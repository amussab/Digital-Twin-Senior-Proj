/**
 * @file pipeline.c
 * Route each input chunk through band-pass, signed statistics and envelope formation.
 * Complete each envelope FFT and bearing-band features, then encode one AI window packet.
 */
#include "pipeline.h"

#include <string.h>

_Static_assert(COE_DMA_BLOCK_SAMPLES > 0U, "Scratch block must be nonempty.");
_Static_assert(COE_MAX_ENVELOPE_SAMPLES > 0U, "Envelope storage must be nonempty.");

/** Start a supported nominal-speed stream with empty records and fresh states. */
bool pipeline_init(PipelineState *pipeline, uint32_t nominal_rpm,
                    PipelineEnvelopeBuffer *envelope_buffer,
                    EnvelopeSpectrumWorkspace *spectrum_workspace,
                    const BearingFeatureProfile *bearing_profile,
                    PipelineWindowCallback callback, void *context)
{
    if (pipeline == NULL)
        return false;
    memset(pipeline, 0, sizeof(*pipeline));
    if (nominal_rpm < COE_MIN_RPM || nominal_rpm > COE_MAX_RPM
        || envelope_buffer == NULL || spectrum_workspace == NULL || callback == NULL
        || !frequency_profile_is_valid(bearing_profile, nominal_rpm))
        return false;

    pipeline->nominal_rpm = nominal_rpm;
    /* 20 revolutions, or the most whole revolutions that fit at slow speeds. */
    const uint64_t samples_per_minute = (uint64_t)COE_INPUT_SAMPLE_RATE_HZ * 60U;
    uint64_t revolutions = ((uint64_t)COE_MAX_WINDOW_SAMPLES * nominal_rpm) / samples_per_minute;
    if (revolutions > COE_WINDOW_REVOLUTIONS)
        revolutions = COE_WINDOW_REVOLUTIONS;
    if (revolutions == 0U)
        return false;
    pipeline->window_revolutions = (uint32_t)revolutions;
    const uint64_t numerator = samples_per_minute * revolutions;
    pipeline->window_samples = (uint32_t)((numerator + nominal_rpm / 2U)
                                          / nominal_rpm);
    const uint32_t maximum_retained = (pipeline->window_samples
                                      + COE_ENVELOPE_DECIMATION_FACTOR - 1U)
                                     / COE_ENVELOPE_DECIMATION_FACTOR;
    if (pipeline->window_samples < 2U || maximum_retained > COE_MAX_ENVELOPE_SAMPLES)
        return false;
    pipeline->on_window = callback;
    pipeline->callback_context = context;
    pipeline->envelope_buffer = envelope_buffer;
    pipeline->spectrum_workspace = spectrum_workspace;
    pipeline->bearing_profile = bearing_profile;
    if (!envelope_spectrum_init(spectrum_workspace))
        return false;
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        bandpass_reset(&pipeline->filters[channel]);
        time_features_reset(&pipeline->statistics[channel]);
        envelope_reset(&pipeline->envelopes[channel]);
    }
    pipeline->valid = true;
    return true;
}

/** Process one channel chunk and append its reduced envelope to the current window. */
static bool process_channel(PipelineState *pipeline, size_t channel,
                              const float *samples, size_t count, size_t *retained)
{
    if (!bandpass_process(&pipeline->filters[channel], samples,
                           pipeline->filtered_block, count)
        || !time_features_update(&pipeline->statistics[channel],
                                   pipeline->filtered_block, count))
        return false;

    /* Time features must consume signed samples before in-place rectification. */
    if (!envelope_process(&pipeline->envelopes[channel], pipeline->filtered_block,
                           pipeline->filtered_block, count, retained))
        return false;
    const uint32_t position = pipeline->envelope_samples_in_window;
    if (position > COE_MAX_ENVELOPE_SAMPLES
        || *retained > COE_MAX_ENVELOPE_SAMPLES - position)
        return false;
    memcpy(pipeline->envelope_buffer->samples[channel] + position, pipeline->filtered_block,
           *retained * sizeof(float));
    return true;
}

/** Finalize all features/packet, publish the complete window, then restart only window counters. */
static bool finish_window(PipelineState *pipeline)
{
    if (pipeline->completed_windows == UINT32_MAX)
        return false;
    const uint32_t first = (COE_ENVELOPE_DECIMATION_FACTOR - pipeline->envelope_start_phase)
                           % COE_ENVELOPE_DECIMATION_FACTOR;
    const uint32_t expected_retained = first < pipeline->window_samples
        ? 1U + (pipeline->window_samples - 1U - first) / COE_ENVELOPE_DECIMATION_FACTOR : 0U;
    if (pipeline->envelope_samples_in_window != expected_retained)
        return false;
    PipelineWindowResult result = {
        .window_id = pipeline->completed_windows + 1U,
        .nominal_rpm = pipeline->nominal_rpm,
        .sample_count = pipeline->window_samples,
        .envelope_sample_count = pipeline->envelope_samples_in_window,
        .envelope_sample_rate_hz = COE_ENVELOPE_SAMPLE_RATE_HZ,
        .envelope_first_input_offset = first
    };
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        if (pipeline->statistics[channel].count != pipeline->window_samples)
            return false;
        result.valid[channel] = time_features_finalize(&pipeline->statistics[channel],
                                                        &result.features[channel]);
        result.envelope[channel] = pipeline->envelope_buffer->samples[channel];
        if (!calculate_envelope_spectrum(pipeline->spectrum_workspace, result.envelope[channel],
                                          result.envelope_sample_count)
            || !extract_frequency_features(result.envelope[channel], result.envelope_sample_count,
                    pipeline->spectrum_workspace->power_density, pipeline->nominal_rpm,
                    pipeline->bearing_profile, &result.frequency_features[channel]))
            return false;
    }
    if (!build_ai_payload(result.features, result.frequency_features, result.valid,
                           result.window_id, pipeline->nominal_rpm,
                           pipeline->bearing_profile->id, &result.ai_payload))
        return false;
    if (!pipeline->on_window(&result, pipeline->callback_context))
        return false;

    ++pipeline->completed_windows;
    pipeline->samples_in_window = 0U;
    pipeline->envelope_samples_in_window = 0U;
    pipeline->envelope_start_phase = pipeline->envelopes[0].phase;
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        time_features_reset(&pipeline->statistics[channel]);
    /* Both filters and decimation phase continue; only the next record starts empty.
     * Buffer contents need no clearing: its count defines which samples are valid.
     */
    return true;
}

/** Consume synchronized channel blocks, splitting them at scratch and window boundaries. */
bool pipeline_process_block(PipelineState *pipeline,
                             const float *const channels[COE_CHANNEL_COUNT],
                             size_t count)
{
    if (pipeline == NULL || !pipeline->valid)
        return false;
    if (count == 0U)
        return true;
    if (channels == NULL)
    {
        pipeline->valid = false;
        return false;
    }
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        if (channels[channel] == NULL)
        {
            pipeline->valid = false;
            return false;
        }
    }

    size_t offset = 0U;
    while (offset < count)
    {
        size_t take = count - offset;
        const uint32_t room = pipeline->window_samples - pipeline->samples_in_window;
        if (take > room)
            take = room;
        if (take > COE_DMA_BLOCK_SAMPLES)
            take = COE_DMA_BLOCK_SAMPLES;

        size_t retained = 0U;
        for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        {
            size_t channel_retained;
            if (!process_channel(pipeline, channel, channels[channel] + offset,
                                   take, &channel_retained))
            {
                pipeline->valid = false;
                return false;
            }
            if (channel == 0U)
                retained = channel_retained;
            if (channel_retained != retained
                || pipeline->envelopes[channel].phase != pipeline->envelopes[0].phase)
            {
                pipeline->valid = false;
                return false;
            }
        }
        pipeline->envelope_samples_in_window += (uint32_t)retained;
        offset += take;
        pipeline->samples_in_window += (uint32_t)take;
        if (pipeline->samples_in_window == pipeline->window_samples
            && !finish_window(pipeline))
        {
            pipeline->valid = false;
            return false;
        }
    }
    return true;
}
