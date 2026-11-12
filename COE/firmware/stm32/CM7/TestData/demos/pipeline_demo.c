/**
 * @file pipeline_demo.c
 * Verify consecutive pipeline windows and measure accumulated block processing work.
 */
#include "pipeline_demo.h"
#include "pipeline.h"
#include "benchmark.h"
#include "bandpass_fixtures.h"
#include "pipeline_expected.h"
#include "pipeline_envelope_expected.h"
#include "final_third_envelopes.h"
#include "final_pipeline_expected.h"
#include "final_features_demo.h"
#include <string.h>
#include <math.h>

volatile bool pipeline_checks_passed;
volatile uint32_t pipeline_checked_windows;
volatile uint32_t pipeline_failed_check;
volatile bool pipeline_timer_ready;
volatile float pipeline_max_window_ms_1750;
volatile float pipeline_max_window_ms_3600;
volatile float pipeline_max_block_time_us;

/* Test storage only: production uses the ADC adapter's buffers instead. */
static float replay_block[COE_CHANNEL_COUNT][1024];
static PipelineState demo_pipeline; /* Small working state stays in fast DTCM. */
static PipelineEnvelopeBuffer envelope_buffer
    __attribute__((section(".pipeline_test_buffers"), aligned(32)));

typedef struct
{
    PipelineWindowResult results[2];
    uint32_t count;
    uint32_t accumulated_cycles;
    uint32_t segment_start;
    uint32_t window_cycles[2];
    bool timed;
    bool validate_envelopes;
    size_t mode;
} DemoOutput;

static DemoOutput output;

/** Compare metadata and every retained envelope sample with the independent Python cascade. */
bool pipeline_envelope_matches_reference(const PipelineWindowResult *result,
                                          size_t mode, size_t window)
{
    if (result == NULL || mode >= 2U || window >= 3U)
        return false;
    const uint32_t expected_count = window == 2U ? final_third_envelope_counts[mode]
                                                : pipeline_envelope_counts[mode][window];
    const uint32_t expected_offset = window == 2U ? final_third_envelope_offsets[mode]
                                                 : pipeline_envelope_first_offsets[mode][window];
    if (result->envelope_sample_count != expected_count
        || result->envelope_sample_rate_hz != COE_ENVELOPE_SAMPLE_RATE_HZ
        || result->envelope_first_input_offset != expected_offset)
        return false;
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        if (result->envelope[channel] == NULL)
            return false;
        const float *expected = window == 2U ? final_third_envelopes[mode][channel]
                                             : pipeline_envelope_expected[mode][window][channel];
        for (size_t sample = 0U; sample < result->envelope_sample_count; ++sample)
        {
            const float actual = result->envelope[channel][sample];
            /* This new combined test allows rounding from both IIR filters.
             * Existing time-feature and standalone tolerances remain unchanged.
             */
            if (!isfinite(actual) || fabsf(actual - expected[sample])
                > 4.0e-5f * (1.0f + fabsf(expected[sample])))
                return false;
        }
    }
    return true;
}

/** Check every final feature, fixed dimensions and packet metadata against Python. */
bool pipeline_final_matches_reference(const PipelineWindowResult *result,
                                       size_t profile, size_t mode, size_t window)
{
    if (result == NULL || profile >= 2U || mode >= 2U || window >= 3U
        || result->ai_payload.channel_valid_mask != 15U)
        return false;
    const uint8_t *packet = result->ai_payload.bytes;
    const uint32_t profile_id = profile == 0U ? COE_CWRU_DRIVE_END_BEARING : COE_PROJECT_BEARING;
    if (memcmp(packet, "COE1", 4U) != 0 || packet[4] != COE_PAYLOAD_SCHEMA_VERSION
        || packet[5] != 0U || packet[6] != COE_FEATURE_COUNT || packet[7] != 0U
        || packet[16] != profile_id || packet[17] != 0U || packet[18] != 0U || packet[19] != 0U
        || packet[20] != 15U || packet[24] != COE_ACCELERATION_UNIT_G)
        return false;
    for (size_t index = 0U; index < COE_FEATURE_COUNT; ++index)
    {
        const float actual = result->ai_payload.features[index];
        const float expected = final_pipeline_expected[profile][mode][window][index];
        const float tolerance = index % COE_FEATURES_PER_CHANNEL < 3U ? 2.0e-5f : 4.0e-5f;
        if (!isfinite(actual) || actual < 0.0f
            || fabsf(actual - expected) > tolerance * (1.0f + fabsf(expected)))
            return false;
    }
    return true;
}

/** Validate untimed records and final features, then copy the result and record timed work. */
static bool capture_window(const PipelineWindowResult *result, void *context)
{
    DemoOutput *sink = context;
    if (sink->count >= 2U)
        return false;
    if (sink->validate_envelopes
        && (!pipeline_envelope_matches_reference(result, sink->mode, sink->count)
            || !pipeline_final_matches_reference(result, 0U, sink->mode, sink->count)))
        return false;
    /* Borrowed envelope pointers are used above only; later checks use copied features/metadata. */
    sink->results[sink->count] = *result;
    if (sink->timed)
    {
        const uint32_t now = benchmark_cycles_now();
        sink->window_cycles[sink->count] = sink->accumulated_cycles
                                          + (now - sink->segment_start);
        sink->accumulated_cycles = 0U;
        sink->segment_start = now;
    }
    ++sink->count;
    return true;
}

/** Compare one feature with its Python reference using the existing tolerance. */
static bool close_to_reference(float actual, float expected)
{
    return isfinite(actual) && fabsf(actual - expected)
                              <= 2.0e-5f * (1.0f + fabsf(expected));
}

/** Replay two windows at one RPM/block size and compare all channels' features. */
static bool check_replay(size_t mode, size_t block_size, bool measure_time)
{
    const uint32_t rpm = mode == 0U ? 1750U : 3600U;
    const uint32_t samples = mode == 0U ? 20571U : 10000U;
    output = (DemoOutput){.timed = pipeline_timer_ready && measure_time,
                          .validate_envelopes = !measure_time, .mode = mode};
    if (!pipeline_init(&demo_pipeline, rpm, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
        || demo_pipeline.window_samples != samples)
        return false;

    const float *channels[COE_CHANNEL_COUNT];
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        channels[channel] = replay_block[channel];

    uint32_t offset = 0U;
    while (offset < 2U * samples)
    {
        size_t take = 2U * samples - offset;
        if (take > block_size)
            take = block_size;
        for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
            for (size_t index = 0U; index < take; ++index)
                replay_block[channel][index] = bandpass_test_input[channel]
                                                  [(offset + index) % 1024U];

        /* Filling test input and comparing results are outside the timer. */
        const uint32_t start = output.timed ? benchmark_cycles_now() : 0U;
        output.segment_start = start;
        if (!pipeline_process_block(&demo_pipeline, channels, take))
            return false;
        if (output.timed)
        {
            const uint32_t end = benchmark_cycles_now();
            output.accumulated_cycles += end - output.segment_start;
            const float elapsed_us = (float)benchmark_cycles_to_us(end - start);
            if (elapsed_us > pipeline_max_block_time_us)
                pipeline_max_block_time_us = elapsed_us;
        }
        offset += (uint32_t)take;
    }

    if (output.count != 2U || demo_pipeline.completed_windows != 2U
        || demo_pipeline.samples_in_window != 0U || !demo_pipeline.valid)
        return false;
    for (size_t window = 0U; window < 2U; ++window)
    {
        const PipelineWindowResult *result = &output.results[window];
        if (result->window_id != window + 1U || result->nominal_rpm != rpm
            || result->sample_count != samples)
            return false;
        for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        {
            const TimeDomainFeatures *actual = &result->features[channel];
            const TimeDomainFeatures *expected = &pipeline_expected[mode][window][channel];
            if (!result->valid[channel] || demo_pipeline.statistics[channel].count != 0U
                || !close_to_reference(actual->rms, expected->rms)
                || !close_to_reference(actual->kurtosis, expected->kurtosis)
                || !close_to_reference(actual->crest_factor, expected->crest_factor))
                return false;
        }
        if (!pipeline_final_matches_reference(result, 0U, mode, window))
            return false;
        if (output.timed)
        {
            const float elapsed_ms = (float)benchmark_cycles_to_us(output.window_cycles[window])
                                      / 1000.0f;
            volatile float *maximum = mode == 0U ? &pipeline_max_window_ms_1750
                                                 : &pipeline_max_window_ms_3600;
            if (elapsed_ms > *maximum)
                *maximum = elapsed_ms;
        }
        if (!measure_time)
            ++pipeline_checked_windows;
    }
    return true;
}

/** Verify invalid arguments, rejected results and stream reinitialization. */
static bool check_api(void)
{
    output = (DemoOutput){0};
    if (pipeline_init(NULL, 1750U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
        || pipeline_init(&demo_pipeline, 1000U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
        || pipeline_init(&demo_pipeline, 1750U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, NULL, &output)
        || pipeline_init(&demo_pipeline, 1750U, NULL, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
        || pipeline_init(&demo_pipeline, 1750U, &envelope_buffer, NULL, &coe_cwru_drive_end_bearing, capture_window, &output)
        || pipeline_init(&demo_pipeline, 1750U, &envelope_buffer, &final_test_workspace, NULL, capture_window, &output))
        return false;
    if (!pipeline_init(&demo_pipeline, 1750U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
        || !pipeline_process_block(&demo_pipeline, NULL, 0U)
        || pipeline_process_block(&demo_pipeline, NULL, 1U)
        || demo_pipeline.valid)
        return false;
    const float *channels[COE_CHANNEL_COUNT] = {replay_block[0], replay_block[1],
                                               NULL, replay_block[3]};
    if (!pipeline_init(&demo_pipeline, 3600U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
        || pipeline_process_block(&demo_pipeline, channels, 1U)
        || demo_pipeline.valid)
        return false;
    /* A rejected payload stops the stream instead of silently dropping a window. */
    output.count = 2U;
    channels[2] = replay_block[2];
    if (!pipeline_init(&demo_pipeline, 3600U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output))
        return false;
    uint32_t remaining = demo_pipeline.window_samples;
    while (remaining > 0U)
    {
        const size_t take = remaining > 256U ? 256U : remaining;
        const bool success = pipeline_process_block(&demo_pipeline, channels, take);
        remaining -= (uint32_t)take;
        if (remaining > 0U && !success)
            return false;
        if (remaining == 0U && (success || demo_pipeline.valid))
            return false;
    }
    /* Reject an impossible append position before any out-of-bounds buffer write. */
    if (!pipeline_init(&demo_pipeline, 1750U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output))
        return false;
    demo_pipeline.envelope_samples_in_window = COE_MAX_ENVELOPE_SAMPLES + 1U;
    if (pipeline_process_block(&demo_pipeline, channels, 1U) || demo_pipeline.valid)
        return false;
    return pipeline_init(&demo_pipeline, 1750U, &envelope_buffer, &final_test_workspace, &coe_cwru_drive_end_bearing, capture_window, &output)
           && demo_pipeline.valid && demo_pipeline.samples_in_window == 0U;
}

/** Run the streaming pipeline checks once and publish Debug Watch results. */
void run_pipeline_demo(void)
{
    pipeline_checks_passed = false;
    pipeline_checked_windows = 0U;
    pipeline_failed_check = 0U;
    pipeline_max_window_ms_1750 = 0.0f;
    pipeline_max_window_ms_3600 = 0.0f;
    pipeline_max_block_time_us = 0.0f;
    pipeline_timer_ready = benchmark_init();
    const size_t block_sizes[] = {17U, 256U, 1024U};
    for (size_t mode = 0U; mode < 2U; ++mode)
        for (size_t test = 0U; test < 3U; ++test)
            if (!check_replay(mode, block_sizes[test], false))
            {
                pipeline_failed_check = (uint32_t)(mode * 3U + test + 1U);
                return;
            }
    if (!check_api())
    {
        pipeline_failed_check = 7U;
        return;
    }
    /* Separate timing replays do not compare envelope arrays inside the callback. */
    for (size_t mode = 0U; mode < 2U; ++mode)
        if (!check_replay(mode, 256U, true))
        {
            pipeline_failed_check = (uint32_t)mode + 8U;
            return;
        }
    pipeline_checks_passed = pipeline_checked_windows == 12U;
}
