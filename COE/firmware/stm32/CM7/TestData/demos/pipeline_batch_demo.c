/**
 * @file pipeline_batch_demo.c
 * Measure processing after a complete four-channel input window is ready.
 * Input preparation and reference comparisons remain outside the timed call.
 */
#include "pipeline_batch_demo.h"
#include "pipeline.h"
#include "pipeline_demo.h"
#include "final_features_demo.h"
#include "benchmark.h"
#include "feature_uart.h"
#include "delivery_timing.h"
#include "bandpass_fixtures.h"
#include <string.h>

volatile bool pipeline_batch_checks_passed;
volatile uint32_t pipeline_batch_checked_windows;
volatile uint32_t pipeline_batch_failed_check;
volatile bool pipeline_batch_timer_ready;
volatile float pipeline_batch_max_ms_1750;
volatile float pipeline_batch_max_ms_3600;
volatile uint32_t pipeline_batch_envelope_samples_1750[3];
volatile uint32_t pipeline_batch_envelope_samples_3600[3];
volatile bool pipeline_batch_fixed_shape_passed;
volatile bool pipeline_batch_deadline_met;
volatile uint32_t pipeline_batch_feature_count_1750;
volatile uint32_t pipeline_batch_feature_count_3600;
volatile uint32_t pipeline_batch_fft_size;
volatile float pipeline_batch_fft_bin_spacing_hz;
volatile uint32_t pipeline_batch_last_profile_id;
volatile float pipeline_batch_features_1750[COE_FEATURE_COUNT];
volatile float pipeline_batch_features_3600[COE_FEATURE_COUNT];
volatile uint8_t pipeline_batch_payload_1750[COE_PAYLOAD_BYTES];
volatile uint8_t pipeline_batch_payload_3600[COE_PAYLOAD_BYTES];

/* Replay data is 1750 and 3600 RPM only; size for the larger 1750-RPM window. */
#define MAX_WINDOW_SAMPLES ((COE_INPUT_SAMPLE_RATE_HZ * 60U * COE_WINDOW_REVOLUTIONS + 875U) / 1750U)
/* The linker places this 329136-byte test buffer in AXI SRAM, outside DTCM.
 * NOLOAD storage: explicitly fill every sample before any processing read.
 * This test buffer is not the production DMA-buffer allocation.
 */
static float complete_window[COE_CHANNEL_COUNT][MAX_WINDOW_SAMPLES]
    __attribute__((section(".pipeline_test_buffers"), aligned(32)));
static PipelineState pipeline; /* Working statistics/delays stay in DTCM. */
static PipelineEnvelopeBuffer envelope_buffer
    __attribute__((section(".pipeline_test_buffers"), aligned(32)));
static PipelineEnvelopeBuffer captured_envelope
    __attribute__((section(".pipeline_test_buffers"), aligned(32)));
static PipelineWindowResult captured;
static uint32_t callback_count;

/** Copy features, metadata and all envelopes into owned result storage inside the timer. */
static bool capture_result(const PipelineWindowResult *result, void *context)
{
    (void)context;
    if (result->envelope_sample_count > COE_MAX_ENVELOPE_SAMPLES)
        return false;
    captured = *result;
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        if (result->envelope[channel] == NULL)
            return false;
        memcpy(captured_envelope.samples[channel], result->envelope[channel],
               result->envelope_sample_count * sizeof(float));
        captured.envelope[channel] = captured_envelope.samples[channel];
    }
    ++callback_count;
    return true;
}

/** Prepare and process three full windows, covering every decimation phase of both speed modes. */
static bool check_mode(size_t profile, size_t mode)
{
    const uint32_t rpm = mode == 0U ? 1750U : 3600U;
    const uint32_t samples = mode == 0U ? 20571U : 10000U;
    const BearingFeatureProfile *bearing = profile == 0U ? &coe_cwru_drive_end_bearing : &coe_project_bearing;
    if (!pipeline_init(&pipeline, rpm, &envelope_buffer, &final_test_workspace, bearing, capture_result, NULL)
        || pipeline.window_samples != samples || samples > MAX_WINDOW_SAMPLES)
        return false;
    const float *channels[COE_CHANNEL_COUNT];
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        channels[channel] = complete_window[channel];

    for (size_t window = 0U; window < 3U; ++window)
    {
        /* All four channels' entire input windows exist before starting time.
         * Repeat the same snippets as the streaming tests for an exact oracle.
         */
        for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
            for (size_t sample = 0U; sample < samples; ++sample)
                complete_window[channel][sample] = bandpass_test_input[channel]
                                   [(window * samples + sample) % 1024U];
        callback_count = 0U;
        captured = (PipelineWindowResult){0};

        delivery_timing_begin();
        const uint32_t start = pipeline_batch_timer_ready ? benchmark_cycles_now() : 0U;
        const bool processed = pipeline_process_block(&pipeline, channels, samples);
        const uint32_t end = pipeline_batch_timer_ready ? benchmark_cycles_now() : 0U;
        /* Stop after results have been copied by the callback and the processing
         * call returns. Input generation and reference comparisons are excluded.
         */
        if (!processed || callback_count != 1U || captured.window_id != window + 1U
            || captured.sample_count != samples || captured.nominal_rpm != rpm)
            return false;
        /* The feature vector is complete before starting UART transmission. */
        if (!feature_uart_send(captured.ai_payload.bytes))
            return false;
        delivery_timing_finish(pipeline_batch_timer_ready
            ? benchmark_cycles_to_us(end - start) / 1000.0f : 0.0f);
        /* The callback copied envelopes, so these references are owned and safe here. */
        if (!pipeline_final_matches_reference(&captured, profile, mode, window)
            || !pipeline_envelope_matches_reference(&captured, mode, window))
            return false;
        if (mode == 0U)
            pipeline_batch_envelope_samples_1750[window] = captured.envelope_sample_count;
        else
            pipeline_batch_envelope_samples_3600[window] = captured.envelope_sample_count;
        const uint32_t feature_count = captured.ai_payload.bytes[6]
                                       | ((uint32_t)captured.ai_payload.bytes[7] << 8U);
        if (feature_count != COE_FEATURE_COUNT)
            return false;
        if (mode == 0U)
            pipeline_batch_feature_count_1750 = feature_count;
        else
            pipeline_batch_feature_count_3600 = feature_count;
        volatile float *feature_copy = mode == 0U ? pipeline_batch_features_1750 : pipeline_batch_features_3600;
        volatile uint8_t *packet_copy = mode == 0U ? pipeline_batch_payload_1750 : pipeline_batch_payload_3600;
        for (size_t index = 0U; index < COE_FEATURE_COUNT; ++index)
            feature_copy[index] = captured.ai_payload.features[index];
        for (size_t index = 0U; index < COE_PAYLOAD_BYTES; ++index)
            packet_copy[index] = captured.ai_payload.bytes[index];
        pipeline_batch_last_profile_id = bearing->id;
        if (pipeline_batch_timer_ready)
        {
            const float elapsed = benchmark_cycles_to_us(end - start) / 1000.0f;
            volatile float *maximum = mode == 0U ? &pipeline_batch_max_ms_1750
                                                 : &pipeline_batch_max_ms_3600;
            if (elapsed > *maximum)
                *maximum = elapsed;
        }
        ++pipeline_batch_checked_windows;
    }
    return pipeline.completed_windows == 3U && pipeline.samples_in_window == 0U;
}

/** Run both nominal speeds and publish complete-window timing/check results. */
void run_pipeline_batch_demo(void)
{
    pipeline_batch_checks_passed = false;
    pipeline_batch_checked_windows = 0U;
    pipeline_batch_failed_check = 0U;
    pipeline_batch_max_ms_1750 = 0.0f;
    pipeline_batch_max_ms_3600 = 0.0f;
    for (size_t window = 0U; window < 3U; ++window)
    {
        pipeline_batch_envelope_samples_1750[window] = 0U;
        pipeline_batch_envelope_samples_3600[window] = 0U;
    }
    pipeline_batch_fixed_shape_passed = false;
    pipeline_batch_deadline_met = false;
    pipeline_batch_feature_count_1750 = pipeline_batch_feature_count_3600 = 0U;
    pipeline_batch_fft_size = COE_FFT_SIZE;
    pipeline_batch_fft_bin_spacing_hz = (float)COE_ENVELOPE_SAMPLE_RATE_HZ / COE_FFT_SIZE;
    pipeline_batch_last_profile_id = 0U;
    for (size_t index = 0U; index < COE_FEATURE_COUNT; ++index)
        pipeline_batch_features_1750[index] = pipeline_batch_features_3600[index] = 0.0f;
    for (size_t index = 0U; index < COE_PAYLOAD_BYTES; ++index)
        pipeline_batch_payload_1750[index] = pipeline_batch_payload_3600[index] = 0U;
    pipeline_batch_timer_ready = benchmark_init();
    for (size_t profile = 0U; profile < 2U; ++profile)
        for (size_t mode = 0U; mode < 2U; ++mode)
            if (!check_mode(profile, mode))
            {
                pipeline_batch_failed_check = (uint32_t)(profile * 2U + mode + 1U);
                return;
            }
    pipeline_batch_checks_passed = pipeline_batch_checked_windows == 12U;
    pipeline_batch_fixed_shape_passed = pipeline_batch_checks_passed
        && pipeline_batch_feature_count_1750 == COE_FEATURE_COUNT
        && pipeline_batch_feature_count_3600 == COE_FEATURE_COUNT;
    pipeline_batch_deadline_met = pipeline_batch_checks_passed && pipeline_batch_timer_ready
        && pipeline_batch_max_ms_1750 > 0.0f && pipeline_batch_max_ms_1750 <= 333.0f
        && pipeline_batch_max_ms_3600 > 0.0f && pipeline_batch_max_ms_3600 <= 333.0f;
}
