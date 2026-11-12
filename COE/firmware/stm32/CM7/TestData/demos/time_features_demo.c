/**
 * @file time_features_demo.c
 * Verify streaming statistics and publish Debug Watch results.
 */
#include "time_features_demo.h"
#include "time_features_reference.h"
#include "benchmark.h"

#include <math.h>
#include "time_features_expected.h"

volatile TimeDomainFeatures demo_features;
volatile bool demo_success = false;
volatile bool demo_checks_passed = false;
volatile bool demo_timer_ready = false;
volatile uint32_t demo_cycles = 0U;
volatile float demo_time_us = 0.0f;
volatile bool demo_streaming_checks_passed = false;
volatile bool demo_window_checks_passed = false;
volatile uint32_t demo_checked_channel_windows = 0U;
volatile uint32_t demo_failed_check = 0U;

/* Shared temporary input buffer, not a stored full window. */
static float test_block[1024];

/** Compare three feature values using the unchanged numerical tolerance. */
static bool close_features(const TimeDomainFeatures *actual,
                           const TimeDomainFeatures *expected)
{
    return fabsf(actual->rms - expected->rms) <= 0.00001f * (1.0f + fabsf(expected->rms))
        && fabsf(actual->kurtosis - expected->kurtosis) <= 0.00001f * (1.0f + fabsf(expected->kurtosis))
        && fabsf(actual->crest_factor - expected->crest_factor) <= 0.00001f * (1.0f + fabsf(expected->crest_factor));
}

/** Verify partition invariance, invalid inputs and reset behavior. */
static bool check_streaming_cases(void)
{
    static const float samples[] = {-3, -1, -1, -1, 1, 1, 1, 3};
    static const size_t block_sizes[] = {1U, 3U, 8U};
    static const float constant[] = {2, 2};
    static const float zeros[] = {0, 0};
    static const float nonfinite[] = {1, NAN};
    static const float offset_samples[] = {1000000.0f, 1000000.125f,
                                         999999.875f, 1000000.25f};
    TimeDomainFeatures reference;
    TimeDomainFeatures actual;
    TimeDomainFeatures again;
    TimeFeatureState state;

    if (!calculate_time_domain_features(samples, 8U, &reference))
        return false;
    for (size_t mode = 0U; mode < 3U; ++mode)
    {
        time_features_reset(&state);
        for (size_t offset = 0U; offset < 8U; offset += block_sizes[mode])
        {
            const size_t count = (8U - offset < block_sizes[mode])
                                 ? 8U - offset : block_sizes[mode];
            if (!time_features_update(&state, samples + offset, count))
                return false;
        }
        if (state.count != 8U || !time_features_finalize(&state, &actual)
            || !close_features(&actual, &reference)
            || !time_features_finalize(&state, &again)
            || !close_features(&actual, &again))
            return false;
    }

    time_features_reset(&state);
    if (!time_features_update(&state, NULL, 0U)
        || time_features_finalize(&state, &actual)) return false;
    if (!time_features_update(&state, constant, 2U)
        || time_features_finalize(&state, &actual)) return false;
    time_features_reset(&state);
    if (!time_features_update(&state, zeros, 2U)
        || time_features_finalize(&state, &actual)) return false;
    time_features_reset(&state);
    if (time_features_update(&state, nonfinite, 2U)
        || time_features_finalize(&state, &actual)
        || time_features_update(&state, samples, 8U)) return false;
    time_features_reset(&state);
    if (time_features_update(&state, NULL, 1U) || state.valid) return false;
    time_features_reset(&state);
    state.count = UINT32_MAX;
    if (time_features_update(&state, samples, 1U) || state.valid) return false;
    time_features_reset(&state);
    if (time_features_update(NULL, samples, 8U)
        || time_features_finalize(NULL, &actual)
        || time_features_finalize(&state, NULL)) return false;
    if (!time_features_update(&state, offset_samples, 1U)
        || time_features_finalize(&state, &actual)) return false;
    if (!time_features_update(&state, offset_samples + 1U, 3U)
        || !time_features_finalize(&state, &actual)
        || !calculate_time_domain_features(offset_samples, 4U, &reference)
        || !close_features(&actual, &reference)) return false;
    return true;
}

/** Generate one deterministic sample for the statistics boundary tests. */
static float synthetic_sample(uint32_t index, uint32_t channel)
{
    const int32_t code = (int32_t)((index * 37U + channel * 11U) % 101U) - 50;
    float sample = (float)code * 0.125f + (float)channel * 0.5f;
    if (index % 97U == 0U)
        sample += (float)(channel + 1U) * 8.0f;
    return sample;
}

/** Verify two four-channel windows at one RPM and block size. */
static bool check_window_boundaries(uint32_t mode, uint32_t block_size)
{
    const uint32_t window_count = (mode == 0U) ? 20571U : 10000U;
    TimeFeatureState states[4];
    uint32_t window_position = 0U;
    uint32_t window_index = 0U;
    for (uint32_t channel = 0U; channel < 4U; ++channel)
        time_features_reset(&states[channel]);

    /* Replay two consecutive windows. All channels share the same boundaries.
     * A real DMA block may contain the end of one window and start of another.
     */
    for (uint32_t first = 0U; first < 2U * window_count; first += block_size)
    {
        const uint32_t remaining = 2U * window_count - first;
        const uint32_t available = (remaining < block_size) ? remaining : block_size;
        uint32_t offset = 0U;
        while (offset < available)
        {
            const uint32_t room = window_count - window_position;
            const uint32_t take = (available - offset < room) ? available - offset : room;
            for (uint32_t channel = 0U; channel < 4U; ++channel)
            {
                for (uint32_t index = 0U; index < take; ++index)
                    test_block[index] = synthetic_sample(first + offset + index, channel);
                if (!time_features_update(&states[channel], test_block, take))
                    return false;
            }
            offset += take;
            window_position += take;
            if (window_position == window_count)
            {
                for (uint32_t channel = 0U; channel < 4U; ++channel)
                {
                    TimeDomainFeatures actual;
                    if (states[channel].count != window_count
                        || !time_features_finalize(&states[channel], &actual)
                        || !close_features(&actual, &expected_windows[mode][window_index][channel]))
                        return false;
                    ++demo_checked_channel_windows;
                    time_features_reset(&states[channel]);
                }
                window_position = 0U;
                ++window_index;
            }
        }
    }
    return window_index == 2U && window_position == 0U;
}

/** Run the statistics checks once and record the small smoke-test duration. */
void run_time_features_demo(void)
{
    static const float samples[] = {-3, -1, -1, -1, 1, 1, 1, 3};
    static const TimeDomainFeatures expected = {1.7320508f, 2.3333333f, 1.7320508f};
    static const uint32_t block_sizes[] = {256U, 512U, 1024U};
    TimeFeatureState state;
    TimeDomainFeatures result;

    demo_checked_channel_windows = 0U;
    demo_failed_check = 0U;
    demo_cycles = 0U;
    demo_time_us = 0.0f;
    demo_timer_ready = benchmark_init();
    time_features_reset(&state);
    const uint32_t start = benchmark_cycles_now();
    demo_success = time_features_update(&state, samples, 3U)
        && time_features_update(&state, samples + 3U, 3U)
        && time_features_update(&state, samples + 6U, 2U);
    const bool finalized = time_features_finalize(&state, &result);
    demo_success = demo_success && finalized;
    const uint32_t end = benchmark_cycles_now();
    demo_features = result;
    if (demo_timer_ready)
    {
        demo_cycles = end - start;
        demo_time_us = benchmark_cycles_to_us(demo_cycles);
    }

    demo_streaming_checks_passed = check_streaming_cases();
    demo_window_checks_passed = true;
    for (uint32_t mode = 0U; mode < 2U; ++mode)
    {
        for (uint32_t size = 0U; size < 3U; ++size)
        {
            if (!check_window_boundaries(mode, block_sizes[size]))
            {
                demo_window_checks_passed = false;
                if (demo_failed_check == 0U)
                    demo_failed_check = 1U + mode * 3U + size;
            }
        }
    }
    demo_checks_passed = demo_success && close_features(&result, &expected)
        && demo_streaming_checks_passed && demo_window_checks_passed
        && demo_checked_channel_windows == 48U;
}
