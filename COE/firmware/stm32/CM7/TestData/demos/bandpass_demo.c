/**
 * @file bandpass_demo.c
 * Verify the CMSIS filter wrapper against recorded Python references.
 */
#include "bandpass_demo.h"
#include "bandpass.h"
#include "pipeline_config.h"
#include "benchmark.h"
#include "bandpass_fixtures.h"

#include <math.h>
#include <string.h>

_Static_assert(COE_CHANNEL_COUNT == BANDPASS_FIXTURE_CHANNELS,
               "The current test fixtures contain four independent signals.");
_Static_assert(COE_DMA_BLOCK_SAMPLES <= BANDPASS_FIXTURE_SAMPLES,
               "Timing test requires a block no larger than the fixture.");

volatile bool bandpass_checks_passed = false;
volatile bool bandpass_continuity_checks_passed = false;
volatile uint32_t bandpass_checked_channel_runs = 0U;
volatile uint32_t bandpass_failed_check = 0U;
volatile float bandpass_max_abs_error = 0.0f;
volatile bool bandpass_timer_ready = false;
volatile uint32_t bandpass_block_cycles = 0U;
volatile float bandpass_block_time_us = 0.0f;

/* Reused across channels: the filter state, not this scratch buffer, is private. */
static float block_output[BANDPASS_FIXTURE_SAMPLES];

/** Compare filtered samples and record the largest observed absolute error. */
static bool samples_match(const float *actual, const float *expected, size_t count)
{
    for (size_t index = 0U; index < count; ++index)
    {
        const float error = fabsf(actual[index] - expected[index]);
        if (!isfinite(actual[index]))
            return false;
        if (error > bandpass_max_abs_error)
            bandpass_max_abs_error = error;
        if (error > 0.00002f * (1.0f + fabsf(expected[index])))
            return false;
    }
    return true;
}

/** Verify four independent channels using the selected block size. */
static bool check_partition(size_t block_size)
{
    BandpassState states[COE_CHANNEL_COUNT];
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        bandpass_reset(&states[channel]);

    for (size_t offset = 0U; offset < BANDPASS_FIXTURE_SAMPLES; offset += block_size)
    {
        const size_t remaining = BANDPASS_FIXTURE_SAMPLES - offset;
        const size_t count = (remaining < block_size) ? remaining : block_size;
        for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        {
            if (!bandpass_process(&states[channel], bandpass_test_input[channel] + offset,
                                  block_output, count)
                || !samples_match(block_output, bandpass_test_expected[channel] + offset, count))
                return false;
        }
    }
    bandpass_checked_channel_runs += COE_CHANNEL_COUNT;
    return true;
}

/** Verify continuous delays, isolated channels and in-place filtering. */
static bool check_continuity_and_inplace(void)
{
    BandpassState state;
    BandpassState idle_state;
    bandpass_reset(&state);
    /* Treat index 333 as an arbitrary analysis boundary. Do not reset the filter. */
    if (!bandpass_process(&state, bandpass_test_input[0], block_output, 333U)
        || !samples_match(block_output, bandpass_test_expected[0], 333U))
        return false;

    /* A different channel's zeros must not affect this active channel. */
    bandpass_reset(&idle_state);
    memset(block_output, 0, 32U * sizeof(float));
    if (!bandpass_process(&idle_state, block_output, block_output, 32U))
        return false;
    for (size_t index = 0U; index < 32U; ++index)
        if (block_output[index] != 0.0f) return false;

    if (!bandpass_process(&state, bandpass_test_input[0] + 333U, block_output,
                          BANDPASS_FIXTURE_SAMPLES - 333U)
        || !samples_match(block_output, bandpass_test_expected[0] + 333U,
                          BANDPASS_FIXTURE_SAMPLES - 333U))
        return false;

    bandpass_reset(&state);
    memcpy(block_output, bandpass_test_input[0], sizeof(block_output));
    return bandpass_process(&state, block_output, block_output, BANDPASS_FIXTURE_SAMPLES)
        && samples_match(block_output, bandpass_test_expected[0], BANDPASS_FIXTURE_SAMPLES);
}

/** Verify argument checks, nonfinite rejection and recovery after reset. */
static bool check_reset_and_invalid_input(void)
{
    BandpassState state;
    static const float nonfinite[] = {1.0f, NAN};
    bandpass_reset(&state);
    if (!bandpass_process(&state, NULL, NULL, 0U)
        || bandpass_process(NULL, bandpass_test_input[0], block_output, 1U))
        return false;
    if (bandpass_process(&state, NULL, block_output, 1U) || state.valid)
        return false;
    if (bandpass_process(&state, bandpass_test_input[0], block_output, 1U))
        return false;
    bandpass_reset(&state);
    if (bandpass_process(&state, bandpass_test_input[0], NULL, 1U) || state.valid)
        return false;
    bandpass_reset(&state);
    if (bandpass_process(&state, nonfinite, block_output, 2U) || state.valid)
        return false;

    static const float infinities[] = {INFINITY, -INFINITY};
    for (size_t index = 0U; index < 2U; ++index)
    {
        bandpass_reset(&state);
        if (bandpass_process(&state, infinities + index, block_output, 1U) || state.valid)
            return false;
    }
    bandpass_reset(&state);
    state.delay[0] = INFINITY;
    if (bandpass_process(&state, bandpass_test_input[0], block_output, 1U) || state.valid)
        return false;

    /* Reset must recover an invalid state and produce the reference impulse. */
    bandpass_reset(&state);
    memset(block_output, 0, 128U * sizeof(float));
    block_output[0] = 1.0f;
    return bandpass_process(&state, block_output, block_output, 128U)
        && samples_match(block_output, bandpass_impulse_expected, 128U);
}

/** Run filter checks once and measure one block for all four channels. */
void run_bandpass_demo(void)
{
    static const size_t block_sizes[] = {1U, 17U, 256U, 1024U};
    bool partitions_passed = true;
    bandpass_checked_channel_runs = 0U;
    bandpass_failed_check = 0U;
    bandpass_max_abs_error = 0.0f;
    bandpass_block_cycles = 0U;
    bandpass_block_time_us = 0.0f;

    for (size_t mode = 0U; mode < 4U; ++mode)
    {
        if (!check_partition(block_sizes[mode]))
        {
            partitions_passed = false;
            if (bandpass_failed_check == 0U)
                bandpass_failed_check = (uint32_t)mode + 1U;
        }
    }
    bandpass_continuity_checks_passed = check_continuity_and_inplace();
    const bool input_checks_passed = check_reset_and_invalid_input();
    if (!bandpass_continuity_checks_passed && bandpass_failed_check == 0U)
        bandpass_failed_check = 5U;
    if (!input_checks_passed && bandpass_failed_check == 0U)
        bandpass_failed_check = 6U;

    /* A separate timing measurement: one 256-sample block for all four channels.
     * This excludes the comparisons and all test data generation above.
     */
    BandpassState states[COE_CHANNEL_COUNT];
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        bandpass_reset(&states[channel]);
    bandpass_timer_ready = benchmark_init();
    bool timed_calls_passed = true;
    const uint32_t start = benchmark_cycles_now();
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        if (!bandpass_process(&states[channel], bandpass_test_input[channel],
                              block_output, COE_DMA_BLOCK_SAMPLES))
            timed_calls_passed = false;
    }
    const uint32_t end = benchmark_cycles_now();
    if (bandpass_timer_ready)
    {
        bandpass_block_cycles = end - start;
        bandpass_block_time_us = benchmark_cycles_to_us(bandpass_block_cycles);
    }

    bandpass_checks_passed = partitions_passed && bandpass_continuity_checks_passed
        && input_checks_passed && timed_calls_passed
        && bandpass_checked_channel_runs == 16U;
}
