/**
 * @file envelope_demo.c
 * Verify envelope samples and continuous decimation against independent Python outputs.
 * The existing pipeline benchmark is separate; this measures the new component only.
 */
#include "envelope_demo.h"
#include "envelope.h"
#include "pipeline_config.h"
#include "benchmark.h"
#include "bandpass_fixtures.h"
#include "envelope_fixtures.h"

#include <math.h>
#include <string.h>

_Static_assert(COE_CHANNEL_COUNT == 4U, "The current fixtures contain four channels.");
_Static_assert(COE_DMA_BLOCK_SAMPLES == 256U, "Regenerate the timing test for another block size.");
_Static_assert(COE_ENVELOPE_DECIMATION_FACTOR == 6U, "The reference samples use factor six.");

volatile bool envelope_checks_passed;
volatile uint32_t envelope_checked_channel_runs;
volatile uint32_t envelope_checked_windows;
volatile uint32_t envelope_failed_check;
volatile float envelope_max_abs_error;
volatile bool envelope_timer_ready;
volatile float envelope_block_time_us;

/* Reusable test buffers; full expected windows live in Flash, not RAM. */
static float test_block[1024];
static float output_block[1024];

/** Compare retained samples to Python and record the largest absolute difference. */
static bool samples_match(const float *actual, const float *expected, size_t count)
{
    for (size_t index = 0U; index < count; ++index)
    {
        const float error = fabsf(actual[index] - expected[index]);
        if (!isfinite(actual[index]))
            return false;
        if (error > envelope_max_abs_error)
            envelope_max_abs_error = error;
        if (error > 2.0e-5f * (1.0f + fabsf(expected[index])))
            return false;
    }
    return true;
}

/** Verify the same 1024 input samples at the selected block size for each channel. */
static bool check_partition(size_t block_size)
{
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        EnvelopeState state;
        envelope_reset(&state);
        size_t retained = 0U;
        for (size_t offset = 0U; offset < 1024U; offset += block_size)
        {
            const size_t remaining = 1024U - offset;
            const size_t take = remaining < block_size ? remaining : block_size;
            size_t produced;
            if (!envelope_process(&state, bandpass_test_expected[channel] + offset,
                                   output_block, take, &produced)
                || retained + produced > 171U
                || !samples_match(output_block, envelope_expected_windows[0][0][channel] + retained, produced)
                || state.phase != (offset + take) % 6U)
                return false;
            retained += produced;
        }
        if (retained != 171U || !state.valid)
            return false;
        ++envelope_checked_channel_runs;
    }
    return true;
}

/** Verify a split at sample 333, channel isolation, sign reversal and in-place use. */
static bool check_continuity_and_inplace(void)
{
    EnvelopeState active;
    EnvelopeState idle;
    size_t produced;
    const float *expected = envelope_expected_windows[0][0][0];
    envelope_reset(&active);
    if (!envelope_process(&active, bandpass_test_expected[0], output_block, 333U, &produced)
        || produced != 56U || active.phase != 3U
        || !samples_match(output_block, expected, produced))
        return false;

    /* Another channel's zero input must not alter the active channel's delays. */
    envelope_reset(&idle);
    memset(test_block, 0, sizeof(test_block));
    if (!envelope_process(&idle, test_block, output_block, 1024U, &produced))
        return false;
    for (size_t index = 0U; index < produced; ++index)
        if (output_block[index] != 0.0f) return false;
    if (!envelope_process(&active, bandpass_test_expected[0] + 333U, output_block, 691U, &produced)
        || produced != 115U || active.phase != 4U
        || !samples_match(output_block, expected + 56U, produced))
        return false;

    envelope_reset(&active);
    memcpy(test_block, bandpass_test_expected[0], sizeof(test_block));
    if (!envelope_process(&active, test_block, test_block, 1024U, &produced)
        || produced != 171U || !samples_match(test_block, expected, produced))
        return false;
    envelope_reset(&active);
    for (size_t index = 0U; index < 1024U; ++index)
        test_block[index] = -bandpass_test_expected[0][index];
    return envelope_process(&active, test_block, output_block, 1024U, &produced)
        && produced == 171U && samples_match(output_block, expected, produced);
}

/** Verify invalid arguments, nonfinite input/state, empty calls and reset recovery. */
static bool check_invalid_input(void)
{
    EnvelopeState state;
    size_t produced = 99U;
    envelope_reset(NULL); /* Documented no-op. */
    envelope_reset(&state);
    if (!envelope_process(&state, NULL, NULL, 0U, &produced) || produced != 0U
        || state.phase != 0U || !state.valid)
        return false;
    if (envelope_process(NULL, test_block, output_block, 1U, &produced) || produced != 0U)
        return false;
    if (envelope_process(&state, NULL, output_block, 1U, &produced) || state.valid || produced != 0U)
        return false;
    if (envelope_process(&state, test_block, output_block, 1U, &produced))
        return false;
    envelope_reset(&state);
    if (envelope_process(&state, test_block, NULL, 1U, &produced) || state.valid)
        return false;
    envelope_reset(&state);
    if (envelope_process(&state, test_block, output_block, 1U, NULL) || state.valid)
        return false;

    const float bad_values[] = {NAN, INFINITY, -INFINITY};
    for (size_t index = 0U; index < 3U; ++index)
    {
        envelope_reset(&state);
        test_block[0] = 1.0f;
        test_block[1] = bad_values[index]; /* Even an unselected input must be checked. */
        produced = 99U;
        if (envelope_process(&state, test_block, output_block, 2U, &produced)
            || state.valid || produced != 0U)
            return false;
    }
    envelope_reset(&state);
    state.delay[0] = INFINITY;
    if (envelope_process(&state, test_block, output_block, 1U, &produced) || state.valid)
        return false;
    envelope_reset(&state);
    state.phase = 6U;
    if (envelope_process(&state, test_block, output_block, 1U, &produced) || state.valid)
        return false;

    envelope_reset(&state);
    memset(test_block, 0, sizeof(test_block));
    if (!envelope_process(&state, test_block, output_block, 3U, &produced)
        || produced != 1U || state.phase != 3U || output_block[0] != 0.0f)
        return false;
    if (!envelope_process(&state, NULL, NULL, 0U, &produced)
        || produced != 0U || state.phase != 3U)
        return false;
    /* A short block can legitimately retain no samples, while phase still advances. */
    return envelope_process(&state, test_block, output_block, 2U, &produced)
        && produced == 0U && state.phase == 5U && state.valid;
}

/** Verify approximately unity low-pass DC gain after a constant magnitude settles. */
static bool check_dc_gain(void)
{
    EnvelopeState state;
    envelope_reset(&state);
    for (size_t index = 0U; index < 256U; ++index)
        test_block[index] = index % 2U == 0U ? 1.0f : -1.0f;
    size_t produced = 0U;
    for (size_t block = 0U; block < 32U; ++block)
        if (!envelope_process(&state, test_block, output_block, 256U, &produced))
            return false;
    return produced > 0U && isfinite(output_block[produced - 1U])
        && fabsf(output_block[produced - 1U] - 1.0f) <= 4.0e-5f;
}

/** Compare every retained sample in two consecutive nominal 20-revolution windows. */
static bool check_window_mode(size_t mode)
{
    const size_t samples = mode == 0U ? 20571U : 10000U;
    EnvelopeState states[COE_CHANNEL_COUNT];
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        envelope_reset(&states[channel]);
    for (size_t window = 0U; window < 2U; ++window)
    {
        size_t retained = 0U;
        for (size_t offset = 0U; offset < samples; offset += COE_DMA_BLOCK_SAMPLES)
        {
            const size_t remaining = samples - offset;
            const size_t take = remaining < COE_DMA_BLOCK_SAMPLES ? remaining : COE_DMA_BLOCK_SAMPLES;
            const size_t global_index = window * samples + offset;
            const size_t first_kept = (6U - global_index % 6U) % 6U;
            const size_t expected_count = first_kept < take ? 1U + (take - 1U - first_kept) / 6U : 0U;
            for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
            {
                for (size_t index = 0U; index < take; ++index)
                    test_block[index] = bandpass_test_expected[channel][(global_index + index) % 1024U];
                size_t produced;
                if (!envelope_process(&states[channel], test_block, output_block, take, &produced)
                    || produced != expected_count
                    || retained + produced > envelope_expected_counts[mode][window]
                    || states[channel].phase != (global_index + take) % 6U
                    || !samples_match(output_block, envelope_expected_windows[mode][window][channel] + retained, produced))
                    return false;
            }
            retained += expected_count;
        }
        if (retained != envelope_expected_counts[mode][window])
            return false;
        ++envelope_checked_windows;
        /* No filter or phase reset here: the next window continues this stream. */
    }
    return true;
}

/** Measure only envelope processing for one 256-sample block on all four channels. */
static bool measure_block(void)
{
    EnvelopeState states[COE_CHANNEL_COUNT];
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        envelope_reset(&states[channel]);
    envelope_timer_ready = benchmark_init();
    bool success = true;
    const uint32_t start = benchmark_cycles_now();
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        size_t produced;
        if (!envelope_process(&states[channel], bandpass_test_expected[channel],
                               output_block, COE_DMA_BLOCK_SAMPLES, &produced)
            || produced != 43U)
            success = false;
    }
    const uint32_t end = benchmark_cycles_now();
    if (envelope_timer_ready)
        envelope_block_time_us = benchmark_cycles_to_us(end - start);
    return success;
}

/** Run the component checks once, then publish their aggregate status and block time. */
void run_envelope_demo(void)
{
    envelope_checks_passed = false;
    envelope_checked_channel_runs = 0U;
    envelope_checked_windows = 0U;
    envelope_failed_check = 0U;
    envelope_max_abs_error = 0.0f;
    envelope_timer_ready = false;
    envelope_block_time_us = 0.0f;
    const size_t block_sizes[] = {1U, 17U, 256U, 1024U};
    for (size_t test = 0U; test < 4U; ++test)
    {
        if (!check_partition(block_sizes[test]))
        {
            envelope_failed_check = (uint32_t)test + 1U;
            return;
        }
    }
    if (!check_continuity_and_inplace())
        envelope_failed_check = 5U;
    else if (!check_invalid_input())
        envelope_failed_check = 6U;
    else if (!check_dc_gain())
        envelope_failed_check = 7U;
    else if (!check_window_mode(0U))
        envelope_failed_check = 8U;
    else if (!check_window_mode(1U))
        envelope_failed_check = 9U;
    if (envelope_failed_check != 0U)
        return;
    if (!measure_block())
    {
        envelope_failed_check = 10U;
        return;
    }
    envelope_checks_passed = envelope_checked_channel_runs == 16U
        && envelope_checked_windows == 4U;
}
