/**
 * @file envelope.c
 * Form and reduce a classical envelope using CMSIS-DSP absolute value and biquads.
 * The small sample-selection function carries its phase between every call.
 */
#include "envelope.h"
#include "pipeline_config.h"
#include "envelope_coefficients.h"
#include "dsp/basic_math_functions.h"
#include "dsp/filtering_functions.h"

#include <math.h>

_Static_assert(COE_INPUT_SAMPLE_RATE_HZ == ENVELOPE_DESIGN_SAMPLE_RATE_HZ,
               "Regenerate envelope coefficients for the new sample rate.");
_Static_assert(COE_ENVELOPE_PASSBAND_HZ == ENVELOPE_DESIGN_PASSBAND_HZ
               && COE_ENVELOPE_STOPBAND_HZ == ENVELOPE_DESIGN_STOPBAND_HZ,
               "Regenerate envelope coefficients for the new filter edges.");
_Static_assert(ENVELOPE_SECTION_COUNT == ENVELOPE_DESIGN_SECTIONS,
               "Envelope delays and coefficients must have matching sections.");
_Static_assert(COE_ENVELOPE_DECIMATION_FACTOR == ENVELOPE_DESIGN_DECIMATION_FACTOR,
               "Revalidate envelope filtering and fixtures for a new factor.");
_Static_assert(COE_INPUT_SAMPLE_RATE_HZ % COE_ENVELOPE_DECIMATION_FACTOR == 0U,
               "Envelope output sample rate must be an integer.");

/** Check that every value is a finite number (neither NaN nor infinity). */
static bool samples_are_finite(const float *samples, size_t count)
{
    for (size_t index = 0U; index < count; ++index)
    {
        if (!isfinite(samples[index]))
        {
            return false;
        }
    }
    return true;
}

/** Rectify and low-pass one block; the descriptor reuses existing filter delays. */
static void filter_envelope(EnvelopeState *state, const float *input,
                             float *output, size_t count)
{
    arm_abs_f32(input, output, (uint32_t)count);
    const arm_biquad_cascade_df2T_instance_f32 filter = {
        .numStages = ENVELOPE_SECTION_COUNT,
        .pState = state->delay,
        .pCoeffs = envelope_cmsis_coefficients
    };
    arm_biquad_cascade_df2T_f32(&filter, output, output, (uint32_t)count);
}

/** Keep one filtered sample in six, compacting the buffer and advancing phase. */
static size_t select_envelope_samples(EnvelopeState *state, float *samples,
                                       size_t count)
{
    size_t retained = 0U;
    for (size_t index = 0U; index < count; ++index)
    {
        if (state->phase == 0U)
        {
            samples[retained++] = samples[index];
        }
        ++state->phase;
        if (state->phase == COE_ENVELOPE_DECIMATION_FACTOR)
        {
            state->phase = 0U;
        }
    }
    return retained;
}

/** Initialize one channel's filter and selection phase for a new stream. */
void envelope_reset(EnvelopeState *state)
{
    if (state != NULL)
    {
        arm_biquad_cascade_df2T_instance_f32 filter;
        arm_biquad_cascade_df2T_init_f32(&filter, ENVELOPE_SECTION_COUNT,
                                        envelope_cmsis_coefficients, state->delay);
        state->phase = 0U;
        state->valid = true;
    }
}

/** Validate a block, form its envelope, then return only the selected samples. */
bool envelope_process(EnvelopeState *state, const float *input, float *output,
                       size_t count, size_t *output_count)
{
    if (output_count == NULL)
    {
        if (state != NULL)
        {
            state->valid = false;
        }
        return false;
    }
    *output_count = 0U;
    if (state == NULL || !state->valid)
    {
        return false;
    }
    if (count == 0U)
    {
        return true;
    }
    if (input == NULL || output == NULL || count > UINT32_MAX
        || state->phase >= COE_ENVELOPE_DECIMATION_FACTOR)
    {
        state->valid = false;
        return false;
    }
    if (!samples_are_finite(input, count)
        || !samples_are_finite(state->delay, 2U * ENVELOPE_SECTION_COUNT))
    {
        state->valid = false;
        return false;
    }

    filter_envelope(state, input, output, count);
    if (!samples_are_finite(output, count)
        || !samples_are_finite(state->delay, 2U * ENVELOPE_SECTION_COUNT))
    {
        state->valid = false;
        return false;
    }
    *output_count = select_envelope_samples(state, output, count);
    return true;
}
