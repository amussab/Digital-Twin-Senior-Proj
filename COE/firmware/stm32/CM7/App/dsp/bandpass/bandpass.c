/**
 * @file bandpass.c
 * Filter one vibration channel with the CMSIS-DSP band-pass kernel.
 * This module checks data and manages delays; CMSIS performs the DSP arithmetic.
 */
#include "bandpass.h"
#include "pipeline_config.h"
#include "bandpass_coefficients.h"
#include "dsp/filtering_functions.h"

#include <math.h>
#include <stdint.h>

/* Stop compilation if settings differ from the exported filter design. */
_Static_assert(COE_INPUT_SAMPLE_RATE_HZ == BANDPASS_DESIGN_SAMPLE_RATE_HZ,
               "Regenerate band-pass coefficients for the new sample rate.");
_Static_assert(COE_BANDPASS_LOW_HZ == BANDPASS_DESIGN_LOW_HZ
               && COE_BANDPASS_HIGH_HZ == BANDPASS_DESIGN_HIGH_HZ,
               "Regenerate band-pass coefficients for the new cutoff frequencies.");
_Static_assert(BANDPASS_SECTION_COUNT == BANDPASS_DESIGN_SECTIONS,
               "Band-pass state and coefficient section counts must match.");

/** Return true when every value is a finite number (neither NaN nor infinity). */
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

/** Clear one channel's filter delays and make the state ready for a new stream. */
void bandpass_reset(BandpassState *state)
{
    if (state != NULL)
    {
        arm_biquad_cascade_df2T_instance_f32 filter;
        arm_biquad_cascade_df2T_init_f32(&filter, BANDPASS_SECTION_COUNT,
                                        bandpass_cmsis_coefficients, state->delay);
        state->valid = true;
    }
}

/** Check, filter and validate one channel block, preserving delays for the next. */
bool bandpass_process(BandpassState *state, const float *input,
                      float *output, size_t count)
{
    if (state == NULL || !state->valid)
    {
        return false;
    }
    if (count == 0U)
    {
        return true;
    }
    if (input == NULL || output == NULL || count > UINT32_MAX)
    {
        state->valid = false;
        return false;
    }

    /* Check input before writing output, including for in-place filtering. */
    if (!samples_are_finite(input, count)
        || !samples_are_finite(state->delay, 2U * BANDPASS_SECTION_COUNT))
    {
        state->valid = false;
        return false;
    }

    /* This descriptor only points to existing delays; it does not reset them. */
    const arm_biquad_cascade_df2T_instance_f32 filter = {
        .numStages = BANDPASS_SECTION_COUNT,
        .pState = state->delay,
        .pCoeffs = bandpass_cmsis_coefficients
    };
    arm_biquad_cascade_df2T_f32(&filter, input, output, (uint32_t)count);

    /* CMSIS reports no arithmetic error, so check the result here. */
    if (!samples_are_finite(output, count)
        || !samples_are_finite(state->delay, 2U * BANDPASS_SECTION_COUNT))
    {
        state->valid = false;
        return false;
    }
    return true;
}
