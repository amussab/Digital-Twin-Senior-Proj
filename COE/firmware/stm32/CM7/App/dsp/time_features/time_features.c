/**
 * @file time_features.c
 * Accumulate one channel's time statistics across blocks of one window.
 * Finalize RMS, Pearson kurtosis and crest factor without storing the window.
 */
#include "time_features.h"

#include <math.h>

/** Start a new statistics window; filter state is managed separately. */
void time_features_reset(TimeFeatureState *state)
{
    if (state != NULL)
    {
        *state = (TimeFeatureState){.valid = true};
    }
}

/** Add this block's samples to the current window in their original order. */
bool time_features_update(TimeFeatureState *state, const float *samples,
                          size_t count)
{
    if (state == NULL || !state->valid)
    {
        return false;
    }
    if (count == 0U)
    {
        return true;
    }
    if (samples == NULL || count > (size_t)(UINT32_MAX - state->count))
    {
        state->valid = false;
        return false;
    }

    for (size_t index = 0U; index < count; ++index)
    {
        const double sample = samples[index];
        if (!isfinite(sample))
        {
            state->valid = false;
            return false;
        }

        /* Incremental centered-moment recurrence (Pebay, SAND2008-6212).
         * The order matters: M4 and M3 must use the previous M2/M3.
         * This follows individual samples, so DMA block size does not
         * change their arithmetic order. M3 is needed to update M4.
         */
        const double old_count = (double)state->count;
        const double new_count = old_count + 1.0;
        const double delta = sample - state->mean;
        const double delta_over_count = delta / new_count;
        const double delta_over_count_squared = delta_over_count
                                                * delta_over_count;
        const double correction = delta * delta_over_count * old_count;

        state->m4 += correction * delta_over_count_squared
                     * (new_count * new_count - 3.0 * new_count + 3.0)
                     + 6.0 * delta_over_count_squared * state->m2
                     - 4.0 * delta_over_count * state->m3;
        state->m3 += correction * delta_over_count * (new_count - 2.0)
                     - 3.0 * delta_over_count * state->m2;
        state->m2 += correction;
        state->mean += delta_over_count;
        state->sum_squares += sample * sample;
        const double magnitude = fabs(sample);
        if (magnitude > state->absolute_peak)
        {
            state->absolute_peak = magnitude;
        }
        ++state->count;
    }

    if (!isfinite(state->mean) || !isfinite(state->m2)
        || !isfinite(state->m3) || !isfinite(state->m4)
        || !isfinite(state->sum_squares))
    {
        state->valid = false;
        return false;
    }
    return true;
}

/** Read the completed window's three features without changing its state. */
bool time_features_finalize(const TimeFeatureState *state,
                            TimeDomainFeatures *result)
{
    if (result == NULL)
    {
        return false;
    }
    *result = (TimeDomainFeatures){0};
    if (state == NULL || !state->valid || state->count < 2U
        || !(state->m2 > 0.0) || !(state->sum_squares > 0.0)
        || !(state->m4 >= 0.0))
    {
        return false;
    }

    const double count = (double)state->count;
    const double rms = sqrt(state->sum_squares / count);
    const TimeDomainFeatures calculated = {
        .rms = (float)rms,
        .kurtosis = (float)(count * state->m4 / (state->m2 * state->m2)),
        .crest_factor = (float)(state->absolute_peak / rms)
    };
    if (!isfinite(calculated.rms) || !isfinite(calculated.kurtosis)
        || !isfinite(calculated.crest_factor))
    {
        return false;
    }
    *result = calculated;
    return true;
}
