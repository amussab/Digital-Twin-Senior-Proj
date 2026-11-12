/**
 * @file time_features_reference.c
 * Independent two-pass statistics reference used only by board tests.
 */
#include "time_features_reference.h"

#include <math.h>

/** Calculate whole-window features in two passes for comparison with streaming results. */
bool calculate_time_domain_features(const float *samples, size_t count,
                                    TimeDomainFeatures *result)
{
    if (result == NULL)
    {
        return false;
    }
    *result = (TimeDomainFeatures){0};

    if (samples == NULL || count < 2U)
    {
        return false;
    }

    /* Use double accumulators to reduce rounding error in long windows.
     * The input and final features remain float32.
     */
    double sum = 0.0;
    double sum_squares = 0.0;
    double absolute_peak = 0.0;

    for (size_t index = 0U; index < count; ++index)
    {
        const double sample = samples[index];
        if (!isfinite(sample))
        {
            return false;
        }

        sum += sample;
        sum_squares += sample * sample;
        const double magnitude = fabs(sample);
        if (magnitude > absolute_peak)
        {
            absolute_peak = magnitude;
        }
    }

    const double mean = sum / (double)count;
    const double rms = sqrt(sum_squares / (double)count);
    double sum_centered_squares = 0.0;
    double sum_centered_fourths = 0.0;

    /* A second pass uses central moments, matching the Python reference.
     * It avoids subtracting large raw moments to obtain the variance.
     */
    for (size_t index = 0U; index < count; ++index)
    {
        const double centered = (double)samples[index] - mean;
        const double centered_square = centered * centered;
        sum_centered_squares += centered_square;
        sum_centered_fourths += centered_square * centered_square;
    }

    const double variance = sum_centered_squares / (double)count;
    if (rms == 0.0 || variance == 0.0)
    {
        /* Zero RMS makes crest factor undefined; a constant signal makes
         * kurtosis undefined. These are invalid results, not fault labels.
         */
        return false;
    }

    const TimeDomainFeatures calculated = {
        .rms = (float)rms,
        .kurtosis = (float)((sum_centered_fourths / (double)count)
                           / (variance * variance)),
        .crest_factor = (float)(absolute_peak / rms)
    };
    if (!isfinite(calculated.rms) || !isfinite(calculated.kurtosis)
        || !isfinite(calculated.crest_factor))
    {
        return false;
    }

    *result = calculated;
    return true;
}
