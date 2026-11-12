/**
 * @file time_features.h
 * Accumulate one channel's statistics across blocks, then finish its window.
 * Features use the signed band-pass output before envelope rectification.
 */
#ifndef COE_TIME_FEATURES_H
#define COE_TIME_FEATURES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Final results for one channel's complete analysis window. */
typedef struct
{
    float rms;          /* Input acceleration units (g if the input is in g). */
    float kurtosis;     /* Pearson kurtosis, dimensionless; Gaussian value is 3. */
    float crest_factor; /* Largest absolute sample / RMS, dimensionless. */
} TimeDomainFeatures;

/**
 * Running statistics for one channel; no samples or heap storage are retained.
 * Centered sums keep the variance/kurtosis calculation numerically stable.
 */
typedef struct
{
    uint32_t count;      /* Number of samples accumulated in the current window. */
    double mean;         /* Running mean of those samples. */
    double m2;           /* Sum of squared deviations from the mean. */
    double m3;           /* Third centered sum; needed when updating m4. */
    double m4;           /* Fourth centered sum; used to calculate kurtosis. */
    double sum_squares;  /* Sum of sample squares, used to calculate RMS. */
    double absolute_peak;
    bool valid;          /* False after invalid input until reset. */
} TimeFeatureState;

/** Start an empty window with valid statistics. A NULL state is ignored. */
void time_features_reset(TimeFeatureState *state);

/**
 * Add count chronological samples to one channel's current window.
 * samples points to at least count floats; input data is not modified or stored.
 * A zero count is a successful no-op for a valid state (samples may be NULL).
 * Return false and invalidate the window for bad input or count overflow.
 * A failed call may have incorporated earlier samples; reset before restarting.
 */
bool time_features_update(TimeFeatureState *state, const float *samples,
                          size_t count);

/**
 * Write RMS, kurtosis and crest factor without changing the accumulated state.
 * Return false for invalid state, fewer than two samples or undefined features
 * (for example constant data). A non-NULL result is cleared on failure.
 * Call reset explicitly when starting the next window.
 */
bool time_features_finalize(const TimeFeatureState *state,
                            TimeDomainFeatures *result);

#endif
