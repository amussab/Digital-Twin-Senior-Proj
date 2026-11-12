/**
 * @file time_features_reference.h
 * Test-only two-pass calculation used to check the streaming statistics.
 * Production processing uses time_features_reset/update/finalize instead.
 */
#ifndef COE_TIME_FEATURES_REFERENCE_H
#define COE_TIME_FEATURES_REFERENCE_H

#include "time_features.h"

/**
 * Calculate one complete channel window's features in two passes.
 * samples contains count signed band-pass samples; result receives the features.
 * Return false for invalid input or undefined features and clear a non-NULL
 * result. RMS has input units; kurtosis and crest factor are dimensionless.
 */
bool calculate_time_domain_features(const float *samples, size_t count,
                                    TimeDomainFeatures *result);

#endif
