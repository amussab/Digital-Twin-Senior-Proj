/**
 * @file frequency_features.h
 * Original-envelope RMS and four fundamental bearing-order band RMS values.
 */
#ifndef COE_FREQUENCY_FEATURES_H
#define COE_FREQUENCY_FEATURES_H

#include "envelope_spectrum.h"
#include <stdint.h>

enum { COE_FTF = 0, COE_BSF, COE_BPFO, COE_BPFI, COE_BEARING_FAMILIES };
enum { COE_PROJECT_BEARING = 1, COE_CWRU_DRIVE_END_BEARING = 2 };

/** Explicit bearing identity and order configuration; keep it alive for the stream. */
typedef struct
{
    uint32_t id;
    float orders[COE_BEARING_FAMILIES]; /* FTF, single BSF, BPFO, BPFI. */
    float half_width_order;
} BearingFeatureProfile;

extern const BearingFeatureProfile coe_project_bearing; /* Provisional supplied orders. */
extern const BearingFeatureProfile coe_cwru_drive_end_bearing;

/** Five g-valued features, using envelope samples before mean removal for the first. */
typedef struct
{
    float envelope_rms;
    float band_rms[COE_BEARING_FAMILIES]; /* FTF, BSF, BPFO, BPFI, in this order. */
} FrequencyDomainFeatures;

/** Reject invalid/overlapping/unsupported bands, including bands above the envelope passband. */
bool frequency_profile_is_valid(const BearingFeatureProfile *profile, uint32_t rpm);

/**
 * Integrate finite nonnegative one-sided PSD bins whose centers lie in each closed
 * order band: sqrt(sum(PSD) * 5000/4096). Envelope RMS includes its mean.
 * Requires the spectrum calculated from this same complete envelope record.
 * Zero power is valid. False zeroes output; no fault classification is performed.
 */
bool extract_frequency_features(const float *envelope, size_t count,
                                 const float power_density[COE_SPECTRUM_BINS],
                                 uint32_t rpm, const BearingFeatureProfile *profile,
                                 FrequencyDomainFeatures *features);

#endif
