/**
 * @file frequency_features.c
 * Match Python step 07: envelope RMS plus FTF/BSF/BPFO/BPFI fundamental-band RMS.
 */
#include "frequency_features.h"
#include "dsp/statistics_functions.h"
#include <math.h>

const BearingFeatureProfile coe_project_bearing = {
    .id = COE_PROJECT_BEARING, .orders = {0.38f, 1.98f, 3.05f, 4.95f}, .half_width_order = 0.10f
};
const BearingFeatureProfile coe_cwru_drive_end_bearing = {
    .id = COE_CWRU_DRIVE_END_BEARING,
    .orders = {0.39828f, 2.35675f, 3.5848f, 5.4152f}, .half_width_order = 0.10f
};

/** Band half width in Hz: the profile's order width, but never narrower than one
 * FFT bin, so slow shafts (for example 250 RPM) still get a band with bins in it.
 * At 1750 and 3600 RPM the order width is wider, so results are unchanged. */
static double band_half_width_hz(const BearingFeatureProfile *profile, double shaft_hz)
{
    const double spacing = (double)COE_ENVELOPE_SAMPLE_RATE_HZ / COE_FFT_SIZE;
    const double width = (double)profile->half_width_order * shaft_hz;
    return width > spacing ? width : spacing;
}

/** Check finite positive orders and disjoint supported bands at the selected speed. */
bool frequency_profile_is_valid(const BearingFeatureProfile *profile, uint32_t rpm)
{
    if (profile == NULL || rpm == 0U
        || (profile->id != COE_PROJECT_BEARING && profile->id != COE_CWRU_DRIVE_END_BEARING)
        || !isfinite(profile->half_width_order) || profile->half_width_order <= 0.0f)
        return false;
    const double shaft_hz = (double)rpm / 60.0;
    const double width_hz = band_half_width_hz(profile, shaft_hz);
    for (size_t family = 0U; family < COE_BEARING_FAMILIES; ++family)
    {
        const double centre_hz = (double)profile->orders[family] * shaft_hz;
        if (!isfinite(centre_hz) || centre_hz <= width_hz
            || centre_hz + width_hz > COE_ENVELOPE_PASSBAND_HZ)
            return false;
        for (size_t other = 0U; other < family; ++other)
            if (fabs(centre_hz - (double)profile->orders[other] * shaft_hz) <= 2.0 * width_hz)
                return false;
    }
    return true;
}

/** Integrate bins in one closed frequency band; use double accumulation for the short sum. */
static float calculate_band_rms(const float *density, double low, double high)
{
    const double spacing = (double)COE_ENVELOPE_SAMPLE_RATE_HZ / COE_FFT_SIZE;
    const size_t first = (size_t)ceil(low / spacing);
    const size_t last = (size_t)floor(high / spacing);
    double power = 0.0;
    for (size_t bin = first; bin <= last; ++bin)
        power += density[bin];
    return (float)sqrt(power * spacing);
}

/** Validate the record/spectrum, then calculate the five defined envelope features. */
bool extract_frequency_features(const float *envelope, size_t count,
                                 const float power_density[COE_SPECTRUM_BINS],
                                 uint32_t rpm, const BearingFeatureProfile *profile,
                                 FrequencyDomainFeatures *features)
{
    if (features == NULL)
        return false;
    *features = (FrequencyDomainFeatures){0};
    if (envelope == NULL || power_density == NULL || count < 3U
        || count > COE_MAX_ENVELOPE_SAMPLES || !frequency_profile_is_valid(profile, rpm))
        return false;
    for (size_t sample = 0U; sample < count; ++sample)
        if (!isfinite(envelope[sample]))
            return false;
    for (size_t bin = 0U; bin < COE_SPECTRUM_BINS; ++bin)
        if (!isfinite(power_density[bin]) || power_density[bin] < 0.0f)
            return false;
    FrequencyDomainFeatures result = {0};
    arm_rms_f32(envelope, (uint32_t)count, &result.envelope_rms);
    if (!isfinite(result.envelope_rms))
        return false;
    const double shaft_hz = (double)rpm / 60.0;
    const double width_hz = band_half_width_hz(profile, shaft_hz);
    for (size_t family = 0U; family < COE_BEARING_FAMILIES; ++family)
    {
        const double centre_hz = (double)profile->orders[family] * shaft_hz;
        result.band_rms[family] = calculate_band_rms(power_density,
            centre_hz - width_hz, centre_hz + width_hz);
        if (!isfinite(result.band_rms[family]))
            return false;
    }
    *features = result;
    return true;
}
