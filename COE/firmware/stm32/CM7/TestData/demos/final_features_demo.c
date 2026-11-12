/**
 * @file final_features_demo.c
 * Check the last three pipeline stages against independent Python fixtures.
 * Test comparisons are separate from the pipeline's measured processing calls.
 */
#include "final_features_demo.h"
#include "frequency_features.h"
#include "feature_payload.h"
#include "final_spectrum_fixtures.h"
#include "final_payload_fixture.h"
#include <math.h>
#include <string.h>

EnvelopeSpectrumWorkspace final_test_workspace;
volatile bool fft_checks_passed;
volatile uint32_t fft_checked_records;
volatile uint32_t fft_failed_check;
volatile float fft_max_amplitude_error;
volatile float fft_max_density_error;
volatile bool frequency_checks_passed;
volatile uint32_t frequency_checked_records;
volatile uint32_t frequency_failed_check;
volatile bool payload_checks_passed;
volatile uint32_t payload_failed_check;

/* Small reusable test objects, rather than large automatic workspace allocations. */
static TimeDomainFeatures test_time[COE_CHANNEL_COUNT];
static FrequencyDomainFeatures test_frequency[COE_CHANNEL_COUNT];
static AiFeaturePayload test_payload;

/** Compare one computed value with a finite Python reference. */
static bool near(float actual, float expected, float tolerance)
{
    return isfinite(actual) && fabsf(actual - expected) <= tolerance * (1.0f + fabsf(expected));
}

/** Compare every amplitude/PSD bin for one complete synthetic record. */
static bool check_spectrum(size_t record)
{
    if (!calculate_envelope_spectrum(&final_test_workspace, final_fft_input[record],
                                      final_fft_counts[record]))
        return false;
    for (size_t bin = 0U; bin < COE_SPECTRUM_BINS; ++bin)
    {
        const float a_error = fabsf(final_test_workspace.amplitude[bin] - final_fft_amplitude[record][bin]);
        const float p_error = fabsf(final_test_workspace.power_density[bin] - final_fft_density[record][bin]);
        if (a_error > fft_max_amplitude_error)
            fft_max_amplitude_error = a_error;
        if (p_error > fft_max_density_error)
            fft_max_density_error = p_error;
        if (!near(final_test_workspace.amplitude[bin], final_fft_amplitude[record][bin], 2.0e-5f)
            || !near(final_test_workspace.power_density[bin], final_fft_density[record][bin], 2.0e-5f))
            return false;
    }
    ++fft_checked_records;
    return true;
}

/** Check all five frequency features using the already verified spectrum at both speeds. */
static bool check_frequency(size_t record)
{
    for (size_t mode = 0U; mode < 2U; ++mode)
    {
        FrequencyDomainFeatures actual;
        const FrequencyDomainFeatures *expected = &final_frequency_expected[record][mode];
        if (!extract_frequency_features(final_fft_input[record], final_fft_counts[record],
                    final_test_workspace.power_density, mode == 0U ? 1750U : 3600U,
                    &coe_cwru_drive_end_bearing, &actual)
            || !near(actual.envelope_rms, expected->envelope_rms, 4.0e-5f))
            return false;
        for (size_t family = 0U; family < COE_BEARING_FAMILIES; ++family)
            if (!near(actual.band_rms[family], expected->band_rms[family], 4.0e-5f))
                return false;
        ++frequency_checked_records;
    }
    return true;
}

/** Reject invalid FFT inputs and prove a valid record remains callable after rejection. */
static bool check_fft_api(void)
{
    float bad[] = {0.1f, NAN, 0.3f};
    if (envelope_spectrum_init(NULL)
        || calculate_envelope_spectrum(NULL, bad, 3U)
        || calculate_envelope_spectrum(&final_test_workspace, NULL, 3U)
        || calculate_envelope_spectrum(&final_test_workspace, final_fft_input[0], 2U)
        || calculate_envelope_spectrum(&final_test_workspace, final_fft_input[0], COE_MAX_ENVELOPE_SAMPLES + 1U)
        || calculate_envelope_spectrum(&final_test_workspace, bad, 3U))
        return false;
    final_test_workspace.ready = false;
    if (calculate_envelope_spectrum(&final_test_workspace, final_fft_input[0], final_fft_counts[0]))
        return false;
    return envelope_spectrum_init(&final_test_workspace)
        && calculate_envelope_spectrum(&final_test_workspace, final_fft_input[0], final_fft_counts[0]);
}

/** Reject unsupported profiles/PSD while retaining zero-power valid feature semantics. */
static bool check_frequency_api(void)
{
    BearingFeatureProfile bad = coe_project_bearing;
    FrequencyDomainFeatures result;
    if (!frequency_profile_is_valid(&coe_project_bearing, 1750U)
        || !frequency_profile_is_valid(&coe_project_bearing, 3600U)
        || frequency_profile_is_valid(NULL, 1750U)
        || frequency_profile_is_valid(&coe_project_bearing, 0U))
        return false;
    bad.orders[COE_BPFI] = 10.0f;
    if (frequency_profile_is_valid(&bad, 3600U))
        return false;
    bad = coe_project_bearing;
    bad.orders[COE_BPFI] = bad.orders[COE_BPFO];
    if (frequency_profile_is_valid(&bad, 1750U))
        return false;
    bad = coe_project_bearing;
    bad.half_width_order = NAN;
    if (frequency_profile_is_valid(&bad, 1750U))
        return false;
    float *density = final_test_workspace.power_density;
    const float saved = density[10];
    density[10] = -1.0f;
    const bool rejected_negative = !extract_frequency_features(final_fft_input[0], final_fft_counts[0],
                                       density, 1750U, &coe_project_bearing, &result);
    density[10] = NAN;
    const bool rejected_nan = !extract_frequency_features(final_fft_input[0], final_fft_counts[0],
                                       density, 1750U, &coe_project_bearing, &result);
    density[10] = saved;
    return rejected_negative && rejected_nan
        && !extract_frequency_features(NULL, 3U, density, 1750U, &coe_project_bearing, &result)
        && !extract_frequency_features(final_fft_input[0], 3U, density, 1750U, &coe_project_bearing, NULL);
}

/** Check exact channel ordering, little-endian bytes, CRC and invalid-channel masks. */
static bool check_payload(void)
{
    bool valid[COE_CHANNEL_COUNT] = {true, true, true, true};
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        const float base = (float)(channel * 10U);
        test_time[channel] = (TimeDomainFeatures){base + 1.0f, base + 2.0f, base + 3.0f};
        test_frequency[channel].envelope_rms = base + 4.0f;
        for (size_t family = 0U; family < COE_BEARING_FAMILIES; ++family)
            test_frequency[channel].band_rms[family] = base + 5.0f + (float)family;
    }
    if (!build_ai_payload(test_time, test_frequency, valid, 17U, 1750U,
                           COE_PROJECT_BEARING, &test_payload)
        || test_payload.channel_valid_mask != 15U
        || memcmp(test_payload.bytes, final_payload_expected, COE_PAYLOAD_BYTES) != 0)
        return false;
    valid[1] = false;
    test_time[1].rms = NAN; /* Undefined data in an explicitly invalid channel is not used. */
    if (!build_ai_payload(test_time, test_frequency, valid, 17U, 3600U,
                           COE_PROJECT_BEARING, &test_payload)
        || test_payload.channel_valid_mask != 13U)
        return false;
    for (size_t feature = 8U; feature < 16U; ++feature)
        if (test_payload.features[feature] != 0.0f)
            return false;
    valid[1] = true;
    if (build_ai_payload(test_time, test_frequency, valid, 17U, 1750U,
                          COE_PROJECT_BEARING, &test_payload)
        || test_payload.channel_valid_mask != 0U)
        return false;
    test_time[1].rms = 11.0f;
    return !build_ai_payload(NULL, test_frequency, valid, 17U, 1750U, COE_PROJECT_BEARING, &test_payload)
        && !build_ai_payload(test_time, test_frequency, valid, 0U, 1750U, COE_PROJECT_BEARING, &test_payload)
        && !build_ai_payload(test_time, test_frequency, valid, 17U, 1000U, COE_PROJECT_BEARING, &test_payload);
}

/** Run all last-stage checks once and publish concise Watch status/counters. */
void run_final_features_demo(void)
{
    fft_checks_passed = frequency_checks_passed = payload_checks_passed = false;
    fft_checked_records = frequency_checked_records = 0U;
    fft_failed_check = frequency_failed_check = payload_failed_check = 0U;
    fft_max_amplitude_error = fft_max_density_error = 0.0f;
    if (!envelope_spectrum_init(&final_test_workspace))
    {
        fft_failed_check = 10U;
        return;
    }
    for (size_t record = 0U; record < 9U; ++record)
    {
        if (!check_spectrum(record))
        {
            fft_failed_check = (uint32_t)record + 1U;
            return;
        }
        if (!check_frequency(record))
        {
            frequency_failed_check = (uint32_t)record + 1U;
            return;
        }
    }
    if (!check_fft_api())
    {
        fft_failed_check = 11U;
        return;
    }
    fft_checks_passed = fft_checked_records == 9U;
    if (!check_frequency_api())
    {
        frequency_failed_check = 10U;
        return;
    }
    frequency_checks_passed = frequency_checked_records == 18U;
    payload_checks_passed = check_payload();
    if (!payload_checks_passed)
        payload_failed_check = 1U;
}
