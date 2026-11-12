/**
 * @file envelope_spectrum.c
 * Match Python step 06 using actual-record Hann normalization and CMSIS RFFT.
 */
#include "envelope_spectrum.h"
#include "envelope_tapers.h"
#include "dsp/statistics_functions.h"
#include "dsp/basic_math_functions.h"
#include "dsp/complex_math_functions.h"
#include "dsp/fast_math_functions.h"
#include <math.h>
#include <string.h>

_Static_assert(COE_FFT_SIZE == 4096U, "Reconfigure the selected CMSIS FFT tables.");
_Static_assert(COE_MAX_ENVELOPE_SAMPLES <= COE_FFT_SIZE, "FFT must fit every complete record.");

/** Initialize a transform descriptor and mark cached coefficients unavailable. */
bool envelope_spectrum_init(EnvelopeSpectrumWorkspace *workspace)
{
    if (workspace == NULL)
        return false;
    workspace->ready = false;
    workspace->taper_count = 0U;
    if (arm_rfft_fast_init_f32(&workspace->fft, COE_FFT_SIZE) != ARM_MATH_SUCCESS)
        return false;
    workspace->ready = true;
    return true;
}

/** Cache a periodic Hann taper and its two normalization sums for this record length. */
static void prepare_taper(EnvelopeSpectrumWorkspace *workspace, size_t count)
{
    if (workspace->taper_count == count)
        return;
    double sum = 0.0;
    double power_sum = 0.0;
    const float *preset = count == 3429U ? envelope_taper_3429
                        : count == 3428U ? envelope_taper_3428
                        : count == 1667U ? envelope_taper_1667
                        : count == 1666U ? envelope_taper_1666 : NULL;
    for (size_t index = 0U; index < count; ++index)
    {
        /* Periodic Hann divides by count, not count-1. */
        const float weight = preset != NULL ? preset[index]
            : (float)(0.5 - 0.5 * cos(6.2831853071795864769 * (double)index / (double)count));
        workspace->taper[index] = weight;
        sum += weight;
        power_sum += (double)weight * weight;
    }
    workspace->taper_sum = (float)sum;
    workspace->taper_power_sum = (float)power_sum;
    workspace->taper_count = count;
}

/** Convert the packed real FFT into correctly normalized amplitude and PSD arrays. */
static bool unpack_spectrum(EnvelopeSpectrumWorkspace *workspace)
{
    /* Interior packed pairs are ordinary complex FFT bins; endpoints are special. */
    arm_cmplx_mag_squared_f32(workspace->packed + 2U, workspace->power_density + 1U,
                              COE_SPECTRUM_BINS - 2U);
    workspace->power_density[0] = workspace->packed[0] * workspace->packed[0];
    workspace->power_density[COE_SPECTRUM_BINS - 1U]
        = workspace->packed[1] * workspace->packed[1];
    const double density_scale = (double)COE_ENVELOPE_SAMPLE_RATE_HZ
                                 * workspace->taper_power_sum;
    for (size_t bin = 0U; bin < COE_SPECTRUM_BINS; ++bin)
    {
        /* CMSIS stores DC in packed[0] and Nyquist in packed[1]. */
        const bool endpoint = bin == 0U || bin == COE_FFT_SIZE / 2U;
        const double power = workspace->power_density[bin];
        const double factor = endpoint ? 1.0 : 2.0;
        float magnitude;
        /* Power is Float32 already; CMSIS uses the M7's Float32 square root. */
        if (arm_sqrt_f32((float)power, &magnitude) != ARM_MATH_SUCCESS)
            return false;
        workspace->amplitude[bin] = (float)(factor * magnitude / workspace->taper_sum);
        workspace->power_density[bin] = (float)(factor * power / density_scale);
        if (!isfinite(workspace->amplitude[bin]) || !isfinite(workspace->power_density[bin]))
            return false;
    }
    return true;
}

/** Validate, center/taper the complete record, transform it once and normalize each bin. */
bool calculate_envelope_spectrum(EnvelopeSpectrumWorkspace *workspace,
                                 const float *envelope, size_t count)
{
    if (workspace == NULL || !workspace->ready || envelope == NULL
        || count < 3U || count > COE_MAX_ENVELOPE_SAMPLES)
        return false;
    for (size_t index = 0U; index < count; ++index)
        if (!isfinite(envelope[index]))
            return false;
    prepare_taper(workspace, count);
    float mean;
    arm_mean_f32(envelope, (uint32_t)count, &mean);
    if (!isfinite(mean))
        return false;
    arm_offset_f32(envelope, -mean, workspace->input, (uint32_t)count);
    arm_mult_f32(workspace->input, workspace->taper, workspace->input, (uint32_t)count);
    memset(workspace->input + count, 0, (COE_FFT_SIZE - count) * sizeof(float));
    arm_rfft_fast_f32(&workspace->fft, workspace->input, workspace->packed, 0U);
    return unpack_spectrum(workspace);
}
