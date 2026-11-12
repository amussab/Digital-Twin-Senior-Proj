/**
 * @file envelope_spectrum.h
 * Mean removal, periodic Hann taper and one CMSIS real FFT per complete envelope.
 * A shared workspace supplies peak amplitudes and one-sided power density.
 */
#ifndef COE_ENVELOPE_SPECTRUM_H
#define COE_ENVELOPE_SPECTRUM_H

#include "pipeline_config.h"
#include "dsp/transform_functions.h"
#include <stdbool.h>
#include <stddef.h>

/** Reusable static storage, approximately 63 KB; do not allocate it on the stack. */
typedef struct
{
    arm_rfft_fast_instance_f32 fft;
    float taper[COE_MAX_ENVELOPE_SAMPLES];
    float input[COE_FFT_SIZE];
    float packed[COE_FFT_SIZE];
    float amplitude[COE_SPECTRUM_BINS];       /* g peak for input calibrated in g. */
    float power_density[COE_SPECTRUM_BINS];   /* g^2/Hz, including DC/Nyquist. */
    size_t taper_count;
    float taper_sum;
    float taper_power_sum;
    bool ready;
} EnvelopeSpectrumWorkspace;

/** Initialize the fixed-size transform; invalidate the cached taper, without clearing arrays. */
bool envelope_spectrum_init(EnvelopeSpectrumWorkspace *workspace);

/**
 * Transform a complete finite record of 3..COE_MAX_ENVELOPE_SAMPLES values.
 * Input is unchanged. Subtract its mean, apply a periodic Hann of actual record
 * length, then zero-pad to 4096. Return false on invalid input/numerical failure.
 * Output arrays are borrowed until the next call. Read them only after success.
 * Call from a single context; streams may share this workspace only sequentially.
 */
bool calculate_envelope_spectrum(EnvelopeSpectrumWorkspace *workspace,
                                 const float *envelope, size_t count);

#endif
