/**
 * @file pipeline_config.h
 * Shared design constants for the four-channel, nominal 20-revolution pipeline.
 */
#ifndef COE_PIPELINE_CONFIG_H
#define COE_PIPELINE_CONFIG_H

/* Current design settings. Changing filter settings requires regenerating
 * bandpass_coefficients.h; bandpass.c checks that the settings agree.
 */
#define COE_CHANNEL_COUNT 4U
#define COE_INPUT_SAMPLE_RATE_HZ 30000U
#define COE_WINDOW_REVOLUTIONS 20U
#define COE_DMA_BLOCK_SAMPLES 256U /* Sample sets; 256 samples per channel. */
#define COE_BANDPASS_LOW_HZ 2000U
#define COE_BANDPASS_HIGH_HZ 8000U

/* Envelope: filter at the input rate, then retain one sample in six. */
#define COE_ENVELOPE_PASSBAND_HZ 350U
#define COE_ENVELOPE_STOPBAND_HZ 1250U
#define COE_ENVELOPE_DECIMATION_FACTOR 6U
#define COE_ENVELOPE_SAMPLE_RATE_HZ (COE_INPUT_SAMPLE_RATE_HZ / COE_ENVELOPE_DECIMATION_FACTOR)

/* One complete-record FFT and eight ordered features per channel. */
#define COE_FFT_SIZE 4096U

/* Any shaft speed in this range is accepted. A window is 20 revolutions when
 * that fits the envelope FFT (about 1500 RPM and above); at slower speeds it is
 * the most whole revolutions that fit (250 RPM: 3 revolutions, 0.72 s).
 */
#define COE_MIN_RPM 60U
#define COE_MAX_RPM 3600U
#define COE_MAX_ENVELOPE_SAMPLES 3600U /* <= COE_FFT_SIZE; 0.72 s, sized to fit RAM_D1 */
#define COE_MAX_WINDOW_SAMPLES (COE_MAX_ENVELOPE_SAMPLES * COE_ENVELOPE_DECIMATION_FACTOR)
#define COE_SPECTRUM_BINS (COE_FFT_SIZE / 2U + 1U)
#define COE_FEATURES_PER_CHANNEL 8U
#define COE_FEATURE_COUNT (COE_CHANNEL_COUNT * COE_FEATURES_PER_CHANNEL)

#endif
