/** @file final_features_demo.h Standalone last-stage checks and shared sequential test workspace. */
#ifndef COE_FINAL_FEATURES_DEMO_H
#define COE_FINAL_FEATURES_DEMO_H

#include "envelope_spectrum.h"
#include <stdint.h>

extern EnvelopeSpectrumWorkspace final_test_workspace; /* Shared by sequential startup demos only. */
extern volatile bool fft_checks_passed;
extern volatile uint32_t fft_checked_records;
extern volatile uint32_t fft_failed_check;
extern volatile float fft_max_amplitude_error;
extern volatile float fft_max_density_error;
extern volatile bool frequency_checks_passed;
extern volatile uint32_t frequency_checked_records;
extern volatile uint32_t frequency_failed_check;
extern volatile bool payload_checks_passed;
extern volatile uint32_t payload_failed_check;

/** Check all FFT bins, five frequency features, invalid inputs and exact packet encoding. */
void run_final_features_demo(void);

#endif
