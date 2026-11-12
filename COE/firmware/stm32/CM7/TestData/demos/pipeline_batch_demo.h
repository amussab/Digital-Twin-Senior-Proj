/**
 * @file pipeline_batch_demo.h
 * Board-test entry point and volatile results for Debug Watch.
 */
#ifndef COE_PIPELINE_BATCH_DEMO_H
#define COE_PIPELINE_BATCH_DEMO_H

#include <stdbool.h>
#include <stdint.h>
#include "feature_payload.h"

extern volatile bool pipeline_batch_checks_passed;
extern volatile uint32_t pipeline_batch_checked_windows;
extern volatile uint32_t pipeline_batch_failed_check;
extern volatile bool pipeline_batch_timer_ready;
extern volatile float pipeline_batch_max_ms_1750;
extern volatile float pipeline_batch_max_ms_3600;
extern volatile uint32_t pipeline_batch_envelope_samples_1750[3];
extern volatile uint32_t pipeline_batch_envelope_samples_3600[3];

/** Run both nominal speeds and publish complete-window timing/check results. */
void run_pipeline_batch_demo(void);

extern volatile bool pipeline_batch_fixed_shape_passed;
extern volatile bool pipeline_batch_deadline_met;
extern volatile uint32_t pipeline_batch_feature_count_1750;
extern volatile uint32_t pipeline_batch_feature_count_3600;
extern volatile uint32_t pipeline_batch_fft_size;
extern volatile float pipeline_batch_fft_bin_spacing_hz;
extern volatile uint32_t pipeline_batch_last_profile_id;
extern volatile float pipeline_batch_features_1750[COE_FEATURE_COUNT];
extern volatile float pipeline_batch_features_3600[COE_FEATURE_COUNT];
extern volatile uint8_t pipeline_batch_payload_1750[COE_PAYLOAD_BYTES];
extern volatile uint8_t pipeline_batch_payload_3600[COE_PAYLOAD_BYTES];

#endif
