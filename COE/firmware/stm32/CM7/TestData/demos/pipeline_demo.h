/**
 * @file pipeline_demo.h
 * Board-test entry point and volatile results for Debug Watch.
 */
#ifndef COE_PIPELINE_DEMO_H
#define COE_PIPELINE_DEMO_H

#include <stdbool.h>
#include <stdint.h>
#include "pipeline.h"

extern volatile bool pipeline_checks_passed;
extern volatile uint32_t pipeline_checked_windows;
extern volatile uint32_t pipeline_failed_check;
extern volatile bool pipeline_timer_ready;
extern volatile float pipeline_max_window_ms_1750;
extern volatile float pipeline_max_window_ms_3600;
extern volatile float pipeline_max_block_time_us;

/** Run the streaming pipeline checks once and publish Debug Watch results. */
void run_pipeline_demo(void);

/** Test-only comparison shared by streaming and batch checks; performs no timing. */
bool pipeline_envelope_matches_reference(const PipelineWindowResult *result,
                                          size_t mode, size_t window);

/** Test-only final-vector comparison, shared with the batch checks. */
bool pipeline_final_matches_reference(const PipelineWindowResult *result,
                                       size_t profile, size_t mode, size_t window);

#endif
