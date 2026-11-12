/**
 * @file bandpass_demo.h
 * Board-test entry point and volatile results for Debug Watch.
 */
#ifndef COE_BANDPASS_DEMO_H
#define COE_BANDPASS_DEMO_H

#include <stdbool.h>
#include <stdint.h>

extern volatile bool bandpass_checks_passed;
extern volatile bool bandpass_continuity_checks_passed;
extern volatile uint32_t bandpass_checked_channel_runs;
extern volatile uint32_t bandpass_failed_check;
extern volatile float bandpass_max_abs_error;
extern volatile bool bandpass_timer_ready;
extern volatile uint32_t bandpass_block_cycles;
extern volatile float bandpass_block_time_us;

/** Run filter checks once and measure one block for all four channels. */
void run_bandpass_demo(void);

#endif
