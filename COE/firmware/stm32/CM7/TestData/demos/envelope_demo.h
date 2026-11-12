/**
 * @file envelope_demo.h
 * Board-test entry point and envelope/decimation results for Debug Watch.
 */
#ifndef COE_ENVELOPE_DEMO_H
#define COE_ENVELOPE_DEMO_H

#include <stdbool.h>
#include <stdint.h>

extern volatile bool envelope_checks_passed;
extern volatile uint32_t envelope_checked_channel_runs;
extern volatile uint32_t envelope_checked_windows;
extern volatile uint32_t envelope_failed_check;
extern volatile float envelope_max_abs_error;
extern volatile bool envelope_timer_ready;
extern volatile float envelope_block_time_us;

/** Run envelope/phase checks once and measure one block for all four channels. */
void run_envelope_demo(void);

#endif
