/**
 * @file time_features_demo.h
 * Board-test entry point and volatile results for Debug Watch.
 */
#ifndef COE_TIME_FEATURES_DEMO_H
#define COE_TIME_FEATURES_DEMO_H

#include "time_features.h"

extern volatile TimeDomainFeatures demo_features;
extern volatile bool demo_success;
extern volatile bool demo_checks_passed;
extern volatile bool demo_timer_ready;
extern volatile uint32_t demo_cycles;
extern volatile float demo_time_us;
extern volatile bool demo_streaming_checks_passed;
extern volatile bool demo_window_checks_passed;
extern volatile uint32_t demo_checked_channel_windows;
extern volatile uint32_t demo_failed_check;

/** Run the statistics checks once and record the small smoke-test duration. */
void run_time_features_demo(void);

#endif
