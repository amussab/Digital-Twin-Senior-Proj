/**
 * @file raw_monitor.h
 * LIVE_RAW bring-up: B1 starts/stops the AD7606, and once per second the
 * measured rate, error counters and per-slot raw statistics are printed on
 * the ST-LINK virtual COM port (USART3, 115200 8N1). No DSP, no packets.
 */
#ifndef COE_RAW_MONITOR_H
#define COE_RAW_MONITOR_H

#include <stdint.h>
#include "ad7606.h"

/* Latest one-second statistics per serial slot V1..V8, in ADC counts (Watch). */
extern volatile float raw_monitor_mean_counts[AD7606_SLOTS];
extern volatile float raw_monitor_rms_counts[AD7606_SLOTS]; /* About the mean. */
extern volatile int32_t raw_monitor_pp_counts[AD7606_SLOTS];
extern volatile uint32_t raw_monitor_clipped[AD7606_SLOTS];  /* Since start. */
extern volatile float raw_monitor_rate_hz;         /* Latest second. */
extern volatile float raw_monitor_average_rate_hz; /* Since start. */
extern volatile uint32_t raw_monitor_start_result; /* ad7606_result_t. */

/** Print the mode banner. The ADC pins must already be held safe. */
void raw_monitor_init(void);

/** Call continuously from the main loop; never from an interrupt. */
void raw_monitor_poll(void);

#endif
