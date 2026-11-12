/**
 * @file acquisition_console.h
 * Shared live-mode console: queued ST-LINK COM output, B1 button, LEDs and
 * the AD7606 start report. Used by LIVE_RAW (raw_monitor) and LIVE (live_stream).
 * Main loop only; never call from an interrupt.
 */
#ifndef COE_ACQUISITION_CONSOLE_H
#define COE_ACQUISITION_CONSOLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "ad7606.h"

/** Queue formatted text; it is dropped if the queue is full. */
void console_emit(const char *format, ...) __attribute__((format(printf, 1, 2)));

/** Write at most 64 queued bytes (about 5.6 ms at 115200 baud). Call each loop. */
void console_flush_some(void);

/** True when nothing is queued: a periodic report may be added. */
bool console_idle(void);

/** Format x with two decimals; newlib-nano printf has no floating point. */
const char *console_fixed2(char *text, size_t size, double x);

/** Debounced B1 press (not release). */
bool console_button_pressed(void);

/** Print the banner shared by both live modes: clock and power order. */
void console_banner(const char *mode_line);

/** Start the AD7606 and print the start-up probe and the result. */
ad7606_result_t console_start_acquisition(void);

/** Green: slow blink idle, fast blink running. Red: any error since start. */
void console_update_leds(bool running, bool error);

#endif
