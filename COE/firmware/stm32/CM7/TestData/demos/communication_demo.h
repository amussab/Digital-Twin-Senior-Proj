/** @file communication_demo.h
 * Packet-decoder checks and a push-button replay trigger, outside production DSP.
 */
#ifndef COE_COMMUNICATION_DEMO_H
#define COE_COMMUNICATION_DEMO_H
#include <stdbool.h>
#include <stdint.h>

extern volatile bool communication_checks_passed;
extern volatile uint32_t communication_checked_cases;
extern volatile uint32_t communication_failed_check;

/** Check the actual receiver decoder against the Python wire fixture/errors. */
void run_communication_demo(void);

/** Debounce B1 and replay twelve freshly calculated feature packets on a press.
 * Call in the main loop. Replays deliberately restart test window IDs at 1.
 */
void poll_communication_replay(void);
#endif
