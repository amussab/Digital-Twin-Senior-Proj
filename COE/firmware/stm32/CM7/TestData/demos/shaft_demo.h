/** @file shaft_demo.h
 * Known-answer checks of the Keyphasor speed and proximity-probe values.
 */
#ifndef COE_SHAFT_DEMO_H
#define COE_SHAFT_DEMO_H
#include <stdbool.h>
#include <stdint.h>

extern volatile bool shaft_checks_passed;
extern volatile uint32_t shaft_checked_cases;
extern volatile uint32_t shaft_failed_check;
extern volatile float shaft_demo_rpm_1750;
extern volatile float shaft_demo_x1_phase_deg;
extern volatile float shaft_demo_x1_um_pp;

/** Run every case once; results in the variables above. */
void run_shaft_demo(void);
#endif
