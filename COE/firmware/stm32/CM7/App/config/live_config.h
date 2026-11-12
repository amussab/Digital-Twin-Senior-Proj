/**
 * @file live_config.h
 * LIVE mode settings: machine speed, bearing profile and calibration.
 */
#ifndef COE_LIVE_CONFIG_H
#define COE_LIVE_CONFIG_H

/* Settled shaft speed of the machine, any value 60..3600 RPM (pipeline_config.h).
 * Change it here, rebuild and flash. 11 Nov 2026: the lab machine runs at 250 RPM.
 */
#define COE_LIVE_NOMINAL_RPM 250U

/* 1: start acquiring on power-up (first try after the delay, so the ADC has
 * power) and retry while stopped, so the node runs with no one at the board.
 * B1 still stops it, and auto-start then waits until B1 is pressed again.
 * 0: wait for B1 as before.
 */
#define COE_LIVE_AUTOSTART 1
#define COE_LIVE_AUTOSTART_DELAY_MS 2000U
#define COE_LIVE_AUTOSTART_RETRY_MS 5000U

/* Bearing orders for the frequency features. coe_project_bearing is the
 * provisional project profile (confirm before AI training).
 */
#define COE_LIVE_BEARING_PROFILE coe_project_bearing

/* Volts at the ADC input per g, for B1X, B1Y, B2X, B2Y: sensor sensitivity
 * x conditioner gain x filter gain. 0 = unknown: do not guess a value.
 * While any value is 0, the DSP runs on volts (for timing and stability
 * checks) and NO packets are sent, because every packet declares g.
 * Set 2 Nov 2026: TE 805-0050 sensors, nominal 100 mV/g (+-10 %); conditioner
 * gain 1 (assumed) and filter gain 1.000 (measured at 100 Hz). B2X carries a
 * copy of the B1Y sensor (the 4th sensor has no cable).
 */
#define COE_LIVE_VOLTS_PER_G {0.100f, 0.100f, 0.100f, 0.100f}

#endif
