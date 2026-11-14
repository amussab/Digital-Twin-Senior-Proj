/**
 * @file live_config.h
 * LIVE mode settings: machine speed, bearing profile and calibration.
 */
#ifndef COE_LIVE_CONFIG_H
#define COE_LIVE_CONFIG_H

/* Machine speed. The node starts at COE_LIVE_START_RPM after every power-up or
 * reset. The laptop then chooses the speed (training page 1750/3600 buttons, or
 * inference_receiver.py --rpm): it sends the speed in its beacon, the ESP32
 * passes it on over the UART return wire (COR1 frame), and LIVE restarts the
 * window at the new speed. Any 60..3600 RPM is accepted (pipeline_config.h).
 * 14 Nov 2026: the lab machine runs at a set speed again (1750 or 3600); a
 * measured speed input from the machine's tachometer comes later.
 */
#define COE_LIVE_START_RPM 1750U

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
