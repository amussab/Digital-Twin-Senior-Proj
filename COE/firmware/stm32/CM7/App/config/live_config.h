/**
 * @file live_config.h
 * LIVE mode settings: bearing profile, calibration of the accelerometers and
 * of the RK4 proximity probes.
 */
#ifndef COE_LIVE_CONFIG_H
#define COE_LIVE_CONFIG_H

/* Machine speed: NOT a setting. Since 16 Nov 2026 LIVE measures it from the RK4
 * Keyphasor on AIN7 (shaft_sensors.c). Windows start once 4 turns agree within
 * 1 %, the window length and bearing bands follow the measured speed, and the
 * window restarts when the speed moves by more than COE_LIVE_SPEED_RESTART.
 * No Keyphasor pulses = no feature windows (nothing is guessed).
 */
#define COE_LIVE_SPEED_RESTART 0.005f /* 0.5 %: 8.75 RPM at 1750 */
/* A window is sent only if its measured mean speed is within this of the speed
 * its bands were computed for, and every turn in it within twice this. */
#define COE_LIVE_SPEED_TOLERANCE 0.01f

/* RK4 proximity probes (PROX 1 on AIN5, PROX 3 on AIN6) and Keyphasor (AIN7):
 * each through a 47 k / 10 k divider, ratio measured 0.170 (15 Nov 2026).
 * Proximitor sensitivity 200 mV/mil (RK4 datasheet) = 127 um per volt. */
#define COE_LIVE_PROBE_DIVIDER {0.170f, 0.170f}
#define COE_LIVE_PROBE_UM_PER_VOLT 127.0f

/* 1: start acquiring on power-up (first try after the delay, so the ADC has
 * power) and retry while stopped, so the node runs with no one at the board.
 * B1 still stops it, and auto-start then waits until B1 is pressed again.
 * 0: wait for B1 as before.
 */
#define COE_LIVE_AUTOSTART 1
#define COE_LIVE_AUTOSTART_DELAY_MS 2000U
#define COE_LIVE_AUTOSTART_RETRY_MS 5000U

/* Bearing orders for the frequency features: the RK4 rotor kit's S6700 bearings
 * (profile 3, since 16 Nov 2026; see frequency_features.c). Profile 1 was the old
 * provisional bearing: its data must not be mixed with profile 3 data.
 */
#define COE_LIVE_BEARING_PROFILE coe_rk4_s6700_bearing

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
