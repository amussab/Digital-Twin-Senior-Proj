/**
 * @file shaft_sensors.h
 * Machine speed from the RK4 Keyphasor and per-window values of the two RK4
 * proximity probes. Pure computation: no HAL, no heap, one sample at a time.
 *
 * Inputs are volts at the ADC (after the 47 k / 10 k dividers):
 * - Keyphasor: about -1.43 V with a -2.5 V notch once per turn. The speed is
 *   measured between notch leading edges: the falling crossing of the middle
 *   level, found with hysteresis and timed to a fraction of a sample.
 *   The levels are learned from the signal (extremes of the last 1-2 s), so no
 *   level is hard-coded.
 * - Probes: negative Proximitor voltage, 200 mV/mil (more negative = larger gap).
 *
 * Per window (the same samples as one feature window) for each probe:
 *   gap_um       mean gap (from the mean probe voltage)
 *   pkpk_um      peak-to-peak displacement (of 8-sample means, 3.75 kHz)
 *   x1_um_pp     once-per-turn (1X) displacement, peak-to-peak
 *   x1_phase_deg 1X phase lag: degrees of rotation from the Keyphasor event to
 *                the closest approach of the shaft to the probe, 0..360
 * 1X uses whole revolutions between Keyphasor events (vector average of the
 * revolutions that end in the window), with each revolution's own mean removed.
 */
#ifndef COE_SHAFT_SENSORS_H
#define COE_SHAFT_SENSORS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SHAFT_PROBES 2U              /* 0 = PROX 1, 1 = PROX 3 */
#define SHAFT_PERIOD_HISTORY 4U      /* revolutions in the speed estimate */
#define SHAFT_MIN_TRACKED_RPM 50.0   /* slower Keyphasor pulses restart the measurement */
#define SHAFT_MAX_TRACKED_RPM 4500.0 /* faster pulses are rejected as glitches */
#define SHAFT_GLITCH_FRACTION 0.7    /* a pulse sooner than 0.7 of the last turn is noise */
#define SHAFT_LOCK_TOLERANCE 0.01   /* the last 4 revolutions within +-1 % of their mean */
#define SHAFT_REV_TOLERANCE 0.01     /* 1X: revolution within +-1 % of its predicted length */
#define SHAFT_MIN_SPAN_VOLTS 0.25f   /* Keyphasor notch depth at the ADC needed to detect it */
#define SHAFT_HYSTERESIS 0.20f       /* of the notch depth, around the middle level */
#define SHAFT_GROUP_SAMPLES 8U       /* pk-pk uses means of 8 samples */
#define SHAFT_PROBE_MIN_VOLTS (-19.0f) /* probe voltage range accepted as a valid gap */
#define SHAFT_PROBE_MAX_VOLTS (-1.0f)
#define SHAFT_CLIP_VOLTS 4.99f       /* ADC input at the +-5 V rail */

/** Calibration, set once. */
typedef struct
{
    double sample_rate_hz;               /* real ADC sample rate */
    float probe_divider[SHAFT_PROBES];   /* ADC volts / probe volts (0.170) */
    float um_per_probe_volt;             /* 127 = 25.4 um/mil / 0.200 V/mil */
} ShaftSensorsConfig;

/** One probe's window values; all zero when not valid. */
typedef struct
{
    bool valid;          /* mean voltage in the linear range and no clipping */
    bool x1_valid;       /* valid, and at least one whole steady revolution */
    float gap_um;
    float pkpk_um;
    float x1_um_pp;
    float x1_phase_deg;
    float mean_probe_volts; /* for the console; any value */
} ShaftProbeWindow;

/** Values for the samples since the last shaft_sensors_start_window(). */
typedef struct
{
    bool speed_measured;   /* at least one whole revolution ended in the window */
    float rpm;             /* mean over those revolutions; 0 when not measured */
    float rpm_min, rpm_max; /* slowest / fastest of those revolutions */
    uint32_t revolutions;
    ShaftProbeWindow probes[SHAFT_PROBES];
} ShaftWindow;

typedef struct
{
    ShaftSensorsConfig config;
    uint64_t sample_index;   /* index of the next sample */
    /* Keyphasor levels: extremes per 1-s segment, thresholds from the last two. */
    uint32_t segment_samples, segment_count;
    float segment_high, segment_low, previous_high, previous_low;
    bool have_previous_segment, levels_ok, armed, have_last_sample;
    float middle, hysteresis, last_keyphasor;
    float level_high, level_low;
    /* Revolution timing (sample units, fractional). */
    bool have_event;
    double last_event;
    double periods[SHAFT_PERIOD_HISTORY];
    uint32_t period_count, period_next;
    uint32_t glitches;       /* rejected pulses (too close together) */
    uint32_t total_revolutions;
    /* Current revolution's 1X sums, with a phasor rotating at the predicted speed. */
    bool rev_active;
    double predicted_period, rot_c, rot_s, step_c, step_s;
    double rev_sum[SHAFT_PROBES], rev_sum_c[SHAFT_PROBES], rev_sum_s[SHAFT_PROBES];
    double rev_c, rev_s;
    uint32_t rev_n;
    /* Window sums. */
    double win_sum[SHAFT_PROBES];
    uint32_t win_n, win_clipped[SHAFT_PROBES];
    float group_sum[SHAFT_PROBES], group_min[SHAFT_PROBES], group_max[SHAFT_PROBES];
    uint32_t group_n;
    bool have_group;
    uint32_t win_revs;
    double win_period_sum, win_period_min, win_period_max;
    double win_x1_c[SHAFT_PROBES], win_x1_s[SHAFT_PROBES];
    uint32_t win_x1_n;
} ShaftSensors;

/** Start from nothing (levels, timing and window). False on a bad config. */
bool shaft_sensors_init(ShaftSensors *shaft, const ShaftSensorsConfig *config);

/** Forget the revolution timing (after lost ADC samples) but keep the learned
 * Keyphasor levels. Also starts a new window. */
void shaft_sensors_restart_timing(ShaftSensors *shaft);

/** Begin a new window (clears only the window sums). */
void shaft_sensors_start_window(ShaftSensors *shaft);

/** Consume count chronological samples (ADC volts) of the three inputs. */
void shaft_sensors_process(ShaftSensors *shaft, const float *keyphasor,
                           const float *const probes[SHAFT_PROBES], size_t count);

/** Current speed from the last 4 revolutions. True only when they agree within
 * +-1 % and the last pulse is less than two revolutions old. */
bool shaft_sensors_speed(const ShaftSensors *shaft, double *rpm);

/** Window values for the samples since the last start; then a new window starts. */
void shaft_sensors_finish_window(ShaftSensors *shaft, ShaftWindow *window);

#endif
