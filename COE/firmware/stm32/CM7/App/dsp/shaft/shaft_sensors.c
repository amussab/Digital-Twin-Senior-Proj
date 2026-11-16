/**
 * @file shaft_sensors.c
 * Keyphasor speed measurement and proximity-probe window values (see the header).
 */
#include "shaft_sensors.h"

#include <float.h>
#include <math.h>
#include <string.h>

#define TWO_PI 6.283185307179586

static void clear_revolution(ShaftSensors *shaft)
{
    shaft->rev_active = false;
    shaft->rev_n = 0U;
    shaft->rev_c = shaft->rev_s = 0.0;
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
        shaft->rev_sum[p] = shaft->rev_sum_c[p] = shaft->rev_sum_s[p] = 0.0;
}

void shaft_sensors_start_window(ShaftSensors *shaft)
{
    if (shaft == NULL)
        return;
    shaft->win_n = 0U;
    shaft->group_n = 0U;
    shaft->have_group = false;
    shaft->win_revs = 0U;
    shaft->win_period_sum = 0.0;
    shaft->win_period_min = DBL_MAX;
    shaft->win_period_max = 0.0;
    shaft->win_x1_n = 0U;
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
    {
        shaft->win_sum[p] = 0.0;
        shaft->win_clipped[p] = 0U;
        shaft->group_sum[p] = 0.0f;
        shaft->group_min[p] = shaft->group_max[p] = 0.0f;
        shaft->win_x1_c[p] = shaft->win_x1_s[p] = 0.0;
    }
}

void shaft_sensors_restart_timing(ShaftSensors *shaft)
{
    if (shaft == NULL)
        return;
    shaft->have_event = false;
    shaft->period_count = shaft->period_next = 0U;
    shaft->armed = false;
    shaft->have_last_sample = false;
    clear_revolution(shaft);
    shaft_sensors_start_window(shaft);
}

bool shaft_sensors_init(ShaftSensors *shaft, const ShaftSensorsConfig *config)
{
    if (shaft == NULL)
        return false;
    memset(shaft, 0, sizeof(*shaft));
    if (config == NULL || !(config->sample_rate_hz > 1000.0) || !(config->um_per_probe_volt > 0.0f))
        return false;
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
        if (!(config->probe_divider[p] > 0.0f))
            return false;
    shaft->config = *config;
    shaft->segment_samples = (uint32_t)(config->sample_rate_hz + 0.5); /* 1 s */
    shaft->segment_high = -FLT_MAX;
    shaft->segment_low = FLT_MAX;
    shaft_sensors_restart_timing(shaft);
    return true;
}

/** 1X vector of the revolution that just ended, if it was steady. */
static void finish_revolution(ShaftSensors *shaft, double period)
{
    if (!shaft->rev_active || shaft->rev_n < 2U
        || fabs(period - shaft->predicted_period) > SHAFT_REV_TOLERANCE * shaft->predicted_period)
        return;
    const double n = (double)shaft->rev_n;
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
    {
        /* Remove this revolution's own mean exactly: sum((x - m) cos) = sum(x cos) - m sum(cos). */
        const double mean = shaft->rev_sum[p] / n;
        shaft->win_x1_c[p] += 2.0 * (shaft->rev_sum_c[p] - mean * shaft->rev_c) / n;
        shaft->win_x1_s[p] += 2.0 * (shaft->rev_sum_s[p] - mean * shaft->rev_s) / n;
    }
    ++shaft->win_x1_n;
}

/** A Keyphasor event at fractional sample time `time`; `index` = current sample. */
static void on_event(ShaftSensors *shaft, double time, uint64_t index)
{
    const double rate = shaft->config.sample_rate_hz;
    if (shaft->have_event)
    {
        const double period = time - shaft->last_event;
        const uint32_t last = (shaft->period_next + SHAFT_PERIOD_HISTORY - 1U) % SHAFT_PERIOD_HISTORY;
        /* A shaft cannot speed up 30 % within one turn: such a pulse is noise. */
        if (period < rate * 60.0 / SHAFT_MAX_TRACKED_RPM
            || (shaft->period_count != 0U && period < SHAFT_GLITCH_FRACTION * shaft->periods[last]))
        {
            ++shaft->glitches; /* Too soon: not a revolution. Ignore this pulse. */
            return;
        }
        if (period > rate * 60.0 / SHAFT_MIN_TRACKED_RPM)
        {
            shaft->period_count = shaft->period_next = 0U; /* Stopped or lost: start again. */
            clear_revolution(shaft);
        }
        else
        {
            finish_revolution(shaft, period);
            ++shaft->win_revs;
            ++shaft->total_revolutions;
            shaft->win_period_sum += period;
            if (period < shaft->win_period_min)
                shaft->win_period_min = period;
            if (period > shaft->win_period_max)
                shaft->win_period_max = period;
            shaft->periods[shaft->period_next] = period;
            shaft->period_next = (shaft->period_next + 1U) % SHAFT_PERIOD_HISTORY;
            if (shaft->period_count < SHAFT_PERIOD_HISTORY)
                ++shaft->period_count;
        }
    }
    shaft->last_event = time;
    shaft->have_event = true;

    /* New revolution: predict its length from the last one. */
    clear_revolution(shaft);
    if (shaft->period_count == 0U)
        return;
    const uint32_t last = (shaft->period_next + SHAFT_PERIOD_HISTORY - 1U) % SHAFT_PERIOD_HISTORY;
    shaft->predicted_period = shaft->periods[last];
    const double step = TWO_PI / shaft->predicted_period;
    const double angle = step * ((double)index - time); /* This sample is after the event. */
    shaft->rot_c = cos(angle);
    shaft->rot_s = sin(angle);
    shaft->step_c = cos(step);
    shaft->step_s = sin(step);
    shaft->rev_active = true;
}

/** Learn the Keyphasor levels: extremes of the last two 1-s segments. */
static void track_levels(ShaftSensors *shaft, float value)
{
    if (value > shaft->segment_high)
        shaft->segment_high = value;
    if (value < shaft->segment_low)
        shaft->segment_low = value;
    if (++shaft->segment_count < shaft->segment_samples)
        return;
    float high = shaft->segment_high, low = shaft->segment_low;
    if (shaft->have_previous_segment)
    {
        high = fmaxf(high, shaft->previous_high);
        low = fminf(low, shaft->previous_low);
    }
    shaft->previous_high = shaft->segment_high;
    shaft->previous_low = shaft->segment_low;
    shaft->have_previous_segment = true;
    shaft->segment_high = -FLT_MAX;
    shaft->segment_low = FLT_MAX;
    shaft->segment_count = 0U;
    shaft->level_high = high;
    shaft->level_low = low;
    const float span = high - low;
    shaft->levels_ok = span >= SHAFT_MIN_SPAN_VOLTS;
    shaft->middle = 0.5f * (high + low);
    shaft->hysteresis = SHAFT_HYSTERESIS * span;
    if (!shaft->levels_ok)
        shaft->armed = false;
}

/** Falling crossing of the middle level after the signal was above middle + hysteresis. */
static void detect_edge(ShaftSensors *shaft, float value)
{
    const uint64_t index = shaft->sample_index;
    if (shaft->levels_ok)
    {
        if (!shaft->armed)
        {
            if (value > shaft->middle + shaft->hysteresis)
                shaft->armed = true;
        }
        else if (value < shaft->middle && shaft->have_last_sample)
        {
            const float drop = shaft->last_keyphasor - value;
            double fraction = drop > 0.0f ? (double)(shaft->last_keyphasor - shaft->middle) / drop : 1.0;
            if (fraction < 0.0)
                fraction = 0.0;
            if (fraction > 1.0)
                fraction = 1.0;
            shaft->armed = false;
            on_event(shaft, (double)(index - 1U) + fraction, index);
        }
    }
    shaft->last_keyphasor = value;
    shaft->have_last_sample = true;
}

void shaft_sensors_process(ShaftSensors *shaft, const float *keyphasor,
                           const float *const probes[SHAFT_PROBES], size_t count)
{
    if (shaft == NULL || keyphasor == NULL || probes == NULL)
        return;
    for (size_t k = 0U; k < count; ++k)
    {
        detect_edge(shaft, keyphasor[k]); /* first: this sample may start a revolution */
        track_levels(shaft, keyphasor[k]);

        for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
        {
            const float x = probes[p][k];
            shaft->win_sum[p] += x;
            shaft->group_sum[p] += x;
            if (fabsf(x) >= SHAFT_CLIP_VOLTS)
                ++shaft->win_clipped[p];
            if (shaft->rev_active)
            {
                shaft->rev_sum[p] += x;
                shaft->rev_sum_c[p] += x * shaft->rot_c;
                shaft->rev_sum_s[p] += x * shaft->rot_s;
            }
        }
        ++shaft->win_n;
        if (++shaft->group_n == SHAFT_GROUP_SAMPLES)
        {
            for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
            {
                const float mean = shaft->group_sum[p] / (float)SHAFT_GROUP_SAMPLES;
                if (!shaft->have_group || mean < shaft->group_min[p])
                    shaft->group_min[p] = mean;
                if (!shaft->have_group || mean > shaft->group_max[p])
                    shaft->group_max[p] = mean;
                shaft->group_sum[p] = 0.0f;
            }
            shaft->have_group = true;
            shaft->group_n = 0U;
        }
        if (shaft->rev_active)
        {
            shaft->rev_c += shaft->rot_c;
            shaft->rev_s += shaft->rot_s;
            ++shaft->rev_n;
            const double c = shaft->rot_c * shaft->step_c - shaft->rot_s * shaft->step_s;
            shaft->rot_s = shaft->rot_s * shaft->step_c + shaft->rot_c * shaft->step_s;
            shaft->rot_c = c;
        }
        ++shaft->sample_index;
    }
}

bool shaft_sensors_speed(const ShaftSensors *shaft, double *rpm)
{
    if (shaft == NULL || rpm == NULL || !shaft->have_event
        || shaft->period_count < SHAFT_PERIOD_HISTORY)
        return false;
    double sum = 0.0;
    for (uint32_t i = 0U; i < SHAFT_PERIOD_HISTORY; ++i)
        sum += shaft->periods[i];
    const double mean = sum / SHAFT_PERIOD_HISTORY;
    for (uint32_t i = 0U; i < SHAFT_PERIOD_HISTORY; ++i)
        if (fabs(shaft->periods[i] - mean) > SHAFT_LOCK_TOLERANCE * mean)
            return false;
    if ((double)shaft->sample_index - shaft->last_event > 2.0 * mean)
        return false; /* No pulse for two revolutions: stopped, or the signal is lost. */
    *rpm = 60.0 * shaft->config.sample_rate_hz / mean;
    return true;
}

void shaft_sensors_finish_window(ShaftSensors *shaft, ShaftWindow *window)
{
    if (shaft == NULL || window == NULL)
        return;
    memset(window, 0, sizeof(*window));
    const double rate = shaft->config.sample_rate_hz;
    window->revolutions = shaft->win_revs;
    window->speed_measured = shaft->win_revs > 0U;
    if (window->speed_measured)
    {
        window->rpm = (float)(60.0 * rate * shaft->win_revs / shaft->win_period_sum);
        window->rpm_min = (float)(60.0 * rate / shaft->win_period_max);
        window->rpm_max = (float)(60.0 * rate / shaft->win_period_min);
    }
    /* The last, partial group of samples also counts for pk-pk. */
    if (shaft->group_n != 0U)
        for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
        {
            const float mean = shaft->group_sum[p] / (float)shaft->group_n;
            if (!shaft->have_group || mean < shaft->group_min[p])
                shaft->group_min[p] = mean;
            if (!shaft->have_group || mean > shaft->group_max[p])
                shaft->group_max[p] = mean;
        }
    const bool any = shaft->win_n != 0U;
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
    {
        ShaftProbeWindow *out = &window->probes[p];
        const float divider = shaft->config.probe_divider[p];
        const float scale = shaft->config.um_per_probe_volt / divider; /* um per ADC volt */
        const float volts = any ? (float)(shaft->win_sum[p] / shaft->win_n) / divider : 0.0f;
        out->mean_probe_volts = volts;
        out->valid = any && shaft->win_clipped[p] == 0U
                     && volts >= SHAFT_PROBE_MIN_VOLTS && volts <= SHAFT_PROBE_MAX_VOLTS;
        if (!out->valid)
            continue;
        out->gap_um = -volts * shaft->config.um_per_probe_volt;
        out->pkpk_um = (shaft->group_max[p] - shaft->group_min[p]) * scale;
        if (shaft->win_x1_n != 0U && window->speed_measured)
        {
            const double c = shaft->win_x1_c[p] / shaft->win_x1_n;
            const double s = shaft->win_x1_s[p] / shaft->win_x1_n;
            double phase = atan2(s, c) * (360.0 / TWO_PI);
            if (phase < 0.0)
                phase += 360.0;
            if (phase >= 360.0)
                phase -= 360.0;
            out->x1_valid = true;
            out->x1_um_pp = (float)(2.0 * sqrt(c * c + s * s)) * scale;
            out->x1_phase_deg = (float)phase < 360.0f ? (float)phase : 0.0f; /* 359.99999 rounds up */
        }
    }
    shaft_sensors_start_window(shaft);
}
