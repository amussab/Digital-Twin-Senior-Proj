/** @file shaft_demo.c
 * Check shaft_sensors.c against signals with a known answer: the Keyphasor notch
 * and the two probe voltages are generated from a shaft angle, so the true speed,
 * gap, pk-pk, 1X amplitude and 1X phase are known exactly. The signals go through
 * the same ADC scaling (47 k / 10 k divider, 16-bit counts, noise) as on the rig.
 * Runs on the board (REPLAY) and on a PC (TestData/shaft/run_shaft_checks.py).
 */
#include "shaft_demo.h"
#include "shaft_sensors.h"
#include <math.h>
#include <string.h>

volatile bool shaft_checks_passed;
volatile uint32_t shaft_checked_cases;
volatile uint32_t shaft_failed_check;
volatile float shaft_demo_rpm_1750;        /* measured, window mean */
volatile float shaft_demo_x1_phase_deg;    /* PROX 1 at 1750 RPM, true 70 */
volatile float shaft_demo_x1_um_pp;        /* PROX 1 at 1750 RPM, true 50 */

#define PI_D 3.141592653589793
#define RATE_HZ 29998.5               /* the node's real ADC rate */
#define DIVIDER 0.170f
#define UM_PER_VOLT 127.0f            /* 25.4 um/mil / 0.200 V/mil */
#define VOLTS_PER_COUNT (10.0f / 65536.0f)
#define KPH_BASE (-8.4f * DIVIDER)    /* measured RK4 levels */
#define KPH_NOTCH (-14.7f * DIVIDER)
#define NOTCH_WIDTH 0.05              /* of a turn */
#define EDGE_WIDTH 0.004              /* of a turn, linear edges */
#define BLOCK 256U

typedef struct
{
    double rpm, rpm_after;            /* speed; step to rpm_after at change_at (0 = none) */
    uint64_t change_at;
    uint64_t keyphasor_until;         /* 0 = always present */
    bool keyphasor;                   /* false: flat signal (no notch) */
    uint32_t glitch_every;            /* one short spike every N turns (0 = none) */
    double probe_volts[SHAFT_PROBES]; /* mean Proximitor voltage */
    double a1_um[SHAFT_PROBES], phase1_deg[SHAFT_PROBES]; /* 1X zero-to-peak */
    double a2_um[SHAFT_PROBES], phase2_deg[SHAFT_PROBES]; /* 2X zero-to-peak */
    bool open[SHAFT_PROBES];          /* input open: the ADC floats at +1.75 V */
    double start_angle;
    double ripple;                    /* speed ripple, fraction (motor drive), at 0.7 Hz */
} Scenario;

typedef struct
{
    const Scenario *s;
    uint64_t n;
    double angle; /* radians, keyphasor event at multiples of 2 pi */
    uint32_t noise;
} Generator;

static void record_check(bool passed)
{
    ++shaft_checked_cases;
    if (!passed && shaft_failed_check == 0U)
        shaft_failed_check = shaft_checked_cases;
}

static float noise(Generator *g) /* uniform +-0.3 mV */
{
    g->noise = g->noise * 1664525U + 1013904223U;
    return ((float)(g->noise >> 8) / 16777216.0f - 0.5f) * 0.0006f;
}

static float adc(float volts) /* 16-bit counts, +-5 V */
{
    float counts = roundf(volts / VOLTS_PER_COUNT);
    if (counts > 32767.0f)
        counts = 32767.0f;
    if (counts < -32768.0f)
        counts = -32768.0f;
    return counts * VOLTS_PER_COUNT;
}

static float keyphasor_volts(const Generator *g)
{
    double turn = g->angle / (2.0 * PI_D);
    double u = turn - floor(turn); /* 0..1 */
    if (u > 0.5)
        u -= 1.0;                  /* -0.5..0.5, the falling edge's middle at 0 */
    const double half = EDGE_WIDTH / 2.0;
    const double depth = (double)KPH_NOTCH - (double)KPH_BASE;
    double v;
    if (u >= -half && u <= half)
        v = KPH_BASE + depth * (u + half) / EDGE_WIDTH;
    else if (u > half && u < NOTCH_WIDTH - half)
        v = KPH_NOTCH;
    else if (u >= NOTCH_WIDTH - half && u <= NOTCH_WIDTH + half)
        v = KPH_NOTCH - depth * (u - (NOTCH_WIDTH - half)) / EDGE_WIDTH;
    else
        v = KPH_BASE;
    if (g->s->glitch_every != 0U)
    {
        const uint64_t turn_index = (uint64_t)floor(turn);
        if (turn_index % g->s->glitch_every == 3U && u >= 0.40 && u < 0.40 + 0.003)
            v = KPH_NOTCH; /* a short spike half way round */
    }
    return (float)v;
}

static double probe_um(const Scenario *s, uint32_t p, double angle)
{
    const double d = PI_D / 180.0;
    return s->a1_um[p] * cos(angle - s->phase1_deg[p] * d)
           + s->a2_um[p] * cos(2.0 * angle - s->phase2_deg[p] * d);
}

/** Next count samples of the three ADC inputs. */
static void generate(Generator *g, float *kph, float *p0, float *p1, size_t count)
{
    float *const probes[SHAFT_PROBES] = {p0, p1};
    for (size_t k = 0U; k < count; ++k)
    {
        const Scenario *s = g->s;
        const bool present = s->keyphasor && (s->keyphasor_until == 0U || g->n < s->keyphasor_until);
        kph[k] = adc((present ? keyphasor_volts(g) : KPH_BASE) + noise(g));
        for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
        {
            const double volts = s->probe_volts[p] + probe_um(s, p, g->angle) / UM_PER_VOLT;
            probes[p][k] = s->open[p] ? adc(1.75f + noise(g)) : adc((float)volts * DIVIDER + noise(g));
        }
        double rpm = (s->change_at != 0U && g->n >= s->change_at) ? s->rpm_after : s->rpm;
        rpm *= 1.0 + s->ripple * sin(2.0 * PI_D * 0.7 * (double)g->n / RATE_HZ);
        g->angle += 2.0 * PI_D * rpm / 60.0 / RATE_HZ;
        ++g->n;
    }
}

static void run(ShaftSensors *shaft, Generator *g, uint64_t samples)
{
    static float kph[BLOCK], p0[BLOCK], p1[BLOCK];
    const float *const probes[SHAFT_PROBES] = {p0, p1};
    while (samples != 0U)
    {
        const size_t take = samples < BLOCK ? (size_t)samples : BLOCK;
        generate(g, kph, p0, p1, take);
        shaft_sensors_process(shaft, kph, probes, take);
        samples -= take;
    }
}

static bool near(double value, double truth, double tolerance)
{
    return fabs(value - truth) <= tolerance;
}

static double angle_error(double a, double b)
{
    double d = fmod(a - b, 360.0);
    if (d > 180.0)
        d -= 360.0;
    if (d < -180.0)
        d += 360.0;
    return fabs(d);
}

/** True pk-pk of the probe's 1X + 2X motion (dense evaluation over one turn). */
static double true_pkpk(const Scenario *s, uint32_t p)
{
    double lo = 1e9, hi = -1e9;
    for (uint32_t i = 0U; i < 7200U; ++i)
    {
        const double v = probe_um(s, p, 2.0 * PI_D * i / 7200.0);
        lo = fmin(lo, v);
        hi = fmax(hi, v);
    }
    return hi - lo;
}

static void init_shaft(ShaftSensors *shaft)
{
    const ShaftSensorsConfig config = {RATE_HZ, {DIVIDER, DIVIDER}, UM_PER_VOLT};
    record_check(shaft_sensors_init(shaft, &config));
}

/** Steady speed: lock, then three 20-turn windows checked against the truth. */
static void check_steady(const Scenario *s, bool keep_1750_result)
{
    static ShaftSensors shaft;
    Generator g = {s, 0U, s->start_angle, 12345U};
    init_shaft(&shaft);
    const double period = RATE_HZ * 60.0 / s->rpm;
    run(&shaft, &g, (uint64_t)(2.0 * RATE_HZ + 6.0 * period)); /* levels (1-2 s) + 4 turns */
    double rpm = 0.0;
    const bool locked = shaft_sensors_speed(&shaft, &rpm);
    const double speed_tolerance = (s->ripple > 0.0 ? 1.2 * s->ripple : 1e-4) * s->rpm;
    record_check(s->keyphasor ? locked && near(rpm, s->rpm, speed_tolerance) : !locked);

    shaft_sensors_start_window(&shaft);
    uint32_t turns = 20U;
    while (turns > 1U && period * turns > 21600.0) /* same window rule as the pipeline */
        --turns;
    const uint64_t window = (uint64_t)(period * turns + 0.5);
    for (uint32_t w = 0U; w < 3U; ++w)
    {
        ShaftWindow out;
        run(&shaft, &g, window);
        shaft_sensors_finish_window(&shaft, &out);
        if (s->keyphasor)
            record_check(out.speed_measured && near(out.rpm, s->rpm, speed_tolerance)
                         && out.rpm_min <= out.rpm && out.rpm <= out.rpm_max
                         && near(out.rpm_max - out.rpm_min, 0.0, 1e-3 * s->rpm + 2.5 * speed_tolerance)
                         && (out.revolutions == turns || out.revolutions + 1U == turns
                             || out.revolutions == turns + 1U));
        else
            record_check(!out.speed_measured && out.rpm == 0.0f && out.revolutions == 0U);
        for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
        {
            const ShaftProbeWindow *pr = &out.probes[p];
            if (s->open[p])
            {
                record_check(!pr->valid && !pr->x1_valid && pr->gap_um == 0.0f
                             && pr->pkpk_um == 0.0f && pr->x1_um_pp == 0.0f);
                continue;
            }
            const double pkpk = true_pkpk(s, p);
            record_check(pr->valid && near(pr->gap_um, -s->probe_volts[p] * UM_PER_VOLT, 0.5)
                         && near(pr->pkpk_um, pkpk, 0.01 * pkpk + 0.5));
            if (s->keyphasor)
                record_check(pr->x1_valid && near(pr->x1_um_pp, 2.0 * s->a1_um[p], 0.005 * 2.0 * s->a1_um[p] + 0.2)
                             && angle_error(pr->x1_phase_deg, s->phase1_deg[p]) <= 0.2);
            else
                record_check(!pr->x1_valid && pr->x1_um_pp == 0.0f && pr->x1_phase_deg == 0.0f);
            if (keep_1750_result && p == 0U && w == 0U)
            {
                shaft_demo_rpm_1750 = out.rpm;
                shaft_demo_x1_phase_deg = pr->x1_phase_deg;
                shaft_demo_x1_um_pp = pr->x1_um_pp;
            }
        }
    }
    if (s->glitch_every != 0U)
        record_check(shaft.glitches != 0U);
}

/** Speed step 1750 -> 1800: the lock drops, then follows the new speed. */
static void check_speed_change(const Scenario *base)
{
    static ShaftSensors shaft;
    Scenario s = *base;
    const double period = RATE_HZ * 60.0 / s.rpm;
    s.change_at = (uint64_t)(2.0 * RATE_HZ + 10.0 * period);
    s.rpm_after = 1800.0;
    Generator g = {&s, 0U, 0.3, 777U};
    init_shaft(&shaft);
    double rpm = 0.0;
    run(&shaft, &g, s.change_at);
    record_check(shaft_sensors_speed(&shaft, &rpm) && near(rpm, 1750.0, 0.2));
    run(&shaft, &g, (uint64_t)(2.0 * period)); /* inside the change: not locked */
    record_check(!shaft_sensors_speed(&shaft, &rpm));
    run(&shaft, &g, (uint64_t)(6.0 * period));
    record_check(shaft_sensors_speed(&shaft, &rpm) && near(rpm, 1800.0, 0.2));
}

/** The Keyphasor stops: the speed is no longer reported after two turns. */
static void check_lost(const Scenario *base)
{
    static ShaftSensors shaft;
    Scenario s = *base;
    const double period = RATE_HZ * 60.0 / s.rpm;
    s.keyphasor_until = (uint64_t)(2.0 * RATE_HZ + 8.0 * period);
    Generator g = {&s, 0U, 1.0, 99U};
    init_shaft(&shaft);
    double rpm = 0.0;
    run(&shaft, &g, s.keyphasor_until);
    record_check(shaft_sensors_speed(&shaft, &rpm));
    run(&shaft, &g, (uint64_t)(2.5 * period));
    record_check(!shaft_sensors_speed(&shaft, &rpm));
    shaft_sensors_restart_timing(&shaft); /* also after lost ADC samples */
    record_check(!shaft_sensors_speed(&shaft, &rpm) && shaft.levels_ok);
}

static void check_bad_config(void)
{
    static ShaftSensors shaft;
    const ShaftSensorsConfig zero_divider = {RATE_HZ, {0.0f, DIVIDER}, UM_PER_VOLT};
    const ShaftSensorsConfig zero_rate = {0.0, {DIVIDER, DIVIDER}, UM_PER_VOLT};
    record_check(!shaft_sensors_init(&shaft, &zero_divider) && !shaft_sensors_init(&shaft, &zero_rate)
                 && !shaft_sensors_init(&shaft, NULL));
}

void run_shaft_demo(void)
{
    shaft_checks_passed = false;
    shaft_checked_cases = shaft_failed_check = 0U;
    /* 1750 RPM: PROX 1 at -9.7 V (1232 um gap), 25 um 1X at 70 deg + 8 um 2X;
     * PROX 3 at -8.2 V, 12 um 1X at 200 deg + 3 um 2X. */
    const Scenario s1750 = {1750.0, 0.0, 0U, 0U, true, 0U, {-9.7, -8.2}, {25.0, 12.0}, {70.0, 200.0},
                            {8.0, 3.0}, {10.0, 300.0}, {false, false}, 0.0, 0.0};
    check_steady(&s1750, true);
    /* 3600 RPM, phases near the 0/360 wrap, start part way round a turn. */
    const Scenario s3600 = {3600.0, 0.0, 0U, 0U, true, 0U, {-10.5, -7.0}, {40.0, 6.0}, {355.0, 2.0},
                            {5.0, 2.0}, {90.0, 45.0}, {false, false}, 2.5, 0.0};
    check_steady(&s3600, false);
    /* 500 RPM (6-turn windows), PROX 3 input open. */
    const Scenario s500 = {500.0, 0.0, 0U, 0U, true, 0U, {-9.0, 0.0}, {15.0, 0.0}, {120.0, 0.0},
                           {4.0, 0.0}, {0.0, 0.0}, {false, true}, 4.0, 0.0};
    check_steady(&s500, false);
    /* No Keyphasor signal: no speed, no 1X; gap and pk-pk still valid. */
    Scenario none = s1750;
    none.keyphasor = false;
    check_steady(&none, false);
    /* A short spike half way round every 7th turn is rejected. */
    Scenario glitch = s1750;
    glitch.glitch_every = 7U;
    check_steady(&glitch, false);
    /* +-0.2 % speed ripple, as from a motor drive: 1X still exact (mean removed per turn). */
    Scenario ripple = s1750;
    ripple.ripple = 0.002;
    check_steady(&ripple, false);
    check_speed_change(&s1750);
    check_lost(&s1750);
    check_bad_config();
    shaft_checks_passed = shaft_failed_check == 0U;
}
