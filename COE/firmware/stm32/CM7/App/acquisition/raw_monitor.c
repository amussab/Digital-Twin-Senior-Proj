/**
 * @file raw_monitor.c
 * Raw AD7606 statistics for wiring, channel-order and rate checks.
 * Voltages assume RANGE low (+/-5 V). They are volts at the ADC input only:
 * no sensor sensitivity is applied and nothing here is acceleration.
 */
#include "raw_monitor.h"
#include "acquisition_console.h"
#include "main.h"
#include <string.h>

volatile float raw_monitor_mean_counts[AD7606_SLOTS];
volatile float raw_monitor_rms_counts[AD7606_SLOTS];
volatile int32_t raw_monitor_pp_counts[AD7606_SLOTS];
volatile uint32_t raw_monitor_clipped[AD7606_SLOTS];
volatile float raw_monitor_rate_hz;
volatile float raw_monitor_average_rate_hz;
volatile uint32_t raw_monitor_start_result;

#define MV_PER_COUNT (1000.0 * AD7606_VOLTS_PER_COUNT)
#define REPORT_MS 1000U

static const char *const slot_names[AD7606_SLOTS] = {
    "V1 AIN1    ", "V2 AIN2 B1X", "V3 AIN3    ", "V4 AIN4 B1Y",
    "V5 AIN5    ", "V6 AIN6 B2X", "V7 AIN7    ", "V8 AIN8 B2Y",
};

/* Per-slot sums over the current report interval. */
typedef struct
{
    int64_t sum;
    uint64_t sum_squares;
    int32_t min, max;
} slot_sums_t;

static slot_sums_t sums[AD7606_SLOTS];
static uint32_t interval_frames;
static ad7606_status_t previous;
static uint64_t total_cycles; /* Between the first and latest BUSY edges. */
static uint32_t first_edge_count;
static bool have_first_edge;
static uint32_t started_tick, report_tick;

static uint32_t isqrt64(uint64_t value)
{
    uint64_t root = 0U, bit = 1ULL << 62;
    while (bit > value)
        bit >>= 2;
    while (bit != 0U)
    {
        if (value >= root + bit)
        {
            value -= root + bit;
            root = (root >> 1) + bit;
        }
        else
            root >>= 1;
        bit >>= 2;
    }
    return (uint32_t)root;
}

static void reset_sums(void)
{
    for (uint32_t slot = 0U; slot < AD7606_SLOTS; ++slot)
    {
        sums[slot].sum = 0;
        sums[slot].sum_squares = 0U;
        sums[slot].min = INT32_MAX;
        sums[slot].max = INT32_MIN;
    }
    interval_frames = 0U;
}

static void accumulate(const ad7606_frame_t *block)
{
    for (uint32_t frame = 0U; frame < AD7606_BLOCK_FRAMES; ++frame)
        for (uint32_t slot = 0U; slot < AD7606_SLOTS; ++slot)
        {
            const int32_t value = block[frame][slot];
            slot_sums_t *s = &sums[slot];
            s->sum += value;
            s->sum_squares += (uint64_t)((int64_t)value * value);
            if (value < s->min)
                s->min = value;
            if (value > s->max)
                s->max = value;
            if (value == INT16_MAX || value == INT16_MIN)
                ++raw_monitor_clipped[slot];
        }
    interval_frames += AD7606_BLOCK_FRAMES;
}

static void report(void)
{
    ad7606_status_t now;
    ad7606_snapshot(&now);
    char a[24], b[24], c[24];

    /* Rate from BUSY edge timestamps: exact frame count over exact time. */
    const uint32_t edges = now.busy_edges - previous.busy_edges;
    const uint32_t cycles = now.last_busy_cycles - previous.last_busy_cycles;
    raw_monitor_rate_hz = 0.0f;
    if (edges != 0U && cycles != 0U && previous.busy_edges != 0U)
    {
        raw_monitor_rate_hz = (float)((double)edges * SystemCoreClock / cycles);
        total_cycles += cycles;
        raw_monitor_average_rate_hz = (float)((double)(now.busy_edges - first_edge_count)
                                              * SystemCoreClock / (double)total_cycles);
    }
    if (!have_first_edge && now.busy_edges != 0U)
    {
        have_first_edge = true;
        first_edge_count = now.busy_edges;
        total_cycles = 0U;
    }
    previous = now;

    const double ns_per_tick = 1e9 / ad7606_timing.timer_clock_hz;
    const double us_per_cycle = 1e6 / SystemCoreClock;
    console_emit("\r\n[%lus] rate %s Hz (avg %s) frames %lu | missed %lu overlap %lu overrun %lu"
                 " dma %lu spi %lu stall %lu\r\n",
                 (unsigned long)((HAL_GetTick() - started_tick) / 1000U),
                 console_fixed2(a, sizeof(a), raw_monitor_rate_hz),
                 console_fixed2(b, sizeof(b), raw_monitor_average_rate_hz),
                 (unsigned long)now.frames, (unsigned long)now.missed_conversions,
                 (unsigned long)now.read_overlaps, (unsigned long)now.ring_overruns,
                 (unsigned long)now.dma_errors, (unsigned long)now.spi_overruns,
                 (unsigned long)ad7606_stalls);
    console_emit("  fill max %lu/%u | CONVST->BUSY ISR %lu-%lu ns | ISR max busy %s us, dma %s us\r\n",
                 (unsigned long)now.max_fill_frames, (unsigned)AD7606_RING_FRAMES,
                 (unsigned long)(now.busy_latency_min_ticks == UINT32_MAX
                                     ? 0U : (uint32_t)(now.busy_latency_min_ticks * ns_per_tick)),
                 (unsigned long)(now.busy_latency_max_ticks * ns_per_tick),
                 console_fixed2(a, sizeof(a), now.busy_isr_max_cycles * us_per_cycle),
                 console_fixed2(b, sizeof(b), now.dma_isr_max_cycles * us_per_cycle));
    console_emit("  slot            mean mV    rms mV     pp mV   min..max counts  clip\r\n");
    for (uint32_t slot = 0U; slot < AD7606_SLOTS; ++slot)
    {
        const slot_sums_t *s = &sums[slot];
        if (interval_frames == 0U)
        {
            console_emit("  %s   (no frames)\r\n", slot_names[slot]);
            continue;
        }
        const double mean = (double)s->sum / interval_frames;
        /* Integer variance avoids losing small AC signals under a DC offset. */
        const int64_t n = (int64_t)interval_frames;
        const uint64_t scaled = (uint64_t)(n * (int64_t)s->sum_squares - s->sum * s->sum);
        const double rms = (double)isqrt64(scaled) / interval_frames;
        raw_monitor_mean_counts[slot] = (float)mean;
        raw_monitor_rms_counts[slot] = (float)rms;
        raw_monitor_pp_counts[slot] = s->max - s->min;
        console_emit("  %s %9s %9s %9s %7ld..%-7ld %4lu\r\n", slot_names[slot],
                     console_fixed2(a, sizeof(a), mean * MV_PER_COUNT),
                     console_fixed2(b, sizeof(b), rms * MV_PER_COUNT),
                     console_fixed2(c, sizeof(c), (s->max - s->min) * MV_PER_COUNT),
                     (long)s->min, (long)s->max, (unsigned long)raw_monitor_clipped[slot]);
    }
    reset_sums();
}

static void start(void)
{
    const ad7606_result_t result = console_start_acquisition();
    raw_monitor_start_result = (uint32_t)result;
    if (result != AD7606_OK)
        return;
    memset(&previous, 0, sizeof(previous));
    memset((void *)raw_monitor_clipped, 0, sizeof(raw_monitor_clipped));
    have_first_edge = false;
    total_cycles = 0U;
    reset_sums();
    started_tick = report_tick = HAL_GetTick();
}

void raw_monitor_init(void)
{
    reset_sums();
    console_banner("LIVE_RAW: AD7606 raw acquisition");
    console_emit("Voltages assume RANGE = +/-5 V; they are ADC input volts, not g.\r\n");
}

void raw_monitor_poll(void)
{
    if (console_button_pressed())
    {
        if (ad7606_running())
        {
            ad7606_stop();
            console_emit("\r\nAD7606 stopped by B1. Pins held low; ADC power may be switched off.\r\n");
        }
        else
            start();
    }
    if (ad7606_check_stall())
        console_emit("\r\nAD7606 STALL: no frame for %u ms; stopped and pins held low."
                     " Check ADC power/BUSY, then press B1.\r\n", (unsigned)AD7606_STALL_MS);

    const ad7606_frame_t *block;
    while ((block = ad7606_peek_block()) != NULL)
    {
        accumulate(block);
        ad7606_release_block();
    }

    if (ad7606_running() && (uint32_t)(HAL_GetTick() - report_tick) >= REPORT_MS)
    {
        report_tick += REPORT_MS;
        if (console_idle()) /* Skip a report rather than block. */
            report();
    }
    console_flush_some();

    ad7606_status_t status;
    ad7606_snapshot(&status);
    console_update_leds(ad7606_running(), ad7606_gap_events(&status) + ad7606_stalls != 0U);
}
