/**
 * @file live_stream.c
 * Feed AD7606 slots 1/3/5/7 into the existing pipeline and send each window.
 */
#include "live_stream.h"
#include "live_config.h"
#include "acquisition_console.h"
#include "ad7606.h"
#include "benchmark.h"
#include "feature_uart.h"
#include "main.h"
#include "pipeline.h"
#include <string.h>

volatile bool live_calibrated;
volatile uint32_t live_windows;
volatile uint32_t live_packets_queued;
volatile uint32_t live_packets_withheld;
volatile uint32_t live_packet_failures;
volatile uint32_t live_discarded_blocks;
volatile uint32_t live_restarts;
volatile uint32_t live_pipeline_errors;
volatile float live_max_block_ms;
volatile float live_dsp_load_percent;
volatile float live_last_rms[COE_CHANNEL_COUNT];
volatile float live_last_window_latency_ms;
volatile float live_max_window_latency_ms;

#define REPORT_MS 1000U
#define LATENCY_LIMIT_MS 333.0f /* S2: features within 333 ms of the window's end. */
#define MAX_BLOCKS_PER_POLL 4U /* Keep UART and console servicing between blocks. */

/* Pipeline order B1X, B1Y, B2X, B2Y from the confirmed AIN2/4/6/8 wiring. */
static const uint32_t slots[COE_CHANNEL_COUNT] = {
    AD7606_SLOT_B1X, AD7606_SLOT_B1Y, AD7606_SLOT_B2X, AD7606_SLOT_B2Y,
};
static const char *const channel_names[COE_CHANNEL_COUNT] = {"B1X", "B1Y", "B2X", "B2Y"};
static const float volts_per_g[COE_CHANNEL_COUNT] = COE_LIVE_VOLTS_PER_G;
static float scale[COE_CHANNEL_COUNT]; /* Counts -> g, or -> V when uncalibrated. */

static PipelineState pipeline;
static EnvelopeSpectrumWorkspace workspace; /* CPU-only FFT work area in DTCM. */
static PipelineEnvelopeBuffer envelope_buffer
    __attribute__((section(".pipeline_live_buffers"), aligned(32)));
static float input[COE_CHANNEL_COUNT][AD7606_BLOCK_FRAMES];
static bool stream_ok;
static uint32_t handled_gaps;
static uint32_t started_tick, report_tick;
static uint32_t interval_dsp_cycles;
static ad7606_status_t previous;
/* Latency bookkeeping for the block being processed: how long its newest
 * frame had already waited in the ring, and when its DSP call began.
 */
static float block_wait_ms;
static uint32_t block_began_cycles;

/** Print the window's 32 ordered features: amplitudes in milli-units. */
static void print_features(const PipelineWindowResult *result)
{
    static const char *const names[COE_FEATURES_PER_CHANNEL] = {
        "RMS", "kurt", "crest", "envRMS", "FTF", "BSF", "BPFO", "BPFI"};
    console_emit("    32 features (amplitudes in %s; kurtosis and crest factor plain), valid mask 0x%lx:\r\n",
                 live_calibrated ? "mg" : "mV", (unsigned long)result->ai_payload.channel_valid_mask);
    for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        console_emit("    %s:", channel_names[channel]);
        for (uint32_t k = 0U; k < COE_FEATURES_PER_CHANNEL; ++k)
        {
            const float value = result->ai_payload.features[channel * COE_FEATURES_PER_CHANNEL + k];
            const bool dimensionless = k == 1U || k == 2U;
            char text[24];
            console_emit(" %s %s", names[k],
                         console_fixed2(text, sizeof(text), dimensionless ? value : value * 1000.0));
        }
        console_emit("\r\n");
    }
}

/** Window callback: queue the packet only when the input really is in g. */
static bool on_window(const PipelineWindowResult *result, void *context)
{
    (void)context;
    ++live_windows;
    /* Conservative S2 latency: from the newest sample of this block (at or
     * after the window's last sample) to features and packet being ready.
     */
    const float latency = block_wait_ms
                          + benchmark_cycles_to_us(benchmark_cycles_now() - block_began_cycles) / 1000.0f;
    live_last_window_latency_ms = latency;
    if (latency > live_max_window_latency_ms)
        live_max_window_latency_ms = latency;
    for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        live_last_rms[channel] = result->features[channel].rms;
    if (!live_calibrated)
        ++live_packets_withheld;
    else if (feature_uart_send(result->ai_payload.bytes))
        ++live_packets_queued;
    else
        ++live_packet_failures;

    char ms[16], max_ms[16];
    console_emit("  window %lu (%lu samples/channel, %lu RPM): features ready %s ms after its"
                 " last sample (max %s ms, limit 333) %s%s\r\n",
                 (unsigned long)result->window_id, (unsigned long)result->sample_count,
                 (unsigned long)result->nominal_rpm,
                 console_fixed2(ms, sizeof(ms), latency),
                 console_fixed2(max_ms, sizeof(max_ms), live_max_window_latency_ms),
                 latency <= LATENCY_LIMIT_MS ? "OK" : "OVER LIMIT",
                 live_calibrated ? ", packet sent" : ", packet not sent (uncalibrated)");
    print_features(result);
    return true;
}

static void restart_stream(void)
{
    stream_ok = pipeline_init(&pipeline, COE_LIVE_NOMINAL_RPM, &envelope_buffer, &workspace,
                              &COE_LIVE_BEARING_PROFILE, on_window, NULL);
    if (!stream_ok)
        ++live_pipeline_errors;
}

static void reset_counters(void)
{
    live_windows = live_packets_queued = live_packets_withheld = 0U;
    live_packet_failures = live_discarded_blocks = live_restarts = 0U;
    live_pipeline_errors = 0U;
    live_max_block_ms = live_dsp_load_percent = 0.0f;
    live_last_window_latency_ms = live_max_window_latency_ms = 0.0f;
    interval_dsp_cycles = 0U;
    handled_gaps = 0U;
    memset(&previous, 0, sizeof(previous));
}

/** Scale one block's four pipeline slots and run the pipeline on them.
 * frames_after: frames already published after this block's newest frame.
 */
static void process_block(const ad7606_frame_t *block, uint32_t frames_after)
{
    const uint32_t began = benchmark_cycles_now();
    block_began_cycles = began;
    block_wait_ms = (float)frames_after * 1000.0f / (ad7606_timing.sample_rate_millihz / 1000.0f);
    for (uint32_t frame = 0U; frame < AD7606_BLOCK_FRAMES; ++frame)
        for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
            input[channel][frame] = (float)block[frame][slots[channel]] * scale[channel];
    const float *const channels[COE_CHANNEL_COUNT] = {input[0], input[1], input[2], input[3]};

    const bool ok = pipeline_process_block(&pipeline, channels, AD7606_BLOCK_FRAMES);
    const uint32_t cycles = benchmark_cycles_now() - began;
    interval_dsp_cycles += cycles;
    const float ms = benchmark_cycles_to_us(cycles) / 1000.0f;
    if (ms > live_max_block_ms)
        live_max_block_ms = ms;
    if (!ok)
    {
        ++live_pipeline_errors;
        stream_ok = false; /* Restart at the next block; this window is lost. */
    }
}

/** Take whole blocks; drop the ones around a gap and restart the windows. */
static void consume_blocks(void)
{
    const ad7606_frame_t *block;
    for (uint32_t done = 0U; done < MAX_BLOCKS_PER_POLL
                             && (block = ad7606_peek_block()) != NULL; ++done)
    {
        ad7606_status_t status;
        ad7606_snapshot(&status);
        const uint32_t gaps = ad7606_gap_events(&status);
        const uint32_t start = ad7606_block_start();
        if (gaps != handled_gaps)
        {
            if ((int32_t)(status.last_gap_frame - start) > 0)
            {
                ++live_discarded_blocks; /* Contains or precedes the gap. */
                ad7606_release_block();
                continue;
            }
            handled_gaps = gaps;
            ++live_restarts;
            restart_stream();
        }
        else if (!stream_ok)
        {
            ++live_restarts;
            restart_stream();
        }
        if (stream_ok)
            process_block(block, status.frames - (start + AD7606_BLOCK_FRAMES));
        else
            ++live_discarded_blocks;
        ad7606_release_block();
    }
}

static void report(void)
{
    ad7606_status_t now;
    ad7606_snapshot(&now);
    char rate[24], block_ms[24], load[24], uart_ms[24], window_ms[24];
    const uint32_t edges = now.busy_edges - previous.busy_edges;
    const uint32_t cycles = now.last_busy_cycles - previous.last_busy_cycles;
    const double rate_hz = (previous.busy_edges != 0U && cycles != 0U)
                               ? (double)edges * SystemCoreClock / cycles : 0.0;
    live_dsp_load_percent = (float)(100.0 * interval_dsp_cycles
                                    / ((double)SystemCoreClock * REPORT_MS / 1000.0));
    interval_dsp_cycles = 0U;
    previous = now;
    console_emit("[%lus] rate %s Hz | windows %lu sent %lu withheld %lu fail %lu |"
                 " gaps %lu dropped blocks %lu restarts %lu stall %lu\r\n"
                 "  fill max %lu/%u | block max %s ms, DSP %s%% | window latency max %s ms (limit 333)"
                 " | UART done %lu err %lu max %s ms\r\n",
                 (unsigned long)((HAL_GetTick() - started_tick) / 1000U),
                 console_fixed2(rate, sizeof(rate), rate_hz),
                 (unsigned long)live_windows, (unsigned long)live_packets_queued,
                 (unsigned long)live_packets_withheld, (unsigned long)live_packet_failures,
                 (unsigned long)ad7606_gap_events(&now), (unsigned long)live_discarded_blocks,
                 (unsigned long)live_restarts, (unsigned long)ad7606_stalls,
                 (unsigned long)now.max_fill_frames, (unsigned)AD7606_RING_FRAMES,
                 console_fixed2(block_ms, sizeof(block_ms), live_max_block_ms),
                 console_fixed2(load, sizeof(load), live_dsp_load_percent),
                 console_fixed2(window_ms, sizeof(window_ms), live_max_window_latency_ms),
                 (unsigned long)feature_uart_packets_sent, (unsigned long)feature_uart_errors,
                 console_fixed2(uart_ms, sizeof(uart_ms), feature_uart_max_tx_us / 1000.0));
}

bool live_stream_init(void)
{
    /* All four in g, or all four in volts: never mix units between channels. */
    live_calibrated = true;
    for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        live_calibrated = live_calibrated && volts_per_g[channel] > 0.0f;
    for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        scale[channel] = live_calibrated ? AD7606_VOLTS_PER_COUNT / volts_per_g[channel]
                                         : AD7606_VOLTS_PER_COUNT;

    restart_stream();
    console_banner("LIVE: AD7606 -> DSP -> ESP32");
    console_emit("Speed %u RPM, window %lu revolutions (%lu samples), bearing profile %lu,"
                 " channels %s/%s/%s/%s = AIN2/4/6/8.\r\n",
                 (unsigned)COE_LIVE_NOMINAL_RPM, (unsigned long)pipeline.window_revolutions,
                 (unsigned long)pipeline.window_samples, (unsigned long)COE_LIVE_BEARING_PROFILE.id,
                 channel_names[0], channel_names[1], channel_names[2], channel_names[3]);
    console_emit(live_calibrated
                     ? "Calibrated (live_config.h): input in g, packets sent to the ESP32.\r\n"
                     : "UNCALIBRATED: COE_LIVE_VOLTS_PER_G is not set in live_config.h.\r\n"
                       "DSP runs on ADC volts for timing checks; NO packets are sent.\r\n");
    return stream_ok;
}

static void start_live(void)
{
    if (console_start_acquisition() == AD7606_OK)
    {
        reset_counters();
        restart_stream();
        started_tick = report_tick = HAL_GetTick();
    }
}

void live_stream_poll(void)
{
    /* Auto-start: no one needs to press B1. Retried while stopped (ADC not yet
     * powered, or a stall), unless B1 stopped it. */
    static bool stopped_by_user;
    static uint32_t next_auto_tick = COE_LIVE_AUTOSTART_DELAY_MS;
    if (console_button_pressed())
    {
        if (ad7606_running())
        {
            ad7606_stop();
            stopped_by_user = true;
            console_emit("\r\nAD7606 stopped by B1 after %lu windows. Pins held low;"
                         " ADC power may be switched off.\r\n", (unsigned long)live_windows);
        }
        else
        {
            stopped_by_user = false;
            start_live();
        }
    }
    else if (COE_LIVE_AUTOSTART && !stopped_by_user && !ad7606_running()
             && (int32_t)(HAL_GetTick() - next_auto_tick) >= 0)
    {
        next_auto_tick = HAL_GetTick() + COE_LIVE_AUTOSTART_RETRY_MS;
        console_emit("\r\nAuto-start (live_config.h COE_LIVE_AUTOSTART).\r\n");
        start_live();
    }
    if (ad7606_check_stall())
        console_emit("\r\nAD7606 STALL: no frame for %u ms; stopped and pins held low."
                     " Check ADC power/BUSY, then press B1.\r\n", (unsigned)AD7606_STALL_MS);

    consume_blocks();

    if (ad7606_running() && (uint32_t)(HAL_GetTick() - report_tick) >= REPORT_MS)
    {
        report_tick += REPORT_MS;
        report(); /* Short; the 3-KB queue absorbs it with the window lines. */
    }
    console_flush_some();

    ad7606_status_t status;
    ad7606_snapshot(&status);
    const bool error = ad7606_gap_events(&status) + ad7606_stalls + live_pipeline_errors
                       + live_packet_failures + feature_uart_errors != 0U;
    console_update_leds(ad7606_running(), error);
}
