/**
 * @file live_stream.c
 * LIVE: the four accelerometers go through the feature pipeline; the RK4
 * Keyphasor gives the speed the windows use; the two RK4 probes are summarised
 * over the same samples. Each window is sent as one schema-2 packet.
 */
#include "live_stream.h"
#include "live_config.h"
#include "acquisition_console.h"
#include "ad7606.h"
#include "benchmark.h"
#include "feature_uart.h"
#include "main.h"
#include "node_packet.h"
#include "pipeline.h"
#include "shaft_sensors.h"
#include <math.h>
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
volatile uint32_t live_rpm;
volatile float live_measured_rpm;
volatile uint32_t live_speed_changes;
volatile uint32_t live_speed_losses;
volatile uint32_t live_unsteady_windows;
volatile float live_last_gap_um[SHAFT_PROBES];
volatile float live_last_x1_um_pp[SHAFT_PROBES];
volatile float live_last_x1_phase_deg[SHAFT_PROBES];

#define REPORT_MS 1000U
#define LATENCY_LIMIT_MS 333.0f /* S2: features within 333 ms of the window's end. */
#define MAX_BLOCKS_PER_POLL 4U /* Keep UART and console servicing between blocks. */

/* Pipeline order B1X, B1Y, B2X, B2Y; slots from the wiring in ad7606.h. */
static const uint32_t slots[COE_CHANNEL_COUNT] = {
    AD7606_SLOT_B1X, AD7606_SLOT_B1Y, AD7606_SLOT_B2X, AD7606_SLOT_B2Y,
};
static const char *const channel_names[COE_CHANNEL_COUNT] = {"B1X", "B1Y", "B2X", "B2Y"};
static const char *const probe_names[SHAFT_PROBES] = {"PROX1", "PROX3"};
static const float volts_per_g[COE_CHANNEL_COUNT] = COE_LIVE_VOLTS_PER_G;
static float scale[COE_CHANNEL_COUNT]; /* Counts -> g, or -> V when uncalibrated. */

static PipelineState pipeline;
static EnvelopeSpectrumWorkspace workspace; /* CPU-only FFT work area in DTCM. */
static PipelineEnvelopeBuffer envelope_buffer
    __attribute__((section(".pipeline_live_buffers"), aligned(32)));
static float input[COE_CHANNEL_COUNT][AD7606_BLOCK_FRAMES];
static ShaftSensors shaft;
static float keyphasor_in[AD7606_BLOCK_FRAMES];
static float probe_in[SHAFT_PROBES][AD7606_BLOCK_FRAMES];
static uint8_t node_packet[COE_NODE_PACKET_BYTES];
static bool shaft_ok;   /* shaft_sensors_init succeeded */
static bool stream_ok;  /* the pipeline is running at live_rpm */
static uint32_t refused_rpm;
static uint32_t handled_gaps;
static uint32_t started_tick, report_tick;
static uint32_t interval_dsp_cycles;
static ad7606_status_t previous;
/* Latency bookkeeping for the block being processed: how long its newest
 * frame had already waited in the ring, how many ms of the block came after
 * the window's last sample (the window can end inside the block), and when
 * its DSP call began.
 */
static float block_wait_ms;
static float block_tail_ms;
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

/** Print the Keyphasor speed and both probes of one window. */
static void print_shaft(const ShaftWindow *w)
{
    char a[16], b[16], c[16];
    if (w->speed_measured)
        console_emit("    Keyphasor: %s RPM (%lu turns, %s..%s)\r\n",
                     console_fixed2(a, sizeof(a), w->rpm), (unsigned long)w->revolutions,
                     console_fixed2(b, sizeof(b), w->rpm_min), console_fixed2(c, sizeof(c), w->rpm_max));
    else
        console_emit("    Keyphasor: no whole turn in this window\r\n");
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
    {
        const ShaftProbeWindow *pr = &w->probes[p];
        char v[16], gap[16], pkpk[16], x1[16], phase[16];
        if (!pr->valid)
        {
            console_emit("    %s: NOT VALID (mean %s V at the probe; valid -19..-1 V, no clipping)\r\n",
                         probe_names[p], console_fixed2(v, sizeof(v), pr->mean_probe_volts));
            continue;
        }
        console_emit("    %s: gap %s um (%s V), pk-pk %s um", probe_names[p],
                     console_fixed2(gap, sizeof(gap), pr->gap_um),
                     console_fixed2(v, sizeof(v), pr->mean_probe_volts),
                     console_fixed2(pkpk, sizeof(pkpk), pr->pkpk_um));
        if (pr->x1_valid)
            console_emit(", 1X %s um pp at %s deg\r\n", console_fixed2(x1, sizeof(x1), pr->x1_um_pp),
                         console_fixed2(phase, sizeof(phase), pr->x1_phase_deg));
        else
            console_emit(", 1X not valid\r\n");
    }
}

/** Sent only if the measured speed stayed at the speed the bands were computed for. */
static bool speed_steady(const ShaftWindow *w, uint32_t nominal_rpm)
{
    const float nominal = (float)nominal_rpm;
    const float tolerance = COE_LIVE_SPEED_TOLERANCE * nominal;
    return w->speed_measured && fabsf(w->rpm - nominal) <= tolerance
           && w->rpm_min >= nominal - 2.0f * tolerance && w->rpm_max <= nominal + 2.0f * tolerance;
}

/** Window callback: add the shaft values, STM32 latency, sample rate and health counters,
 * send the schema-3 packet. */
static bool on_window(const PipelineWindowResult *result, void *context)
{
    (void)context;
    ++live_windows;
    ShaftWindow shaft_window;
    shaft_sensors_finish_window(&shaft, &shaft_window);
    /* STM32 latency (S2 and the end-to-end chain): from the window's last sample
     * (block_tail_ms before the block's newest sample, which waited block_wait_ms
     * in the ring) to the packet being handed to the UART, which follows at once. */
    const float latency = block_tail_ms + block_wait_ms
                          + benchmark_cycles_to_us(benchmark_cycles_now() - block_began_cycles) / 1000.0f;
    const uint32_t latency_us = (uint32_t)ceilf(latency * 1000.0f);
    const bool steady = speed_steady(&shaft_window, result->nominal_rpm);
    const char *outcome;
    if (!live_calibrated)
    {
        ++live_packets_withheld;
        outcome = "packet not sent (uncalibrated)";
    }
    else if (!steady)
    {
        ++live_unsteady_windows;
        outcome = "packet NOT sent: the speed moved during this window";
    }
    else if (!node_packet_encode(result->ai_payload.bytes, &shaft_window, latency_us,
                                 ad7606_timing.sample_rate_millihz,
                                 &(NodeHealth){live_restarts, live_unsteady_windows, live_packet_failures},
                                 node_packet))
    {
        ++live_packet_failures;
        outcome = "packet NOT sent: encoding refused a value";
    }
    else if (feature_uart_send(node_packet, COE_NODE_PACKET_BYTES))
    {
        ++live_packets_queued;
        outcome = "packet sent";
    }
    else
    {
        ++live_packet_failures;
        outcome = "packet NOT sent: UART queue";
    }
    live_last_window_latency_ms = latency;
    if (latency > live_max_window_latency_ms)
        live_max_window_latency_ms = latency;
    for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
        live_last_rms[channel] = result->features[channel].rms;
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
    {
        live_last_gap_um[p] = shaft_window.probes[p].gap_um;
        live_last_x1_um_pp[p] = shaft_window.probes[p].x1_um_pp;
        live_last_x1_phase_deg[p] = shaft_window.probes[p].x1_phase_deg;
    }

    char ms[16], max_ms[16];
    console_emit("  window %lu (%lu samples/channel, bands for %lu RPM): features ready %s ms after its"
                 " last sample (max %s ms, limit 333) %s, %s\r\n",
                 (unsigned long)result->window_id, (unsigned long)result->sample_count,
                 (unsigned long)result->nominal_rpm,
                 console_fixed2(ms, sizeof(ms), latency),
                 console_fixed2(max_ms, sizeof(max_ms), live_max_window_latency_ms),
                 latency <= LATENCY_LIMIT_MS ? "OK" : "OVER LIMIT", outcome);
    print_shaft(&shaft_window);
    print_features(result);
    return true;
}

/** (Re)start the pipeline at live_rpm; a new shaft window starts with it. */
static void restart_stream(void)
{
    stream_ok = pipeline_init(&pipeline, live_rpm, &envelope_buffer, &workspace,
                              &COE_LIVE_BEARING_PROFILE, on_window, NULL)
                && pipeline.window_samples > AD7606_BLOCK_FRAMES; /* at most one window end per block */
    shaft_sensors_start_window(&shaft);
}

/** Between blocks: start, follow or stop the windows with the Keyphasor speed. */
static void follow_speed(void)
{
    double rpm = 0.0;
    const bool locked = shaft_ok && shaft_sensors_speed(&shaft, &rpm);
    live_measured_rpm = locked ? (float)rpm : 0.0f;
    if (!locked)
    {
        if (stream_ok)
        {
            stream_ok = false;
            live_rpm = 0U;
            ++live_speed_losses;
            console_emit("\r\nKeyphasor speed lost or not steady: windows paused until 4 turns agree.\r\n");
        }
        return;
    }
    if (stream_ok && fabs(rpm - live_rpm) <= COE_LIVE_SPEED_RESTART * live_rpm)
        return;
    if (!stream_ok && refused_rpm != 0U && fabs(rpm - refused_rpm) <= COE_LIVE_SPEED_RESTART * refused_rpm)
        return; /* Already refused at about this speed. */
    const uint32_t before = live_rpm;
    live_rpm = (uint32_t)(rpm + 0.5);
    restart_stream();
    char text[16];
    if (!stream_ok)
    {
        refused_rpm = live_rpm;
        console_emit("\r\nKeyphasor speed %s RPM: outside what the pipeline accepts (%u..%u RPM with"
                     " this bearing profile); no windows.\r\n",
                     console_fixed2(text, sizeof(text), rpm), (unsigned)COE_MIN_RPM, (unsigned)COE_MAX_RPM);
        live_rpm = 0U;
        return;
    }
    refused_rpm = 0U;
    if (before != 0U)
        ++live_speed_changes;
    console_emit("\r\nKeyphasor speed %s RPM: windows at %lu RPM, %lu turns (%lu samples)%s.\r\n",
                 console_fixed2(text, sizeof(text), rpm), (unsigned long)live_rpm,
                 (unsigned long)pipeline.window_revolutions, (unsigned long)pipeline.window_samples,
                 before != 0U ? " (speed changed; window restarted)" : "");
}

static void reset_counters(void)
{
    live_windows = live_packets_queued = live_packets_withheld = 0U;
    live_packet_failures = live_discarded_blocks = live_restarts = 0U;
    live_pipeline_errors = 0U;
    live_speed_changes = live_speed_losses = live_unsteady_windows = 0U;
    live_max_block_ms = live_dsp_load_percent = 0.0f;
    live_last_window_latency_ms = live_max_window_latency_ms = 0.0f;
    interval_dsp_cycles = 0U;
    handled_gaps = 0U;
    memset(&previous, 0, sizeof(previous));
}

/** Scale one block; feed the shaft sensors always and the pipeline while it runs.
 * frames_after: frames already published after this block's newest frame.
 */
static void process_block(const ad7606_frame_t *block, uint32_t frames_after)
{
    const uint32_t began = benchmark_cycles_now();
    block_began_cycles = began;
    block_wait_ms = (float)frames_after * 1000.0f / (ad7606_timing.sample_rate_millihz / 1000.0f);
    for (uint32_t frame = 0U; frame < AD7606_BLOCK_FRAMES; ++frame)
    {
        for (uint32_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
            input[channel][frame] = (float)block[frame][slots[channel]] * scale[channel];
        keyphasor_in[frame] = (float)block[frame][AD7606_SLOT_KEYPHASOR] * AD7606_VOLTS_PER_COUNT;
        probe_in[0][frame] = (float)block[frame][AD7606_SLOT_PROX1] * AD7606_VOLTS_PER_COUNT;
        probe_in[1][frame] = (float)block[frame][AD7606_SLOT_PROX3] * AD7606_VOLTS_PER_COUNT;
    }
    const float *const channels[COE_CHANNEL_COUNT] = {input[0], input[1], input[2], input[3]};
    const float *const probes[SHAFT_PROBES] = {probe_in[0], probe_in[1]};

    /* The shaft window must hold exactly the pipeline window's samples: feed the
     * samples up to the window end first, so on_window sees all of them. */
    size_t first = AD7606_BLOCK_FRAMES;
    if (stream_ok)
    {
        const uint32_t room = pipeline.window_samples - pipeline.samples_in_window;
        if (room < first)
            first = room;
    }
    block_tail_ms = (float)(AD7606_BLOCK_FRAMES - first) * 1000.0f
                    / (ad7606_timing.sample_rate_millihz / 1000.0f);
    shaft_sensors_process(&shaft, keyphasor_in, probes, first);
    bool ok = true;
    if (stream_ok)
        ok = pipeline_process_block(&pipeline, channels, AD7606_BLOCK_FRAMES);
    if (first < AD7606_BLOCK_FRAMES)
    {
        const float *const rest[SHAFT_PROBES] = {probe_in[0] + first, probe_in[1] + first};
        shaft_sensors_process(&shaft, keyphasor_in + first, rest, AD7606_BLOCK_FRAMES - first);
    }

    const uint32_t cycles = benchmark_cycles_now() - began;
    interval_dsp_cycles += cycles;
    const float ms = benchmark_cycles_to_us(cycles) / 1000.0f;
    if (ms > live_max_block_ms)
        live_max_block_ms = ms;
    if (!ok)
    {
        ++live_pipeline_errors;
        stream_ok = false; /* Restarted by follow_speed; this window is lost. */
    }
}

/** Take whole blocks; drop the ones around a gap and restart the timing. */
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
            /* Samples are missing: the window and the turn timing are broken. */
            handled_gaps = gaps;
            ++live_restarts;
            shaft_sensors_restart_timing(&shaft);
            stream_ok = false;
            live_rpm = 0U;
        }
        follow_speed();
        process_block(block, status.frames - (start + AD7606_BLOCK_FRAMES));
        ad7606_release_block();
    }
}

static void report(void)
{
    ad7606_status_t now;
    ad7606_snapshot(&now);
    char rate[24], block_ms[24], load[24], uart_ms[24], window_ms[24], speed[24], hi[16], lo[16];
    const uint32_t edges = now.busy_edges - previous.busy_edges;
    const uint32_t cycles = now.last_busy_cycles - previous.last_busy_cycles;
    const double rate_hz = (previous.busy_edges != 0U && cycles != 0U)
                               ? (double)edges * SystemCoreClock / cycles : 0.0;
    live_dsp_load_percent = (float)(100.0 * interval_dsp_cycles
                                    / ((double)SystemCoreClock * REPORT_MS / 1000.0));
    interval_dsp_cycles = 0U;
    previous = now;
    console_emit("[%lus] rate %s Hz | windows %lu sent %lu withheld %lu unsteady %lu fail %lu |"
                 " gaps %lu dropped blocks %lu restarts %lu stall %lu\r\n"
                 "  fill max %lu/%u | block max %s ms, DSP %s%% | window latency max %s ms (limit 333)"
                 " | UART done %lu err %lu max %s ms, RX err %lu\r\n",
                 (unsigned long)((HAL_GetTick() - started_tick) / 1000U),
                 console_fixed2(rate, sizeof(rate), rate_hz),
                 (unsigned long)live_windows, (unsigned long)live_packets_queued,
                 (unsigned long)live_packets_withheld, (unsigned long)live_unsteady_windows,
                 (unsigned long)live_packet_failures,
                 (unsigned long)ad7606_gap_events(&now), (unsigned long)live_discarded_blocks,
                 (unsigned long)live_restarts, (unsigned long)ad7606_stalls,
                 (unsigned long)now.max_fill_frames, (unsigned)AD7606_RING_FRAMES,
                 console_fixed2(block_ms, sizeof(block_ms), live_max_block_ms),
                 console_fixed2(load, sizeof(load), live_dsp_load_percent),
                 console_fixed2(window_ms, sizeof(window_ms), live_max_window_latency_ms),
                 (unsigned long)feature_uart_packets_sent, (unsigned long)feature_uart_errors,
                 console_fixed2(uart_ms, sizeof(uart_ms), feature_uart_max_tx_us / 1000.0),
                 (unsigned long)feature_uart_rx_errors);
    if (live_measured_rpm > 0.0f)
        console_emit("  Keyphasor %s RPM (windows at %lu RPM), turns %lu, rejected pulses %lu\r\n",
                     console_fixed2(speed, sizeof(speed), live_measured_rpm), (unsigned long)live_rpm,
                     (unsigned long)shaft.total_revolutions, (unsigned long)shaft.glitches);
    else
        console_emit("  Keyphasor: NO steady speed (levels %s..%s V at the ADC, notch %s) -> no windows\r\n",
                     console_fixed2(lo, sizeof(lo), shaft.level_low),
                     console_fixed2(hi, sizeof(hi), shaft.level_high),
                     shaft.levels_ok ? "seen" : "not seen (need 0.25 V)");
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
    stream_ok = false;
    live_rpm = 0U;

    console_banner("LIVE: AD7606 -> DSP -> ESP32");
    console_emit("Speed: measured by the RK4 Keyphasor on AIN7 (no set value); windows start when"
                 " 4 turns agree within 1 %%.\r\nBearing profile %lu, channels %s/%s/%s/%s = AIN2/1/3/4,"
                 " %s AIN5, %s AIN6.\r\n",
                 (unsigned long)COE_LIVE_BEARING_PROFILE.id,
                 channel_names[0], channel_names[1], channel_names[2], channel_names[3],
                 probe_names[0], probe_names[1]);
    console_emit(live_calibrated
                     ? "Calibrated (live_config.h): input in g, packets sent to the ESP32.\r\n"
                     : "UNCALIBRATED: COE_LIVE_VOLTS_PER_G is not set in live_config.h.\r\n"
                       "DSP runs on ADC volts for timing checks; NO packets are sent.\r\n");
    return true;
}

static void start_live(void)
{
    if (console_start_acquisition() == AD7606_OK)
    {
        reset_counters();
        const ShaftSensorsConfig config = {
            ad7606_timing.sample_rate_millihz / 1000.0,
            COE_LIVE_PROBE_DIVIDER, COE_LIVE_PROBE_UM_PER_VOLT};
        shaft_ok = shaft_sensors_init(&shaft, &config);
        if (!shaft_ok)
            console_emit("Shaft sensor settings refused (live_config.h): no speed, no windows.\r\n");
        stream_ok = false;
        live_rpm = 0U;
        live_measured_rpm = 0.0f;
        refused_rpm = 0U;
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
