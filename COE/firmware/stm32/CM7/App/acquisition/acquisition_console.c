/**
 * @file acquisition_console.c
 * Queued USART3 output, B1, LEDs and the AD7606 start-up report.
 */
#include "acquisition_console.h"
#include "main.h"
#include "processing_clock.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

int _write(int file, char *ptr, int len); /* syscalls.c -> BSP COM1. */

#define DEBOUNCE_MS 30U
#define CHUNK_BYTES 64U

/* Output is queued and written in short chunks so the ring keeps draining. */
static char output[3072];
static uint32_t output_length, output_position;
static uint32_t led_tick;

void console_emit(const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    const uint32_t space = sizeof(output) - output_length;
    const int written = vsnprintf(&output[output_length], space, format, arguments);
    va_end(arguments);
    if (written > 0 && (uint32_t)written < space)
        output_length += (uint32_t)written;
    else
        output[output_length] = '\0';
}

void console_flush_some(void)
{
    if (output_position == output_length)
    {
        output_position = output_length = 0U;
        return;
    }
    uint32_t count = output_length - output_position;
    if (count > CHUNK_BYTES)
        count = CHUNK_BYTES;
    (void)_write(1, &output[output_position], (int)count);
    output_position += count;
}

bool console_idle(void)
{
    return output_length == 0U;
}

const char *console_fixed2(char *text, size_t size, double x)
{
    const bool negative = x < 0.0;
    const uint32_t hundredths = (uint32_t)((negative ? -x : x) * 100.0 + 0.5);
    snprintf(text, size, "%s%lu.%02lu", negative ? "-" : "",
             (unsigned long)(hundredths / 100U), (unsigned long)(hundredths % 100U));
    return text;
}

bool console_button_pressed(void)
{
    static bool stable, last_raw;
    static uint32_t changed_tick;
    const bool raw = BSP_PB_GetState(BUTTON_USER) == (int32_t)BUTTON_PRESSED;
    const uint32_t now = HAL_GetTick();
    if (raw != last_raw)
    {
        last_raw = raw;
        changed_tick = now;
    }
    else if (raw != stable && (uint32_t)(now - changed_tick) >= DEBOUNCE_MS)
    {
        stable = raw;
        return stable;
    }
    return false;
}

void console_banner(const char *mode_line)
{
    console_emit("\r\n=== COE CM7 %s ===\r\n"
                 "Clock %s, core measured %lu Hz. ADC pins held low.\r\n"
                 "Switch on the 24 V supply (ADC power), then press B1 to start.\r\n"
                 "Press B1 again to stop BEFORE switching the ADC power off.\r\n",
                 mode_line,
                 processing_clock_source == PROCESSING_CLOCK_HSE ? "HSE 8 MHz (ST-LINK MCO)"
                                                                 : "HSI 64 MHz (HSE failed)",
                 (unsigned long)processing_clock_measured_hz);
    led_tick = HAL_GetTick();
}

static const char *result_text(ad7606_result_t result)
{
    switch (result)
    {
    case AD7606_OK: return "ok";
    case AD7606_ERR_CYCLE_COUNTER: return "cycle counter unavailable";
    case AD7606_ERR_TIMER_CLOCK: return "TIM4 clock cannot give the sample rate";
    case AD7606_ERR_SPI_CLOCK: return "SPI1 kernel clock (PLL1 Q) not available";
    case AD7606_ERR_READ_TOO_SLOW: return "read does not fit in one sample period";
    case AD7606_ERR_BUSY_HIGH: return "BUSY stayed high after RESET (ADC wiring/power?)";
    case AD7606_ERR_NO_RESPONSE: return "no BUSY pulse after a test CONVST (ADC unpowered? BUSY/CONVST wiring?)";
    }
    return "unknown";
}

/** Plain reading of one pull test: driven low/high, or floating. */
static const char *line_state(uint32_t high_with_pull_down, uint32_t high_with_pull_up)
{
    const uint32_t all = AD7606_PROBE_SAMPLES;
    if (high_with_pull_down == 0U && high_with_pull_up == 0U)
        return "driven LOW";
    if (high_with_pull_down == all && high_with_pull_up == all)
        return "driven HIGH";
    if (high_with_pull_down == 0U && high_with_pull_up == all)
        return "FLOATING (not driven: wire, pin or ADC power?)";
    return "unstable (noisy or weakly driven)";
}

static void print_probe(void)
{
    ad7606_probe_t p;
    memcpy(&p, (const void *)&ad7606_probe, sizeof(p));
    console_emit("\r\nTimeline (us): RESET 100 ns @1000, CONVST @5000, RESET 50 us @10000-10050,"
                 " CONVST @15000, end @20000\r\n  BUSY starts %s; %lu changes:",
                 p.timeline_start_level ? "HIGH" : "low", (unsigned long)p.timeline_edges);
    uint32_t level = p.timeline_start_level;
    for (uint32_t i = 0U; i < p.timeline_edges && i < AD7606_TIMELINE_EDGES; ++i)
    {
        level ^= 1U;
        console_emit(" %s@%lu.%01lu", level ? "up" : "down",
                     (unsigned long)(p.timeline_ns[i] / 1000U),
                     (unsigned long)(p.timeline_ns[i] % 1000U / 100U));
        if (i % 6U == 5U)
            console_emit("\r\n   ");
    }
    console_emit("\r\n  (healthy: up just after each CONVST, down about 4 us later;"
                 " RESET should not raise BUSY)\r\n");

    static const char *const names[AD7606_SHORT_LINES] = {
        "RST", "CONVST A", "CS", "SCLK", "BUSY", "DOUTA", "CONVST B"};
    console_emit("Short test (pull-downs on, one line driven high):\r\n  nothing driven -> high:");
    for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i)
        if ((p.short_baseline >> i) & 1U)
            console_emit(" %s", names[i]);
    console_emit("\r\n");
    for (uint32_t d = 0U; d < AD7606_SHORT_LINES; ++d)
    {
        if (d == AD7606_LINE_BUSY || d == AD7606_LINE_DOUTA)
            continue; /* Inputs are not driven. */
        console_emit("  %s high -> also high:", names[d]);
        const uint32_t extra = p.shorts[d] & ~p.short_baseline;
        for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i)
        {
            if (((extra >> i) & 1U) == 0U)
                continue;
            const bool expected = i == AD7606_LINE_BUSY || ((d == 1U && i == 6U) || (d == 6U && i == 1U));
            console_emit(" %s%s", names[i], expected ? "" : " (SHORT?)");
        }
        console_emit(extra == 0U ? " none\r\n" : "\r\n");
    }
    console_emit("\r\nProbe: BUSY high %lu/%u before RESET, %lu/%u while RESET held high"
                 " (expect low while RESET is high)\r\n",
                 (unsigned long)p.busy_high_before_reset, (unsigned)AD7606_PROBE_SAMPLES,
                 (unsigned long)p.busy_high_during_reset, (unsigned)AD7606_PROBE_SAMPLES);
    if (p.busy_high_after_reset)
        console_emit("BUSY was high after RESET; one CONVST was sent to clear it.\r\n");
    console_emit("After RESET (high samples of %u): BUSY no-pull %lu, pull-down %lu,"
                 " pull-up %lu -> %s\r\n",
                 (unsigned)AD7606_PROBE_SAMPLES, (unsigned long)p.busy_high_no_pull,
                 (unsigned long)p.busy_high_pull_down, (unsigned long)p.busy_high_pull_up,
                 line_state(p.busy_high_pull_down, p.busy_high_pull_up));
    console_emit("  DOUTA (CS low) pull-down %lu, pull-up %lu -> %s\r\n",
                 (unsigned long)p.douta_high_pull_down, (unsigned long)p.douta_high_pull_up,
                 line_state(p.douta_high_pull_down, p.douta_high_pull_up));
    console_emit("  CS high: DOUTA -> %s (expect FLOATING: CS reaches the ADC)\r\n",
                 line_state(p.douta_cs_high_pull_down, p.douta_cs_high_pull_up));
    console_emit("  CS high: BUSY -> %s (expect driven; FLOATING: wire is on FD/a data pin)\r\n",
                 line_state(p.busy_cs_high_pull_down, p.busy_cs_high_pull_up));
    console_emit("  CS low, SCLK/RD high: DOUTA -> %s (driven: serial mode; FLOATING: parallel mode)\r\n",
                 line_state(p.douta_rd_high_pull_down, p.douta_rd_high_pull_up));
    console_emit("  Test CONVST (1 us high): BUSY rise %ld ns, fall %ld ns, edges %lu"
                 " (expect rise <100, fall about 4000, edges 2; -1 = none)\r\n",
                 (long)p.busy_rise_ns, (long)p.busy_fall_ns, (unsigned long)p.busy_edges_seen);
}

ad7606_result_t console_start_acquisition(void)
{
    const ad7606_result_t result = ad7606_start();
    if (result == AD7606_OK || result == AD7606_ERR_BUSY_HIGH
        || result == AD7606_ERR_NO_RESPONSE)
        print_probe();
    if (result != AD7606_OK)
    {
        console_emit("\r\nAD7606 start failed: %s. Pins held low.\r\n", result_text(result));
        return result;
    }
    char a[24];
    console_emit("\r\nAD7606 started: %s S/s per channel (TIM4 %lu Hz / %lu), SCLK %lu Hz,"
                 " read %lu ns, clock %s\r\n",
                 console_fixed2(a, sizeof(a), ad7606_timing.sample_rate_millihz / 1000.0),
                 (unsigned long)ad7606_timing.timer_clock_hz,
                 (unsigned long)ad7606_timing.period_ticks,
                 (unsigned long)ad7606_timing.sclk_hz, (unsigned long)ad7606_timing.read_ns,
                 processing_clock_source == PROCESSING_CLOCK_HSE ? "HSE 8 MHz" : "HSI 64 MHz");
    return result;
}

void console_update_leds(bool running, bool error)
{
    const uint32_t now = HAL_GetTick();
    const uint32_t blink_ms = running ? 100U : 500U;
    if ((uint32_t)(now - led_tick) >= blink_ms)
    {
        BSP_LED_Toggle(LED_GREEN);
        led_tick = now;
    }
    if (error)
        BSP_LED_On(LED_RED);
    else
        BSP_LED_Off(LED_RED);
}
