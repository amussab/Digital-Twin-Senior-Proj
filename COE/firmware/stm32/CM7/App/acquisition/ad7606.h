/**
 * @file ad7606.h
 * Hardware-timed AD7606 acquisition into an AXI SRAM ring of raw frames.
 *
 * TIM4 CH4 (PD15) and CH2 (PB7) pulse CONVST A and B together at the
 * sample rate (CA and CB may also be joined on PD15). BUSY falling (PB9,
 * EXTI9) starts one SPI1 receive-only read of all eight results on DOUTA;
 * DMA2 Stream0 stores them in the next ring frame and its completion
 * interrupt publishes the frame. No DSP runs in these interrupts.
 * USES: TIM4 CH2+CH4, SPI1, DMA2 Stream0, DMAMUX1 channel 8, EXTI9, CM7 only.
 */
#ifndef COE_AD7606_H
#define COE_AD7606_H

#include <stdbool.h>
#include <stdint.h>
#include "pipeline_config.h"

/* One serial frame on DOUTA: V1..V8, 16-bit two's complement, MSB first. */
#define AD7606_SLOTS 8U

/* Confirmed physical wiring: ADC input AINn is serial slot n-1. The pipeline
 * channels B1X, B1Y, B2X, B2Y use AIN2/4/6/8. The AINn-to-slot order is still
 * to be verified on the board with raw_monitor (signal on one input only).
 */
#define AD7606_SLOT_B1X 1U /* AIN2 */
#define AD7606_SLOT_B1Y 3U /* AIN4 */
#define AD7606_SLOT_B2X 5U /* AIN6 */
#define AD7606_SLOT_B2Y 7U /* AIN8 */

/* RANGE low (held by PC7 or tied to 0 V): +/-5 V full scale, 16-bit two's complement. */
#define AD7606_RANGE_VOLTS 5.0f
#define AD7606_VOLTS_PER_COUNT (2.0f * AD7606_RANGE_VOLTS / 65536.0f)

#define AD7606_SAMPLE_RATE_HZ COE_INPUT_SAMPLE_RATE_HZ /* Per channel. */
#define AD7606_MIN_SAMPLE_RATE_HZ 25000U
/* SPI1 kernel clock (PLL1 Q, 200 MHz) / 32 = 6.25 MHz. 16 gives 12.5 MHz once
 * framing is verified; 20 MHz is allowed only for VDRIVE > 4.75 V.
 */
#define AD7606_SPI_DIVIDER 32U
#define AD7606_CONVST_HIGH_NS 1000U /* Datasheet minimum 25 ns. */
#define AD7606_STALL_MS 50U         /* Stop if no frame arrives for this long. */

/* Ring of 16 blocks x 256 frames (64 KiB, about 136 ms at 30 kS/s). */
#define AD7606_BLOCK_FRAMES COE_DMA_BLOCK_SAMPLES
#define AD7606_RING_BLOCKS 16U
#define AD7606_RING_FRAMES (AD7606_BLOCK_FRAMES * AD7606_RING_BLOCKS)

typedef int16_t ad7606_frame_t[AD7606_SLOTS];

/* Results of ad7606_start(). */
typedef enum
{
    AD7606_OK = 0,
    AD7606_ERR_CYCLE_COUNTER,  /* DWT unavailable. */
    AD7606_ERR_TIMER_CLOCK,    /* TIM4 clock gives a rate below the minimum. */
    AD7606_ERR_SPI_CLOCK,      /* PLL1 Q output is not enabled. */
    AD7606_ERR_READ_TOO_SLOW,  /* Conversion + 128 SCLKs exceed the period. */
    AD7606_ERR_BUSY_HIGH,      /* BUSY did not stay low after RESET. */
    AD7606_ERR_NO_RESPONSE,    /* A test CONVST pulse produced no BUSY pulse. */
} ad7606_result_t;

/* Timing actually configured by the latest start (Watch). */
typedef struct
{
    uint32_t timer_clock_hz;     /* TIM4 kernel clock. */
    uint32_t period_ticks;       /* TIM4 ticks per conversion (ARR + 1). */
    uint32_t sample_rate_millihz;
    uint32_t period_cycles;      /* Cortex-M7 cycles per conversion. */
    uint32_t sclk_hz;
    uint32_t read_ns;            /* 128 SCLK periods. */
} ad7606_timing_t;

/* Counters since the latest start (Watch; read in main with ad7606_snapshot). */
typedef struct
{
    uint32_t busy_edges;          /* BUSY falling edges seen. */
    uint32_t frames;              /* Complete frames published to the ring. */
    uint32_t missed_conversions;  /* Inferred from BUSY gaps above 1.5 periods. */
    uint32_t read_overlaps;       /* BUSY fell while the previous read ran. */
    uint32_t ring_overruns;       /* Conversions not read: ring full. */
    uint32_t dma_errors;
    uint32_t spi_overruns;
    uint32_t max_fill_frames;     /* Highest unread ring content. */
    uint32_t busy_latency_min_ticks; /* TIM4 ticks, CONVST rise to BUSY ISR. */
    uint32_t busy_latency_max_ticks;
    uint32_t busy_isr_max_cycles;
    uint32_t dma_isr_max_cycles;
    uint32_t last_busy_cycles;    /* DWT at the latest BUSY edge. */
    uint32_t last_frame_tick;     /* HAL tick of the latest published frame. */
    /* Ring frame index of the first frame after the latest gap event; frames
     * before it are not continuous with frames from it onward.
     */
    uint32_t last_gap_frame;
} ad7606_status_t;

/* Start-up probe of the ADC outputs, recorded by each ad7606_start (Watch).
 * The pull tests use the STM32's weak (about 40 kOhm) internal resistors:
 * a driven line ignores them, a floating line follows them.
 */
#define AD7606_PROBE_SAMPLES 64U
#define AD7606_TIMELINE_EDGES 32U
/* Short test lines (bit numbers): RST, CONVST A, CS, SCLK, BUSY, DOUTA, CONVST B. */
#define AD7606_LINE_BUSY 4U
#define AD7606_LINE_DOUTA 5U
#define AD7606_SHORT_LINES 7U
typedef struct
{
    /* shorts[d]: lines read high (with pull-down) while only output line d is
     * driven high (0 for the two inputs). Any bit other than BUSY means two
     * lines are connected (CONVST A/B together is normal when joined).
     */
    uint32_t shorts[AD7606_SHORT_LINES];
    uint32_t short_baseline; /* Lines high with nothing driven (pull-downs only). */
    /* 20-ms BUSY timeline: short RESET at 1 ms, CONVST at 5 ms, 50-us RESET
     * at 10 ms, CONVST at 15 ms. Each entry is the time (ns) of a BUSY
     * change; levels alternate from timeline_start_level.
     */
    uint32_t timeline_start_level;
    uint32_t timeline_edges;
    uint32_t timeline_ns[AD7606_TIMELINE_EDGES];
    uint32_t busy_high_before_reset; /* High samples of 64, power-up state. */
    uint32_t busy_high_during_reset; /* While RST is held high for 100 us. */
    uint32_t busy_high_after_reset;  /* 1: high 2 ms after RESET, cleared by one CONVST. */
    uint32_t busy_high_no_pull;    /* High samples of 64 after RESET. */
    uint32_t busy_high_pull_down;
    uint32_t busy_high_pull_up;
    uint32_t douta_high_pull_down; /* CS and SCLK low: DOUTA/D7 is driven. */
    uint32_t douta_high_pull_up;
    /* CS high: DOUTA three-states (proves CS reaches the ADC). BUSY is
     * always driven; a "BUSY" wire that floats here is on FD or a data pin.
     */
    uint32_t douta_cs_high_pull_down, douta_cs_high_pull_up;
    uint32_t busy_cs_high_pull_down, busy_cs_high_pull_up;
    /* CS low, SCLK/RD high: serial mode keeps DOUTA driven, parallel mode
     * three-states D7 while RD is high.
     */
    uint32_t douta_rd_high_pull_down, douta_rd_high_pull_up;
    int32_t busy_rise_ns;          /* After the test CONVST rising edge; -1: none. */
    int32_t busy_fall_ns;          /* -1: no fall within the trace. */
    uint32_t busy_edges_seen;      /* BUSY level changes in the 50-us trace. */
} ad7606_probe_t;

extern volatile ad7606_probe_t ad7606_probe;
extern volatile ad7606_status_t ad7606_status;
extern volatile ad7606_timing_t ad7606_timing;
extern volatile uint32_t ad7606_stalls; /* Automatic stops; kept across starts. */

/**
 * Pulse RESET, check one test conversion, then configure SPI1/DMA/EXTI and
 * start TIM4 conversions. CS and SCLK are driven high only after the test
 * conversion answered, which limits driving an unpowered ADC to about 2 us.
 * Precondition: ad7606_pins_hold_safe() was called and the ADC is powered.
 * Counters and the ring restart from zero. On failure the pins stay safe.
 */
ad7606_result_t ad7606_start(void);

/** Stop conversions, finish or abort a running read, hold all pins safe. */
void ad7606_stop(void);

bool ad7606_running(void);

/**
 * Stop automatically when no frame arrived for AD7606_STALL_MS (for example,
 * the ADC lost power). Call from main. Return true if it stopped now.
 */
bool ad7606_check_stall(void);

/** Oldest complete 256-frame block, or NULL. Valid until released. */
const ad7606_frame_t *ad7606_peek_block(void);

/** Return the block from ad7606_peek_block() to the ring. */
void ad7606_release_block(void);

/** Ring frame index of the block ad7606_peek_block() returns (frames since start). */
uint32_t ad7606_block_start(void);

/** Copy the counters consistently with interrupts briefly masked. */
void ad7606_snapshot(ad7606_status_t *status);

/** Events that break sample continuity: missed, overlap, overrun, DMA, SPI. */
static inline uint32_t ad7606_gap_events(const ad7606_status_t *status)
{
    return status->missed_conversions + status->read_overlaps + status->ring_overruns
           + status->dma_errors + status->spi_overruns;
}

#endif
