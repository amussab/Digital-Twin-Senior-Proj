/**
 * @file processing_clock.h
 * Set the operating clock for the unmodified NUCLEO-H755ZI-Q board.
 */
#ifndef COE_PROCESSING_CLOCK_H
#define COE_PROCESSING_CLOCK_H

#include <stdbool.h>
#include <stdint.h>

/* Values of processing_clock_source (Watch). */
#define PROCESSING_CLOCK_NONE 0U
#define PROCESSING_CLOCK_HSE 1U /* ST-LINK 8-MHz MCO; preferred for sample timing. */
#define PROCESSING_CLOCK_HSI 2U /* Internal 64-MHz oscillator; fallback. */

/* Watch: the PLL1 reference actually in use. */
extern volatile uint32_t processing_clock_source;
/* Watch: Cortex-M7 clock measured against HSI (independent of the PLL source). */
extern volatile uint32_t processing_clock_measured_hz;

/**
 * Set M7=400 MHz, AHB/M4=200 MHz and APB buses=100 MHz using direct SMPS.
 * PLL1 uses HSE bypass when it starts and measures correctly, otherwise HSI.
 * Call once after generated base-clock initialization, before releasing M4
 * and before initializing peripherals. Return false on initialization failure.
 */
bool configure_processing_clock(void);

#endif
