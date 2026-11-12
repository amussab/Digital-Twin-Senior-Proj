/**
 * @file benchmark.h
 * Measure processing time using the Cortex-M7 hardware cycle counter.
 * Initialize once, read before/after processing, then convert the difference.
 */
#ifndef COE_BENCHMARK_H
#define COE_BENCHMARK_H

#include <stdbool.h>
#include <stdint.h>

/** Enable/reset the counter; return true if it advances and the clock is known. */
bool benchmark_init(void);

/** Read the counter. Unsigned end-start handles a wrap for short intervals. */
uint32_t benchmark_cycles_now(void);

/**
 * Convert elapsed cycles to microseconds using SystemCoreClock.
 * Initialize successfully first and keep the CPU clock fixed while measuring.
 * At 400 MHz, one counter period is approximately 10.74 seconds.
 */
float benchmark_cycles_to_us(uint32_t elapsed_cycles);

#endif
