/** Host counter model: arithmetic and IRQ ordering, not electrical timing. */
#ifndef HOST_BENCHMARK_H
#define HOST_BENCHMARK_H
#include <stdbool.h>
#include <stdint.h>
extern uint32_t fake_tick;
static inline bool benchmark_init(void) { return true; }
static inline uint32_t benchmark_cycles_now(void) { return fake_tick * 400000U; }
static inline float benchmark_cycles_to_us(uint32_t cycles) { return (float)cycles / 400.0f; }
#endif
