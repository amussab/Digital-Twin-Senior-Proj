/**
 * @file benchmark.c
 * Measure Cortex-M7 processing time using the hardware cycle counter.
 */
#include "benchmark.h"

#include "stm32h7xx.h"

/** Enable the cycle counter and check that it advances at the current CPU clock. */
bool benchmark_init(void)
{
    SystemCoreClockUpdate();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55UL; /* Unlock the Cortex-M7 DWT registers. */
    if ((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) != 0U)
    {
        return false;
    }
    /* Keep the shared counter running: a UART interval may be in progress. */
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB();
    __ISB();

    const uint32_t before = DWT->CYCCNT;
    for (volatile uint32_t index = 0U; index < 16U; ++index)
    {
        __NOP();
    }
    return (DWT->CYCCNT != before) && (SystemCoreClock != 0U);
}

/** Read the counter after completing earlier memory operations. */
uint32_t benchmark_cycles_now(void)
{
    __DSB();
    __ISB();
    return DWT->CYCCNT;
}

/** Convert an elapsed cycle count to microseconds at the current CPU clock. */
float benchmark_cycles_to_us(uint32_t elapsed_cycles)
{
    return (float)((double)elapsed_cycles * 1000000.0
                   / (double)SystemCoreClock);
}
