/**
 * @file processing_clock.c
 * Select the stock board's 400-MHz M7 processing clock before releasing M4.
 * PLL1 prefers the ST-LINK's crystal-referenced 8-MHz clock (accurate sample
 * timing) and falls back to the internal 64-MHz HSI oscillator.
 */
#include "processing_clock.h"
#include "benchmark.h"
#include "stm32h7xx_hal.h"

volatile uint32_t processing_clock_source = PROCESSING_CLOCK_NONE;
volatile uint32_t processing_clock_measured_hz;

/* HSI is factory-trimmed to about 1% near room temperature. An unexpected
 * HSE frequency (for example after an ST-LINK firmware change) moves the
 * measured core clock much further than this.
 */
#define CLOCK_CHECK_TOLERANCE_HZ 12000000U /* 3% of 400 MHz. */
/* LPTIM1 counts HSI / 32 = 2 MHz: slow enough for its two-equal-reads rule.
 * (At 64 MHz matching reads were so rare that whole wraps were missed.)
 */
#define CLOCK_CHECK_PRESCALER 32U
#define CLOCK_CHECK_TICKS 40000U /* 20 ms. */

/** Configure and lock PLL1 from HSE bypass or HSI; P=400, Q=200, R=400 MHz. */
static bool start_pll(uint32_t source)
{
    RCC_OscInitTypeDef oscillator = {0};
    oscillator.PLL.PLLState = RCC_PLL_ON;
    oscillator.PLL.PLLSource = source;
    oscillator.PLL.PLLP = 2U;
    oscillator.PLL.PLLQ = 4U; /* 200 MHz: SPI1 kernel clock (SPI123SEL default). */
    oscillator.PLL.PLLR = 2U;
    oscillator.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
    oscillator.PLL.PLLFRACN = 0U;
    if (source == RCC_PLLSOURCE_HSE)
    {
        /* UM2408 default: ST-LINK MCO drives 8 MHz into PH0 (no crystal).
         * 8 MHz / M=2 * N=200 / P=2 = 400 MHz; 4-MHz input, wide VCO 800 MHz.
         */
        oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSE;
        oscillator.HSEState = RCC_HSE_BYPASS;
        oscillator.PLL.PLLM = 2U;
        oscillator.PLL.PLLN = 200U;
        oscillator.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
    }
    else
    {
        /* HSI 64 MHz / M=4 * N=50 / P=2 = 400 MHz; 16-MHz input, wide VCO 800 MHz. */
        oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSI;
        oscillator.HSIState = RCC_HSI_DIV1;
        oscillator.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
        oscillator.PLL.PLLM = 4U;
        oscillator.PLL.PLLN = 50U;
        oscillator.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
    }
    return HAL_RCC_OscConfig(&oscillator) == HAL_OK;
}

/** Switch SYSCLK with the fixed AHB/2 and APB/2 dividers. */
static bool select_system_clock(uint32_t source)
{
    RCC_ClkInitTypeDef clock = {0};
    clock.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK
                      | RCC_CLOCKTYPE_D1PCLK1 | RCC_CLOCKTYPE_PCLK1
                      | RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1;
    clock.SYSCLKSource = source;
    clock.SYSCLKDivider = RCC_SYSCLK_DIV1; /* Cortex-M7: 400 MHz. */
    clock.AHBCLKDivider = RCC_HCLK_DIV2;  /* AXI/AHB and Cortex-M4: 200 MHz. */
    clock.APB1CLKDivider = RCC_APB1_DIV2;
    clock.APB2CLKDivider = RCC_APB2_DIV2;
    clock.APB3CLKDivider = RCC_APB3_DIV2;
    clock.APB4CLKDivider = RCC_APB4_DIV2; /* All APB buses: 100 MHz. */
    /* Conservative Flash latency at VOS1 / 200-MHz AXI clock. HAL raises
     * latency before switching clocks and updates SysTick/SystemCoreClock.
     */
    return HAL_RCC_ClockConfig(&clock, FLASH_LATENCY_4) == HAL_OK;
}

/** Read LPTIM1's asynchronous counter; two equal reads are required. */
static uint32_t lptim1_count(void)
{
    uint32_t first, second;
    do
    {
        first = LPTIM1->CNT;
        second = LPTIM1->CNT;
    } while (first != second);
    return first;
}

/** Count DWT cycles across HSI periods: LPTIM1 runs from per_ck = HSI. */
static uint32_t measure_core_clock_hz(void)
{
    if (!benchmark_init())
        return 0U;
    __HAL_RCC_CLKP_CONFIG(RCC_CLKPSOURCE_HSI);
    __HAL_RCC_LPTIM1_CONFIG(RCC_LPTIM1CLKSOURCE_CLKP);
    __HAL_RCC_LPTIM1_CLK_ENABLE();
    LPTIM1->CR = 0U;
    LPTIM1->CFGR = 5UL << LPTIM_CFGR_PRESC_Pos; /* Internal clock / 32. */
    LPTIM1->CR = LPTIM_CR_ENABLE;
    LPTIM1->ARR = 0xFFFFU;
    const uint32_t limit = 40000000U; /* About 100 ms at any expected clock. */
    uint32_t began = benchmark_cycles_now();
    while ((LPTIM1->ISR & LPTIM_ISR_ARROK) == 0U)
        if (benchmark_cycles_now() - began > limit)
            break;
    LPTIM1->ICR = LPTIM_ICR_ARROKCF;
    LPTIM1->CR |= LPTIM_CR_CNTSTRT;

    uint32_t measured = 0U;
    const uint32_t first = lptim1_count();
    uint32_t count = first;
    began = benchmark_cycles_now();
    while (count == first && benchmark_cycles_now() - began < limit)
        count = lptim1_count(); /* Start on a counter edge. */
    const uint32_t start_cycles = benchmark_cycles_now();
    /* Sum small steps so a counter wrap can never be missed. */
    uint32_t elapsed = 0U;
    while (elapsed < CLOCK_CHECK_TICKS
           && benchmark_cycles_now() - start_cycles < limit)
    {
        const uint32_t next = lptim1_count();
        elapsed += (next - count) & 0xFFFFU;
        count = next;
    }
    const uint32_t cycles = benchmark_cycles_now() - start_cycles;
    if (elapsed >= CLOCK_CHECK_TICKS)
        measured = (uint32_t)((uint64_t)cycles * (HSI_VALUE / CLOCK_CHECK_PRESCALER)
                              / elapsed);

    LPTIM1->CR = 0U;
    __HAL_RCC_LPTIM1_CLK_DISABLE();
    __HAL_RCC_LPTIM1_CONFIG(RCC_LPTIM1CLKSOURCE_D2PCLK1);
    return measured;
}

/** Configure power and clocks, then verify 400/200/100-MHz operating rates. */
bool configure_processing_clock(void)
{
    /* Direct SMPS supports VOS1 / 400 MHz on the stock board.
     * VOS0 / 480 MHz requires an LDO hardware power configuration.
     */
    if (HAL_PWREx_ConfigSupply(PWR_DIRECT_SMPS_SUPPLY) != HAL_OK)
        return false;
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    const uint32_t start = HAL_GetTick();
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY))
        if ((uint32_t)(HAL_GetTick() - start) > 100U)
            return false;

    /* HSE start-up waits at most HSE_STARTUP_TIMEOUT (100 ms). */
    bool use_hse = start_pll(RCC_PLLSOURCE_HSE)
                   && select_system_clock(RCC_SYSCLKSOURCE_PLLCLK);
    if (use_hse)
    {
        processing_clock_measured_hz = measure_core_clock_hz();
        const uint32_t error = processing_clock_measured_hz > 400000000U
                                   ? processing_clock_measured_hz - 400000000U
                                   : 400000000U - processing_clock_measured_hz;
        use_hse = error <= CLOCK_CHECK_TOLERANCE_HZ;
    }
    if (!use_hse)
    {
        /* Leave PLL1 so it can be reconfigured, then stop the external input. */
        if (!select_system_clock(RCC_SYSCLKSOURCE_HSI)
            || !start_pll(RCC_PLLSOURCE_HSI)
            || !select_system_clock(RCC_SYSCLKSOURCE_PLLCLK))
            return false;
        __HAL_RCC_HSE_CONFIG(RCC_HSE_OFF);
        processing_clock_measured_hz = measure_core_clock_hz();
    }
    processing_clock_source = use_hse ? PROCESSING_CLOCK_HSE : PROCESSING_CLOCK_HSI;

    SystemCoreClockUpdate();
    return SystemCoreClock == 400000000U
           && HAL_RCC_GetHCLKFreq() == 200000000U
           && HAL_RCC_GetPCLK1Freq() == 100000000U
           && HAL_RCC_GetPCLK2Freq() == 100000000U;
}
