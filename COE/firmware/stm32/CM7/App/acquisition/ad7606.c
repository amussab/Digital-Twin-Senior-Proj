/**
 * @file ad7606.c
 * TIM4-timed AD7606 conversions, SPI1 receive-only reads by DMA2 Stream0.
 * Direct register access: the HAL SPI/TIM modules are not enabled.
 */
#include "ad7606.h"
#include "ad7606_pins.h"
#include "benchmark.h"
#include "stm32h7xx_hal.h"
#include <string.h>

volatile ad7606_probe_t ad7606_probe;
volatile ad7606_status_t ad7606_status;
volatile ad7606_timing_t ad7606_timing;
volatile uint32_t ad7606_stalls;

/* DMA2 cannot reach DTCM: the ring lives in AXI SRAM. NOLOAD: DMA writes
 * every frame before it is published. Blocks are 4096 bytes, so whole
 * 32-byte cache lines belong to one block if D-cache is enabled later.
 */
static ad7606_frame_t ring[AD7606_RING_FRAMES]
    __attribute__((section(".acquisition_dma"), aligned(32)));
_Static_assert((AD7606_RING_FRAMES & (AD7606_RING_FRAMES - 1U)) == 0U,
               "Ring index uses a power-of-two mask");
_Static_assert(sizeof(ad7606_frame_t) * AD7606_BLOCK_FRAMES % 32U == 0U,
               "Blocks own complete cache lines");

static volatile uint32_t head; /* Frames published; written by the DMA ISR. */
static volatile uint32_t tail; /* Frames released; written by main. */
static volatile bool reading;  /* A DMA read is in progress. */
static volatile bool running;

#define BUSY_EXTI_LINE (1UL << 9)
#define CONVERSION_MAX_NS 4500U /* AD7606 tCONV 4.15 us max, no oversampling. */
#define READ_OVERHEAD_NS 1000U  /* ISR entry, CS and DMA set-up allowance. */

#define DMA_STREAM0_FLAGS (DMA_LIFCR_CFEIF0 | DMA_LIFCR_CDMEIF0 | DMA_LIFCR_CTEIF0 \
                           | DMA_LIFCR_CHTIF0 | DMA_LIFCR_CTCIF0)
#define SPI_ALL_FLAGS 0x00000FF8UL /* EOTC..SUSPC in SPI_IFCR. */
#define SPI_CR1_IDLE SPI_CR1_SSI   /* Software NSS high: master, no MODF. */

/** TIM4 kernel clock: twice PCLK1 when APB1 is divided (TIMPRE = 0). */
static uint32_t tim4_clock_hz(void)
{
    if ((RCC->CFGR & RCC_CFGR_TIMPRE) != 0U)
        return 0U; /* Not used by this firmware's clock tree. */
    const uint32_t pclk1 = HAL_RCC_GetPCLK1Freq();
    return (RCC->D2CFGR & RCC_D2CFGR_D2PPRE1_2) != 0U ? 2U * pclk1 : pclk1;
}

/** SPI1 baud-rate field for a power-of-two divider from 2 to 256. */
static uint32_t spi_mbr(uint32_t divider)
{
    uint32_t mbr = 0U;
    while ((2UL << mbr) < divider)
        ++mbr;
    return mbr;
}

static void wait_cycles(uint32_t cycles)
{
    const uint32_t began = benchmark_cycles_now();
    while (benchmark_cycles_now() - began < cycles)
    {
    }
}

static void set_alternate(GPIO_TypeDef *port, uint32_t pin, uint32_t alternate,
                          uint32_t speed)
{
    GPIO_InitTypeDef init = {0};
    init.Pin = pin;
    init.Mode = GPIO_MODE_AF_PP;
    init.Pull = GPIO_NOPULL;
    init.Speed = speed;
    init.Alternate = alternate;
    HAL_GPIO_Init(port, &init);
}

/** Manual CONVST A and B level (GPIO mode, outside acquisition): back to back. */
static void convst_write(bool high)
{
    AD7606_CONVST_PORT->BSRR = high ? AD7606_CONVST_PIN : (uint32_t)AD7606_CONVST_PIN << 16U;
    AD7606_CONVSTB_PORT->BSRR = high ? AD7606_CONVSTB_PIN : (uint32_t)AD7606_CONVSTB_PIN << 16U;
}

/** Disable every acquisition interrupt and peripheral, then hold pins safe. */
static void shut_down(void)
{
    /* CONVST first: no conversion may start after this point. */
    TIM4->CR1 &= ~TIM_CR1_CEN;
    TIM4->CCER = 0U;
    convst_write(false);
    GPIO_InitTypeDef convst = {0};
    convst.Mode = GPIO_MODE_OUTPUT_PP;
    convst.Speed = GPIO_SPEED_FREQ_LOW;
    convst.Pin = AD7606_CONVST_PIN;
    HAL_GPIO_Init(AD7606_CONVST_PORT, &convst);
    convst.Pin = AD7606_CONVSTB_PIN;
    HAL_GPIO_Init(AD7606_CONVSTB_PORT, &convst);
    EXTI_D1->IMR1 &= ~BUSY_EXTI_LINE;
    HAL_NVIC_DisableIRQ(EXTI9_5_IRQn);

    /* A read takes about 20 us at 6.25 MHz; abort it after 100 us. */
    const uint32_t began = benchmark_cycles_now();
    while (reading && benchmark_cycles_now() - began < SystemCoreClock / 10000U)
    {
    }
    HAL_NVIC_DisableIRQ(DMA2_Stream0_IRQn);
    DMA2_Stream0->CR &= ~DMA_SxCR_EN;
    while ((DMA2_Stream0->CR & DMA_SxCR_EN) != 0U)
    {
    }
    DMA2->LIFCR = DMA_STREAM0_FLAGS;
    SPI1->CR1 = SPI_CR1_IDLE;
    SPI1->IFCR = SPI_ALL_FLAGS;
    EXTI_D1->PR1 = BUSY_EXTI_LINE;
    HAL_NVIC_ClearPendingIRQ(EXTI9_5_IRQn);
    HAL_NVIC_ClearPendingIRQ(DMA2_Stream0_IRQn);
    reading = false;
    running = false;
    ad7606_pins_hold_safe(); /* RST, SCLK, CS and CONVST A/B low. */
}

/** Calculate and check timer, SPI and read timing for the current clocks. */
static ad7606_result_t configure_timing(void)
{
    ad7606_timing_t timing = {0};
    timing.timer_clock_hz = tim4_clock_hz();
    if (timing.timer_clock_hz == 0U)
        return AD7606_ERR_TIMER_CLOCK;
    timing.period_ticks = (timing.timer_clock_hz + AD7606_SAMPLE_RATE_HZ / 2U)
                          / AD7606_SAMPLE_RATE_HZ;
    timing.sample_rate_millihz = (uint32_t)((uint64_t)timing.timer_clock_hz * 1000U
                                            / timing.period_ticks);
    if (timing.period_ticks > 0x10000U /* TIM4 is 16-bit. */
        || timing.sample_rate_millihz < AD7606_MIN_SAMPLE_RATE_HZ * 1000U)
        return AD7606_ERR_TIMER_CLOCK;
    timing.period_cycles = (uint32_t)((uint64_t)SystemCoreClock * timing.period_ticks
                                      / timing.timer_clock_hz);

    /* SPI1 kernel clock: SPI123SEL = 0 selects PLL1 Q. */
    if ((RCC->PLLCFGR & RCC_PLLCFGR_DIVQ1EN) == 0U
        || (RCC->D2CCIP1R & RCC_D2CCIP1R_SPI123SEL) != 0U)
        return AD7606_ERR_SPI_CLOCK;
    PLL1_ClocksTypeDef pll1;
    HAL_RCCEx_GetPLL1ClockFreq(&pll1);
    timing.sclk_hz = pll1.PLL1_Q_Frequency / AD7606_SPI_DIVIDER;
    if (timing.sclk_hz == 0U)
        return AD7606_ERR_SPI_CLOCK;
    timing.read_ns = (uint32_t)(128ULL * 1000000000ULL / timing.sclk_hz);
    const uint64_t period_ns = (uint64_t)timing.period_ticks * 1000000000ULL
                               / timing.timer_clock_hz;
    if (CONVERSION_MAX_NS + READ_OVERHEAD_NS + timing.read_ns >= period_ns)
        return AD7606_ERR_READ_TOO_SLOW;
    memcpy((void *)&ad7606_timing, &timing, sizeof(timing));
    return AD7606_OK;
}

/** Count high readings with one internal pull applied, then remove the pull. */
static uint32_t count_high(GPIO_TypeDef *port, uint32_t pin, uint32_t pull)
{
    const uint32_t shift = 2U * (uint32_t)__builtin_ctz(pin);
    port->PUPDR = (port->PUPDR & ~(3UL << shift)) | (pull << shift);
    wait_cycles(SystemCoreClock / 100000U); /* 10 us to settle. */
    uint32_t high = 0U;
    for (uint32_t sample = 0U; sample < AD7606_PROBE_SAMPLES; ++sample)
    {
        if ((port->IDR & pin) != 0U)
            ++high;
        wait_cycles(SystemCoreClock / 4000000U);
    }
    port->PUPDR &= ~(3UL << shift); /* Back to no pull. */
    return high;
}

/** One manual 1-us CONVST pulse; time BUSY edges over 50 us. */
static void trace_test_conversion(ad7606_probe_t *probe)
{
    const uint32_t cycles_per_us = SystemCoreClock / 1000000U;
    probe->busy_rise_ns = -1;
    probe->busy_fall_ns = -1;
    probe->busy_edges_seen = 0U;
    uint32_t level = AD7606_BUSY_PORT->IDR & AD7606_BUSY_PIN;
    bool convst_high = true;
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();
    convst_write(true);
    const uint32_t began = DWT->CYCCNT;
    uint32_t elapsed;
    while ((elapsed = DWT->CYCCNT - began) < 50U * cycles_per_us)
    {
        if (convst_high && elapsed >= cycles_per_us)
        {
            convst_write(false);
            convst_high = false;
        }
        const uint32_t now_level = AD7606_BUSY_PORT->IDR & AD7606_BUSY_PIN;
        if (now_level == level)
            continue;
        level = now_level;
        ++probe->busy_edges_seen;
        const int32_t ns = (int32_t)(elapsed * 1000U / cycles_per_us);
        if (level != 0U && probe->busy_rise_ns < 0)
            probe->busy_rise_ns = ns;
        else if (level == 0U && probe->busy_rise_ns >= 0 && probe->busy_fall_ns < 0)
            probe->busy_fall_ns = ns;
    }
    convst_write(false);
    __set_PRIMASK(primask);
}

typedef struct
{
    GPIO_TypeDef *port;
    uint32_t pin;
} line_t;

/* Order matches AD7606_LINE_* in ad7606.h. */
static const line_t lines[AD7606_SHORT_LINES] = {
    {AD7606_RST_PORT, AD7606_RST_PIN},     {AD7606_CONVST_PORT, AD7606_CONVST_PIN},
    {AD7606_CS_PORT, AD7606_CS_PIN},       {AD7606_SCLK_PORT, AD7606_SCLK_PIN},
    {AD7606_BUSY_PORT, AD7606_BUSY_PIN},   {AD7606_DOUTA_PORT, AD7606_DOUTA_PIN},
    {AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN},
};

static bool is_output_line(uint32_t line)
{
    return line != AD7606_LINE_BUSY && line != AD7606_LINE_DOUTA;
}

static void set_mode(const line_t *line, uint32_t mode, uint32_t pull)
{
    const uint32_t shift = 2U * (uint32_t)__builtin_ctz(line->pin);
    line->port->MODER = (line->port->MODER & ~(3UL << shift)) | (mode << shift);
    line->port->PUPDR = (line->port->PUPDR & ~(3UL << shift)) | (pull << shift);
}

/** Drive one control line high at a time; see which other lines follow. */
static void find_shorts(ad7606_probe_t *probe)
{
    for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i)
    {
        lines[i].port->BSRR = lines[i].pin << 16U;
        set_mode(&lines[i], 0U, GPIO_PULLDOWN);
    }
    wait_cycles(SystemCoreClock / 100000U);
    probe->short_baseline = 0U;
    for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i)
        if ((lines[i].port->IDR & lines[i].pin) != 0U)
            probe->short_baseline |= 1UL << i;

    for (uint32_t driver = 0U; driver < AD7606_SHORT_LINES; ++driver)
    {
        probe->shorts[driver] = 0U;
        if (!is_output_line(driver))
            continue;
        for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i) /* All inputs with pull-down. */
            set_mode(&lines[i], 0U, GPIO_PULLDOWN);
        lines[driver].port->BSRR = lines[driver].pin;
        set_mode(&lines[driver], 1U, GPIO_NOPULL);
        wait_cycles(SystemCoreClock / 100000U); /* 10 us. */
        uint32_t high = 0U;
        for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i)
            if (i != driver && (lines[i].port->IDR & lines[i].pin) != 0U)
                high |= 1UL << i;
        probe->shorts[driver] = high;
        lines[driver].port->BSRR = lines[driver].pin << 16U;
    }
    /* Back to the safe state: outputs low, inputs without pulls. */
    for (uint32_t i = 0U; i < AD7606_SHORT_LINES; ++i)
    {
        lines[i].port->BSRR = lines[i].pin << 16U;
        set_mode(&lines[i], is_output_line(i) ? 1U : 0U, GPIO_NOPULL);
    }
}

/** Scripted RESET/CONVST sequence over 20 ms, logging every BUSY change. */
static void record_timeline(ad7606_probe_t *probe)
{
    const uint32_t per_us = SystemCoreClock / 1000000U;
    /* Step times in cycles: pin, set (1) or clear (0). */
    const struct { uint32_t at; GPIO_TypeDef *port; uint32_t pin; bool high; } steps[] = {
        {1000U * per_us, AD7606_RST_PORT, AD7606_RST_PIN, true},
        {1000U * per_us + per_us / 10U, AD7606_RST_PORT, AD7606_RST_PIN, false}, /* 100 ns. */
        {5000U * per_us, AD7606_CONVST_PORT, AD7606_CONVST_PIN, true},
        {5000U * per_us, AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN, true},
        {5001U * per_us, AD7606_CONVST_PORT, AD7606_CONVST_PIN, false},
        {5001U * per_us, AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN, false},
        {10000U * per_us, AD7606_RST_PORT, AD7606_RST_PIN, true},
        {10050U * per_us, AD7606_RST_PORT, AD7606_RST_PIN, false},                /* 50 us. */
        {15000U * per_us, AD7606_CONVST_PORT, AD7606_CONVST_PIN, true},
        {15000U * per_us, AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN, true},
        {15001U * per_us, AD7606_CONVST_PORT, AD7606_CONVST_PIN, false},
        {15001U * per_us, AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN, false},
    };
    const uint32_t step_count = sizeof(steps) / sizeof(steps[0]);
    uint32_t next_step = 0U;
    uint32_t level = AD7606_BUSY_PORT->IDR & AD7606_BUSY_PIN;
    probe->timeline_start_level = level != 0U;
    probe->timeline_edges = 0U;

    const uint32_t primask = __get_PRIMASK();
    __disable_irq();
    const uint32_t began = DWT->CYCCNT;
    uint32_t elapsed;
    while ((elapsed = DWT->CYCCNT - began) < 20000U * per_us)
    {
        if (next_step < step_count && elapsed >= steps[next_step].at)
        {
            steps[next_step].port->BSRR = steps[next_step].high
                                              ? steps[next_step].pin
                                              : steps[next_step].pin << 16U;
            ++next_step;
        }
        const uint32_t now_level = AD7606_BUSY_PORT->IDR & AD7606_BUSY_PIN;
        if (now_level == level)
            continue;
        level = now_level;
        if (probe->timeline_edges < AD7606_TIMELINE_EDGES)
            probe->timeline_ns[probe->timeline_edges] = (uint32_t)((uint64_t)elapsed * 1000U / per_us);
        ++probe->timeline_edges;
    }
    AD7606_RST_PORT->BSRR = (uint32_t)AD7606_RST_PIN << 16U;
    convst_write(false);
    __set_PRIMASK(primask);
}

ad7606_result_t ad7606_start(void)
{
    if (running)
        return AD7606_OK;
    if (!benchmark_init())
        return AD7606_ERR_CYCLE_COUNTER;
    const ad7606_result_t timing_result = configure_timing();
    if (timing_result != AD7606_OK)
        return timing_result;

    memset((void *)&ad7606_status, 0, sizeof(ad7606_status));
    ad7606_status.busy_latency_min_ticks = UINT32_MAX;
    head = 0U;
    tail = 0U;
    reading = false;

    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_DMA2_CLK_ENABLE();
    __HAL_RCC_TIM4_CLK_ENABLE();
    __HAL_RCC_SYSCFG_CLK_ENABLE();

    /* RESET pulse (datasheet minimum 50 ns) is required after power-up.
     * BUSY is sampled before and during it to see whether RESET acts.
     */
    ad7606_probe_t probe = {0};
    probe.busy_high_before_reset = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_NOPULL);
    record_timeline(&probe);
    find_shorts(&probe);
    HAL_GPIO_WritePin(AD7606_RST_PORT, AD7606_RST_PIN, GPIO_PIN_SET);
    wait_cycles(SystemCoreClock / 20000U); /* 50 us, then 64 samples. */
    probe.busy_high_during_reset = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_NOPULL);
    HAL_GPIO_WritePin(AD7606_RST_PORT, AD7606_RST_PIN, GPIO_PIN_RESET);
    HAL_Delay(2U);

    /* On the board this AD7606 holds BUSY high after RESET until the next
     * CONVST, which then converts normally: clear it with one conversion.
     */
    probe.busy_high_after_reset =
        HAL_GPIO_ReadPin(AD7606_BUSY_PORT, AD7606_BUSY_PIN) == GPIO_PIN_SET;
    if (probe.busy_high_after_reset)
    {
        convst_write(true);
        wait_cycles(SystemCoreClock / 1000000U);
        convst_write(false);
        const uint32_t began = benchmark_cycles_now();
        while (HAL_GPIO_ReadPin(AD7606_BUSY_PORT, AD7606_BUSY_PIN) == GPIO_PIN_SET
               && benchmark_cycles_now() - began < SystemCoreClock / 10000U) /* 100 us. */
        {
        }
    }

    /* Is each ADC output driven, or floating? CS and SCLK start low. */
    probe.busy_high_no_pull = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_NOPULL);
    probe.busy_high_pull_down = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_PULLDOWN);
    probe.busy_high_pull_up = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_PULLUP);
    probe.douta_high_pull_down = count_high(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_PULLDOWN);
    probe.douta_high_pull_up = count_high(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_PULLUP);

    HAL_GPIO_WritePin(AD7606_CS_PORT, AD7606_CS_PIN, GPIO_PIN_SET);
    probe.douta_cs_high_pull_down = count_high(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_PULLDOWN);
    probe.douta_cs_high_pull_up = count_high(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_PULLUP);
    probe.busy_cs_high_pull_down = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_PULLDOWN);
    probe.busy_cs_high_pull_up = count_high(AD7606_BUSY_PORT, AD7606_BUSY_PIN, GPIO_PULLUP);
    HAL_GPIO_WritePin(AD7606_CS_PORT, AD7606_CS_PIN, GPIO_PIN_RESET);

    HAL_GPIO_WritePin(AD7606_SCLK_PORT, AD7606_SCLK_PIN, GPIO_PIN_SET);
    probe.douta_rd_high_pull_down = count_high(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_PULLDOWN);
    probe.douta_rd_high_pull_up = count_high(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_PULLUP);
    HAL_GPIO_WritePin(AD7606_SCLK_PORT, AD7606_SCLK_PIN, GPIO_PIN_RESET);

    trace_test_conversion(&probe);
    memcpy((void *)&ad7606_probe, &probe, sizeof(probe));
    if (probe.busy_high_pull_down == AD7606_PROBE_SAMPLES)
    {
        shut_down(); /* Driven high while idle. */
        return AD7606_ERR_BUSY_HIGH;
    }
    if (probe.busy_rise_ns < 0 || probe.busy_fall_ns < 0)
    {
        shut_down();
        return AD7606_ERR_NO_RESPONSE;
    }

    /* The ADC answered, so it is powered: deselect it. */
    HAL_GPIO_WritePin(AD7606_CS_PORT, AD7606_CS_PIN, GPIO_PIN_SET);

    /* SPI1: master, receive-only, 16-bit, mode 2 (CPOL=1, CPHA=0): DOUTA
     * changes on SCLK rising and is sampled on falling. CFG1/CFG2/CR2 are
     * written only while SPE = 0. AFCNTR keeps SCLK idling high between reads.
     */
    SPI1->CR1 = SPI_CR1_IDLE;
    SPI1->CFG1 = (spi_mbr(AD7606_SPI_DIVIDER) << SPI_CFG1_MBR_Pos)
                 | SPI_CFG1_RXDMAEN
                 | (15U << SPI_CFG1_DSIZE_Pos); /* FTHLV = 0: one frame per request. */
    SPI1->CFG2 = SPI_CFG2_AFCNTR | SPI_CFG2_SSM | SPI_CFG2_CPOL | SPI_CFG2_MASTER
                 | SPI_CFG2_COMM_1; /* COMM = 10: simplex receiver. */
    SPI1->CR2 = AD7606_SLOTS << SPI_CR2_TSIZE_Pos;
    SPI1->IFCR = SPI_ALL_FLAGS;
    /* Slowest edge rate: long jumper wires ring with fast edges, and a ring
     * on SCLK counts as an extra clock (seen as doubled values). Low speed
     * still supports well above 6.25 MHz.
     */
    set_alternate(AD7606_SCLK_PORT, AD7606_SCLK_PIN, GPIO_AF5_SPI1, GPIO_SPEED_FREQ_LOW);
    set_alternate(AD7606_DOUTA_PORT, AD7606_DOUTA_PIN, GPIO_AF5_SPI1, GPIO_SPEED_FREQ_HIGH);

    /* DMA2 Stream0 <- DMAMUX1 channel 8 <- SPI1_RX: 16-bit peripheral to memory. */
    DMA2_Stream0->CR = 0U;
    while ((DMA2_Stream0->CR & DMA_SxCR_EN) != 0U)
    {
    }
    DMA2->LIFCR = DMA_STREAM0_FLAGS;
    DMA2_Stream0->PAR = (uint32_t)&SPI1->RXDR;
    DMA2_Stream0->FCR = 0U; /* Direct mode. */
    DMA2_Stream0->CR = DMA_SxCR_PL_1 | DMA_SxCR_MSIZE_0 | DMA_SxCR_PSIZE_0
                       | DMA_SxCR_MINC | DMA_SxCR_TCIE | DMA_SxCR_TEIE | DMA_SxCR_DMEIE;
    DMAMUX1_Channel8->CCR = DMA_REQUEST_SPI1_RX;
    HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 1U, 0U);
    HAL_NVIC_ClearPendingIRQ(DMA2_Stream0_IRQn);
    HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);

    /* BUSY falling edge on PB9 -> EXTI9, highest priority. */
    SYSCFG->EXTICR[2] = (SYSCFG->EXTICR[2] & ~SYSCFG_EXTICR3_EXTI9_Msk)
                        | (1UL << SYSCFG_EXTICR3_EXTI9_Pos); /* Port B. */
    EXTI->RTSR1 &= ~BUSY_EXTI_LINE;
    EXTI->FTSR1 |= BUSY_EXTI_LINE;
    EXTI_D1->PR1 = BUSY_EXTI_LINE;
    EXTI_D1->IMR1 |= BUSY_EXTI_LINE;
    HAL_NVIC_SetPriority(EXTI9_5_IRQn, 0U, 0U);
    HAL_NVIC_ClearPendingIRQ(EXTI9_5_IRQn);
    HAL_NVIC_EnableIRQ(EXTI9_5_IRQn);

    /* TIM4 PWM mode 1 on CH4 (CONVST A) and CH2 (CONVST B), same counter and
     * compare value: both high for the first CCR ticks of each period, so each
     * update produces the rising edges that start a conversion, together.
     * Starting with CNT = CCR keeps the outputs low until the first update.
     */
    const uint32_t high_ticks = (uint32_t)((uint64_t)ad7606_timing.timer_clock_hz
                                           * AD7606_CONVST_HIGH_NS / 1000000000ULL);
    TIM4->CR1 = 0U;
    TIM4->PSC = 0U;
    TIM4->ARR = ad7606_timing.period_ticks - 1U;
    TIM4->CCR4 = high_ticks;
    TIM4->CCR2 = high_ticks;
    TIM4->CCMR2 = (6UL << TIM_CCMR2_OC4M_Pos) | TIM_CCMR2_OC4PE;
    TIM4->CCMR1 = (6UL << TIM_CCMR1_OC2M_Pos) | TIM_CCMR1_OC2PE;
    TIM4->CCER = TIM_CCER_CC4E | TIM_CCER_CC2E;
    TIM4->CR1 = TIM_CR1_ARPE;
    TIM4->EGR = TIM_EGR_UG;
    TIM4->CNT = high_ticks;
    TIM4->SR = 0U;
    set_alternate(AD7606_CONVST_PORT, AD7606_CONVST_PIN, GPIO_AF2_TIM4, GPIO_SPEED_FREQ_LOW);
    set_alternate(AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN, GPIO_AF2_TIM4, GPIO_SPEED_FREQ_LOW);

    ad7606_status.last_frame_tick = uwTick;
    ad7606_status.last_busy_cycles = DWT->CYCCNT;
    running = true;
    TIM4->CR1 = TIM_CR1_ARPE | TIM_CR1_CEN;
    return AD7606_OK;
}

void ad7606_stop(void)
{
    if (running)
        shut_down();
}

bool ad7606_running(void)
{
    return running;
}

bool ad7606_check_stall(void)
{
    if (!running || (uint32_t)(uwTick - ad7606_status.last_frame_tick) <= AD7606_STALL_MS)
        return false;
    shut_down();
    ++ad7606_stalls;
    return true;
}

const ad7606_frame_t *ad7606_peek_block(void)
{
    const uint32_t first = tail;
    if (head - first < AD7606_BLOCK_FRAMES)
        return NULL;
    __DMB(); /* Read frame data only after observing its publication. */
    const ad7606_frame_t *block = &ring[first & (AD7606_RING_FRAMES - 1U)];
    if ((SCB->CCR & SCB_CCR_DC_Msk) != 0U)
        SCB_InvalidateDCache_by_Addr((void *)block,
                                     (int32_t)(sizeof(ad7606_frame_t) * AD7606_BLOCK_FRAMES));
    return block;
}

void ad7606_release_block(void)
{
    __DMB(); /* Finish reading before DMA may reuse the frames. */
    tail = tail + AD7606_BLOCK_FRAMES;
}

uint32_t ad7606_block_start(void)
{
    return tail;
}

void ad7606_snapshot(ad7606_status_t *status)
{
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();
    memcpy(status, (const void *)&ad7606_status, sizeof(*status));
    __set_PRIMASK(primask);
}

/** BUSY fell: results are ready. Start one 8-frame read into the next slot. */
void EXTI9_5_IRQHandler(void)
{
    const uint32_t entry = DWT->CYCCNT;
    const uint32_t latency = TIM4->CNT; /* Ticks since this CONVST rising edge. */
    EXTI_D1->PR1 = BUSY_EXTI_LINE;

    volatile ad7606_status_t *status = &ad7606_status;
    if (status->busy_edges != 0U)
    {
        const uint32_t gap = entry - status->last_busy_cycles;
        const uint32_t period = ad7606_timing.period_cycles;
        if (gap > period + period / 2U)
        {
            status->missed_conversions += (gap + period / 2U) / period - 1U;
            status->last_gap_frame = head;
        }
    }
    status->last_busy_cycles = entry;
    ++status->busy_edges;
    if (latency < status->busy_latency_min_ticks)
        status->busy_latency_min_ticks = latency;
    if (latency > status->busy_latency_max_ticks)
        status->busy_latency_max_ticks = latency;

    if (reading)
    {
        ++status->read_overlaps;
        status->last_gap_frame = head + 1U; /* The read in progress is still published. */
    }
    else if (head - tail >= AD7606_RING_FRAMES)
    {
        ++status->ring_overruns;
        status->last_gap_frame = head;
    }
    else
    {
        reading = true;
        AD7606_CS_PORT->BSRR = (uint32_t)AD7606_CS_PIN << 16U; /* CS low: MSB of V1 on DOUTA. */
        DMA2->LIFCR = DMA_STREAM0_FLAGS;
        DMA2_Stream0->M0AR = (uint32_t)&ring[head & (AD7606_RING_FRAMES - 1U)][0];
        DMA2_Stream0->NDTR = AD7606_SLOTS;
        DMA2_Stream0->CR |= DMA_SxCR_EN;
        SPI1->CR1 = SPI_CR1_IDLE | SPI_CR1_SPE;
        SPI1->CR1 = SPI_CR1_IDLE | SPI_CR1_SPE | SPI_CR1_CSTART;
    }
    const uint32_t cycles = DWT->CYCCNT - entry;
    if (cycles > status->busy_isr_max_cycles)
        status->busy_isr_max_cycles = cycles;
}

/** Eight results are in memory: deselect the ADC and publish the frame. */
void DMA2_Stream0_IRQHandler(void)
{
    const uint32_t entry = DWT->CYCCNT;
    const uint32_t flags = DMA2->LISR & DMA_STREAM0_FLAGS;
    DMA2->LIFCR = flags;
    volatile ad7606_status_t *status = &ad7606_status;

    /* The SPI finished clocking before DMA moved the last result. */
    for (uint32_t wait = 0U; (SPI1->SR & SPI_SR_EOT) == 0U && wait < 200U; ++wait)
    {
    }
    AD7606_CS_PORT->BSRR = AD7606_CS_PIN;
    if ((SPI1->SR & SPI_SR_OVR) != 0U)
    {
        ++status->spi_overruns;
        status->last_gap_frame = head + 1U; /* This frame is suspect. */
    }
    SPI1->IFCR = SPI_ALL_FLAGS;
    SPI1->CR1 = SPI_CR1_IDLE; /* SPE = 0 ends the transfer; TSIZE reloads next time. */

    if ((flags & (DMA_LISR_TEIF0 | DMA_LISR_DMEIF0)) != 0U)
    {
        ++status->dma_errors;
        status->last_gap_frame = head;
        DMA2_Stream0->CR &= ~DMA_SxCR_EN;
    }
    else if ((flags & DMA_LISR_TCIF0) != 0U)
    {
        __DMB(); /* Frame data before the new head. */
        const uint32_t published = head + 1U;
        head = published;
        ++status->frames;
        status->last_frame_tick = uwTick;
        const uint32_t fill = published - tail;
        if (fill > status->max_fill_frames)
            status->max_fill_frames = fill;
    }
    reading = false;
    const uint32_t cycles = DWT->CYCCNT - entry;
    if (cycles > status->dma_isr_max_cycles)
        status->dma_isr_max_cycles = cycles;
}
