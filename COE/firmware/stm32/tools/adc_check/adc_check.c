/**
 * @file adc_check.c
 * Standalone AD7606 "is it alive?" test for the NUCLEO-H755ZI-Q (CM7 only).
 * Not part of the firmware: no HAL, no DSP, no SPI. Once per second it
 * pulses RESET, then sends 10 CONVST pulses and checks that BUSY answers
 * each one with a high pulse of about 4 us. Result on the LEDs and on the
 * ST-LINK COM port (115200 8N1).
 *
 * Wires (same pins as the main firmware):
 *   PB8 CN7-2  -> ADC RST        PD15 CN7-18  -> ADC CA (and CB if joined)
 *   PB9 CN7-4  <- ADC BUSY       PB7  CN10-16 -> ADC CB, if wired separately
 *   GND CN7-8  -- ADC GND
 * PB7 (and PA4, CN7-17) are pulsed together with PD15, so CB may be on any.
 * ADC board: 5V and GND from the 5-V converter. D15, RANGE, OS0-OS2 either
 * to GND, or to the STM32 pins that hold them low: OS0 CN7-1 (PC6),
 * OS1 CN7-3 (PB15), OS2 CN7-7 (PB12), RANGE CN7-11 (PC7), D15 CN7-20 (PG9).
 * CS (PD14) and RD/SCLK (PA5) are held low in case they are still connected.
 *
 * LEDs: green blinking = ADC converts; red = BUSY stuck high;
 *       yellow = BUSY never rises (ADC unpowered, or CONVST/BUSY wiring).
 */
#include "stm32h7xx.h"
#include <stdbool.h>
#include <stdint.h>

#define CPU_HZ 64000000U /* Reset clock: HSI 64 MHz, no PLL. */
#define CYCLES_PER_US (CPU_HZ / 1000000U)
#define PULSES 10U

/* CONVST A on PD15, CONVST B on PB7 (or PA4): set and cleared back to back. */
static void convst(bool high)
{
    GPIOD->BSRR = high ? (1UL << 15) : (1UL << 15) << 16;
    GPIOB->BSRR = high ? (1UL << 7) : (1UL << 7) << 16;
    GPIOA->BSRR = high ? (1UL << 4) : (1UL << 4) << 16;
}

static void wait_us(uint32_t us)
{
    const uint32_t start = DWT->CYCCNT;
    while (DWT->CYCCNT - start < us * CYCLES_PER_US)
    {
    }
}

static bool busy_high(void)
{
    return (GPIOB->IDR & (1UL << 9)) != 0U;
}

/** Set one pin's 2-bit field in a GPIO register. */
static void set_field(volatile uint32_t *reg, uint32_t pin, uint32_t value)
{
    *reg = (*reg & ~(3UL << (2U * pin))) | (value << (2U * pin));
}

static void uart_put(char c)
{
    while ((USART3->ISR & USART_ISR_TXE_TXFNF) == 0U)
    {
    }
    USART3->TDR = (uint8_t)c;
}

static void uart_text(const char *text)
{
    while (*text != '\0')
        uart_put(*text++);
}

static void uart_number(uint32_t value)
{
    char digits[11];
    int count = 0;
    do
    {
        digits[count++] = (char)('0' + value % 10U);
        value /= 10U;
    } while (value != 0U);
    while (count > 0)
        uart_put(digits[--count]);
}

static void setup(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; /* Cycle counter for timing. */
    DWT->LAR = 0xC5ACCE55UL;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOAEN | RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOCEN
                    | RCC_AHB4ENR_GPIODEN | RCC_AHB4ENR_GPIOEEN | RCC_AHB4ENR_GPIOGEN;
    (void)RCC->AHB4ENR;

    /* ADC configuration pins low: PC6 OS0, PB15 OS1, PB12 OS2, PC7 RANGE, PG9 D15. */
    GPIOC->BSRR = ((1UL << 6) | (1UL << 7)) << 16;
    GPIOB->BSRR = ((1UL << 15) | (1UL << 12)) << 16;
    GPIOG->BSRR = (1UL << 9) << 16;
    set_field(&GPIOC->MODER, 6U, 1U);
    set_field(&GPIOC->MODER, 7U, 1U);
    set_field(&GPIOB->MODER, 15U, 1U);
    set_field(&GPIOB->MODER, 12U, 1U);
    set_field(&GPIOG->MODER, 9U, 1U);

    /* Outputs to the ADC, all low: PB8 RST, PD15 CONVST, PD14 CS, PA5 SCLK. */
    GPIOB->BSRR = ((1UL << 8) | (1UL << 7)) << 16;
    set_field(&GPIOB->MODER, 7U, 1U); /* PB7: CONVST B, optional wire. */
    GPIOD->BSRR = ((1UL << 15) | (1UL << 14)) << 16;
    GPIOA->BSRR = ((1UL << 5) | (1UL << 4)) << 16;
    set_field(&GPIOA->MODER, 4U, 1U); /* PA4: CONVST B, optional wire. */
    set_field(&GPIOB->MODER, 8U, 1U);
    set_field(&GPIOD->MODER, 15U, 1U);
    set_field(&GPIOD->MODER, 14U, 1U);
    set_field(&GPIOA->MODER, 5U, 1U);
    /* Inputs from the ADC, digital, no pulls: PB9 BUSY, PA6 DOUTA. */
    set_field(&GPIOB->MODER, 9U, 0U);
    set_field(&GPIOB->PUPDR, 9U, 0U);
    set_field(&GPIOA->MODER, 6U, 0U);
    set_field(&GPIOA->PUPDR, 6U, 0U);

    /* LEDs: PB0 green, PE1 yellow, PB14 red. */
    set_field(&GPIOB->MODER, 0U, 1U);
    set_field(&GPIOE->MODER, 1U, 1U);
    set_field(&GPIOB->MODER, 14U, 1U);

    /* USART3 TX on PD8 (AF7) = ST-LINK virtual COM port, 115200 8N1. */
    RCC->APB1LENR |= RCC_APB1LENR_USART3EN;
    (void)RCC->APB1LENR;
    set_field(&GPIOD->MODER, 8U, 2U);
    GPIOD->AFR[1] = (GPIOD->AFR[1] & ~(0xFUL << 0)) | (7UL << 0);
    USART3->BRR = (CPU_HZ + 115200U / 2U) / 115200U;
    USART3->CR1 = USART_CR1_TE | USART_CR1_UE;
}

static void leds(bool green, bool yellow, bool red)
{
    GPIOB->BSRR = green ? (1UL << 0) : (1UL << 16);
    GPIOE->BSRR = yellow ? (1UL << 1) : (1UL << 17);
    GPIOB->BSRR = red ? (1UL << 14) : (1UL << 30);
}

int main(void)
{
    setup();
    uart_text("\r\n=== AD7606 standalone check: RESET, 10 x CONVST, watch BUSY ===\r\n");
    bool blink = false;
    for (uint32_t round = 1U;; ++round)
    {
        const bool idle_high = busy_high();

        GPIOB->BSRR = 1UL << 8; /* RESET pulse, 1 us (minimum 50 ns). */
        wait_us(1U);
        GPIOB->BSRR = (1UL << 8) << 16;
        wait_us(1000U);
        const bool after_reset_high = busy_high();

        uint32_t answered = 0U, min_ns = UINT32_MAX, max_ns = 0U;
        for (uint32_t pulse = 0U; pulse < PULSES; ++pulse)
        {
            __disable_irq();
            convst(true); /* Conversion starts. */
            const uint32_t start = DWT->CYCCNT;
            uint32_t rise = 0U, fall = 0U;
            bool rose = false;
            while (DWT->CYCCNT - start < 50U * CYCLES_PER_US)
            {
                const uint32_t now = DWT->CYCCNT - start;
                if (now >= CYCLES_PER_US)
                    convst(false); /* Low again after 1 us. */
                const bool high = busy_high();
                if (!rose && high)
                {
                    rose = true;
                    rise = now;
                }
                else if (rose && !high)
                {
                    fall = now;
                    break;
                }
            }
            convst(false);
            __enable_irq();
            if (rose && fall != 0U)
            {
                ++answered;
                const uint32_t ns = (fall - rise) * 1000U / CYCLES_PER_US;
                if (ns < min_ns)
                    min_ns = ns;
                if (ns > max_ns)
                    max_ns = ns;
            }
            wait_us(100U);
        }

        uart_text("[");
        uart_number(round);
        uart_text("] BUSY idle ");
        uart_text(idle_high ? "HIGH" : "low");
        uart_text(", after RESET ");
        uart_text(after_reset_high ? "HIGH" : "low");
        uart_text(", answered ");
        uart_number(answered);
        uart_text("/10 CONVST pulses");
        if (answered != 0U)
        {
            uart_text(", BUSY high ");
            uart_number(min_ns);
            uart_text("-");
            uart_number(max_ns);
            uart_text(" ns (expect about 4000)");
        }
        if (answered == PULSES)
        {
            uart_text(" -> ADC CONVERTS OK\r\n");
            blink = !blink;
            leds(blink, false, false);
        }
        else if (answered != 0U)
        {
            uart_text(" -> FAIL: only some pulses answered (loose CONVST/BUSY wire?)\r\n");
            leds(false, true, true);
        }
        else if (after_reset_high)
        {
            uart_text(" -> FAIL: BUSY stuck high\r\n");
            leds(false, false, true);
        }
        else
        {
            uart_text(" -> FAIL: no BUSY pulse (ADC power? CONVST/BUSY wires?)\r\n");
            leds(false, true, false);
        }
        wait_us(1000000U);
    }
}
