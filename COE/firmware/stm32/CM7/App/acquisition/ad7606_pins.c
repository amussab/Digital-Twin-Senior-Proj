/**
 * @file ad7606_pins.c
 * Safe idle state for the AD7606 interface pins.
 */
#include "ad7606_pins.h"

/** Write the output latch low before switching a pin to output mode. */
static void hold_output_low(GPIO_TypeDef *port, uint32_t pin, uint32_t speed)
{
    GPIO_InitTypeDef init = {0};
    HAL_GPIO_WritePin(port, (uint16_t)pin, GPIO_PIN_RESET);
    init.Pin = pin;
    init.Mode = GPIO_MODE_OUTPUT_PP;
    init.Pull = GPIO_NOPULL;
    init.Speed = speed;
    HAL_GPIO_Init(port, &init);
}

void ad7606_pins_hold_safe(void)
{
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    hold_output_low(AD7606_OS0_PORT, AD7606_OS0_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_OS1_PORT, AD7606_OS1_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_OS2_PORT, AD7606_OS2_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_RANGE_PORT, AD7606_RANGE_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_BYTESEL_PORT, AD7606_BYTESEL_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_RST_PORT, AD7606_RST_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_SCLK_PORT, AD7606_SCLK_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_CS_PORT, AD7606_CS_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_CONVST_PORT, AD7606_CONVST_PIN, GPIO_SPEED_FREQ_LOW);
    hold_output_low(AD7606_CONVSTB_PORT, AD7606_CONVSTB_PIN, GPIO_SPEED_FREQ_LOW);

    GPIO_InitTypeDef input = {0};
    input.Mode = GPIO_MODE_INPUT;
    input.Pull = GPIO_NOPULL;
    input.Pin = AD7606_BUSY_PIN;
    HAL_GPIO_Init(AD7606_BUSY_PORT, &input);
    input.Pin = AD7606_DOUTA_PIN;
    HAL_GPIO_Init(AD7606_DOUTA_PORT, &input);
}
