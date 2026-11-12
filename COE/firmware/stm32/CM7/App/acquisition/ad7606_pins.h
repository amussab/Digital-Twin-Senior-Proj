/**
 * @file ad7606_pins.h
 * AD7606 interface pins on the NUCLEO-H755ZI-Q CN7 connector.
 */
#ifndef COE_AD7606_PINS_H
#define COE_AD7606_PINS_H

#include "stm32h7xx_hal.h"

/* STM32 outputs to the ADC. */
#define AD7606_RST_PORT GPIOB /* CN7 pin 2. */
#define AD7606_RST_PIN GPIO_PIN_8
#define AD7606_SCLK_PORT GPIOA /* CN7 pin 10, RD/SCLK, SPI1_SCK AF5. */
#define AD7606_SCLK_PIN GPIO_PIN_5
#define AD7606_CS_PORT GPIOD /* CN7 pin 16. */
#define AD7606_CS_PIN GPIO_PIN_14
#define AD7606_CONVST_PORT GPIOD /* CN7 pin 18, CONVST A (and B if joined), TIM4_CH4 AF2. */
#define AD7606_CONVST_PIN GPIO_PIN_15
/* CONVST B on its own wire: CN10 pin 16 (D0), TIM4_CH2 AF2. Driven by the
 * same timer counter as CONVST A, so both edges coincide; leaving it
 * unconnected is fine when CA and CB are joined on CN7 pin 18.
 */
#define AD7606_CONVSTB_PORT GPIOB
#define AD7606_CONVSTB_PIN GPIO_PIN_7

/* ADC configuration inputs, held low by the STM32 (instead of wires to GND):
 * OS0-OS2 low = no oversampling, RANGE low = +/-5 V, D15/BYTE SEL low =
 * serial read. Wiring them to GND on the ADC board instead is equivalent.
 */
#define AD7606_OS0_PORT GPIOC /* CN7 pin 1. */
#define AD7606_OS0_PIN GPIO_PIN_6
#define AD7606_OS1_PORT GPIOB /* CN7 pin 3. */
#define AD7606_OS1_PIN GPIO_PIN_15
#define AD7606_OS2_PORT GPIOB /* CN7 pin 7. */
#define AD7606_OS2_PIN GPIO_PIN_12
#define AD7606_RANGE_PORT GPIOC /* CN7 pin 11. */
#define AD7606_RANGE_PIN GPIO_PIN_7
#define AD7606_BYTESEL_PORT GPIOG /* CN7 pin 20, ADC pin D15. */
#define AD7606_BYTESEL_PIN GPIO_PIN_9

/* ADC outputs at about VDRIVE (4 V): 5-V-tolerant digital inputs, no pulls.
 * Never put these pins in analog mode while the ADC may be powered.
 */
#define AD7606_BUSY_PORT GPIOB /* CN7 pin 4, EXTI9. */
#define AD7606_BUSY_PIN GPIO_PIN_9
#define AD7606_DOUTA_PORT GPIOA /* CN7 pin 12, D7/DOUTA, SPI1_MISO AF5. */
#define AD7606_DOUTA_PIN GPIO_PIN_6

/**
 * Drive RST, SCLK, CS, CONVST A/B and the configuration pins (OS0-2, RANGE,
 * D15) low, and make BUSY/DOUTA digital inputs.
 * Call as early as possible after reset, in every firmware mode: the ADC's
 * digital pins must not be driven high while it may be unpowered.
 */
void ad7606_pins_hold_safe(void);

#endif
