/* Host-only HAL test double. The production sender source is compiled unchanged.
 * This models calls/completion events, not USART/DMA electrical behavior.
 */
#ifndef HOST_STM32_HAL_H
#define HOST_STM32_HAL_H
#include <stdint.h>
#include <stddef.h>
typedef enum { HAL_OK, HAL_ERROR, HAL_BUSY } HAL_StatusTypeDef;
typedef struct { uint32_t Pin, Mode, Pull, Speed, Alternate; } GPIO_InitTypeDef;
typedef struct { uint32_t Request, Direction, PeriphInc, MemInc, PeriphDataAlignment,
    MemDataAlignment, Mode, Priority, FIFOMode; } DMA_InitTypeDef;
typedef struct { void *Instance; DMA_InitTypeDef Init; void *Parent; } DMA_HandleTypeDef;
typedef struct { uint32_t BaudRate, WordLength, StopBits, Parity, Mode, HwFlowCtl,
    OverSampling, OneBitSampling, ClockPrescaler; } UART_InitTypeDef;
typedef struct { void *Instance; UART_InitTypeDef Init;
    struct { uint32_t AdvFeatureInit; } AdvancedInit; DMA_HandleTypeDef *hdmatx; } UART_HandleTypeDef;
typedef struct { uint32_t CCR; } HostSCB;
extern HostSCB fake_scb;
extern UART_HandleTypeDef *fake_uart;
extern uint8_t *fake_buffer;
extern uint32_t fake_tick, fake_start_calls, fake_abort_calls, fake_cache_calls;
extern uint32_t fake_dma_irqs, fake_uart_irqs;
extern HAL_StatusTypeDef fake_start_status, fake_init_status;
#define SCB (&fake_scb)
#define SCB_CCR_DC_Msk 4U
#define GPIOD ((void *)4)
#define USART2 ((void *)2)
#define DMA1_Stream0 ((void *)1)
#define DMA1_Stream0_IRQn 11
#define USART2_IRQn 38
#define GPIO_PIN_5 32U
#define GPIO_PIN_6 64U
#define GPIO_PULLUP 1U
#define GPIO_MODE_AF_PP 2U
#define GPIO_NOPULL 0U
#define GPIO_SPEED_FREQ_HIGH 3U
#define GPIO_AF7_USART2 7U
#define DMA_REQUEST_USART2_TX 44U
#define DMA_MEMORY_TO_PERIPH 1U
#define DMA_PINC_DISABLE 0U
#define DMA_MINC_ENABLE 1U
#define DMA_PDATAALIGN_BYTE 0U
#define DMA_MDATAALIGN_BYTE 0U
#define DMA_NORMAL 0U
#define DMA_PRIORITY_LOW 0U
#define DMA_FIFOMODE_DISABLE 0U
#define UART_WORDLENGTH_8B 8U
#define UART_STOPBITS_1 1U
#define UART_PARITY_NONE 0U
#define UART_MODE_TX 1U
#define UART_MODE_TX_RX 3U
#define UART_HWCONTROL_NONE 0U
#define UART_OVERSAMPLING_16 16U
#define UART_ONE_BIT_SAMPLE_DISABLE 0U
#define UART_PRESCALER_DIV1 1U
#define UART_ADVFEATURE_NO_INIT 0U
#define __HAL_RCC_GPIOD_CLK_ENABLE() ((void)0)
#define __HAL_RCC_USART2_CLK_ENABLE() ((void)0)
#define __HAL_RCC_DMA1_CLK_ENABLE() ((void)0)
#define __HAL_LINKDMA(h, member, d) ((h)->member = &(d), (d).Parent = (h))
#define __DSB() ((void)0)
static inline void HAL_GPIO_Init(void *port, GPIO_InitTypeDef *pin) { (void)port; (void)pin; }
static inline HAL_StatusTypeDef HAL_DMA_Init(DMA_HandleTypeDef *dma) { (void)dma; return fake_init_status; }
static inline HAL_StatusTypeDef HAL_UART_Init(UART_HandleTypeDef *uart) { (void)uart; return fake_init_status; }
static inline HAL_StatusTypeDef HAL_UARTEx_DisableFifoMode(UART_HandleTypeDef *uart) { (void)uart; return HAL_OK; }
static inline void HAL_NVIC_SetPriority(int irq, unsigned priority, unsigned sub) { (void)irq; (void)priority; (void)sub; }
static inline void HAL_NVIC_EnableIRQ(int irq) { (void)irq; }
static inline uint32_t HAL_GetTick(void) { return fake_tick; }
static inline HAL_StatusTypeDef HAL_UART_Transmit_DMA(UART_HandleTypeDef *uart, uint8_t *bytes, unsigned size)
{
    (void)size;
    fake_uart = uart;
    fake_buffer = bytes;
    ++fake_start_calls;
    return fake_start_status;
}
static inline HAL_StatusTypeDef HAL_UART_AbortTransmit(UART_HandleTypeDef *uart)
{ (void)uart; ++fake_abort_calls; return HAL_OK; }
static inline void SCB_CleanDCache_by_Addr(uint32_t *bytes, unsigned size)
{ (void)bytes; (void)size; ++fake_cache_calls; }
static inline void HAL_DMA_IRQHandler(DMA_HandleTypeDef *dma) { (void)dma; ++fake_dma_irqs; }
static inline void HAL_UART_IRQHandler(UART_HandleTypeDef *uart) { (void)uart; ++fake_uart_irqs; }
static inline HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *handle, uint8_t *byte, uint16_t n)
{ (void)handle; (void)byte; (void)n; return HAL_OK; }
static inline HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *handle, const uint8_t *bytes, uint16_t n, uint32_t timeout)
{ (void)handle; (void)bytes; (void)n; (void)timeout; return HAL_OK; }
static inline void HAL_Delay(uint32_t ms) { fake_tick += ms; }
#endif
