/** @file feature_uart.c
 * HAL USART2/DMA transport with two owned packets and no heap allocation.
 * USART2, PD5/PD6 and DMA1 Stream0 belong to this module, on CM7 only.
 */
#include "feature_uart.h"
#include "delivery_timing.h"
#include "latency_packet.h"
#include "benchmark.h"
#include "stm32h7xx_hal.h"
#include <string.h>

volatile bool feature_uart_ready;
volatile uint32_t feature_uart_packets_sent;
volatile uint32_t feature_uart_errors;
volatile uint32_t feature_uart_queue_full_count;
volatile bool feature_uart_timer_ready;
volatile float feature_uart_last_tx_us, feature_uart_max_tx_us;
static uint32_t tx_start_cycles;
static uint8_t rx_byte;

static UART_HandleTypeDef uart;
static DMA_HandleTypeDef dma;
/* DMA1 cannot read DTCM. The linker assigns these cache-line-aligned packets
 * to AXI SRAM. NOLOAD: send() writes every byte before DMA reads a slot.
 */
static uint8_t packets[2][COE_PAYLOAD_BYTES]
    __attribute__((section(".feature_uart_dma"), aligned(32)));
_Static_assert(COE_PAYLOAD_BYTES % 32U == 0U, "Own complete cache lines");
static uint32_t first_slot;
static uint32_t packet_count; /* Includes the packet currently being sent. */
static uint32_t start_tick;
static bool transmitting;
static volatile bool transmission_done;
static volatile bool transmission_failed;

/** Stop after an error rather than silently dropping or duplicating a window. */
static void stop_transmitter(void)
{
    feature_uart_ready = false;
    ++feature_uart_errors;
    (void)HAL_UART_AbortTransmit(&uart);
    packet_count = 0U;
    transmitting = false;
}

/** Configure the two peripheral clocks, TX pin, DMA request and interrupts. */
bool feature_uart_init(void)
{
    GPIO_InitTypeDef pin = {0};
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_USART2_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();
    /* USART2/3 use the existing APB1 clock selected at board startup.
     * Do not change their shared clock selector after COM1 initializes.
     */
    pin.Pin = GPIO_PIN_5 | GPIO_PIN_6;
    pin.Mode = GPIO_MODE_AF_PP;
    pin.Pull = GPIO_PULLUP; /* Keep the unused/unwired ACK input idle-high. */
    pin.Speed = GPIO_SPEED_FREQ_HIGH;
    pin.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOD, &pin);

    dma.Instance = DMA1_Stream0;
    dma.Init.Request = DMA_REQUEST_USART2_TX;
    dma.Init.Direction = DMA_MEMORY_TO_PERIPH;
    dma.Init.PeriphInc = DMA_PINC_DISABLE;
    dma.Init.MemInc = DMA_MINC_ENABLE;
    dma.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    dma.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    dma.Init.Mode = DMA_NORMAL;
    dma.Init.Priority = DMA_PRIORITY_LOW;
    dma.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&dma) != HAL_OK)
        return false;

    uart.Instance = USART2;
    uart.Init.BaudRate = COE_FEATURE_UART_BAUD;
    uart.Init.WordLength = UART_WORDLENGTH_8B;
    uart.Init.StopBits = UART_STOPBITS_1;
    uart.Init.Parity = UART_PARITY_NONE;
    uart.Init.Mode = UART_MODE_TX_RX; /* PD6 receives optional host-delivery ACKs. */
    uart.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    uart.Init.OverSampling = UART_OVERSAMPLING_16;
    uart.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    uart.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    uart.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    __HAL_LINKDMA(&uart, hdmatx, dma);
    if (HAL_UART_Init(&uart) != HAL_OK
        || HAL_UARTEx_DisableFifoMode(&uart) != HAL_OK)
        return false;

    /* Normal-mode DMA completion enables the UART TC interrupt. Both handlers
     * are required before HAL reports the last stop bit as transmitted.
     */
    HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 6U, 0U);
    HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);
    HAL_NVIC_SetPriority(USART2_IRQn, 6U, 0U);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
    feature_uart_timer_ready = benchmark_init();
    if (HAL_UART_Receive_IT(&uart, &rx_byte, 1U) != HAL_OK)
        return false;
    feature_uart_ready = true;
    return true;
}

/** Consume interrupt events in main, then start at most one queued packet. */
void feature_uart_process(void)
{
    if (!feature_uart_ready)
        return;
    if (transmission_failed)
    {
        stop_transmitter();
        return;
    }
    if (transmitting && transmission_done)
    {
        transmitting = false;
        transmission_done = false;
        first_slot = (first_slot + 1U) % 2U;
        --packet_count;
        ++feature_uart_packets_sent;
    }
    if (transmitting && (uint32_t)(HAL_GetTick() - start_tick) > 100U)
    {
        stop_transmitter();
        return;
    }
    if (!transmitting && packet_count != 0U)
    {
        uint8_t *packet = packets[first_slot];
        /* Cache is currently disabled. This remains safe if D-cache is enabled
         * later: each 160-byte slot owns five complete 32-byte cache lines.
         */
        if ((SCB->CCR & SCB_CCR_DC_Msk) != 0U)
            SCB_CleanDCache_by_Addr((uint32_t *)packet, COE_PAYLOAD_BYTES);
        __DSB();
        transmission_done = false;
        start_tick = HAL_GetTick();
        tx_start_cycles = benchmark_cycles_now();
        transmitting = true;
        if (HAL_UART_Transmit_DMA(&uart, packet, COE_PAYLOAD_BYTES) != HAL_OK)
            stop_transmitter();
    }
}

/** Copy into a free slot; DMA retains its own buffer until TC completes. */
bool feature_uart_send(const uint8_t packet[COE_PAYLOAD_BYTES])
{
    feature_uart_process();
    if (!feature_uart_ready || packet == NULL)
        return false;
    if (packet_count == 2U)
    {
        ++feature_uart_queue_full_count;
        return false;
    }
    const uint32_t slot = (first_slot + packet_count) % 2U;
    memcpy(packets[slot], packet, COE_PAYLOAD_BYTES);
    delivery_timing_bind_packet(packet);
    ++packet_count;
    feature_uart_process();
    return feature_uart_ready;
}

/** Forward the DMA event to HAL; no processing or packet copying in this ISR. */
void DMA1_Stream0_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&dma);
}

/** Forward USART2's final-byte completion event to HAL. */
void USART2_IRQHandler(void)
{
    HAL_UART_IRQHandler(&uart);
}

/** Record completion only for the USART2 handle owned by this module. */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *handle)
{
    if (handle == &uart) {
        if (feature_uart_timer_ready) {
            feature_uart_last_tx_us = benchmark_cycles_to_us(benchmark_cycles_now() - tx_start_cycles);
            if (feature_uart_last_tx_us > feature_uart_max_tx_us)
                feature_uart_max_tx_us = feature_uart_last_tx_us;
        }
        transmission_done = true;
    }
}

/** Defer USART2 error cleanup to main rather than aborting inside an ISR. */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *handle)
{
    if (handle == &uart)
        transmission_failed = true;
}

/** Arm the next ACK byte after bounded parsing; GPIO19->PD6 is optional. */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *handle)
{
    if (handle != &uart) return;
    delivery_timing_receive_byte(rx_byte);
    if (HAL_UART_Receive_IT(&uart, &rx_byte, 1U) != HAL_OK)
        transmission_failed = true;
}

/** Send the 12-byte timing marker before starting the measured DSP interval.
 * Marker setup is outside the reported interval. Feature vectors still use DMA.
 */
bool feature_uart_send_timing_request(uint32_t id)
{
    uint8_t bytes[LATENCY_REQUEST_BYTES];
    const uint32_t began = HAL_GetTick();
    while (feature_uart_ready && packet_count != 0U && (uint32_t)(HAL_GetTick() - began) < 100U)
        feature_uart_process();
    if (!feature_uart_ready || packet_count != 0U) return false;
    latency_request_encode(id, bytes);
    return HAL_UART_Transmit(&uart, bytes, sizeof(bytes), 20U) == HAL_OK;
}
