/** @file feature_uart.h
 * Send one completed feature packet over USART2 using HAL transmit DMA.
 * Call all APIs from the CM7 main context; interrupt handlers only signal events.
 */
#ifndef COE_FEATURE_UART_H
#define COE_FEATURE_UART_H

#include "feature_payload.h"
#define COE_FEATURE_UART_BAUD 115200U

extern volatile bool feature_uart_ready;
extern volatile uint32_t feature_uart_packets_sent;
extern volatile uint32_t feature_uart_errors;
extern volatile uint32_t feature_uart_queue_full_count;
extern volatile bool feature_uart_timer_ready;
extern volatile float feature_uart_last_tx_us, feature_uart_max_tx_us;
/** Optional timing marker; call from main while no other producer is active. */
bool feature_uart_send_timing_request(uint32_t id);

/** Configure PD5 TX/DMA1 Stream0 and optional PD6 ACK input at 115200 baud. */
bool feature_uart_init(void);

/** Copy one 160-byte packet into owned storage and start DMA when idle.
 * Two slots hold the active packet and one pending packet. Returns false on
 * unavailable hardware or a full queue; never overwrites an active packet.
 * Success means accepted for transmission, not received by the ESP32.
 */
bool feature_uart_send(const uint8_t packet[COE_PAYLOAD_BYTES]);

/** Handle completion/errors and start the next packet without waiting.
 * Call regularly from main and before enqueueing. A HAL error or 100-ms
 * transmit timeout stops this transmitter until reset; no automatic retries.
 */
void feature_uart_process(void);

#endif
