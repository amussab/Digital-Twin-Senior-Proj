/** @file feature_uart.h
 * Send one completed feature packet over USART2 using HAL transmit DMA.
 * Call all APIs from the CM7 main context; interrupt handlers only signal events.
 */
#ifndef COE_FEATURE_UART_H
#define COE_FEATURE_UART_H

#include "feature_payload.h"
#include <stddef.h>
#define COE_FEATURE_UART_BAUD 115200U
/* One queue slot: the largest packet (schema 2, 204 bytes) in whole 32-byte cache lines. */
#define FEATURE_UART_SLOT_BYTES 224U

extern volatile bool feature_uart_ready;
extern volatile uint32_t feature_uart_packets_sent;
extern volatile uint32_t feature_uart_errors;
extern volatile uint32_t feature_uart_queue_full_count;
extern volatile bool feature_uart_timer_ready;
extern volatile float feature_uart_last_tx_us, feature_uart_max_tx_us;
/** Receive errors on PD6 (noise, overrun). Each loses at most one incoming
 * message; reception restarts and feature packets keep going. */
extern volatile uint32_t feature_uart_rx_errors;
/** Last valid speed command from the ESP32 (COR1 frame), 0 until one arrives.
 * Not used by LIVE since 16 Nov 2026 (the speed is measured by the Keyphasor);
 * the current ESP32 firmware no longer sends it. */
extern volatile uint32_t feature_uart_requested_rpm;
/** Optional timing marker; call from main while no other producer is active. */
bool feature_uart_send_timing_request(uint32_t id);

/** Configure PD5 TX/DMA1 Stream0 and optional PD6 ACK input at 115200 baud. */
bool feature_uart_init(void);

/** Copy one packet (1..FEATURE_UART_SLOT_BYTES bytes: 160 for schema 1, 204 for
 * schema 2) into owned storage and start DMA when idle.
 * Two slots hold the active packet and one pending packet. Returns false on
 * unavailable hardware, a bad length or a full queue; never overwrites an active packet.
 * Success means accepted for transmission, not received by the ESP32.
 */
bool feature_uart_send(const uint8_t *packet, size_t length);

/** Handle completion/errors and start the next packet without waiting.
 * Call regularly from main and before enqueueing. A HAL error or 100-ms
 * transmit timeout stops this transmitter until reset; no automatic retries.
 */
void feature_uart_process(void);

#endif
