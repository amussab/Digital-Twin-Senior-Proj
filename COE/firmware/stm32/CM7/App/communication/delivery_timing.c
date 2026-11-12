/** @file delivery_timing.c
 * Measure processing-start to HTTP-acceptance ACK on one STM32 clock.
 * No subtraction between clocks, no sum of maxima, no success on timeout.
 */
#include "delivery_timing.h"
#include "feature_uart.h"
#include "latency_packet.h"
#include "stm32h7xx_hal.h"
#include <string.h>
volatile DeliveryTimingResult delivery_timing_results[12];
volatile uint32_t delivery_timing_completed, delivery_timing_timeouts;
volatile bool delivery_timing_batch_complete;
volatile float delivery_timing_max_ms_1750, delivery_timing_max_ms_3600;
static bool enabled;
static volatile bool active, acknowledged;
static uint32_t next_id, start_tick, slot, sent_before;
static volatile uint32_t current_id;
static volatile uint32_t end_tick;
static volatile uint32_t expected_window, expected_rpm, expected_profile;
static LatencyStream ack_stream;
static volatile LatencyAck received_ack;

/** Read the existing COE1 metadata without assuming alignment. */
static uint32_t read_u32(const uint8_t *p)
{ return (uint32_t)p[0] | (uint32_t)p[1] << 8U | (uint32_t)p[2] << 16U | (uint32_t)p[3] << 24U; }
void delivery_timing_start_batch(void)
{
    active = false;
    enabled = true;
    slot = delivery_timing_completed = delivery_timing_timeouts = 0U;
    delivery_timing_batch_complete = false;
    delivery_timing_max_ms_1750 = delivery_timing_max_ms_3600 = 0.0f;
    for (unsigned i = 0U; i < 12U; ++i) delivery_timing_results[i] = (DeliveryTimingResult){0};
}
void delivery_timing_begin(void)
{
    if (!enabled || slot >= 12U) return;
    active = acknowledged = false;
    expected_window = expected_rpm = expected_profile = 0U;
    if (next_id == UINT32_MAX || !feature_uart_send_timing_request(++next_id)) {
        ++delivery_timing_timeouts;
        enabled = false;
        return;
    }
    current_id = next_id;
    delivery_timing_results[slot].measurement_id = current_id;
    sent_before = feature_uart_packets_sent;
    start_tick = HAL_GetTick();
    active = true;
}
void delivery_timing_bind_packet(const uint8_t *bytes)
{
    if (!active) return;
    expected_window = read_u32(bytes + 8U);
    expected_rpm = read_u32(bytes + 12U);
    expected_profile = read_u32(bytes + 16U);
    delivery_timing_results[slot].window_id = expected_window;
    delivery_timing_results[slot].rpm = expected_rpm;
    delivery_timing_results[slot].bearing_profile = expected_profile;
}
void delivery_timing_receive_byte(uint8_t byte)
{
    LatencyAck ack;
    if (!latency_ack_push(&ack_stream, byte, &ack) || !active || acknowledged
        || ack.measurement_id != current_id || ack.window_id != expected_window
        || ack.rpm != expected_rpm || ack.bearing_profile != expected_profile) return;
    received_ack = ack;
    end_tick = HAL_GetTick();
    acknowledged = true; /* Fields are published before this flag. */
}
void delivery_timing_finish(float processing_ms)
{
    if (!active) return;
    delivery_timing_results[slot].processing_ms = processing_ms;
    /* SysTick measures wall time across waits; DWT would wrap during long network retries. */
    while (!acknowledged && feature_uart_ready && (uint32_t)(HAL_GetTick() - start_tick) < 12000U) {
        feature_uart_process();
        HAL_Delay(1U);
    }
    active = false;
    feature_uart_process();
    /* Local TX timing is useful even when the host cannot be reached. */
    if (feature_uart_timer_ready && feature_uart_packets_sent != sent_before) {
        delivery_timing_results[slot].uart_tx_ms = feature_uart_last_tx_us / 1000.0f;
        delivery_timing_results[slot].uart_tx_valid = true;
    }
    if (!acknowledged) {
        ++delivery_timing_timeouts;
        enabled = false; /* Avoid twelve long waits with an absent host/return wire. */
        ++slot;
        return;
    }
    volatile DeliveryTimingResult *result = &delivery_timing_results[slot++];
    result->esp_queue_ms = (float)received_ack.esp_queue_us / 1000.0f;
    result->esp_network_ms = (float)received_ack.esp_network_us / 1000.0f;
    result->scope_roundtrip_ms = (float)(uint32_t)(end_tick - start_tick);
    result->valid = true;
    ++delivery_timing_completed;
    volatile float *maximum = result->rpm == 1750U ? &delivery_timing_max_ms_1750 : &delivery_timing_max_ms_3600;
    if (result->scope_roundtrip_ms > *maximum) *maximum = result->scope_roundtrip_ms;
    delivery_timing_batch_complete = delivery_timing_completed == 12U;
}
