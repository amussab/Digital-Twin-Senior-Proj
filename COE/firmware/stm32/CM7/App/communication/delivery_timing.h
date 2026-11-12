/** @file delivery_timing.h
 * Optional B1 replay measurements; production COE1 vectors stay unchanged.
 * Wall-clock total ends at return ACK reception and has 1-ms resolution.
 */
#ifndef COE_DELIVERY_TIMING_H
#define COE_DELIVERY_TIMING_H
#include <stdbool.h>
#include <stdint.h>
typedef struct {
    uint32_t measurement_id, window_id, rpm, bearing_profile;
    float processing_ms, uart_tx_ms, esp_queue_ms, esp_network_ms, scope_roundtrip_ms;
    bool uart_tx_valid; /* Local transmission completed; independent of host delivery. */
    bool valid; /* Matched host-delivery ACK received. */
} DeliveryTimingResult;
extern volatile DeliveryTimingResult delivery_timing_results[12];
extern volatile uint32_t delivery_timing_completed, delivery_timing_timeouts;
extern volatile bool delivery_timing_batch_complete;
extern volatile float delivery_timing_max_ms_1750, delivery_timing_max_ms_3600;
/** Enable/reset one 12-window B1 timing run. Startup replay does not wait for Wi-Fi. */
void delivery_timing_start_batch(void);
/** Send a request while idle, then start timing immediately before DSP. */
void delivery_timing_begin(void);
/** Bind the next transmitted COE1 vector's metadata to the active timing ID. */
void delivery_timing_bind_packet(const uint8_t *bytes);
/** Publish the elapsed DSP time, then await host acceptance before reference checks. */
void delivery_timing_finish(float processing_ms);
/** Consume RX interrupt bytes. A matched ACK records its arrival tick, not polling time. */
void delivery_timing_receive_byte(uint8_t byte);
#endif
