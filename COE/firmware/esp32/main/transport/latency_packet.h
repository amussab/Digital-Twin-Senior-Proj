/** @file latency_packet.h
 * Small CRC-protected timing messages alongside the unchanged COE1 payload.
 * COT1 names the next feature vector; COA1 confirms HTTP host acceptance.
 */
#ifndef COE_LATENCY_PACKET_H
#define COE_LATENCY_PACKET_H
#include "feature_packet.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define LATENCY_REQUEST_BYTES 12U
#define LATENCY_ACK_BYTES 36U
typedef struct {
    uint32_t measurement_id, window_id, rpm, bearing_profile;
    uint32_t esp_total_us, esp_queue_us, esp_network_us;
} LatencyAck;
typedef struct { uint8_t bytes[LATENCY_ACK_BYTES]; size_t count; } LatencyStream;
/** Encode a nonzero request ID, in little-endian order with CRC-32/ISO-HDLC. */
void latency_request_encode(uint32_t id, uint8_t bytes[LATENCY_REQUEST_BYTES]);
/** Encode the matching host acceptance and ESP-local elapsed times. */
void latency_ack_encode(const LatencyAck *ack, uint8_t bytes[LATENCY_ACK_BYTES]);
/** Consume arbitrary fragments/noise; leave output unchanged on rejection. */
bool latency_request_push(LatencyStream *stream, uint8_t byte, uint32_t *id);
/** Reject corrupted, malformed or internally inconsistent acknowledgments. */
bool latency_ack_push(LatencyStream *stream, uint8_t byte, LatencyAck *ack);
/** Two frame types on the same UART; control bytes never enter a COE1 body. */
typedef struct {
    FeaturePacketStream features;
    LatencyStream requests;
    unsigned request_bytes_left;
    uint32_t measurement_id;
} FeatureTimingStream;
/** Decode unchanged features plus their optional preceding measurement ID. */
bool latency_feature_push(FeatureTimingStream *stream, uint8_t byte,
                          FeaturePacket *packet, uint32_t *measurement_id);
/** Drop all partial data after hardware loss, retaining the feature reject count. */
void latency_feature_reset(FeatureTimingStream *stream);
#endif
