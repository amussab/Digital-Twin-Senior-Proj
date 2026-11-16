/** @file feature_packet.h
 * Portable decoder for the STM32's COE1 feature packets.
 * Contains no ESP-IDF or STM32 dependencies; also verified by MCU test code.
 * Schema 1 (160 bytes): header + 32 features + CRC.
 * Schema 2 (204 bytes, LIVE since 16 Nov 2026): schema 1's header and features, then
 * the shaft block (Keyphasor speed, PROX 1 and PROX 3 values), then CRC.
 * Layout: coe_firmware/CM7/App/payload/node_packet.h.
 */
#ifndef COE_RECEIVER_PACKET_H
#define COE_RECEIVER_PACKET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FEATURE_PACKET_BYTES 160U        /* schema 1 */
#define FEATURE_PACKET_SHAFT_BYTES 204U  /* schema 2 */
#define FEATURE_PACKET_MAX_BYTES FEATURE_PACKET_SHAFT_BYTES
#define FEATURE_PACKET_VALUES 32U
#define FEATURE_PACKET_PROBES 2U          /* PROX 1, PROX 3 */
#define FEATURE_PACKET_MIN_RPM 60U
#define FEATURE_PACKET_MAX_RPM 4000U
/* shaft_flags bits (schema 2) */
#define FEATURE_SHAFT_SPEED_MEASURED 0x01U
#define FEATURE_SHAFT_PROX1_VALID 0x02U
#define FEATURE_SHAFT_PROX3_VALID 0x04U
#define FEATURE_SHAFT_PROX1_X1_VALID 0x08U
#define FEATURE_SHAFT_PROX3_X1_VALID 0x10U

/** One independently decoded window; values are in channel-major order. */
typedef struct
{
    uint32_t window_id;
    uint32_t rpm;
    uint32_t bearing_profile;
    uint32_t valid_channels;
    float features[FEATURE_PACKET_VALUES];
    /* Schema 2 only (all zero for schema 1). */
    uint32_t schema;              /* 1 or 2 */
    float measured_rpm;           /* Keyphasor, mean of the window's revolutions */
    uint32_t revolutions;
    uint32_t shaft_flags;
    float probes[FEATURE_PACKET_PROBES][4]; /* gap_um, pkpk_um, x1_um_pp, x1_phase_deg */
} FeaturePacket;

/** Bounded storage for a fragmented byte stream; initialize with {0}. */
typedef struct
{
    uint8_t bytes[FEATURE_PACKET_MAX_BYTES];
    size_t count;
    uint32_t rejected_packets;
} FeaturePacketStream;

/** Check magic, schema, metadata, IEEE Float32 values and CRC; leave output
 * untouched on rejection. Invalid channels must contain eight zero values.
 * The schema field (bytes 4..5) gives the length: 160 bytes for schema 1,
 * 204 for schema 2; bytes must hold that many.
 */
bool feature_packet_decode(const uint8_t *bytes, FeaturePacket *output);

/** Packet length for the schema in a header's bytes 4..7, or 0 if unknown. */
size_t feature_packet_length(const uint8_t header[8]);

/** Consume one byte; return true and set output only for a valid full packet.
 * Searches for COE1, supports arbitrary UART chunks, and recovers after a
 * corrupt/truncated frame. Unknown schemas and CRC-valid bad metadata are rejected.
 */
bool feature_packet_push(FeaturePacketStream *stream, uint8_t byte, FeaturePacket *output);

/** Discard a partial frame after UART data loss, preserving the reject count. */
void feature_packet_reset(FeaturePacketStream *stream);

#endif
