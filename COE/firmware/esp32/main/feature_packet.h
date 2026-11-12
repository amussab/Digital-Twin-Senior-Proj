/** @file feature_packet.h
 * Portable decoder for the STM32's fixed COE1 feature packets.
 * Contains no ESP-IDF or STM32 dependencies; also verified by MCU test code.
 */
#ifndef COE_RECEIVER_PACKET_H
#define COE_RECEIVER_PACKET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FEATURE_PACKET_BYTES 160U
#define FEATURE_PACKET_VALUES 32U

/** One independently decoded window; values are in channel-major order. */
typedef struct
{
    uint32_t window_id;
    uint32_t rpm;
    uint32_t bearing_profile;
    uint32_t valid_channels;
    float features[FEATURE_PACKET_VALUES];
} FeaturePacket;

/** Bounded storage for a fragmented byte stream; initialize with {0}. */
typedef struct
{
    uint8_t bytes[FEATURE_PACKET_BYTES];
    size_t count;
    uint32_t rejected_packets;
} FeaturePacketStream;

/** Check magic, schema, metadata, IEEE Float32 values and CRC; leave output
 * untouched on rejection. Invalid channels must contain eight zero values.
 */
bool feature_packet_decode(const uint8_t bytes[FEATURE_PACKET_BYTES], FeaturePacket *output);

/** Consume one byte; return true and set output only for a valid full packet.
 * Searches for COE1, supports arbitrary UART chunks, and recovers after a
 * corrupt/truncated frame. CRC-valid unknown schemas/metadata are rejected.
 */
bool feature_packet_push(FeaturePacketStream *stream, uint8_t byte, FeaturePacket *output);

/** Discard a partial frame after UART data loss, preserving the reject count. */
void feature_packet_reset(FeaturePacketStream *stream);

#endif
