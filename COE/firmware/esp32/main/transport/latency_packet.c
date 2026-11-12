/** @file latency_packet.c
 * Portable timing-frame encoder/parser; no device APIs and no heap allocation.
 */
#include "latency_packet.h"
#include <string.h>
/** Read a potentially unaligned little-endian integer. */
static uint32_t read_u32(const uint8_t *p)
{ return (uint32_t)p[0] | (uint32_t)p[1] << 8U | (uint32_t)p[2] << 16U | (uint32_t)p[3] << 24U; }
/** Write a little-endian integer without casting the byte storage. */
static void write_u32(uint8_t *p, uint32_t value)
{ for (unsigned i = 0U; i < 4U; ++i) p[i] = (uint8_t)(value >> (8U * i)); }
/** Protect every preceding field with the same CRC polynomial as COE1. */
static uint32_t crc32(const uint8_t *p, size_t n)
{
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0U; i < n; ++i) {
        crc ^= p[i];
        for (unsigned bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
    }
    return crc ^ UINT32_MAX;
}
void latency_request_encode(uint32_t id, uint8_t bytes[LATENCY_REQUEST_BYTES])
{
    memcpy(bytes, "COT1", 4U);
    write_u32(bytes + 4U, id);
    write_u32(bytes + 8U, crc32(bytes, 8U));
}
void latency_ack_encode(const LatencyAck *ack, uint8_t bytes[LATENCY_ACK_BYTES])
{
    memcpy(bytes, "COA1", 4U);
    write_u32(bytes + 4U, ack->measurement_id);
    write_u32(bytes + 8U, ack->window_id);
    write_u32(bytes + 12U, ack->rpm);
    write_u32(bytes + 16U, ack->bearing_profile);
    write_u32(bytes + 20U, ack->esp_total_us);
    write_u32(bytes + 24U, ack->esp_queue_us);
    write_u32(bytes + 28U, ack->esp_network_us);
    write_u32(bytes + 32U, crc32(bytes, 32U));
}
/** Use a bounded sliding window so a corrupt/truncated frame can recover. */
static bool push_frame(LatencyStream *stream, uint8_t byte, const char *magic, size_t n)
{
    if (stream->count == n) {
        memmove(stream->bytes, stream->bytes + 1U, n - 1U);
        --stream->count;
    }
    stream->bytes[stream->count++] = byte;
    return stream->count == n && memcmp(stream->bytes, magic, 4U) == 0
        && crc32(stream->bytes, n - 4U) == read_u32(stream->bytes + n - 4U);
}
bool latency_request_push(LatencyStream *stream, uint8_t byte, uint32_t *id)
{
    if (!push_frame(stream, byte, "COT1", LATENCY_REQUEST_BYTES)) return false;
    const uint32_t value = read_u32(stream->bytes + 4U);
    if (value == 0U) return false;
    *id = value;
    stream->count = 0U;
    return true;
}
bool latency_ack_push(LatencyStream *stream, uint8_t byte, LatencyAck *ack)
{
    if (!push_frame(stream, byte, "COA1", LATENCY_ACK_BYTES)) return false;
    const uint8_t *p = stream->bytes;
    const LatencyAck decoded = {read_u32(p + 4U), read_u32(p + 8U), read_u32(p + 12U),
        read_u32(p + 16U), read_u32(p + 20U), read_u32(p + 24U), read_u32(p + 28U)};
    if (decoded.measurement_id == 0U || decoded.window_id == 0U
        || decoded.rpm < 60U || decoded.rpm > 3600U
        || (decoded.bearing_profile != 1U && decoded.bearing_profile != 2U)
        || (uint64_t)decoded.esp_queue_us + decoded.esp_network_us != decoded.esp_total_us)
        return false;
    *ack = decoded;
    stream->count = 0U;
    return true;
}

/** Demultiplex only outside a feature frame, even if an ID contains COE1 bytes. */
bool latency_feature_push(FeatureTimingStream *stream, uint8_t byte,
                          FeaturePacket *packet, uint32_t *measurement_id)
{
    if (stream->request_bytes_left != 0U) {
        (void)latency_request_push(&stream->requests, byte, &stream->measurement_id);
        if (--stream->request_bytes_left == 0U) stream->requests.count = 0U;
        return false;
    }
    if (stream->features.count < 4U) {
        uint32_t unused;
        (void)latency_request_push(&stream->requests, byte, &unused);
        const size_t n = stream->requests.count;
        if (n >= 4U && memcmp(stream->requests.bytes + n - 4U, "COT1", 4U) == 0) {
            memcpy(stream->requests.bytes, "COT1", 4U);
            stream->requests.count = 4U;
            stream->request_bytes_left = LATENCY_REQUEST_BYTES - 4U;
            stream->measurement_id = 0U;
            feature_packet_reset(&stream->features);
            return false;
        }
    }
    if (!feature_packet_push(&stream->features, byte, packet)) return false;
    *measurement_id = stream->measurement_id;
    stream->measurement_id = 0U;
    stream->requests.count = 0U;
    return true;
}
void latency_feature_reset(FeatureTimingStream *stream)
{
    feature_packet_reset(&stream->features);
    stream->requests.count = 0U;
    stream->request_bytes_left = 0U;
    stream->measurement_id = 0U;
}
