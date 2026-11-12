/** @file feature_packet.c
 * Validate the wire format independently of the STM32 payload encoder.
 */
#include "feature_packet.h"
#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == 4U && FLT_RADIX == 2 && FLT_MANT_DIG == 24
               && FLT_MAX_EXP == 128, "Protocol requires IEEE binary32 floats");
static const uint8_t magic[4] = {'C', 'O', 'E', '1'};

/** Read an unsigned little-endian field without alignment assumptions. */
static uint32_t read_u32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U)
           | ((uint32_t)bytes[2] << 16U) | ((uint32_t)bytes[3] << 24U);
}

/** Compute CRC-32/ISO-HDLC over the 156 bytes preceding the checksum. */
static uint32_t packet_crc(const uint8_t *bytes)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0U; index < FEATURE_PACKET_BYTES - 4U; ++index)
    {
        crc ^= bytes[index];
        for (unsigned bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
    }
    return crc ^ UINT32_MAX;
}

/** Decode only after packet integrity and every value have been checked. */
bool feature_packet_decode(const uint8_t bytes[FEATURE_PACKET_BYTES], FeaturePacket *output)
{
    if (bytes == NULL || output == NULL || memcmp(bytes, magic, sizeof(magic)) != 0
        || bytes[4] != 1U || bytes[5] != 0U || bytes[6] != 32U || bytes[7] != 0U
        || packet_crc(bytes) != read_u32(bytes + 156U))
        return false;

    FeaturePacket decoded = {0};
    decoded.window_id = read_u32(bytes + 8U);
    decoded.rpm = read_u32(bytes + 12U);
    decoded.bearing_profile = read_u32(bytes + 16U);
    decoded.valid_channels = read_u32(bytes + 20U);
    if (decoded.window_id == 0U || decoded.rpm < 60U || decoded.rpm > 3600U
        || (decoded.bearing_profile != 1U && decoded.bearing_profile != 2U)
        || (decoded.valid_channels & ~15U) != 0U || read_u32(bytes + 24U) != 1U)
        return false;
    for (size_t index = 0U; index < FEATURE_PACKET_VALUES; ++index)
    {
        const uint32_t bits = read_u32(bytes + 28U + 4U * index);
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (!isfinite(value) || value < 0.0f
            || ((decoded.valid_channels & (1U << (index / 8U))) == 0U && value != 0.0f))
            return false;
        decoded.features[index] = value;
    }
    *output = decoded;
    return true;
}

/** Drop one leading byte so the next possible magic can be found. */
static void discard_first_byte(FeaturePacketStream *stream)
{
    --stream->count;
    memmove(stream->bytes, stream->bytes + 1U, stream->count);
}

/** Keep partial magic or one candidate frame, then validate at exactly 160 bytes. */
bool feature_packet_push(FeaturePacketStream *stream, uint8_t byte, FeaturePacket *output)
{
    if (stream == NULL || output == NULL)
        return false;
    stream->bytes[stream->count++] = byte;
    while (stream->count >= sizeof(magic)
           && memcmp(stream->bytes, magic, sizeof(magic)) != 0)
        discard_first_byte(stream);
    if (stream->count != FEATURE_PACKET_BYTES)
        return false;
    if (feature_packet_decode(stream->bytes, output))
    {
        stream->count = 0U;
        return true;
    }
    ++stream->rejected_packets;
    discard_first_byte(stream);
    while (stream->count >= sizeof(magic)
           && memcmp(stream->bytes, magic, sizeof(magic)) != 0)
        discard_first_byte(stream);
    return false;
}

/** Reset framing after an overflow or parity/framing error. */
void feature_packet_reset(FeaturePacketStream *stream)
{
    if (stream != NULL)
        stream->count = 0U;
}
