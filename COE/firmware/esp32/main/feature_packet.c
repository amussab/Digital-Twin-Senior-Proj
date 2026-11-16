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

/** Compute CRC-32/ISO-HDLC over the bytes preceding the checksum. */
static uint32_t packet_crc(const uint8_t *bytes, size_t count)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0U; index < count; ++index)
    {
        crc ^= bytes[index];
        for (unsigned bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
    }
    return crc ^ UINT32_MAX;
}

static float read_f32(const uint8_t *bytes)
{
    const uint32_t bits = read_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

size_t feature_packet_length(const uint8_t header[8])
{
    if (header == NULL || header[5] != 0U || header[6] != FEATURE_PACKET_VALUES || header[7] != 0U)
        return 0U;
    return header[4] == 1U ? FEATURE_PACKET_BYTES : header[4] == 2U ? FEATURE_PACKET_SHAFT_BYTES
         : header[4] == 3U ? FEATURE_PACKET_LATENCY_BYTES : 0U;
}

/** A value that must be finite and >= 0 when its flag is set, and exactly 0 otherwise. */
static bool flagged_value(float value, bool valid)
{
    return valid ? isfinite(value) && value >= 0.0f : value == 0.0f;
}

/** Schema 2 shaft block: flags consistent with the values. */
static bool decode_shaft(const uint8_t *bytes, FeaturePacket *decoded)
{
    decoded->measured_rpm = read_f32(bytes + 156U);
    decoded->revolutions = read_u32(bytes + 160U);
    decoded->shaft_flags = read_u32(bytes + 164U);
    const uint32_t flags = decoded->shaft_flags;
    if ((flags & ~0x1FU) != 0U)
        return false;
    const bool speed = (flags & FEATURE_SHAFT_SPEED_MEASURED) != 0U;
    if (speed ? !(isfinite(decoded->measured_rpm) && decoded->measured_rpm >= (float)FEATURE_PACKET_MIN_RPM
                  && decoded->measured_rpm <= (float)FEATURE_PACKET_MAX_RPM && decoded->revolutions != 0U)
              : (decoded->measured_rpm != 0.0f || decoded->revolutions != 0U))
        return false;
    for (size_t probe = 0U; probe < FEATURE_PACKET_PROBES; ++probe)
    {
        const bool valid = (flags & (FEATURE_SHAFT_PROX1_VALID << probe)) != 0U;
        const bool x1 = (flags & (FEATURE_SHAFT_PROX1_X1_VALID << probe)) != 0U;
        if (x1 && (!valid || !speed))
            return false;
        for (size_t field = 0U; field < 4U; ++field)
        {
            const float value = read_f32(bytes + 168U + 16U * probe + 4U * field);
            if (!flagged_value(value, field < 2U ? valid : x1)
                || (field == 3U && value >= 360.0f))
                return false;
            decoded->probes[probe][field] = value;
        }
    }
    return true;
}

/** Decode only after packet integrity and every value have been checked. */
bool feature_packet_decode(const uint8_t *bytes, FeaturePacket *output)
{
    if (bytes == NULL || output == NULL || memcmp(bytes, magic, sizeof(magic)) != 0)
        return false;
    const size_t length = feature_packet_length(bytes);
    if (length == 0U || packet_crc(bytes, length - 4U) != read_u32(bytes + length - 4U))
        return false;

    FeaturePacket decoded = {0};
    decoded.schema = bytes[4];
    decoded.window_id = read_u32(bytes + 8U);
    decoded.rpm = read_u32(bytes + 12U);
    decoded.bearing_profile = read_u32(bytes + 16U);
    decoded.valid_channels = read_u32(bytes + 20U);
    if (decoded.window_id == 0U || decoded.rpm < FEATURE_PACKET_MIN_RPM || decoded.rpm > FEATURE_PACKET_MAX_RPM
        || decoded.bearing_profile < 1U || decoded.bearing_profile > FEATURE_PACKET_MAX_PROFILE
        || (decoded.valid_channels & ~15U) != 0U || read_u32(bytes + 24U) != 1U)
        return false;
    for (size_t index = 0U; index < FEATURE_PACKET_VALUES; ++index)
    {
        const float value = read_f32(bytes + 28U + 4U * index);
        if (!isfinite(value) || value < 0.0f
            || ((decoded.valid_channels & (1U << (index / 8U))) == 0U && value != 0.0f))
            return false;
        decoded.features[index] = value;
    }
    if (decoded.schema >= 2U && !decode_shaft(bytes, &decoded))
        return false;
    if (decoded.schema == 3U)
    {
        decoded.stm_latency_us = read_u32(bytes + 200U);
        decoded.sample_rate_millihz = read_u32(bytes + 204U);
        if (decoded.stm_latency_us == 0U || decoded.stm_latency_us > FEATURE_PACKET_MAX_LATENCY_US
            || decoded.sample_rate_millihz < FEATURE_PACKET_MIN_RATE_MILLIHZ
            || decoded.sample_rate_millihz > FEATURE_PACKET_MAX_RATE_MILLIHZ)
            return false;
        decoded.rpm_min = read_f32(bytes + 208U);
        decoded.rpm_max = read_f32(bytes + 212U);
        decoded.adc_gaps = read_u32(bytes + 216U);
        decoded.unsteady_windows = read_u32(bytes + 220U);
        decoded.stm_send_failures = read_u32(bytes + 224U);
        const bool speed = (decoded.shaft_flags & FEATURE_SHAFT_SPEED_MEASURED) != 0U;
        if (speed ? !(isfinite(decoded.rpm_min) && isfinite(decoded.rpm_max) && decoded.rpm_min > 0.0f
                      && decoded.rpm_min <= decoded.rpm_max && decoded.rpm_max <= FEATURE_PACKET_MAX_TURN_RPM)
                  : (decoded.rpm_min != 0.0f || decoded.rpm_max != 0.0f))
            return false;
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

/** Keep partial magic or one candidate frame; validate once its schema's length is in. */
bool feature_packet_push(FeaturePacketStream *stream, uint8_t byte, FeaturePacket *output)
{
    if (stream == NULL || output == NULL)
        return false;
    if (stream->count == FEATURE_PACKET_MAX_BYTES) /* cannot happen; never overflow */
        discard_first_byte(stream);
    stream->bytes[stream->count++] = byte;
    for (;;)
    {
        while (stream->count >= sizeof(magic)
               && memcmp(stream->bytes, magic, sizeof(magic)) != 0)
            discard_first_byte(stream);
        if (stream->count < 8U)
            return false;
        const size_t length = feature_packet_length(stream->bytes);
        if (length == 0U)
        {
            ++stream->rejected_packets; /* unknown schema */
            discard_first_byte(stream);
            continue;
        }
        if (stream->count < length)
            return false;
        if (feature_packet_decode(stream->bytes, output))
        {
            /* After a resync the buffer can hold bytes past this packet: keep them. */
            stream->count -= length;
            memmove(stream->bytes, stream->bytes + length, stream->count);
            return true;
        }
        ++stream->rejected_packets;
        discard_first_byte(stream);
    }
}

/** Reset framing after an overflow or parity/framing error. */
void feature_packet_reset(FeaturePacketStream *stream)
{
    if (stream != NULL)
        stream->count = 0U;
}
