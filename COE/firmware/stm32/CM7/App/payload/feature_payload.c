/**
 * @file feature_payload.c
 * Build a fixed channel-major feature vector and a portable window packet.
 */
#include "feature_payload.h"
#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(COE_CHANNEL_COUNT == 4U && COE_FEATURES_PER_CHANNEL == 8U,
               "Changing feature layout requires a new payload schema.");
_Static_assert(sizeof(float) == 4U && FLT_RADIX == 2 && FLT_MANT_DIG == 24
               && FLT_MAX_EXP == 128, "Payload requires IEEE binary32.");

/** Write an integer explicitly in little-endian byte order, without padding/alignment assumptions. */
static void write_u32(uint8_t *destination, uint32_t value)
{
    for (size_t byte = 0U; byte < 4U; ++byte)
        destination[byte] = (uint8_t)(value >> (8U * byte));
}

/** Standard reflected CRC-32/ISO-HDLC, matching Python zlib.crc32. */
static uint32_t calculate_crc32(const uint8_t *bytes, size_t count)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0U; index < count; ++index)
    {
        crc ^= bytes[index];
        for (size_t bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
    }
    return ~crc;
}

/** Store metadata, Float32 bits and checksum in the fixed packet. */
static void encode_packet(AiFeaturePayload *payload, uint32_t window_id,
                           uint32_t rpm, uint32_t profile_id)
{
    memcpy(payload->bytes, "COE1", 4U);
    write_u32(payload->bytes + 4U, COE_PAYLOAD_SCHEMA_VERSION | (COE_FEATURE_COUNT << 16U));
    write_u32(payload->bytes + 8U, window_id);
    write_u32(payload->bytes + 12U, rpm);
    write_u32(payload->bytes + 16U, profile_id);
    write_u32(payload->bytes + 20U, payload->channel_valid_mask);
    write_u32(payload->bytes + 24U, COE_ACCELERATION_UNIT_G);
    for (size_t index = 0U; index < COE_FEATURE_COUNT; ++index)
    {
        uint32_t bits;
        memcpy(&bits, &payload->features[index], sizeof(bits));
        write_u32(payload->bytes + COE_PAYLOAD_HEADER_BYTES + 4U * index, bits);
    }
    write_u32(payload->bytes + COE_PAYLOAD_BYTES - 4U,
              calculate_crc32(payload->bytes, COE_PAYLOAD_BYTES - 4U));
}

/** Assemble each valid channel, validate values, then encode the completed vector. */
bool build_ai_payload(const TimeDomainFeatures time_features[COE_CHANNEL_COUNT],
                       const FrequencyDomainFeatures frequency_features[COE_CHANNEL_COUNT],
                       const bool valid[COE_CHANNEL_COUNT], uint32_t window_id,
                       uint32_t rpm, uint32_t bearing_profile_id, AiFeaturePayload *payload)
{
    if (payload == NULL)
        return false;
    memset(payload, 0, sizeof(*payload));
    if (time_features == NULL || frequency_features == NULL || valid == NULL
        || window_id == 0U || rpm < COE_MIN_RPM || rpm > COE_MAX_RPM
        || (bearing_profile_id != COE_PROJECT_BEARING && bearing_profile_id != COE_CWRU_DRIVE_END_BEARING))
        return false;
    for (size_t channel = 0U; channel < COE_CHANNEL_COUNT; ++channel)
    {
        if (!valid[channel])
            continue;
        float *row = payload->features + channel * COE_FEATURES_PER_CHANNEL;
        row[0] = time_features[channel].rms;
        row[1] = time_features[channel].kurtosis;
        row[2] = time_features[channel].crest_factor;
        row[3] = frequency_features[channel].envelope_rms;
        for (size_t family = 0U; family < COE_BEARING_FAMILIES; ++family)
            row[4U + family] = frequency_features[channel].band_rms[family];
        for (size_t index = 0U; index < COE_FEATURES_PER_CHANNEL; ++index)
            if (!isfinite(row[index]) || row[index] < 0.0f)
            {
                memset(payload, 0, sizeof(*payload));
                return false;
            }
        payload->channel_valid_mask |= 1U << channel;
    }
    encode_packet(payload, window_id, rpm, bearing_profile_id);
    return true;
}
