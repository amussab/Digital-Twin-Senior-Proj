/**
 * @file feature_payload.h
 * Assemble eight features/channel and encode one fixed 160-byte window packet.
 */
#ifndef COE_FEATURE_PAYLOAD_H
#define COE_FEATURE_PAYLOAD_H

#include "time_features.h"
#include "frequency_features.h"

#define COE_PAYLOAD_SCHEMA_VERSION 1U
#define COE_ACCELERATION_UNIT_G 1U
#define COE_PAYLOAD_HEADER_BYTES 28U
#define COE_PAYLOAD_BYTES (COE_PAYLOAD_HEADER_BYTES + COE_FEATURE_COUNT * 4U + 4U)

/** Owned fixed output: 32 readable Float32 features plus their canonical wire representation. */
typedef struct
{
    float features[COE_FEATURE_COUNT];
    uint8_t bytes[COE_PAYLOAD_BYTES];
    uint32_t channel_valid_mask; /* Bit 0..3 corresponds to bearing 1 X/Y, bearing 2 X/Y. */
} AiFeaturePayload;

/**
 * Combine time/frequency features in channel-major order and serialize little-endian.
 * Per channel: bandpass RMS, Pearson kurtosis, crest factor, envelope RMS,
 * FTF RMS, single-BSF RMS, BPFO RMS, BPFI RMS. Acceleration features are in g.
 * Invalid channels contain eight zeros with a cleared validity bit. Valid
 * channels must contain finite nonnegative values; failure clears the output.
 * Packet contains magic/schema/count/window/RPM/profile/validity/unit, values,
 * then standard CRC-32. No allocation, transmission or AI inference is performed.
 */
bool build_ai_payload(const TimeDomainFeatures time_features[COE_CHANNEL_COUNT],
                       const FrequencyDomainFeatures frequency_features[COE_CHANNEL_COUNT],
                       const bool valid[COE_CHANNEL_COUNT], uint32_t window_id,
                       uint32_t rpm, uint32_t bearing_profile_id, AiFeaturePayload *payload);

#endif
