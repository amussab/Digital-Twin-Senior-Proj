/** @file feature_json.h
 * Encode one validated feature vector as bounded JSON for the receiving host.
 */
#ifndef COE_FEATURE_JSON_H
#define COE_FEATURE_JSON_H
#include "feature_packet.h"
#define FEATURE_JSON_BYTES 2048U
/** Accept 1-32 letters, digits, hyphens or underscores as a device identifier. */
bool feature_device_id_is_valid(const char *device_id);
/** Serialize a validated packet plus delivery identity; return length or zero
 * on invalid input/insufficient space. Failure clears the output string.
 * Nine significant digits preserve Float32 values through JSON parsing.
 */
size_t feature_packet_to_json(const FeaturePacket *packet, const char *device_id,
                             uint64_t session_id, uint32_t sequence,
                             char *output, size_t capacity);
#endif
