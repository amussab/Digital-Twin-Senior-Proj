/** @file feature_json.c
 * Portable JSON encoding; no network calls, allocation or SDK dependencies.
 */
#include "feature_json.h"
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

/** Restrict device IDs so they can be inserted into JSON and HTTP headers safely. */
bool feature_device_id_is_valid(const char *device_id)
{
    if (device_id == NULL || device_id[0] == '\0')
        return false;
    size_t count = 0U;
    while (device_id[count] != '\0')
    {
        const char character = device_id[count];
        if (++count > 32U || !((character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9') || character == '-' || character == '_'))
            return false;
    }
    return true;
}

/** Append formatted text only if the full text and terminating NUL fit. */
static bool append_json(char *output, size_t capacity, size_t *used, const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    const int count = vsnprintf(output + *used, capacity - *used, format, arguments);
    va_end(arguments);
    if (count < 0 || (size_t)count >= capacity - *used)
        return false;
    *used += (size_t)count;
    return true;
}

/** Build the host-facing representation without modifying the original features. */
size_t feature_packet_to_json(const FeaturePacket *packet, const char *device_id,
                             uint64_t session_id, uint32_t sequence,
                             char *output, size_t capacity)
{
    if (output == NULL || capacity == 0U)
        return 0U;
    output[0] = '\0';
    if (packet == NULL || sequence == 0U || !feature_device_id_is_valid(device_id))
        return 0U;
    size_t used = 0U;
    const unsigned schema = packet->schema == 2U ? 2U : 1U;
    if (!append_json(output, capacity, &used,
        "{\"schema_version\":%u,\"device_id\":\"%s\",\"session_id\":\"%016" PRIx64
        "\",\"sequence\":%" PRIu32 ",\"window_id\":%" PRIu32 ",\"rpm\":%" PRIu32
        ",\"bearing_profile\":%" PRIu32 ",\"valid_channels\":%" PRIu32
        ",\"acceleration_unit\":\"g\",\"features\":[",
        schema, device_id, session_id, sequence, packet->window_id, packet->rpm,
        packet->bearing_profile, packet->valid_channels))
        goto failed;
    for (size_t index = 0U; index < FEATURE_PACKET_VALUES; ++index)
    {
        const float value = packet->features[index];
        if (!isfinite(value) || value < 0.0f
            || !append_json(output, capacity, &used, index == 0U ? "%.9g" : ",%.9g", (double)value))
            goto failed;
    }
    if (!append_json(output, capacity, &used, "]"))
        goto failed;
    if (schema == 2U)
    {
        /* Keyphasor speed and the two proximity probes (0 when not valid, see shaft_flags). */
        static const char *const names[FEATURE_PACKET_PROBES] = {"prox1", "prox3"};
        if (!isfinite(packet->measured_rpm)
            || !append_json(output, capacity, &used,
                            ",\"measured_rpm\":%.9g,\"revolutions\":%" PRIu32 ",\"shaft_flags\":%" PRIu32,
                            (double)packet->measured_rpm, packet->revolutions, packet->shaft_flags))
            goto failed;
        for (size_t probe = 0U; probe < FEATURE_PACKET_PROBES; ++probe)
        {
            const float *v = packet->probes[probe];
            if (!isfinite(v[0]) || !isfinite(v[1]) || !isfinite(v[2]) || !isfinite(v[3])
                || !append_json(output, capacity, &used,
                                ",\"%s_gap_um\":%.9g,\"%s_pkpk_um\":%.9g,\"%s_x1_um_pp\":%.9g,"
                                "\"%s_x1_phase_deg\":%.9g",
                                names[probe], (double)v[0], names[probe], (double)v[1],
                                names[probe], (double)v[2], names[probe], (double)v[3]))
                goto failed;
        }
    }
    if (!append_json(output, capacity, &used, "}"))
        goto failed;
    return used;
failed:
    output[0] = '\0';
    return 0U;
}
