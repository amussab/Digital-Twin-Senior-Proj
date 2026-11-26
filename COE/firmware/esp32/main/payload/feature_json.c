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
    const unsigned schema = packet->schema == 3U ? 3U : packet->schema == 2U ? 2U : 1U;
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
    if (schema >= 2U)
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
    if (schema == 3U)
    {
        /* End-to-end latency chain, in us. uart: the packet's time on the 115200-baud
         * wire plus the ESP32 UART's 10-character idle timeout (an upper bound).
         * wifi: the previous accepted HTTP request's round trip (0 = none yet).
         * The host adds its own time (receive -> AI result) to node_latency_us. */
        const uint32_t uart_us = (uint32_t)(((uint64_t)(FEATURE_PACKET_LATENCY_BYTES + 10U) * 10U * 1000000U
                                             + FEATURE_PACKET_UART_BAUD - 1U) / FEATURE_PACKET_UART_BAUD);
        const uint64_t node_us = (uint64_t)packet->stm_latency_us + uart_us + packet->esp_latency_us
                                 + packet->wifi_latency_us;
        const uint32_t rate = packet->sample_rate_millihz;
        if (packet->stm_latency_us == 0U || node_us > UINT32_MAX
            || rate < FEATURE_PACKET_MIN_RATE_MILLIHZ || rate > FEATURE_PACKET_MAX_RATE_MILLIHZ
            || !append_json(output, capacity, &used,
                            ",\"stm_latency_us\":%" PRIu32 ",\"uart_latency_us\":%" PRIu32
                            ",\"esp_latency_us\":%" PRIu32 ",\"wifi_latency_us\":%" PRIu32
                            ",\"node_latency_us\":%" PRIu32,
                            packet->stm_latency_us, uart_us, packet->esp_latency_us,
                            packet->wifi_latency_us, (uint32_t)node_us)
            /* ADC sample rate per channel, exact to 1 mHz, e.g. 29998.500 */
            || !append_json(output, capacity, &used, ",\"sample_rate_hz\":%" PRIu32 ".%03" PRIu32,
                            rate / 1000U, rate % 1000U))
            goto failed;
        /* Speed range in the window, and node health: STM32 counters since LIVE started,
         * ESP32 counters since it started, the Wi-Fi link and its signal. */
        static const char *const links[] = {"none", "node", "hotspot"};
        const bool linked = packet->wifi_link != FEATURE_WIFI_LINK_NONE;
        if (!isfinite(packet->rpm_min) || !isfinite(packet->rpm_max)
            || packet->wifi_link > FEATURE_WIFI_LINK_HOTSPOT
            || (linked ? packet->wifi_rssi_dbm < -127 || packet->wifi_rssi_dbm > 0 : packet->wifi_rssi_dbm != 0)
            || !append_json(output, capacity, &used,
                            ",\"rpm_min\":%.9g,\"rpm_max\":%.9g,\"adc_gaps\":%" PRIu32
                            ",\"unsteady_windows\":%" PRIu32 ",\"stm_send_failures\":%" PRIu32
                            ",\"esp_rx_rejected\":%" PRIu32 ",\"esp_dropped\":%" PRIu32
                            ",\"wifi_link\":\"%s\"",
                            (double)packet->rpm_min, (double)packet->rpm_max, packet->adc_gaps,
                            packet->unsteady_windows, packet->stm_send_failures,
                            packet->esp_rx_rejected, packet->esp_dropped, links[packet->wifi_link])
            || !(linked ? append_json(output, capacity, &used, ",\"wifi_rssi_dbm\":%" PRId32,
                                      packet->wifi_rssi_dbm)
                        : append_json(output, capacity, &used, ",\"wifi_rssi_dbm\":null")))
            goto failed;
    }
    if (!append_json(output, capacity, &used, "}"))
        goto failed;
    return used;
failed:
    output[0] = '\0';
    return 0U;
}
