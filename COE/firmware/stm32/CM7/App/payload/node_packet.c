/**
 * @file node_packet.c
 * Schema-2 packet: schema-1 features + shaft block (see node_packet.h).
 */
#include "node_packet.h"
#include <math.h>
#include <string.h>

_Static_assert(COE_NODE_PACKET_BYTES == 204U, "Schema 2 is 204 bytes");

static uint32_t read_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
}

static void write_u32(uint8_t *p, uint32_t value)
{
    for (size_t i = 0U; i < 4U; ++i)
        p[i] = (uint8_t)(value >> (8U * i));
}

static void write_f32(uint8_t *p, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    write_u32(p, bits);
}

static uint32_t crc32(const uint8_t *bytes, size_t count)
{
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0U; i < count; ++i)
    {
        crc ^= bytes[i];
        for (size_t bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
    }
    return ~crc;
}

static bool usable(float value)
{
    return isfinite(value) && value >= 0.0f;
}

bool node_packet_encode(const uint8_t feature_packet[COE_PAYLOAD_BYTES], const ShaftWindow *shaft,
                        uint8_t output[COE_NODE_PACKET_BYTES])
{
    if (output == NULL)
        return false;
    memset(output, 0, COE_NODE_PACKET_BYTES);
    if (feature_packet == NULL || shaft == NULL || memcmp(feature_packet, "COE1", 4U) != 0
        || read_u32(feature_packet + 4U) != (COE_PAYLOAD_SCHEMA_VERSION | (COE_FEATURE_COUNT << 16U))
        || crc32(feature_packet, COE_PAYLOAD_BYTES - 4U) != read_u32(feature_packet + COE_PAYLOAD_BYTES - 4U))
        return false;

    uint32_t flags = 0U;
    if (shaft->speed_measured)
    {
        if (!isfinite(shaft->rpm) || shaft->rpm < (float)COE_MIN_RPM || shaft->rpm > (float)COE_MAX_RPM
            || shaft->revolutions == 0U)
            return false;
        flags |= COE_SHAFT_SPEED_MEASURED;
    }
    memcpy(output, feature_packet, COE_SHAFT_BLOCK_OFFSET);
    write_u32(output + 4U, COE_NODE_SCHEMA_VERSION | (COE_FEATURE_COUNT << 16U));
    write_f32(output + 156U, shaft->speed_measured ? shaft->rpm : 0.0f);
    write_u32(output + 160U, shaft->speed_measured ? shaft->revolutions : 0U);
    for (uint32_t p = 0U; p < SHAFT_PROBES; ++p)
    {
        const ShaftProbeWindow *probe = &shaft->probes[p];
        uint8_t *field = output + 168U + 16U * p;
        if (!probe->valid)
            continue;
        if (!usable(probe->gap_um) || !usable(probe->pkpk_um))
        {
            memset(output, 0, COE_NODE_PACKET_BYTES);
            return false;
        }
        flags |= COE_SHAFT_PROX1_VALID << p;
        write_f32(field, probe->gap_um);
        write_f32(field + 4U, probe->pkpk_um);
        if (!probe->x1_valid || !shaft->speed_measured)
            continue;
        if (!usable(probe->x1_um_pp) || !usable(probe->x1_phase_deg) || probe->x1_phase_deg >= 360.0f)
        {
            memset(output, 0, COE_NODE_PACKET_BYTES);
            return false;
        }
        flags |= COE_SHAFT_PROX1_X1_VALID << p;
        write_f32(field + 8U, probe->x1_um_pp);
        write_f32(field + 12U, probe->x1_phase_deg);
    }
    write_u32(output + 164U, flags);
    write_u32(output + COE_NODE_PACKET_BYTES - 4U, crc32(output, COE_NODE_PACKET_BYTES - 4U));
    return true;
}
