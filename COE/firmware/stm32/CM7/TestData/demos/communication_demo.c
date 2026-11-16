/** @file communication_demo.c
 * Verify the portable ESP receiver decoder on CM7 and provide B1 test replay.
 * These tests do not establish that the ESP32 has physically received a packet.
 */
#include "communication_demo.h"
#include "delivery_timing.h"
#include "feature_packet.h"
#include "final_payload_fixture.h"
#include "pipeline_batch_demo.h"
#include "feature_uart.h"
#include "main.h"
#include "node_packet.h"
#include <string.h>

volatile bool communication_checks_passed;
volatile uint32_t communication_checked_cases;
volatile uint32_t communication_failed_check;

/** Preserve the first failed case and expose how many cases ran. */
static void record_check(bool passed)
{
    ++communication_checked_cases;
    if (!passed && communication_failed_check == 0U)
        communication_failed_check = communication_checked_cases;
}

/** Refresh the CRC of deliberately malformed metadata so rejection tests
 * exercise schema/value validation rather than only the checksum check.
 */
#define RATE 29998500U /* 29,998.5 S/s, the board's AD7606 rate in mHz */
static const NodeHealth HEALTH = {3U, 7U, 1U}; /* gaps, unsteady windows, send failures */

static void refresh_crc_length(uint8_t *bytes, size_t length)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0U; index < length - 4U; ++index)
    {
        crc ^= bytes[index];
        for (unsigned bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
    }
    crc ^= UINT32_MAX;
    for (unsigned index = 0U; index < 4U; ++index)
        bytes[length - 4U + index] = (uint8_t)(crc >> (8U * index));
}

static void refresh_crc(uint8_t *bytes)
{
    refresh_crc_length(bytes, FEATURE_PACKET_BYTES);
}

static void put_f32(uint8_t *bytes, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    for (unsigned index = 0U; index < 4U; ++index)
        bytes[index] = (uint8_t)(bits >> (8U * index));
}

/** Confirm metadata and every channel-major value in the independent fixture. */
static bool matches_fixture(const FeaturePacket *packet)
{
    if (packet->window_id != 17U || packet->rpm != 1750U
        || packet->bearing_profile != 1U || packet->valid_channels != 15U)
        return false;
    for (size_t index = 0U; index < FEATURE_PACKET_VALUES; ++index)
        if (packet->features[index] != (float)((index / 8U) * 10U + index % 8U + 1U))
            return false;
    return true;
}

/** Feed arbitrary bytes and count only valid, correctly decoded fixture packets. */
static uint32_t feed_bytes(FeaturePacketStream *stream, const uint8_t *bytes, size_t length)
{
    uint32_t accepted = 0U;
    FeaturePacket packet;
    for (size_t index = 0U; index < length; ++index)
        if (feature_packet_push(stream, bytes[index], &packet))
        {
            if (!matches_fixture(&packet))
                record_check(false);
            ++accepted;
        }
    return accepted;
}

/** Schema 2 (LIVE since 16 Nov 2026): the STM32 encoder (node_packet.c) and the
 * ESP32 decoder agree; every shaft-block rule is enforced on both sides. */
static void check_shaft_packets(void)
{
    static uint8_t node[COE_NODE_PACKET_BYTES], bad[COE_NODE_PACKET_BYTES];
    ShaftWindow w;
    memset(&w, 0, sizeof(w));
    w.speed_measured = true;
    w.rpm = 1749.93f;
    w.rpm_min = 1749.5f;
    w.rpm_max = 1750.25f;
    w.revolutions = 20U;
    w.probes[0] = (ShaftProbeWindow){true, true, 1231.9f, 52.25f, 50.0f, 70.5f, -9.7f};
    w.probes[1] = (ShaftProbeWindow){true, false, 1041.4f, 20.0f, 0.0f, 0.0f, -8.2f};
    FeaturePacket out;
    const bool encoded = node_packet_encode(final_payload_expected, &w, 12345U, RATE, &HEALTH, node);
    record_check(encoded && feature_packet_length(node) == FEATURE_PACKET_LATENCY_BYTES
                 && memcmp(node, final_payload_expected, 4U) == 0 && node[4] == 3U
                 && memcmp(node + 5U, final_payload_expected + 5U, 151U) == 0);
    record_check(feature_packet_decode(node, &out) && matches_fixture(&out) && out.schema == 3U
                 && out.stm_latency_us == 12345U && out.sample_rate_millihz == 29998500U
                 && out.rpm_min == 1749.5f && out.rpm_max == 1750.25f && out.adc_gaps == 3U
                 && out.unsteady_windows == 7U && out.stm_send_failures == 1U
                 && out.measured_rpm == 1749.93f && out.revolutions == 20U
                 && out.shaft_flags == (FEATURE_SHAFT_SPEED_MEASURED | FEATURE_SHAFT_PROX1_VALID
                                        | FEATURE_SHAFT_PROX3_VALID | FEATURE_SHAFT_PROX1_X1_VALID)
                 && out.probes[0][0] == 1231.9f && out.probes[0][1] == 52.25f
                 && out.probes[0][2] == 50.0f && out.probes[0][3] == 70.5f
                 && out.probes[1][0] == 1041.4f && out.probes[1][1] == 20.0f
                 && out.probes[1][2] == 0.0f && out.probes[1][3] == 0.0f);

    /* The encoder refuses a bad input packet and impossible shaft values. */
    ShaftWindow v = w;
    memcpy(bad, final_payload_expected, FEATURE_PACKET_BYTES);
    bad[60] ^= 1U;
    record_check(!node_packet_encode(bad, &w, 12345U, RATE, &HEALTH, node) && node[0] == 0U);
    v.probes[0].gap_um = -1.0f;
    record_check(!node_packet_encode(final_payload_expected, &v, 12345U, RATE, &HEALTH, node));
    v = w;
    v.probes[0].x1_phase_deg = 360.0f;
    record_check(!node_packet_encode(final_payload_expected, &v, 12345U, RATE, &HEALTH, node));
    v = w;
    v.rpm = 4100.0f;
    record_check(!node_packet_encode(final_payload_expected, &v, 12345U, RATE, &HEALTH, node));
    /* The STM32 latency must be 1 us .. 10 s. */
    record_check(!node_packet_encode(final_payload_expected, &w, 0U, RATE, &HEALTH, node)
                 && !node_packet_encode(final_payload_expected, &w, COE_NODE_MAX_LATENCY_US + 1U, RATE, &HEALTH, node)
                 && node_packet_encode(final_payload_expected, &w, COE_NODE_MAX_LATENCY_US, RATE, &HEALTH, node));
    /* The sample rate must be 1 kS/s .. 200 kS/s. */
    record_check(!node_packet_encode(final_payload_expected, &w, 12345U, COE_NODE_MIN_RATE_MILLIHZ - 1U, &HEALTH, node)
                 && !node_packet_encode(final_payload_expected, &w, 12345U, COE_NODE_MAX_RATE_MILLIHZ + 1U, &HEALTH, node)
                 && node_packet_encode(final_payload_expected, &w, 12345U, COE_NODE_MIN_RATE_MILLIHZ, &HEALTH, node));
    /* Health counters are required; the slowest turn must not be faster than the fastest. */
    v = w;
    v.rpm_min = 1751.0f;
    record_check(!node_packet_encode(final_payload_expected, &w, 12345U, RATE, NULL, node)
                 && !node_packet_encode(final_payload_expected, &v, 12345U, RATE, &HEALTH, node));
    /* No Keyphasor speed: no 1X either, even if the probe had it. */
    v = w;
    v.speed_measured = false;
    record_check(node_packet_encode(final_payload_expected, &v, 12345U, RATE, &HEALTH, node) && feature_packet_decode(node, &out)
                 && out.shaft_flags == (FEATURE_SHAFT_PROX1_VALID | FEATURE_SHAFT_PROX3_VALID)
                 && out.measured_rpm == 0.0f && out.revolutions == 0U && out.rpm_min == 0.0f && out.rpm_max == 0.0f
                 && out.probes[0][2] == 0.0f && out.probes[0][3] == 0.0f && out.probes[0][0] == 1231.9f);
    /* A probe that is not valid sends zeros and no flag. */
    v = w;
    v.probes[1].valid = false;
    record_check(node_packet_encode(final_payload_expected, &v, 12345U, RATE, &HEALTH, node) && feature_packet_decode(node, &out)
                 && (out.shaft_flags & FEATURE_SHAFT_PROX3_VALID) == 0U && out.probes[1][0] == 0.0f);

    /* Decoder: CRC-valid packets that break a rule are rejected. */
    (void)node_packet_encode(final_payload_expected, &w, 12345U, RATE, &HEALTH, node);
    for (unsigned test = 0U; test < 14U; ++test)
    {
        memcpy(bad, node, sizeof(bad));
        switch (test)
        {
        case 0U: bad[164] |= 0x20U; break;           /* unknown flag bit */
        case 1U: bad[164] = 0x1DU; break;            /* PROX1 1X without PROX1 valid */
        case 2U: bad[164] = 0x0EU; break;            /* 1X without a measured speed */
        case 3U: put_f32(bad + 156U, 0.0f); break;   /* speed flag but 0 RPM */
        case 4U: bad[160] = 0U; break;               /* speed flag but 0 turns */
        case 5U: put_f32(bad + 196U, 1.0f); break;   /* PROX3 1X value, flag clear */
        case 6U: put_f32(bad + 180U, 360.0f); break; /* phase out of range */
        case 7U: memset(bad + 200U, 0, 4U); break;   /* STM32 latency 0 */
        case 8U: bad[203] = 0x01U; break;            /* STM32 latency > 10 s */
        case 9U: memset(bad + 204U, 0, 4U); break;   /* sample rate 0 */
        case 10U: bad[207] = 0x10U; break;           /* sample rate > 200 kS/s */
        case 11U: put_f32(bad + 208U, 1751.0f); break; /* slowest turn faster than the fastest */
        case 12U: put_f32(bad + 212U, 9000.0f); break; /* turn speed above the limit */
        default: put_f32(bad + 168U, -5.0f); break;  /* negative gap */
        }
        refresh_crc_length(bad, sizeof(bad));
        record_check(!feature_packet_decode(bad, &out));
    }
    memcpy(bad, node, sizeof(bad));
    bad[190] ^= 1U; /* corruption in the shaft block, CRC not refreshed */
    record_check(!feature_packet_decode(bad, &out));

    /* Stream: schema 1 and 3 packets back to back, and recovery after a bad schema-3 packet. */
    FeaturePacketStream stream = {0};
    record_check(feed_bytes(&stream, node, sizeof(node)) == 1U
                 && feed_bytes(&stream, final_payload_expected, FEATURE_PACKET_BYTES) == 1U
                 && feed_bytes(&stream, node, sizeof(node)) == 1U);
    record_check(feed_bytes(&stream, bad, sizeof(bad)) == 0U
                 && feed_bytes(&stream, final_payload_expected, FEATURE_PACKET_BYTES) == 1U
                 && feed_bytes(&stream, node, 100U) == 0U && feed_bytes(&stream, node + 100U, sizeof(node) - 100U) == 1U);
}

/** Exercise malformed packets and stream recovery without modifying DSP output. */
void run_communication_demo(void)
{
    communication_checked_cases = 0U;
    communication_failed_check = 0U;
    communication_checks_passed = false;
    FeaturePacket output;
    uint8_t packet[FEATURE_PACKET_BYTES];
    record_check(feature_packet_decode(final_payload_expected, &output) && matches_fixture(&output));
    memcpy(packet, final_payload_expected, sizeof(packet));
    refresh_crc(packet);
    record_check(memcmp(packet, final_payload_expected, sizeof(packet)) == 0);

    /* Wrong magic, schema, count, window, RPM, profile, mask, unit, NaN, infinity,
     * negative acceleration, and a nonzero value on a channel marked invalid.
     */
    const size_t offsets[] = {0U, 4U, 6U, 8U, 13U, 16U, 20U, 24U, 31U, 31U, 31U, 20U};
    const uint8_t values[] = {'X', 4U, 31U, 0U, 0x10U, 4U, 31U, 2U, 0x7FU, 0x7FU, 0xBFU, 14U};
    /* Offset 4 = 4: schema 4 does not exist (1, 2 and 3 do).
     * Offset 13 = 0x10: RPM 1750 becomes 4310, above the 4000 limit (any 60..4000 is valid).
     * Offset 16 = 4: bearing profile 4 does not exist (1..3 are valid). */
    for (size_t test = 0U; test < sizeof(offsets) / sizeof(offsets[0]); ++test)
    {
        memcpy(packet, final_payload_expected, sizeof(packet));
        packet[offsets[test]] = values[test];
        if (test == 8U) /* Quiet NaN 0x7fc00000. */
            packet[30] = 0xC0U;
        if (test == 9U) /* Positive infinity 0x7f800000. */
            packet[30] = 0x80U;
        refresh_crc(packet);
        const FeaturePacket saved = output;
        record_check(!feature_packet_decode(packet, &output)
                     && memcmp(&saved, &output, sizeof(output)) == 0);
    }
    memcpy(packet, final_payload_expected, sizeof(packet));
    packet[60] ^= 1U; /* Corruption without CRC refresh. */
    record_check(!feature_packet_decode(packet, &output));

    /* All-invalid window: zero values, zero validity mask, intact schema. */
    memcpy(packet, final_payload_expected, sizeof(packet));
    packet[20] = 0U;
    memset(packet + 28U, 0, 128U);
    refresh_crc(packet);
    record_check(feature_packet_decode(packet, &output) && output.valid_channels == 0U);

    FeaturePacketStream stream = {0};
    const uint8_t noise[] = {'C', 'O', 'E', 'X', 0U, 'C', 'C', 'O'};
    record_check(feed_bytes(&stream, noise, sizeof(noise)) == 0U
                 && feed_bytes(&stream, final_payload_expected, 2U) == 0U
                 && feed_bytes(&stream, final_payload_expected + 2U, 158U) == 1U);
    record_check(feed_bytes(&stream, final_payload_expected, 160U) == 1U
                 && feed_bytes(&stream, final_payload_expected, 160U) == 1U);
    memcpy(packet, final_payload_expected, sizeof(packet));
    packet[60] ^= 1U;
    record_check(feed_bytes(&stream, packet, 160U) == 0U
                 && feed_bytes(&stream, final_payload_expected, 160U) == 1U
                 && stream.rejected_packets != 0U);
    record_check(feed_bytes(&stream, final_payload_expected, 80U) == 0U
                 && feed_bytes(&stream, final_payload_expected, 160U) == 1U);
    record_check(feed_bytes(&stream, final_payload_expected, 50U) == 0U
                 && feed_bytes(&stream, final_payload_expected + 51U, 109U) == 0U
                 && feed_bytes(&stream, final_payload_expected, 160U) == 1U);
    (void)feed_bytes(&stream, final_payload_expected, 50U);
    feature_packet_reset(&stream);
    record_check(stream.count == 0U
                 && feed_bytes(&stream, final_payload_expected, 160U) == 1U);
    record_check(feature_packet_decode(final_payload_expected, &output) && output.schema == 1U
                 && output.shaft_flags == 0U && output.measured_rpm == 0.0f);
    check_shaft_packets();
    communication_checks_passed = communication_failed_check == 0U;
}

/** Accept one stable rising edge; holding B1 never repeats the batch test. */
void poll_communication_replay(void)
{
    static bool last_read;
    static bool stable_pressed;
    static uint32_t change_tick;
    const bool pressed = BSP_PB_GetState(BUTTON_USER) != 0;
    const uint32_t now = HAL_GetTick();
    if (pressed != last_read)
    {
        last_read = pressed;
        change_tick = now;
    }
    if ((uint32_t)(now - change_tick) >= 30U && pressed != stable_pressed)
    {
        stable_pressed = pressed;
        if (pressed && feature_uart_ready)
        {
            delivery_timing_start_batch();
            run_pipeline_batch_demo();
        }
    }
}
