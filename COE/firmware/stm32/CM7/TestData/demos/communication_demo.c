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
static void refresh_crc(uint8_t *bytes)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0U; index < 156U; ++index)
    {
        crc ^= bytes[index];
        for (unsigned bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0xEDB88320U : 0U);
    }
    crc ^= UINT32_MAX;
    for (unsigned index = 0U; index < 4U; ++index)
        bytes[156U + index] = (uint8_t)(crc >> (8U * index));
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
    const size_t offsets[] = {0U, 4U, 6U, 8U, 12U, 16U, 20U, 24U, 31U, 31U, 31U, 20U};
    const uint8_t values[] = {'X', 2U, 31U, 0U, 0U, 3U, 31U, 2U, 0x7FU, 0x7FU, 0xBFU, 14U};
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
