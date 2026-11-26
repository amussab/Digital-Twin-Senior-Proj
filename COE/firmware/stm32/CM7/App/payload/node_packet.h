/**
 * @file node_packet.h
 * The packet LIVE sends to the ESP32: schema 3 = the schema-1 feature packet
 * (same header and the same 32 features, byte for byte) plus a shaft block
 * with the Keyphasor speed and the two proximity probes, the STM32's own
 * latency, then CRC-32. (Schema 2, 16 Nov 2026, was the same without the latency.)
 *
 * Offset Size Field (little-endian; f32 = IEEE binary32)
 *   0..155       as schema 1, except bytes 4..5 = schema 2
 *   156   f32    measured_rpm: mean of the Keyphasor revolutions in the window (0 if none)
 *   160   u32    revolutions: Keyphasor revolutions that ended in the window
 *   164   u32    shaft_flags: bit 0 speed measured, 1 PROX 1 valid, 2 PROX 3 valid,
 *                3 PROX 1 1X valid, 4 PROX 3 1X valid
 *   168   4xf32  PROX 1: gap_um, pkpk_um, x1_um_pp, x1_phase_deg
 *   184   4xf32  PROX 3: same
 *   200   u32    stm_latency_us: from the window's last sample to this packet being
 *                handed to the UART (1..10,000,000 us); part of the end-to-end latency
 *   204   u32    CRC-32/ISO-HDLC of bytes 0..203
 * A value that is not valid is 0. The header RPM (offset 12) is the whole-RPM
 * speed the window and bearing bands were computed for (from the Keyphasor).
 */
#ifndef COE_NODE_PACKET_H
#define COE_NODE_PACKET_H

#include "feature_payload.h"
#include "shaft_sensors.h"

#define COE_NODE_SCHEMA_VERSION 3U
#define COE_NODE_MAX_LATENCY_US 10000000U
#define COE_NODE_MIN_RATE_MILLIHZ 1000000U   /* sample rate limits: 1 kS/s */
#define COE_NODE_MAX_RATE_MILLIHZ 200000000U /* .. 200 kS/s (AD7606 maximum) */
#define COE_SHAFT_BLOCK_OFFSET (COE_PAYLOAD_BYTES - 4U)
#define COE_SHAFT_BLOCK_BYTES 44U
/* shaft block, STM32 latency (u32 us), sample rate (u32 mHz), speed min/max (2 x f32),
 * node health (3 x u32), CRC */
#define COE_NODE_PACKET_BYTES (COE_SHAFT_BLOCK_OFFSET + COE_SHAFT_BLOCK_BYTES + 4U + 4U + 8U + 12U + 4U)
#define COE_NODE_MAX_TURN_RPM (2.0f * (float)COE_MAX_RPM) /* limit for a single turn's speed */

/** STM32 health counters since LIVE started (B1), sent with every window. */
typedef struct
{
    uint32_t adc_gaps;         /* ADC sample gaps that broke the stream (window restarted) */
    uint32_t unsteady_windows; /* windows withheld because the speed moved */
    uint32_t send_failures;    /* packets not sent: encoding refused or UART queue full */
} NodeHealth;

#define COE_SHAFT_SPEED_MEASURED 0x01U
#define COE_SHAFT_PROX1_VALID 0x02U
#define COE_SHAFT_PROX3_VALID 0x04U
#define COE_SHAFT_PROX1_X1_VALID 0x08U
#define COE_SHAFT_PROX3_X1_VALID 0x10U

/** Build the schema-3 packet from a CRC-valid schema-1 packet, the window's shaft
 * values (incl. the slowest/fastest turn), the STM32 latency, the ADC sample rate and
 * the health counters. False (output zeroed) on a bad input packet, a non-finite,
 * negative or out-of-range shaft value (speed: 0 < min <= max <= COE_NODE_MAX_TURN_RPM),
 * a latency outside 1..COE_NODE_MAX_LATENCY_US, a sample rate outside
 * COE_NODE_MIN/MAX_RATE_MILLIHZ, or no health counters. */
bool node_packet_encode(const uint8_t feature_packet[COE_PAYLOAD_BYTES], const ShaftWindow *shaft,
                        uint32_t stm_latency_us, uint32_t sample_rate_millihz,
                        const NodeHealth *health, uint8_t output[COE_NODE_PACKET_BYTES]);

#endif
