/**
 * @file node_packet.h
 * The packet LIVE sends to the ESP32: schema 2 = the schema-1 feature packet
 * (same header and the same 32 features, byte for byte) plus a shaft block
 * with the Keyphasor speed and the two proximity probes, then CRC-32.
 *
 * Offset Size Field (little-endian; f32 = IEEE binary32)
 *   0..155       as schema 1, except bytes 4..5 = schema 2
 *   156   f32    measured_rpm: mean of the Keyphasor revolutions in the window (0 if none)
 *   160   u32    revolutions: Keyphasor revolutions that ended in the window
 *   164   u32    shaft_flags: bit 0 speed measured, 1 PROX 1 valid, 2 PROX 3 valid,
 *                3 PROX 1 1X valid, 4 PROX 3 1X valid
 *   168   4xf32  PROX 1: gap_um, pkpk_um, x1_um_pp, x1_phase_deg
 *   184   4xf32  PROX 3: same
 *   200   u32    CRC-32/ISO-HDLC of bytes 0..199
 * A value that is not valid is 0. The header RPM (offset 12) is the whole-RPM
 * speed the window and bearing bands were computed for (from the Keyphasor).
 */
#ifndef COE_NODE_PACKET_H
#define COE_NODE_PACKET_H

#include "feature_payload.h"
#include "shaft_sensors.h"

#define COE_NODE_SCHEMA_VERSION 2U
#define COE_SHAFT_BLOCK_OFFSET (COE_PAYLOAD_BYTES - 4U)
#define COE_SHAFT_BLOCK_BYTES 44U
#define COE_NODE_PACKET_BYTES (COE_SHAFT_BLOCK_OFFSET + COE_SHAFT_BLOCK_BYTES + 4U)

#define COE_SHAFT_SPEED_MEASURED 0x01U
#define COE_SHAFT_PROX1_VALID 0x02U
#define COE_SHAFT_PROX3_VALID 0x04U
#define COE_SHAFT_PROX1_X1_VALID 0x08U
#define COE_SHAFT_PROX3_X1_VALID 0x10U

/** Build the schema-2 packet from a CRC-valid schema-1 packet and the window's
 * shaft values. False (output zeroed) on a bad input packet or a non-finite,
 * negative or out-of-range shaft value. */
bool node_packet_encode(const uint8_t feature_packet[COE_PAYLOAD_BYTES], const ShaftWindow *shaft,
                        uint8_t output[COE_NODE_PACKET_BYTES]);

#endif
