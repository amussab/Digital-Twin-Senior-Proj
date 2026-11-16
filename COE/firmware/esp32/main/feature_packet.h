/** @file feature_packet.h
 * Portable decoder for the STM32's COE1 feature packets.
 * Contains no ESP-IDF or STM32 dependencies; also verified by MCU test code.
 * Schema 1 (160 bytes): header + 32 features + CRC.
 * Schema 2 (204 bytes, 16 Nov 2026): schema 1's header and features, then
 * the shaft block (Keyphasor speed, PROX 1 and PROX 3 values), then CRC.
 * Schema 3 (232 bytes, LIVE since 16 Nov 2026 afternoon): schema 2 plus, before the CRC,
 * the STM32's latency (u32 us at 200, for the end-to-end latency chain), the ADC sample
 * rate (u32 mHz at 204), the window's slowest and fastest turn (2 x f32 RPM at 208) and
 * the STM32 health counters (3 x u32 at 216: ADC gaps, unsteady windows, send failures).
 * Layout: coe_firmware/CM7/App/payload/node_packet.h.
 */
#ifndef COE_RECEIVER_PACKET_H
#define COE_RECEIVER_PACKET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FEATURE_PACKET_BYTES 160U        /* schema 1 */
#define FEATURE_PACKET_SHAFT_BYTES 204U  /* schema 2 */
#define FEATURE_PACKET_LATENCY_BYTES 232U /* schema 3 */
#define FEATURE_PACKET_MAX_BYTES FEATURE_PACKET_LATENCY_BYTES
#define FEATURE_PACKET_MAX_LATENCY_US 10000000U /* STM32 latency limit, 10 s */
#define FEATURE_PACKET_UART_BAUD 115200U       /* STM32 -> ESP32 wire, 8N1 */
#define FEATURE_PACKET_MIN_RATE_MILLIHZ 1000000U   /* ADC sample rate limits, 1 kS/s */
#define FEATURE_PACKET_MAX_RATE_MILLIHZ 200000000U /* .. 200 kS/s */
#define FEATURE_PACKET_MAX_TURN_RPM 8000.0f /* limit for a single turn's speed */
/* Which Wi-Fi link the ESP32 reports the signal of (wifi_link). */
#define FEATURE_WIFI_LINK_NONE 0U
#define FEATURE_WIFI_LINK_NODE 1U    /* a laptop is on the node's own Wi-Fi COE-NODE */
#define FEATURE_WIFI_LINK_HOTSPOT 2U /* the ESP32 joined a hotspot */
#define FEATURE_PACKET_MAX_PROFILE 3U          /* 1..3; 3 = RK4 S6700 */
#define FEATURE_PACKET_VALUES 32U
#define FEATURE_PACKET_PROBES 2U          /* PROX 1, PROX 3 */
#define FEATURE_PACKET_MIN_RPM 60U
#define FEATURE_PACKET_MAX_RPM 4000U
/* shaft_flags bits (schema 2) */
#define FEATURE_SHAFT_SPEED_MEASURED 0x01U
#define FEATURE_SHAFT_PROX1_VALID 0x02U
#define FEATURE_SHAFT_PROX3_VALID 0x04U
#define FEATURE_SHAFT_PROX1_X1_VALID 0x08U
#define FEATURE_SHAFT_PROX3_X1_VALID 0x10U

/** One independently decoded window; values are in channel-major order. */
typedef struct
{
    uint32_t window_id;
    uint32_t rpm;
    uint32_t bearing_profile;
    uint32_t valid_channels;
    float features[FEATURE_PACKET_VALUES];
    /* Schema 2 and 3 only (all zero for schema 1). */
    uint32_t schema;              /* 1, 2 or 3 */
    float measured_rpm;           /* Keyphasor, mean of the window's revolutions */
    uint32_t revolutions;
    uint32_t shaft_flags;
    float probes[FEATURE_PACKET_PROBES][4]; /* gap_um, pkpk_um, x1_um_pp, x1_phase_deg */
    /* Latency chain (schema 3). stm_latency_us comes from the packet; the forwarder
     * fills the other two just before it builds the JSON (0 from the decoder). */
    uint32_t stm_latency_us;  /* STM32: window's last sample -> packet handed to its UART */
    uint32_t esp_latency_us;  /* ESP32: packet complete -> this HTTP request starts */
    uint32_t wifi_latency_us; /* previous accepted HTTP request's round trip; 0 = none yet */
    uint32_t sample_rate_millihz; /* schema 3: ADC samples per second per channel x 1000 */
    /* Schema 3, from the STM32: slowest / fastest turn in the window (0 without a speed)
     * and its health counters since LIVE started. */
    float rpm_min, rpm_max;
    uint32_t adc_gaps;          /* ADC sample gaps that broke the stream */
    uint32_t unsteady_windows;  /* windows withheld because the speed moved */
    uint32_t stm_send_failures; /* packets the STM32 could not send */
    /* Filled by the ESP32 before the JSON is built (0 from the decoder). */
    uint32_t esp_rx_rejected;   /* UART packets rejected (CRC, rules) + UART errors since ESP start */
    uint32_t esp_dropped;       /* windows not delivered: queue full or HTTP failed after retries */
    uint32_t wifi_link;         /* FEATURE_WIFI_LINK_... */
    int32_t wifi_rssi_dbm;      /* signal of that link, -127..0 dBm; 0 when no link */
} FeaturePacket;

/** Bounded storage for a fragmented byte stream; initialize with {0}. */
typedef struct
{
    uint8_t bytes[FEATURE_PACKET_MAX_BYTES];
    size_t count;
    uint32_t rejected_packets;
} FeaturePacketStream;

/** Check magic, schema, metadata, IEEE Float32 values and CRC; leave output
 * untouched on rejection. Invalid channels must contain eight zero values.
 * The schema field (bytes 4..5) gives the length: 160 bytes for schema 1,
 * 204 for schema 2, 232 for schema 3; bytes must hold that many.
 */
bool feature_packet_decode(const uint8_t *bytes, FeaturePacket *output);

/** Packet length for the schema in a header's bytes 4..7, or 0 if unknown. */
size_t feature_packet_length(const uint8_t header[8]);

/** Consume one byte; return true and set output only for a valid full packet.
 * Searches for COE1, supports arbitrary UART chunks, and recovers after a
 * corrupt/truncated frame. Unknown schemas and CRC-valid bad metadata are rejected.
 */
bool feature_packet_push(FeaturePacketStream *stream, uint8_t byte, FeaturePacket *output);

/** Discard a partial frame after UART data loss, preserving the reject count. */
void feature_packet_reset(FeaturePacketStream *stream);

#endif
