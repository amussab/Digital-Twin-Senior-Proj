/* Host-test stand-in for payload sizes; DSP headers are unnecessary here.
 * run_host_checks.py force-includes it (-include), so it uses the real header's
 * guard and the real feature_payload.h is skipped everywhere (also from node_packet.h). */
#ifndef COE_FEATURE_PAYLOAD_H
#define COE_FEATURE_PAYLOAD_H
#include <stdbool.h>
#include <stdint.h>
#define COE_PAYLOAD_SCHEMA_VERSION 1U
#define COE_PAYLOAD_BYTES 160U
#define COE_FEATURE_COUNT 32U
#define COE_MIN_RPM 60U
#define COE_MAX_RPM 4000U
#endif
