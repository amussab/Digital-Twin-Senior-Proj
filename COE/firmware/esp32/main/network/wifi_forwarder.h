/** @file wifi_forwarder.h
 * Queue owned, validated vectors for HTTP POST in a separate FreeRTOS task.
 */
#ifndef COE_WIFI_FORWARDER_H
#define COE_WIFI_FORWARDER_H
#include "feature_packet.h"
typedef enum { FORWARD_DISABLED, FORWARD_QUEUED, FORWARD_QUEUE_FULL } ForwardStatus;
/** Initialize once. Disabled/unconfigured Wi-Fi leaves UART reception available. */
void wifi_forwarder_init(void);
/** Nonblocking copy into a 16-vector queue, called only by the receiving task.
 * Full queue rejects the new vector; HTTP failures are logged after three tries.
 * Acceptance does not mean host delivery. There is no persistent offline store.
 */
ForwardStatus wifi_forwarder_submit(const FeaturePacket *packet, uint32_t measurement_id, int64_t received_us);
#endif
