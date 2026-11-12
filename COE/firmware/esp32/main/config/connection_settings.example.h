/** Default connection settings, used when connection_settings.local.h does not exist.
 * As is, the node runs only its own open Wi-Fi "COE-NODE" (what the ICS team uses).
 * To also join a phone hotspot, copy this file to connection_settings.local.h
 * (git ignores it, so passwords stay private) and fill in SSID and password.
 */
#define COE_ENABLE_WIFI 1
#define COE_WIFI_SSID ""     /* optional hotspot; empty = COE-NODE only */
#define COE_WIFI_PASSWORD "" /* empty, or 8-63 characters */
#define COE_HTTP_URL "http://192.168.4.2:8000/features" /* fallback until a receiver announces itself */
#define COE_DEVICE_ID "coe-node-01"
