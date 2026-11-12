/** @file connection_settings.h
 * Choose local connection settings without putting credentials in shared code.
 */
#ifndef COE_CONNECTION_SETTINGS_H
#define COE_CONNECTION_SETTINGS_H
#if __has_include("connection_settings.local.h")
#include "connection_settings.local.h"
#else
#include "connection_settings.example.h"
#endif
#define COE_FEATURE_UART_BAUD 115200
#define COE_DISPLAY_FEATURES 1
#define COE_LED_GPIO 8
#define COE_FORWARD_QUEUE_LENGTH 16U
#define COE_HTTP_ATTEMPTS 3U
#define COE_HTTP_TIMEOUT_MS 3000
#define COE_WIFI_WAIT_MS 5000
#define COE_WIFI_TX_POWER_QDBM 40 /* 10 dBm, in 0.25 dBm steps (full power is 80) */
#endif
