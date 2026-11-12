/** @file status_led.h
 * Board RGB LED, independent stages:
 *   red   = the ESP just started (shown on every boot; repeated red flashes
 *           mean it keeps restarting, i.e. its 5 V supply dips)
 *   blue  = the ESP is running and its own Wi-Fi "COE-NODE" is up
 *           (does not need the STM32)
 *   green = at least one valid packet arrived from the STM32 (latched)
 * It does not indicate Wi-Fi delivery or a healthy bearing.
 */
#ifndef COE_STATUS_LED_H
#define COE_STATUS_LED_H
/** Initialize the LED and show red (just booted). */
void status_led_init(void);
/** Show blue once the node's own Wi-Fi is broadcasting (unless already green). */
void status_led_wifi_ready(void);
/** Latch green after the first valid received packet, until reset. */
void status_led_received(void);
#endif
