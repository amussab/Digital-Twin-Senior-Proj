/** @file status_led.c
 * Use Espressif's LED-strip/RMT driver for the nanoESP32-C6 RGB LED.
 * Called from the receiver task and the Wi-Fi event task, so writes are locked.
 */
#include "status_led.h"
#include "connection_settings.h"
#include "led_strip.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <stdbool.h>
#include <stdint.h>
static led_strip_handle_t led;
static SemaphoreHandle_t lock;
static bool received;

static void show(uint8_t red, uint8_t green, uint8_t blue)
{
    ESP_ERROR_CHECK(led_strip_set_pixel(led, 0, red, green, blue));
    ESP_ERROR_CHECK(led_strip_refresh(led));
}

/** Configure one WS2812 pixel on GPIO8 using the vendor driver; show red. */
void status_led_init(void)
{
    const led_strip_config_t config = {
        .strip_gpio_num = COE_LED_GPIO,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
    };
    const led_strip_rmt_config_t rmt = {
        .resolution_hz = 10000000,
        .flags.with_dma = false,
    };
    lock = xSemaphoreCreateMutex();
    configASSERT(lock != NULL);
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&config, &rmt, &led));
    show(16, 0, 0);
}

void status_led_wifi_ready(void)
{
    xSemaphoreTake(lock, portMAX_DELAY);
    if (!received)
        show(0, 0, 16);
    xSemaphoreGive(lock);
}

/** Set a modest green brightness once; subsequent packets need no LED transfer. */
void status_led_received(void)
{
    if (received)
        return;
    xSemaphoreTake(lock, portMAX_DELAY);
    show(0, 16, 0);
    received = true;
    xSemaphoreGive(lock);
}
