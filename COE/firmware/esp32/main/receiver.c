/** @file receiver.c
 * Receive and display CRC-validated STM32 feature vectors on ESP32-C6 UART1.
 * Valid packets turn the LED green; a separate task forwards owned copies via HTTP.
 * UART0 remains the USB/CH343 programming and console interface.
 */
#include "feature_packet.h"
#include "connection_settings.h"
#include "latency_packet.h"
#include "esp_timer.h"
#include "status_led.h"
#include "esp_system.h"
#include "wifi_forwarder.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include <inttypes.h>
#include <stdio.h>

static const char *tag = "coe_receiver";
static QueueHandle_t uart_events;
static FeatureTimingStream stream;
static uint32_t received_packets;
static uint32_t uart_errors;

/** Configure a separate data UART, leaving the board's printed TX/RX unused. */
static void configure_feature_uart(void)
{
    const uart_config_t config = {
        .baud_rate = COE_FEATURE_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_1, &config));
    /* IDF 6 added DTR/DSR arguments. Both versions leave flow-control pins unused. */
#if ESP_IDF_VERSION_MAJOR >= 6
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_1, 19, 18, UART_PIN_NO_CHANGE,
                                UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
#else
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_1, 19, 18, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
#endif
    /* UART is idle-high; keep RX stable when the STM32 wire is absent. */
    ESP_ERROR_CHECK(gpio_set_pull_mode(GPIO_NUM_18, GPIO_PULLUP_ONLY));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_1, 2048, 0, 16, &uart_events, 0));
}

/** Print the actual received metadata and all 32 Float32 values as one JSON line. */
static void display_packet(const FeaturePacket *packet)
{
    ++received_packets;
    if (!COE_DISPLAY_FEATURES) {
        ESP_LOGI(tag, "UART accepted packet=%" PRIu32 ", RPM=%" PRIu32 ", profile=%" PRIu32 ", window=%" PRIu32 ", features=32",
                 received_packets, packet->rpm, packet->bearing_profile, packet->window_id);
        if (packet->schema == 3U)
        {
            ESP_LOGI(tag, "  STM32 latency %.2f ms, sample rate %.3f S/s, turns %.1f..%.1f RPM",
                     packet->stm_latency_us / 1000.0, packet->sample_rate_millihz / 1000.0,
                     (double)packet->rpm_min, (double)packet->rpm_max);
            ESP_LOGI(tag, "  STM32 health: ADC gaps %" PRIu32 ", unsteady windows %" PRIu32
                     ", send failures %" PRIu32, packet->adc_gaps, packet->unsteady_windows,
                     packet->stm_send_failures);
        }
        if (packet->schema >= 2U)
            ESP_LOGI(tag, "  Keyphasor %.2f RPM (%" PRIu32 " turns), PROX1 gap %.1f um 1X %.1f um pp @ %.1f deg,"
                     " PROX3 gap %.1f um 1X %.1f um pp @ %.1f deg, flags 0x%02" PRIx32,
                     (double)packet->measured_rpm, packet->revolutions,
                     (double)packet->probes[0][0], (double)packet->probes[0][2], (double)packet->probes[0][3],
                     (double)packet->probes[1][0], (double)packet->probes[1][2], (double)packet->probes[1][3],
                     packet->shaft_flags);
        return;
    }
    printf("{\"received_packets\":%" PRIu32 ",\"crc_valid\":true,\"window_id\":%" PRIu32
           ",\"rpm\":%" PRIu32 ",\"bearing_profile\":%" PRIu32
           ",\"valid_channels\":%" PRIu32 ",\"acceleration_unit\":\"g\",\"features\":[",
           received_packets, packet->window_id, packet->rpm,
           packet->bearing_profile, packet->valid_channels);
    for (size_t index = 0U; index < FEATURE_PACKET_VALUES; ++index)
        printf(index == 0U ? "%.9g" : ",%.9g", (double)packet->features[index]);
    printf("]");
    if (packet->schema == 3U)
        printf(",\"stm_latency_us\":%" PRIu32 ",\"sample_rate_millihz\":%" PRIu32,
               packet->stm_latency_us, packet->sample_rate_millihz);
    if (packet->schema >= 2U) /* prox: gap_um, pkpk_um, x1_um_pp, x1_phase_deg */
        printf(",\"measured_rpm\":%.9g,\"revolutions\":%" PRIu32 ",\"shaft_flags\":%" PRIu32
               ",\"prox1\":[%.9g,%.9g,%.9g,%.9g],\"prox3\":[%.9g,%.9g,%.9g,%.9g]",
               (double)packet->measured_rpm, packet->revolutions, packet->shaft_flags,
               (double)packet->probes[0][0], (double)packet->probes[0][1], (double)packet->probes[0][2],
               (double)packet->probes[0][3], (double)packet->probes[1][0], (double)packet->probes[1][1],
               (double)packet->probes[1][2], (double)packet->probes[1][3]);
    printf("}\n");
    fflush(stdout);
}

/** Coordinate the three actions for an already CRC/schema-validated packet. */
static void accept_packet(const FeaturePacket *packet, uint32_t measurement)
{
    const int64_t received_us = esp_timer_get_time();
    FeaturePacket forwarded = *packet;
    forwarded.esp_rx_rejected = stream.features.rejected_packets + uart_errors; /* node health */
    if (wifi_forwarder_submit(&forwarded, measurement, received_us) == FORWARD_QUEUE_FULL)
        ESP_LOGW(tag, "Forwarding queue full; UART packet received but not forwarded");
    status_led_received();
    display_packet(packet);
}

/** Consume the indicated UART bytes in bounded chunks and retain partial packets. */
static void read_feature_bytes(size_t remaining)
{
    uint8_t bytes[128];
    FeaturePacket packet;
    while (remaining != 0U)
    {
        const size_t wanted = remaining < sizeof(bytes) ? remaining : sizeof(bytes);
        const int count = uart_read_bytes(UART_NUM_1, bytes, wanted, pdMS_TO_TICKS(20));
        if (count <= 0)
        {
            ESP_LOGW(tag, "UART read timed out; discarding partial packet");
            ESP_ERROR_CHECK(uart_flush_input(UART_NUM_1));
            xQueueReset(uart_events);
            latency_feature_reset(&stream);
            ++uart_errors;
            return;
        }
        remaining -= (size_t)count;
        for (int index = 0; index < count; ++index) {
            uint32_t measurement;
            if (latency_feature_push(&stream, bytes[index], &packet, &measurement))
                accept_packet(&packet, measurement);
        }
    }
}

/** Run one receiving task; errors are visible and corrupt packets never reach output. */
void app_main(void)
{
    /* 9 = brownout: the 5 V supply dipped (see status_led.h, red flashes). */
    ESP_LOGI(tag, "Boot: reset reason %d", (int)esp_reset_reason());
    status_led_init();
    wifi_forwarder_init();
    configure_feature_uart();
    ESP_LOGI(tag, "Ready: UART1 RX=GPIO18 TX=GPIO19, %d 8N1; press STM32 B1 to replay", COE_FEATURE_UART_BAUD);
    uart_event_t event;
    for (;;)
    {
        if (xQueueReceive(uart_events, &event, portMAX_DELAY) != pdTRUE)
            continue;
        if (event.type == UART_DATA)
        {
            const uint32_t rejected_before = stream.features.rejected_packets;
            read_feature_bytes(event.size);
            if (stream.features.rejected_packets != rejected_before)
                ESP_LOGW(tag, "Rejected packets=%" PRIu32, stream.features.rejected_packets);
        }
        else if (event.type == UART_FIFO_OVF || event.type == UART_BUFFER_FULL
                 || event.type == UART_PARITY_ERR || event.type == UART_FRAME_ERR || event.type == UART_BREAK)
        {
            ++uart_errors;
            ESP_LOGW(tag, "UART error=%d, total=%" PRIu32 "; discarding partial data",
                     (int)event.type, uart_errors);
            ESP_ERROR_CHECK(uart_flush_input(UART_NUM_1));
            xQueueReset(uart_events);
            latency_feature_reset(&stream);
        }
    }
}
