idf_component_register(
    SRCS "receiver.c" "feature_packet.c" "indicator/status_led.c"
         "payload/feature_json.c" "network/wifi_forwarder.c" "transport/latency_packet.c"
    INCLUDE_DIRS "." "config" "indicator" "payload" "network" "transport"
    REQUIRES esp_driver_uart esp_driver_gpio esp_wifi esp_event esp_netif nvs_flash
             esp_http_client esp_http_server mbedtls esp_hw_support esp_timer)
