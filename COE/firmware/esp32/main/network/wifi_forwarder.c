/** @file wifi_forwarder.c
 * Wi-Fi station setup and bounded HTTP forwarding, independent of UART reads.
 */
#include "wifi_forwarder.h"
#include "connection_settings.h"
#include "feature_json.h"
#include "latency_packet.h"
#include "status_led.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_http_server.h"
#include "esp_system.h"
#include "nvs.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_random.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "esp_crt_bundle.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "lwip/sockets.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

typedef struct { FeaturePacket packet; uint32_t sequence, measurement_id; int64_t received_us; } ForwardJob;
static const char *tag = "coe_wifi";
static QueueHandle_t jobs;
static EventGroupHandle_t connection;
static uint64_t session_id;
static uint32_t next_sequence;
#define CONNECTED_BIT BIT0 /* joined a hotspot */
#define AP_CLIENT_BIT BIT1 /* a laptop is on the node's own Wi-Fi */

/* Host discovery: the training recorder broadcasts "COE-HOST <port>" on UDP 8001.
 * The ESP sends to the laptop that last announced itself, so no laptop IP is
 * fixed in the firmware. Until a beacon arrives, COE_HTTP_URL is used. */
#define DISCOVERY_PORT 8001
static char target_url[96] = COE_HTTP_URL;
static SemaphoreHandle_t target_lock;

static void copy_target(char *out, size_t size)
{
    xSemaphoreTake(target_lock, portMAX_DELAY);
    strlcpy(out, target_url, size);
    xSemaphoreGive(target_lock);
}

/** Listen for recorder beacons and follow the announcing laptop. */
static void discovery_task(void *argument)
{
    (void)argument;
    const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    struct sockaddr_in local = {.sin_family = AF_INET, .sin_port = htons(DISCOVERY_PORT),
                                .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (sock < 0 || bind(sock, (struct sockaddr *)&local, sizeof(local)) != 0)
    {
        ESP_LOGE(tag, "Discovery socket failed; using fixed target %s", COE_HTTP_URL);
        if (sock >= 0) close(sock);
        vTaskDelete(NULL);
        return;
    }
    char text[32];
    for (;;)
    {
        struct sockaddr_in from;
        socklen_t from_length = sizeof(from);
        const int length = recvfrom(sock, text, sizeof(text) - 1U, 0,
                                    (struct sockaddr *)&from, &from_length);
        unsigned port = 0U;
        if (length <= 0)
            continue;
        text[length] = '\0';
        if (sscanf(text, "COE-HOST %u", &port) != 1 || port == 0U || port > 65535U)
            continue;
        char address[16], url[sizeof(target_url)];
        inet_ntoa_r(from.sin_addr, address, sizeof(address));
        snprintf(url, sizeof(url), "http://%s:%u/features", address, port);
        xSemaphoreTake(target_lock, portMAX_DELAY);
        const bool changed = strcmp(url, target_url) != 0;
        if (changed)
            strlcpy(target_url, url, sizeof(target_url));
        xSemaphoreGive(target_lock);
        if (changed)
            ESP_LOGI(tag, "Recorder found: HTTP target now %s", url);
    }
}

/* The node always runs its own open Wi-Fi "COE-NODE" (192.168.4.1): a laptop that
 * joins it receives the data directly. It also joins a hotspot as a station: the
 * one saved through POST /wifi (from the training recorder page), otherwise the
 * one in connection_settings. */
#define NODE_AP_SSID "COE-NODE"
#define WIFI_NVS_NAMESPACE "coe_wifi"
#define RECONNECT_US 5000000LL /* slow retries keep the node's own Wi-Fi usable */
static esp_timer_handle_t reconnect_timer, restart_timer;
static bool station_enabled;

static bool wifi_credentials_valid(const char *ssid, const char *password)
{
    const size_t ssid_length = strlen(ssid), password_length = strlen(password);
    return ssid_length > 0U && ssid_length <= 32U && password_length <= 63U
           && (password_length == 0U || password_length >= 8U);
}

/** Saved hotspot from NVS, else the build-time default. */
static void station_credentials(char ssid[33], char password[64])
{
    strlcpy(ssid, COE_WIFI_SSID, 33);
    strlcpy(password, COE_WIFI_PASSWORD, 64);
    nvs_handle_t nvs;
    if (nvs_open(WIFI_NVS_NAMESPACE, NVS_READONLY, &nvs) != ESP_OK)
        return;
    char saved_ssid[33], saved_password[64];
    size_t ssid_size = sizeof(saved_ssid), password_size = sizeof(saved_password);
    if (nvs_get_str(nvs, "ssid", saved_ssid, &ssid_size) == ESP_OK
        && nvs_get_str(nvs, "pass", saved_password, &password_size) == ESP_OK
        && wifi_credentials_valid(saved_ssid, saved_password))
    {
        strlcpy(ssid, saved_ssid, 33);
        strlcpy(password, saved_password, 64);
    }
    nvs_close(nvs);
}

static void restart_node(void *argument)
{
    (void)argument;
    esp_restart();
}

/** POST /wifi, body "<hotspot name>\n<password>": save, reply, restart onto it. */
static esp_err_t wifi_settings_post(httpd_req_t *request)
{
    char body[128];
    const int length = request->content_len;
    if (length <= 0 || length >= (int)sizeof(body))
        return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST, "Expected name\\npassword");
    int received = 0;
    while (received < length)
    {
        const int part = httpd_req_recv(request, body + received, (size_t)(length - received));
        if (part <= 0)
            return ESP_FAIL;
        received += part;
    }
    body[length] = '\0';
    char *split = strchr(body, '\n');
    if (split == NULL)
        return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST, "Expected name\\npassword");
    *split = '\0';
    char *password = split + 1;
    password[strcspn(password, "\r\n")] = '\0';
    if (!wifi_credentials_valid(body, password))
        return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                                   "Name 1-32 characters; password empty or 8-63 characters");
    nvs_handle_t nvs;
    esp_err_t error = nvs_open(WIFI_NVS_NAMESPACE, NVS_READWRITE, &nvs);
    if (error == ESP_OK)
    {
        error = nvs_set_str(nvs, "ssid", body);
        if (error == ESP_OK)
            error = nvs_set_str(nvs, "pass", password);
        if (error == ESP_OK)
            error = nvs_commit(nvs);
        nvs_close(nvs);
    }
    if (error != ESP_OK)
        return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR, "Could not save");
    ESP_LOGI(tag, "New hotspot saved: \"%s\"; restarting", body);
    httpd_resp_sendstr(request, "saved");
    esp_timer_start_once(restart_timer, 1000000LL);
    return ESP_OK;
}

static void start_settings_server(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = NULL;
    const httpd_uri_t wifi_uri = {.uri = "/wifi", .method = HTTP_POST, .handler = wifi_settings_post};
    if (httpd_start(&server, &config) != ESP_OK
        || httpd_register_uri_handler(server, &wifi_uri) != ESP_OK)
        ESP_LOGE(tag, "Settings server failed; hotspot can only be changed by reflashing");
}

static void reconnect(void *argument)
{
    (void)argument;
    const esp_err_t error = esp_wifi_connect();
    if (error != ESP_OK)
        ESP_LOGW(tag, "Wi-Fi connect request failed: %s", esp_err_to_name(error));
}

/** Track IP availability and reconnect when the hotspot connection is lost. */
static void wifi_event(void *argument, esp_event_base_t base, int32_t id, void *data)
{
    (void)argument;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START && station_enabled)
        reconnect(NULL);
    else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED)
    {
        if (data != NULL) {
            const wifi_event_sta_disconnected_t *event = data;
            ESP_LOGW(tag, "Hotspot not connected: reason=%u; retry in 5 s", (unsigned)event->reason);
        }
        xEventGroupClearBits(connection, CONNECTED_BIT);
        esp_timer_start_once(reconnect_timer, RECONNECT_US);
    }
    else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_START)
    {
        ESP_LOGI(tag, "%s is broadcasting", NODE_AP_SSID);
        status_led_wifi_ready();
    }
    else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STACONNECTED)
    {
        ESP_LOGI(tag, "A laptop joined %s", NODE_AP_SSID);
        xEventGroupSetBits(connection, AP_CLIENT_BIT);
    }
    else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STADISCONNECTED)
    {
        wifi_sta_list_t clients;
        if (esp_wifi_ap_get_sta_list(&clients) == ESP_OK && clients.num == 0)
            xEventGroupClearBits(connection, AP_CLIENT_BIT);
    }
    else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP)
    {
        xEventGroupSetBits(connection, CONNECTED_BIT);
        const ip_event_got_ip_t *event = data;
        ESP_LOGI(tag, "Wi-Fi ready: IP=" IPSTR ", gateway=" IPSTR,
                 IP2STR(&event->ip_info.ip), IP2STR(&event->ip_info.gw));
        ESP_LOGI(tag, "Waiting for a recorder beacon on UDP %d (fallback %s)",
                 DISCOVERY_PORT, COE_HTTP_URL);
    }
}

/** POST one JSON vector; only a completed HTTP 2xx response confirms host acceptance. */
static bool post_vector(const ForwardJob *job)
{
    char body[FEATURE_JSON_BYTES];
    const size_t length = feature_packet_to_json(&job->packet, COE_DEVICE_ID,
                                                 session_id, job->sequence, body, sizeof(body));
    if (length == 0U)
        return false;
    char identity[80];
    snprintf(identity, sizeof(identity), "%s-%016" PRIx64 "-%" PRIu32,
             COE_DEVICE_ID, session_id, job->sequence);
    char url[sizeof(target_url)];
    copy_target(url, sizeof(url));
    const esp_http_client_config_t config = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = COE_HTTP_TIMEOUT_MS,
        .disable_auto_redirect = true,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL)
        return false;
    esp_err_t error = esp_http_client_set_header(client, "Content-Type", "application/json");
    if (error == ESP_OK)
        error = esp_http_client_set_header(client, "Idempotency-Key", identity);
    if (error == ESP_OK)
        error = esp_http_client_set_post_field(client, body, (int)length);
    if (error == ESP_OK)
        error = esp_http_client_perform(client);
    const int status = esp_http_client_get_status_code(client);
    const bool accepted = error == ESP_OK && status >= 200 && status < 300;
    if (!accepted)
        ESP_LOGW(tag, "POST sequence=%" PRIu32 " failed: %s, HTTP=%d",
                 job->sequence, esp_err_to_name(error), status);
    (void)esp_http_client_cleanup(client);
    return accepted;
}

/** Own network waits/retries in this task so UART reception keeps running. */
static void forwarding_task(void *argument)
{
    (void)argument;
    uint32_t forwarded = 0U, failed = 0U;
    ForwardJob job;
    for (;;)
    {
        if (xQueueReceive(jobs, &job, portMAX_DELAY) != pdTRUE)
            continue;
        const int64_t dequeued_us = esp_timer_get_time();
        bool accepted = false;
        unsigned attempts = 0U;
        for (unsigned attempt = 0U; attempt < COE_HTTP_ATTEMPTS && !accepted; ++attempt)
        {
            ++attempts;
            const EventBits_t bits = xEventGroupWaitBits(connection, CONNECTED_BIT | AP_CLIENT_BIT,
                pdFALSE, pdFALSE, pdMS_TO_TICKS(COE_WIFI_WAIT_MS));
            if ((bits & (CONNECTED_BIT | AP_CLIENT_BIT)) != 0U)
                accepted = post_vector(&job);
            if (!accepted && attempt + 1U < COE_HTTP_ATTEMPTS)
                vTaskDelay(pdMS_TO_TICKS(500));
        }
        if (accepted) {
            const int64_t accepted_us = esp_timer_get_time();
            const uint32_t queue_us = (uint32_t)(dequeued_us - job.received_us);
            const uint32_t network_us = (uint32_t)(accepted_us - dequeued_us);
            if (job.measurement_id != 0U) {
                const LatencyAck ack = {job.measurement_id, job.packet.window_id, job.packet.rpm,
                    job.packet.bearing_profile, queue_us + network_us, queue_us, network_us};
                uint8_t bytes[LATENCY_ACK_BYTES];
                latency_ack_encode(&ack, bytes);
                if (uart_write_bytes(UART_NUM_1, bytes, sizeof(bytes)) != (int)sizeof(bytes))
                    ESP_LOGW(tag, "Could not return delivery ACK for measurement=%" PRIu32, job.measurement_id);
            }
            ESP_LOGI(tag, "Host accepted sequence=%" PRIu32 ", forwarded=%" PRIu32
                     ", measurement=%" PRIu32 ", queue_ms=%.3f, network_ms=%.3f, attempts=%u",
                     job.sequence, ++forwarded, job.measurement_id, queue_us / 1000.0,
                     network_us / 1000.0, attempts);
        }
        else
            ESP_LOGW(tag, "Discarded sequence=%" PRIu32 " after retries, failed=%" PRIu32,
                     job.sequence, ++failed);
    }
}

/** Start the station and its sender task only after local settings are supplied. */
void wifi_forwarder_init(void)
{
    if (!COE_ENABLE_WIFI)
    {
        ESP_LOGI(tag, "Forwarding disabled; UART and its green LED can be tested independently");
        return;
    }
    if (!feature_device_id_is_valid(COE_DEVICE_ID)
        || (strncmp(COE_HTTP_URL, "http://", 7U) != 0 && strncmp(COE_HTTP_URL, "https://", 8U) != 0))
    {
        ESP_LOGE(tag, "Invalid connection settings; forwarding disabled");
        return;
    }
    esp_err_t error = nvs_flash_init();
    if (error == ESP_ERR_NVS_NO_FREE_PAGES || error == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        error = nvs_flash_init();
    }
    ESP_ERROR_CHECK(error);
    connection = xEventGroupCreate();
    target_lock = xSemaphoreCreateMutex();
    jobs = xQueueCreate(COE_FORWARD_QUEUE_LENGTH, sizeof(ForwardJob));
    if (connection == NULL || jobs == NULL || target_lock == NULL)
    {
        ESP_LOGE(tag, "Cannot allocate forwarding resources");
        if (jobs != NULL) vQueueDelete(jobs);
        if (connection != NULL) vEventGroupDelete(connection);
        jobs = NULL;
        return;
    }
    session_id = ((uint64_t)esp_random() << 32U) | esp_random();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    if (esp_netif_create_default_wifi_sta() == NULL || esp_netif_create_default_wifi_ap() == NULL)
        ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    const esp_timer_create_args_t reconnect_args = {.callback = reconnect, .name = "wifi_retry"};
    const esp_timer_create_args_t restart_args = {.callback = restart_node, .name = "restart"};
    ESP_ERROR_CHECK(esp_timer_create(&reconnect_args, &reconnect_timer));
    ESP_ERROR_CHECK(esp_timer_create(&restart_args, &restart_timer));
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL));

    char ssid[33], password[64];
    station_credentials(ssid, password);
    station_enabled = wifi_credentials_valid(ssid, password);
    wifi_config_t station = {0};
    memcpy(station.sta.ssid, ssid, strlen(ssid));
    memcpy(station.sta.password, password, strlen(password));
    station.sta.threshold.authmode = password[0] == '\0' ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK;
    wifi_config_t access_point = {.ap = {.ssid = NODE_AP_SSID, .ssid_len = sizeof(NODE_AP_SSID) - 1U,
                                         .channel = 1, .max_connection = 4,
                                         .authmode = WIFI_AUTH_OPEN}};
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &access_point));
    if (station_enabled)
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &station));
    ESP_ERROR_CHECK(esp_wifi_start());
    /* Lower transmit power: smaller current bursts, so the ESP stays up on the
     * shared 5 V converter (at full power it reset there). The laptop is near. */
    if (esp_wifi_set_max_tx_power(COE_WIFI_TX_POWER_QDBM) != ESP_OK)
        ESP_LOGW(tag, "Could not lower Wi-Fi transmit power");
    ESP_LOGI(tag,"Own Wi-Fi \"%s\" (open, node 192.168.4.1); hotspot \"%s\"",
             NODE_AP_SSID, station_enabled ? ssid : "(none)");
    start_settings_server();
    /* Lower latency on this USB-powered demo; higher power consumption is intentional. */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    if (xTaskCreate(discovery_task, "host_discovery", 3072, NULL, 1, NULL) != pdPASS)
        ESP_LOGW(tag, "Cannot create discovery task; using fixed target %s", COE_HTTP_URL);
    if (xTaskCreate(forwarding_task, "feature_post", 8192, NULL, 1, NULL) != pdPASS)
    {
        ESP_LOGE(tag, "Cannot create forwarding task; UART remains available");
        vQueueDelete(jobs);
        jobs = NULL;
    }
}

/** Copy the decoded vector before the receiver reuses its local packet storage. */
ForwardStatus wifi_forwarder_submit(const FeaturePacket *packet, uint32_t measurement_id, int64_t received_us)
{
    if (jobs == NULL)
        return FORWARD_DISABLED;
    if (packet == NULL || next_sequence == UINT32_MAX)
        return FORWARD_QUEUE_FULL;
    const ForwardJob job = {.packet = *packet, .sequence = ++next_sequence,
                            .measurement_id = measurement_id, .received_us = received_us};
    return xQueueSend(jobs, &job, 0) == pdTRUE ? FORWARD_QUEUED : FORWARD_QUEUE_FULL;
}
