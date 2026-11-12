# ESP32-C6: UART reception and Wi-Fi forwarding

This is the **separate ESP-IDF firmware project** in `coe_esp32`, beside
`coe_firmware` (STM32) and `coe_pipeline` (Python reference). Open this folder
in its own editor window. Build here with ESP-IDF and flash the ESP32 through
CH343 USB. Each firmware has its own CMake configuration and build outputs.
STM32 communication verification reuses this portable packet decoder; it
does not build the ESP application. Keep the sibling folder names unchanged.

The two phases are independent:

1. **UART reception:** collect a full COE1 packet, verify CRC/schema/metadata/
   values, then latch the RGB LED **green** and display the 32 actual features.
2. **Wi-Fi forwarding:** copy the validated vector into a bounded queue; a
   separate task connects to Wi-Fi and sends **HTTP POST JSON** to your host.
   Only an HTTP 2xx response counts as host acceptance.

Green means a validated UART packet has arrived since reset. It does not mean
Wi-Fi delivery, valid sensor measurements on every channel, or healthy bearings.
It stays green until reset; failed packets cannot turn it on initially.

## Structure

```text
main/
  receiver.c                         UART setup/reads and orchestration
  feature_packet.c/.h                 Portable CRC/schema decoder
  config/connection_settings.h        Shared fixed settings/default selection
  config/connection_settings.example.h  Copy for your local credentials/URL
  indicator/status_led.c/.h           Board RGB LED, vendor LED-strip driver
  payload/feature_json.c/.h           Fixed vector -> bounded JSON
  network/wifi_forwarder.c/.h         Wi-Fi setup, owned queue and HTTP sender
  idf_component.yml                  Espressif LED-strip dependency
tests/
  host_receiver.py                    Example HTTP host and JSONL capture
  test_forwarding.py                  Actual C JSON + loopback HTTP checks
```

Functions have descriptions and one purpose. `accept_packet` is the small
orchestrator that calls the indicator, display and forwarding functions.
The addressable WS2812 RGB LED on the nanoESP32-C6 uses **GPIO8**, not a simple
GPIO on/off LED. Its timing is handled by Espressif's `led_strip` RMT driver.
The board power LED is separate and does not indicate packet acceptance.

## First test: UART only

Wi-Fi is **disabled by default**. No SSID, password or destination is required
for the first phase.

1. Wire STM32 **PD5 / D53 / CN9 pin 6 -> ESP GPIO18 / pin marked 18**.
2. Wire STM32 **GND / CN9 pin 12 -> ESP GND**.
3. Power each board by its own USB cable; do not join their 3.3-V supply pins.
4. Flash the STM32 build through ST-LINK, and this firmware through **CH343 USB**.
5. Open the ESP monitor, wait for `Ready`, then press and release STM32 **B1**.

UART1 uses GPIO18 RX and unused GPIO19 TX at **115200 8N1**. Leave the ESP board's
printed TX/RX pins alone; UART0/CH343 remains its programming/console port.
The STM32 sends twelve freshly computed replay packets per button press:
two profiles, two RPM modes and three windows each. Expect green plus twelve
CRC-valid vectors. IDs restart per test case; use profile/RPM/window to match
the existing STM32 Watch arrays. Those packet IDs are test-session identifiers.

On this computer, **ESP-IDF v6.1** and the official VS Code extension are
installed. The target is already configured as `esp32c6`. The full build and
flash on **COM6** passed, and the actual board reached the receiver-ready state.
The original project path built successfully with this SDK; no alias or second
source copy is needed. GPIO18 has an idle-high pull-up for unwired operation.
The detected 16 MB BOYA flash uses its native driver.

Open a PowerShell terminal in this folder and run:

```powershell
cd 'C:\Users\USER2\OneDrive - KFUPM\Senior Design\coe_esp32'
.\start_esp32.ps1
```

That opens an activated ESP-IDF terminal in the same folder. To watch the
already-flashed receiver:

```powershell
idf.py -p COM6 monitor
```

After changing code, build and flash from that terminal:

```powershell
idf.py build
idf.py -p COM6 flash monitor
```

Use `Ctrl+]` to exit the monitor and release COM6 before another program uses it.
Run `.\start_esp32.ps1 -OpenEditor` to open this project in its own VS Code window.
For a new SDK/board setup elsewhere, select `esp32c6` with `idf.py set-target esp32c6`.

COM6 is the detected CH343 port for this run; change it if Windows assigns another port. Component Manager fetches the pinned
Espressif `led_strip` 3.0.3 dependency during configuration. STM32's Arm GCC
cannot compile this ESP32-C6 RISC-V application. ESP-IDF 6 changed the UART pin
API; the receiver handles both signatures. Keep the UART0 console defaults.

## Second test: add your Wi-Fi and destination

Copy `main/config/connection_settings.example.h` to
`main/config/connection_settings.local.h`, then edit:

```c
#define COE_ENABLE_WIFI 1
#define COE_WIFI_SSID "your-network-name"
#define COE_WIFI_PASSWORD "your-password"
#define COE_HTTP_URL "http://192.168.1.100:8000/features"
#define COE_DEVICE_ID "coe-node-01"
```

The **URL** contains the host IP/name, port and receiving API path. These must
match the receiving program; an IP alone does not define an HTTP endpoint.
Use a different device ID for each node (1-32 letters/digits/hyphens/underscores).
SSID may be up to 32 bytes; ordinary WPA2 passwords are 8-63 bytes (an empty
password selects an open network). Credentials are never logged by this code;
the local header is ignored by Git. Rebuild/reflash after changes. Leave
`COE_ENABLE_WIFI 0` whenever you want to isolate UART testing.

HTTP and HTTPS URLs are supported. HTTPS uses Espressif's certificate bundle;
custom/self-signed host certificates require a corresponding trust setup.
The initial test host below uses HTTP on your local network. There is no server
authentication/token in this example; add the receiving platform's required
authentication when its API is known. Configure the Wi-Fi credentials locally.

## Example host for your colleague

No Python packages are required. On the computer that will receive the vector:

```powershell
python tests/host_receiver.py --bind 0.0.0.0 --port 8000
```

Set COE_HTTP_URL to that computer's **LAN IP**, e.g.
`http://192.168.1.100:8000/features`. `127.0.0.1` would refer to the ESP itself.
Both devices need network reachability, and the host must allow inbound TCP
on the chosen port. The test host validates each message, appends it to
`tests/received_features.jsonl`, then returns HTTP 200. Change `--output` if
you want another capture location. This example host contains no AI model.

Each JSON object contains:

| Field | Meaning |
| --- | --- |
| schema_version | 1 |
| device_id | Configured node identifier |
| session_id | 16-hex-digit identifier generated at ESP boot |
| sequence | Increasing delivery number within that ESP session |
| window_id | Original STM32 window identifier |
| rpm | 1750 or 3600 |
| bearing_profile | 1 = provisional project; 2 = CWRU |
| valid_channels | Four channel-valid bits |
| acceleration_unit | g (kurtosis/crest factor remain dimensionless) |
| features | 32 ordered numeric values, unchanged Float32 values |

The original binary packet is decoded before forwarding; its CRC is verified
on ESP. The HTTP payload is JSON, so it does not retain the original binary CRC.
The feature order and validity rules remain as defined in the
[STM32 firmware guide, section 6](../coe_firmware/README.md#6-feature-vector-and-packet).
The host can use `message.features` as one matrix row. Each channel contributes
RMS, Pearson kurtosis, crest factor, envelope RMS, FTF, BSF, BPFO and BPFI band
RMS; channels are B1X, B1Y, B2X, B2Y.

## Buffering and failures

The queue owns copies of **16 pending vectors** plus the worker's active one.
HTTP waits/retries cannot overwrite the UART's local packet storage. A full
queue rejects a new vector and logs a warning; reception/green indication
continues. The sender tries each queued vector up to three times, waiting up
to 5 s for Wi-Fi and using a 3-s HTTP timeout per attempt. Failed vectors are
then discarded with a log message. There is no persistent offline backlog.

Retries use the same `Idempotency-Key` header composed of device/session/
sequence. The host must deduplicate that identity: a request might be saved
even if its response is lost. The example host keeps the last 10,000 identities
in memory, acknowledges identical retries and rejects conflicting duplicates.
Its deduplication history resets on restart. A deployment host needs durable
handling if guaranteed delivery/deduplication is required.

Observe `Host accepted ...` separately from UART reception. A 2xx is the host's
acknowledgment of the HTTP request, not an acknowledgment sent back to STM32.
The original 333-ms specification concerns MCU processing/payload availability;
Wi-Fi/HTTP delivery time is a separate measurement.

## Validation status

The STM32 sender has a passing Arm build and passing host software checks, but
physical UART operation is **not verified until you wire and run both boards**.
The decoder already passed 22 cases plus 480 stream recovery cases. The new
C JSON encoder and example HTTP host have reproducible checks:

```text
python3 tests/test_forwarding.py
```

Run those in Linux/WSL with GCC; they do not need ESP-IDF. The test runs only a
loopback HTTP host and uses labeled synthetic values, not claimed board data.
Read `tests/implementation_report.json` for the original implementation results. A host
serializer test is not an ESP firmware build or physical Wi-Fi/LED validation.

Sources: [MuseLab board hardware](https://github.com/wuxx/nanoESP32-C6),
[Espressif LED driver](https://components.espressif.com/components/espressif/led_strip/versions/3.0.3/readme),
[Espressif HTTP client](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/api-reference/protocols/esp_http_client.html).

Current software results: **5 host test groups passed**, including **3,200
Float32 round-trip values**, HTTP acceptance/retry/error cases and **2,049
buffer capacities checked under address/undefined-behavior sanitizers**.
All existing DSP sources/fixtures remain unchanged. The original report
records the pre-relocation ELF; rebuilding updates debug source paths.
The ESP-IDF build, physical flash and UART-ready startup now passed.
See [actual board build/startup results](tests/board_build_report.json) and
[captured startup log](tests/board_startup.log). STM32 feature reception,
green LED response to a valid packet and Wi-Fi/HTTP delivery remain pending.
The earlier implementation and separation reports describe their original runs.


## Version history and layer timing

The working hotspot baseline is on local Git `main`, tagged
`working-hotspot-2026-10-05`. Timing development is on `codex/layer-latency` in
both firmware repositories. Keep both on the same branch and rebuild before
flashing. Private hotspot settings are in ignored
`main/config/connection_settings.local.h`.

See the [STM32 firmware guide](../coe_firmware/README.md) for wiring
(section 3.3), the timing run and its Watch list (section 7), and the branches
used for recovery (section 10).
