# ICS update, 16 Nov 2026: what the node sends now

The node (STM32 + ESP32) sends **one JSON message per window** with everything in it: the 32
features for the AI model, plus the machine speed, both proximity probes, the latency, the sampling
rate and the node's health. One message, one place to read from: the AI model and the digital twin
take what they need from the same message.

It arrives as `POST /features` on your laptop (port 8000, laptop on the Wi-Fi **COE-NODE**), about
every 0.69 s at 1750 RPM and every 0.33 s at 3600 RPM. `schema_version` is now **3**.

## Example message (one window)

```json
{"schema_version": 3, "device_id": "coe-node", "session_id": "1a2b3c4d5e6f7081", "sequence": 118,
 "window_id": 118, "rpm": 1750, "bearing_profile": 3, "valid_channels": 15, "acceleration_unit": "g",
 "features": [0.0123, 3.02, 3.9, "... 32 numbers: B1X rms .. B2Y bpfi"],

 "measured_rpm": 1749.6, "revolutions": 20, "shaft_flags": 31,
 "rpm_min": 1749.1, "rpm_max": 1750.2,
 "prox1_gap_um": 1226.7, "prox1_pkpk_um": 22.1, "prox1_x1_um_pp": 14.5, "prox1_x1_phase_deg": 105.0,
 "prox3_gap_um": 1024.5, "prox3_pkpk_um": 25.0, "prox3_x1_um_pp": 17.6, "prox3_x1_phase_deg": 104.0,

 "sample_rate_hz": 29998.5,

 "stm_latency_us": 4200, "uart_latency_us": 21007, "esp_latency_us": 150, "wifi_latency_us": 35000,
 "node_latency_us": 60357,

 "adc_gaps": 0, "unsteady_windows": 0, "stm_send_failures": 0, "esp_rx_rejected": 0, "esp_dropped": 0,
 "wifi_link": "node", "wifi_rssi_dbm": -52}
```

## The fields

| Group | Fields | Meaning |
|---|---|---|
| AI model | `features` (32), `valid_channels` | Same order as before: B1X, B1Y, B2X, B2Y × rms, kurtosis, crest, envelope_rms, ftf, bsf, bpfo, bpfi. `bearing_profile` 3 = RK4 S6700 |
| Speed | `measured_rpm`, `rpm`, `revolutions`, `rpm_min`, `rpm_max` | Speed from the Keyphasor (ADC channel 7): exact mean, whole RPM, whole turns in the window, slowest and fastest turn |
| Proximity probes | `prox1_*`, `prox3_*` (ADC channels 5, 6) | Per probe: `gap_um` (distance probe-shaft), `pkpk_um` (vibration pk-pk), `x1_um_pp` and `x1_phase_deg` (once-per-turn part and where in the turn) |
| Valid flags | `shaft_flags` | bit 0 speed measured, 1 PROX 1 valid, 2 PROX 3 valid, 3 PROX 1 1X valid, 4 PROX 3 1X valid. A value whose bit is clear is 0 |
| Sampling | `sample_rate_hz` | ADC samples per second per channel (29998.5) |
| Latency | `stm_latency_us`, `uart_latency_us`, `esp_latency_us`, `wifi_latency_us`, `node_latency_us` | The node's part of the end-to-end latency (see below) |
| Node health | `adc_gaps`, `unsteady_windows`, `stm_send_failures`, `esp_rx_rejected`, `esp_dropped` | Problems counted since the node started; all 0 = nothing lost |
| Wi-Fi | `wifi_link`, `wifi_rssi_dbm` | `node` = laptop on COE-NODE; signal in dBm (-60 or better good, below -75 weak) |

## End-to-end latency (spec: under 500 ms)

From a window's last sample to your AI result (collecting the window is not counted):

**end-to-end = `node_latency_us` / 1000 + your time from receiving the POST to the AI result (ms).**

Note `time.perf_counter()` when the POST arrives and again after `predict()`. Our
`COE/inference/inference_receiver.py` already does this and prints it per window; it is also in
`GET /digital-twin` (`ai_result.end_to_end_ms`).

## What to check in your code

- Accept `schema_version` 1, 2 and 3 (the simplest: take `COE/inference/host_receiver.py` from `main`).
- Accept `bearing_profile` 1–3 and `rpm` 60–4000 (the node measures the speed; it is not exactly
  1750 or 3600).
- Read the extra fields you need from the same message, and ignore the rest.

Questions: Abdullah (COE).
