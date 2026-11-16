# ICS update, 16 Nov 2026: what changed in the node and what to change on your side

Two things changed in the node firmware (STM32 + ESP32). Your receiver code needs a small update
to accept the new packets.

## 1. Backup firmware with your training conditions

Your model was trained on the 13 Nov recordings (node firmware `639376b`). Since then the node
changed: new bearings (profile 3: ball band at 2 × BSF, ± 0.15 order), a wider envelope
(700 Hz) and B2X is now its own sensor. A model trained on 13 Nov data would see different
features.

If there is no time to retrain, flash the **backup** STM32 file
`COE/firmware/prebuilt/coe_node_stm32_LIVE_ICS_BACKUP.hex`. It makes the same kind of features
as your training data:

| | Normal (`coe_node_stm32_LIVE.hex`) | Backup (`..._ICS_BACKUP.hex`), as 13 Nov |
|---|---|---|
| `bearing_profile` in the JSON | 3 | **4** |
| Ball band (`*_bsf`) | 2 × BSF = 10.2786 × speed, ± 0.15 order | **BSF = 5.1393 × speed, ± 0.08 order** |
| FTF / BPFO / BPFI bands | 0.4518 / 4.9699 / 6.0301, ± 0.15 | the same orders, **± 0.08** |
| Envelope low-pass | 700 Hz | **350 Hz** (the old filter) |
| `B2X_*` features (16–23) | the B2X sensor | **a copy of B1Y**, as the old wiring gave |

Speed and probes are measured in both (Keyphasor, PROX 1, PROX 3). The ESP32 file is the same
for both. Data recorded with the backup is profile 4: keep it apart from profile 3 data.

## 2. End-to-end latency in every window (packet schema 3)

Spec: under 500 ms from a window's last sample to your AI result (collecting the window is not
counted). The node now sends its part in the JSON, in whole microseconds:

| Field | Part |
|---|---|
| `stm_latency_us` | STM32: DSP, features, packet |
| `uart_latency_us` | STM32 -> ESP32 wire (fixed 18924) |
| `esp_latency_us` | ESP32 queue |
| `wifi_latency_us` | Wi-Fi: the previous request's round trip (0 for the first) |
| `node_latency_us` | the sum of the four |

**End-to-end = `node_latency_us` + your time from receiving the POST to the AI result.**
`COE/inference/inference_receiver.py` already does this and prints it per window; it is also in
`GET /digital-twin` (`ai_result.end_to_end_ms`).

## 3. What to change in your code (fpr branch)

The node now sends `schema_version` **3** and, with the backup, `bearing_profile` **4**. Your
receiver currently accepts only `schema_version` 1 and `bearing_profile` 1–2, so it would answer
400 and drop every window. Change:

- Accept `schema_version` 1, 2 and 3 (your `SCHEMA_VERSION` check in the node adapter, and the
  `host_receiver.py` limits). The simplest: take `COE/inference/host_receiver.py` from `main`.
- Accept `bearing_profile` 1–4.
- Accept `rpm` 60–4000 (the node measures the speed; it is not exactly 1750 or 3600).
- Ignore extra fields you don't use (`measured_rpm`, `prox1_*`, `prox3_*`, `*_latency_us`), or
  use them. The 32 `features` are in the same order as before.
- For the latency: note `time.perf_counter()` when the POST arrives, and after `predict()`
  add `(now - arrival) * 1000` ms to `node_latency_us / 1000`.

Questions: Abdullah (COE).
