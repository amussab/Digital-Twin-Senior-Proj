# IS1 end-to-end latency measurement (UP2), 2026-10-24 13:30 UTC

Spec IS1: sensor -> classify+RUL -> dashboard < 500 ms. Tool: `tools/DemoMeasure is1` (n = 484 ONNX-active windows of 509 sent, 0 without a snapshot; payloads = AI-engine/models/up2_demo_payloads.bin, [SYNTHETIC] synth_v3 stream; contract v3b incl. post-process).
Method: t_close = sender timestamp when the COE window is complete (payload ready); then a [SIMULATION] Wi-Fi hop delay, uniform 10-60 ms (NOT measured on a real Wi-Fi link; real value needs the ESP32-C6 + AP); then UDP 127.0.0.1 -> backend (decode, feature engineering, ONNX classifier-slot + RUL-slot, SignalR broadcast) -> .NET SignalR client receive (stands in for the browser; Blazor WASM render time is NOT included). One monotonic clock. Windows are sent one at a time at 20 Hz. Marker: rpm nudged by <= 0.2 rpm to identify each window's snapshot.

## Measured legs (this PC, Windows host)
| Leg | p50 ms | p95 ms | max ms | Provenance |
|---|---|---|---|---|
| Wi-Fi hop (injected delay realised) | 33.6 | 63.7 | 65.5 | [SIMULATION] assumed 10-60 ms jitter |
| UDP + backend (decode, features, 2x ONNX, SignalR) + client receive | 4.7 | 6.0 | 16.2 | [MEASURED host] localhost; not the target server / Pi 5 |
| Sender window-close -> client receive (two rows above) | 39.9 | 68.7 | 80.0 | [MEASURED host + SIMULATION] |

## Budget including COE processing (t20 -> payload ready), from AI-engine/reports/coe_verification.json
| Stage | p50 ms | p95 ms | max ms | Provenance |
|---|---|---|---|---|
| COE t20 -> payload ready, host PC (Python/NumPy; worst of 1750/3600 rpm) | 5.3 | 6.8 | 9.0 | [MEASURED host], not STM32 |
| COE t20 -> payload ready, STM32 pessimistic (10 cyc/flop, 480 MHz M7 assumed, incl. wait for last DMA block) | - | 28.4 | 28.4 | [CALCULATED] (nominal 2 cyc/flop: 12.5 ms) |
| Per-window sum: measured legs + COE host (p50 / p95 / max) | 45.2 | 75.5 | 89.0 | [MEASURED host + SIMULATION] |
| Per-window sum: measured legs + COE STM32 pessimistic (conservative) | 68.3 | 97.1 | 108.4 | mixed incl. [CALCULATED] |

Verdict vs IS1 (< 500 ms): p95 = 97.1 ms, max = 108.4 ms (STM32 pessimistic) -> within budget for this setup: **MET (design / simulation)**; the hardware leg and the real Wi-Fi hop are still to be measured.
Not measured: real Wi-Fi latency/jitter, STM32 timing, browser (Blazor WASM) render time, final server (laptop vs Raspberry Pi 5).

## S9 re-measure (dashboard client rate) with the v3b contract and C# post-process active
`dotnet tools/DemoMeasure/bin/Release/net10.0/DemoMeasure.dll rate http://127.0.0.1:5081 40` against backend `--Backend:ModelContractPath=AI-engine/models/model_contract_v3b.json --Backend:Replay:Source=up2 --Backend:Replay:RateHz=10` (50 ms re-broadcast timer).
| Metric | Value | Verdict | Provenance |
|---|---|---|---|
| Client snapshot rate (40 s, 1201 snapshots) | 30.02 Hz; per-second min 29, max 31; inter-arrival p50 44.2 / p95 61.7 / max 65.5 ms | S9 >= 10 Hz: PASS | [MEASURED host] .NET SignalR client on localhost, [SYNTHETIC] payload stream; Blazor browser render not included |
| Backend per-stage (/api/metrics, 510 payloads) | inference p50 3.7 / p95 5.2 ms (max 54.8 cold); TFT-slot ONNX p95 1.4 ms; RUL-slot ONNX p95 3.2 ms; ingest->broadcast p95 5.6 ms | S8 < 200 ms: PASS | [MEASURED host] |

Browser check: headless Chrome (playwright-core) loaded the published Blazor dashboard from the v3b backend; it showed Live, machine label "[SYNTHETIC / SIMULATED SOURCE]", Health stage n/6 (see evidence/dashboard_up2_20261024.png). Screenshot = single frame, Rank 4; the live run is the Rank 9 evidence.
