# IS1 end-to-end latency measurement (UP2), 2026-10-24 06:26 UTC

Spec IS1: sensor -> classify+RUL -> dashboard < 500 ms. Tool: `tools/DemoMeasure is1` (n = 484 ONNX-active windows of 509 sent, 0 without a snapshot; payloads = AI-engine/models/up2_demo_payloads.bin, [SYNTHETIC] synth_v3 stream; contract v3).
Method: t_close = sender timestamp when the COE window is complete (payload ready); then a [SIMULATION] Wi-Fi hop delay, uniform 10-60 ms (NOT measured on a real Wi-Fi link; real value needs the ESP32-C6 + AP); then UDP 127.0.0.1 -> backend (decode, feature engineering, ONNX classifier-slot + RUL-slot, SignalR broadcast) -> .NET SignalR client receive (stands in for the browser; Blazor WASM render time is NOT included). One monotonic clock. Windows are sent one at a time at 20 Hz. Marker: rpm nudged by <= 0.2 rpm to identify each window's snapshot.

## Measured legs (this PC, Windows host)
| Leg | p50 ms | p95 ms | max ms | Provenance |
|---|---|---|---|---|
| Wi-Fi hop (injected delay realised) | 43.4 | 63.7 | 74.4 | [SIMULATION] assumed 10-60 ms jitter |
| UDP + backend (decode, features, 2x ONNX, SignalR) + client receive | 4.6 | 16.5 | 24.8 | [MEASURED host] localhost; not the target server / Pi 5 |
| Sender window-close -> client receive (two rows above) | 50.0 | 73.3 | 92.5 | [MEASURED host + SIMULATION] |

## Budget including COE processing (t20 -> payload ready), from AI-engine/reports/coe_verification.json
| Stage | p50 ms | p95 ms | max ms | Provenance |
|---|---|---|---|---|
| COE t20 -> payload ready, host PC (Python/NumPy; worst of 1750/3600 rpm) | 5.3 | 6.8 | 9.0 | [MEASURED host], not STM32 |
| COE t20 -> payload ready, STM32 pessimistic (10 cyc/flop, 480 MHz M7 assumed, incl. wait for last DMA block) | - | 28.4 | 28.4 | [CALCULATED] (nominal 2 cyc/flop: 12.5 ms) |
| Per-window sum: measured legs + COE host (p50 / p95 / max) | 55.4 | 80.1 | 101.5 | [MEASURED host + SIMULATION] |
| Per-window sum: measured legs + COE STM32 pessimistic (conservative) | 78.5 | 101.7 | 120.9 | mixed incl. [CALCULATED] |

Verdict vs IS1 (< 500 ms): p95 = 101.7 ms, max = 120.9 ms (STM32 pessimistic) -> within budget for this setup: **MET (design / simulation)**; the hardware leg and the real Wi-Fi hop are still to be measured.
Not measured: real Wi-Fi latency/jitter, STM32 timing, browser (Blazor WASM) render time, final server (laptop vs Raspberry Pi 5).

Note: the realised Wi-Fi delay exceeds the nominal 10-60 ms by up to ~14 ms because of OS sleep granularity in the tool; it is a conservative (larger) simulated hop.
Backend `/api/metrics` also showed a one-off cold-path max of ~534 ms on the very first broadcast after process start (SignalR/JIT warm-up, not a steady-state value); the 484 ONNX-active windows above start after 25 warm-up windows and were measured on an already-warm process.

## S9 re-measure (dashboard client rate) with v3 models running
`dotnet tools/DemoMeasure/bin/Release/net10.0/DemoMeasure.dll rate http://127.0.0.1:5081 40` against `run-up2-demo` equivalent (`--Backend:Replay:Source=up2`, v3 contract, 10 Hz payloads, 50 ms re-broadcast timer):
| Metric | Value | Verdict | Provenance |
|---|---|---|---|
| Client snapshot rate (40 s, 1207 snapshots) | 30.17 Hz; per-second min 28, max 36; inter-arrival p50 40.7 / p95 60.8 / max 76.4 ms | S9 >= 10 Hz: PASS | [MEASURED host], SignalR .NET client on localhost, [SYNTHETIC] payload stream; the Blazor browser render is not included |
| Backend per-stage (from /api/metrics) | inference p50 9.0 / p95 16.1 ms; TFT-slot ONNX p95 7.5 ms; RUL-slot ONNX p95 14.5 ms; ingest->broadcast p95 16.9 ms | S8 < 200 ms: PASS | [MEASURED host] |

## COE simulator -> UDP -> backend live run (`scripts/coe_fault_scenario.py`, 160 windows at 4/s)
Raw simulated samples -> COE pipeline -> 152-byte payload -> UDP :5005 -> v3 backend. Via `/api/state`: commissioning (22 windows) -> Healthy (63 % conf.) -> Outer-race fault (97-100 %) with HI rising 7.8 -> 83.6 %. [SIMULATION] + [SYNTHETIC] models.
Findings, stated plainly: (1) the simulator faults bearing 1 only, yet bearing 2 is also classified outer_race (the simulator transmits 34 % of the impacts to bearing 2, and the classifier was trained on synth_v3 payload-level features, not COE-pipeline output: a domain gap that the COE-through-generator step of the plan would close). (2) RUL is ~0.003-0.05 h because COE windows are contiguous (0.69 s apart in t20) so the cadence-based RUL scale is seconds, not hours; RUL is not meaningful for this scenario (meaningful RUL needs the t20 time base of a degradation run, e.g. the up2 replay stream).
