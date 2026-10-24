# IS1 end-to-end latency measurement (UP2), 2026-10-24 21:48 UTC

Spec IS1: sensor -> classify+RUL -> dashboard < 500 ms. Tool: `tools/DemoMeasure is1` (n = 484 ONNX-active windows of 509 sent, 0 without a snapshot; payloads = AI-engine/models/up2_demo_payloads.bin, [SYNTHETIC] synth_v3 stream; contract 3b-coe = v3b + COE-adapted classifier + cross-bearing gate; payload stream is the synth_v3 replay, so classes shown are not meaningful for this latency run).
Method: t_close = sender timestamp when the COE window is complete (payload ready); then a [SIMULATION] Wi-Fi hop delay, uniform 10-60 ms (NOT measured on a real Wi-Fi link; real value needs the ESP32-C6 + AP); then UDP 127.0.0.1 -> backend (decode, feature engineering, ONNX classifier-slot + RUL-slot, SignalR broadcast) -> .NET SignalR client receive (stands in for the browser; Blazor WASM render time is NOT included). One monotonic clock. Windows are sent one at a time at 20 Hz. Marker: rpm nudged by <= 0.2 rpm to identify each window's snapshot.

## Measured legs (this PC, Windows host)
| Leg | p50 ms | p95 ms | max ms | Provenance |
|---|---|---|---|---|
| Wi-Fi hop (injected delay realised) | 33.8 | 63.9 | 65.2 | [SIMULATION] assumed 10-60 ms jitter |
| UDP + backend (decode, features, 2x ONNX, SignalR) + client receive | 4.2 | 7.3 | 9.8 | [MEASURED host] localhost; not the target server / Pi 5 |
| Sender window-close -> client receive (two rows above) | 39.6 | 68.5 | 72.4 | [MEASURED host + SIMULATION] |

## Budget including COE processing (t20 -> payload ready), from AI-engine/reports/coe_verification.json
| Stage | p50 ms | p95 ms | max ms | Provenance |
|---|---|---|---|---|
| COE t20 -> payload ready, host PC (Python/NumPy; worst of 1750/3600 rpm) | 5.3 | 6.8 | 9.0 | [MEASURED host], not STM32 |
| COE t20 -> payload ready, STM32 pessimistic (10 cyc/flop, 480 MHz M7 assumed, incl. wait for last DMA block) | - | 28.4 | 28.4 | [CALCULATED] (nominal 2 cyc/flop: 12.5 ms) |
| Per-window sum: measured legs + COE host (p50 / p95 / max) | 44.9 | 75.3 | 81.4 | [MEASURED host + SIMULATION] |
| Per-window sum: measured legs + COE STM32 pessimistic (conservative) | 68.0 | 97.0 | 100.8 | mixed incl. [CALCULATED] |

Verdict vs IS1 (< 500 ms): p95 = 97.0 ms, max = 100.8 ms (STM32 pessimistic) -> within budget for this setup: **MET (design / simulation)**; the hardware leg and the real Wi-Fi hop are still to be measured.
Not measured: real Wi-Fi latency/jitter, STM32 timing, browser (Blazor WASM) render time, final server (laptop vs Raspberry Pi 5).

## S9 re-measure (dashboard client rate), contract 3b-coe, replay source up2 at 10 Hz (50 ms re-broadcast timer)
`dotnet tools/DemoMeasure/bin/Release/net10.0/DemoMeasure.dll rate http://127.0.0.1:5081 40`
| Item | Result | Provenance |
|---|---|---|
| Client snapshot rate (40 s, 1200 snapshots) | 29.99 Hz; per-second min 30, max 30; inter-arrival p50 44.8 / p95 60.6 / max 68.3 ms; S9 >= 10 Hz: PASS | [MEASURED host] .NET SignalR client, [SYNTHETIC] payload stream; browser render not included |

## End-to-end demo runs (scripts executed, logs in this session) [SIMULATION]
`scripts/run-up2-coe-demo.ps1 -Rate 4` (default contract 3b-coe, `--hours-per-window 0.1667`): COE simulator (outer-race fault on bearing 1 only, severity ramp) -> CoePipeline -> UDP 5005 -> backend -> dashboard (HTTP 200 on :5080). `/api/state` polled every ~1.5 s:
| t | B1 class (stage) | B2 class (stage), transmitted_from_partner |
|---|---|---|
| commissioning (first ~45 windows) | commissioning | commissioning |
| window 45 | healthy (2) | healthy (2) |
| window 67 | outer_race (3) | healthy (2) |
| window 112 | outer_race (5) | healthy (3), gated = true |
| window 223 | outer_race (5) | healthy (4), gated = true |
Bearing 2 never reported a fault while only bearing 1 was faulted (0 of the polled snapshots). The gate fired from window ~112 (B2 raw argmax outer_race, 73 % raw probability; class_probs left unchanged by design).
CAVEAT (honest): the stage / RUL head was trained on synth_v3 payload features and is NOT usable on COE-pipeline features (healthy COE bearings read stage >= 2, here rising to 3-4 on the healthy bearing 2 as transmitted vibration lifts its HI). Stage, RUL and health-index displays in the COE-source demo are NOT evidence; only the fault-class output (adapted classifier + gate, see AI-engine/reports/coe_adapt_20261024T213622Z.md) is.
`scripts/run-up2-demo.ps1 -RateHz 20` (contract v3b, replay up2): dashboard serves; state loops healthy -> inner_race stage 3 -> 4 -> 5 -> 6 with bearing 2 healthy (stage 2) throughout.
Screenshots: `evidence/dashboard_up2_coe_20261025.png`, `evidence/dashboard_up2_replay_20261025.png` (headless Chrome via playwright-core).
