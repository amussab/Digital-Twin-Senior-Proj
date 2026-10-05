# Integration measurements — v2 models in the loop (2026-10-05, ~05:00 local)

[MEASURED loopback on the dev laptop (Windows, Release build); real ONNX models exported from
AI-engine a78a828, which are MEASURED-on-public-data models, not rig-trained. Replay stream =
`AI-engine/models/demo_payloads.bin` (held-out XJTU-SY Bearing2_5 run-to-failure as bearing 1,
the healthy phase of Bearing3_4 as bearing 2).]

| Item | Spec | Measured | Verdict | How to reproduce |
|---|---|---|---|---|
| Snapshot rate received by a SignalR client | S9 ≥10 Hz | 901 snapshots / 30 s = **30 Hz** | PASS | `dotnet run --project tools/DemoMeasure -- rate http://localhost:5080 30` with replay on |
| UDP send → client snapshot, all windows (n=339) | IS1 <500 ms | p50 25.5 ms, **p95 31.8 ms**, max 167 ms (first ONNX call) | PASS | backend with `--Backend:Replay:Enabled=false`, then `DemoMeasure -- latency http://localhost:5080 5005 AI-engine/models/demo_payloads.bin 10` |
| Same, ONNX-active windows only (n=303) | IS1 | p50 25.6, p95 31.4, max 49.3 ms | PASS | same |
| TFT inference per window inside the backend | S8 <200 ms | p50 23.2 ms, p95 31.7 ms, max 108 ms | PASS | `GET /api/metrics` |
| Golden parity C# vs Python (v2 contract, rtf_onset_gate) | — | 18/18 backend tests pass incl. `GoldenParityTests` | PASS | `dotnet test DigitalTwin.slnx` |

Caveats:
- IS1 here covers payload ingest → dashboard client on one machine. It does not include the
  COE node's acquisition/DSP leg (COE spec S5 ≤333 ms) or the Wi-Fi hop.
- On this stream, bearing 1 (true fault: outer race) is called "Cage fault" near end of life.
  That is the v2 model's real behaviour on a held-out bearing; IS3 is NOT claimed from this stream.

## Update (~08:30): ONNX warm-up at backend startup
The backend now runs 3 dummy TFT and 3 dummy N-HiTS inferences when the engine loads (18 ms total),
so the first real window no longer pays session initialisation. Re-measured twice, same command:

| Run | All windows (n=339) p50 / p95 / max | ONNX-active windows (n=303) p50 / p95 / max |
|---|---|---|
| 1 (first client after process start) | 25.5 / 30.5 / **137.5 ms** | 25.6 / 30.3 / **39.8 ms** |
| 2 (steady state) | 25.1 / 30.0 / 39.0 ms | 25.1 / 30.0 / 39.0 ms |

The remaining 137 ms one-off in run 1 falls on a pre-ONNX commissioning window, i.e. first-payload
JIT/connection setup after a process start. With ONNX active, the max is ≤ 40 ms.
