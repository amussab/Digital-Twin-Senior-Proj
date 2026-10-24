# DigitalTwin.Backend

ASP.NET Core (net10.0, Kestrel) ingest + inference + SignalR service for the OH-2 pump digital twin.
Data path: ESP32-C6 node -> UDP/HTTP (152-byte payload) -> decode -> per-bearing features -> ONNX (TFT + N-HiTS)
-> SignalR `/hubs/dashboard` -> dashboard.

## Run

```
dotnet run --project src/DigitalTwin.Backend                       # listens on http://0.0.0.0:5080, UDP 5005
dotnet run --project src/DigitalTwin.Backend --Backend:Replay:Enabled=true --Backend:Replay:Synthetic=true --Backend:Replay:RateHz=2
dotnet run --project src/DigitalTwin.Backend --Urls=http://192.168.1.50:5080   # bind to one LAN IP
```

Without `AI-engine/models/model_contract.json` (+ the two ONNX files it names) the service starts the
**StubInferenceEngine** and logs `*** USING SIMULATED STUB INFERENCE ENGINE ***`; `/api/state`, `/api/metrics`
and `/api/health` report `isSimulated: true`. Stub output is a heuristic, not a model prediction.

## Endpoints

| Endpoint | Purpose |
|---|---|
| UDP `:5005` | one datagram = one 152-byte little-endian payload |
| `POST /api/payload` | same payload as `application/octet-stream` (fallback) |
| `/hubs/dashboard` (SignalR) | event `snapshot` carrying `DashboardSnapshot` (camelCase JSON), pushed on every payload and re-broadcast every 50 ms (20 Hz nominal; S9 requires ≥10 Hz) |
| `GET /api/state` | engine info, both bearings (class, confidence, HI, RUL, buffer fill), current snapshot |
| `GET /api/metrics` | p50/p95/max per stage: `decode`, `features`, `inference`, `broadcast`, `total_ingest_to_broadcast`; reject count |
| `GET /api/health` | liveness + `isSimulated` |

Payloads other than exactly 152 bytes are rejected, logged at error level and counted. The 168-byte FDR variant
is rejected with an explicit message until COE reconciles the layout.

Snapshot mapping: the worse bearing (ready beats warming, then higher HI, then lower RUL) fills
`FaultClass`, `FaultConfidencePercent`, `HealthIndexPercent` and `AiRulHours`. `AiRulHours = -1` and
`FaultClass = commissioning | warming_up | no_data` mean "no estimate yet". Each bearing first collects
`baseline_windows` healthy windows (contract; override with `Backend:BaselineWindowsOverride`), then fills
its encoder buffer.

## Config (`appsettings.json`, section `Backend`)

`MachineId`, `ModelContractPath`, `BroadcastIntervalMs` (50 = 20 Hz nominal, 2x margin over the S9 10 Hz floor given Windows timer granularity on
Windows timers), `ConnectionTimeoutSeconds`, `BaselineWindowsOverride`, `HoursPerWindow`,
`Udp.{Enabled,Port,BindAddress}`, `Replay.{Enabled,Path,Synthetic,RateHz,Loop}`. Replay `Path` is a `.bin`
(concatenated 152-byte records) or `.csv` (rpm, 32 features, 4 displacements per row).

## C5: LAN-only enforcement

1. Kestrel binds via `Urls` (default `http://0.0.0.0:5080`; set a specific LAN IP to bind one interface).
2. `PrivateNetworkGuard` middleware returns 403 for any remote address outside loopback, 10/8, 172.16/12,
   192.168/16, 169.254/16, IPv6 ULA/link-local.
3. CORS only allows origins whose host is a private/loopback IP, `localhost` or `*.local`.
4. The service makes no outbound calls (no HttpClient, no telemetry, no cloud SDKs).
5. Firewall (Windows, run as admin), allow only the LAN subnet:
```
New-NetFirewallRule -DisplayName "DT Backend HTTP" -Direction Inbound -Protocol TCP -LocalPort 5080 -RemoteAddress 192.168.1.0/24 -Action Allow
New-NetFirewallRule -DisplayName "DT Backend UDP"  -Direction Inbound -Protocol UDP -LocalPort 5005 -RemoteAddress 192.168.1.0/24 -Action Allow
```
Do not port-forward or tunnel these ports.

## Evidence commands (IS1 / S8 / S9)

```
dotnet test DigitalTwin.Backend.slnx --logger "console;verbosity=detailed"   # prints MEASURED rate and latency lines
curl http://localhost:5080/api/metrics                                       # live per-stage p50/p95/max
```
Test numbers are in-process (no network, stub engine) and `[SYNTHETIC]`; re-measure on the chosen server
hardware with the real ONNX engine before quoting them.

## Status after integration (see /INTEGRATION.md)

- `OnnxInferenceEngine` mirrors `AI-engine/aiengine/engine.py` + `rul.py` for contract v2 (per-bearing, 21 engineered
  features, class-conditioned RUL threshold, loglin RUL). Golden parity test runs and passes.
- Not implemented: persisted per-machine baselines (re-collected at each start).

## UP2: contract v3, COE simulator, hardware-day swap

**Model contract switch.** `Backend:ModelContractPath` (appsettings, env `Backend__ModelContractPath`, or CLI
`--Backend:ModelContractPath=AI-engine/models/model_contract_v3.json`). Contract v2 (`model_contract.json`) is still the default and its
tests stay green. v3 (`rul.method = "direct_v3"`): classifier ONNX in the TFT slot, `v3_rul.onnx` in the N-HiTS slot
(`encoder_cont` [1,48,21] -> `rul_head` [1,3] = rul_windows, z, stage). `OnnxInferenceEngine` then uses
`rul_hours = max(head[0],0) * cadence`, `stage = (int)head[2]` and skips `RulEstimator`/`Stage` (mirrors `engine.py`).
An optional contract `postprocess` block (planned v3b causal stage/RUL smoothing) is parsed into `ModelContract.Postprocess`;
`IRulPostProcessor`/`RulPostProcessors.Create` is the hook (TODO until the Python reference exists; an unknown block is logged, not applied).
Golden test: `GoldenV3ParityTests` against `golden_vectors_v3.json` ([SYNTHETIC] vectors).

**Demo commands.**
```
./scripts/run-up2-demo.ps1          # replay source 'up2' = AI-engine/models/up2_demo_payloads.bin ([SYNTHETIC] healthy -> fault -> failure)
./scripts/run-up2-coe-demo.ps1      # backend (replay off) + COE simulator (AI-engine/aiengine/coe) sending a ramping fault over UDP :5005
```
The dashboard header shows the machine label (`[SYNTHETIC / SIMULATED SOURCE]`). With replay disabled the label is exactly `Backend:MachineId`.

**Single ingest entry.** UDP `:5005` (one 152-byte datagram = one payload; `UdpIngestService`, C5 private/loopback sender guard unchanged)
is the only path for the COE simulator (`python -m aiengine.coe.run --source sim --udp 127.0.0.1:5005`) and the future STM32 + ESP32-C6 node.
HTTP `POST /api/payload` stays as a fallback. Replay sources are demo-only.

**Hardware day swap: change nothing but the sender.** Stop the simulator; power the node and point its UDP sender at `<server LAN IP>:5005`
(same 152-byte layout, little-endian, `t20_ms` = ms since stream start). Open UDP 5005 in the firewall for the node's subnet (see C5 above).
No backend, contract, model or dashboard change. Re-measure IS1 with `tools/DemoMeasure is1` (real Wi-Fi) and replace the `[SIMULATION]` hop.
