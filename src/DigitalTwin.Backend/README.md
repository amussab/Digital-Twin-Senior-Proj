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
