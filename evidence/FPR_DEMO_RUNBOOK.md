# FPR demo runbook (review 2026-11-16)

All commands: PowerShell from the repo root `Repo/DT-main` (branch `fix/rig-new-bearing`). Python =
`C:\Users\lenovo\Documents\KFUPM\Term261\ICS414\Repo\DT-up2\AI-engine\.venv\Scripts\python.exe` (the scripts already use it).
Deeper rig detail: `RIG_DAY.md`. Evidence numbers: `AI-engine/notebooks/FPR_Evidence.html`.

## What is real and what is simulated (say this out loud)

- REAL: the node, the Wi-Fi link, the features, the **fault class** (model `v4_real`, trained on real rig windows, ONE physical bearing per fault, severities estimated), the dashboard, the backend, the latency numbers.
- SIMULATION: **stage, AI RUL, physics RUL, Miner D** come from `v7_node`, trained on simulated degradation whose endpoints are resampled from the rig windows (1 bearing per fault, damage curve assumed). It has NOT been validated on real run-to-failure data; on the one real failure trace (ball BA-1, n=1) it scores 83 % MAPE. RUL time unit = simulated (table) hours, not wall-clock.
- Exact sentence for the live tiles: "Fault class comes from a model trained on real rig data. Stage and remaining life come from a model trained on simulated degradation anchored on our rig recordings; it has not been validated on real run-to-failure data, and the time unit is simulated hours."
- The dashboard shows no label on the live-rig stage/RUL tiles (user decision). Do not use live-rig screenshots as S7/S9 evidence; the evidence is the notebook.
- Numbers to quote (all tagged in the notebook): S7 11.47 % [SIMULATION, learnable profile; hard profile 21.1 %; trivial baseline 63.7 %]; IS3 headline = UP2 v3b F1 0.9587, stage max 1, 80/80 [SYNTHETIC]; v7 anchored-sim IS3 = **0.8486 and 76/80** (0.0014 below 0.85, reported as measured); IS2 = UP2 v3b 16.3 % / 49.2 % [SYNTHETIC, common-mode caveat]; IS1 = UP2 placeholder, not rerun on this build; S8 p95 about 10-14 ms [MEASURED host]; C5 live tests.

## STM32 firmware generations (read before touching the node) - added 2026-11-16
- **Current node (flashed on rig day, our release `COE/firmware/stm32/release/rig_runtime_config`)**: schema 1, bearing profile 2 (S6700 orders, bsf band at 1 x BSF), B2X = copy of B1Y, rpm typed with `node_config.py --rpm` (needs ST-LINK USB; reverts to 1800 on reboot). `v4_real` + `baseline_v4_real.json` match this firmware. USE THIS FOR THE REVIEW.
- **Ewidah's newer firmware** (`origin/main`, merged into this branch 2026-11-16; `COE/ICS_START_HERE.md`, prebuilt `COE/firmware/prebuilt/coe_node_stm32_LIVE.hex` + `coe_node_esp32.bin`): node MEASURES the speed from the Keyphasor (ADC ch 7) - no `node_config.py`, no ST-LINK needed for speed, `rpm` = measured speed, window only while speed steady within 1 %; schema 2 adds `measured_rpm`, probe gap/pk-pk/1X; bearing profile 3 (bsf band = 2 x BSF = 10.28); B2X is a real independent sensor. **The merged STM32 SOURCE is Ewidah's** (our typed-rpm `rpm_command` console code is now unused); hardware still runs our old build until you flash.
- **Code side is already ready (this branch)**: `COE/*/host_receiver.py` accept schema 1+2 and profiles 1-3, rpm up to 4000; the engine already reads `rpm` from every window and matches baselines within 5 % (a measured 1750 matches the 1800 baseline). So after a reflash the rpm follows the rig automatically.
- **But the models are NOT validated on the new firmware**: profile 3 moves the bsf feature to 2 x BSF and B2X becomes independent; `v4_real` was trained on profile 2 with B2X := B2Y. The receiver prints `WARNING: node reports bearing_profile 3` once. Class output after a reflash = unvalidated until new labelled data is recorded and `v4_real` retrained (do not mix profile-2 and profile-3 recordings). Not flashed yet; decision recorded in `.claude/RIG_STATE.md`. Recorder page: the typed-speed box is gone (merged); it now shows the measured Keyphasor speed, which only appears with the new firmware (old firmware: it says so).

## (a) LIVE rig demo

### 10-line pre-review checklist
1. VPN (ProTUN) off. Firewall rule for TCP 8000 exists (one-time, admin; `RIG_DAY.md` 0.2).
2. 24 V supply ON, ST-LINK USB plugged in (leave it in), ESP LED goes red, blue, green (3-10 s).
3. Laptop on Wi-Fi **COE-NODE** (not KFUPM-WiFi; it drifts).
4. `python COE\training\node_config.py --query` returns an `@CONFIG` line (rpm, bearing orders S6700 values: ftf 0.4518 bsf 5.1393 bpfo 4.9699 bpfi 6.0301).
5. VFD set to the speed you will type (1800 or 3600 only). Typed rpm must match the VFD.
6. `./scripts/run-rig-live.ps1` (default = hybrid_rig, v7_node RUL, baseline file loaded). Wait for "Dashboard : http://localhost:5080/".
7. `python scripts\rig_dryrun.py --count 5` prints windows with rpm and both bearings; baseline phase leaves `warming` / `settle`.
8. Bearing 2 stays `healthy`, RUL n/a (housing 2 is always healthy).
9. With a faulted bearing in housing 1: class matches, stage >= 2 within about a minute, RUL appears. (Healthy bearing: RUL n/a is correct.)
10. Close other programs on COM7 (VS Code STM32 extension / serial monitors), disable laptop sleep, plug in the charger.

### Start
```powershell
./scripts/run-rig-live.ps1                    # = -Contract hybrid_rig -RulContract v7_node -RulTimescale 1 -Baseline AI-engine/models/baseline_v4_real.json
# useful flags: -NoBrowser  -Republish (rebuild dashboard)  -SettleS 120 -BaselineS 60  -Contract v4_real (class only, no stage/RUL)
./scripts/stop-rig-live.ps1                   # stops backend + receiver, frees ports 8000/5080
```
Logs: `logs/backend.log`, `logs/receiver.log`, `logs/live_<time>.jsonl` (every window and prediction; has `rulSource`).

### Speed (1800 / 3600 only)
Set the VFD, then tell the node (not persisted): `python COE\training\node_config.py --rpm 3600` (or `--rpm 1800`). **The node reverts to 1800 after any STM32 reboot / power cycle**: re-run `--rpm 3600` and check `--query`. The baseline file holds both 1800 and 3600, so a speed change needs no new baseline.

### Warm-up
Without the baseline file: about 120 s settle + 60 s baseline per rpm (3 min). With `AI-engine/models/baseline_v4_real.json` loaded (default) the baseline is already known, so a faulted bearing at start is judged against the healthy baseline and classified right away (the settle/hold of the first windows still applies). Recommendation: start the stack, run `rig_dryrun.py`, and let it stabilise 3 minutes BEFORE the reviewers arrive. Override with `-SettleS`/`-BaselineS` (env `NODE_SETTLE_S`, `NODE_BASELINE_S`).

### Bearing swap
Stop the rig, swap the bearing in **housing 1** (B1 sensors; B2X is a copy of B1Y, so housing 2 stays healthy), restart the rig, confirm speed (`--query`; re-send `--rpm`). The receiver keeps running; the baseline is the stored healthy one, so the faulted bearing is not accepted as "normal". If bearing 1 still shows `healthy` after the swap, wait 1-2 min (class needs a few consecutive windows).

### Power-off / restart
- Rig power off: stop the VFD, 24 V OFF. Backend can stay up (dashboard keeps the last state).
- Restart: 24 V ON, wait for green LED, Wi-Fi COE-NODE, `node_config.py --rpm <N>` (node starts at 1800), then the VFD. No need to restart the stack unless the receiver died.
- Backend/receiver restart: `./scripts/stop-rig-live.ps1` then `./scripts/run-rig-live.ps1`. The baseline is reloaded from `AI-engine/models/baseline_v4_real.json`; it is only re-saved when a new rpm baseline is captured.

### Dashboard tiles
Class and confidence per bearing (real, v4_real); health stage 1-6; AI RUL and physics RUL (hours, simulated table hours) and Miner D: **only for a bearing called faulted at stage >= 2**, else `n/a`; baseline phase; inference ms; update rate. `http://localhost:5080/api/metrics` = latency stages; `http://localhost:5080/api/live/latest` = the Live block (class/stage/RUL, `rulSource`, `rulContract`).

### Dry-run command (prints class, rpm, stage, RUL per bearing from the backend)
```powershell
python scripts\rig_dryrun.py --backend http://127.0.0.1:5080 --count 10
```
Example line: `seq 41 rpm 1800 base=... | B1: inner_race stage=5 RUL=14.2 h phys=5.8 h [ok] | B2: healthy stage=1 RUL=n/a ...`

### Troubleshooting
| Symptom | Fix |
|---|---|
| COE-NODE not listed | VPN off; check ESP 5 V/GND jumper (LED off); see `COE/README.md`. |
| `node_config.py` no `@` reply | ST-LINK USB plugged in; another program holds the COM port (kill `cube stlink-server`); retry once. |
| `rig_dryrun.py` "no live window yet" | node not posting: Wi-Fi, firewall, port 8000 busy (`stop-rig-live.ps1`), receiver log. |
| rpm shows 1800 but VFD is 3600 | node rebooted; re-run `--rpm 3600`. |
| Class right, no RUL | bearing not yet stage >= 2, baseline still capturing, or it is the healthy bearing (n/a is correct). |
| Dashboard blank | `./scripts/run-rig-live.ps1 -Republish`; use 127.0.0.1 not localhost (2 s DNS delay). |
| Anything fails live | switch to (b) Docker replay and say it is a simulation replay. |

## (b) Docker replay FALLBACK (SIMULATION, labelled on screen)
```powershell
docker compose --profile fpr-demo up -d --build backend-fpr fpr-replay     # -> http://127.0.0.1:5080/  (contract v7_node by default; FPR_CONTRACT overrides)
docker compose --profile fpr-demo logs -f fpr-replay                        # receiver / replay log
python scripts\rig_dryrun.py --count 5                                      # same monitor
docker compose --profile fpr-demo down                                      # stop
```
Native without Docker: `./scripts/run-fpr-demo.ps1 -Contract v7_node`. Badge on screen: `[SIMULATION; rig-anchored replay]`. The three units (inner, outer, cage) are simulated VALIDATION trajectories; commissioning about 15 s, then class/stage/AI RUL/physics RUL for bearing 1. Say: "This is a simulation replay of rig-anchored data, not the rig."
Do not run the evidence notebook (it starts its own backend on port 5080) while this stack is up; stop it first.

## (c) Notebook / HTML
- Open (no execution needed): `AI-engine/notebooks/FPR_Evidence.html` in a browser. Order: summary and KPI banner, all-specs table with status, then one section per spec with its highest-level evidence.
- Re-execute (about 6.5 min; needs the .NET SDK on PATH; port 5080 free):
```powershell
cd AI-engine/notebooks
$env:PYTHONPATH = "$PWD/.."
..\..\..\DT-up2\AI-engine\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute FPR_Evidence.ipynb --inplace --ExecutePreprocessor.timeout=-1
..\..\..\DT-up2\AI-engine\.venv\Scripts\python.exe -m jupyter nbconvert --to html FPR_Evidence.ipynb
```
- Fast check of the helpers: `cd AI-engine; $env:PYTHONPATH=$PWD; <python> -m pytest tests/fpr_evidence -q`.

## Honest talking points
1. Only the fault class on the live rig is a model trained on real data (1 bearing per fault, severities estimated by us on rig day; cross-speed transfer is weak: 0.36-0.79 on real data).
2. Stage and RUL: simulation-trained, anchored on rig windows. S7 11.47 % passes only on the learnable life-variance profile; the hard profile gives 21.1 %; on the real ball failure trace (n=1) 83 %.
3. IS3 on v7 is 0.8486 (0.0014 short; bootstrap noise is of that size but it is a miss as measured) and 76/80 caught by stage 3 (4 inner-race misses). The IS3 headline is UP2 v3b on synthetic data; both are shown.
4. IS2 is the UP2 result and is common-mode (twin and AI read the same signal); the feature twin on the rig was not scored. IS1's Wi-Fi hop is unmeasured (node has no timestamp); the value shown is a UP2 placeholder to be re-tested in the final integrated run.
5. ME/COE items are owner-supplied statements; artifacts are with the owners.
6. B2X is a copy of B1Y (4th sensor has no cable): housing 2 is a single-axis healthy reference.
