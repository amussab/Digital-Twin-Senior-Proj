# Rig day: flash → record → train → live

Goal for the day: train the AI on **real rig data** at several bearing health levels, then show the
**full integrated system live**: node → laptop AI → dashboard. All of the setup is done already,
so the day is only flashing once, recording and training.

All commands run in PowerShell from the repo root:
`cd C:\Users\lenovo\Documents\KFUPM\Term261\ICS414\Repo\DT-main` (branch `fix/rig-new-bearing`).
`$PY` means the AI Python: `$PY = "..\DT-up2\AI-engine\.venv\Scripts\python.exe"`.

---

## 0. Before you start (10 min)

| # | Do | Check |
|---|---|---|
| 0.1 | **Disconnect the VPN (ProTUN).** It hid COE-NODE last time. | — |
| 0.2 | Only once, in an **administrator** PowerShell: `netsh advfirewall firewall add rule name="COE node TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any` | — |
| 0.3 | Plug in the **ethernet cable** (internet). Wi-Fi is for the node only. | — |
| 0.4 | `./scripts/preflight.ps1 -Quick` | `PREFLIGHT PASSED`. Warnings for ST-LINK and COE-NODE are expected until steps 1 and 2. |

## 1. Flash the STM32 (only once, about 5 min; the ESP32 is not touched)

1. Switch the **24 V supply OFF**. The pins float during reset, so never flash with it on.
2. Plug the Nucleo's **ST-LINK USB** into this laptop. Leave it plugged in all day: speed changes use it.
3. Run `COE\firmware\stm32\tools\flash_release.ps1`. It checks the SHA-256 hashes, flashes both cores and resets.
   It ends with `Flashed ... and reset the board`.
4. Switch the **24 V ON**. The ESP LED goes red, then blue, then **green** (about 3–10 s).
5. Connect Wi-Fi to **COE-NODE**, then run `python COE\training\node_config.py --query`.
   You should get: `@CONFIG rpm=1800 ... ftf=0.4518 bsf=5.1393 bpfo=4.9699 bpfi=6.0301 hw=0.08 fw=...`

If no `@CONFIG` reply comes back, see Troubleshooting. Data can still be recorded at the default
1800 rpm, but changing speed needs that reply.

**No reflash is needed for any of these changes:**
- Speed: `python COE\training\node_config.py --rpm 3600` (any value 60–3600), or use the buttons on the recorder page.
- A different bearing type: `python COE\training\node_config.py --bearing <ftf> <bsf> <bpfo> <bpfi> [half_width]`.
  These are multiples of shaft speed. Send me the geometry and I'll compute them.
- Settings are not saved on the board. After a power cycle the node starts at **1800 rpm and the 12 Nov
  bearing**, so set them again if you need different values.
- **The node has no speed sensor.** The rpm you set must match the VFD, otherwise the BPFO/BPFI/BSF/FTF
  features are measured at the wrong frequencies.

## 2. Record the training data (most of the day)

Start the recorder with `python COE\training\training_recorder.py` and open **http://localhost:8000**.
The top indicator turns green and says "Data arriving". Only one program can use port 8000: stop the recorder
before going live in step 4.

### 2.0 Recommended: let the run-sheet driver do it

With the recorder running, in a second terminal:

```powershell
cd COE\training
python rig_session.py --make-plan   # writes rig_plan.csv: 28 mounts x (1800, 3600) = 56 runs, priority order
python rig_session.py               # step by step; rerun to resume after any stop
```

For each mount it tells you which bearing to fit and which VFD speed to set, then waits for **Enter**. It then
sets the node RPM, checks the packets report that RPM, records 4 minutes with all labels filled in, checks the
window count, backs up to `..\rig_data\<date>_training_data` and prints coverage. Each bearing is recorded at
1800 and then 3600 before the next swap. At the swap prompt, type a different bearing ID if needed, or
`s` to skip / `q` to quit. Edit `rig_plan.csv` first if your bearing set differs from the default
(`H-01..03`, `OR/IR/BA/CA-L<level>-a/b`). Claude can run this for you: you only swap bearings,
set the VFD and say "ready".

The order of the plan already follows the priority below, so stopping early still leaves a trainable set.

### 2.1 Fields to fill in for every run (manual mode, on the page)

| Field | Value |
|---|---|
| Condition | Healthy / Inner race / Outer race / Ball / Cage |
| Health level | 0 healthy · 1 incipient · 2 moderate · 3 severe |
| Bearing ID | The physical bearing, e.g. `B-03`. **Label every bearing.** The train/test split is done by bearing ID. |
| Position | Which bearing housing holds the swapped bearing: **bearing 1 (B1 sensors)** or **bearing 2 (B2Y sensor)**. Prefer **bearing 1**: it has two independent axes, while B2X is a copy of B1Y. |
| Commissioning baseline | Tick **only** for the healthy reference run in 2.2. |
| Shaft speed | Set the VFD first, then press **1800/3600 → Set** on the page. Recording is locked until the node confirms. |

### 2.2 Protocol (for each speed: 1800 first, then 3600)

Each run should be **at least 4 minutes**. The first 2 minutes of every run are discarded as warm-up.

1. **Commissioning baseline:** mount a healthy bearing, condition *Healthy*, level 0, **tick baseline**, record 4 min.
2. **Healthy:** 3 runs. Unmount and remount between runs, using a different healthy bearing if you have one.
3. **Each fault type at each health level:** **2 runs on 2 different physical bearings**. The minimum is level 2.
   Levels 1 and 3 give the severity/stage head something to learn.
4. Watch the coverage table on the page. To see what's still missing, run:
   `cd AI-engine; & $PY -m aiengine.data_coverage ..\COE\training\training_data --tier minimum`

Priority if time is short: baseline + healthy at 1800, then each fault at level 2 at 1800, then 3600,
then levels 1 and 3. Two runs on two bearings per class is the minimum needed for a test on a bearing
the model has never seen.

**Back up the data at every break:** copy `COE\training\training_data\` to
`..\rig_data\<date>_training_data\`. Git ignores that folder.

## 3. Train on the real data (about 2 min)

```powershell
cd AI-engine
& $PY -m aiengine.train_real --data ..\COE\training\training_data --out v4_real
& $PY -m aiengine.train_real --data ..\COE\training\training_data --out v4_real_sim --include-sim   # optional comparison
```

- The held-out bearings are chosen and frozen in `splits.json` **before** fitting. Each one is scored once.
- Read `AI-engine/reports/train_real_<time>.md` for per-class F1, the confusion matrix, per-rpm results,
  stage error and latency against **IS3** (macro-F1 ≥ 0.85, stage error ≤ 1, fault caught by stage 3) and **S8** (< 200 ms).
- A class with fewer than 2 runs is flagged **INSUFFICIENT DATA**. Record more rather than claiming it.
- Every number is tagged `[MEASURED on rig]`. Don't tune after looking at the held-out scores.
- Outputs: `AI-engine/models/model_contract_v4_real.json`, the ONNX files, `baseline_v4_real.json`.
- To check a recorded run with the new model:
  `& $PY -m aiengine.node_eval <run.csv> --contract v4_real --baseline models\baseline_v4_real.json`

## 4. Go live: the full integrated system (the final prototype)

1. **Stop the recorder** (Ctrl+C) so port 8000 is free.
2. `./scripts/run-rig-live.ps1 -Contract v4_real -Baseline AI-engine/models/baseline_v4_real.json`
   (before training, run it without arguments and it uses `v3e_node`).
3. The dashboard opens at **http://localhost:5080**. Open the **Node (live)** panel; its badge reads `[LIVE rig data]`.
   - Without a saved baseline, the panel shows `warming_up` for 3 minutes, then classifies.
   - It shows per-bearing class and probabilities, onset, the 32 features with trends, rpm, and latencies
     (inference, then backend, then dashboard render).
   - Every live window and its prediction is also logged to `logs/live_*.jsonl`.
4. **Demo:** swap in a faulty bearing (or switch to 3600 with `node_config.py --rpm 3600` after setting the VFD)
   and show the class change on the dashboard. Screen-record it: a video plus the logged data is Rank 8,
   and running it live at the presentation is Rank 9.
5. Stop with `./scripts/stop-rig-live.ps1`.

## 5. End of day

- Copy `COE\training\training_data\`, `logs\` and `AI-engine\reports\train_real_*` into `..\rig_data\`.
- Commit the reports, the `v4_real` models and `splits.json`. Push `fix/rig-new-bearing`, then open a PR into `main` for COE to review the firmware.
- Update `.claude/RIG_STATE.md`.

## What is and isn't claimed

| Claim | Evidence after today |
|---|---|
| Fault class F1 (IS3) | Real held-out bearings, if each class has 2 or more bearings. Otherwise cross-validation only, and say so. |
| Inference < 200 ms (S8) | Measured live: about 2–10 ms on this laptop. |
| End-to-end < 500 ms (IS1) | Laptop receive → dashboard measured (about 0.3–0.4 s on a replay). The node → Wi-Fi leg is **unmeasured**: the node sends no timestamp. |
| RUL (S7) | **Not valid on node data.** A run-to-failure would be needed. The dashboard shows n/a. |
| Physics twin (IS2) | **n/a on the node.** It needs raw waveform or proximity data, which the node doesn't send. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| COE-NODE isn't listed | VPN off; check the ESP 5 V/GND jumper (LED off); see `COE/README.md`. |
| LED stays blue, never green | Press the STM32's black RESET button. |
| Joined Wi-Fi but no data | Firewall rule (0.2); restart the recorder; check nothing else holds port 8000. |
| `node_config.py` gives no `@` reply | Check the ST-LINK USB is plugged in (the COM port appears in Device Manager); close any other program using the COM port (`com.ps1`, a serial monitor); retry once. |
| `@ERR RPM ... band` | That rpm puts a bearing band past 400 Hz or makes bands overlap for the active bearing. Use a lower rpm, or check the bearing values. |
| Dashboard is empty | `logs/receiver.log` and `logs/backend.err.log`; check port 5080 is free. |
| Need to fall back | Old behaviour: `COE\firmware\stm32\tools\flash.ps1` builds from source; `git log` lists every change. |
