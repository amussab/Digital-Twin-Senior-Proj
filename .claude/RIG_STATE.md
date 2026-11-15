# RIG_STATE — resume ledger for the real-rig integration

## RIG DAY RESULT (2026-11-14/15) — newest; supersedes the START HERE block below
DONE: flashed our `release/rig_runtime_config` (24 V ON + ST-LINK; 24 V OFF fails: target reads 0.02 V); runtime RPM/BEARING commands WORK on hardware
(`node_config.py --query` -> rpm 1800 + new bearing orders). ESP32 NOT reflashed (subagent verdict: COE's new ESP32 firmware is only a speed relay
we don't use; COE's prebuilt STM32 hex has OLD bearing orders -> do not use). Open COE items: port his PD6 RX-error fix; STM32 rebooted twice
(once at 3600 rpm, once during the bearing failure) -> node reverts to 1800 rpm default after any reboot, check `window_id` resets.
DATA INVENTORY (per-run table, formats, all locations): `Repo/rig_data/README.md`. The raw data is git-ignored (not in the repo): only on disk in `Repo/DT-main/COE/training/training_data` + backup `Repo/rig_data/2026-11-14_training_data`.
Data (`COE/training/training_data`, backup `Repo/rig_data/2026-11-14_training_data`): healthy H-01 (1800 baseline + first 1800 run as healthy + 3600 baseline),
inner (green, IR-1), outer (white, OR-1), cage (black, CA-1) at 1800+3600, ball (gold, BA-1) at 1800 only. ONE bearing per fault; severities ESTIMATED
(inner/outer 3, cage/ball 2), notes say so. Housing 1 = swapped bearing, housing 2 = always-healthy bearing. B2X = copy of B1Y (other housing) ->
policy B2X:=B2Y (`real_data.apply_b2x_policy`, contract `channel_policy`, applied in train_real, sim rows, NodeEngine).
Excluded/special: `Repo/rig_data/excluded/` (run 232516: STM32 reboot), `Repo/rig_data/run_to_failure/run_20261115_012341` (+CONTEXT.md; ball bearing broke at 3600,
rpm column invalid after reboots; NOT training data), `rig_data/old_b2x_copy/`.
Model `v4_real` (commit 7bb6a78): 10 runs/5,076 windows. [MEASURED on rig] CV macro-F1 0.45 (structurally pessimistic), within-mount time-blocked 1.00 (optimistic),
cross-speed 0.36-0.79 (fails 0.85). [SIMULATION] held-out sim units 0.80. IS3 NOT scorable on rig (1 bearing/fault). v3e_node [SIMULATION] 0.877 is the only IS3 pass.
S8 p95 13.6 ms MET; S9 21 Hz but RUL/physics empty (partial); IS1 partial; S7/IS2 not met (v4_real has no RUL). Full table: `evidence/RIG_SPEC_SCORECARD.md`.
Live stack with v4_real verified via replay (`run-rig-live.ps1 -Contract v4_real -Baseline AI-engine/models/baseline_v4_real.json -NoBrowser`; stop with stop-rig-live.ps1).
Not done: real-time demo with the node + screen recording; push (ask first, then PR into main). Offered: sim-trained RUL head on node features [SIMULATION].
Rule from user session: never fabricate evidence; synthetic data only as tagged [SIMULATION]; no jittered-real "held-out bearings". New tools: `rig_within_mount_eval`, `v4_sim_test`.
Gotchas: `cube stlink-server-start` (VS Code STM32 ext) can lock COM7 -> kill it; Wi-Fi drifts to KFUPM-WiFi; recorder jsonl rows are {message, received_at}.

## START HERE (updated 2026-11-13 night, for the rig-day session on 2026-11-14)
Goal of rig day: flash once -> record labelled real data at several bearing health levels (1800 + 3600 rpm)
-> train `v4_real` on it -> show the full system live (node -> AI -> dashboard). User only swaps bearings,
sets the VFD, switches 24 V, says "ready". Claude drives everything else.

Order of work (details + commands: `RIG_DAY.md`):
1. `./scripts/preflight.ps1 -Quick` (last full run 2026-11-13: PASS except expected warnings: VPN ProTUN up,
   firewall rule TCP 8000 missing (admin), ST-LINK unplugged, not on COE-NODE).
2. User: 24 V off + ST-LINK USB in -> `COE\firmware\stm32\tools\flash_release.ps1` -> 24 V on -> green LED.
3. `netsh wlan connect name=COE-NODE`; `python COE\training\node_config.py --query` -> expect `@CONFIG rpm=1800 ...`.
   FIRST hardware test of the runtime RPM/BEARING commands. If no reply: record at default 1800 only, debug later.
4. Ask the user the measured ball diameter (11 balls confirmed): 1.2 mm = defaults OK; 1.5875 mm ->
   `node_config.py --bearing 0.437 3.873 4.802 6.198` AND realign sim (`aiengine/coe/config.py` orders) + retrain v3e.
5. Ask which damaged bearings exist (type x level x count) -> `python COE\training\rig_session.py --make-plan`, then trim
   `COE/training/rig_plan.csv`. Need >= 2 physical bearings per class for a held-out test.
6. Recorder: `python COE\training\training_recorder.py` (port 8000); driver: `python COE\training\rig_session.py`
   (resumable; backs up to `Repo/rig_data/<date>_training_data`).
7. Train: `cd AI-engine; $PY -m aiengine.data_coverage ..\COE\training\training_data` then
   `$PY -m aiengine.train_real --data ..\COE\training\training_data --out v4_real` (+ `--include-sim` variant).
   Report vs IS3/S8 with [MEASURED on rig] tags; held-out bearings frozen in splits.json before fitting; no tuning on test.
8. Live: stop recorder; `./scripts/run-rig-live.ps1 -Contract v4_real -Baseline AI-engine/models/baseline_v4_real.json`;
   dashboard http://localhost:5080 (Node (live) panel). User screen-records a bearing swap (Rank 8; Rank 9 live).
9. End: back up data/logs to Repo/rig_data, commit reports+models+splits, update this file, ask before pushing
   `fix/rig-new-bearing` (then PR into main for COE review).

$PY = `C:\Users\lenovo\Documents\KFUPM\Term261\ICS414\Repo\DT-up2\AI-engine\.venv\Scripts\python.exe` (AI venv;
recorder/node_config/rig_session use system `python`). Use 127.0.0.1 not localhost in scripts (2 s DNS delay here).

Known open issues: dashboard "AI RUL Trend" shows -1 h on node source (should be n/a); bearing-1 residual "cage"
false alarms (~17 % on the fw250 dev file; expected to drop with valid bands, unproven); RUL + physics twin n/a on
node data; node->laptop Wi-Fi latency unmeasurable (node sends no timestamp).

---
# History (chronological, appended by agents)
# RIG_STATE ledger (branch fix/rig-new-bearing) -- AI-engine node-matching work, 2026-11-12

Python: Repo/DT-up2/AI-engine/.venv/Scripts/python.exe, run from DT-main/AI-engine with PYTHONPATH=$PWD,
AIENGINE_DATA_DIR=Repo/Digital-Twin-Senior-Proj/AI-engine/data. Git-ignored caches copied from DT-up2:
AI-engine/data/cache_up2, AI-engine/checkpoints/up2_v3.

## Step A (sim matches firmware) -- DONE
- Envelope LP = Butterworth order 7 via buttord(350,1250,0.1,60) at 30 kS/s (== firmware coefficient table, tested).
- Order bands = periodic Hann, mean removed, 4096 FFT, one-sided Hann-power PSD, sqrt(sum PSD * df), ceil/floor bins,
  half width max(0.08*shaft, df); bands placed with NOMINAL rpm (node has no tach); payload rpm = nominal.
- Units: pipeline outputs g (calibrate no longer multiplies by G).
- Time features: central-moment Pearson kurtosis, rms from raw sum of squares (firmware), crest = |peak|/rms.
- New bearing orders FTF .4518 BSF 5.1393 BPFO 4.9699 BPFI 6.0301, half width .08 in aiengine/config.py (+RIG_HALF_WIDTH_ORDER);
  OPERATING_RPM = (1800, 3600). rpm_norm centre/scale 2675/925 kept: 1800 -> -0.946, 3600 -> +1.0 (in range).
- B2X = copy of B1Y in both sim sources.
- Noise calibrated to measured healthy rig: Student-t(df 6) sensor noise; per-run df 4.5-8, per-channel noise x0.8-1.25.
  [CALIBRATED to MEASURED healthy rig] sim healthy median @1800: rms 0.039 g, kurt 3.6, crest 5.8, env_rms 0.031 g
  vs real 0.029-0.053 / 3.4-5.6 / 5.3-7.3 / 0.024-0.045.
- Tests: tests/coe updated + firmware-formula reference test (sine + AM at BPFO). Full suite 83 passed, 1 skipped.

## Next: B node_adapter.py, C retrain v3d_node, D healthy-real false-alarm check

## Step B -- DONE: AI-engine/aiengine/node_adapter.py (node_json_to_raw, NodeEngine -> HybridEngine), tests/test_node_adapter.py. 91 tests pass.

## Step C -- DONE: contract v3d_node (models/model_contract_v3d_node.json, v3d_node_cls.onnx, golden_vectors_v3d_node.json); v3b_coe untouched.
- Dataset synth_coe_node (420 runs, split by run 253/63/104 registered in splits.json BEFORE fit). Cmds (AIENGINE_COE_VARIANT=node):
  python -m aiengine.coe_adapt_eval data|select|test|export  (logs in AI-engine/checkpoints/node_*.log, git-ignored)
- Selected on val only: mix_w0.3 + cross_bearing_gate tau 0.75 (family). Tau grid extended to 1.0/1.5 for B2X=B1Y; old tau 0.6 not kept. val F1 0.8467.
- [SIMULATION] test (scored once): macro-F1 0.8716, acc 0.8905; F1 healthy .910 outer .893 inner .887 ball .908 cage .761;
  B2 false-fault 8.2 %, B1 0.5 %, missed-fault 18.5 %. v3b_coe-as-deployed on same data: 0.5315.
- Mixed with old synth_v3 (old bearing, payload-level) as before; RUL/stage heads = v3b unchanged (NOT valid on node features).

## Step D -- DONE [MEASURED on rig, healthy only, firmware RPM=250 so order bands invalid], 931 windows, after 24+25 warm-up:
- B1 healthy 43.4 % (rpm as sent) / 48.3 % (rpm forced 1800); B2 86.7 % / 90.0 %. onset never flagged. Host latency p50 1.9 ms, p95 2.4 ms.
- NOT tuned. Open: real B1Y has kurtosis ~6.5 (heavier than sim), cage/ball false alarms on B1; needs real 1800-rpm capture with fw 1800 + seeded faults.

## Step E -- contract v3e_node (baseline-relative classifier) -- DONE 2026-11-12 (aiengine/node_v3e.py; node_eval.py)
- Classifier inputs = 16 baseline-relative columns (features.V3E_COLUMNS): dropped kurtosis_max, crest_max, raw_*_share (absolute);
  added crest_rise (22nd engineered column, features.ENGINEERED_EXT, engine enables it when contract engineered_order has it; older contracts unchanged).
  Trained ONLY on synth_coe_node train (no synth_v3). Grid 4 xgb hp x gate (family/env x 7 tau) on val; selected depth4 lr.05, gate family tau .75.
- [SIMULATION] test (scored once): macro-F1 0.877 (>= 0.85); F1 healthy .915 outer .893 inner .927 ball .907 cage .744; B2 false-fault 4.4 %, missed 18.6 %.
  v3d_node (same procedure) test 0.8716.
- [SIMULATION] heavy-tail val variant (Student-t df 3.5 on B1Y/B2Y; B1Y kurt med 7.0 / crest 8.8 vs 3.8 / 5.9 normal): healthy specificity
  v3d 0.926 vs v3e 0.909 (clean-partner 0.925 vs 0.907). v3e did NOT improve on the proxy. Val F1 v3d .8467 vs v3e .8393.
- node_adapter default contract -> v3e_node (meets 0.85 on sim test). v3d_node still available.
- [MEASURED on rig, healthy only, DEV file, fw RPM=250 so order bands invalid] v3e_node, 907 windows after warm-up: B1 healthy 36.6 % (cage 51.6, ball 11.2),
  B2 healthy 99.6 %; identical with rpm forced 1800. Worse than v3d on B1 (43-48 %). NOT tuned. Conclusion: relative features alone do not fix B1;
  the cage false alarms probably come from band (ftf/bsf) ratios vs the 24-window baseline with fw250 order bands -> needs fw 1800 capture.
- Tool: python -m aiengine.node_eval <jsonl|csv> [--contract v3e_node] [--rpm-override N]  (reports/node_eval_*.json)

## AI: commissioning-baseline policy (contract v3e_node.1) -- DONE 2026-11-12
- Finding: dev-file B1 false alarms came from a 24-window baseline taken during rig warm-up (B1Y kurtosis median 5.4 -> 6.7), not fw-250 bands.
- Policy (contract feature_engineering.baseline.policy): settle_seconds 120, baseline_seconds 60 -> windows = s*rpm/1200 (1800 rpm: 180 + 90). Median, keyed per rpm (+-5 %),
  rpm change -> stored baseline or warm-up + buffers reset; warming output fault_class "warming_up". NodeEngine.save_baseline/load_baseline (JSON, contract version checked).
  Contracts without policy / policy=None = legacy 24-window (sim scoring, golden vectors untouched). node_eval --legacy-baseline. Live runner aiengine/node_live.py (+ README paste snippet).
- [MEASURED on rig, healthy only, DEV file, fw RPM=250 so order bands invalid; rpm forced 1800; not tuned] healthy fraction B1/B2:
  v3e settle180+base90 (default) 81.1/94.9 %; settle180+base100 79.8/95.6 %; baseline 180-280, scored 280+ 79.7/95.7 % (was 36.6/99.6 % with 24-window).
  v3d same three: 80.1/92.6, 78.5/93.3, 78.5/93.2 % (was 43-48/87-90). Remaining B1 error mostly cage (~17 %).
- Sim: NodeEngine not in the sim scoring path; sim test macro-F1 0.877 (v3e) unchanged by construction. Sim runs are 70 windows < 270 -> sim keeps the 24-window baseline (not mixed).

## RESUME HERE (paused 2026-11-12 by user; continue Saturday)
- Branch `fix/rig-new-bearing` (worktree Repo/DT-main), NOT pushed. Commits fabb638..f12409a.
- Runtime RPM firmware (6758ced), node_rpm.py + recorder Shaft-speed control (cdceba8), docs (f12409a) are committed but
  NEVER COMPILED: gnu-tools-for-stm32 bundle was still downloading in VS Code. Release ELFs in
  COE/firmware/stm32/release/rig_1800_runtime_rpm/ do NOT exist yet.
- Saturday order: (1) finish bundle install in VS Code (STM32Cube > Bundles: gnu-tools-for-stm32), (2) tools/build.ps1 -Mode LIVE,
  fix compile errors, (3) copy ELFs to release/ + SHA-256, (4) 24 V off, ST-LINK USB in, tools/flash_release.ps1, 24 V on,
  (5) python COE/training/node_rpm.py --query, (6) record commissioning baseline (2 min settle + 1 min) at 1800 and 3600,
  (7) fresh healthy runs scored ONCE with `python -m aiengine.node_eval <csv> --contract v3e_node`, (8) live run via node_live.
- Dev-file result (fw250): v3e_node B1 81.1 % / B2 94.9 % healthy with settle policy [MEASURED on rig, healthy only, DEV].
- Uncommitted on purpose: COE/firmware/stm32/.settings/*, .vscode/c_cpp_properties.json (VS Code extension edits).

## Agent A (firmware) -- 2026-11-13
- Review of 6758ced found a compile blocker: live_stream.c had raw newlines inside the "@OK/@ERR" string literals (never compiled). Fixed.
  Also: rx_tail now volatile (ISR reads it), RX ring 128, line max 96, replies end \r\n.
- New (21addf9): `BEARING <ftf> <bsf> <bpfo> <bpfi> [hw]`, `CONFIG?` (RPM? gives the same line). RPM and BEARING are validated jointly
  (against the pending partner), applied at the next window boundary; packet bearing_profile stays 1, runtime orders only via CONFIG?.
- Host: COE/training/node_config.py (--rpm/--bearing/--query/--selftest); node_rpm.py is a wrapper (recorder API unchanged).
- Tests: python mirror of the C parser (firmware/stm32/tools/tests); C itself NOT compiled on host (no host C compiler);
  syntax-checked only with the ST clangd bundle (only missing-libc errors).
- Build status: see the final entry below if a build was made. tools/preflight_fw.ps1 = report-only pre-flash check.

## Agent B (ML) -- recorder labels + train_real (2026-11-13)
- Recorder (COE/training): new run fields `severity` 0-3, `bearing_id`, `position` (both/b1/b2), `is_baseline`; CSV columns appended at the END; page has the selectors, a baseline checkbox and a Coverage table (minutes per condition x level x rpm). Old run files load (healthy -> 0, faulty -> unknown level).
- Tomorrow, from AI-engine/ (.venv python): `python -m aiengine.data_coverage <training_data dir>` (what is missing), then `python -m aiengine.train_real --data <training_data dir> [--include-sim]` -> models/model_contract_v4_real.json (+ v4_real_cls.onnx, v4_real_sev.onnx, golden_vectors_v4_real.json, baseline_v4_real.json), report reports/train_real_<ts>.{md,json}. Live: NODE_CONTRACT=v4_real NODE_BASELINE=AI-engine/models/baseline_v4_real.json. Replay: `python -m aiengine.node_eval <csv> --contract v4_real --baseline AI-engine/models/baseline_v4_real.json`.
- Protocol (data_coverage.py docstring): per speed 1 baseline run (3 min), 3 healthy runs, each fault x level 2 runs on DIFFERENT bearing_id (minimum = level 2 only; --tier full = levels 1-3); record >= 4 min per run (first 120 s dropped as settle). Held-out runs are chosen by hash and registered in splits.json section rig_real BEFORE fitting (needs >= 3 mounts per condition, else CV only, said in the report). Use a NEW bearing_id per physical bearing.
- Stage head: severity 0/1/2/3 -> stage 1/3/4/6 (assumption, stated in the report). The exported RUL output is a constant 0: NOT a RUL model.
- [SIMULATION rehearsal] (scratch Repo/rig_data/rehearsal/, 56 fake runs): end-to-end OK (train ~66 s, ONNX parity 8e-8, NodeEngine p95 ~10 ms); held-out macro-F1 0.847 (sim cage/ball hard) -- says nothing about rig results. Dev file (healthy only) -> clean "cannot train: only healthy" exit 2.
- Agent A BUILD (2026-11-13): compiler arrived; tools/build.ps1 -Mode LIVE built clean (0 warnings after initialising pipeline.c geometry outputs; the
  predecessor code otherwise compiled once the string-literal bug was fixed). Release in firmware/stm32/release/rig_runtime_config/ (commit 0c0a251):
  CM7 SHA256 397f4fe2..., CM4 ec012521... (full in SHA256SUMS.txt), BUILD_INFO.txt. NOT flashed. Order: preflight_fw.ps1, flash_release.ps1, node_config.py --query.

## Agent C (integration) -- 2026-11-13
- Path (a): receiver (COE/inference) runs NodeEngine via model_hook -> aiengine.node_live.predict_full, POSTs camelCase JSON to backend
  POST /api/live/prediction (non-blocking forwarder, queue 20, drop+count). Backend maps it to state + SignalR snapshot (new `Live` block);
  NO C# model runs (C# contracts are v3b, invalid for node features). RUL = n/a, physics twin = "n/a: node sends features only".
  Stage/health index shown but labelled unvalidated on node data. Provenance badge: "[LIVE rig data]"; replayer sets "replay":true -> "[REPLAY of recorded rig data]".
- Local-network guard (C5/I1) unchanged; new endpoints /api/live/prediction, /api/live/render-ack, /api/live/latest.
- Start: scripts/run-rig-live.ps1 [-Contract v4_real -Baseline path.json -SettleS -BaselineS]; stop: scripts/stop-rig-live.ps1. Uses dotnet (Docker daemon not used).
  Logs in logs/ (git-ignored): backend.log, receiver.log, live_<time>.jsonl (every window + prediction + latencies). Metrics: GET :5080/api/metrics (live_* stages).
- Rehearsal: COE/inference/replay_node.py <jsonl|csv> --rpm-override 1800 --speed 10 (use 127.0.0.1; "localhost" adds ~2 s IPv6 fallback on Windows).
- [MEASURED host, replayed rig data] 931 windows: node-ACK RTT p50 6.4 / p95 26 ms; inference p50 4.4 / p95 10 ms; laptop-receive->backend-ack p50 9 / p95 31 ms;
  laptop-receive->dashboard render (headless Edge, no GPU) p50 288 / p95 353 ms. Node->laptop Wi-Fi hop NOT measurable (node sends no timestamp).
- Dashboard SignalR channel capacity 64 -> 2 (latest wins; 64 gave ~600 ms render lag). Screenshot: Repo/rig_data/rehearsal_dashboard.png.
- dotnet test: 37 passed (2 new LiveNodeTests).

## RESUME HERE (2026-11-13 night, coordinator) -- supersedes the 11-12 "RESUME HERE"
- Rig day runbook: `RIG_DAY.md` (repo root). Preflight: `scripts/preflight.ps1 [-Quick] [-Rehearse]`.
- Preflight 2026-11-13: all checks pass except expected warnings (ST-LINK unplugged, not on COE-NODE, VPN ProTUN up,
  firewall rule for TCP 8000 missing -> admin step 0.2). Tests: AI 107, COE host 12, backend 37 passed. Rehearsal
  (120 replayed real windows -> receiver -> NodeEngine -> backend /api/live/latest) OK.
- Flash files: COE/firmware/stm32/release/rig_runtime_config (hash-verified). Runtime RPM/BEARING/CONFIG? untested on hardware.
- Known dashboard nit: "AI RUL Trend" shows -1 hours on node source (should be n/a); header said "Disconnected" in headless screenshot.

# FPR session (2026-11-15, FPR = 2026-11-16) — resume ledger
Plan: `.claude/FPR_PLAN.md` (+ REVISION 1 after critique, binding). User decisions: anchored-sim RUL set (tagged), no more rig time, extend twin + re-freeze IS2, no merge of origin/main.
Honest ceiling: S7/IS3 Rank 6, IS2 Rank 3-6 (sim-anchored, curve assumed); live demo Rank 9 for the pipeline only. Tell the user.
Status: [ ] generator rig_anchor (agent) [ ] feature twin (agent) [ ] v5 RUL head + baseline [ ] v5 classifier/stage [ ] IS2 eval [ ] integration [ ] evidence + reviewer
- 2026-11-15 20:50 (coord): v5_node scored ONCE on registered test: ALL GATES FAIL (S7 23.39 %, F1 0.8375, stage max 2, caught 46/48) -> kept in evidence. v6_node (val-only, FRESH test rig_anchor_test2 registered before fit, commit 48c6643) in progress by ML agent.
  Integration done (f013bac, 47fc966, 1f04307): NodeEngine v5/v6 + FeaturePhysicsTwin, fpr_is2 harness (select/test once), dashboard Live fields, demo units AI-engine/demo, compose `fpr-demo`+`evidence-fpr`. Docker daemon was DOWN -> only `compose config` verified.
  IS2 on v5 VAL: 4/6 negative controls PASS -> valid_for_test=false (twin Miner r^3 saturates). Twin recalibration on VAL only in progress (twin agent), then `fpr_is2 predict/select --contract v6_node`, `test --split rig_anchor_test2:test` ONCE.
  Evidence notebook agent building notebooks/FPR_Evidence.ipynb (+html) with computed statuses. After final: re-render demo units for chosen contract, update README limits, re-publish dashboard (header tag fix), run docker if daemon up.
- 2026-11-15 ~22:30 (coord, usage limit near; RESUME HERE): v6_node scored ONCE on fresh rig_anchor_test2: ALL FAIL (S7 22.14 %, F1 0.828, stage max 2, caught 63/64); oracle with hidden life/shape = 13.4 % => S7<=15 % needs narrower assumed life variance (UP2 synth_v3 precedent). Twin recal (04ea7c4): IS2 valid_for_test=false (2/6 controls pass) -> DO NOT score IS2 test; report Rank 3 design-level. Evidence notebook built (2f6bbdf): 13/17 met only via owner-supplied.
  RUNNING agent a93ada928be7baf4a (v7_node: research table + synthetic 'learnable' profile, val-only, fresh test once; outputs reports/s7_research_*, reports/v7_node_*). USER ORDER: ingest synthetic until val passes, tagged; one fresh-test pass; no re-scoring.
  TO FINISH: (1) read reports/s7_research_*.md + v7_node_*.md; (2) point notebook final_reports() at v7 if scored, re-execute: cd AI-engine/notebooks; PYTHONPATH=$PWD/.. ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute FPR_Evidence.ipynb --inplace --ExecutePreprocessor.timeout=-1 ; then --to html; refresh evidence/FPR_Evidence.md numbers;
  (3) re-render demo units for chosen contract (python -m aiengine.fpr_demo render), update README limits, republish dashboard (scripts/run-fpr-demo.ps1 -Republish); (4) docker daemon was DOWN: try `docker compose --profile fpr-demo up --build backend-fpr fpr-replay`, else native scripts/run-fpr-demo.ps1; (5) evidence-reviewer pass; (6) ask user before pushing/PR.
