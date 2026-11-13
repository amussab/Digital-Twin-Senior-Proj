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
