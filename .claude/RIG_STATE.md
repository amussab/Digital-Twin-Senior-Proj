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
