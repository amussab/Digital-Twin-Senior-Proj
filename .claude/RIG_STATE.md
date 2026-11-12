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
