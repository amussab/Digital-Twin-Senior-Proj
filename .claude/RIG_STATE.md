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
