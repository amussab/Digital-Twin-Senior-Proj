# AI-engine — hybrid N-HiTS (RUL) + TFT (fault class), per biaxial bearing

Team M001's production AI engine (ICS, owner Jaddoua). Interface contract: [`DESIGN.md`](DESIGN.md).
`AI-test/` is the earlier synthetic-only testbench; this package reuses its ideas and is trained
on **real public bearing data** (XJTU-SY, IMS Test 1, MaFaulDa), to be fine-tuned on rig data
once that exists. `pytorch-forecasting` ships no pretrained weights: "pretrained" here means our
own pretraining on public data; `finetune` adapts it to a new machine (rehearsed on IMS).

- **Model unit** = one biaxial bearing (16 COE C7 features: X's 8 then Y's 8). The rig payload
  carries two (features[0:16], features[16:32]).
- **TFT** classifies every window into healthy / outer_race / inner_race / ball / cage from the
  last `encoder_length` windows + the current one (engineered, baseline-normalised inputs;
  no dataset identity; speed only via `rpm_norm`).
- **N-HiTS** forecasts the health index (HI); `rul.py` turns the forecast into RUL hours against
  the failure threshold of the TFT-predicted class; stage = 1 before the causal onset, else from
  the implied fraction of degradation consumed.

## Run

```bash
PY=../AI-test/.venv/Scripts/python.exe        # shared venv (torch CPU)
$PY app.py selftest                            # no model needed
$PY app.py data                                # caches -> features -> labels -> frozen splits.json
$PY app.py train --run main                    # pretrain TFT + N-HiTS (IMS excluded)
$PY app.py select rul --run main               # LOBO over train+val bearings (never test)
$PY app.py train --run main --skip-tft --skip-nhits --rul-from reports/model_selection_rul_<ts>.json
$PY app.py finetune --run main                 # IMS ft_train bearings
$PY app.py evaluate --run main                 # held-out test -> reports/spec_check_<ts>.{json,md}, reports/latest.md
$PY app.py export --run main                   # models/{tft,nhits}.onnx + model_contract.json + golden_vectors.json
$PY app.py demo                                # 152-byte payload replay through HybridEngine (ONNX Runtime)
```
`--datasets synthetic_rig` runs everything on SYNTHETIC data (pipeline proof only).

## Files

| File | Role |
|---|---|
| `aiengine/features.py` | per-bearing engineered inputs + HI, baselines (C#-portable; `engineer_arrays` is the reference) |
| `aiengine/labels.py` | causal onset (HI > μ+3σ ×3), stages 1–6, observable class |
| `aiengine/splits.py` → `splits.json` | frozen bearing/record-level splits per dataset |
| `aiengine/models.py`, `train.py` | TimeSeriesDataSets, TFT/N-HiTS builders, pretrain, `finetune()` |
| `aiengine/select.py` | LOBO RUL-estimator selection, TFT val grid |
| `aiengine/rul.py` | forecast → RUL (trend / onset-scaled / blend), stage estimate |
| `aiengine/evaluate.py` | S7 (pre-registered), S8, IS3, IS1-ICS-leg scoring and reports |
| `aiengine/export.py` | ONNX, parity (ORT vs PyTorch, tensor builder vs TimeSeriesDataSet), contract, golden vectors |
| `aiengine/engine.py` | `HybridEngine` runtime (payload → per-bearing JSON), mirrored by C# |
| `aiengine/datasets/synthetic_rig.py` | SYNTHETIC two-bearing rig stand-in |

## Results

**Current results are SYNTHETIC only** (pipeline proof). Copied from
`reports/spec_check_20261004T192939Z_synthetic.md`. They are not a measurement of any real bearing.
Real-data results (XJTU-SY / MaFaulDa / IMS) will replace this table once those caches exist.

| Spec | Target | Measured [SYNTHETIC] | Verdict |
|---|---|---|---|
| S7 RUL MAPE (pre-registered) | <= 15 % | 18.9 % | FAIL |
| S8 TFT latency p95 (1 thread, ONNX) | < 200 ms | 1.7 ms | PASS |
| IS3a macro-F1 | >= 0.85 | 0.860 | PASS |
| IS3b stage within +/-1 | >= 0.95 | 1.000 | PASS |
| IS3c caught by stage 3 | 1.00 | 1.000 | PASS |
| IS1 ICS leg p95 | < 500 ms | 2.3 ms | PASS |
