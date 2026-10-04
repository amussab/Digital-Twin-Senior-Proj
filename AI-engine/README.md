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
| `aiengine/labels.py` | causal onset (HI > μ+3·max(σ,0.01) of HI windows 6–29, 12 consecutive), stages 1–6, observable class |
| `aiengine/robustness.py` | SECONDARY LOBO robustness over all eligible XJTU-SY bearings |
| `aiengine/demo_stream.py` | `models/demo_payloads.bin` real held-out replay stream (152-byte payloads) |
| `aiengine/splits.py` → `splits.json` | frozen bearing/record-level splits per dataset |
| `aiengine/models.py`, `train.py` | TimeSeriesDataSets, TFT/N-HiTS builders, pretrain, `finetune()` |
| `aiengine/select.py` | LOBO RUL-estimator selection, TFT val grid |
| `aiengine/rul.py` | forecast → RUL (trend / onset-scaled / blend), stage estimate |
| `aiengine/evaluate.py` | S7 (pre-registered), S8, IS3, IS1-ICS-leg scoring and reports |
| `aiengine/export.py` | ONNX, parity (ORT vs PyTorch, tensor builder vs TimeSeriesDataSet), contract, golden vectors |
| `aiengine/engine.py` | `HybridEngine` runtime (payload → per-bearing JSON), mirrored by C# |
| `aiengine/datasets/synthetic_rig.py` | SYNTHETIC two-bearing rig stand-in |

## Real-data run (2026-10-05) — exact reproduce commands

```bash
PY=../AI-test/.venv/Scripts/python.exe; D="--datasets xjtu_sy ims"
$PY app.py selftest
$PY app.py train  $D --run real --epochs 30 --threads 20                     # pretrain, IMS excluded
$PY app.py select tft $D --run real --epochs 20 --threads 8 --tft-encoders 6 12 --weight-powers 0.25 0.5
cp checkpoints/real/select_tft/tft_02/tft.ckpt checkpoints/real/tft.ckpt     # val winner: enc 12, power 0.25
$PY app.py select rul $D --run real --epochs 12 --threads 8                  # LOBO over 12 train+val bearings
$PY app.py train  $D --run real --skip-tft --skip-nhits --tft-encoder 12 --class-weight-power 0.25     --rul-from reports/model_selection_rul_20261004T212405Z.json
$PY app.py finetune $D --run real --epochs 8 --threads 8                     # IMS ft_train B1/B2/B3
for k in $(seq 0 14); do $PY app.py robustness $D --run real --fold $k --epochs 10 --threads 2; done
$PY app.py robustness $D --run real                                          # aggregate SECONDARY LOBO
$PY app.py export $D --run real                                              # models/ (pretrained = XJTU-SY)
$PY app.py evaluate $D --run real --tag real --robustness --threads 8
$PY app.py demo-payloads $D                                                  # models/demo_payloads.bin
```
Seed 20261004. Splits frozen in `splits.json` before any real-data training (DESIGN.md change log).

## Results

**[MEASURED on XJTU-SY / IMS public bearing datasets, not the team rig.]** Copied from
`reports/spec_check_20261004T214921Z_real.md` (JSON alongside; per-window predictions in
`*.tft_preds.csv.gz` / `*.rul_preds.csv.gz`). Test bearings: XJTU-SY Bearing1_4 (cage), 2_5 (outer),
3_4 (inner) with the pretrained model; IMS test1 B4 (ball) with the IMS-fine-tuned model.

| Spec | Target | Measured | Verdict |
|---|---|---|---|
| S7 RUL MAPE (pre-registered, 4 test bearings) | <= 15 % | 259.5 % (XJTU-only 189.9 %; per bearing 1_4 464 %, 2_5 51 %, 3_4 55 %, IMS B4 469 %) | FAIL |
| S8 TFT latency p95 (1 thread, ONNX) | < 200 ms | 11.4 ms | PASS |
| IS3a macro-F1, testable classes {healthy, outer, inner, cage} (primary) | >= 0.85 | 0.566 | FAIL |
| IS3a macro-F1, all 5 classes (ball: 1987 test windows, 0 training bearings) | >= 0.85 | 0.288 | FAIL |
| IS3b stage error, max abs (literal) | <= 1 | 4 (mean 0.90; 72.1 % of windows within ±1) | FAIL |
| IS3c caught by stage 3 | 4/4 | 2/4 (2_5 at stage 2, 3_4 at stage 3; 1_4 cage and IMS B4 ball never caught) | FAIL |
| IS1 ICS leg p95 (features+TFT+N-HiTS+RUL) | < 500 ms | 14.8 ms | PASS |

SECONDARY robustness (`reports/model_selection_robustness_lobo_20261004T214321Z.json`, LOBO over all 15
XJTU-SY bearings, TFT + N-HiTS retrained and the estimator selected inside each fold): S7 median 73.7 %,
mean 221.7 %, 0/15 bearings ≤ 15 %. Fine-tune demonstration (IMS B4): S7 406.4 % before fine-tuning,
468.6 % after; B4 classification is untestable (ball has no output channel).

Synthetic pipeline-proof results (superseded): `reports/spec_check_20261004T192939Z_synthetic.md`.
