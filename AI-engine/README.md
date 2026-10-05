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

**v2 (IS3 with MaFaulDa, 2026-10-05)** -- TFT re-trained jointly; N-HiTS/RUL reused from `real`:
```bash
D2="--datasets xjtu_sy ims mafaulda"
$PY app.py data $D2                                     # freezes the mafaulda section of splits.json (XJTU/IMS unchanged)
for k in 0 1 2 3 4 5 6 7; do $PY -m aiengine.v2 fit --idx $k --epochs 25 --threads 3 & done; wait   # grid on VAL only
$PY -m aiengine.v2 aggregate                            # -> reports/model_selection_tft_v2_<ts>.json (winner idx 2 + onset gate)
mkdir -p checkpoints/real_v2 && cp checkpoints/real/{nhits.ckpt,nhits_ft.ckpt,rul_calibration.json,rul_calibration_ft.json} checkpoints/real_v2/
cp checkpoints/real_v2/select/tft_02/tft.ckpt checkpoints/real_v2/tft.ckpt; cp checkpoints/real_v2/tft.ckpt checkpoints/real_v2/tft_ft.ckpt
$PY -c "from aiengine import v2; [v2.write_decision(v2.CHECKPOINT_DIR/'real_v2'/f, 'rtf_onset_gate', 'model_selection_tft_v2') for f in ('tft.ckpt','tft_ft.ckpt')]"
$PY app.py export   $D2 --run real_v2                   # models/ (v1 copy kept in checkpoints/real/models_v1_backup/)
$PY app.py evaluate $D2 --run real_v2 --tag real_v2 --robustness --threads 12 --note "v2 = second look at XJTU/IMS test bearings; v2 choices made on validation only ..."
```
`models/demo_payloads.bin` is unchanged (it carries raw features; the inputs did not change).
Seed 20261004. Splits frozen in `splits.json` before any real-data training (DESIGN.md change log).

## Results

**[MEASURED on XJTU-SY / IMS / MaFaulDa public bearing datasets, not the team rig.]**
v1 = `reports/spec_check_20261004T214921Z_real.md` (TFT on XJTU-SY only; 4 output channels, no ball).
v2 = `reports/spec_check_20261005T012112Z_real_v2.md`: **v2 = second look at the XJTU/IMS test bearings;
every v2 choice was made on validation only** (`reports/model_selection_tft_v2_20261005T011306Z.json`).
Test sets: XJTU-SY Bearing1_4 (cage), 2_5 (outer), 3_4 (inner); IMS test1 B4 (ball); v2 adds 233 MaFaulDa
test records (held-out speed blocks). Per-window predictions in `*.tft_preds.csv.gz` / `*.rul_preds.csv.gz`.

| Spec | Target | v1 | v2 | v2 verdict |
|---|---|---|---|---|
| S7 RUL MAPE (pre-registered, 4 RTF test bearings) | <= 15 % | 259.5 % | 281.8 % (1_4 567 %, 2_5 47 %, 3_4 44 %, IMS B4 469 %) | FAIL |
| S8 TFT latency p95 (1 thread, ONNX) | < 200 ms | 11.4 ms | 11.1 ms | PASS |
| IS3a macro-F1, all 5 classes | >= 0.85 | 0.288 (ball untrainable) | 0.581 | FAIL |
| IS3a per dataset (classes present) | -- | -- | MaFaulDa 0.848, XJTU-SY 0.450, IMS 0.679 | -- |
| IS3b stage error, max abs (literal) | <= 1 | 4 (72.1 % within +/-1) | 4 (72.1 % within +/-1) | FAIL |
| IS3c caught by stage 3 | 4/4 | 2/4 | 1/4 (2_5 at stage 2; 3_4 at stage 4; IMS B4 at stage 6; 1_4 never) | FAIL |
| IS1 ICS leg p95 | < 500 ms | 14.8 ms | 14.4 ms | PASS |

v2 validation (selection only, not a result): pooled val macro-F1 0.917 (MaFaulDa 0.911, XJTU-SY 0.859). It
did not transfer to test because validation has no inner-race and no IMS-domain fault: on test, IMS B4
ball windows are called inner_race 75 % of the time (the only IMS-domain fault in training is B3's inner
race, a source shortcut), XJTU-SY cage stays at F1 0 and inner race at 0.19, MaFaulDa cage recall drops
to 0.60 (151 cage windows called outer). Per-source confusion matrices are in the report.

SECONDARY robustness (`reports/model_selection_robustness_lobo_20261004T214321Z.json`, LOBO over all 15
XJTU-SY bearings, v1 models): S7 median 73.7 %, mean 221.7 %, 0/15 bearings <= 15 %.

Synthetic pipeline-proof results (superseded): `reports/spec_check_20261004T192939Z_synthetic.md`.
