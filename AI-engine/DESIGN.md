# AI-engine — design contract

Owner: ICS (Jaddoua). Coordinator-authored 2026-10-04. This file is the interface that the
ml-engineer, data-engineer, backend-engineer and digital-twin-engineer agents build against.
Change it deliberately and note the change at the bottom.

`AI-test/` is the earlier synthetic-only testbench. Reuse its code freely (dsp, payload, models,
rul, export). `AI-engine/` is the production engine. It is trained on real public run-to-failure
and seeded-fault data, then fine-tuned on rig data once that exists.

## 1. What it must achieve (from `Project architecture/overview/Specifications.md`)

| Spec | Target | Model / part |
|---|---|---|
| S7 (ICS1) | RUL MAPE ≤15% on held-out degradation window | N-HiTS (+ TFT-selected threshold) |
| S8 (ICS2) | Classification <200 ms/window | TFT |
| IS1 | Sensor→classify+RUL→dashboard <500 ms (ICS leg measured here) | engine + backend |
| IS3 | macro-F1 ≥85% over {healthy, outer_race, inner_race, ball, cage}; stage error ≤1; fault caught by stage 3 | TFT + staging |
| IS2 | Physics-AI RUL residual ≤5% stage 1→3 | needs a physics RUL (digital twin) |
| C4 / C5 | Local only; no cloud | engine is offline; backend binds to LAN |

## 2. Package layout

```
AI-engine/
  aiengine/
    config.py          constants mirrored from team docs + spec targets
    dsp.py             COE Components 3–7 (from AI-test), plus bearing_features(x, y, rpm, cfg) → 16
    geometry.py        bearing geometries per dataset → characteristic orders
    datasets/          one adapter per dataset → window table (schema §3), cached to parquet
      xjtu_sy.py  ims.py  mafaulda.py  synthetic_rig.py  common.py
    features.py        per-bearing engineered inputs (C#-portable), baselines
    labels.py          onset detection, ground-truth health stages, observable class, RUL labels
    splits.py          frozen bearing-level train/val/test splits → splits.json
    models.py          N-HiTS, TFT builders
    train.py           pretrain + finetune entry points
    rul.py             HI forecast → RUL; class-conditioned thresholds
    evaluate.py        spec scoring → reports/*.json + *.md (provenance on the face)
    export.py          ONNX + model_contract.json + golden test vectors
    engine.py          HybridEngine: runtime inference on 32-feature rig windows (reference impl)
  app.py               CLI: data | train | finetune | evaluate | export | demo | selftest
  models/              committed ONNX + model_contract.json + golden vectors (small)
  reports/             tracked evidence reports
  notebooks/           PPR evidence notebook
  data/                gitignored (raw/ + cache/)
```

## 3. Window table schema (every dataset adapter returns exactly this)

One row = one 20-revolution window of ONE bearing.

| column | type | meaning |
|---|---|---|
| dataset | str | `xjtu_sy`, `ims`, `mafaulda`, `synthetic_rig`, `rig` |
| unit_id | str | globally unique sequence key = one bearing in one run/record, e.g. `xjtu_sy:Bearing2_3`, `ims:test1:B3`, `mafaulda:overhang/cage_fault/20g/41.5:overhang` |
| bearing_key | str | physical bearing identity used for splitting (several units can share one: all MaFaulDa records of one fault-type+position+severity share an identity group, see §5) |
| window_index | int | 0..n-1, consecutive within unit, time-ordered |
| t_hours | float | machine time of this window since run start (NaN for non-time-series records) |
| rpm | float | measured shaft speed for the window |
| fault_class | str | ground truth for the unit: healthy / outer_race / inner_race / ball / cage / mixed / unknown |
| run_to_failure | bool | true only for units that end in failure |
| life_hours | float | total life of the unit (NaN if not run-to-failure) |
| rul_hours | float | life_hours − t_hours (NaN if not run-to-failure) |
| severity | str | seeded-fault severity tag (MaFaulDa mass, etc.), else "" |
| x_bp_rms … x_bpfi_mag | float32 ×8 | COE C7 features of axis X (order: bp_rms, bp_kurtosis, bp_crest, env_rms, ftf_mag, bsf_mag, bpfo_mag, bpfi_mag) |
| y_bp_rms … y_bpfi_mag | float32 ×8 | same, axis Y |
| fs_hz | float | source sample rate |
| band_hz | str | band-pass actually used (e.g. "2000-8000") |
| orders | str | JSON of the FTF/BSF/BPFO/BPFI orders used for this bearing |

Mapping to the rig payload: bearing 1 = features[0:16] (X = 0–7, Y = 8–15), bearing 2 =
features[16:32].

Windowing: cut each raw record into consecutive non-overlapping 20-rev windows using the record's
rpm. For snapshot-style run-to-failure data (XJTU-SY, IMS), keep **the first 20-rev window of
each snapshot** (one window per time step, rig-equivalent). Seeded-fault records (MaFaulDa) keep
all their windows.

## 4. Engineered model inputs (per bearing, C#-portable)

Baseline = per-unit healthy reference:
- **Run-to-failure units:** median of the first 24 windows.
- **MaFaulDa:** median over the speed-matched `normal` record at the same bearing position
  (nearest rpm). This is the commissioning baseline of that machine at that speed.
- **Rig:** the commissioning capture (spec M5).

Inputs are those of AI-test's `features.py`, re-expressed per bearing. `plane_asymmetry` is
replaced by `axis_asymmetry` (X vs Y envelope ratio). The ml-engineer may add or remove inputs
if validation supports it, but every input must stay arithmetic on (window, baseline).

## 5. Splits (frozen before any test scoring)

- Run-to-failure bearings (XJTU-SY 15, IMS test1 B1–B4): split by bearing, stratified by
  operating condition. Test bearings are listed in `splits.json`. Because N is small, also run
  leave-one-bearing-out cross-validation on train+val bearings for model selection only.
- MaFaulDa: split by **(position, fault type, severity) groups × speed blocks**, so test speeds
  are never seen for that group. Records at neighbouring speeds are near-duplicates, so a random
  record split leaks.
- IMS test1 is also the **fine-tuning demonstration domain**: pretrain without it, fine-tune on
  its train portion, test on its held-out portion. This is the "new machine" rehearsal for the
  team's rig.

## 6. Labels

- **Onset** (first predicting time): HI exceeds baseline μ+3σ (from the 24 baseline windows) for
  3 consecutive windows. The onset detector runs causally, so the same rule works at runtime.
- **Ground-truth health stage (1–6) for run-to-failure units:** stage 1 before onset. The
  degradation interval [t_onset, T_fail] is split into 5 equal-time parts → stages 2–6.
  `[CLAUDE-SYNTHESIS]` A defined labelling, documented as such.
- **Observable class:** `healthy` until onset, then the unit's fault_class (same detectability
  policy as AI-test). `mixed` / `unknown` units are excluded from classification but kept for RUL.

## 7. S7 metric — pre-registered (written before any test result exists)

- **Degradation window** = [t_onset, T_fail − 0.1·(T_fail − t_onset)]. The last 10% is excluded
  because RUL_true → 0 makes per-window percentage error unbounded there.
- **MAPE** = mean over held-out test bearings of (mean over that bearing's degradation-window
  windows of |RUL_pred − RUL_true| / RUL_true). Hours. Bearing-averaged, so long-lived bearings
  don't dominate.
- Also reported, never substituted: RMSE (h), PHM-2012 score, percent-of-life error, and MAPE
  over the full degradation window.

## 8. Runtime contract (engine.py, mirrored by C# via ONNX)

Input per rig window: `rpm` (f32), `features[32]` (f32), `t20_ms`, plus the stored per-bearing
baselines. HybridEngine keeps a rolling buffer per bearing. TFT needs `tft_encoder_length`
windows; N-HiTS needs `nhits_encoder_length` RUL-cadence steps. Before the buffers fill, it
returns `warming_up`.

Output per window per bearing:
```json
{ "bearing": 1, "fault_class": "outer_race", "class_probs": {...}, "confidence": 0.93,
  "health_index": 0.41, "health_stage": 3, "rul_hours": 12.4, "rul_interval_hours": [9.1, 16.0],
  "top_features": [["bpfo_share", 0.31], ...], "latency_ms": {"features": .., "tft": .., "nhits": ..} }
```
The backend maps the worse of the two bearings onto `DashboardSnapshot`.

`model_contract.json` carries: input names/order, encoder lengths, class labels in
output-channel order, feature constants, thresholds, and golden input→output vectors for C#
parity tests.

## Change log
- 2026-10-04: initial contract (coordinator).
