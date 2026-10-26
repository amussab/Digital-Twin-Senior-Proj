# UP2 evidence summary (Team M001, 25/26 Oct 2026)

Real-Time Edge Digital Twin for Bearing Fault Diagnosis and RUL Prediction in OH-2 Centrifugal Pumps.
Standalone summary for the coach. The numbers here are recomputed live by `AI-engine/notebooks/UP2_Evidence.ipynb` (HTML copy:
`UP2_Evidence.html`); the master table and figures are stored in `evidence/up2_figures/`.

## 1. Status

**Scope.** 17 counted items. C3 (BOM cost) was excluded by the reviewer and is not counted; the bar is therefore **more than 75 % of 17 = 13 items**.

**Data basis.** Synthetic run-to-failure data and the COE / FE-twin simulators, **approved by the coach**. Every number keeps its provenance tag (`[SYNTHETIC]`, `[SIMULATION]`, `[MEASURED host]`, `[CALCULATED]`, `[OWNER-SUPPLIED: ME]`, `[MEASURED on <public dataset>]`).
Evidence rank follows the course ladder: a working demo run live at the presentation is Rank 9; re-scoring stored predictions is Rank 6.

**Count.** **13 of 17 items are MET (76.5 %): the >75 % bar (13) is reached.**

| Tier | Items |
|---|---|
| MET (live / measured) | C4, C5, S6, S7, S8, S9, IS3 |
| MET (physical, PPR Rank 9) | C6 (enclosure), S1 (biaxial IEPE sets): ME hardware shown at the PPR, presented by ME (`[OWNER-SUPPLIED: ME]`) |
| MET (simulation + calculation) | S4, S5 (COE pipeline, AD7606 timing and STM32 time calculated), IS1 (live end-to-end run, Wi-Fi hop emulated at 10-60 ms), IS2 (clarified wording v2, section 6c) |
| PARTIAL (50 % per ME) | S2 (FE stiffness identification), S3 (IP54 enclosure) |
| Owner evidence pending | C1 (ME), C2 (COE electrical) |

Owner evidence is recorded in one place (`up2_helpers.OVERRIDES`); counts are recomputed from the registry.

**Deployed model.** Gradient-boosted heads in the N-HiTS / TFT ONNX slots (approved change from N-HiTS/TFT for accuracy). Same synthetic data, for the design decision:

| Item | Deployed (gradient-boosted heads) | N-HiTS / TFT, same synthetic data |
|---|---|---|
| S7 RUL MAPE <= 15 % | 12.64 % MET | N-HiTS 46.9 % |
| S8 < 200 ms | classifier ONNX p95 sub-millisecond (0.04-0.4 ms, live) MET | TFT ONNX about 1.5-12 ms |
| IS3 F1 / stage / caught | F1 0.959, stage max 1, 80/80 MET | TFT F1 0.924, stage max 3, 40/40 |

Rank of the ML rows: re-scoring the stored fresh-test2 predictions is Rank 6; the notebook also **runs the deployed ONNX engine live** on a 16-bearing subset (S7 13.17 %, F1 0.964, stage max 1, 16/16 caught, recomputed from model outputs), Rank 9 (working simulation demo).

**Second independent confirmation (test3).** The deployed v3b heads were also scored once on a second fresh set, `synth_v3_test3` (100 runs, registered before the v3c fit): S7 **10.75 %**, IS3 F1 **0.9615**, stage max **1**, caught **80/80** (`reports/up2c_spec_check_20261024T175404Z.md`).
Times seen: old test set twice (v3, v3b); `synth_v3_test2` once for v3b (then re-scored read-only); `synth_v3_test3` once.

## 2. What changed since PPR (Oct 5)

| Item | PPR (old) | UP2 (new) | What changed |
|---|---|---|---|
| C4 no cloud | MET | MET (rank 7) | now also shown with every network call blocked while the v3b engine classifies 120 windows |
| C5 LAN only | MET | MET | 19 backend guard tests pass live |
| S4 sample rate | not evidenced | MET (simulation + calculation) | COE pipeline in code (`aiengine/coe`): 30 kS/s x 4 ch gap-free, 16.3 us slack per sample set |
| S5 payload <= 333 ms | not evidenced | MET (simulation + calculation) | host worst case ~7 ms; STM32 calculated 28.4 ms |
| S6 fixed AI input | partial (public data) | MET (live / measured, simulator) | 28 speeds, 1000-3600 rpm, always 32 features / 152 bytes (below ~1465 rpm the first 4096 envelope points feed the FFT; the 1750/3600 rpm modes use the full window) |
| S7 RUL MAPE <= 15 % | 281.8 % (public data, PPR model) | MET (12.64 %), deployed gradient-boosted head | new synthetic data + new model (N-HiTS on the same data: 46.9 %) |
| S8 inference < 200 ms | MET (11.1 ms) | MET (classifier ONNX p95 measured live, sub-millisecond, 0.04-0.4 ms) | deployed slot holds a gradient-boosted model |
| S9 dashboard >= 10 Hz | MET (30 Hz) | MET (live / measured) | client receives ~30 snapshots/s (50 ms re-broadcast timer); new data at the 10 Hz replay rate |
| IS1 < 500 ms | partial | MET (simulation + calculation), live end-to-end run | p95 ~98 ms with the Wi-Fi hop emulated at 10-60 ms and the STM32 leg calculated; budget holds up to a ~458 ms hop (table below) |
| IS2 physics-AI agreement | wording v1 | **MET under the clarified wording v2**: v3b median 16.3 %, p95 49.2 % of L_ref (targets 18 % / 63 %) on fresh test4 | wording clarified per coach (section 6c) |
| IS3 F1 / stage / caught | F1 0.581 (public data, PPR model) | MET (F1 0.959, stage max 1, 80/80 caught), deployed gradient-boosted heads | new synthetic data + new heads (TFT on the same data: stage max 3) |
| C6, S1 (ME) | - | MET (physical, PPR Rank 9) | ME hardware, presented by ME |
| S2, S3 (ME) | - | PARTIAL (50 %) | reported by ME |

PPR public-data numbers (XJTU-SY / IMS / MaFaulDa, `[MEASURED on public data]`) remain in the notebook for reference.

### IS1 Wi-Fi hop sensitivity `[CALCULATED]`

Budget = Wi-Fi hop + backend leg measured on this PC (p95 7.2 / max 14.0 ms) + COE STM32 leg (28.4 ms, calculated).

| Wi-Fi hop | total with backend p95 | total with backend max | < 500 ms |
|---|---|---|---|
| 60 ms (top of the emulated 10-60 ms) | 95.6 ms | 102.4 ms | yes |
| 100 ms | 135.6 ms | 142.4 ms | yes |
| 300 ms | 335.6 ms | 342.4 ms | yes |

The budget holds for a Wi-Fi hop up to ~458 ms.

## 3. Architecture and the hardware-ready boundary

```
SampleSource --256x8 int16 blocks + Keyphasor edges--> CoePipeline (C1-C9) --152-byte payload--> UDP :5005 --> backend (ONNX, SignalR) --> dashboard
 simulator | file replay | HardwareSource (stub)
```

The only interface the rest of the system sees is the **152-byte payload** (rpm f32, 32 features f32, 4 displacement f32, t20_ms u32), sent one per UDP
datagram, accepted from private/loopback senders only (C5). The simulator, a recorded-file replay, or the real AD7606/STM32 node all sit behind
`SampleSource`/the payload, so swapping in hardware changes nothing downstream.

## 4. How to run the live demo

From the repo root (PowerShell), with the backend and dashboard published once:

- Dashboard on the replay stream (healthy -> fault -> failure, `[SYNTHETIC]`): `./scripts/run-up2-demo.ps1` then open `http://localhost:5080/` (LAN address is printed).
- COE simulator through the real pipeline and UDP into the backend: `./scripts/run-up2-coe-demo.ps1` (`scripts/coe_fault_scenario.py`).
- Whole scorecard: `cd AI-engine/notebooks; ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute UP2_Evidence.ipynb --inplace --ExecutePreprocessor.timeout=-1` (about 6 minutes), or open the notebook and Run All.
- Individual checks: `cd AI-engine; .venv/Scripts/python.exe -m aiengine.up2_eval` (re-scores saved predictions, measures latency), `-m aiengine.coe.verify --quick`, `-m aiengine.up2d_eval` (IS2 wording v2; re-scores saved test4 predictions), `-m pytest tests/coe tests/up2 tests/twin_up2 -q`; `dotnet test tests/DigitalTwin.Backend.Tests -c Release`.

## 5. Method notes

- **Data.** ML numbers are on `aiengine/synth_v3` (`[SYNTHETIC]`, coach-approved): a physics-informed generator of 2-bearing biaxial feature windows in the COE
  payload layout. Behaviour (damage curve, defect-order bands, kurtosis/crest rise-and-fall, life scaling with rpm, noise, sensor drift, rare transients) follows the
  bearing-diagnostics literature; magnitudes are calibrated to the XJTU-SY end-of-life envelope ratio and the knob ranges were set on validation runs.
- **S7 definition** (pre-registered, DESIGN.md section 7, unchanged since PPR): MAPE of RUL in absolute hours, per bearing over [detected onset, T_fail - 10 % of the
  degradation span], averaged over bearings. True-onset variant 14.72 %, percent-of-life error 5.81 points.
- **Post-process.** The deployed v3b adds a causal post-process (stage median filter, log-EMA of RUL, stage scale), parameters chosen on validation runs and
  reimplemented in C# from `model_contract_v3b.json` (28 golden cases).
- **Test sets.** Headline numbers come from a fresh set (`synth_v3_test2`, 100 runs, new seed, registered before any v3b fit, predicted once); `synth_v3_test3` and
  `synth_v3_test4` were each scored once.
- **Healthy-companion bearings** are predicted at stage 2 in 99.8 % of windows (true stage 1); IS3b is defined on run-to-failure bearings.
- **Latency.** Classifier ONNX p95 sub-millisecond (0.04-0.4 ms across runs; the notebook's live cell is canonical), `[MEASURED host]` on this PC with synthetic inputs.
  IS1 combines the measured backend leg, a Wi-Fi hop emulated at 10-60 ms and the calculated STM32 leg.
- **COE-pipeline features.** The classifier adapted to COE-pipeline windows is in section 6b.

## 6. Owner-supplied items

| Item | Owner | Status |
|---|---|---|
| C6 UV-stabilized enclosure | ME | MET (physical, PPR Rank 9), presented by ME |
| S1 IEPE accelerometers (2x biaxial) | ME | MET (physical, PPR Rank 9), presented by ME |
| S2 FE stiffness identification <= 10 % | ME | PARTIAL (50 %) |
| S3 IP54 enclosure | ME | PARTIAL (50 %) |
| C1 OH-2 / Goulds 3196 envelope | ME | presented by the owner (datasheet table, sensor-mounting CAD) |
| C2 mains node, SELV IEPE 24 V / 4 mA | COE | presented by the owner (power tree) |
| C3 BOM <= 4,000 SAR | - | excluded by the reviewer |

## 6c. IS2 under the clarified wording v2 `[SYNTHETIC]` `[SIMULATION]`

**Clarified wording v2** (the coach requested a clarification on 2026-10-25; text in `Project architecture/overview/Specifications.md`). For the faulted bearing of every run-to-failure run, over every window in true health stages 1-3 after the AI warm-up, the disagreement e = |RUL_physics - RUL_AI| / L_ref (FE-twin physics RUL vs the deployed AI RUL, as a fraction of the bearing's reference (rated) life L_ref at its duty speed) shall satisfy (a) the median over bearings of each bearing's median e <= 18 % of L_ref and (b) the 95th percentile over bearings of each bearing's 95th-percentile e <= 63 % of L_ref.

| IS2, fresh `synth_v3_test4` (100 runs, scored once; 15,863 windows, 80 bearings) | v2 (a) median | v2 (b) p95 | verdict |
|---|---|---|---|
| v3b (deployed) | 16.3 % of L_ref (<= 18) | 49.2 % (<= 63) | **PASS** |
| v3c (research variant, not deployed) | 18.3 % | 47.8 % | FAIL |

Thresholds 0.18 / 0.63 were frozen on validation sets (`synth_v3_val4`, `synth_v3_val2`) and committed (355831c) before test4 was scored.
Six deliberately wrong AI RULs (x1.5, x0.6, +0.3 L_ref, time-reversed, constant 0.5 L_ref, lognormal noise) all fall outside the thresholds. Rank: re-scoring the saved predictions is Rank 6; the notebook also runs the deployed ONNX engine live on 20 bearings of test4 (median 14.8 %, p95 51.8 %: PASS), Rank 9.
Wording clarified per coach: v1 bounded the maximum ratio to RUL_AI, which grows without bound as RUL_AI approaches zero near failure (v1 values: v3b 233 %, v3c 216 % on test4). Method details: `AI-engine/reports/up2d_is2_20261024T225605Z.md`.

## 6b. COE-integration readiness: classifier adapted to COE-pipeline features `[SIMULATION]` `[SYNTHETIC]`

Raw samples from the COE rig simulator were run through the real causal `CoePipeline` (dataset `synth_coe`, 420 runs, split by run; the 106-run COE-test was scored once; `reports/coe_adapt_20261024T213622Z.md`, re-scored live in the notebook, "Additional items", part A).

| COE-test, per bearing-window | macro-F1 | bearing-2 false-fault | bearing-1 false-fault | missed-fault | cage F1 |
|---|---|---|---|---|---|
| v3b as deployed | 0.584 | 82.1 % | 74.7 % | 2.2 % | 0.24 |
| gate alone (ablation) | 0.725 | 4.4 % | 2.6 % | 13.9 % | 0.25 |
| retrain alone (ablation) | 0.777 | 53.0 % | 39.5 % | 15.4 % | 0.79 |
| **selected: retrain on synth_v3 + COE-train (COE weight 0.6) + cross-bearing gate (tau 0.6)** | **0.875** | **0.6 %** | **0.0 %** | 22.8 % | 0.83 |

Supporting evidence for S4-S6 and the hardware-day swap (not a counted item). Hardware day: re-fit `tau` (healthy-partner commissioning data), the stage cut and the classifier with rig seeded faults. The adapted classifier and contract (`model_contract_v3b_coe.json`) serve COE-pipeline sources; the deployed v3b contract is unchanged.

## 7. COE hardware-day swap steps

1. Firmware implements C1-C9 (STM32 runs the pipeline, sends the 152-byte payload via UART -> ESP32-C6 -> Wi-Fi -> UDP 5005); `aiengine/coe` is the golden reference.
2. Cross-check: dump raw blocks (`record_blocks`, `.npz` layout), replay with `CoePipeline(FileReplaySource(file))`, compare features with the firmware payload for the same window.
3. Enter commissioning values (sensitivity/offset per channel, proximity sensitivity, ADC range, speed band, per-revolution deviation).
4. Optional raw bring-up stream via `HardwareSource.blocks()`.
5. Re-measure on hardware and replace the `[CALCULATED]`/`[MEASURED host]` entries: t20 to payload-ready on the STM32 (S5), sample count and overruns (S4), the real Wi-Fi hop (IS1), window loss.
Nothing downstream (payload codec, backend, models) changes. Models should then be re-evaluated on rig data (fine-tuning on real fault data is the step that turns the `[SYNTHETIC]` S7/IS3 rows into measured ones).
