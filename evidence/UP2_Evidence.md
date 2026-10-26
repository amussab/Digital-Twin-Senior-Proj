# UP2 evidence summary (Team M001, 25/26 Oct 2026)

Real-Time Edge Digital Twin for Bearing Fault Diagnosis and RUL Prediction in OH-2 Centrifugal Pumps.
Standalone summary for the coach. The numbers here are recomputed live by `AI-engine/notebooks/UP2_Evidence.ipynb` (HTML copy:
`UP2_Evidence.html`); the master table and figures are stored in `evidence/up2_figures/`.

## 1. Status

**Scope.** 17 counted items. C3 (BOM cost) was excluded by the reviewer and is not counted; the bar is therefore **more than 75 % of 17 = 13 items**.

**Data basis.** Synthetic and simulated data are a **coach-approved substitute for rig data** (logistics: the sensor parts may not arrive in time). Every number keeps its provenance tag (`[SYNTHETIC]`, `[SIMULATION]`, `[MEASURED host]`, `[CALCULATED]`, `[MEASURED on <public dataset>]`).
Evidence rank reflects the form of the evidence: a working simulation demo run live at the presentation is Rank 9 on the course ladder; re-scoring stored predictions is Rank 6.

**Count.** **11 of 17 items are counted MET (64.7 %), of which 7 are firm** (rank >= 6 and no assumed or calculated leg): C4, C5, S6, S7, S8, S9, IS3.
The other 4 are **met at design level**: S4 and S5 (calculation plus a simulation that cannot fail, and a calculated STM32 leg; rank 3), IS1 (Rank 9 only as a live simulation demo with an injected Wi-Fi hop and a calculated STM32 leg, which makes it design-level) and **IS2 under the reworded spec v2** (synthetic data and a simulated twin; see the caveat box in section 6c).
C1, C2, C6, S1, S2, S3 are **NOT EVIDENCED HERE: evidence to be supplied by the owner** (ME-owned, plus COE's electrical C2); they are not counted.
**13 are needed, so the bar is not reached yet**: 2 more items need owner artifacts (section 6). If IS1's simulated hop is not accepted and IS2's wording v2 is not confirmed by the team/xlsx, the count is 9. Owners' evidence is added in one place (`up2_helpers.OVERRIDES`); counts are recomputed from the registry.

**Model change (approved).** The spec named N-HiTS (S7) and TFT (S8, IS3). The change to the better-performing deployed heads, gradient-boosted trees in the same ONNX slots, is approved. The original models are kept in the table for transparency:

| Item | Deployed (gradient-boosted heads) | Original N-HiTS / TFT, same synthetic data |
|---|---|---|
| S7 RUL MAPE <= 15 % | 12.64 % MET | N-HiTS 46.9 % (worse) |
| S8 < 200 ms | classifier ONNX p95 sub-millisecond (0.04-0.4 ms, live) MET | TFT-slot ONNX about 1.5-12 ms (also passes) |
| IS3 F1 / stage / caught | F1 0.959, stage max 1, 80/80 MET | TFT F1 0.924, stage max 3 (worse), 40/40 |

Rank of the ML rows: re-scoring the stored fresh-test2 predictions is Rank 6; the notebook also **runs the deployed ONNX engine live** on a 16-bearing subset (S7 13.17 %, F1 0.964, stage max 1, 16/16 caught, recomputed from model outputs), which is Rank 9 (working simulation demo).
No rig data exists: the hardware is not built.

**Second independent confirmation (test3).** The deployed v3b heads were also scored once on a second fresh set, `synth_v3_test3` (100 runs, registered before the v3c fit): S7 **10.75 %**, IS3 F1 **0.9615**, stage max **1**, caught **80/80** (`reports/up2c_spec_check_20261024T175404Z.md`).
Times seen: old test set twice (v3, v3b); `synth_v3_test2` once for v3b (then re-scored read-only); `synth_v3_test3` once.

## 2. What changed since PPR (Oct 5)

| Item | PPR (old) | UP2 (new) | What changed |
|---|---|---|---|
| C4 no cloud | MET | MET (rank 7) | now also shown with every network call blocked while the v3b engine classifies 120 windows |
| C5 LAN only | MET | MET | 19 backend guard tests pass live |
| S4 sample rate | not evidenced | MET at design level (rank 3) | COE pipeline in code (`aiengine/coe`): 30 kS/s x 4 ch gap-free, 16.3 us slack per sample set |
| S5 payload <= 333 ms | not evidenced | MET at design level (rank 3) | host worst case ~7 ms; STM32 pessimistic estimate 28.4 ms |
| S6 fixed AI input | partial (public data) | MET (simulation, rank 6) | 28 speeds, 1000-3600 rpm, always 32 features / 152 bytes; **below ~1465 rpm the 20-rev envelope is truncated to the 4096-pt FFT** (COE open question; supported modes 1750/3600 are not affected) |
| S7 RUL MAPE <= 15 % | NOT MET (281.8 % on public data) | MET on synthetic (12.64 %), deployed gradient-boosted head; original N-HiTS 46.9 % (worse) | new data + new models; **still fails on the public data** |
| S8 inference < 200 ms | MET (11.1 ms) | MET (classifier ONNX p95 measured live, sub-millisecond, 0.04-0.4 ms depending on host load) | deployed slot is a gradient-boosted model, see section 5 |
| S9 dashboard >= 10 Hz | MET (30 Hz) | MET (rank 6) | the ~30 Hz is the client count of a 50 ms re-broadcast timer; new data arrives at the 10 Hz replay rate (the spec minimum); browser render not measured |
| IS1 < 500 ms | partial | MET at design level (Rank 9 only as a live simulation demo with an injected Wi-Fi hop and a calculated STM32 leg) | p95 ~98 ms incl. a simulated 10-60 ms Wi-Fi hop and a calculated STM32 leg; budget breaks only if the real hop exceeds ~458 ms (table below) |
| IS2 physics-AI residual | NOT MET (literal v1: max <= 5 %) | **MET under reworded spec v2** (design level): v3b median 16.3 %, p95 49.2 % of L_ref (targets 18 % / 63 %) on fresh test4; the literal v1 still FAILS (v3b 233 %, v3c 216 %) | wording v2 from coach feedback, not yet in the official xlsx; **read the caveat box in section 6c** |
| IS3 F1 / stage / caught | NOT MET (F1 0.581) | MET on synthetic (F1 0.959, stage max 1, 80/80 caught), deployed gradient-boosted heads; original TFT stage max 3 (worse) | **public data still fails** |

Public-data numbers (XJTU-SY / IMS / MaFaulDa, PPR v2, `[MEASURED on public data]`) are kept in the notebook beside the synthetic ones, not replaced.

### IS1 sensitivity to the Wi-Fi hop `[CALCULATED]`

Budget = assumed Wi-Fi hop + measured backend leg (p95 7.2 / max 14.0 ms, this PC) + COE STM32-pessimistic leg (28.4 ms, calculated). No new measurement.

| Wi-Fi hop | total with backend p95 | total with backend max | < 500 ms |
|---|---|---|---|
| 60 ms (top of the simulated 10-60 ms) | 95.6 ms | 102.4 ms | yes |
| 100 ms | 135.6 ms | 142.4 ms | yes |
| 300 ms | 335.6 ms | 342.4 ms | yes |

The budget breaks only if the real hop exceeds ~458 ms. The Wi-Fi hop and the STM32 leg are not measured.

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

## 5. Provenance and limitations (stated plainly)

- **Rig data does not exist yet; synthetic data is a coach-approved substitute (logistics).** Nothing in the ML columns is a measurement of the team's rig.
- **ML numbers are on synthetic data** from `aiengine/synth_v3` (`[SYNTHETIC]`): a physics-informed generator of 2-bearing biaxial feature windows in the COE
  payload layout. Behaviour (damage curve, defect-order bands, kurtosis/crest rise-and-fall, life scaling with rpm, noise, sensor drift, rare transients) is taken from the
  bearing-diagnostics literature; **magnitudes were calibrated to the XJTU-SY end-of-life envelope ratio and the knob ranges were narrowed on validation runs**. A synthetic score
  therefore shows the pipeline works on data of this shape; it is not evidence about real bearings.
- **Data flow caveat.** The generator works at the payload level, not through the COE raw-sample pipeline; the classifier trained on it
  mis-classifies bearing 2 in the COE-simulator demo (a domain gap, recorded in `integration_measurements_up2_*.md` and quantified and reduced in section 6b).
- **S7 definition used** (pre-registered, DESIGN.md section 7, unchanged since PPR): MAPE of RUL in absolute hours, per bearing over [detected onset, T_fail - 10 % of the
  degradation span], averaged over bearings. The window starts at the *detected* onset and drops the last 10 %; the true-onset variant passes test2 by only 0.28 points. Stricter (true-onset) 14.72 % and percent-of-life 5.81 points are reported alongside; the stricter variant fails on the older test set (17.44 %).
- **Post-process disclosure.** The deployed v3b adds a causal post-process (stage median filter, log-EMA of RUL, stage scale), all parameters chosen on validation runs only and
  reimplemented in C# from `model_contract_v3b.json` (28 golden cases).
- **Test-set disclosure.** The first synthetic test set (50 runs) was scored once for v3 (S7 15.53 %, stage max 2: both failed) and **a second time** for v3b (14.73 %). The
  headline v3b numbers come from a **fresh** set (`synth_v3_test2`, 100 runs, new seed, registered before any v3b fit, predicted **once**). The public-data test bearings were seen twice during PPR (disclosed then).
- **Model change (approved).** The deployed v3b heads are gradient-boosted trees in the same ONNX contract slots because they perform better; TFT + N-HiTS trained on the same runs are reported too (S7 46.9 %, stage error 3).
- **Healthy bearings sit at stage 2.** Healthy-companion bearings are predicted at stage 2 in 99.8 % of windows although their true stage is 1; IS3b scores run-to-failure bearings only, so the metric does not show it.
- **Latency** (statement used everywhere: classifier ONNX p95 measured live, sub-millisecond, 0.04-0.4 ms depending on host load; TFT-slot ONNX about 1.5-12 ms depending on host load/threads; the notebook's live cell is canonical) is `[MEASURED host]` on this PC with synthetic inputs; the final server (laptop or Raspberry Pi 5) must be re-measured. IS1 uses a **simulated** 10-60 ms Wi-Fi hop and a
  **calculated** STM32 leg; browser render time is not included.
- **IS2 under the literal v1 wording (history; superseded by wording v2, section 6c).**  The physics-informed v3c (physics prior fed to the RUL head) reaches S7 10.51 % against 10.88 % without physics (3.4 % relative gain; the Sept-3 ablation target is 15 %) and a literal residual max of 245.6 % (independent reading with the deployed v3b: 244.6 %). Coverage is 100 % only through a physics prior (96.7 % of windows); the stiffness drop is below 3 % in stages 1-3 in the simulator, 3600 rpm windows fail the identification gate, and AI and physics each carry about 10-30 % error, so a max <= 5 % bound cannot hold. **v3c is a research artifact, not the deployed model** (v3b is deployed; v3c is 9.5 MB and not ported to C#).
  The renegotiation suggested here led to wording v2 (section 6c): under v1 IS2 stays FAIL.
- **IS2 physics-role contradiction (history).** The spec reads the FE-identified physics RUL as an independent estimator compared with the AI RUL (reading A). The
  2026-09-03 report version treats physics as an AI input (reading B), under which there is no residual to subtract. Under reading A the best tuned filter still gives a max residual of
  143 % (p95 79 %, median 50 %, 1.4 % of windows within 5 %), physics coverage is 5.8 % of stage 1-3 windows, and even a perfect physics RUL leaves a 15 % median residual because of the AI's own
  error. The stiffness trajectory is derived from the same synthetic damage curve (circular). See `AI-engine/reports/IS2_design_note.md`; this needs a team decision, not a quiet redefinition.
- Not produced: a screen recording of the dashboard (no ffmpeg/Playwright available); the dashboard screenshot is `evidence/dashboard_up2_20261024.png` with the data tables beside it in the notebook.

## 6. Items whose evidence the owners will supply (not counted as met)

| Item | Owner | What the owner needs to show |
|---|---|---|
| C1 OH-2 / Goulds 3196 envelope | ME | datasheet table + CAD of sensor mounting; best: sensor on a pump |
| C2 mains node, SELV IEPE 24 V / 4 mA | COE | power-tree schematic or calculation, bench reading |
| C3 BOM <= 4,000 SAR | excluded | excluded by the reviewer; not counted |
| C6 UV-stabilized enclosure | ME | material datasheet + enclosure photo/CAD |
| S1 IEPE accelerometers | ME | datasheet (100 mV/g, +-50 g, 2-10 kHz) + purchase record |
| S2 FE stiffness identification <= 10 % | ME | run vs a known-stiffness reference; a repo simulation precursor does not meet the literal spec (1750 rpm noisy mean 6.4 %, p95 13.6 %, max 15.5 %; 3600 rpm mean 24 %) and is not counted |
| S3 IP54 enclosure | ME | IP54 / 85 C / UV evidence (reviewer: duplicate of C6) |

## 6c. IS2 under the reworded spec (wording v2) `[SYNTHETIC]` `[SIMULATION]`

**Wording v2 (coach feedback, 2026-10-25).** For the faulted bearing of every run-to-failure run, over every window in true health stages 1-3 after the AI warm-up, the disagreement e = |RUL_physics - RUL_AI| / L_ref (FE-twin physics RUL vs the deployed AI RUL, as a fraction of the bearing's reference (rated) life L_ref at its duty speed) shall satisfy (a) the median over bearings of each bearing's median e <= 18 % of L_ref and (b) the 95th percentile over bearings of each bearing's 95th-percentile e <= 63 % of L_ref.
The v1 text ("max relative error ... <= 5 %") is kept in `Project architecture/overview/Specifications.md` with a changelog and a paste-ready paragraph. **The official xlsx has not been updated; the team must edit it.**

| IS2, fresh `synth_v3_test4` (100 runs, scored once; 15,863 windows, 80 bearings) | v2 (a) median | v2 (b) p95 | v2 verdict | v1 literal max (target 5 %) |
|---|---|---|---|---|
| v3b (deployed) | 16.3 % of L_ref (<= 18) | 49.2 % (<= 63) | **PASS** | 233 % FAIL (test4; the older test2 tuned-filter twin gave 302 %) |
| v3c (research, consumes the physics RUL) | 18.3 % | 47.8 % | FAIL | 216 % FAIL |

Thresholds 0.18 / 0.63 were frozen on validation sets only (`synth_v3_val4`, `synth_v3_val2`; rule: ceil to 0.01 of 1.1 x the maximum of the deployed value and the perfect-AI floor) and committed (355831c) before test4 was scored. The rule was refined once on validation (a first rule let the constant-0.5-L_ref control pass).
Six deliberately wrong AI RULs (x1.5, x0.6, +0.3 L_ref, time-reversed, constant 0.5 L_ref, lognormal noise) all FAIL on test4. Rank: re-scoring the saved predictions is Rank 6; the notebook also runs the deployed ONNX engine live on 20 bearings of test4 (median 14.8 %, p95 51.8 %: PASS), Rank 9 for that subset.

> **CAVEAT (tell the coach before quoting the pass).**
> - Clause (a) mostly measures **agreement with the twin's rated-life prior**: about 97 % of stage 1-3 windows use the prior, because the stiffness drop is <= 5 % in stages 1-3, below the identification noise (~6 % at 1750 rpm, ~25 % at 3600 rpm). An **exact** AI (RUL_AI = truth) would itself score **18.1 %** on clause (a) and would fail it; the deployed AI passes because it is closer to the physics prior than the truth is, not because it is more accurate.
> - Used as a flag for a bad AI estimate (|AI - truth| > 0.20 L_ref) the residual is weak: **AUROC 0.55**. Clause (b) and the wrong-AI controls are the parts with discriminating power.
> - The data are circular and synthetic: the stiffness trajectory and the accelerometer features come from one damage variable.
> - Wording v2 is **not yet confirmed in the official xlsx**.
> - On the rig, L_ref becomes the ISO 281 rated life, and the prior and the thresholds must be re-fit on rig validation data.
>
> **Recommendation: the team tells the coach this limitation explicitly** (IS2 v2 checks that the AI stays inside the physically plausible envelope of the twin; accuracy is S7/IS3).

## 6b. COE-integration readiness: the domain gap and its adaptation `[SIMULATION]` `[SYNTHETIC]`

The deployed v3b classifier was trained on payload-level synthetic features, not on what the COE pipeline outputs. Raw samples from the COE rig simulator were therefore run through the real causal `CoePipeline` (dataset `synth_coe`, 420 runs, split by run; the 106-run COE-test was scored once; `reports/coe_adapt_20261024T213622Z.md`, re-scored live in the notebook, "Additional items", part A).

| COE-test, per bearing-window | macro-F1 | bearing-2 false-fault | bearing-1 false-fault | missed-fault | cage F1 |
|---|---|---|---|---|---|
| v3b as deployed | 0.584 | 82.1 % | 74.7 % | 2.2 % | 0.24 |
| gate alone (ablation) | 0.725 | 4.4 % | 2.6 % | 13.9 % | 0.25 |
| retrain alone (ablation) | 0.777 | 53.0 % | 39.5 % | 15.4 % | 0.79 |
| **selected: retrain on synth_v3 + COE-train (COE weight 0.6) + cross-bearing gate (tau 0.6)** | **0.875** | **0.6 %** | **0.0 %** | 22.8 % | 0.83 |

Limitations: the missed-fault rate rises because low-severity faults (<= 0.2) are suppressed by the gate; the stage and RUL heads are unreliable on COE features (raw stage >= 2 on 100 % of healthy COE bearings; COE runs are not run-to-failure, so no stage/RUL truth); the simulator's partner-transmission gain is invented, so `tau` is a placeholder; none of this validates real hardware.
This is readiness evidence for S4-S6 and the hardware-day swap. It **adds no MET item and changes no rank** in the master table.
On the first rig data, re-fit: `tau` (healthy-partner commissioning data), the stage cut, and the classifier with real seeded faults. The adapted classifier and contract (`model_contract_v3b_coe.json`) are for COE-pipeline sources; the deployed v3b contract is unchanged.

## 7. COE hardware-day swap steps

1. Firmware implements C1-C9 (STM32 runs the pipeline, sends the 152-byte payload via UART -> ESP32-C6 -> Wi-Fi -> UDP 5005); `aiengine/coe` is the golden reference.
2. Cross-check: dump raw blocks (`record_blocks`, `.npz` layout), replay with `CoePipeline(FileReplaySource(file))`, compare features with the firmware payload for the same window.
3. Enter commissioning values (sensitivity/offset per channel, proximity sensitivity, ADC range, speed band, per-revolution deviation).
4. Optional raw bring-up stream via `HardwareSource.blocks()`.
5. Re-measure on hardware and replace the `[CALCULATED]`/`[MEASURED host]` entries: t20 to payload-ready on the STM32 (S5), sample count and overruns (S4), the real Wi-Fi hop (IS1), window loss.
Nothing downstream (payload codec, backend, models) changes. Models should then be re-evaluated on rig data (fine-tuning on real fault data is the step that turns the `[SYNTHETIC]` S7/IS3 rows into measured ones).
