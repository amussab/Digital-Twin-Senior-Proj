# UP2 evidence summary (Team M001, 25/26 Oct 2026)

Real-Time Edge Digital Twin for Bearing Fault Diagnosis and RUL Prediction in OH-2 Centrifugal Pumps.
Standalone summary for the coach. The numbers here are recomputed live by `AI-engine/notebooks/UP2_Evidence.ipynb` (HTML copy:
`UP2_Evidence.html`); the master table and figures are stored in `evidence/up2_figures/`.

## 1. Status in one paragraph

**10 of 18 items are MET** (55.6 %), 1 is NOT MET (IS2), 7 have **no evidence from ICS or COE** (C1, C2, C3, C6, S1, S2, S3: owned by ME, plus COE's
electrical C2). That is **below the 75 % bar (14 of 18)**; four more items need an owner's artifact (section 6). If the simulated Wi-Fi hop of IS1 is
not accepted, the count is 9. The ICS department's own **1 constraint + 2 specs (C5, S7, S8) are all met**, on the evidence forms stated below.
No rig data exists: the hardware is not built, so ML results are on synthetic data and COE results are simulation / host measurements.

## 2. What changed since PPR (Oct 5)

| Item | PPR (old) | UP2 (new) | What changed |
|---|---|---|---|
| C4 no cloud | MET | MET | now also shown with every network call blocked while the v3b engine classifies 120 windows |
| C5 LAN only | MET | MET | 19 backend guard tests pass live |
| S4 sample rate | not evidenced | MET (simulation + calculation) | COE pipeline in code (`aiengine/coe`): 30 kS/s x 4 ch gap-free, 16.3 us slack per sample set |
| S5 payload <= 333 ms | not evidenced | MET (host measured, STM32 calculated) | host worst case ~7 ms; STM32 pessimistic estimate 28.4 ms |
| S6 fixed AI input | partial (public data) | MET (simulation) | 28 speeds, 1000-3600 rpm, always 32 features / 152 bytes |
| S7 RUL MAPE <= 15 % | NOT MET (281.8 % on public data) | MET on synthetic (12.64 %) | new data + new models; **still fails on the public data** |
| S8 inference < 200 ms | MET (11.1 ms) | MET (0.3 ms classifier slot) | deployed slot is a gradient-boosted model, see section 5 |
| S9 dashboard >= 10 Hz | MET (30 Hz) | MET (30 Hz) | re-measured with the v3b models running |
| IS1 < 500 ms | partial | MET (simulation) | p95 ~98 ms incl. a simulated 10-60 ms Wi-Fi hop and a calculated STM32 leg |
| IS2 residual <= 5 % | NOT MET | **NOT MET** | max residual 143 % (median 50 %) on synthetic; see section 5 |
| IS3 F1 / stage / caught | NOT MET (F1 0.581) | MET on synthetic (F1 0.959, stage max 1, 80/80 caught) | **public data still fails** |

Public-data numbers (XJTU-SY / IMS / MaFaulDa, PPR v2, `[MEASURED on public data]`) are kept in the notebook beside the synthetic ones, not replaced.

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
- Whole scorecard: `cd AI-engine/notebooks; ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute UP2_Evidence.ipynb --inplace --ExecutePreprocessor.timeout=-1` (about 3 min), or open the notebook and Run All.
- Individual checks: `cd AI-engine; .venv/Scripts/python.exe -m aiengine.up2_eval` (re-scores saved predictions, measures latency), `-m aiengine.coe.verify --quick`, `-m aiengine.up2_residual` (IS2), `-m pytest tests/coe tests/up2 tests/twin_up2 -q`; `dotnet test tests/DigitalTwin.Backend.Tests -c Release`.

## 5. Provenance and limitations (stated plainly)

- **Rig data does not exist yet.** Nothing in the ML columns is a measurement of the team's rig.
- **ML numbers are on synthetic data** from `aiengine/synth_v3` (`[SYNTHETIC]`): a physics-informed generator of 2-bearing biaxial feature windows in the COE
  payload layout. Behaviour (damage curve, defect-order bands, kurtosis/crest rise-and-fall, life scaling with rpm, noise, sensor drift, rare transients) is taken from the
  bearing-diagnostics literature; **magnitudes were calibrated to the XJTU-SY end-of-life envelope ratio and the knob ranges were narrowed on validation runs**. A synthetic score
  therefore shows the pipeline works on data of this shape; it is not evidence about real bearings.
- **Data flow caveat.** The generator works at the payload level, not through the COE raw-sample pipeline; the classifier trained on it
  mis-classifies bearing 2 in the COE-simulator demo (a domain gap, recorded in `integration_measurements_up2_*.md`).
- **S7 definition used** (pre-registered, DESIGN.md section 7, unchanged since PPR): MAPE of RUL in absolute hours, per bearing over [detected onset, T_fail - 10 % of the
  degradation span], averaged over bearings. Stricter (true-onset) 14.72 % and percent-of-life 5.81 points are reported alongside; the stricter variant fails on the older test set (17.44 %).
- **Post-process disclosure.** The deployed v3b adds a causal post-process (stage median filter, log-EMA of RUL, stage scale), all parameters chosen on validation runs only and
  reimplemented in C# from `model_contract_v3b.json` (28 golden cases).
- **Test-set disclosure.** The first synthetic test set (50 runs) was scored once for v3 (S7 15.53 %, stage max 2: both failed) and **a second time** for v3b (14.73 %). The
  headline v3b numbers come from a **fresh** set (`synth_v3_test2`, 100 runs, new seed, registered before any v3b fit, predicted **once**). The public-data test bearings were seen twice during PPR (disclosed then).
- **Model naming.** The spec says pretrained N-HiTS (RUL) and TFT (class). The deployed v3b heads are gradient-boosted trees placed in the same ONNX contract slots, because they met the targets
  on synthetic data; TFT + N-HiTS trained on the same runs are reported too and do **not** meet S7 (46.9 %) or IS3 stage error (3).
- **Latency** is `[MEASURED host]` on this PC with synthetic inputs; the final server (laptop or Raspberry Pi 5) must be re-measured. IS1 uses a **simulated** 10-60 ms Wi-Fi hop and a
  **calculated** STM32 leg; browser render time is not included.
- **IS2 (FAIL) and the open physics-role contradiction.** The spec reads the FE-identified physics RUL as an independent estimator compared with the AI RUL (reading A). The
  2026-09-03 report version treats physics as an AI input (reading B), under which there is no residual to subtract. Under reading A the best tuned filter still gives a max residual of
  143 % (p95 79 %, median 50 %, 1.4 % of windows within 5 %), physics coverage is 5.8 % of stage 1-3 windows, and even a perfect physics RUL leaves a 15 % median residual because of the AI's own
  error. The stiffness trajectory is derived from the same synthetic damage curve (circular). See `AI-engine/reports/IS2_design_note.md`; this needs a team decision, not a quiet redefinition.
- Not produced: a screen recording of the dashboard (no ffmpeg/Playwright available); the dashboard screenshot is `evidence/dashboard_up2_20261024.png` with the data tables beside it in the notebook.

## 6. Items not evidenced from ICS/COE (not counted as met)

| Item | Owner | What the owner needs to show |
|---|---|---|
| C1 OH-2 / Goulds 3196 envelope | ME | datasheet table + CAD of sensor mounting; best: sensor on a pump |
| C2 mains node, SELV IEPE 24 V / 4 mA | COE | power-tree schematic or calculation, bench reading |
| C3 BOM <= 4,000 SAR | ME (ICS rejected it) | priced BOM with quotes/receipts |
| C6 UV-stabilized enclosure | ME | material datasheet + enclosure photo/CAD |
| S1 IEPE accelerometers | ME | datasheet (100 mV/g, +-50 g, 2-10 kHz) + purchase record |
| S2 FE stiffness identification <= 10 % | ME | run vs a known-stiffness reference; a repo simulation precursor does not meet the literal spec (1750 rpm noisy mean 6.4 %, p95 13.6 %, max 15.5 %; 3600 rpm mean 24 %) and is not counted |
| S3 IP54 enclosure | ME | IP54 / 85 C / UV evidence (reviewer: duplicate of C6) |

## 7. COE hardware-day swap steps

1. Firmware implements C1-C9 (STM32 runs the pipeline, sends the 152-byte payload via UART -> ESP32-C6 -> Wi-Fi -> UDP 5005); `aiengine/coe` is the golden reference.
2. Cross-check: dump raw blocks (`record_blocks`, `.npz` layout), replay with `CoePipeline(FileReplaySource(file))`, compare features with the firmware payload for the same window.
3. Enter commissioning values (sensitivity/offset per channel, proximity sensitivity, ADC range, speed band, per-revolution deviation).
4. Optional raw bring-up stream via `HardwareSource.blocks()`.
5. Re-measure on hardware and replace the `[CALCULATED]`/`[MEASURED host]` entries: t20 to payload-ready on the STM32 (S5), sample count and overruns (S4), the real Wi-Fi hop (IS1), window loss.
Nothing downstream (payload codec, backend, models) changes. Models should then be re-evaluated on rig data (fine-tuning on real fault data is the step that turns the `[SYNTHETIC]` S7/IS3 rows into measured ones).
