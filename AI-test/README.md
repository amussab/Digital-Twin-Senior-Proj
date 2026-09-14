# ICS AI testbench — N-HiTS (RUL) and TFT (fault classification)

A runnable harness for the two ICS-owned models in
`Project architecture/ICS/ICS AI, Dashboard & Digital Twin Layer.md`:

| Spec | Model | Target |
|---|---|---|
| **ICS1** | N-HiTS, fine-tuned | RUL, MAPE ≤ 15% |
| **ICS2** | TFT, fine-tuned | Fault classification, < 200 ms/window |
| I1 (shared) | both | ICS leg of the < 500 ms end-to-end budget |
| I3 (shared) | TFT + staging | macro-F1 ≥ 85%, stage error ≤ 1, fault caught by stage 3 |

It trains both models, scores them against those specs, exports them to ONNX
with a machine-readable contract for the ASP.NET backend, and converts public
bearing datasets into the project's own feature space.

> **All numbers this produces are from synthetic data.** No fault-seeded or
> run-to-failure data from the team's rig exists yet. This verifies that the
> pipeline, models, latency path and export boundary behave as designed. It is
> not a measurement of accuracy on a real bearing and must not be quoted as
> one. Every report the tool writes carries that statement on its face.

---

## Quick start

```bash
cd AI-test
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe app.py selftest    # no model needed, ~5 s
.venv/Scripts/python.exe app.py all         # generate, train, evaluate, export (~6 min CPU)
```

| Command | What it does |
|---|---|
| `app.py selftest` | Payload codec, feature contract, bearing-order consistency, DSP chain |
| `app.py generate` | Synthetic run-to-failure corpus |
| `app.py train` | Train both models + fit the RUL calibration |
| `app.py evaluate` | Score against ICS1/ICS2/I1/I3, write a report to `reports/` |
| `app.py export` | ONNX + `model_contract.json` for the .NET host |
| `app.py replay` | Stream payloads through both models with per-window timing |
| `app.py datasets` | List public datasets; `--load femto` converts one |

---

## Results as measured

From a full run on held-out runs the models never saw
(`reports/latest.md` is regenerated each time):

| Spec | Target | Measured | Verdict |
|---|---|---|---|
| ICS1 — RUL MAPE | ≤ 15% | **38.4%** | **FAIL** |
| ICS2 — classification latency | ≤ 200 ms | 15.0 ms (p95) | PASS |
| I3a — macro-F1 | ≥ 0.85 | 0.980 | PASS |
| I3b — stage error within ±1 | ≥ 0.95 | 1.000 | PASS |
| I3c — caught by stage 3 | 1.00 | 1.000 | PASS |
| I1 — ICS inference leg | ≤ 500 ms | 20.0 ms (p95) | PASS |

Latency has ~10× headroom on a development laptop, and both ONNX graphs are
under 700 KB — so ICS2 and I1 are not at risk even if the Pi 5 wins the
still-open hardware decision. **ICS1 is the finding that matters.**

### Why ICS1 fails, and what would fix it

Not under-training. The harness computes the information floor on its own:

- **Information floor: 31.8% MAPE.** Across the faulted test runs, total life
  varies by 2.5×. A health index tells you *how far through its life* a bearing
  is, not *how long that life is*. Since `RUL = total_life × (1 − fraction_consumed)`,
  the relative error in RUL can be no better than the relative error in total
  life. **No forecaster of any quality beats this bound while RUL is expressed
  in absolute hours.**
- Measured 38.4% against that 31.8% floor: N-HiTS is already extracting most of
  what this formulation makes available. It does beat the population-prior-only
  baseline (44.1%), so the forecast is contributing — just not enough to matter.
- Expressed scale-invariantly, the same predictions give **15.9 percentage
  points** of percent-of-life-remaining error.

This is exactly risk 2 in `ICS External Datasets & Literature Precedent.md`
("RUL label scale mismatch"), measured. Published sub-15% MAPE results on
bearing benchmarks are real, but they come from models trained *directly* on
`features → RUL` as supervised regression, not from trend-extrapolating a
scalar health index.

**Three options, for the team to choose between:**

1. **Restate ICS1 scale-invariantly** — percent-of-life-remaining, where the
   current pipeline already achieves ~16 points. This is also what makes
   pretraining on FEMTO-ST/XJTU-SY coherent at all (risk 2's own
   recommendation), so it is the option that solves two problems at once.
2. **Change the formulation** to direct supervised RUL regression. This is what
   the sub-15% literature does. N-HiTS cannot be used this way as a
   pytorch-forecasting forecaster — a forecaster always receives its target's
   own history, and RUL history is a linear countdown the model would simply
   continue, and which is never observable on a live machine.
3. **Keep ICS1 at ≤15% absolute-hours MAPE** and accept that it is very likely
   unmeetable. Worth deciding deliberately rather than discovering at FPR.

Given that Dr. Al-Mohair already flagged the RUL spec by name at CDR, option 1
or 2 with the reasoning written up is a much stronger position than carrying a
"Met (design)" claim this harness contradicts.

---

## Design decisions worth knowing

**N-HiTS forecasts a health index, not RUL.** A pytorch-forecasting model is
given its target's own history as encoder input. Trained on RUL it would learn
to continue a countdown, and at inference on a real machine there is no RUL
history to continue. It therefore forecasts an observable health index, and
`rul.py` solves for when that forecast crosses a failure threshold.

**The classifier selects the regressor's failure threshold.** A cage defect
never reaches the health index an outer-race defect reaches at the same true
damage — measured end-of-life values run 0.74 (cage) to 1.01 (outer race). A
single pooled threshold would systematically over-predict life for weak fault
types, so TFT's predicted class picks the threshold. This is the one place the
two models are coupled.

**Splits are by run, never by window.** Windows from one bearing's history are
strongly autocorrelated; a random window split would put near-duplicates on
both sides and report an accuracy the system will not reproduce.

**Labels follow a detectability policy.** A faulted run is labelled `healthy`
until the defect is physically detectable (health stage 2). Labelling window 0
with the eventual fault type would ask the classifier to predict the future
from noise, and would inflate every metric computed against those labels.

**Class weights are inverse-frequency.** That policy makes `healthy` dominant;
unweighted, the classifier maximises accuracy by under-calling faults — the
exact failure mode Ch. 2.3 identifies as the ethical risk. `missed_fault_rate`
is reported separately from macro-F1 for the same reason.

---

## The .NET boundary

`app.py export` writes `artifacts/`:

- `nhits.onnx` (624 KB), `tft.onnx` (692 KB)
- `model_contract.json` — everything the C# side needs, so nothing is hardcoded
  there: input tensor names and column order, categorical encodings, the class
  labels **in output-channel order**, every feature-engineering constant, the
  payload layout, and the RUL calibration.

Both graphs are verified against PyTorch after export (max abs diff 7.5e-09 for
N-HiTS, 9.5e-07 for TFT). An export that loads but computes something else is
worse than one that fails, so parity is checked, not assumed.

ONNX constant-folding prunes unused inputs, so the C# call site is smaller than
the PyTorch signature suggests — the contract records exactly which inputs each
graph requires.

> **⚠️ Class order.** `NaNLabelEncoder` sorts labels alphabetically, so output
> channel 0 is `ball`, **not** `healthy` as `config.FAULT_CLASSES` declares.
> This bug was live in this codebase and cost a macro-F1 of 0.002 before it was
> caught — it is invisible in the training loss. The C# side must read
> `classes.labels` from the contract. Never hardcode the config's order.

Feature engineering (`features.py`) is deliberately restricted to what can be
re-implemented in C# in a few dozen lines — no fitted scalers, no pandas in the
per-window path. Its constants travel in the contract.

---

## Public datasets (risk 1)

The literature doc notes the reprocessing step is unscoped:

> "…they are not usable as pretraining input until they're run through the
> *same* feature-extraction pipeline the rig's own data will use."

`dsp.py` is that step — COE Components 3–7 re-implemented in Python,
parameterised by sample rate, passband and bearing geometry, so CWRU, FEMTO-ST,
XJTU-SY and the rig's own data all land in one identical 32-value contract.

It doubles as an independent check on COE's spec. Two results from `selftest`:

- **COE's bearing orders are internally consistent** — they imply an 8-element
  bearing at d/D = 0.2375, and all four orders reproduce from that geometry to
  within 0.0065 order.
- **The DSP chain recovers seeded faults**: synthesised BPFO, BPFI and BSF
  impacts each come back dominant in the right order family (≈98% of band
  energy), and a healthy signal shows no dominance with kurtosis ≈ 3.

Independent validation of the geometry code: the derived CWRU 6205 orders
(FTF 0.40, BSF 2.36, BPFO 3.58, BPFI 5.42) match CWRU's published values.

`app.py datasets` lists what each dataset can support. Nothing is downloaded
automatically; missing data prints the corrected URLs from the literature doc.

> **⚠️ FEMTO-ST geometry is a placeholder** and its derived orders should not be
> trusted until checked against the dataset's own documentation. CWRU and
> XJTU-SY figures match published values.

**Channel-count gap (risk 1, unresolved).** FEMTO-ST and XJTU-SY have two
accelerometers on one plane; this project has four across two. The default
policy **refuses** rather than padding, because duplicating one plane onto the
other teaches the model that plane asymmetry is always zero — suppressing
exactly the feature that localises a fault to one bearing. The recommended path
is a 2-channel (16-feature) variant for pretraining, extended to 4 channels at
fine-tuning. `--mirror-plane` exists for smoke-testing and is flagged as
fabrication.

**Domain adaptation (risk 3) is not implemented here.** Kumar et al. 2024's
DAN-vs-SEDA result (99.55% vs 24.77% on the same synthetic→real transfer)
is the reason it cannot be assumed — the transfer has to be *measured*. Doing
that needs the target-domain data, which does not exist yet.

---

## On the word "pretrained"

`pytorch-forecasting` supplies the N-HiTS and TFT *architectures*; it ships no
pretrained weights for either. "Pretrained, fine-tuned" is accurate for the
team's actual plan — pretrain on FEMTO-ST/CWRU, fine-tune on rig data — and
inaccurate if a reader takes it to mean off-the-shelf weights. Worth one
clarifying clause wherever it appears, since a reviewer who checks the library
will find no pretrained weights in it. As of now neither pretraining nor
fine-tuning has run: both models here are trained from initialisation on
synthetic data.

---

## Files

| File | Role |
|---|---|
| `config.py` | Single source of truth — feature layout, bearing orders, spec targets |
| `payload.py` | COE's 152-byte wire format |
| `dsp.py` | COE Components 3–7 in Python; bearing orders from geometry |
| `synth.py` | Synthetic run-to-failure generator |
| `features.py` | Host-side feature engineering + health index (C#-portable) |
| `data.py` | Labelling policy, run-level splits, TimeSeriesDataSets |
| `models.py` | N-HiTS and TFT construction, weighted cross-entropy |
| `train.py` | Training + RUL calibration |
| `rul.py` | Health-index forecast → RUL via threshold crossing |
| `evaluate.py` | Spec-check harness and report writer |
| `export_onnx.py` | ONNX export, parity check, .NET contract |
| `external.py` | CWRU / FEMTO-ST / XJTU-SY adapters |
| `selftest.py` | Checks needing no model or downloads |
| `replay.py` | Per-window live replay |

---

## Open items this testbench depends on

1. **Payload size.** COE's working doc specifies 152 bytes; the FDR deck says
   168. `payload.py` implements 152 and rejects any other length loudly. The
   C# parser should not be finalised until this is settled.
2. **The §2 physics contradiction** in the ICS layer doc — physics as model
   input vs. an independent RUL compared via residual — is still open and
   decides what I2 means. This harness takes no position on it.
3. **Deployment hardware** (laptop vs. Pi 5) is open. Latency here is from a
   development laptop; re-run on the chosen target before quoting a figure.
4. **ICS1's target** — see the three options above.
