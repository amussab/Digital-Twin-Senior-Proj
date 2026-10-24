# Team M001 — Constraints & Specifications (machine-readable copy)

Transcribed 2026-10-04 from `overview/252-Specifications-TEAM M001.xlsx` (the team's current
spec workbook, the version submitted with the department approval columns filled in). **The
xlsx is the source of truth; this file is a read-friendly copy.** If the workbook changes,
re-transcribe it here; don't edit numbers only in this file.

Column meanings: **OTS** = off-the-shelf. **Depts** = the departments the row is assigned to.
**Approval / reviewer comment** = what the department reviewers wrote in the approval columns.
Rows with an approval of "No" are still open.

## Constraints

| # | Short ID | Constraint (verbatim) | OTS | Depts (approval) | Reviewer comment |
|---|---|---|---|---|---|
| C1 | M1 | OH-2 (API 610) pumps — Goulds 3196 STi+MTi class: drivers ≤40 HP (STi) / ≈122 HP (MTi); 60 Hz grid → 1750 or 3500 rpm; housing envelope ≤10 mm/s RMS | No | ME (Yes) | — |
| C2 | M2 | Mains-powered node; SELV (≤50V DC) IEPE sensor excitation: 24V/4mA through the coax, AC-coupled to ADC range | No | COE (Yes) | — |
| C3 | M3 | Node BOM ≤4,000 SAR (placeholder, pending quotes); target sell price ≤8,000 SAR/node | No | ICS (**No**) | "Constraint not related to CS" |
| C4 | C1 | All real-time data acquisition and signal processing shall execute entirely on local hardware without reliance on cloud-based computation | No | COE (Yes) | — |
| C5 | I1 | Dashboard accessible only on the local network; no external/cloud exposure of vibration data | No | ICS (Yes) | — |
| C6 | — | Enclosure fabricated from a UV-stabilized, weather-resistant polymer (e.g., ABS/PETG) to protect internal electronics from dust, moisture, and UV/thermal degradation during continuous operation | No | ME (Yes) | — |

## Specifications

| # | Short ID | Specification (verbatim) | OTS | Depts (approval) | Reviewer comment |
|---|---|---|---|---|---|
| S1 | M4 | 2× biaxial (X,Y) IEPE sets, one per radial bearing housing = 4 channels: 100 mV/g, ±50g, flat 2–10 kHz; dual-mode input. No thrust bearing/axial load in the rig, so no Z-axis channel | **Yes** | ME (Yes) | — |
| S2 | M5 | Finite-element digital twin: 1D beam model identifies bearing stiffness K from measured shaft displacement with ≤10% error against a known-stiffness reference case | No | ME (Yes) | — |
| S3 | M6 | Enclosure IP54 (IEC 60529) with continuous operation, no material/seal degradation, ambient up to 85 °C, direct UV | No | ME (**No**) | "duplication with const. 6" |
| S4 | C2 | COE acquisition subsystem sample rate ≥25 kSamples/s per channel (4 channels: 2 radial bearings × X,Y) | No | COE (Yes) | — |
| S5 | C3 | After a 20-revolution window is acquired, COE processes it and makes the output payload available within 333 ms | No | COE (Yes) | — |
| S6 | C4 | COE transforms each 20-revolution window into an AI-ready representation with fixed dimensionality and defined ordering, independent of shaft speed over 1000–3600 RPM | No | COE (Yes) | — |
| **S7** | **ICS1 / I2** | **Pump-bearing RUL: MAPE ≤15% on held-out degradation window (= ≥85% mean accuracy) — pretrained NHiTS, fine-tuned** | No | ICS (Yes) | — |
| **S8** | **ICS2 / I3** | **Fault-classification inference <200 ms/window — pretrained TFT, fine-tuned** | No | ICS (Yes) | (comment cell holds an unrelated template example: "audit results … within 200 ms") |
| S9 | ICS3 / I4 | Dashboard & digital-twin presentation: UI refresh ≥10 Hz; physics RUL + AI RUL + fault ID updated together within N1's budget | No | ICS (**No**) | (comment cell holds an unrelated template example: "auditing dashboard shall update within 2 seconds") |

## Integrated specifications

| # | Short ID | Integrated specification (verbatim) | Depts (approval) | Reviewer comment |
|---|---|---|---|---|
| IS1 | N1 | Sensor → classify+RUL → dashboard <500 ms | COE (No: "not clear"), ICS (No: "Not clear. Specification should be specific and clear"), ME (No: "reflect the integration, how this is governed by all departments") | open |
| IS2 | N2 | **Wording v2 (coach feedback 2026-10-25; v1 text kept below; xlsx not yet updated by the team):** Physics-AI RUL agreement. For the faulted bearing of every run-to-failure run, over every window in true health stages 1→3 after the AI warm-up, the disagreement e = \|RUL_physics − RUL_AI\| / L_ref between the FE-twin physics RUL and the deployed AI RUL, as a fraction of the bearing's reference (rated) life L_ref at its duty speed, shall satisfy (a) median over bearings of each bearing's median e ≤ 18 % of L_ref and (b) 95th percentile over bearings of each bearing's 95th-percentile e ≤ 63 % of L_ref. | COE (No: "not clear"), ICS (No), ME (Yes) — approvals refer to v1 | open (v2 pending team/xlsx) |
| IS3 | N3 | Classification & stage performance: macro-F1 ≥85% over {Healthy, OD, ID, BD, CF}; stage error ≤1; any injected fault caught by stage 3 at the latest | COE (No: "not clear"), ICS (No), ME (Yes) | open |

## ICS-relevant reading of the table (for PPR planning)

`[CLAUDE-SYNTHESIS]`. This is the coordinator's reading, not a team decision.

- **ICS-unique constraint:** C5 (local-network-only dashboard). C3 is rejected for ICS by the
  reviewer. C4 (no cloud) is COE-assigned, but the AI engine also runs fully offline, which gives
  ICS evidence toward it.
- **ICS-unique specs:** S7 (RUL, N-HiTS), S8 (classification latency, TFT), S9 (dashboard ≥10 Hz;
  approval still "No").
- **Integrated specs ICS contributes to:** IS1 (latency), IS2 (residual, which needs a physics RUL
  from the ME FE twin), IS3 (macro-F1 / staging / early catch).
- **PPR rubric, per department, Exemplary:** *more than 1 constraint* and *1 specification* met
  with satisfactory-or-better evidence. ICS has only one approved unique constraint (C5), so
  Exemplary needs C4 (offline AI) counted as ICS evidence too. The team should confirm that this
  counting is acceptable.
- **Team table, Exemplary:** more than 50% of all constraints + specs + integrated specs met with
  satisfactory-or-better evidence. That is 18 items, so at least 10 must be met. **(PPR history.)** For UP2 the reviewer excluded C3 (BOM), so there are 17 counted items and the bar is more than 75 %, i.e. at least 13 met.

## IS2 wording history and changelog

**v1 (superseded, kept verbatim):** "Physics-AI residual: max relative error between the FE-identified physics RUL
and the AI RUL ≤5%, health stage 1→3". COE (No: "not clear"), ICS (No), ME (Yes).

**v2 (2026-10-25, coach feedback):** the text in the IS2 row above.

| Date | Change | Reason |
|---|---|---|
| 2026-10-25 | v1 → v2: normaliser RUL_AI → L_ref; aggregation "max over windows" → per-bearing median and 95th percentile, then median / 95th percentile over bearings; threshold 5 % → 18 % (median) and 63 % (tail) of L_ref; window set made explicit (faulted bearing of run-to-failure runs, true stages 1→3, after AI warm-up) | (1) Coach and COE/ICS reviewers: "not clear". (2) v1 divides by the AI's own RUL, so late windows with a small RUL_AI dominate the max (window max 233 % on test4 although the median is 27.5 %); a max over ~16 000 windows is set by a single window. (3) In stages 1→3 the stiffness drop is ≤ ~5 %, below the twin's identification noise (~6 % at 1750 rpm, ~25 % at 3600 rpm), so the physics RUL is the rated-life prior on ~97 % of windows; even an exact AI (RUL_AI = true RUL) shows a median disagreement of 15–18 % of L_ref with it, so v1's 5 % cannot be met by any AI. |

**Why L_ref and these statistics.** Expressing the error as a fraction of a fixed reference life (rather than of the
remaining life) is the prognostic-horizon convention (accuracy bands as a fraction of end-of-life; Saxena et al.,
"Metrics for Offline Evaluation of Prognostic Performance", IJPHM 1(1), 2010 `[CITED — check the exact wording before
quoting]`). The spec answers one safety question: *are the two estimators consistent to within a tolerable fraction of
the asset's life, so that a larger disagreement flags an implausible estimate?* The median clause bounds typical
agreement; the tail clause bounds how far any one bearing's estimates may drift from the twin.

**How the thresholds were set (validation only, `[SYNTHETIC]`+`[SIMULATION]`).** Rule: T = ceil to 0.01 of
1.1 × max over the validation sets of max(deployed-AI value, perfect-AI floor). Validation sets: `synth_v3_val4`
(fresh, registered 2026-10-25, never used to fit a model) and `synth_v3_val2`. Values (fraction of L_ref):
v3b MED 0.153 / 0.156, P95 0.477 / 0.536; perfect-AI floor MED 0.153 / 0.155, P95 0.457 / 0.571 → T_MED = 0.18,
T_P95 = 0.63. Six deliberately wrong AIs (×1.5, ×0.6, +0.3 L_ref, time-reversed, constant 0.5 L_ref, lognormal
noise σ 0.5) all FAIL on both validation sets. Thresholds were frozen in `AI-engine/aiengine/up2d_is2.py` and
committed (355831c) before `synth_v3_test4` was scored. Disclosure: the rule was refined once on validation (first
rule, margin 1.2 / 0.05 steps → 0.20 / 0.65, let the constant-0.5 L_ref control pass).

**Result on fresh `synth_v3_test4` (scored once, `AI-engine/reports/up2d_is2_20261024T225605Z.md`).** Deployed v3b:
MED 16.3 %, P95 49.2 % of L_ref → **PASS** (v1 literal max 233 % → FAIL). v3c (research): MED 18.3 % → FAIL. All six
controls FAIL. **Limitation:** the exact-AI floor on test4 is MED 18.1 % (would FAIL clause (a)): clause (a) measures
agreement with the twin's mostly-prior RUL, not AI accuracy (that is S7/IS3); clause (b) and the controls are the
discriminating parts. Circular on synthetic data (K(t) and the features share one damage variable); L_ref is the
synthetic median TRAIN life (72.9 h at 1750 rpm, 46.5 h at 3600 rpm) and becomes the ISO 281 rated life at the duty
point on the rig, with thresholds re-derived on rig validation data.

**Ready-to-paste text for the team xlsx (IS2 / N2 row; do not paste the reasoning above):**

> Physics-AI RUL agreement: for the faulted bearing of each run-to-failure run, over all windows in health stages 1→3
> after the AI warm-up, the disagreement between the FE-twin physics RUL and the AI RUL, expressed as a fraction of the
> bearing's reference (rated) life L_ref at its duty speed, |RUL_phys − RUL_AI| / L_ref, shall have (a) a median over
> bearings of each bearing's median ≤ 18 % of L_ref and (b) a 95th percentile over bearings of each bearing's 95th
> percentile ≤ 63 % of L_ref. Thresholds set from validation runs only; verified on a fresh held-out test set.
