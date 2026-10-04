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
| IS2 | N2 | Physics-AI residual: max relative error between the FE-identified physics RUL and the AI RUL ≤5%, health stage 1→3 | COE (No: "not clear"), ICS (No), ME (Yes) | open |
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
  satisfactory-or-better evidence. That is 18 items, so at least 10 must be met.
