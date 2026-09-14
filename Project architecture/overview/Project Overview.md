# Real-Time Edge AI Bearing Monitoring System
## Project Overview

Extracted 2026-09-14 from the team's full project context (`Claude outputs/PROJECT_CONTEXT_EXPORT.md`)
and the FDR presentation. This is the cross-discipline, whole-project document — team, deadlines,
the final spec/constraint table, the one open contradiction that touches all three disciplines,
and the standing rules for evidence and report writing. **For the deep technical detail behind any
one discipline's blocks, see that discipline's own folder** — `COE/` (edge acquisition &
processing), `ICS/` (AI, fault classification & dashboard), `ME/` (physical system & FE digital
twin) — this file summarizes, it doesn't duplicate.

---

## 1. Project definition

**Title**: *"Real-Time Edge Digital Twin for Bearing Fault Diagnosis and RUL Prediction in OH-2
Centrifugal Pumps."*

**The problem**: unplanned rotating-machinery failures cost a median of ~$125,000/hour. Existing
tools force a choice between automated detectors that can't diagnose, and expert platforms that
need a certified analyst.

**The answer**: an edge-processed digital twin — accelerometers on the bearings feed two
fine-tuned AI models (RUL regression + fault classification), running entirely on the local
network. No cloud, no certified analyst required. (Whether a physics-based model is a third,
independent cross-check on the AI output, or feeds into the AI models as an input feature, is
**currently an open, unresolved contradiction — see §3 before writing or presenting anything that
touches this.**)

**Target machine**: OH-2 (API 610) centrifugal pumps, specifically Goulds 3196 i-FRAME STi/MTi
class (drivers up to 40 HP / ≈122 HP; 1750 or 3500 rpm at 60 Hz). Test rig: a Bently Nevada/Baker
Hughes RK4 rotor kit (motor + controller + Proximitor probe assembly) — not itself part of the
product, it stands in for the pump for bench testing.

**Six deliverables**:

1. Finite-Element Digital Twin (ME) — hand-coded 1D Euler-Bernoulli beam model; inverse-solves
   bearing stiffness K from measured shaft displacement.
2. Sensor Node Hardware (ME/COE) — dual biaxial (X,Y) IEPE accelerometer sets, one per radial
   bearing housing, with conditioning and edge acquisition.
3. Physics-Based RUL Estimator (ME) — tracks identified stiffness K(t); degradation trend gives
   an "AI-independent" remaining-useful-life estimate. *(§3 contradiction lives here.)*
4. Fine-Tuned N-HiTS + TFT Models (ICS) — pretrained models fine-tuned for RUL regression and
   fault classification, served from the always-on server.
5. Local-Network Web Dashboard (ICS) — live fault classification, physics RUL, AI RUL, and their
   residual on one multi-panel view; no cloud exposure.
6. Final Documentation Package (all) — report series, finalized specification set, project
   notebook/Gantt chart records.

**Elevator pitch numbers**: $125K/hr median downtime cost; <500ms sensor-to-dashboard latency
budget (final ID **I1**).

---

## 2. Team & roles

| Name | Student ID | Dept | Owns |
|---|---|---|---|
| Abdullah Mohammed Ewidah | 202244620 | COE | Edge Acquisition & Processing — see `COE/` |
| Abdulrazaq Musaab Alsayed Ahmad ("Mussab") | 202267720 | ICS | Dashboard & digital-twin presentation — see `ICS/` |
| **Muhammad Salah Jaddoua ("Jaddou")** | 202246240 | ICS | RUL regression (N-HiTS) + fault classification (TFT) — see `ICS/` |
| Abdelrahman Khaled Mohamed Elshenawy ("Shinawi") | 202269160 | ME | Physical System & Simulation — see `ME/` |

- **Coach**: Dr. Uthman Baroudi — ubaroudi@kfupm.edu.sa (COE, room 309, Mon). De facto COE
  reviewer.
- **Reviewers**: Dr. Hani Kaid Al-Mohair — hanik@kfupm.edu.sa (ICS reviewer); Dr. Fadi Al-Badour —
  fbadour@kfupm.edu.sa (ME reviewer).
- **⚠️ Al-Badour rejected the project outright at CDR** (May 21, 2026): *"The project is not
  accepted and none of the concepts was properly evaluated."* Harsher and more technical than the
  other two reviewers — forced most of the original spec/constraint rewrite (market-based cost,
  real pump targeting instead of a generic rotor kit, measurable failure criterion instead of a
  vibration alarm, per-department spec separation). Still the ME/FDR reviewer — treat him as the
  harshest, most technically literal grader.
- **Recurring reviewer theme across every round so far** (Scope Form, CDR, Update Presentation I):
  claims without evidence, generic stakeholders, unjustified constraints/specs. This is *the*
  thing to keep getting right.
- **Course-wide contact**: TEAM.Design@kfupm.edu.sa.

---

## 3. ⚠️⚠️ OPEN CONTRADICTION — physics model's role (blocks work in all three disciplines)

**Report version** (`report_draft_v1.docx`, corrected 2026-09-03): the physics model (bearing
kinematics/BCFs + Miner's cumulative-damage rule) is software **context feeding the AI models'
input** — physics-derived features engineered into N-HiTS's and TFT's input, not a separate
estimator. No independent physics RUL number exists to compare against the AI's output. The
integrated spec (then N2, now I2) was defined as an **ablation study**: N-HiTS trained with
physics-derived input features must beat an ablated DSP-only-feature baseline by ≥15% relative
MAPE. Under this framing, the Ch.2.3 ethics safety argument rests on TFT's attention-based
confidence signal, not a residual check.

**FDR presentation version** (slides 3, 4, 5, 13, 16, 17 — internally consistent with each other):
the physics model is an **independent, parallel RUL estimator**. Deliverable #3 is literally named
"Physics-Based RUL Estimator," described as giving an "AI-independent" RUL estimate. Block B3
("Stiffness RUL Estimator," owned by ME) is drawn as its own block, separate from ICS's N-HiTS,
with a dashed arrow captioned: *"physics RUL feeds the residual check... against [the] AI RUL."*
The final table's **I2 = "Physics-AI RUL residual ≤5%, stages 1→3."** ICS's own B9 mechanism
description says the .NET backend "combines the AI outputs with B3's physics RUL and [the]
residual into one payload."

**Why this is a whole-project problem, not just ICS's**: it changes what ME's physics model has to
output (a standalone RUL number vs. a feature set), what COE's payload needs to carry to the host
(just DSP features vs. also room for ME's own RUL value), what ICS's model inputs and dashboard
actually are (§ handled in `ICS/`'s own file), and what the Ch.2.3 ethics section's safety
mechanism is. Three explanations are possible and this export can't tell which is true: (a) the
Sept-3 "correction" was itself a mistake; (b) the FDR deck is stale and doesn't reflect a real team
decision; (c) the team reverted after Sept 3 and it never made it back into this workspace.

**Do not silently pick one — ask the team directly**, then propagate that single answer everywhere
it touches: report text, both diagrams, Table 1's I2 row, Ch.2.3 ethics, and each discipline's
folder in this repo.

---

## 4. System architecture, at a glance

Three disciplines, one pipeline, nine blocks (B1–B9), final IDs per the FDR's summary table
(§5 below has the full spec table; each discipline's own folder has the block-by-block mechanism,
evidence, and pass/fail criteria):

**ME · Physical System** — B1 Instrumented Rotor & Sensor Mounting (M1) → B2 Finite-Element
Digital Twin, identifies stiffness K (M2) → B3 Fault Database & RUL Regression / "Stiffness RUL
Estimator" (M3). Full detail: `ME/`.

**COE · Edge Acquisition & Processing** — B4 IEPE Sensor Conditioning (C2) → B5 Edge DAQ &
Windowing (COE1) → B6 Local Processing + WiFi, builds the fixed-size payload (COE2, COE3, C4).
Full detail: `COE/`.

**ICS · AI & Dashboard** — B7 N-HiTS (fine-tuned), continuous RUL regression (ICS1) + B8 TFT
(fine-tuned), per-window fault classification (ICS2), both fed by COE's payload → B9 .NET
Dashboard & Digital Twin Backend, combines everything and serves the web dashboard (ICS3). Full
detail: `ICS/`.

**Cross-discipline links**: B3(ME)→B9(ICS) is the disputed link — see §3. B6(COE)→B7/B8(ICS) is
the WiFi feature-window handoff, within I1's <500ms end-to-end budget.

**⚠️ Known internal inconsistency in the FDR deck itself**: slide 4's block diagram still labels
blocks with an older ID scheme (M5, M6, C2, C4, I2, I3, I4, N2) that doesn't match the final table
on slide 17 (M2, M3, COE1, COE3, ICS1, ICS2, ICS3, I2) — slide 4 evidently wasn't updated when the
scheme was finalized. Don't copy slide 4's ID labels into anything new; slide 17 is authoritative.

**Where FDR says the project stands** (slide 18): conceptual design complete — all 6 constraints,
12 specs, 3 integrated specs justified and traceable to the final table; full block diagram
defined. Next: rig assembly, fault-injection data collection, model fine-tuning, dashboard
integration.

---

## 5. Final constraint/specification table (supersedes every earlier version)

**⚠️ The ID scheme changed completely** between the Sept-3 "v4" table (still what's in
`build_report.py` and the report draft) and this FDR-final table. The report's Table 1 has **not**
been rebuilt to match — required before the Second Report Instalment ships, independent of §3.

| # | Constraint / Specification | Off-the-shelf | ME | COE | ICS |
|---|---|---|---|---|---|
| C1 | Target machine & sensor range: OH-2 (API 610), ≤10 mm/s RMS | | X | | |
| C2 | SELV ≤50V DC IEPE excitation, 24V/4mA | | X | X | |
| C3 | BOM ≤4,000 SAR; sell price ≤8,000 SAR/node | | X | | X |
| C4 | All real-time DAQ/processing on local hardware, no cloud | | | X | |
| C5 | Dashboard: local network only, no external exposure | | | | X |
| C6 | ≤1% window loss over 1-hr run, rig-floor RF | | X | X | |
| M1 | 4-ch biaxial IEPE, 100mV/g, ±50g, flat 0.5Hz–10kHz | **Met** | X | | |
| M2 | FE beam model identifies K, ≤15% error vs. known case | | X | | |
| M3 | Detect/localize {OD,ID,BD,CF}; failure = K drop ≥25% or stage 6 | | X | | |
| COE1 | ≥4 channels @ ≥25 kS/s/ch simultaneous acquisition | | | X | |
| COE2 | ≤333 ms per 4-ch, 20-rev window | | | X | |
| COE3 | Fixed-dimension window across 1000–3600 RPM | | | X | |
| ICS1 | RUL: MAPE ≤15% — pretrained N-HiTS, fine-tuned | | | | X |
| ICS2 | Fault classification <200ms/window — pretrained TFT | | | | X |
| ICS3 | UI refresh ≥10Hz; joint physics+AI+fault update | | | | X |
| I1 | End-to-end latency <500ms, sensor→dashboard | | X | X | X |
| I2 | **Physics-AI RUL residual ≤5%, stages 1→3** *(disputed — §3)* | | X | X | X |
| I3 | Macro-F1 ≥85%; any fault caught by stage 3 | | X | X | X |

18 rows total (6 constraints + 12 specs [9 dept-owned + 3 integrated] — hits the guideline's 6–12
constraint floor exactly, and the ≤12-spec cap exactly). M1 is the project's one off-the-shelf item
(1/18 ≈ 5.6% off-shelf, under the 25% ceiling and the ≤1-per-dept cap). Every department has ≥1
uniquely-owned constraint and exactly 3 uniquely-owned specs.

**Numeric deltas from the Sept-3 "v4" table worth flagging in the report's own changelog**: a 6th
constraint (C6, window-loss/RF reliability) was added, resolving the "one short of the floor" gap
that was previously open; ICS's RUL MAPE target loosened from ≤10% to ≤15%; ME's failure criterion
changed from Miner's-rule damage (D≥0.95) to a stiffness-drop threshold (K drop ≥25% — itself a
symptom of §3's contradiction); accelerometer channel count is 4 (biaxial), not the earlier
6-channel triaxial draft, consistent with the rig having no axial load.

Citable sources already in hand for References (IEEE format): Goulds 3196 "Modular
Interchangeability" chart; ISO 281/Miner's rule; ISO 20816-3 (severity reference only); PHM09/
FEMTO-ST run-to-failure bearing datasets; Challu et al. AAAI-23 (N-HiTS); Lim et al. *IJF* 37(4)
2021 (TFT); Fentaye & Kyprianidis 2024, *Aeronautical Journal* 128(1325) (TFT on gas-turbine
prognostics); pytorch-forecasting docs; Analog Devices AD7606 datasheet; SMACQ SRD-1104 manual;
RK4 rotor kit datasheet (Baker Hughes); Microsoft Learn ASP.NET Core SignalR docs; ONNX Runtime
docs; dotnet/core supported-os.md; McKinsey "Prediction at scale" (2024); IoT Analytics PdM market
highlights; IEC 62368-1:2023.

---

## 6. Architecture decision history (condensed)

### 6.1 Distributed node + server (handheld is DEPRECATED)

The Aug-31 handheld pivot (single Pi-5 unit, integrated 7″ touchscreen, battery) is dead. At
Update Presentation I (Sept 1), the coach pushed back live: *"what's the point of a handheld if I
have to stand next to the machine — I want to sit in my office and monitor my machines."* Landed
on: sensor node(s) near the pump broadcast over WiFi to a separate always-on server running the AI
engine + digital twin, serving a web dashboard reachable from any device on the local network.
**Report framing note**: write this around the coach's stated need for centralized/remote
monitoring — never "to boost the ME member's contribution" anywhere near the actual report.

### 6.2 AI model direction: pretrained N-HiTS + TFT (2026-09-02, real instructor consultation)

Following consultation with Dr. Moayad, the model direction changed from an earlier "dual-head
1D-CNN + LSTM" to pretrained N-HiTS and TFT from the `pytorch-forecasting` library, fine-tuned on
project data. Reasoning detail in `ICS/`. Genuine Expert Consultation material for Ch.3.1.

### 6.3 Hardware — resolved vs. still open, as of FDR

- **Node MCU**: resolved — ESP32-C6 (WiFi bridge) + STM32H755/Nucleo-H755ZI-Q (processing). No
  longer provisional.
- **Server hardware (laptop vs. Raspberry Pi 5)**: **still explicitly unresolved.** FDR slide 13
  says the always-on server is "hardware still being sized between a laptop and a Raspberry Pi 5."
- **Node power**: mains power via wall cable, folded into constraint C2 (SELV ≤50V DC, 24V/4mA
  IEPE excitation over coax).
- **Dashboard/UI**: resolved — a conventional multi-panel web telemetry layout, not a bespoke 2.5D
  cross-section SVG.
- **Enclosure**: resolved — 220×170×110mm PETG/ABS 3D-print, thermal analysis done (detail in
  `ME/`).
- **Filter topology/Keyphasor conditioning circuit**: still open (detail in `COE/`).

### 6.4 Stakeholders — framing rule (unchanged, still applies)

The real consultations are with the university instructors (coach + reviewers) — genuine Expert
Consultation material for Ch.3.1. The original CDR stakeholder table (Plant Reliability Managers /
Maintenance Planners / Maintenance Technicians) is research/industry-literature-derived, not from
real contacts. **Framing rule**: state customer needs as fact, cite the literature/market evidence
behind each one, never write a sentence stating the team *didn't* contact real stakeholders.
General rule: state what actually is/was done — never state what didn't happen.

---

## 7. Evidence quality standard (applies to every design claim, everywhere)

**Why this exists**: every review so far (Scope Form, CDR, Update Presentation I) made the same
complaint — claims without evidence, generic justification, unjustified constraints/specs.

**Two-tier provenance labeling** (working files only, not the final report): `[SOURCED: <doc,
page/slide/section>]` — from an official document or explicit user statement, quotable/checkable.
`[CLAUDE-SYNTHESIS]` — Claude's own reasoning/inference, not a citable fact. `[USER-STATED]` — a
decision/fact the user stated directly. Never let a synthesized claim read with the authority of a
sourced one.

**Strength-of-Evidence scale**: 0 Words only (never acceptable) · 1 Basic sketches/diagrams · 2
Calculations & predictions (supports feasibility, not prototype) · 3 Simulations & CAD (design,
not prototype) · 4 Picture of demo · 5 Video of demo · 6 Sim/CAD + data · 7 Picture of demo + data
· 8 Video of demo + data · 9 Physical prototype/working sim live at presentation. **Ranks 2–3 are
the legitimate ceiling right now** — confirmed by the FDR itself, which reports Rank 2–3 across
every discipline's evidence status. Rank 0 is banned.

**Evidence Chain procedure** (5 steps, every design choice, every time it's defended): (1) state
the requirement in checkable numeric form; (2) name the candidate — an actual part/method/
architecture, not a category; (3) prove it from a primary source — datasheet, standard clause,
cited figure (this is the step that actually earns Rank 2; skip it and the claim is Rank 0
regardless of prose); (4) state what else it gives or costs — interface, protocol, voltage,
latency, power, price; (5) show it fits the whole system — interface/power/timing compatibility
with its neighbors. If step 3 genuinely can't be completed yet, flag red rather than forcing a
placeholder through.

**Standing rules**: never claim a test was run, prototype built, or number measured when it
wasn't; every "MET (design)" claim needs a real calculation, datasheet reference, or cited
standard; where something isn't decided/verified, say so plainly — future-tense, in-progress
framing is explicitly allowed at conceptual-design stage; state what actually is/was done — never
state what didn't happen.

---

## 8. Report writing & color-coding conventions

**Layer 1 — Claude-authorship review coding (everywhere in the document)**: Orange (`#D97757`) =
Claude-drafted content. Red (`#FF0000`) = flagged for the user's attention — unresolved decision,
unverified number, anything needing judgment. Once reviewed/cleared, recolor to default/black —
that transition *is* the "reviewed and passed" signal.

**Layer 2 — Appendix E discipline color coding (Appendix E only)**: official requirement to
color-code tracked changes by discipline. Legend (avoids red/orange to never collide with Layer 1):
COE = Blue (Ewidah); ICS = Green (User + Abdulrazaq); ME = Purple (Elshenawy). State this legend
explicitly at the top of Appendix E.

**Section completeness checklist** (report-instalment scope: front matter, Ch.1-3, References,
Appendix E) — numeric rules worth remembering: Abstract ≤500 words, future-tense/no-success
language pre-results; Acknowledgments ~100-150 words needing each member's own input;
**2.1.6 Constraints** 6-12 enumerated items (now exactly 6, at the floor); **2.1.7 Specs + Table
1** — max 12 specs total incl. integrated, ≥3 integrated, each dept ≥1 constraint + ≥2 specs
uniquely owned, max 1 off-shelf spec/dept, off-shelf ≤25% overall; **2.3 Ethical section** needs
Concerns/NSPE/Stakeholders+Ranking/Proposed Action, **currently written for the AI-input framing —
re-check against whichever side of §3 wins**; Ch.3 (biggest single rubric item, 20 pts) needs
Internal Search + External Search + Expert Consultation all three, plus a defined scoring
methodology (Weaknesses Analysis/AHP/Pugh — not the old CDR +/-/0 method Al-Badour flagged);
Appendix E is point-by-point, every comment individually addressed, Track Changes used, Layer-2
legend declared at top, written **last**.

**Review workflow**: checklist first, rubric second (cite exact point/weight and which reviewer
will read it), report findings using the color convention, never silently pad a section to hit a
word-count band.

---

## 9. Deadlines & grading

| Item | Due |
|---|---|
| First Report Instalment + final Specifications | 3 Sept 2026 — done |
| First Peer Evaluation | 10 Sept 2026 |
| Project Notebook 1 | 10 Sept 2026 |
| FDR — M001 slot, Room 219 | Week 4 (~6–7 Sept 2026) — **done, this file reflects it** |
| **Second Report Instalment — the live next deliverable** | **22 Sept 2026** |
| Second Peer Evaluation | 29 Sept 2026 |
| PPR Demonstration | ~Week 7–8 |
| Project Notebook 2 | 14 Oct 2026 |
| Update Presentation II | ~Week 10 |
| Third Report Instalment | 5 Nov 2026 |
| Final Prototype Review (FPR) | ~Week 13 |
| Poster Submission | 19 Nov 2026 |
| Project Notebook 3 | 26 Nov 2026 |
| Web Page Info + 1-min YouTube video | 26 Nov 2026 |
| Overall Presentation + Final Report | 10 Dec 2026 |
| Senior Design Expo | 5–10 Dec 2026 |
| Final Peer Evaluation | 17 Dec 2026 |

No late submissions accepted barring an official Students' Affairs excuse.

**Grading**: 50% Individual/Discipline-specific, 50% Group Work. FDR: 30-min slot (2 min setup, 18
min presentation, 10 min Q&A) — **already delivered**. FDR rubric: Elevator Pitch (10),
Deliverables incl. prototype quality (10), Final Design Block Diagram with all disciplines'
contributions (27 pts, the single biggest item), Final Design Details per discipline (23),
constraints/specs/integrated-specs attainment table, Visual Support & Delivery (20). Combined
weight: FDR Presentation ~13% (also seen as ~16.5% on one FDR slide — reconcile against Blackboard
if it matters), Written Report (= Second Report Instalment) 3.5%. Bonuses (capped at +4%): MS
Planner submission up to +2%, Feasibility Plan/accelerator progress up to +4% combined, ITT
screening +1%, Best Presentation +1%, Multidisciplinary (4 disciplines) +1% — doesn't apply, team
has 3.

---

## 10. Standing rules: AI Declaration, purchasing

**AI Declaration Statement**: every report submission needs a table (which sections had AI
assistance, purpose, tool used). Enforced regardless of what's declared — declaring AI use carries
no rubric penalty, vague content scores zero regardless of declaration.

**Purchasing**: budget up to 15,000 SAR total per project, requestable until Week 10, approved by
coach + TEAM Design office. Verbal coach approval on a BOM before any purchase. **>1,500 SAR/item**:
formal Purchase Order, coach-signed, submitted to TEAM Design office (68-335) *before* purchasing.
**<1,500 SAR/item**: keep receipts, get coach's initials. Excludes food/drinks, trips, gas,
transport, compensation, awards, personal mobile devices, hospitality. Receipts need vendor info,
itemized list, tax ID/QR code (KFUPM tax ID: 300000865600003).

---

## 11. Project status & next actions

1. **Resolve §3** (physics model framing) — blocks work in all three disciplines.
2. **Rebuild `build_report.py`'s Table 1 and every cross-referencing paragraph** to this file's §5
   ID scheme — the report currently reflects a spec table one full revision behind what was
   actually presented and defended at FDR.
3. **Pull the FDR's richer technical content into Ch.3** where relevant (each discipline's own
   folder now has this detail).
4. **Second Report Instalment is due 22 Sept 2026.** That deliverable needs #1 and #2 resolved to
   avoid submitting a report that contradicts the team's own FDR.
5. Still-open team sign-off items, unconfirmed either way: each member's own Acknowledgments line;
   Table 2 real course codes for the ME and ICS rows (only COE's has a real-course-code draft, not
   verified against transcripts); laptop-vs-Pi5 confirmation (genuinely open, §6.3); MS Planner
   actually in place; Track Changes turned on; TOC/List of Figures/Tables field refresh.
6. Still-open engineering items directly from the FDR's own "Evidence status" lines: real rig
   frequency sweep and real fault-seeded database (ME); prototype timing/integration tests (COE);
   filter topology/cutoff and the Keyphasor conditioning circuit (COE); actual model fine-tuning on
   real data, not just architecture selection (ICS). A separate mismatch worth resolving alongside
   these: COE's own working docs describe a 152-byte/30kS-s payload while the FDR deck says
   168-byte/50kS-s — reconcile before any downstream document treats either number as final.

---

## 12. Files in this repo

- `COE/` — edge acquisition & processing node: sensor conditioning, ADC/DMA pipeline, payload
  construction, spec pass criteria.
- `ICS/` — AI models (N-HiTS, TFT), dashboard/digital-twin backend, model-selection evidence.
- `ME/` — physical system, FE digital twin, fault database (currently a placeholder — needs the
  same treatment as COE/ICS).
- `Component selection (pugh matrix)/DCDC_Converter_Pugh_Matrix.xlsx` — DC-DC converter concept
  scoring.
- This file (`overview/`) — cross-discipline context: team, deadlines, the final spec table, the
  open physics-framing contradiction, and the evidence/report-writing standing rules.
