# FPR_PLAN — close S7, IS2, IS3 on the node-feature AI (written 2026-11-15, FPR = 2026-11-16)

Branch `fix/rig-new-bearing` (worktree Repo/DT-main). Python = `Repo/DT-up2/AI-engine/.venv/Scripts/python.exe`, run from `AI-engine/`
with PYTHONPATH=$PWD. Rules in force: `.claude/rules/*`, evidence ladder, provenance tags, splits registered before fitting,
never fabricate (standing rule), `[SIMULATION anchored on MEASURED rig windows]` is the tag for everything built from rig anchors.
User decisions (2026-11-15 interview): anchored-sim RUL set approved; NO more rig time; extend twin + re-freeze IS2 thresholds on
new validation; no merge of origin/main; real damage sizes unknown (severities stay ESTIMATED).

## 0. What the real data actually gives (measured, run-medians, B1Y)
healthy 1800: rms .026 env .022; faults at 1800: inner .27/.23, outer .49/.42, cage .17/.15, ball .16/.13 (5-19x healthy).
So the rig faults are already SEVERE (late-stage). There is no real mild/incipient data and no real degradation path except the
ball BA-1 failure trace (rms .32 -> .52 -> 1.69 g, 143 windows, rpm column invalid, last 229 windows = free spin).
Per fault: ONE physical bearing; ball only at 1800 (3600 ball run excluded: jammed). Healthy: one bearing H-01 + B2 always healthy.

## 1. Rig-anchored degradation set (`aiengine/rig_anchor/`, new; owner: data engineer)
Unit = one run-to-failure of the faulted bearing + healthy partner (same 32-feature node window layout, B2X:=B2Y policy).
Window features are built IN THE REAL FEATURE SPACE, not by the old synth_v3 formulas:
 - log-ratio space vs the unit's commissioning baseline (what the contract features already use).
 - x_t = exp((1-g_t) * log h_t + g_t * log f_t) element-wise per channel feature, where h_t is a REAL healthy window and f_t a REAL
   fault window of that fault class (same rpm key), drawn as a time-correlated (AR(1)-ordered) resample of the anchor pools so
   real window noise/kurtosis/impulsiveness structure is preserved. g_t in [0, 1+e] is the damage curve; g=1 is the MEASURED fault
   level, g>1 extends to failure using the late-stage amplification measured in the real ball failure trace (rms ratio failure/segment0).
 - anchor placement from the RECORDED run metadata (severity 0-3, ESTIMATED by the user on rig day, already in each run's .json):
   the measured fault windows are placed at the stage the severity maps to (train_real rule sev 3 -> stage 6 region, sev 2 -> stage 4
   region; exact u_anchor per severity is a documented assumption, frozen before fitting) instead of at "end of life". So g=1 sits at
   u_anchor(sev), and g<1 (stages 2-4 for sev 3) is interpolation toward healthy, g>1 extrapolation toward failure.
 - damage curve g(u): onset at random healthy run-in, accelerating curve (parameters a, p per unit) as synth_v3; per-unit life from an
   ISO-281-style speed law x lognormal rate; failure = u=1; stage = DESIGN.md rule on true u (stage 1 pre-onset, 2..6 equal parts).
 - per-unit nuisance: sensor gain per channel, mount offset (log-ratio shift sigma larger than the observed run-to-run spread), speed 1800/3600.
 - Partner bearing: healthy real windows + transmitted share (as synth_v3 `trans`).
Provenance on every frame/report: `[SIMULATION anchored on MEASURED rig windows]`; the file states anchors = 5 bearings, 1 per fault.
Known circularity (stated in the report, not hidden): the damage curve family is our assumption; test units use curve parameters OUTSIDE
the training range (shape OOD) and a wrong-shape control must FAIL; held-out units share the 1 real anchor bearing per fault.
Splits: unit-level train/val/test registered in splits.json (section `rig_anchor`) BEFORE any fit; test scored once.

## 2. Models (owner: ml engineer) -> contract `v5_node`
 - RUL head: gradient-boosted regressor (log RUL) on the v3b-style window aggregates of baseline-relative node features (HI + families),
   post-process (stage median, log-EMA) unchanged; trained on rig_anchor train units; selected on val; S7 = MAPE <= 15 % on held-out
   degradation window (definition as in Specifications.md S7: the degradation part, true stages>=2) on test units.
 - Classifier + stage head: 5-class + stage (1..6) from the same features; trained on rig_anchor train units + real windows (v4_real data,
   weight chosen on val). IS3 = macro-F1 >= 0.85 on test units; stage error <= 1; fault caught by stage 3 at latest (per-unit).
 - Keep v4_real / v3e_node results in the evidence as history. v5_node must keep S8 (<200 ms; measure) and the ONNX export + golden vectors.
 - Extra honest checks (reported, not gated): (a) cross-speed anchored (train 1800 anchors -> test units built from 3600 recordings),
   (b) leave-one-fault-recording-out, (c) the REAL ball failure trace n=1: score the RUL/stage head on segments 0-2 with true RUL =
   windows-to-disintegration, tagged `[MEASURED on rig, n=1, rpm column invalid]`.

## 3. Physics digital twin on node features (owner: twin engineer) -> `aiengine/twin/feature_physics.py`
The FE-beam twin needs prox amp/phase (node has none) => IS2 impossible on node data. Add FeaturePhysicsTwin:
 - state: defect-energy ratio r_t = env-band energy / commissioning baseline (per bearing, from the 4 order bands + envelope rms).
 - law (physics, cited): ISO 281 life exponent p=3: damage rate dD/dt = r^p / L_ref(rpm); Miner D>=0.95 = failure (team spec M6);
   plus Paris-type exponential growth of ln r for RUL extrapolation (same least-squares + significance gate structure as `fit_rul`).
 - L_ref(rpm) = ISO 281 basic rating life of S6700/61700 (C from catalogue [CITED], to be filled; if not found: marked ASSUMED).
 - Output: physics_rul_hours, D, health index; feeds NodeEngine -> dashboard (so S9's "physics RUL + AI RUL + fault ID" gets values).
 - IS2 evaluation: wording v2 (e = |RUL_phys - RUL_AI|/L_ref, per-bearing median/p95 aggregation), thresholds RE-FROZEN on rig_anchor val
   units (rule as in `up2d_is2.py`, controls must fail) BEFORE scoring test once. v1 literal (5 %) result reported next to it. Old IS2 numbers kept as history.

## 4. Integration (owner: backend/integration engineer)
NodeEngine loads `v5_node` (RUL + stage + physics), live runner/receiver/backend `Live` block carry rul/physics/stage; dashboard n/a -> values.
Tests: pytest (AI-engine/tests) + dotnet test (tests/DigitalTwin.Backend.Tests -c Release). Replay scenario: a rig_anchor unit rendered as a
jsonl in node format (healthy -> failure) replayed through the receiver; plus the real healthy/fault recordings replayed for the rig-measured part.

## 5. Evidence + scorecard (owner: evidence author) -> Rank 9
 - `evidence/FPR_Evidence.md` + notebook `AI-engine/notebooks/FPR_Evidence.ipynb` (+html): every claim recomputed live from files; per-item
   tag, rank achieved, rank achievable, limits. Update `evidence/RIG_SPEC_SCORECARD.md` (new rows, old rows kept), RIG_STATE, FACTS, memory.
 - Rank 9 here = the working demo run live (replay -> NodeEngine -> dashboard showing RUL/stage/physics). The DATA behind S7/IS2/IS3 stays
   labelled simulation-anchored; the report never says "measured on rig" for them. IS1 tested last in one integrated run (user will do).
 - evidence-reviewer subagent (read-only) grades before hand-off.

## Order / time-box (all today)
1 generator + tests + registered split (h1) -> 2 train/eval v5 (h2-3) || 3 twin + IS2 (parallel, no shared files) -> 4 integration -> 5 evidence -> review -> commit.
Stop rule: if a gate fails on test, report FAIL with the number; do not retune on test; do not redefine the metric.
Commits small on `fix/rig-new-bearing`; no push without asking.

---
## REVISION 1 (after subagent critique, 2026-11-15) — supersedes the sections above where it conflicts
Critique verdict: S7/IS3 on an anchored set are close to circular (labels are a monotone function of the assumed curve), anchors are shared
(1 bearing/fault), stage labels before/after the anchor are assumed, IS2 twin and AI read the same signal, L_ref vs simulated life scale can
make IS2 pass vacuously, scope too big for one day. Accepted changes:
1. WORDING. Tag = `[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]` (short: `[SIM-ANCHORED]`). Test units are
   "held-out simulated trajectories", never "held-out bearings". Say in every report: "no real run-to-failure data exists for 4 of 5 fault classes".
2. EVIDENCE RANKS (honest maximum): S7 Rank 6, IS3 Rank 6 (classifier part; stage/early-catch by construction on the assumed curve), IS2 Rank 3-6
   (common-mode, partly by construction). A live demo is Rank 9 for the PIPELINE only. Scorecard has two columns: "demonstrated live" and
   "spec evidence rank". "Level 9 for all specs" is NOT reachable with this data; say so to the user, do not paper over it.
3. LEAKAGE CONTROL. Anchor pools are split by TIME BLOCKS inside each real run (train pool first 60 %, val 20 %, test last 20 %); test trajectories use
   ONLY test-pool windows. Real windows are NOT added to classifier training as separate rows (they enter only via train-pool anchors). Real-window
   checks are reported separately as [MEASURED on rig] (v4_real cross-speed numbers stay as history).
4. CONTROLS FOR S7: a trivial baseline (linear/log-linear regression on one HI) must be reported next to the GBM; a wrong-shape control; if the baseline
   matches the GBM, say the number measures the assumed curve.
5. PLAUSIBILITY: figure + table comparing generator trajectories' feature ranges (rms/env/kurtosis/bands) with the real BA-1 failure trace and the
   real fault runs; state where they differ. Real ball-failure trace scored as [MEASURED on rig, n=1], reported not gated.
6. IS2: first compute the scale check (simulated life in hours vs L_ref from ISO 281 at the duty speed); simulated unit life is scaled so that
   life/L_ref is a stated, justified ratio (not vacuous); negative controls (AI x1.5, x0.6, +0.3 L_ref, constant 0.5 L_ref) must FAIL on val before freeze;
   report the share of windows where the twin issues a trend RUL vs the prior. v1 literal (5 %) result reported beside it; old UP2 results kept.
7. SCOPE CUT. No new notebook: extend `UP2_Evidence` pattern via `evidence/FPR_Evidence.md` + a `fpr_eval` CLI that recomputes every number from stored
   artifacts (re-score = Rank 6 live re-check; demo = Rank 9). Cross-speed and leave-one-recording-out = reported, not gated. Headline history kept:
   synth_v3 S7 12.64 % / IS3 0.959 [SYNTHETIC] stay in the evidence as before.
8. ORDER: (a) scale check -> (b) generator + split + plausibility -> (c) v5 RUL head + baseline -> (d) v5 classifier/stage -> (e) feature twin + IS2
   (twin module can be written in parallel, IS2 eval after (b)) -> (f) integration -> (g) evidence + reviewer.
Generator output schema (contract between agents): pandas table, one row per window per bearing, columns of synth_v3 `SCHEMA` + extras
(`unit_id`,`window_index`,`t_hours`,`rpm`,`fault_class`,`run_to_failure`,`life_hours`,`rul_hours`,`true_stage`,`u_true`,`damage`,`faulted_bearing`,
`split`,`provenance`) PLUS the 32 raw node features per WINDOW in `aiengine.real_data.FEATURE_COLUMNS` order (B2X:=B2Y applied), so NodeEngine can replay it.
