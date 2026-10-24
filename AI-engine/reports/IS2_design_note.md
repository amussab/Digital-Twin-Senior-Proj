# IS2 design note: what "physics-AI residual" can mean (open team contradiction)

Status: for the team to decide. This note does not choose. Source of the contradiction:
`Project architecture/overview/Project Overview.md` section 3. Numbers: `reports/up2_residual_<ts>.md`
([SIMULATION] physics twin on a [SYNTHETIC] damage-derived stiffness trajectory).

## The two readings

**A. Physics as an independent estimator** (FDR deck; Specifications.md IS2: "max relative error between the
FE-identified physics RUL and the AI RUL <=5%, health stage 1->3"). The FE twin yields its own RUL from the
stiffness-drop criterion (K drop >= 25 %); the backend reports both RULs and their residual.

**B. Physics as an AI input feature** (report version of 2026-09-03). Physics-derived quantities enter the
N-HiTS/TFT input; there is no second RUL number to subtract. IS2 would then be an ablation (physics features vs
DSP-only baseline, >=15 % relative MAPE gain) and the safety argument rests on the classifier's confidence.

## What each means for IS2

| | Reading A (independent) | Reading B (input feature) |
|---|---|---|
| What is measured | |RUL_phys - RUL_AI| / RUL_AI per window | MAPE(with physics inputs) vs MAPE(DSP-only) |
| Current evidence | Simulation: FAIL by a wide margin (see residual report) | Not run. Needs displacement/K features in the synthetic windows and a retrain; the generator currently has no physics-derived column |
| Structural limit | The residual is bounded below by the AI's own error vs truth (|truth-AI|/AI). With S7 at ~13-15 % MAPE a <=5 % residual needs both estimators far more accurate than S7 requires, or a physics RUL that tracks the AI's errors (which would make it non-independent) | No such bound; the target is a relative gain, but the claim "physics helps" is circular on synthetic data because the stiffness trajectory and the vibration features are generated from one damage variable |
| Observability | K moves <~5 % during stages 1-3 (damage 0..0.4 of life) against ~3-6 % per-window identification noise at 1750 rpm and unusable identification at 3600 rpm (twin README), so an independent physics RUL is weakest exactly in the stages IS2 names | Same data, but a feature need not be a forecast |
| Payload / COE | 4 displacement floats already in the 152-byte payload; RUL computed on the host | Same payload; features computed on the host |
| Dashboard | Shows physics RUL, AI RUL and residual (Project Overview section 2 item 5) | No residual panel; confidence signal instead |
| Ethics section | Residual as a cross-check | Confidence signal |

## Consequences to decide, not decided here

- Under A, either the 5 % threshold or the metric (e.g. residual vs truth on a calibrated rig, median instead of
  max, stages 2-3 only, or a relative-to-S7 bound) has to be revisited by the team; the simulation shows the
  literal wording is not attainable with an independent physics estimator and a 13-15 % AI.
- Under B, IS2's text, Table 1 row and the physics twin's deliverable change, and the work item is the ablation.
- Mixing them (physics RUL used as an AI input AND compared to the AI RUL) makes the residual partly a measure of
  the AI copying its input; if the team wants that, the report must say so.
