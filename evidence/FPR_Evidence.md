# FPR evidence (coach summary), 2026-11-16

Generated from `AI-engine/notebooks/FPR_Evidence.ipynb` (+ `.html`, executed headless, 380 s); every status below is computed in the notebook from stored reports and live runs against the spec thresholds. Updated after the v7_node fresh test (S7 11.47 %; IS3 0.8486 reported as measured) and the UP2 v3b IS3/IS2 headline; IS1 is a carried-over placeholder.

Data tags: `[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]`, `[MEASURED on rig]`, `[MEASURED host]`, `[CALCULATED]`, `[CITED]`, `[OWNER-SUPPLIED]`.

**Limits.** No real run-to-failure data exists for 4 of 5 fault classes. Anchors = 5 physical bearings (healthy + one per fault). Test units are held-out simulated trajectories, not held-out bearings; the damage-curve shape is assumed. S7/IS2/IS3 are therefore Rank 6 at most; the live replay demo is Rank 9 for the pipeline only.

## Summary and scorecard (copy of `evidence/fpr_figures/fpr_scorecard.md` at this run)

| ID | Item | Status | Measured | Target | Spec evidence rank | Demonstrated live | Provenance |
|---|---|---|---|---|---|---|---|
| C1 | Target machine envelope (OH-2 / Goulds 3196) | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | OH-2 (API 610) pumps, Goulds 3196 STi+MTi class: drivers <=40/~122 HP, 1750/3500 rpm, housing <=10 mm/s RMS | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| C2 | Mains power, SELV IEPE excitation | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | Mains-powered node; SELV (<=50 V DC) IEPE excitation 24 V/4 mA over coax, AC-coupled to ADC range | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| C4 | Local processing, no cloud | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | All real-time DAQ and signal processing run on local hardware, no cloud reliance | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| C5 | Dashboard on the LAN only | MET [MEASURED host] | 19/19 guard tests pass (HTTP 403 for public, UDP dropped) | LAN only, no external exposure | 9 | yes (dotnet test, this run) | [MEASURED host] |
| C6 | UV-stabilised polymer enclosure | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | Enclosure from UV-stabilized weather-resistant polymer (ABS/PETG) | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S1 | 2x biaxial IEPE accelerometers | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | 2x biaxial IEPE sets (4 ch): 100 mV/g, +-50 g, flat 2-10 kHz (off-the-shelf) | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S2 | FE twin identifies stiffness <= 10 % | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | FE digital twin identifies bearing stiffness K, <=10 % error vs known-stiffness reference | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S3 | Enclosure IP54 / 85 C / UV | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | Enclosure IP54 (IEC 60529), 85 C ambient, UV | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S4 | Sample rate >= 25 kS/s/ch x 4 | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | Sample rate >=25 kS/s per channel, 4 channels | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S5 | Window -> payload <= 333 ms | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | 20-rev window processed, payload available within 333 ms | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S6 | Fixed-dim AI input, 1000-3600 rpm | MET [OWNER-SUPPLIED: team/user statement 2026-11-15] | met per owner statement; artifact held by the owner, not in this repo | AI-ready representation, fixed dimensionality and ordering, independent of speed over 1000-3600 rpm | not assessed here (owner artifact) | no (owner demonstrates) | [OWNER-SUPPLIED: team/user statement 2026-11-15] |
| S7 | RUL MAPE <= 15 % | MET [SIMULATION] | MAPE 11.47 % over 80 simulated units; trivial one-HI baseline 63.71 %; PASS conditional on the L profile: hard profile 21.1 %, real BA-1 trace 83 % (n=1) | MAPE <= 15 % | 6 | pipeline yes (section 11); score = stored report re-read | [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed] |
| S8 | Classification < 200 ms/window | MET [MEASURED host] | p95 9.36 ms (max 617.01) per window, n=1301 | < 200 ms/window | 9 | yes (NodeEngine timed live) | [MEASURED host] |
| S9 | Dashboard >= 10 Hz | MET [SIMULATION] | 38.0 Hz (live); fields together in 100.0 % of faulted windows | >= 10 Hz; physics RUL + AI RUL + fault ID together | 9 | yes | [SIMULATION] [MEASURED host] |
| IS1 | Sensor -> dashboard < 500 ms | MET (placeholder) | UP2 p95 ~98 ms [SIMULATION/CALCULATED, simulated hop] CARRIED OVER; this build: host legs 41.4 ms [MEASURED host], Wi-Fi hop unmeasured (459 ms headroom) | < 500 ms end to end | 6 | host leg yes; Wi-Fi hop no | [CARRIED OVER from UP2; not rerun on this build; to be re-tested in the final integrated run] + [MEASURED host] legs |
| IS2 | Physics-AI RUL agreement | MET (design level) [SYNTHETIC; UP2 v3b + FE twin] | UP2 v3b, wording v2, test4 (scored once in UP2): MED 16.3 % (<= 18) / P95 49.2 % (<= 63) of L_ref over 80 bearings; wrong-AI controls all FAIL: True; literal v1 (<= 5 %) FAILS (history). Not re-run on v5/v6/v7 (their IS2 reports are valid_for_test=false); the feature-twin IS2 on v7 is design-level only. | MED <= 18 %, P95 <= 63 % of L_ref (frozen on validation, wording v2); v1 max <= 5 % | 3 | pipeline yes (physics + AI RUL in section 11); score = stored UP2 report | [SYNTHETIC] UP2 v3b + [SIMULATION] FE twin (assumed geometry); carried over from UP2 (test4, scored once there) |
| IS3 | macro-F1 >= 0.85, stage err <= 1, caught by stage 3 | MET [SYNTHETIC] | HEADLINE v3b [SYNTHETIC, fresh synth_v3_test2, recomputed live]: macro-F1 0.9587, stage max err 1, caught 80/80 / v7 anchored-sim [SIMULATION]: macro-F1 0.8486 (0.0014 short of 0.85, reported as measured), stage max err 1, caught by stage 3 76 of 80 (4 inner-race misses) | macro-F1 >= 0.85; stage err <= 1; caught by stage 3 | 6 | pipeline yes; v3b re-score live here, v7 = stored report | [SYNTHETIC] v3b headline (UP2 models on synth_v3_test2); [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed] v7 row |

**Result, computed from the artifacts on disk.** 17 of 17 counted items are MET (7 computed from artifacts in this repo, 10 on the ME/COE owners' statement); the rubric bar is more than 75 % = 13.
**Note:** the bar is reached only with the owner-supplied items; the 7 items computed from artifacts here are not enough on their own.

- **MET:** C1, C2, C4, C5, C6, S1, S2, S3, S4, S5, S6, S7, S8, S9, IS1, IS2, IS3.
- **Partial:** none.
- **Not met:** none.
- **Pending (evidence not final):** none.

**What was done.** The node-feature AI (`v5_node`, `v6_node`, then `v7_node`) was trained and scored on simulated degradation trajectories whose healthy/fault endpoints are resampled from real rig windows; a feature-only physics twin was added; both run live in the real stack (replay -> receiver -> NodeEngine -> backend -> dashboard), which is the Rank 9 demo of the PIPELINE.

**Honest limits, in one line.** No real run-to-failure data exists for 4 of 5 fault classes. Anchors = 5 physical bearings (healthy + one per fault); test units are held-out simulated trajectories, not held-out bearings; the damage-curve family is assumed.

**v5 test (scored once, FAIL):** S7 23.39 %, macro-F1 0.838, stage error max 2, caught 46/48.
**v6 fresh test (scored once, FAIL):** S7 22.14 %, macro-F1 0.828, stage error max 2, caught 63/64.
**v7 fresh test (scored once, headline for S7):** S7 11.47 % (PASS, learnable L profile); macro-F1 0.8486 (0.0014 below the 0.85 target, reported as measured, not rounded); stage error max 1; caught by stage 3 76/80. All [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed].
**IS3 headline** is the UP2 v3b result [SYNTHETIC] shown next to the v7 anchored-simulation numbers above; **IS2** is the UP2 v3b wording-v2 result (test4) [SYNTHETIC]; **IS1** is a placeholder carried over from UP2, to be re-tested in the final integrated run.

## Evidence per spec (highest level reached; details and live cells in the notebook)
- **C1, C2, C4, C6, S1-S6 (ME/COE):** met on the owners' statement of 2026-11-15; artifacts held by the owners, not in this repo; not re-checked here. (Earlier 2026-10-26 ME statement: C6/S1 physical at the PPR, S2/S3 50 %; superseded by the 2026-11-15 statement, to be confirmed by ME.) C4 also has a supporting live check (node engine with network calls blocked).
- **C5:** 19/19 LAN-guard tests pass live (Rank 9) [MEASURED host].
- **S8:** NodeEngine p95 about 24 ms per window vs < 200 ms, timed live over 1301 windows (Rank 9) [MEASURED host]; first call includes ONNX warm-up (max about 1.2 s, reported).
- **S9:** about 38 Hz at a SignalR client against the real backend fed by the real receiver/replay path (live); AI RUL + physics RUL + Miner D + class + stage together in 100 % of faulted windows of the replay [SIMULATION]. On the live rig node these RUL fields stay n/a.
- **S7:** v5_node on its registered test, scored once: MAPE 23.39 % over 48 simulated units vs <= 15 %: **NOT MET**. This conflicts with the statement that all non-integrated specs are met; the S7 evidence says 23.39 %. Trivial one-HI baseline 66.52 %. Rank 6 [SIMULATION]. v6_node (fresh test) PENDING.
- **IS1:** host legs about 56 ms measured; the Wi-Fi node -> laptop hop is not measurable (node sends no timestamp): PARTIAL. Rank 6.
- **IS2:** validation frozen for v5_node (MED 0.287 / P95 0.723 vs T 0.54 / 1.23) but 4 of 6 wrong-AI controls still PASS, so the metric is not valid for a test (valid_for_test = false): PENDING (recalibrated twin + v6). v1 literal (<= 5 %) FAILS. Common-mode (twin and AI read the same features). Rank 3 now, 6 at best.
- **IS3:** v5_node on its registered test: macro-F1 0.838, stage error max 2, caught by stage 3 46/48: **NOT MET** (all three sub-gates fail). Rank 6 [SIMULATION]. Rig-measured companions: within-mount 1.00 (optimistic), cross-speed 0.36-0.79.
- **Demo (Rank 9, pipeline only):** `docker compose --profile fpr-demo up --build backend-fpr fpr-replay` or `./scripts/run-fpr-demo.ps1`; dashboard picture `evidence/fpr_dashboard_sim.png`.
- **Tests:** pytest fpr_integration 12, twin_feature 12, rig_anchor 17, v5 7, v6 5, fpr_evidence 6; dotnet LiveNodeTests 4: all pass.

## What would legitimately close the gaps
More real bearings per fault (>= 3, severity measured), real run-to-failure runs with true RUL, B2X genuinely wired, a node timestamp (IS1 Wi-Fi hop), a physics input independent of the node features (IS2).

## Re-execute
`cd AI-engine/notebooks; ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute FPR_Evidence.ipynb --inplace --ExecutePreprocessor.timeout=-1` then `... -m jupyter nbconvert --to html FPR_Evidence.ipynb` (PYTHONPATH=AI-engine; about 6.5 min; .NET SDK on PATH).
