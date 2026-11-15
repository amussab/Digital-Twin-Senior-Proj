# ICS + Integrated spec scorecard on the rig AI (`v4_real`) — 2026-11-15

Every number below is tagged by how it was obtained. Nothing here is invented; where the data cannot support a spec, it says so.
Data: 10 rig runs / 5,076 windows (healthy, inner, outer, cage at 1800 + 3600 rpm; ball at 1800 rpm only), ONE physical bearing per fault,
fault severities ESTIMATED (not measured), B2X on the node is a copy of B1Y (policy B2X:=B2Y applied). Commits eca261f..7bb6a78 + this file.

| Spec | Requirement | Result | Tag | Verdict |
|---|---|---|---|---|
| C5 (ICS constraint) | Dashboard LAN-only, no external exposure | Local-network guard unchanged; backend tests 37/37 pass (incl. guard) | [MEASURED host tests] | MET (as before; live run on :5080, LAN address shown) |
| S8 | Classification < 200 ms/window | NodeEngine p95 **13.6 ms** (train_real stream check, 158 windows); live path inference p50 3.5 / p95 5.0 ms (362 windows) | [MEASURED host] | **MET** |
| S9 | UI refresh >= 10 Hz; physics RUL + AI RUL + fault ID updated together | Client snapshot rate **21.1 Hz** (min 20 / max 22 per s) at real node cadence, 20 s, SignalR client. BUT physics RUL = null and AI RUL = n/a on node data (0/423 snapshots carry physics RUL) | [MEASURED host, replay of rig data] | **PARTIAL**: rate met, "physics RUL + AI RUL" not available from node features |
| S7 | RUL MAPE <= 15 % on held-out degradation window | No RUL model: the node sends window features only and the rig has no run-to-failure data with ground-truth RUL (the one failure, ball BA-1, has invalid rpm labels, see run_to_failure/) | — | **NOT MET / NOT MEASURABLE** |
| IS1 | sensor -> classify + RUL -> dashboard < 500 ms | Host leg: receive -> inference p95 5 ms, receive -> backend ack p50 16 / p95 32 ms, node-ACK RTT p95 28 ms (replay, 362 windows); earlier headless-Edge render p95 353 ms (v3e). Wi-Fi node->laptop hop NOT measurable (node sends no timestamp). RUL not produced. | [MEASURED host] | **PARTIAL / CONDITIONAL** (classification path well under 500 ms; RUL and Wi-Fi leg not covered) |
| IS2 | physics-AI RUL agreement (stages 1->3) | Physics twin and AI RUL are both n/a on node data | — | **NOT MET / NOT MEASURABLE** |
| IS3 | macro-F1 >= 85 %, stage error <= 1, fault caught by stage 3 | see below | — | **NOT SCORABLE (insufficient bearings); evidence below is NOT an IS3 pass** |

## IS3 evidence that exists (all [MEASURED on rig], all optimistic or pessimistic proxies)
- Held-out BY BEARING: impossible, 1 bearing per fault (train_real: "need >= 3 runs/mounts per condition"; IS3 NOT SCORED).
- Leave-one-bearing-out CV: macro-F1 0.45 (best variant). Structurally pessimistic (each fold removes a class).
- Within-mount time-blocked (same bearing/mount in train and test): macro-F1 1.000 real-only / 0.999 real+sim. OPTIMISTIC upper bound; not IS3.
- Held-out OPERATING SPEED (train one speed, test the other, same bearings): 1800->3600 macro-F1 0.68 (real-only) / 0.79 (+sim, cage F1 0.30);
  3600->1800 macro-F1 0.48 (real-only) / 0.36 (+sim; inner 0.0, cage 0.04). All below 0.85: the classifier does NOT yet generalise across speed with this data.
- Stage error / caught-by-stage-3: severities are estimated, not measured, so no valid stage ground truth exists: not scored.

## What would legitimately close the gaps (needs data, not more tuning)
1. IS3: >= 3 physical bearings per fault (2 train + 1 held out, frozen in splits.json before fitting), measured severity levels.
2. S7 / IS2 / S9-RUL: run-to-failure runs with true RUL labels recorded as features-over-time + a physics-twin input that the node does not currently provide.
3. IS1: a node timestamp (firmware) to measure the Wi-Fi hop.
4. Cross-speed: more speeds or more bearings; the 4-sensor re-run helps only once B2X is genuinely wired.

## Simulation-based evidence (added 2026-11-15) — all [SIMULATION], coach-approved synthetic basis, NOT rig measurements
- `v4_real` on the HELD-OUT SIMULATED test units (synth_coe_node split=test, 208 units, 9,360 windows; scored once): **macro-F1 0.802**
  (ball .80 cage .72 healthy .84 inner .83 outer .82; 1800 rpm 0.819 / 3600 rpm 0.789). Below the 0.85 target.
- Sim-weight variants (1.0, 3.0) chosen on the simulation VALIDATION split only: val macro-F1 0.763 (w=0.3, shipped), 0.775 (w=1), 0.768 (w=3). None reaches 0.85;
  they were discarded (not shipped). Reason: the real faults are far stronger (RMS 5-19x healthy) than the simulated ones, so the real data pulls the model away from the simulated distribution.
- Existing simulation-only model `v3e_node` (trained only on synth_coe_node, 2026-11-12): **[SIMULATION] test macro-F1 0.877 (>= 0.85)** on the same held-out simulated units. That is the model with a simulation IS3 pass; it has no rig-measured evidence of equal strength.
- A synthetic "held-out bearing" built by jittering the real windows was deliberately NOT made: it would derive from the same real bearings, so it would only look like generalisation.
- S7 / IS2 [SIMULATION] evidence (RUL and physics twin) exists from the UP2 prototype (v3 contracts, `Repo/DT-up2`, `UP2_Evidence.ipynb`); `v4_real` itself outputs no RUL.


## FPR section (2026-11-15, evidence author) - computed by `AI-engine/notebooks/FPR_Evidence.ipynb`; the rows above are kept as history

Counts of the 17 counted items at this run: **13 MET** (3 computed from artifacts here: C5, S8, S9; 10 owner-supplied ME/COE statement 2026-11-15), 1 PARTIAL (IS1), 2 NOT MET (S7, IS3), 1 PENDING (IS2); bar = 13.
Data tag for S7/IS2/IS3: `[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]`; no real run-to-failure data exists for 4 of 5 fault classes; test units are held-out simulated trajectories, not held-out bearings.

| Spec | FPR result (v5_node, scored once; v6_node PENDING) | Rank | Verdict |
|---|---|---|---|
| C5 | 19/19 LAN-guard tests live [MEASURED host] | 9 | MET |
| S8 | NodeEngine p95 ~24 ms/window live [MEASURED host] | 9 | MET |
| S9 | ~38 Hz live; AI RUL + physics RUL + Miner D together in 100 % of faulted replay windows [SIMULATION]; n/a on the live rig node | 9 (pipeline) | MET [SIMULATION] |
| S7 | MAPE 23.39 % (> 15 %), baseline 66.52 % [SIMULATION] | 6 | NOT MET (conflicts with the all-met statement) |
| IS1 | host legs ~56 ms; Wi-Fi hop unmeasurable (no node timestamp) | 6 | PARTIAL |
| IS2 | val frozen, 4/6 controls pass -> not valid for test; v1 literal FAIL | 3 | PENDING |
| IS3 | macro-F1 0.838, stage max 2, caught 46/48 [SIMULATION] | 6 | NOT MET |
| C1, C2, C4, C6, S1-S6 | owner statement 2026-11-15, artifacts with owners | n/a | MET [OWNER-SUPPLIED] |
