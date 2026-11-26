# FPR IS2 (wording v2) on rig_anchor simulated units: contract v5_node, split val

Provenance: [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]

Limits: COMMON MODE: twin and AI read the same node features and the same baseline, so agreement is consistency of two readings of one signal, not independent confirmation. No real run-to-failure data exists for 4 of 5 fault classes. Anchors = 5 physical bearings (H-01 + one per fault); test units are held-out simulated trajectories, not held-out bearings; the damage-curve family is assumed. L_ref uses an ASSUMED simulation-equivalent duty (P/C = 0.5). The thresholds are tied to this simulated population.

## Frozen thresholds (from VALIDATION only)
T_MED = 0.54, T_P95 = 1.23 (x L_ref). Rule: ceil to 0.01 of 1.1 x max over validation sets of max(v3b value, perfect-AI floor). Preds: fpr_is2_v5_node_val_20261115T165844Z.csv.gz.

## Result
| quantity | value | threshold | verdict |
|---|---|---|---|
| MED (median over bearings of per-bearing median e) | 0.2870 | 0.54 | PASS |
| P95 (95th pct over bearings of per-bearing p95 e) | 0.7231 | 1.23 | PASS |
| **IS2 v2** | n_bearings 48, n_windows 9548 | both | **PASS** |

## Context (what the number means)
- Exact-AI floor (AI := true RUL, same windows): MED 0.4825, P95 1.1091. A perfect AI still disagrees with this twin by that much; thresholds are never tighter than this floor.
- Twin issues a TREND RUL in 38.0 % of scored windows; the rest is the rated-life prior (sources {'prior': 5915, 'trend': 3633}). Where the twin is the prior, agreement mostly measures the AI against the rated-life assumption.
- Residual-flag AUROC for 'AI wrong by > 0.20 L_ref': 0.31426020008838856.
- Twin vs truth (median |.|/L_ref) 0.468; AI vs truth 0.207.
- Scale (simulated life vs ISO 281 L_ref): {'units': 48, 'life_over_Lref_median': 0.98325, 'onset_to_failure_over_Lref_median': 0.6659999999999999, 'onset_to_failure_over_Lref_min_max': [0.31725, 1.1655000000000002], 'L_ref_h': {'1800': np.float64(74.07407407407408), '3600': np.float64(37.03703703703704)}, 'duty': 'P/C = 0.50 [ASSUMED]'}

## Literal v1 (max over windows of |RUL_phys - RUL_AI| / RUL_AI <= 5 %)
max 1.492, p95 1.000, median 0.651, share of windows <= 5 %: 0.035 -> **FAIL**

## Negative controls (wrong AI RULs on the same windows; must all FAIL)
| control | MED | P95 | verdict |
|---|---|---|---|
| AI x 1.5 (optimistic bias) | 0.4825 | 1.0847 | PASS |
| AI x 0.6 (pessimistic bias) | 0.2130 | 0.5761 | PASS |
| AI + 0.3 L_ref | 0.5646 | 1.0231 | FAIL |
| AI time-reversed within bearing | 0.3185 | 0.6012 | PASS |
| AI = 0.5 L_ref constant | 0.3194 | 0.5000 | PASS |
| AI = truth x lognormal(0, 0.5) noise | 0.4719 | 2.2142 | FAIL |

All controls FAIL: **False**

## Honest reading
- Twin and AI read the same features (common mode): a pass says the two estimators are consistent on this simulated population, not that either is right about a real bearing.
- Data are simulated, rig-anchored; the damage curve is assumed; the thresholds are tied to this population and to the ASSUMED duty used for L_ref.
