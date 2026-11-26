# rig_anchor scale check (ISO 281 rating life vs simulated unit life)

Provenance: [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]. Generated 2026-11-15T16:07:53+00:00 by `python -m aiengine.rig_anchor.scale`.

## Inputs
- Bearing S6700 / 61700, open, 10x15x3 mm, Z=11 balls, Dw=1.2 mm, Dpw=12.45 mm, contact angle 0 (COE, measured 2026-11-12; orders {'FTF': 0.4518, 'BSF': 5.1393, 'BPFO': 4.9699, 'BPFI': 6.0301}).
- Basic dynamic load rating C: **489 N [CITED: supplier listing W 61700, 110 lbf]**; other listings of the 61700 family give 853-859 N (FAG F61700-2RSR 857 N; a 4 mm-wide sealed version 859 N). The exact brand/suffix of the rig bearing is unknown -> **ASSUMED within 489-857 N**. My own ISO 281 formula estimate C = fc (i cos a)^0.7 Z^(2/3) Dw^1.8 with an older-table fc=55.1 (recalled, ASSUMED) gives 378 N [CALCULATED]; modern catalogue ratings are higher.
- L10 depends on C only through P/C: L10h = 1e6/(60 n) (C/P)^3 (ISO 281 basic rating life, a1=1, 90 % reliability). The rig's real equivalent radial load is NOT known (not measured) -> P/C is a stated assumption.

## L10 table (hours)
| P/C | P (N, C=489) | rpm | L10 (h) | simulated median unit life (h) | sim life / L10 |
|---|---|---|---|---|---|
| 0.05 | 24.5 | 1800 | 74074.1 | 51.9 | 0.0007 |
| 0.05 | 24.5 | 3600 | 37037.0 | 25.9 | 0.0007 |
| 0.1 | 48.9 | 1800 | 9259.3 | 51.9 | 0.0056 |
| 0.1 | 48.9 | 3600 | 4629.6 | 25.9 | 0.0056 |
| 0.2 | 97.8 | 1800 | 1157.4 | 51.9 | 0.0448 |
| 0.2 | 97.8 | 3600 | 578.7 | 25.9 | 0.0448 |
| 0.3 | 146.7 | 1800 | 342.9 | 51.9 | 0.1512 |
| 0.3 | 146.7 | 3600 | 171.5 | 25.9 | 0.1512 |
| 0.5 | 244.5 | 1800 | 74.1 | 51.9 | 0.7 |
| 0.5 | 244.5 | 3600 | 37.0 | 25.9 | 0.7 |

## Choice
- Simulation-equivalent duty **P/C = 0.5** (P = 244 N): L_ref = L10 = 74.1 h at 1800 rpm and 37.0 h at 3600 rpm.
- Simulated onset-to-failure life = LIFE_RHO x L_ref x lognormal(0, 0.35) with LIFE_RHO = 0.7: median 51.9 h (1800) and 25.9 h (3600) = 311 / 156 windows at the 10-min snapshot cadence. **Ratio life/L_ref = 0.7.**
- Why not the light real duty: at P/C = 0.1 L10 is 9259 h (1800 rpm); a simulated unit of tens of hours then has life/L_ref ~ 0.006, so an IS2 error normalised by L_ref (e = |RUL_phys - RUL_AI| / L_ref) would be < 1 % of L_ref for ANY prediction between 0 and the unit life = vacuous pass. With life/L_ref = 0.7, an 18 % L_ref band is about 26 % of the unit's median life, so a constant or x1.5/x0.6 wrong RUL can fail it (to be verified by the IS2 controls on val).
- Honesty: P/C = 0.5 is the heavy end of the ISO 281 range, a stated simulation-equivalent (accelerated) duty, not a rig measurement. LIFE_RHO = 0.7 is an assumption: a seeded-defect bearing's onset-to-failure time is taken as at most its rated life. The absolute hours therefore carry no physical claim; only the life/L_ref ratio and the speed law (life in revolutions fixed, hours ~ 1/rpm) are used.
