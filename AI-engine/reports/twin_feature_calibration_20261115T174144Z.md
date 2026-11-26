# FeaturePhysicsTwin bounded-link calibration [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed; calibration = fit to the generator]

UTC 20261115T174144Z; command `python -m aiengine.twin.calibrate_feature_physics`.

Law: g(r) = 1 + k ln(max(r,1)) (equivalent-load factor), dD/dt = g^3 / L_ref(rpm) (ISO 281 p = 3 on the LOAD factor), failure D >= 0.95;
trend RUL extrapolates ln r to ln r_fail, Miner RUL uses the same law in closed form; rul = min.

Criterion (declared before running): min over grid of median over faulted val bearings of median_w |RUL_phys-RUL_true|/L_ref (true stage 1-3, window>=24, RUL>0)
Val units: 48 (sha256 of the sorted unit list ade2ae724bd0e86cd23257b6e35dcf902b9ea991462a80a55be6ea3baf1225cb).

## Calibrated: k = 0.01, r_fail = 10.0   (criterion = 0.1770 of L_ref)

| stage | truth RUL/L_ref | twin RUL/L_ref | median D | trend share |
|---|---|---|---|---|
| 1 | 0.803 | 0.728 | 0.213 | 0.11 |
| 2 | 0.657 | 0.555 | 0.390 | 0.44 |
| 3 | 0.511 | 0.406 | 0.532 | 0.99 |

k = 0 reference (feature-blind rated-life clock, excluded from the choice): criterion 0.1688.

Legacy law (g = r, p = 3 on r, r_fail 20): criterion 0.5145; per stage twin RUL/L_ref 1: 0.301, 2: 0.084, 3: 0.000

## Limits
Fit of two scalars to the simulation generator on VAL units; the law is a modelling assumption; IS2 on this twin stays common-mode (same features as the AI) and Rank <= 6. Test / rig_anchor_test2 units not used.

## Grid (criterion)

| k | r_fail 10 | r_fail 20 | r_fail 50 |
|---|---|---|---|
| 0.0 | 0.169 | 0.169 | 0.169 |
| 0.01 | 0.177 | 0.177 | 0.177 |
| 0.02 | 0.186 | 0.186 | 0.186 |
| 0.05 | 0.179 | 0.179 | 0.179 |
| 0.1 | 0.198 | 0.198 | 0.198 |
| 0.2 | 0.200 | 0.200 | 0.200 |
