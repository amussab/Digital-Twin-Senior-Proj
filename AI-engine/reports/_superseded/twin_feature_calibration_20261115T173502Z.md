# FeaturePhysicsTwin bounded-link calibration [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed; calibration = fit to the generator]

UTC 20261115T173502Z; command `python -m aiengine.twin.calibrate_feature_physics`.

Law: g(r) = 1 + k ln(max(r,1)) (equivalent-load factor), dD/dt = g^3 / L_ref(rpm) (ISO 281 p = 3 on the LOAD factor), failure D >= 0.95;
trend RUL extrapolates ln r to ln r_fail, Miner RUL uses the same law in closed form; rul = min.

Criterion (declared before running): min over grid of median over faulted val bearings of median_w |RUL_phys-RUL_true|/L_ref (true stage 1-3, window>=24, RUL>0)
Val units: 48 (sha256 of the sorted unit list ade2ae724bd0e86cd23257b6e35dcf902b9ea991462a80a55be6ea3baf1225cb).

## Calibrated: k = 0.05, r_fail = 10.0   (criterion = 0.1789 of L_ref)

| stage | truth RUL/L_ref | twin RUL/L_ref | median D | trend share |
|---|---|---|---|---|
| 1 | 0.803 | 0.703 | 0.218 | 0.11 |
| 2 | 0.657 | 0.523 | 0.399 | 0.44 |
| 3 | 0.511 | 0.351 | 0.552 | 0.99 |

Legacy law (g = r, p = 3 on r, r_fail 20): criterion 0.5145; per stage twin RUL/L_ref 1: 0.301, 2: 0.084, 3: 0.000

## Limits
Fit of two scalars to the simulation generator on VAL units; the law is a modelling assumption; IS2 on this twin stays common-mode (same features as the AI) and Rank <= 6. Test / rig_anchor_test2 units not used.

## Grid (criterion)

| k | r_fail 10 | r_fail 15 | r_fail 20 | r_fail 30 | r_fail 50 | r_fail 100 |
|---|---|---|---|---|---|---|
| 0.05 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 |
| 0.1 | 0.198 | 0.198 | 0.198 | 0.198 | 0.198 | 0.198 |
| 0.2 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| 0.3 | 0.266 | 0.266 | 0.266 | 0.266 | 0.266 | 0.266 |
| 0.5 | 0.367 | 0.367 | 0.367 | 0.367 | 0.367 | 0.367 |
| 0.75 | 0.445 | 0.445 | 0.445 | 0.445 | 0.445 | 0.445 |
| 1.0 | 0.488 | 0.488 | 0.488 | 0.488 | 0.488 | 0.488 |
| 1.5 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 |
| 2.0 | 0.606 | 0.606 | 0.606 | 0.606 | 0.606 | 0.606 |
