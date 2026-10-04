# Physics digital twin (initial, replaceable version)

**Status: INITIAL and REPLACEABLE.** The ME member owns the final FE rotor model. This package
gives the pipeline (stiffness identification, physics RUL, IS2 residual) something real to run
against until that model arrives. Every result below is **[SIMULATION]** (evidence Rank 6 at best:
simulation + data). Nothing here is a measurement on the RK4 rig.

## Model

- `fe_beam.py`: 1D Euler-Bernoulli FE rotor, lateral plane, Hermite elements (2 DOF/node, 28
  elements), consistent mass, lumped disk mass, bearing springs K1/K2 plus fixed dashpots at the
  bearing nodes. Steady-state unbalance response from `(K - w^2 M + j w C) q = F`, `F = m e w^2` at the disk.
  Output at the two probe nodes: 1x amplitude (um, peak) and phase (rad). Phase is `arg(X)` for
  `x = Re(X e^{jwt})`; the probe/Keyphasor sign convention must be fixed at install (COE doc).
- `identify.py`: inverse problem. Solves for K1, K2 (log-parameterised, multi-start) by nonlinear
  least squares on the relative complex residual. Returns K plus a linearised 1-sigma relative
  uncertainty (`rel_std`) as the confidence measure.
- `physics_rul.py`: `PhysicsTwin` (stateful). First 20 windows = commissioning: the unbalance is
  calibrated by weighted linear LS at the nominal K, K0 = median identified K. Then
  degradation = `1 - K/K0`; failure at a 25 % drop [TEAM DOC]. RUL = exponential trend fit
  `ln(K/K0) = a + bt`, extrapolated to `ln(0.75)`, interval from slope +/- 1.96 se. RUL is `None`
  until 8 points exist or while the slope is not significantly negative. Windows whose predicted K
  uncertainty exceeds 15 % are not used (see limits).
- C# port: `src/DigitalTwin.Twin` (`FeBeamPhysicsTwin.Update(...)`). Parity test
  `tests/DigitalTwin.Twin.Tests` replays `AI-engine/reports/twin_golden.json` (written by `demo_twin`).

## Assumptions (all in `params.py`)

| Parameter | Value | Provenance |
|---|---|---|
| E, rho | 200 GPa, 7850 kg/m3 | [CITED] textbook steel |
| Shaft diameter / length | 10 mm / 0.56 m | [ASSUMED - needs ME confirmation] (RK4 datasheet 141592 geometry not retrieved) |
| Bearing nodes | x = 0.06 m, 0.50 m | [ASSUMED - needs ME confirmation] |
| Probe nodes | x = 0.10 m, 0.46 m | [ASSUMED - needs ME confirmation]; one radial probe per plane [TEAM DOC] |
| Disk mass / position | 0.8 kg at 0.22 m | [ASSUMED - needs ME confirmation] |
| Healthy bearing K | 5e4 N/m each | [ASSUMED - needs ME confirmation] |
| Bearing damping | 200 N s/m, held fixed | [ASSUMED] |
| Nominal unbalance | 1e-5 kg m, 0.5 rad | [ASSUMED], simulation only; real value from commissioning |
| Failure criterion | K drop >= 25 % | [TEAM DOC] Project Overview |
| Run speeds | 1750, 3600 rpm; window 20 rev | [TEAM DOC] COE docs |
| Noise in tests | +/-5 % amplitude, +/-0.05 rad phase (uniform) | [ASSUMED] sensor model |

Simplifications: translational lateral DOF only (one plane, no gyroscopics, no cross-coupled
bearing terms), one probe per plane (so X/Y orbit is not measured, per the COE doc), no shaft
damping. Eigen-check: first natural frequency 41 Hz vs 42 Hz from a closed-form beam estimate.

## Verification results [SIMULATION] (`python -m aiengine.twin.demo_twin`)

S2 (K identification error <= 10 %, known unbalance, forward-model data): noise-free error
< 0.05 % in all cases. With noise, at 1750 rpm the mean error is 4.8-6.5 % (p95 10-14 %, max up to
15 %), so the **mean** target holds but individual noisy windows can exceed 10 %. At 3600 rpm the
mean error is 19-24 %: the speed is above the bearing mode and K is weakly observable, so S2 is **not**
met there. Full per-case table: `AI-engine/reports/twin_s2_verification.json`. The test generates and
identifies with the same model (an "inverse crime"), so it is an upper bound on what a real rig
will show.

Physics RUL (K1 exponential decay, 1 window/h, 10 seeds): mean error 16 % at 50 % of life,
11 % at 70 %, 24 % at 85 %.

## How IS2 would be computed

`residual = |physics_RUL - AI_RUL| / AI_RUL`, evaluated per window and aggregated (max or p95 as
the spec wording decides) over health stages 1 to 3, with the target <= 5 %. Both RULs must use the
same failure definition and time base. Note the contradiction in Project Overview section 3:
the physics RUL here is independent of the AI (stiffness threshold) while the AI RUL
forecasts a health index with a class-selected threshold. A 5 % agreement between two independently
noisy estimates is demanding: the physics RUL's own error above is 11-24 % in simulation, so IS2 may
fail on noise alone, whatever the models' correctness.

## Honest limits

- The public datasets (XJTU-SY, IMS, MaFaulDa) have **no displacement probes**, so the physics twin
  cannot be exercised on them. IS2 can only be exercised in simulation until the rig runs.
- All geometry is assumed. Until ME supplies the real rotor (and a modal/impact test or a
  known-stiffness reference run), absolute K values are not physical statements about the RK4.
- A 25 % K drop as the failure criterion is a team-doc choice, not a validated bearing-life limit.
- Unbalance calibration assumes the commissioning bearings are at nominal stiffness; if they are not,
  the unbalance estimate absorbs the error and K0 is biased.
- Confidence gating: windows at speeds with poor K sensitivity are skipped (3600 rpm in this model).
  If only 3600 rpm windows are available, the twin keeps its last estimate.
