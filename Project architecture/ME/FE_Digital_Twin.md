# FE digital twin of the RK4 rig (ME)

Owner: Abdelrahman Elshenawy (ME). Code: `src/DigitalTwin.Twin` (`FeBeam.cs`, `BearingStiffness.cs`,
`RotorParams.cs`). Tests: `tests/DigitalTwin.Twin.Tests/MeModelTests.cs`. The integration code that
turns stiffness into a physics RUL (`FeBeamPhysicsTwin.cs`, `Identifier.cs`, `PhysicsRul.cs`) is
unchanged.

## Rig geometry (measured on the rig, 16 Nov 2026)

Positions from the shaft end at the coupling.

| Item | Position (mm) | Notes |
|---|---|---|
| Bearing 1 (block A) | 20 | S6700 in a PLA ring |
| PROX 1 (block B) | 78 | vertical |
| Disk 1 | 190 | 800 g, 75 mm diameter, 25.5 mm wide |
| Disk 2 | 355 | 800 g, 75 mm diameter, 25.5 mm wide |
| PROX 3 (block C) | 470 | vertical |
| Bearing 2 (block D) | 530 | S6700 in a PLA ring |
| Shaft end | 550 | shaft diameter 9.9 mm |

## Model

- Euler-Bernoulli beam elements, 2 degrees of freedom per node (deflection, slope), 57 elements.
  Nodes sit at every bearing, probe and disk and at mid-span.
- Bearings are springs K at their nodes. Disks are point masses with diametral inertia. The 20 mm
  overhang past each bearing is included.
- Unbalance force m·e·ω², shared equally by the two disks. Shaft loss factor 0.02 (assumed; fit it
  from a run-up).
- Equation solved: [K(1 + iη) − ω²M]·x = F.

## Healthy stiffness K0 (calculation, not a measurement)

- Load per bearing from the shaft and disk weights: 9.56 N (A) and 9.40 N (D).
- Hertz point contact for one S6700 ball (11 balls of 1.2 mm, pitch 12.45 mm, groove radius
  0.52 D), Hamrock and Brewe (1983): Kn = 3.628 × 10⁹ N/m^1.5, K = 1.5W/δ = 7.5 MN/m.
- PLA ring (from `adapter_6f.step`: OD 17.2 mm, bore 15.3 mm, width 3.4 mm, E = 3.5 GPa):
  about 192 MN/m, in series with the bearing.
- K0 = 7.23 MN/m (A) and 7.19 MN/m (D).

## Checks (tests)

| Check | Hand calculation | FE model |
|---|---|---|
| 10 N at mid-span, rigid bearings | PL³/48EI = 293.04 µm | 293.04 µm |
| First critical speed | between 1473 rpm (Dunkerley) and 1606 rpm (Rayleigh) | 1509 rpm |
| Probe reading at 1750 rpm, 10 g·mm unbalance | n/a | 17.2 and 17.7 µm pk-pk |

The probe reading in the last row is close to the first rig readings sent by the node (16 to 19 µm
pk-pk at 1749 rpm, 16 Nov 2026), which fits a residual unbalance of about 10 g·mm. This is a
consistency check, not a measurement of the unbalance.

## Limit of the method with this bearing

The S6700 (about 7 MN/m) is about 100 times stiffer than the shaft between the bearings. The shaft
then sets the probe reading: a 10 % drop in K changes it by less than 0.1 %. The identifier reports
this itself (relative uncertainty on K of 470 to 980 % for 5 % probe noise), so the twin keeps the
healthy K0 and gives no physics RUL until K shows a clear downward trend. With a soft bearing
(45 kN/m) the same method identifies K to about 6 % at 1000 rpm.

## Next check on the rig

A slow run-up from about 500 to 2500 rpm with the COE inference receiver running. The probe
amplitude should peak near 1509 rpm. The measured peak speed validates the model, and the width
of the peak gives the damping.
