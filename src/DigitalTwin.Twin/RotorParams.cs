namespace DigitalTwin.Twin;

/// <summary>One disk on the shaft: axial position from the coupling end of the shaft, mass, and
/// diametral moment of inertia (m(3r^2 + w^2)/12 for a solid disk).</summary>
public sealed record DiskSpec(double PositionM, double MassKg, double DiametralInertiaKgM2);

/// <summary>
/// ME finite-element model of the team's Bently Nevada RK4 rig (owner: ME, A. Elshenawy).
/// All positions are measured on the rig from the shaft end at the coupling (0 m), 2026-11-16.
/// Bearings: S6700 deep-groove ball bearings in a 100 % infill PLA ring, in blocks A and D.
/// Probes: PROX 1 in block B, PROX 3 in block C, both vertical.
/// Stiffness K0 is a Hertz contact calculation (see BearingStiffness), not a measurement.
/// </summary>
public sealed record RotorParams
{
    public double E { get; init; } = 2.0e11;              // steel shaft, Pa
    public double Rho { get; init; } = 7850.0;            // kg/m3
    public double ShaftD { get; init; } = 0.0099;         // measured, caliper
    public double ShaftLength { get; init; } = 0.550;     // measured, coupling end to far end
    public double[] BearingPositions { get; init; } = { 0.020, 0.530 };   // blocks A and D
    public double[] ProbePositions { get; init; } = { 0.078, 0.470 };     // PROX 1 (block B), PROX 3 (block C)
    public DiskSpec[] Disks { get; init; } =
    {
        new(0.190, 0.800, BearingStiffness.DiskInertia(0.800, 0.0375, 0.0255)),   // 800 g, 75 mm dia, 25.5 mm wide
        new(0.355, 0.800, BearingStiffness.DiskInertia(0.800, 0.0375, 0.0255)),
    };
    public double MaxElementLength { get; init; } = 0.010;  // mesh: about 57 elements

    /// <summary>Healthy bearing stiffness [N/m]: Hertz (S6700) in series with the PLA ring.</summary>
    public double[] KNominal { get; init; } = BearingStiffness.RigK0();
    public double[] CBearing { get; init; } = { 0.0, 0.0 };   // bearing damping, N s/m (not modelled)
    /// <summary>Structural loss factor of the shaft, ASSUMED 0.02 (about 1 % damping). Fit it from
    /// the width of the resonance peak in a rig run-up.</summary>
    public double ShaftLossFactor { get; init; } = 0.02;

    // Physics RUL settings (unchanged from the integration twin).
    public double FailureDrop { get; init; } = 0.25;
    public double RelNoise { get; init; } = 0.05;
    public int NCommission { get; init; } = 20;
    public int MinRulPoints { get; init; } = 8;
    public int RulFitPoints { get; init; } = 200;
    public double MaxRelStd { get; init; } = 0.15;

    public static RotorParams Default { get; } = new();
}
