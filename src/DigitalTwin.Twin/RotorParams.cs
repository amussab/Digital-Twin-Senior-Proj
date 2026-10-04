namespace DigitalTwin.Twin;

/// <summary>
/// Rotor and twin parameters. Mirrors AI-engine/aiengine/twin/params.py (single provenance table
/// lives there). Geometry values are [ASSUMED - needs ME confirmation]; failure drop is [TEAM DOC].
/// INITIAL, REPLACEABLE: the ME member owns the final FE model.
/// </summary>
public sealed record RotorParams
{
    public double E { get; init; } = 2.0e11;
    public double Rho { get; init; } = 7850.0;
    public double ShaftD { get; init; } = 0.010;
    public double Length { get; init; } = 0.560;
    public int NElem { get; init; } = 28;
    public int[] BearingNodes { get; init; } = { 3, 25 };
    public int[] ProbeNodes { get; init; } = { 5, 23 };
    public int DiskNode { get; init; } = 11;
    public double DiskMass { get; init; } = 0.8;
    public double[] KNominal { get; init; } = { 5.0e4, 5.0e4 };
    public double[] CBearing { get; init; } = { 200.0, 200.0 };
    public double FailureDrop { get; init; } = 0.25;
    public double RelNoise { get; init; } = 0.05;
    public int NCommission { get; init; } = 20;
    public int MinRulPoints { get; init; } = 8;
    public int RulFitPoints { get; init; } = 200;
    public double MaxRelStd { get; init; } = 0.15;

    public static RotorParams Default { get; } = new();
}
