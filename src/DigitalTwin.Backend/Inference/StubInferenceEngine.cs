using DigitalTwin.Backend.Models;

namespace DigitalTwin.Backend.Inference;

/// <summary>
/// SIMULATED engine used when the trained models/contract are absent. Heuristic only: class = dominant
/// bearing-order family once HI rises, RUL = linear in (1 - HI). It is NOT a model and carries no accuracy claim.
/// </summary>
public sealed class StubInferenceEngine : IInferenceEngine
{
    private readonly double _hoursPerWindow;
    public StubInferenceEngine(double hoursPerWindow, FeatureConstants? features = null)
    { _hoursPerWindow = hoursPerWindow; Features = features ?? FeatureConstants.StubDefaults; }

    public string Name => "StubInferenceEngine";
    public bool IsSimulated => true;
    public string Provenance => "SIMULATED: heuristic stub, no trained model loaded";
    public FeatureConstants Features { get; }
    public int RequiredHistory => 1;

    public BearingTracker CreateTracker(int bearing, int baselineWindowsOverride) =>
        new(bearing, new FeatureEngineer(Features), baselineWindowsOverride > 0 ? baselineWindowsOverride : Features.BaselineWindows, RequiredHistory);

    public BearingResult Infer(BearingTracker t)
    {
        var d = t.Latest!;
        double V(string n) => d[t.FeatureIndex(n)];
        var hi = V("hi");
        string[] fam = { "cage", "ball", "outer_race", "inner_race" };
        double[] shares = { V("ftf_share"), V("bsf_share"), V("bpfo_share"), V("bpfi_share") };
        var k = Array.IndexOf(shares, shares.Max());
        var faulty = hi > 0.15;
        var cls = faulty ? fam[k] : "healthy";
        var conf = faulty ? Math.Clamp(50 + 50 * V("family_contrast") * 2, 0, 99) : Math.Clamp(99 - hi * 200, 50, 99);
        var rul = Math.Max(0, 1.0 - hi) * 1000 * _hoursPerWindow;
        var probs = new Dictionary<string, double> { [cls] = conf / 100 };
        return new BearingResult(t.Bearing, BearingStates.Ready, cls, conf, hi, rul, probs, t.Fill, t.Required);
    }
}
