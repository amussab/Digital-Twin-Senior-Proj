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

    public BearingResult Infer(BearingTracker t)
    {
        var d = t.Latest!;
        var hi = d["hi"];
        string[] fam = { "cage", "ball", "outer_race", "inner_race" };
        double[] shares = { d["ftf_share"], d["bsf_share"], d["bpfo_share"], d["bpfi_share"] };
        var k = Array.IndexOf(shares, shares.Max());
        var faulty = hi > 0.15;
        var cls = faulty ? fam[k] : "healthy";
        var conf = faulty ? Math.Clamp(50 + 50 * d["family_contrast"] * 2, 0, 99) : Math.Clamp(99 - hi * 200, 50, 99);
        var rul = Math.Max(0, 1.0 - hi) * 1000 * _hoursPerWindow;
        var probs = new Dictionary<string, double> { [cls] = conf / 100 };
        return new BearingResult(t.Bearing, BearingStates.Ready, cls, conf, hi, rul, probs, t.Fill, t.Required);
    }
}
