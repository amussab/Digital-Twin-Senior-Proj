using DigitalTwin.Backend.Models;

namespace DigitalTwin.Backend.Inference;

public interface IInferenceEngine
{
    string Name { get; }
    /// <summary>True when outputs are NOT from the trained models (stub). Surfaced in logs and /api/state.</summary>
    bool IsSimulated { get; }
    string Provenance { get; }
    FeatureConstants Features { get; }
    /// <summary>Engineered-feature rows each bearing must buffer before inference is meaningful.</summary>
    int RequiredHistory { get; }
    /// <summary>Run classification + RUL for one bearing whose tracker is Ready.</summary>
    BearingResult Infer(BearingTracker tracker);
    /// <summary>Builds the per-bearing tracker consistent with this engine's contract.</summary>
    BearingTracker CreateTracker(int bearing, int baselineWindowsOverride);
}

/// <summary>Digital-twin hook (IS2). A physics agent supplies the real implementation later.</summary>
public interface IPhysicsTwin
{
    /// <summary>Physics-based RUL in hours for the current window, or null when unavailable.</summary>
    double? EstimateRulHours(WindowPayload window, BearingResult worstBearing);
}

/// <summary>Twins that can drop their state when the demo source changes.</summary>
public interface IResettable { void Reset(); }

public sealed class NullPhysicsTwin : IPhysicsTwin
{
    public double? EstimateRulHours(WindowPayload window, BearingResult worstBearing) => null;
}

/// <summary>
/// IPhysicsTwin over DigitalTwin.Twin.FeBeamPhysicsTwin, fed from the payload displacement fields
/// (p1 amp/phase, p2 amp/phase). Time base is a window-index clock (window k = k x HoursPerWindow h),
/// the same base the AI RUL uses, so replay faster than real time keeps both RULs comparable.
/// INITIAL/REPLACEABLE: rotor geometry is assumed (see AI-engine/aiengine/twin/README.md).
/// </summary>
public sealed class FeBeamPhysicsTwinAdapter : IPhysicsTwin, IResettable
{
    private static readonly DateTimeOffset Epoch = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
    private DigitalTwin.Twin.FeBeamPhysicsTwin _twin = new();
    private readonly double _hoursPerWindow;
    private long _n;
    public FeBeamPhysicsTwinAdapter(double hoursPerWindow) => _hoursPerWindow = hoursPerWindow;

    public DigitalTwin.Twin.TwinUpdate? Last { get; private set; }
    public void Reset() { _twin = new(); _n = 0; Last = null; }

    public double? EstimateRulHours(WindowPayload w, BearingResult worstBearing)
    {
        var d = w.Displacements;
        if (d.Length < 4 || !(d[0] > 0) || !(d[2] > 0)) return null;
        var t = Epoch.AddHours(_n++ * _hoursPerWindow);
        try { Last = _twin.Update(w.Rpm, d[0], d[1], d[2], d[3], t); }
        catch (Exception) { return null; }   // a bad window must not break the pipeline
        return Last.PhysicsRulHours;
    }
}
