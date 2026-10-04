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
}

/// <summary>Digital-twin hook (IS2). A physics agent supplies the real implementation later.</summary>
public interface IPhysicsTwin
{
    /// <summary>Physics-based RUL in hours for the current window, or null when unavailable.</summary>
    double? EstimateRulHours(WindowPayload window, BearingResult worstBearing);
}

public sealed class NullPhysicsTwin : IPhysicsTwin
{
    public double? EstimateRulHours(WindowPayload window, BearingResult worstBearing) => null;
}
