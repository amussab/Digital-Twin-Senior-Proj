namespace DigitalTwin.Backend.Models;

/// <summary>One decoded 152-byte COE window (little-endian, see PayloadCodec).</summary>
public sealed record WindowPayload(
    float Rpm,
    float[] Features,          // 32 floats: B1-X, B1-Y, B2-X, B2-Y x 8 features
    float[] Displacements,     // p1_amp_um, p1_phase_rad, p2_amp_um, p2_phase_rad
    uint T20Ms);

public static class BearingStates
{
    public const string Commissioning = "commissioning"; // collecting the healthy baseline
    public const string WarmingUp = "warming_up";        // baseline done, encoder buffer not full
    public const string Ready = "ready";
}

public sealed record BearingResult(
    int Bearing,
    string State,
    string FaultClass,
    double ConfidencePercent,
    double? HealthIndex,
    double? RulHours,
    IReadOnlyDictionary<string, double>? ClassProbs,
    int BufferFill,
    int BufferRequired,
    BearingDetails? Details = null);

/// <summary>Extra per-bearing engine output (stage, forecast, threshold class, RUL interval) for the API and parity tests.</summary>
public sealed record BearingDetails(
    int HealthStage, bool OnsetDetected, string RulThresholdClass, double[] HiForecast,
    double RulLoHours, double RulHiHours, double TftMs, double NhitsMs,
    [property: System.Text.Json.Serialization.JsonIgnore] double[]? TftEncoder = null,
    [property: System.Text.Json.Serialization.JsonIgnore] double[]? TftDecoder = null);

public sealed record PipelineState(
    WindowPayload? Window,
    IReadOnlyList<BearingResult> Bearings,
    DateTimeOffset? LastPayloadUtc,
    double LastProcessingMs,
    long PayloadCount,
    double? PhysicsRulHours);
