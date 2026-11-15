namespace DigitalTwin.Backend.Models;

/// <summary>One bearing's result from the Python NodeEngine (camelCase JSON on POST /api/live/prediction).</summary>
public sealed record LiveBearing(
    int Bearing, string Status, string FaultClass,
    Dictionary<string, double>? ClassProbs = null, double? Confidence = null,
    double? HealthIndex = null, int? HealthStage = null, bool? OnsetDetected = null, string? Reason = null,
    double? AiRulHours = null, double? PhysicsRulHours = null, double? MinerD = null, double? PhysicsHealthIndex = null,
    string? PhysicsSource = null, bool? RulValid = null, string? RulNote = null, string? TimeBase = null);

/// <summary>Body of POST /api/live/prediction: node window + AI result produced by COE/inference (Python NodeEngine).
/// Timestamps are epoch milliseconds on the laptop clock (the node sends none; the node->laptop hop is not measurable).</summary>
public sealed class LiveWindow
{
    public long Seq { get; set; }
    public long WindowId { get; set; }
    public double Rpm { get; set; }
    public int ValidChannels { get; set; } = 15;
    public string DeviceId { get; set; } = "";
    public string SessionId { get; set; } = "";
    public string Contract { get; set; } = "";
    /// <summary>"[LIVE rig data]" for the real node, "[REPLAY of recorded rig data]" for the rehearsal replayer.</summary>
    public string Provenance { get; set; } = "[LIVE rig data]";
    /// <summary>hybrid_rig only (null otherwise): non-visual provenance of stage/RUL/physics, e.g. "sim-trained head (hybrid_rig), not validated on rig".</summary>
    public string? RulSource { get; set; }
    public string? RulContract { get; set; }
    public string? ClassContract { get; set; }
    public double? RulTimescale { get; set; }
    public double[] Features { get; set; } = Array.Empty<double>();
    public string? BaselinePhase { get; set; }
    public double InferenceMs { get; set; }
    public long ReceivedAtMs { get; set; }
    public long PredictedAtMs { get; set; }
    public long PostedAtMs { get; set; }
    public long PostsDropped { get; set; }
    public List<LiveBearing> Bearings { get; set; } = new();
}

/// <summary>Live block attached to the dashboard snapshot (null for replay/sim sources).</summary>
public sealed record LiveInfo(
    long Seq, long WindowId, double Rpm, int ValidChannels, string DeviceId, string SessionId, string Contract, string Provenance,
    double[] Features, string? BaselinePhase, double InferenceMs,
    long ReceivedAtMs, long PredictedAtMs, long PostedAtMs, long BackendAckMs, long PostsDropped,
    IReadOnlyList<LiveBearing> Bearings,
    string? RulSource = null, string? RulContract = null, string? ClassContract = null, double? RulTimescale = null);

public sealed record RenderAck(long Seq, double RenderMs);
