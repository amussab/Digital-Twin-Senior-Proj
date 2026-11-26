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
    /// <summary>1X probe readings for the ME FE twin; null from old node firmware.</summary>
    public LiveProbes? Probes { get; set; }
}

/// <summary>Live block attached to the dashboard snapshot (null for replay/sim sources).</summary>
public sealed record LiveInfo(
    long Seq, long WindowId, double Rpm, int ValidChannels, string DeviceId, string SessionId, string Contract, string Provenance,
    double[] Features, string? BaselinePhase, double InferenceMs,
    long ReceivedAtMs, long PredictedAtMs, long PostedAtMs, long BackendAckMs, long PostsDropped,
    IReadOnlyList<LiveBearing> Bearings,
    string? RulSource = null, string? RulContract = null, string? ClassContract = null, double? RulTimescale = null,
    LiveTwinInfo? FeTwin = null);

/// <summary>1X proximity probe readings of one live window (node firmware schema 2+), as sent by COE/inference.
/// Amplitudes are peak-to-peak um, phases are lag degrees from the Keyphasor event.</summary>
public sealed record LiveProbes(
    double? MeasuredRpm,
    double? P1GapUm, double? P1X1UmPp, double? P1X1PhaseDeg, bool P1X1Valid,
    double? P3GapUm, double? P3X1UmPp, double? P3X1PhaseDeg, bool P3X1Valid);

/// <summary>ME FE digital twin on the live rig (see Services/LiveFeTwin). Amplitudes um pk-pk, stiffness N/m.
/// Status: waiting | commissioning | running | holding.</summary>
public sealed record LiveTwinInfo(
    string Status, string Note, int CommissionDone, int CommissionNeeded, double? MeasuredRpm,
    double? MeasP1UmPp, double? MeasP3UmPp, double? ModelP1UmPp, double? ModelP3UmPp, double? ResidualPct,
    double? K1, double? K2, double? StiffnessDropPct, double K0A, double K0D,
    double? PhysicsRulHours, double? UnbalanceGmm, double ModelCriticalRpm);

public sealed record RenderAck(long Seq, double RenderMs);
