namespace DigitalTwin.Dashboard.Models;

public sealed record DashboardSnapshot(
    string SchemaVersion,
    DateTimeOffset TimestampUtc,
    string MachineId,
    double Rpm,
    double AiRulHours,
    string FaultClass,
    double FaultConfidencePercent,
    double HealthIndexPercent,
    double ProcessingLatencyMs,
    bool IsConnected,
    double? PhysicsRulHours = null,
    double? RulResidualPercent = null,
    string? Bearing1Fault = null,
    string? Bearing2Fault = null,
    int? HealthStage = null,
    LiveInfo? Live = null);

public sealed record LiveBearing(
    int Bearing, string Status, string FaultClass,
    Dictionary<string, double>? ClassProbs = null, double? Confidence = null,
    double? HealthIndex = null, int? HealthStage = null, bool? OnsetDetected = null, string? Reason = null);

public sealed record LiveInfo(
    long Seq, long WindowId, double Rpm, int ValidChannels, string DeviceId, string SessionId, string Contract, string Provenance,
    double[] Features, string? BaselinePhase, double InferenceMs,
    long ReceivedAtMs, long PredictedAtMs, long PostedAtMs, long BackendAckMs, long PostsDropped,
    IReadOnlyList<LiveBearing> Bearings);
