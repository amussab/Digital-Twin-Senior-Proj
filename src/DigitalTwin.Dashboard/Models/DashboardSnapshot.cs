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
    double? HealthIndex = null, int? HealthStage = null, bool? OnsetDetected = null, string? Reason = null,
    double? AiRulHours = null, double? PhysicsRulHours = null, double? MinerD = null, double? PhysicsHealthIndex = null,
    string? PhysicsSource = null, bool? RulValid = null, string? RulNote = null, string? TimeBase = null);

public sealed record LiveInfo(
    long Seq, long WindowId, double Rpm, int ValidChannels, string DeviceId, string SessionId, string Contract, string Provenance,
    double[] Features, string? BaselinePhase, double InferenceMs,
    long ReceivedAtMs, long PredictedAtMs, long PostedAtMs, long BackendAckMs, long PostsDropped,
    IReadOnlyList<LiveBearing> Bearings,
    string? RulSource = null, string? RulContract = null, string? ClassContract = null, double? RulTimescale = null,
    LiveTwinInfo? FeTwin = null);

/// <summary>ME FE digital twin on the live rig (copy of the backend record).</summary>
public sealed record LiveTwinInfo(
    string Status, string Note, int CommissionDone, int CommissionNeeded, double? MeasuredRpm,
    double? MeasP1UmPp, double? MeasP3UmPp, double? ModelP1UmPp, double? ModelP3UmPp, double? ResidualPct,
    double? K1, double? K2, double? StiffnessDropPct, double K0A, double K0D,
    double? PhysicsRulHours, double? UnbalanceGmm, double ModelCriticalRpm);
