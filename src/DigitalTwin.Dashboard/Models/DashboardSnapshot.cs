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
    double? RulResidualPercent = null);
