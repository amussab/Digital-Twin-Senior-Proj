namespace DigitalTwin.Backend.Models;

/// <summary>
/// Wire contract with the dashboard (copy of feature/dashboard DigitalTwin.Dashboard.Models.DashboardSnapshot;
/// same members, same order). SignalR serialises to camelCase JSON, so the namespace does not matter on the wire.
/// FaultClass "warming_up"/"commissioning" and AiRulHours = -1 mean "no estimate yet".
/// </summary>
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
