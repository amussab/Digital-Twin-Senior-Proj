using System.Runtime.CompilerServices;
using DigitalTwin.Dashboard.Models;

namespace DigitalTwin.Dashboard.Services;

public sealed class MockDashboardDataSource : IDashboardDataSource
{
    public async IAsyncEnumerable<DashboardSnapshot> StreamAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var step = 0;

        while (!cancellationToken.IsCancellationRequested)
        {
            string? bearing1Fault = step switch
            {
                >= 15 and < 30 => "Outer-ring fault",
                >= 45 => "Outer-ring fault",
                _ => null
            };

            string? bearing2Fault = step switch
            {
                >= 30 and < 45 => "Inner-ring fault",
                >= 45 => "Inner-ring fault",
                _ => null
            };

            var faulted = bearing1Fault is not null || bearing2Fault is not null;

            var faultClass = (bearing1Fault, bearing2Fault) switch
            {
                (not null, not null) => "Multiple bearing faults",
                (not null, null) => bearing1Fault,
                (null, not null) => bearing2Fault,
                _ => "Healthy"
            };

            var health = Math.Max(42, 96 - (step * 0.7));
            var aiRul = Math.Max(12, 240 - (step * 2.8));
            var physicsRul = Math.Max(12, aiRul * 1.03);

            yield return new DashboardSnapshot(
                SchemaVersion: "1.0",
                TimestampUtc: DateTimeOffset.UtcNow,
                MachineId: "RK4-01",
                Rpm: step % 12 < 6 ? 1750 : 3600,
                AiRulHours: aiRul,
                FaultClass: faultClass,
                FaultConfidencePercent: faulted ? 91.4 : 96.8,
                HealthIndexPercent: health,
                ProcessingLatencyMs: 146 + (step % 7),
                IsConnected: true,
                PhysicsRulHours: physicsRul,
                RulResidualPercent: 3.0,
                Bearing1Fault: bearing1Fault,
                Bearing2Fault: bearing2Fault);

            step = (step + 1) % 60;

            await Task.Delay(
                TimeSpan.FromMilliseconds(500),
                cancellationToken);
        }
    }
}