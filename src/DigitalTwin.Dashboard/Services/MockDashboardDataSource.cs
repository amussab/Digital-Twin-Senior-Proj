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
            var faulted = step > 18;
            var health = Math.Max(42, 96 - (step * 0.7));
            var aiRul = Math.Max(12, 240 - (step * 2.8));
            var physicsRul = Math.Max(12, aiRul * 1.03);

            yield return new DashboardSnapshot(
                SchemaVersion: "1.0",
                TimestampUtc: DateTimeOffset.UtcNow,
                MachineId: "RK4-01",
                Rpm: step % 12 < 6 ? 1750 : 3600,
                AiRulHours: aiRul,
                FaultClass: faulted ? "Outer-race defect" : "Healthy",
                FaultConfidencePercent: faulted ? 91.4 : 96.8,
                HealthIndexPercent: health,
                ProcessingLatencyMs: 146 + (step % 7),
                IsConnected: true,
                PhysicsRulHours: physicsRul,
                RulResidualPercent: 3.0);

            step = (step + 1) % 60;
            await Task.Delay(TimeSpan.FromMilliseconds(500), cancellationToken);
        }
    }
}
