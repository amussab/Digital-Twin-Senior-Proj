using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.AspNetCore.SignalR.Client;

// INTEGRATION NOTE: DashboardSnapshot and IDashboardDataSource below are standalone copies of the types in
// feature/dashboard (same namespaces, same members) so this library compiles on its own. On the integration
// branch delete this file's two copies (see Contracts region) and reference the dashboard's own types.
namespace DigitalTwin.Dashboard.Models
{
    #region Contracts
    public sealed record DashboardSnapshot(
        string SchemaVersion, DateTimeOffset TimestampUtc, string MachineId, double Rpm, double AiRulHours,
        string FaultClass, double FaultConfidencePercent, double HealthIndexPercent, double ProcessingLatencyMs,
        bool IsConnected, double? PhysicsRulHours = null, double? RulResidualPercent = null);
    #endregion
}

namespace DigitalTwin.Dashboard.Services
{
    using DigitalTwin.Dashboard.Models;

    #region Contracts
    public interface IDashboardDataSource
    {
        IAsyncEnumerable<DashboardSnapshot> StreamAsync(CancellationToken cancellationToken);
    }
    #endregion

    /// <summary>
    /// IDashboardDataSource backed by the backend's SignalR hub (/hubs/dashboard, event "snapshot").
    /// Reconnects automatically. Register with: services.AddSingleton&lt;IDashboardDataSource&gt;(
    ///   new SignalRDashboardDataSource("http://BACKEND-LAN-IP:5080/hubs/dashboard"));
    /// </summary>
    public sealed class SignalRDashboardDataSource : IDashboardDataSource
    {
        private readonly string _hubUrl;
        public SignalRDashboardDataSource(string hubUrl) => _hubUrl = hubUrl;

        public async IAsyncEnumerable<DashboardSnapshot> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var channel = Channel.CreateBounded<DashboardSnapshot>(new BoundedChannelOptions(64)
            { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true });

            await using var conn = new HubConnectionBuilder().WithUrl(_hubUrl).WithAutomaticReconnect().Build();
            conn.On<DashboardSnapshot>("snapshot", s => channel.Writer.TryWrite(s));
            conn.Closed += _ => { channel.Writer.TryComplete(); return Task.CompletedTask; };

            // Retry the first connect until the backend is up.
            while (true)
            {
                try { await conn.StartAsync(cancellationToken); break; }
                catch (OperationCanceledException) { yield break; }
                catch { await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken); }
            }

            await foreach (var s in channel.Reader.ReadAllAsync(cancellationToken)) yield return s;
        }
    }
}
