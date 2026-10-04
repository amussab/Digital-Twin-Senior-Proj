using System.Runtime.CompilerServices;
using System.Threading.Channels;
using DigitalTwin.Dashboard.Models;
using Microsoft.AspNetCore.SignalR.Client;

namespace DigitalTwin.Dashboard.Services;

/// <summary>
/// IDashboardDataSource backed by the backend's SignalR hub (/hubs/dashboard, event "snapshot").
/// Reconnects automatically. Selected in Program.cs via the "DataSource" config key.
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
