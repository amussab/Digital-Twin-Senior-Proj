using DigitalTwin.Backend.Config;
using DigitalTwin.Backend.Models;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;

namespace DigitalTwin.Backend.Services;

public sealed class StateStore
{
    private PipelineState _state = new(null, Array.Empty<BearingResult>(), null, 0, 0, null);
    public PipelineState Current => Volatile.Read(ref _state);
    public void Set(PipelineState s) => Volatile.Write(ref _state, s);
}

public sealed class SnapshotFactory
{
    private readonly BackendOptions _o;
    public SnapshotFactory(IOptions<BackendOptions> o) => _o = o.Value;

    /// <summary>Picks the worse bearing: ready beats warming; then higher health index; then lower RUL.</summary>
    public static BearingResult? Worst(IReadOnlyList<BearingResult> bearings) =>
        bearings.OrderByDescending(b => b.State == BearingStates.Ready ? 2 : b.HealthIndex.HasValue ? 1 : 0)
                .ThenByDescending(b => b.HealthIndex ?? -1)
                .ThenBy(b => b.RulHours ?? double.MaxValue)
                .FirstOrDefault();

    public DashboardSnapshot Build(PipelineState s, DateTimeOffset now)
    {
        var connected = s.LastPayloadUtc is { } t && (now - t).TotalSeconds <= _o.ConnectionTimeoutSeconds;
        var w = Worst(s.Bearings);
        if (w is null || s.Window is null)
            return new DashboardSnapshot("1.0", now, _o.MachineId, 0, -1, "no_data", 0, 0, 0, false);

        var rul = w.RulHours ?? -1;
        double? residual = null;
        // IS2 residual (twin README): |physics - AI| / AI x 100.
        if (s.PhysicsRulHours is { } p && w.RulHours is { } a && a > 0) residual = Math.Abs(p - a) / a * 100;
        return new DashboardSnapshot("1.0", now, _o.MachineId, s.Window.Rpm, rul, w.FaultClass,
            Math.Round(w.ConfidencePercent, 2), Math.Round(Math.Clamp((w.HealthIndex ?? 0) * 100, 0, 100), 2),
            Math.Round(s.LastProcessingMs, 3), connected, s.PhysicsRulHours, residual);
    }
}

public sealed class DashboardHub : Hub
{
    private readonly StateStore _store; private readonly SnapshotFactory _f;
    public DashboardHub(StateStore store, SnapshotFactory f) { _store = store; _f = f; }

    public override async Task OnConnectedAsync()
    {
        await Clients.Caller.SendAsync(SnapshotPublisher.EventName, _f.Build(_store.Current, DateTimeOffset.UtcNow));
        await base.OnConnectedAsync();
    }
}

public sealed class SnapshotPublisher
{
    public const string EventName = "snapshot";
    private readonly IHubContext<DashboardHub> _hub;
    public SnapshotPublisher(IHubContext<DashboardHub> hub) => _hub = hub;
    public Task PublishAsync(DashboardSnapshot s, CancellationToken ct = default) =>
        _hub.Clients.All.SendAsync(EventName, s, ct);
}

/// <summary>Re-broadcasts the latest state every BroadcastIntervalMs even between payloads (spec S9, &gt;=10 Hz).</summary>
public sealed class BroadcastTimer : BackgroundService
{
    private readonly StateStore _store; private readonly SnapshotFactory _f; private readonly SnapshotPublisher _pub;
    private readonly BackendOptions _o; private readonly ILogger<BroadcastTimer> _log;
    public BroadcastTimer(StateStore s, SnapshotFactory f, SnapshotPublisher p, IOptions<BackendOptions> o, ILogger<BroadcastTimer> l)
    { _store = s; _f = f; _pub = p; _o = o.Value; _log = l; }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(_o.BroadcastIntervalMs));
        while (await timer.WaitForNextTickAsync(ct))
        {
            try { await _pub.PublishAsync(_f.Build(_store.Current, DateTimeOffset.UtcNow), ct); }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { _log.LogWarning(ex, "periodic broadcast failed"); }
        }
    }
}
