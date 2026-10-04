using System.Diagnostics;
using DigitalTwin.Backend.Config;
using DigitalTwin.Backend.Ingest;
using DigitalTwin.Backend.Inference;
using DigitalTwin.Backend.Models;
using Microsoft.Extensions.Options;

namespace DigitalTwin.Backend.Services;

/// <summary>decode -> per-bearing features -> inference -> state -> immediate broadcast, with per-stage timing.</summary>
public sealed class Pipeline
{
    private readonly IInferenceEngine _engine; private readonly IPhysicsTwin _twin;
    private readonly StateStore _store; private readonly SnapshotFactory _factory; private readonly SnapshotPublisher _pub;
    private readonly LatencyRecorder _lat; private readonly ILogger<Pipeline> _log;
    private readonly BearingTracker[] _trackers;
    private readonly object _gate = new();
    private long _count;

    public Pipeline(IInferenceEngine engine, IPhysicsTwin twin, StateStore store, SnapshotFactory factory,
        SnapshotPublisher pub, LatencyRecorder lat, IOptions<BackendOptions> opt, ILogger<Pipeline> log)
    {
        _engine = engine; _twin = twin; _store = store; _factory = factory; _pub = pub; _lat = lat; _log = log;
        var fe = new FeatureEngineer(engine.Features);
        var baseline = opt.Value.BaselineWindowsOverride > 0 ? opt.Value.BaselineWindowsOverride : engine.Features.BaselineWindows;
        _trackers = new[] { new BearingTracker(1, fe, baseline, engine.RequiredHistory), new BearingTracker(2, fe, baseline, engine.RequiredHistory) };
    }

    public IInferenceEngine Engine => _engine;

    public async Task<(bool Ok, string Error)> IngestAsync(ReadOnlyMemory<byte> raw, string source, CancellationToken ct = default)
    {
        var t0 = Stopwatch.GetTimestamp();
        if (!PayloadCodec.TryDecode(raw.Span, out var window, out var err))
        {
            _lat.CountReject();
            _log.LogError("Payload from {Source} {Error}", source, err);
            return (false, err);
        }
        var t1 = Stopwatch.GetTimestamp();

        PipelineState state; double featMs = 0, infMs = 0;
        lock (_gate)
        {
            var results = new List<BearingResult>(2);
            foreach (var tr in _trackers)
            {
                var a = Stopwatch.GetTimestamp();
                tr.Update(window!.Features, window.Rpm);
                var b = Stopwatch.GetTimestamp();
                featMs += LatencyRecorder.ToMs(b - a);
                if (tr.State == BearingStates.Ready)
                {
                    results.Add(_engine.Infer(tr));
                    infMs += LatencyRecorder.ToMs(Stopwatch.GetTimestamp() - b);
                }
                else
                {
                    results.Add(new BearingResult(tr.Bearing, tr.State,
                        tr.State == BearingStates.Commissioning ? "commissioning" : "warming_up",
                        0, tr.Latest?["hi"], null, null,
                        tr.State == BearingStates.Commissioning ? tr.CommissioningFill : tr.Fill,
                        tr.State == BearingStates.Commissioning ? _trackers[0].Required : tr.Required));
                }
            }
            var worst = SnapshotFactory.Worst(results);
            var phys = worst is null ? null : _twin.EstimateRulHours(window!, worst);
            var decodeMs = LatencyRecorder.ToMs(t1 - t0);
            state = new PipelineState(window, results, DateTimeOffset.UtcNow, decodeMs + featMs + infMs, ++_count, phys);
            _store.Set(state);
            _lat.Record("decode", decodeMs); _lat.Record("features", featMs); _lat.Record("inference", infMs);
        }

        var t2 = Stopwatch.GetTimestamp();
        await _pub.PublishAsync(_factory.Build(state, DateTimeOffset.UtcNow), ct);
        var t3 = Stopwatch.GetTimestamp();
        _lat.Record("broadcast", LatencyRecorder.ToMs(t3 - t2));
        _lat.Record("total_ingest_to_broadcast", LatencyRecorder.ToMs(t3 - t0));
        return (true, "");
    }
}
