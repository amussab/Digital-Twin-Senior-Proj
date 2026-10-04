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
    private BearingTracker[] _trackers;
    private readonly int _baselineOverride;
    private readonly object _gate = new();
    private long _count;
    private long? _t0Ms; private long _wrap; private long _lastMs = -1;

    public Pipeline(IInferenceEngine engine, IPhysicsTwin twin, StateStore store, SnapshotFactory factory,
        SnapshotPublisher pub, LatencyRecorder lat, IOptions<BackendOptions> opt, ILogger<Pipeline> log)
    {
        _engine = engine; _twin = twin; _store = store; _factory = factory; _pub = pub; _lat = lat; _log = log;
        _baselineOverride = opt.Value.BaselineWindowsOverride;
        _trackers = NewTrackers();
    }

    private BearingTracker[] NewTrackers() => new[] { _engine.CreateTracker(1, _baselineOverride), _engine.CreateTracker(2, _baselineOverride) };

    /// <summary>Forget all per-bearing state (new run / source switch). Also resets the physics twin if it supports it.</summary>
    public void Reset()
    {
        lock (_gate)
        {
            _trackers = NewTrackers(); _t0Ms = null; _wrap = 0; _lastMs = -1; _count = 0;
            (_twin as IResettable)?.Reset();
            _store.Set(new PipelineState(null, Array.Empty<BearingResult>(), null, 0, 0, null));
        }
    }

    /// <summary>Hours since the first window from the payload t20_ms (uint32 wrap handled), as engine.py::_hours.</summary>
    private double Hours(uint t20)
    {
        if (_lastMs >= 0 && t20 < _lastMs) _wrap += 1L << 32;
        _lastMs = t20;
        var ms = t20 + _wrap;
        _t0Ms ??= ms;
        return (ms - _t0Ms.Value) / 3.6e6;
    }

    public IInferenceEngine Engine => _engine;
    public IReadOnlyList<BearingTracker> Trackers => _trackers;

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

        PipelineState state; double featMs = 0, infMs = 0, tftMs = 0, nhMs = 0;
        lock (_gate)
        {
            var results = new List<BearingResult>(2);
            var tHours = Hours(window!.T20Ms);
            foreach (var tr in _trackers)
            {
                var a = Stopwatch.GetTimestamp();
                tr.Update(window!.Features, window.Rpm, tHours);
                var b = Stopwatch.GetTimestamp();
                featMs += LatencyRecorder.ToMs(b - a);
                if (tr.State == BearingStates.Ready)
                {
                    var br = _engine.Infer(tr);
                    results.Add(br);
                    infMs += LatencyRecorder.ToMs(Stopwatch.GetTimestamp() - b);
                    if (br.Details is { } d) { tftMs += d.TftMs; nhMs += d.NhitsMs; }
                }
                else
                {
                    results.Add(new BearingResult(tr.Bearing, tr.State,
                        tr.State == BearingStates.Commissioning ? "commissioning" : "warming_up",
                        0, tr.LatestHi, null, null,
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
            if (tftMs > 0) { _lat.Record("tft_onnx", tftMs); _lat.Record("nhits_onnx", nhMs); }
        }

        var t2 = Stopwatch.GetTimestamp();
        await _pub.PublishAsync(_factory.Build(state, DateTimeOffset.UtcNow), ct);
        var t3 = Stopwatch.GetTimestamp();
        _lat.Record("broadcast", LatencyRecorder.ToMs(t3 - t2));
        _lat.Record("total_ingest_to_broadcast", LatencyRecorder.ToMs(t3 - t0));
        return (true, "");
    }
}
