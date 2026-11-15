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
            results = _engine.PostProcessWindow(results, _trackers).ToList();
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

    private long _liveSeen;
    /// <summary>Live path: the Python NodeEngine already produced the per-bearing result; we only map it to state and broadcast.
    /// No C# model runs (the C# ONNX contracts are v3b, not valid for node firmware features). Physics twin = n/a (node sends features only).</summary>
    public async Task<(bool Ok, string Error)> IngestLiveAsync(LiveWindow w, CancellationToken ct = default)
    {
        var t0 = Stopwatch.GetTimestamp();
        if (w.Features.Length != 32 || w.Features.Any(x => !double.IsFinite(x))) { _lat.CountReject(); return (false, "features must be 32 finite numbers"); }
        if (w.Bearings.Count == 0) return (false, "no bearings");
        var ackMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var results = w.Bearings.Select(b =>
        {
            var ok = b.Status == "ok";
            var det = ok ? new BearingDetails(b.HealthStage ?? 0, b.OnsetDetected ?? false, "", Array.Empty<double>(), 0, 0, 0, 0) : null;
            // RUL only when the engine marks it valid (simulated-trajectory replay, faulted bearing, stage >= 2); live node -> null -> n/a
            double? rul = ok && b.RulValid == true && b.AiRulHours is double ai && double.IsFinite(ai) ? ai : null;
            return new BearingResult(b.Bearing, ok ? BearingStates.Ready : BearingStates.WarmingUp,
                ok ? b.FaultClass : "warming_up", (b.Confidence ?? 0) * 100, b.HealthIndex, rul, b.ClassProbs, 0, 0, det);
        }).ToList();
        var physRul = w.Bearings.Where(b => b.RulValid == true && b.PhysicsRulHours is double pr && double.IsFinite(pr))
            .OrderBy(b => b.AiRulHours ?? double.MaxValue).Select(b => b.PhysicsRulHours).FirstOrDefault();
        PipelineState state;
        lock (_gate)
        {
            var info = new LiveInfo(w.Seq, w.WindowId, w.Rpm, w.ValidChannels, w.DeviceId, w.SessionId, w.Contract, w.Provenance,
                w.Features, w.BaselinePhase, w.InferenceMs, w.ReceivedAtMs, w.PredictedAtMs, w.PostedAtMs, ackMs, w.PostsDropped, w.Bearings);
            var win = new WindowPayload((float)w.Rpm, w.Features.Select(x => (float)x).ToArray(), new float[4], 0);
            state = new PipelineState(win, results, DateTimeOffset.UtcNow, w.InferenceMs, ++_liveSeen, physRul, info);
            _store.Set(state);
        }
        if (w.ReceivedAtMs > 0) _lat.Record("live_recv_to_backend_ack", ackMs - w.ReceivedAtMs);
        if (w.PostedAtMs > 0) _lat.Record("live_post_to_backend_ack", Math.Max(0, ackMs - w.PostedAtMs));
        _lat.Record("live_inference", w.InferenceMs);
        await _pub.PublishAsync(_factory.Build(state, DateTimeOffset.UtcNow), ct);
        _lat.Record("live_ingest_to_broadcast", LatencyRecorder.ToMs(Stopwatch.GetTimestamp() - t0));
        return (true, "");
    }
}
