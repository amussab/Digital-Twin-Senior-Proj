using System.Collections.Concurrent;
using System.Diagnostics;

namespace DigitalTwin.Backend.Services;

public sealed record StageStats(string Stage, long Count, double P50Ms, double P95Ms, double MaxMs);

/// <summary>Per-stage latency ring buffers (last 2000 samples each) for IS1/S8 evidence.</summary>
public sealed class LatencyRecorder
{
    private const int Capacity = 2000;
    private sealed class Ring { public readonly double[] Data = new double[Capacity]; public long N; }
    private readonly ConcurrentDictionary<string, Ring> _rings = new();
    private long _rejects;

    public static double ToMs(long stopwatchTicks) => stopwatchTicks * 1000.0 / Stopwatch.Frequency;

    public void Record(string stage, double ms)
    {
        var r = _rings.GetOrAdd(stage, _ => new Ring());
        lock (r) { r.Data[r.N % Capacity] = ms; r.N++; }
    }

    public void CountReject() => Interlocked.Increment(ref _rejects);
    public long Rejects => Interlocked.Read(ref _rejects);

    private long _udpNonLan;
    public void CountUdpNonLan() => Interlocked.Increment(ref _udpNonLan);
    public long UdpRejectedNonLan => Interlocked.Read(ref _udpNonLan);

    public IReadOnlyList<StageStats> Snapshot() =>
        _rings.OrderBy(k => k.Key).Select(kv =>
        {
            double[] copy; long n;
            lock (kv.Value) { n = kv.Value.N; copy = kv.Value.Data.Take((int)Math.Min(n, Capacity)).ToArray(); }
            Array.Sort(copy);
            double P(double q)
            {
                if (copy.Length == 0) return 0;
                var i = (int)Math.Ceiling(q * copy.Length) - 1;
                return copy[Math.Clamp(i, 0, copy.Length - 1)];
            }
            return new StageStats(kv.Key, n, Math.Round(P(0.5), 4), Math.Round(P(0.95), 4), Math.Round(P(1.0), 4));
        }).ToList();
}
