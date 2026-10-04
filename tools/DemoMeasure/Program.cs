// Measurement client for the PPR demo. Connects to the backend's SignalR hub exactly like the dashboard does.
//   rate    <baseUrl> <seconds>                 : snapshot rate + per-bearing alert timeline against a running replay
//   latency <baseUrl> <udpPort> <payloads.bin> <hz> : sends demo payloads over UDP (rpm nudged by k*0.002 as a marker) to a
//                                                  backend with replay disabled; reports UDP-send -> client-snapshot latency
using System.Diagnostics;
using System.Net.Sockets;
using Microsoft.AspNetCore.SignalR.Client;

var mode = args.Length > 0 ? args[0] : "rate";
var baseUrl = args.Length > 1 ? args[1] : "http://localhost:5080";

var snaps = new System.Collections.Concurrent.ConcurrentQueue<(long Ticks, Snap S)>();
var conn = new HubConnectionBuilder().WithUrl(baseUrl.TrimEnd('/') + "/hubs/dashboard").Build();
conn.On<Snap>("snapshot", s => snaps.Enqueue((Stopwatch.GetTimestamp(), s)));
await conn.StartAsync();
double Ms(long a, long b) => (b - a) * 1000.0 / Stopwatch.Frequency;

if (mode == "rate")
{
    var secs = args.Length > 2 ? double.Parse(args[2]) : 30;
    await Task.Delay(500);
    while (snaps.TryDequeue(out _)) { }
    var t0 = Stopwatch.GetTimestamp();
    await Task.Delay(TimeSpan.FromSeconds(secs));
    var t1 = Stopwatch.GetTimestamp();
    var list = snaps.ToArray();
    var el = Ms(t0, t1) / 1000;
    var gaps = list.Zip(list.Skip(1), (a, b) => Ms(a.Ticks, b.Ticks)).OrderBy(x => x).ToArray();
    Console.WriteLine($"MEASURED client snapshot rate: {list.Length} snapshots in {el:F2}s = {list.Length / el:F2} Hz; inter-arrival p50={gaps[gaps.Length / 2]:F1} ms p95={gaps[(int)(gaps.Length * 0.95)]:F1} ms max={gaps[^1]:F1} ms");
    // 1-second windows: minimum rate over any 1 s window
    var perSec = list.GroupBy(x => (int)(Ms(t0, x.Ticks) / 1000)).Select(g => g.Count()).ToArray();
    Console.WriteLine($"MEASURED per-second counts: min={perSec.Min()} max={perSec.Max()} (S9 needs >= 10 Hz)");
    string? last = null; var n = 0;
    foreach (var (ticks, s) in list)
    {
        var key = $"{s.FaultClass} | b1={s.Bearing1Fault ?? "-"} | b2={s.Bearing2Fault ?? "-"}";
        if (key != last) { Console.WriteLine($"  t={Ms(t0, ticks) / 1000,6:F2}s  {key}  aiRul={s.AiRulHours:F2}h hi={s.HealthIndexPercent:F1}% conf={s.FaultConfidencePercent:F1}% phys={(s.PhysicsRulHours?.ToString("F2") ?? "null")} resid={(s.RulResidualPercent?.ToString("F1") ?? "null")} machine={s.MachineId}"); last = key; }
        n++;
    }
    var lastSec = -1;
    foreach (var (ticks, s) in list)
    {
        var sec = (int)(Ms(t0, ticks) / 2000);
        if (s.PhysicsRulHours is not null && sec != lastSec) { lastSec = sec; Console.WriteLine($"  [series] t={Ms(t0, ticks) / 1000,5:F1}s ai={s.AiRulHours:F2}h physics={s.PhysicsRulHours:F2}h residual={s.RulResidualPercent:F1}%"); }
    }
    var withPhys = list.Count(x => x.S.PhysicsRulHours is not null);
    Console.WriteLine($"snapshots with PhysicsRulHours != null: {withPhys}/{list.Length}; with RulResidualPercent: {list.Count(x => x.S.RulResidualPercent is not null)}");
    Console.WriteLine($"last: aiRul={list[^1].S.AiRulHours:F2} class={list[^1].S.FaultClass} latencyMs(server decode+feat+inf)={list[^1].S.ProcessingLatencyMs}");
}
else
{
    var port = int.Parse(args[2]); var bytes = File.ReadAllBytes(args[3]); var hz = double.Parse(args[4]);
    var count = bytes.Length / 152;
    using var udp = new UdpClient(); udp.Connect("127.0.0.1", port);
    var lat = new List<double>(); var latReady = new List<double>();
    await Task.Delay(500);
    for (var k = 0; k < count; k++)
    {
        var pkt = bytes.AsSpan(k * 152, 152).ToArray();
        var rpm = BitConverter.ToSingle(pkt, 0) + (k + 1) * 0.002f;       // marker only
        BitConverter.GetBytes(rpm).CopyTo(pkt, 0);
        var t0 = Stopwatch.GetTimestamp();
        await udp.SendAsync(pkt, pkt.Length);
        double? got = null;
        var dl = t0 + Stopwatch.Frequency * 2;
        while (got is null && Stopwatch.GetTimestamp() < dl)
        {
            foreach (var (ticks, s) in snaps) if (ticks >= t0 && Math.Abs(s.Rpm - rpm) < 5e-4) { got = Ms(t0, ticks); break; }
            if (got is null) await Task.Delay(1);
        }
        if (got is null) { Console.WriteLine($"no snapshot for payload {k}"); continue; }
        lat.Add(got.Value);
        if (k >= 24 + 12) latReady.Add(got.Value);   // after baseline + TFT buffer: real ONNX inference in the loop
        await Task.Delay(TimeSpan.FromSeconds(1.0 / hz));
    }
    double P(List<double> l, double q) { var s = l.OrderBy(x => x).ToArray(); return s[Math.Clamp((int)Math.Ceiling(q * s.Length) - 1, 0, s.Length - 1)]; }
    Console.WriteLine($"MEASURED UDP-send -> client-snapshot latency, ALL n={lat.Count}: p50={P(lat, .5):F2} p95={P(lat, .95):F2} max={lat.Max():F2} ms");
    Console.WriteLine($"MEASURED same, ONNX-active windows only (payload >= 37) n={latReady.Count}: p50={P(latReady, .5):F2} p95={P(latReady, .95):F2} max={latReady.Max():F2} ms (IS1 budget 500 ms)");
}
await conn.DisposeAsync();

public sealed record Snap(string SchemaVersion, DateTimeOffset TimestampUtc, string MachineId, double Rpm, double AiRulHours, string FaultClass,
    double FaultConfidencePercent, double HealthIndexPercent, double ProcessingLatencyMs, bool IsConnected, double? PhysicsRulHours,
    double? RulResidualPercent, string? Bearing1Fault, string? Bearing2Fault);
