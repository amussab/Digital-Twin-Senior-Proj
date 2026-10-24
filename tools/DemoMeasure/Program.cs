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
    var fullSecs = (int)el;   // only complete 1-s windows; the trailing partial bucket would read as a false low
    var perSec = list.GroupBy(x => (int)(Ms(t0, x.Ticks) / 1000)).Where(g => g.Key < fullSecs).Select(g => g.Count()).ToArray();
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
else if (mode == "is1")
{
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    var port = int.Parse(args[2]); var all = File.ReadAllBytes(args[3]); var n = int.Parse(args[4]);
    var wMin = double.Parse(args[5], ci); var wMax = double.Parse(args[6], ci);
    var coePath = args[7]; var outPath = args[8];
    var hz = args.Length > 9 ? double.Parse(args[9], ci) : 10; var seed = args.Length > 10 ? int.Parse(args[10]) : 1;
    var rng = new Random(seed);
    n = Math.Min(n, all.Length / 152);
    using var udp = new UdpClient(); udp.Connect("127.0.0.1", port);
    void Precise(double ms)
    {
        var end = Stopwatch.GetTimestamp() + (long)(ms * Stopwatch.Frequency / 1000.0);
        while (Stopwatch.GetTimestamp() < end) { if (end - Stopwatch.GetTimestamp() > Stopwatch.Frequency / 500) Thread.Sleep(1); else Thread.SpinWait(50); }
    }
    var wifi = new List<double>(); var backend = new List<double>(); var meas = new List<double>(); var missed = 0;
    await Task.Delay(500);
    for (var k = 0; k < n; k++)
    {
        var pkt = all.AsSpan(k * 152, 152).ToArray();
        var rpm = BitConverter.ToSingle(pkt, 0) + ((k % 50) + 1) * 0.004f;            // marker only (<= 0.2 rpm)
        BitConverter.GetBytes(rpm).CopyTo(pkt, 0);
        var tClose = Stopwatch.GetTimestamp();                                       // COE window close / payload ready (sender side)
        var d = wMin + (wMax - wMin) * rng.NextDouble();                              // [SIMULATION] Wi-Fi hop delay
        Precise(d);
        var tSend = Stopwatch.GetTimestamp();
        await udp.SendAsync(pkt, pkt.Length);
        long? got = null; var dl = tSend + Stopwatch.Frequency * 2;
        while (got is null && Stopwatch.GetTimestamp() < dl)
        {
            foreach (var (ticks, s) in snaps) if (ticks >= tSend && Math.Abs(s.Rpm - rpm) < 0.0015) { got = ticks; break; }
            if (got is null) await Task.Delay(1);
        }
        if (got is null) { missed++; continue; }
        if (k >= 25) { wifi.Add(Ms(tClose, tSend)); backend.Add(Ms(tSend, got.Value)); meas.Add(Ms(tClose, got.Value)); }   // k>=25: TFT buffer full, ONNX active
        while (snaps.TryDequeue(out _)) { }
        await Task.Delay(TimeSpan.FromSeconds(1.0 / hz));
    }
    double P(IReadOnlyList<double> l, double q) { var s2 = l.OrderBy(x => x).ToArray(); return s2[Math.Clamp((int)Math.Ceiling(q * s2.Length) - 1, 0, s2.Length - 1)]; }
    using var cj = System.Text.Json.JsonDocument.Parse(File.ReadAllText(coePath));
    var coe2 = cj.RootElement.GetProperty("COE2");
    double hostP50 = 0, hostP95 = 0, hostMax = 0;
    foreach (var r in coe2.GetProperty("by_rpm").EnumerateObject())
    {
        var t = r.Value.GetProperty("t20_to_payload_ms");
        hostP50 = Math.Max(hostP50, t.GetProperty("p50").GetDouble()); hostP95 = Math.Max(hostP95, t.GetProperty("p95").GetDouble()); hostMax = Math.Max(hostMax, t.GetProperty("max").GetDouble());
    }
    double stm = 0, stmNom = 0;
    foreach (var r in coe2.GetProperty("stm32_estimate").GetProperty("by_rpm").EnumerateObject())
    {
        var e = r.Value;
        stm = Math.Max(stm, e.GetProperty("pessimistic_10cyc_per_flop").GetProperty("plus_wait_for_block_end_ms").GetDouble());
        stmNom = Math.Max(stmNom, e.GetProperty("nominal_2cyc_per_flop").GetProperty("plus_wait_for_block_end_ms").GetDouble());
    }
    var totHost = meas.Select(m => m + hostP95).ToList(); var totStm = meas.Select(m => m + stm).ToList();
    string F(double v) => v.ToString("F1", ci);
    var sb = new System.Text.StringBuilder();
    sb.AppendLine($"# IS1 end-to-end latency measurement (UP2), {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC");
    sb.AppendLine();
    sb.AppendLine($"Spec IS1: sensor -> classify+RUL -> dashboard < 500 ms. Tool: `tools/DemoMeasure is1` (n = {meas.Count} ONNX-active windows of {n} sent, {missed} without a snapshot; payloads = AI-engine/models/up2_demo_payloads.bin, [SYNTHETIC] synth_v3 stream; contract v3).");
    sb.AppendLine($"Method: t_close = sender timestamp when the COE window is complete (payload ready); then a [SIMULATION] Wi-Fi hop delay, uniform {wMin}-{wMax} ms (NOT measured on a real Wi-Fi link; real value needs the ESP32-C6 + AP); then UDP 127.0.0.1 -> backend (decode, feature engineering, ONNX classifier-slot + RUL-slot, SignalR broadcast) -> .NET SignalR client receive (stands in for the browser; Blazor WASM render time is NOT included). One monotonic clock. Windows are sent one at a time at {hz} Hz. Marker: rpm nudged by <= 0.2 rpm to identify each window's snapshot.");
    sb.AppendLine();
    sb.AppendLine("## Measured legs (this PC, Windows host)");
    sb.AppendLine("| Leg | p50 ms | p95 ms | max ms | Provenance |");
    sb.AppendLine("|---|---|---|---|---|");
    sb.AppendLine($"| Wi-Fi hop (injected delay realised) | {F(P(wifi, .5))} | {F(P(wifi, .95))} | {F(wifi.Max())} | [SIMULATION] assumed {wMin}-{wMax} ms jitter |");
    sb.AppendLine($"| UDP + backend (decode, features, 2x ONNX, SignalR) + client receive | {F(P(backend, .5))} | {F(P(backend, .95))} | {F(backend.Max())} | [MEASURED host] localhost; not the target server / Pi 5 |");
    sb.AppendLine($"| Sender window-close -> client receive (two rows above) | {F(P(meas, .5))} | {F(P(meas, .95))} | {F(meas.Max())} | [MEASURED host + SIMULATION] |");
    sb.AppendLine();
    sb.AppendLine("## Budget including COE processing (t20 -> payload ready), from AI-engine/reports/coe_verification.json");
    sb.AppendLine("| Stage | p50 ms | p95 ms | max ms | Provenance |");
    sb.AppendLine("|---|---|---|---|---|");
    sb.AppendLine($"| COE t20 -> payload ready, host PC (Python/NumPy; worst of 1750/3600 rpm) | {F(hostP50)} | {F(hostP95)} | {F(hostMax)} | [MEASURED host], not STM32 |");
    sb.AppendLine($"| COE t20 -> payload ready, STM32 pessimistic (10 cyc/flop, 480 MHz M7 assumed, incl. wait for last DMA block) | - | {F(stm)} | {F(stm)} | [CALCULATED] (nominal 2 cyc/flop: {F(stmNom)} ms) |");
    sb.AppendLine($"| Per-window sum: measured legs + COE host (p50 / p95 / max) | {F(P(meas, .5) + hostP50)} | {F(P(totHost, .95))} | {F(meas.Max() + hostMax)} | [MEASURED host + SIMULATION] |");
    sb.AppendLine($"| Per-window sum: measured legs + COE STM32 pessimistic (conservative) | {F(P(meas, .5) + stm)} | {F(P(totStm, .95))} | {F(meas.Max() + stm)} | mixed incl. [CALCULATED] |");
    sb.AppendLine();
    var worst = meas.Max() + stm;
    sb.AppendLine($"Verdict vs IS1 (< 500 ms): p95 = {F(P(totStm, .95))} ms, max = {F(worst)} ms (STM32 pessimistic) -> " + (worst < 500 ? "within budget for this setup: **MET (design / simulation)**; the hardware leg and the real Wi-Fi hop are still to be measured." : "max EXCEEDS 500 ms; not met on the worst case."));
    sb.AppendLine("Not measured: real Wi-Fi latency/jitter, STM32 timing, browser (Blazor WASM) render time, final server (laptop vs Raspberry Pi 5).");
    File.WriteAllText(outPath, sb.ToString());
    Console.WriteLine(sb.ToString());
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
