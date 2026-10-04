using System.Globalization;
using System.Net;
using System.Net.Sockets;
using DigitalTwin.Backend.Config;
using DigitalTwin.Backend.Models;
using DigitalTwin.Backend.Services;
using Microsoft.Extensions.Options;

namespace DigitalTwin.Backend.Ingest;

/// <summary>UDP listener: one datagram = one 152-byte payload. Anything else is logged and counted as a reject.</summary>
public sealed class UdpIngestService : BackgroundService
{
    private readonly Pipeline _pipe; private readonly UdpOptions _o; private readonly ILogger<UdpIngestService> _log;
    public UdpIngestService(Pipeline p, IOptions<BackendOptions> o, ILogger<UdpIngestService> l) { _pipe = p; _o = o.Value.Udp; _log = l; }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        if (!_o.Enabled) { _log.LogInformation("UDP ingest disabled"); return; }
        using var udp = new UdpClient(new IPEndPoint(IPAddress.Parse(_o.BindAddress), _o.Port));
        _log.LogInformation("UDP ingest listening on {Ep}", udp.Client.LocalEndPoint);
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var r = await udp.ReceiveAsync(ct);
                await _pipe.IngestAsync(r.Buffer, "udp:" + r.RemoteEndPoint, ct);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { _log.LogWarning(ex, "UDP receive error"); }
        }
    }
}

/// <summary>Streams a recorded run (.bin/.csv) or a SYNTHETIC generated run through the same pipeline (rig-less demo).</summary>
public sealed class ReplaySource : BackgroundService
{
    private readonly Pipeline _pipe; private readonly ReplayOptions _o; private readonly ILogger<ReplaySource> _log;
    public ReplaySource(Pipeline p, IOptions<BackendOptions> o, ILogger<ReplaySource> l) { _pipe = p; _o = o.Value.Replay; _log = l; }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        if (!_o.Enabled) return;
        var records = _o.Synthetic && string.IsNullOrEmpty(_o.Path) ? SyntheticWindows.DegradingRun(600) : Load(_o.Path);
        _log.LogWarning("REPLAY enabled: {N} records at {Hz} Hz from {Src}", records.Count, _o.RateHz,
            string.IsNullOrEmpty(_o.Path) ? "SYNTHETIC generator" : _o.Path);
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(1.0 / Math.Max(_o.RateHz, 0.001)));
        do
        {
            foreach (var rec in records)
            {
                if (!await timer.WaitForNextTickAsync(ct)) return;
                await _pipe.IngestAsync(PayloadCodec.Encode(rec), "replay", ct);
            }
        } while (_o.Loop && !ct.IsCancellationRequested);
    }

    public static List<WindowPayload> Load(string path)
    {
        var list = new List<WindowPayload>();
        if (path.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
        {
            uint t = 0;
            foreach (var line in File.ReadLines(path))
            {
                var p = line.Split(',');
                if (p.Length < 37 || !float.TryParse(p[0], NumberStyles.Float, CultureInfo.InvariantCulture, out _)) continue; // header/blank
                var v = p.Take(37).Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                list.Add(new WindowPayload(v[0], v[1..33], v[33..37], t += 1200));
            }
        }
        else
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length % PayloadCodec.Size != 0)
                throw new InvalidDataException($"{path}: {bytes.Length} bytes is not a multiple of {PayloadCodec.Size}");
            for (var o = 0; o < bytes.Length; o += PayloadCodec.Size)
            {
                if (!PayloadCodec.TryDecode(bytes.AsSpan(o, PayloadCodec.Size), out var w, out var err)) throw new InvalidDataException(err);
                list.Add(w!);
            }
        }
        return list;
    }
}

/// <summary>SYNTHETIC payload generator (not rig data): healthy plateau, then exponential growth with an outer-race signature.</summary>
public static class SyntheticWindows
{
    public static List<WindowPayload> DegradingRun(int n, int healthyWindows = 60, int seed = 7)
    {
        var rng = new Random(seed); var list = new List<WindowPayload>(n);
        for (var i = 0; i < n; i++)
        {
            var g = i < healthyWindows ? 1.0 : Math.Exp(3.0 * (i - healthyWindows) / Math.Max(1, n - healthyWindows));
            list.Add(Make(i, 1750f, g, 1.0, rng));
        }
        return list;
    }

    /// <param name="growth">amplitude multiplier on bearing 1 outer-race signature</param>
    public static WindowPayload Make(int index, float rpm, double growth, double b2Growth, Random rng)
    {
        var f = new float[32];
        for (var ch = 0; ch < 4; ch++)
        {
            var gain = ch < 2 ? growth : b2Growth;
            double Jit() => 1 + 0.02 * (rng.NextDouble() - 0.5);
            var o = ch * 8;
            f[o + 0] = (float)(0.5 * (1 + 0.3 * (gain - 1)) * Jit());   // bp_rms
            f[o + 1] = (float)(3.0 + 0.5 * (gain - 1));                  // kurtosis
            f[o + 2] = (float)(4.0 + 0.3 * (gain - 1));                  // crest
            f[o + 3] = (float)(0.2 * gain * Jit());                      // env_rms
            f[o + 4] = (float)(0.01 * Jit());                            // ftf
            f[o + 5] = (float)(0.01 * Jit());                            // bsf
            f[o + 6] = (float)(0.01 * (1 + 3 * (gain - 1)) * Jit());     // bpfo
            f[o + 7] = (float)(0.01 * Jit());                            // bpfi
        }
        return new WindowPayload(rpm, f, new[] { 20f, 0.1f, 22f, 0.2f }, (uint)(index * 1200));
    }
}
