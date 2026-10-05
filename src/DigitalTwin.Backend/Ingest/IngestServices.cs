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
    private readonly Pipeline _pipe; private readonly UdpOptions _o; private readonly ILogger<UdpIngestService> _log; private readonly LatencyRecorder _lat;
    public UdpIngestService(Pipeline p, IOptions<BackendOptions> o, ILogger<UdpIngestService> l, LatencyRecorder lat) { _pipe = p; _o = o.Value.Udp; _log = l; _lat = lat; }

    /// <summary>C5 for UDP: only private/loopback senders are accepted (same ranges as the HTTP guard).</summary>
    public static bool IsAllowedSender(IPEndPoint? remote) => remote is null || PrivateNetwork.IsPrivate(remote.Address);

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
                if (!IsAllowedSender(r.RemoteEndPoint))
                {
                    _lat.CountUdpNonLan();
                    _log.LogWarning("C5: dropped UDP datagram from non-LAN sender {Ep}", r.RemoteEndPoint);
                    continue;
                }
                await _pipe.IngestAsync(r.Buffer, "udp:" + r.RemoteEndPoint, ct);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { _log.LogWarning(ex, "UDP receive error"); }
        }
    }
}

/// <summary>Runtime-selectable demo source (config default; POST /api/demo/source switches it live).</summary>
public sealed class DemoControl
{
    public const string Real = "demo";          // recorded XJTU-SY payloads (MEASURED public data)
    public const string TwinSim = "twin-sim";   // [SIMULATION] synthetic degrading run with FE-beam displacement
    public const string Synthetic = "synthetic";
    private readonly string _machineId;
    private volatile string _source;
    private int _version;
    public DemoControl(IOptions<BackendOptions> o)
    {
        _machineId = o.Value.MachineId;
        var r = o.Value.Replay;
        _source = r.Source.ToLowerInvariant() is Real or TwinSim or Synthetic ? r.Source.ToLowerInvariant() : Real;
        if (r.Synthetic && string.IsNullOrEmpty(r.Path) && r.Source == "") _source = Synthetic;
    }
    public string Source => _source;
    public int Version => Volatile.Read(ref _version);
    public bool TrySet(string name)
    {
        name = name.ToLowerInvariant();
        if (name is not (Real or TwinSim or Synthetic)) return false;
        _source = name; Interlocked.Increment(ref _version); return true;
    }
    /// <summary>Label shown on the dashboard (MachineId). Simulation sources are labelled explicitly.</summary>
    public string MachineLabel => _source switch
    {
        TwinSim => _machineId + " [SIMULATION: twin demo]",
        Synthetic => _machineId + " [SYNTHETIC]",
        _ => _machineId + " [REPLAY: XJTU-SY public data]",
    };
}

/// <summary>Streams a recorded run (.bin/.csv) or a generated [SIMULATION] run through the same pipeline (rig-less demo).</summary>
public sealed class ReplaySource : BackgroundService
{
    private readonly Pipeline _pipe; private readonly BackendOptions _opt; private readonly ReplayOptions _o;
    private readonly DemoControl _ctl; private readonly ILogger<ReplaySource> _log;
    public ReplaySource(Pipeline p, DemoControl c, IOptions<BackendOptions> o, ILogger<ReplaySource> l)
    { _pipe = p; _ctl = c; _opt = o.Value; _o = o.Value.Replay; _log = l; }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        if (!_o.Enabled) return;
        await Task.Yield();
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(1.0 / Math.Max(_o.RateHz, 0.001)));
        while (!ct.IsCancellationRequested)
        {
            var ver = _ctl.Version; var src = _ctl.Source;
            List<WindowPayload> records;
            try { records = Records(src); }
            catch (Exception ex) { _log.LogError(ex, "REPLAY: cannot load source {Src}; replay stopped", src); return; }
            _log.LogWarning("REPLAY source={Src}: {N} records at {Hz} Hz ({Prov})", src, records.Count, _o.RateHz,
                src == DemoControl.Real ? "MEASURED XJTU-SY public data, not the rig" : "[SIMULATION] generated data");
            _pipe.Reset();
            foreach (var rec in records)
            {
                if (ver != _ctl.Version) break;
                if (!await timer.WaitForNextTickAsync(ct)) return;
                await _pipe.IngestAsync(PayloadCodec.Encode(rec), "replay", ct);
            }
            if (ver == _ctl.Version && !_o.Loop) return;
        }
    }

    private List<WindowPayload> Records(string src) => src switch
    {
        DemoControl.TwinSim => SyntheticWindows.DegradingRun(400, hoursPerWindow: _opt.HoursPerWindow),
        DemoControl.Synthetic => SyntheticWindows.DegradingRun(600),
        _ => Load(Paths.Resolve(_o.Path)),
    };

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
    public static List<WindowPayload> DegradingRun(int n, int healthyWindows = 60, int seed = 7, double hoursPerWindow = 0)
    {
        var rng = new Random(seed); var list = new List<WindowPayload>(n);
        var beam = new DigitalTwin.Twin.FeBeam(); var unb = System.Numerics.Complex.FromPolarCoordinates(1e-5, 0.5);
        for (var i = 0; i < n; i++)
        {
            var g = i < healthyWindows ? 1.0 : Math.Exp(3.0 * (i - healthyWindows) / Math.Max(1, n - healthyWindows));
            var w = Make(i, 1750f, g, 1.0, rng);
            // Displacements from the FE-beam forward model with bearing-1 stiffness decaying to a 30 % drop
            // at the end of the run (SYNTHETIC; +/-5 % amplitude, +/-0.05 rad phase noise, assumed unbalance).
            var prog = i < healthyWindows ? 0.0 : (double)(i - healthyWindows) / Math.Max(1, n - healthyWindows);
            var k0 = beam.P.KNominal;
            var pr = beam.Probes(k0[0] * Math.Exp(Math.Log(0.70) * prog), k0[1], 1750, unb);
            double N(double v, double rel) => v * (1 + rel * (2 * rng.NextDouble() - 1));
            double Np(double v) => v + 0.05 * (2 * rng.NextDouble() - 1);
            var disp = new[] { (float)N(pr.A1, 0.05), (float)Np(pr.Ph1), (float)N(pr.A2, 0.05), (float)Np(pr.Ph2) };
            var t20 = hoursPerWindow > 0 ? (uint)Math.Round(i * hoursPerWindow * 3.6e6) : w.T20Ms;
            list.Add(w with { Displacements = disp, T20Ms = t20 });
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
