using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text.Json;
using DigitalTwin.Backend.Ingest;
using DigitalTwin.Backend.Inference;
using DigitalTwin.Backend.Models;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.SignalR.Client;
using Xunit;
using Microsoft.Extensions.DependencyInjection;
using DigitalTwin.Backend;
using Xunit.Abstractions;

namespace DigitalTwin.Backend.Tests;

public class CodecTests
{
    [Fact]
    public void RoundTrip_PreservesEveryField()
    {
        var w = new WindowPayload(1750.25f, Enumerable.Range(0, 32).Select(i => i * 0.5f + 0.125f).ToArray(),
            new[] { 1.5f, -0.25f, 2.75f, 3.125f }, 123456u);
        var bytes = PayloadCodec.Encode(w);
        Assert.Equal(152, bytes.Length);
        Assert.True(PayloadCodec.TryDecode(bytes, out var d, out var err), err);
        Assert.Equal(w.Rpm, d!.Rpm);
        Assert.Equal(w.Features, d.Features);
        Assert.Equal(w.Displacements, d.Displacements);
        Assert.Equal(w.T20Ms, d.T20Ms);
    }

    [Fact]
    public void Rejects168BytePayloadLoudly()
    {
        Assert.False(PayloadCodec.TryDecode(new byte[168], out var d, out var err));
        Assert.Null(d);
        Assert.Contains("168", err);
        Assert.Contains("REJECTED", err);
    }
}

public class BackendFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder b)
    {
        b.UseSetting("Backend:Udp:Enabled", "false");
        b.UseSetting("Backend:Replay:Enabled", "false");
        b.UseSetting("Backend:BaselineWindowsOverride", "3");
        b.UseSetting("Backend:PhysicsTwin", "None");
        b.UseSetting("Backend:ModelContractPath", "does-not-exist/model_contract.json");
        b.UseSetting("Urls", "http://127.0.0.1:0");
    }
}

public class EndToEndTests : IClassFixture<BackendFactory>
{
    private readonly BackendFactory _f; private readonly ITestOutputHelper _out;
    public EndToEndTests(BackendFactory f, ITestOutputHelper o) { _f = f; _out = o; }

    private HubConnection Connect() => new HubConnectionBuilder()
        .WithUrl(new Uri(_f.Server.BaseAddress, "/hubs/dashboard"), o =>
        {
            o.HttpMessageHandlerFactory = _ => _f.Server.CreateHandler();
            o.Transports = HttpTransportType.LongPolling;
        }).Build();

    [Fact]
    public async Task Http_Rejects168Bytes_AndCountsIt()
    {
        var c = _f.CreateClient();
        var r = await c.PostAsync("/api/payload", new ByteArrayContent(new byte[168]));
        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        Assert.Contains("168", await r.Content.ReadAsStringAsync());
        var m = await c.GetFromJsonAsync<JsonElement>("/api/metrics");
        Assert.True(m.GetProperty("rejects").GetInt64() >= 1);
    }

    [Fact]
    public async Task Udp_Datagram_Reaches_Pipeline()
    {
        int port; using (var probe = new UdpClient(0)) port = ((IPEndPoint)probe.Client.LocalEndPoint!).Port;
        using var f = new BackendFactory().WithWebHostBuilder(b =>
        {
            b.UseSetting("Backend:Udp:Enabled", "true");
            b.UseSetting("Backend:Udp:Port", port.ToString());
            b.UseSetting("Backend:Udp:BindAddress", "127.0.0.1");
        });
        var client = f.CreateClient();
        using var udp = new UdpClient();
        var rng = new Random(1);
        var bytes = PayloadCodec.Encode(SyntheticWindows.Make(0, 1750f, 1, 1, rng));
        long count = 0;
        for (var i = 0; i < 50 && count == 0; i++)
        {
            await udp.SendAsync(bytes, bytes.Length, new IPEndPoint(IPAddress.Loopback, port));
            await Task.Delay(100);
            count = (await client.GetFromJsonAsync<JsonElement>("/api/state")).GetProperty("payloadCount").GetInt64();
        }
        Assert.True(count > 0, "UDP datagram never reached the pipeline");
    }

    [Fact]
    public async Task Broadcasts_AtLeast10Hz_And_MeasuresIngestToBroadcastLatency()
    {
        var http = _f.CreateClient();
        var seen = new System.Collections.Concurrent.ConcurrentQueue<(long ticks, DashboardSnapshot snap)>();
        var hub = Connect();
        hub.On<DashboardSnapshot>("snapshot", s => seen.Enqueue((Stopwatch.GetTimestamp(), s)));
        await hub.StartAsync();

        // 1) Pure timer rate: no payloads at all for 3 s.
        await Task.Delay(500);
        while (seen.TryDequeue(out _)) { }
        var start = Stopwatch.GetTimestamp();
        await Task.Delay(3000);
        var n = seen.Count; var secs = (Stopwatch.GetTimestamp() - start) / (double)Stopwatch.Frequency;
        var rate = n / secs;
        _out.WriteLine($"MEASURED idle broadcast rate: {n} snapshots in {secs:F2}s = {rate:F2} Hz (spec S9 >= 10 Hz)");
        Assert.True(rate >= 10.0, $"broadcast rate {rate:F2} Hz < 10 Hz");

        // 2) Ingest -> broadcast latency: unique rpm per payload, wait for the snapshot carrying it.
        var rng = new Random(3); var lat = new List<double>();
        for (var i = 0; i < 40; i++)
        {
            var rpm = 1750f + (i + 1) * 0.25f;
            var w = SyntheticWindows.Make(i, rpm, i < 10 ? 1 : 1 + 0.1 * (i - 10), 1, rng);
            var t0 = Stopwatch.GetTimestamp();
            var resp = await http.PostAsync("/api/payload", new ByteArrayContent(PayloadCodec.Encode(w)));
            Assert.Equal(HttpStatusCode.Accepted, resp.StatusCode);
            double? got = null;
            var deadline = Stopwatch.GetTimestamp() + Stopwatch.Frequency;
            while (got is null && Stopwatch.GetTimestamp() < deadline)
            {
                foreach (var (ticks, s) in seen) if (Math.Abs(s.Rpm - rpm) < 1e-3 && ticks >= t0) { got = (ticks - t0) * 1000.0 / Stopwatch.Frequency; break; }
                if (got is null) await Task.Delay(1);
            }
            Assert.NotNull(got);
            lat.Add(got!.Value);
            await Task.Delay(30);
        }
        lat.Sort();
        _out.WriteLine($"MEASURED POST->client snapshot latency (in-process, long-polling) n={lat.Count}: p50={lat[lat.Count / 2]:F1} ms p95={lat[(int)(lat.Count * 0.95)]:F1} ms max={lat[^1]:F1} ms (IS1 budget 500 ms total)");
        Assert.True(lat[(int)(lat.Count * 0.95)] < 500);

        var m = await http.GetFromJsonAsync<JsonElement>("/api/metrics");
        foreach (var st in m.GetProperty("stages").EnumerateArray())
            _out.WriteLine($"MEASURED stage {st.GetProperty("stage").GetString()}: p50={st.GetProperty("p50Ms").GetDouble()} p95={st.GetProperty("p95Ms").GetDouble()} max={st.GetProperty("maxMs").GetDouble()} ms (n={st.GetProperty("count").GetInt64()})");

        var state = await http.GetFromJsonAsync<JsonElement>("/api/state");
        Assert.True(state.GetProperty("engine").GetProperty("isSimulated").GetBoolean());
        Assert.Equal(2, state.GetProperty("bearings").GetArrayLength());
        var last = seen.Last().snap;
        Assert.Equal("1.0", last.SchemaVersion);
        Assert.NotEqual("no_data", last.FaultClass);
        await hub.DisposeAsync();
    }
}

/// <summary>Locates AI-engine/models/golden_vectors.json by walking up from the test binary. Fails (not skips) when absent.</summary>
public static class Golden
{
    public static string Dir()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            var p = Path.Combine(d.FullName, "AI-engine", "models");
            if (File.Exists(Path.Combine(p, "golden_vectors.json"))) return p;
        }
        throw new FileNotFoundException("AI-engine/models/golden_vectors.json not found above " + AppContext.BaseDirectory);
    }
}

public class GoldenParityTests
{
    private readonly ITestOutputHelper _out;
    public GoldenParityTests(ITestOutputHelper o) => _out = o;

    private static double[] Arr(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    /// <summary>
    /// Feeds golden_vectors.json inputs (same 16 values to both bearings, as engine.py::golden_vectors does) through the C#
    /// BearingTracker + OnnxInferenceEngine and compares engineered row, TFT tensors, class probabilities, HI, forecast,
    /// RUL, stage and class with the Python engine outputs. Contract tolerance: 1e-4 abs (probs/HI), 1e-3 rel (RUL).
    /// </summary>
    [Fact]
    public void CSharpPipeline_MatchesPythonGoldenVectors()
    {
        var dir = Golden.Dir();
        var contract = ModelContract.Load(Path.Combine(dir, "model_contract.json"));
        using var engine = new OnnxInferenceEngine(contract, Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance);
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "golden_vectors.json")));
        var root = doc.RootElement;
        var cases = root.GetProperty("cases").EnumerateArray().ToDictionary(c => c.GetProperty("window_index").GetInt32());
        var tr = engine.CreateTracker(1, 0);
        double t0 = -1, maxEng = 0, maxEnc = 0, maxProb = 0, maxHi = 0, maxFc = 0, maxRulRel = 0;
        var checkedN = 0;
        foreach (var inp in root.GetProperty("inputs").EnumerateArray())
        {
            var f16 = Arr(inp.GetProperty("features16")).Select(v => (float)v).ToArray();
            var full = f16.Concat(f16).ToArray();
            var ms = inp.GetProperty("t20_ms").GetUInt32();
            if (t0 < 0) t0 = ms;
            tr.Update(full, inp.GetProperty("rpm").GetDouble(), (ms - t0) / 3.6e6);
            var wi = inp.GetProperty("window_index").GetInt32();
            if (!cases.TryGetValue(wi, out var cs)) { if (tr.State == BearingStates.Ready) engine.Infer(tr); continue; }
            Assert.Equal(BearingStates.Ready, tr.State);
            var r = engine.Infer(tr);
            var o = cs.GetProperty("output");

            maxEng = Math.Max(maxEng, Arr(cs.GetProperty("engineered")).Zip(tr.Latest!).Max(p => Math.Abs(p.First - p.Second)));
            var enc = cs.GetProperty("tft_encoder_cont").EnumerateArray().SelectMany(row => Arr(row)).ToArray();
            var dec = cs.GetProperty("tft_decoder_cont").EnumerateArray().SelectMany(row => Arr(row)).ToArray();
            maxEnc = Math.Max(maxEnc, enc.Zip(r.Details!.TftEncoder!).Max(p => Math.Abs(p.First - p.Second)));
            maxEnc = Math.Max(maxEnc, dec.Zip(r.Details.TftDecoder!).Max(p => Math.Abs(p.First - p.Second)));

            Assert.Equal(o.GetProperty("fault_class").GetString(), r.FaultClass);
            foreach (var p in o.GetProperty("class_probs").EnumerateObject())
                maxProb = Math.Max(maxProb, Math.Abs(p.Value.GetDouble() - r.ClassProbs![p.Name]));
            maxHi = Math.Max(maxHi, Math.Abs(o.GetProperty("health_index").GetDouble() - r.HealthIndex!.Value));
            maxFc = Math.Max(maxFc, Arr(o.GetProperty("hi_forecast")).Zip(r.Details.HiForecast).Max(p => Math.Abs(p.First - p.Second)));
            var rulPy = o.GetProperty("rul_hours").GetDouble();
            maxRulRel = Math.Max(maxRulRel, Math.Abs(rulPy - r.RulHours!.Value) / Math.Max(Math.Abs(rulPy), 1e-9));
            Assert.Equal(o.GetProperty("health_stage").GetInt32(), r.Details.HealthStage);
            Assert.Equal(o.GetProperty("rul_threshold_class").GetString(), r.Details.RulThresholdClass);
            Assert.Equal(o.GetProperty("onset_detected").GetBoolean(), r.Details.OnsetDetected);
            if (checkedN == 0)
            {
                var pyH = o.GetProperty("class_probs").GetProperty("healthy").GetDouble(); var csH = r.ClassProbs!["healthy"];
                _out.WriteLine($"sample window {wi}: py healthy={pyH:R} cs healthy={csH:R}; py rul={rulPy:R} cs rul={r.RulHours:R}");
            }
            checkedN++;
        }
        _out.WriteLine($"PARITY cases={checkedN}: max|engineered|={maxEng:E2} max|tft tensors|={maxEnc:E2} max|prob|={maxProb:E2} max|HI|={maxHi:E2} max|forecast|={maxFc:E2} max rel RUL={maxRulRel:E2}");
        Assert.Equal(cases.Count, checkedN);
        Assert.True(maxEng < 1e-6, $"engineered {maxEng}");
        Assert.True(maxEnc < 1e-4, $"tft tensors {maxEnc}");
        Assert.True(maxProb < 1e-4, $"probs {maxProb}");
        Assert.True(maxHi < 1e-4, $"hi {maxHi}");
        Assert.True(maxFc < 1e-4, $"forecast {maxFc}");
        Assert.True(maxRulRel < 1e-3, $"rul rel {maxRulRel}");
    }
}

public class FeatureEngineerTests
{
    [Fact]
    public void Healthy_Window_Has_HealthIndex_Zero_And_Growth_Raises_It()
    {
        var c = FeatureConstants.StubDefaults;
        var fe = new FeatureEngineer(c);
        var rng = new Random(5);
        var baseWins = Enumerable.Range(0, 24).Select(i => SyntheticWindows.Make(i, 1750f, 1, 1, rng).Features[..16]).ToList();
        var b = fe.ComputeBaseline(baseWins);
        var healthy = fe.Compute(SyntheticWindows.Make(0, 1750f, 1, 1, rng).Features[..16], 1750, b);
        var worn = fe.Compute(SyntheticWindows.Make(0, 1750f, 6, 1, rng).Features[..16], 1750, b);
        Assert.True(healthy[c.Index("hi")] < 0.1);
        Assert.True(worn[c.Index("hi")] > healthy[c.Index("hi")] + 0.3);
        Assert.True(worn[c.Index("bpfo_share")] > 0.5);
    }
}

/// <summary>C5: the remote-address guard returns 403 for any non-private address.</summary>
public class LanOnlyTests
{
    private static async Task<int> Status(string ip)
    {
        var ctx = new Microsoft.AspNetCore.Http.DefaultHttpContext();
        ctx.Connection.RemoteIpAddress = IPAddress.Parse(ip);
        ctx.Response.Body = new MemoryStream();
        var guard = new DigitalTwin.Backend.Services.PrivateNetworkGuard(c => Task.CompletedTask,
            Microsoft.Extensions.Logging.Abstractions.NullLogger<DigitalTwin.Backend.Services.PrivateNetworkGuard>.Instance);
        await guard.Invoke(ctx);
        return ctx.Response.StatusCode;
    }

    [Theory]
    [InlineData("8.8.8.8")]
    [InlineData("203.0.113.9")]
    [InlineData("172.32.0.1")]
    [InlineData("2001:4860:4860::8888")]
    public async Task NonPrivateRemote_Gets403(string ip) => Assert.Equal(403, await Status(ip));

    [Theory]
    [InlineData("127.0.0.1")]
    [InlineData("10.1.2.3")]
    [InlineData("172.20.0.5")]
    [InlineData("192.168.1.50")]
    [InlineData("fe80::1")]
    public async Task PrivateRemote_Passes(string ip) => Assert.Equal(200, await Status(ip));
}

public class RealEngineIntegrationTests : IClassFixture<RealEngineFactory>
{
    private readonly RealEngineFactory _f; private readonly ITestOutputHelper _out;
    public RealEngineIntegrationTests(RealEngineFactory f, ITestOutputHelper o) { _f = f; _out = o; }

    private (DigitalTwin.Backend.Services.Pipeline Pipe, DigitalTwin.Backend.Services.StateStore Store, DigitalTwin.Backend.Services.SnapshotFactory Snap) Get()
    {
        _ = _f.Server;
        var sp = _f.Services;
        return (sp.GetRequiredService<DigitalTwin.Backend.Services.Pipeline>(), sp.GetRequiredService<DigitalTwin.Backend.Services.StateStore>(), sp.GetRequiredService<DigitalTwin.Backend.Services.SnapshotFactory>());
    }

    [Fact]
    public async Task DemoReplay_Bearing1_ReachesFault_Bearing2_StaysHealthy_RealOnnx()
    {
        var (pipe, store, snap) = Get();
        pipe.Reset();
        Assert.False(pipe.Engine.IsSimulated);
        var recs = ReplaySource.Load(DigitalTwin.Backend.Config.Paths.Resolve("AI-engine/models/demo_payloads.bin"));
        Assert.Equal(339, recs.Count);
        int b1Fault = 0, b2Fault = 0, ready = 0; var lat = new List<double>();
        foreach (var r in recs)
        {
            var t0 = Stopwatch.GetTimestamp();
            await pipe.IngestAsync(PayloadCodec.Encode(r), "test");
            lat.Add((Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency);
            var s = snap.Build(store.Current, DateTimeOffset.UtcNow);
            if (store.Current.Bearings.All(b => b.State == BearingStates.Ready)) { ready++; if (s.Bearing1Fault != null) b1Fault++; if (s.Bearing2Fault != null) b2Fault++; }
            Assert.Null(s.PhysicsRulHours);   // public data has no displacement probes: physics stays null, not fabricated
        }
        _out.WriteLine($"ready windows {ready}: bearing1 fault alerts {b1Fault}, bearing2 fault alerts {b2Fault}");
        var last = snap.Build(store.Current, DateTimeOffset.UtcNow);
        Assert.NotNull(last.Bearing1Fault);                    // run-to-failure bearing flagged by the end
        Assert.True(b1Fault > ready * 0.4, "bearing 1 should be flagged for a large part of the degraded run");
        Assert.True(b2Fault < ready * 0.1, "healthy stream bearing 2 should rarely alarm");
        lat.Sort();
        _out.WriteLine($"MEASURED in-process ingest->snapshot (real ONNX) p50={lat[lat.Count / 2]:F1} ms p95={lat[(int)(lat.Count * 0.95)]:F1} ms");
        Assert.True(lat[(int)(lat.Count * 0.95)] < 500);
    }

    [Fact]
    public async Task TwinSim_PopulatesPhysicsRulAndResidual_AsSimulation()
    {
        var (pipe, store, snap) = Get();
        pipe.Reset();
        var recs = SyntheticWindows.DegradingRun(240, hoursPerWindow: 0.05);
        DigitalTwin.Backend.Models.DashboardSnapshot? last = null; var withPhys = 0;
        foreach (var r in recs)
        {
            await pipe.IngestAsync(PayloadCodec.Encode(r), "test");
            last = snap.Build(store.Current, DateTimeOffset.UtcNow);
            if (last.PhysicsRulHours is not null) withPhys++;
        }
        _out.WriteLine($"[SIMULATION] windows with physics RUL: {withPhys}; last ai={last!.AiRulHours:F2} h physics={last.PhysicsRulHours:F2} h residual={last.RulResidualPercent:F1} %");
        Assert.True(withPhys > 60);
        Assert.NotNull(last.RulResidualPercent);
    }
}

public class RealEngineFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder b)
    {
        b.UseSetting("Backend:Udp:Enabled", "false");
        b.UseSetting("Backend:Replay:Enabled", "false");
        b.UseSetting("Backend:ModelContractPath", "AI-engine/models/model_contract.json");
        b.UseSetting("Backend:DashboardPath", "");
        b.UseSetting("Urls", "http://127.0.0.1:0");
    }
}
