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
        b.UseSetting("Backend:BaselineWindowsOverride", "3");
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

/// <summary>Skips with a clear message until the engine exports golden_vectors.json.</summary>
public sealed class GoldenFactAttribute : FactAttribute
{
    public static string? Find()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            var p = Path.Combine(d.FullName, "AI-engine", "models", "golden_vectors.json");
            if (File.Exists(p)) return p;
        }
        return null;
    }
    public GoldenFactAttribute()
    {
        if (Find() is null)
            Skip = "golden_vectors.json not found under AI-engine/models/: pending export from the AI engine. Parity test will run once it exists.";
    }
}

public class GoldenParityTests
{
    /// <summary>
    /// ASSUMED schema (to be matched to the engine's export): {"vectors":[{"bearing":1,"raw_windows":[[16 floats]...],"rpm":[..],
    /// "expected":{"fault_class":"..","health_index":x,"rul_hours":y}}]}. Feeds windows through FeatureEngineer + OnnxInferenceEngine
    /// using model_contract.json from the same folder.
    /// </summary>
    [GoldenFact]
    public void CSharpPipeline_MatchesPythonGoldenVectors()
    {
        var gv = GoldenFactAttribute.Find()!;
        var contract = ModelContract.Load(Path.Combine(Path.GetDirectoryName(gv)!, "model_contract.json"));
        using var engine = new OnnxInferenceEngine(contract, 1.0, Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance);
        var fe = new FeatureEngineer(contract.Features);
        using var doc = JsonDocument.Parse(File.ReadAllText(gv));
        foreach (var v in doc.RootElement.GetProperty("vectors").EnumerateArray())
        {
            var bearing = v.GetProperty("bearing").GetInt32();
            var wins = v.GetProperty("raw_windows").EnumerateArray().Select(a => a.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray()).ToList();
            var rpms = v.GetProperty("rpm").EnumerateArray().Select(x => x.GetDouble()).ToList();
            var tr = new BearingTracker(1, fe, contract.Features.BaselineWindows, contract.RequiredHistory);
            for (var i = 0; i < wins.Count; i++)
            {
                var full = new float[32]; wins[i].CopyTo(full, 0);
                tr.Update(full, rpms[i]);
            }
            Assert.Equal(BearingStates.Ready, tr.State);
            var r = engine.Infer(tr);
            var exp = v.GetProperty("expected");
            Assert.Equal(exp.GetProperty("fault_class").GetString(), r.FaultClass);
            Assert.InRange(r.HealthIndex!.Value, exp.GetProperty("health_index").GetDouble() - 1e-3, exp.GetProperty("health_index").GetDouble() + 1e-3);
        }
    }
}

public class FeatureEngineerTests
{
    [Fact]
    public void Healthy_Window_Has_HealthIndex_Zero_And_Growth_Raises_It()
    {
        var fe = new FeatureEngineer(FeatureConstants.StubDefaults);
        var rng = new Random(5);
        var baseWins = Enumerable.Range(0, 24).Select(i => SyntheticWindows.Make(i, 1750f, 1, 1, rng).Features[..16]).ToList();
        var b = fe.ComputeBaseline(baseWins);
        var healthy = fe.Compute(SyntheticWindows.Make(0, 1750f, 1, 1, rng).Features[..16], 1750, b);
        var worn = fe.Compute(SyntheticWindows.Make(0, 1750f, 6, 1, rng).Features[..16], 1750, b);
        Assert.True(healthy["hi"] < 0.1);
        Assert.True(worn["hi"] > healthy["hi"] + 0.3);
        Assert.True(worn["bpfo_share"] > 0.5);
    }
}
