using System.Text;
using DigitalTwin.Backend.Config;
using DigitalTwin.Backend.Ingest;
using DigitalTwin.Backend.Inference;
using DigitalTwin.Backend.Services;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);
builder.Services.Configure<BackendOptions>(builder.Configuration.GetSection(BackendOptions.Section));

builder.Services.AddSingleton<IInferenceEngine>(sp =>
{
    var opt = sp.GetRequiredService<IOptions<BackendOptions>>().Value;
    var log = sp.GetRequiredService<ILoggerFactory>().CreateLogger("Engine");
    var path = Paths.Resolve(opt.ModelContractPath);
    try
    {
        if (File.Exists(path))
        {
            var contract = ModelContract.Load(path);
            var engine = new OnnxInferenceEngine(contract, log);
            log.LogInformation("Loaded ONNX engine from {Path} (contract v{V}); provenance: {P}", path, contract.ContractVersion, contract.Provenance);
            return engine;
        }
        log.LogWarning("model_contract.json not found at {Path}", path);
    }
    catch (Exception ex)
    {
        log.LogError(ex, "Failed to load model contract/ONNX models from {Path}", path);
    }
    log.LogWarning("*** USING SIMULATED STUB INFERENCE ENGINE (IsSimulated=true): outputs are NOT model predictions ***");
    return new StubInferenceEngine(opt.HoursPerWindow);
});
builder.Services.AddSingleton<IPhysicsTwin>(sp =>
{
    var opt = sp.GetRequiredService<IOptions<BackendOptions>>().Value;
    var log = sp.GetRequiredService<ILoggerFactory>().CreateLogger("Twin");
    if (string.Equals(opt.PhysicsTwin, "FeBeam", StringComparison.OrdinalIgnoreCase))
    {
        log.LogWarning("Physics twin: FE-beam (INITIAL, assumed rotor geometry, [SIMULATION]-verified only)");
        return new FeBeamPhysicsTwinAdapter(opt.HoursPerWindow);
    }
    log.LogInformation("Physics twin disabled (Backend:PhysicsTwin={V}); PhysicsRulHours = null", opt.PhysicsTwin);
    return new NullPhysicsTwin();
});
builder.Services.AddSingleton<DemoControl>();
builder.Services.AddSingleton<StateStore>();
builder.Services.AddSingleton<SnapshotFactory>();
builder.Services.AddSingleton<SnapshotPublisher>();
builder.Services.AddSingleton<LatencyRecorder>();
builder.Services.AddSingleton<Pipeline>();
builder.Services.AddHostedService<BroadcastTimer>();
builder.Services.AddHostedService<UdpIngestService>();
builder.Services.AddHostedService<ReplaySource>();
builder.Services.AddSignalR();
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p
    .SetIsOriginAllowed(PrivateNetwork.IsPrivateOrigin).AllowAnyHeader().AllowAnyMethod().AllowCredentials()));

var app = builder.Build();
app.UseMiddleware<PrivateNetworkGuard>();
app.UseCors();

// Serve the published Blazor WASM dashboard from this process (one LAN URL).
var dashDir = app.Services.GetRequiredService<IOptions<BackendOptions>>().Value.DashboardPath;
if (!string.IsNullOrWhiteSpace(dashDir)) dashDir = Paths.Resolve(dashDir);
var serveDash = !string.IsNullOrWhiteSpace(dashDir) && Directory.Exists(dashDir);
if (serveDash)
{
    var provider = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
    provider.Mappings[".dat"] = "application/octet-stream";
    provider.Mappings[".blat"] = "application/octet-stream";
    provider.Mappings[".glb"] = "model/gltf-binary";   // 3D pump viewer asset
    provider.Mappings[".gltf"] = "model/gltf+json";
    var fp = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(Path.GetFullPath(dashDir));
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = fp });
    app.UseStaticFiles(new StaticFileOptions { FileProvider = fp, ContentTypeProvider = provider });
    app.Logger.LogInformation("Serving dashboard from {Dir}", Path.GetFullPath(dashDir));
}

app.MapHub<DashboardHub>("/hubs/dashboard");

app.MapPost("/api/payload", async (HttpRequest req, Pipeline pipe) =>
{
    using var ms = new MemoryStream();
    var buf = new byte[512]; int n, total = 0;
    while ((n = await req.Body.ReadAsync(buf)) > 0)
    {
        total += n; if (total > 4096) return Results.BadRequest(new { error = "body too large" });
        ms.Write(buf, 0, n);
    }
    var (ok, err) = await pipe.IngestAsync(ms.ToArray(), "http:" + req.HttpContext.Connection.RemoteIpAddress);
    return ok ? Results.Accepted() : Results.BadRequest(new { error = err });
}).Accepts<byte[]>("application/octet-stream");

app.MapGet("/api/state", (StateStore s, SnapshotFactory f, IInferenceEngine e) =>
{
    var st = s.Current;
    return Results.Ok(new
    {
        engine = new { e.Name, e.IsSimulated, e.Provenance },
        payloadCount = st.PayloadCount,
        lastPayloadUtc = st.LastPayloadUtc,
        rpm = st.Window?.Rpm,
        bearings = st.Bearings,
        snapshot = f.Build(st, DateTimeOffset.UtcNow),
    });
});

app.MapGet("/api/metrics", (LatencyRecorder l, IInferenceEngine e) =>
    Results.Ok(new { isSimulated = e.IsSimulated, rejects = l.Rejects, stages = l.Snapshot() }));

app.MapGet("/api/demo", (DemoControl c) => Results.Ok(new { source = c.Source, machine = c.MachineLabel }));
app.MapPost("/api/demo/source", (string name, DemoControl c) =>
    c.TrySet(name) ? Results.Ok(new { source = c.Source }) : Results.BadRequest(new { error = "name must be demo | twin-sim | synthetic" }));

app.MapGet("/api/health", (IInferenceEngine e) => Results.Ok(new { status = "ok", isSimulated = e.IsSimulated }));

if (serveDash)
    app.MapFallbackToFile("index.html", new StaticFileOptions { FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(Path.GetFullPath(dashDir)) });

app.Logger.LogInformation("C5: LAN-only. Make sure Urls binds to a LAN address and no outbound calls are configured.");
app.Run();

public partial class Program { }
