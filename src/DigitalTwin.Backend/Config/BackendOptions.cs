namespace DigitalTwin.Backend.Config;

public sealed class BackendOptions
{
    public const string Section = "Backend";
    public string MachineId { get; set; } = "OH2-PUMP-01";
    /// <summary>Path to model_contract.json (the ONLY source of model metadata).</summary>
    public string ModelContractPath { get; set; } = "AI-engine/models/model_contract.json";
    /// <summary>Re-broadcast period; 100 ms = 10 Hz (spec S9).</summary>
    // 50 ms (20 Hz nominal): Windows timer granularity (~15.6 ms) stretches a 100 ms period to
    // ~9.9 Hz measured, which would fail S9 (>=10 Hz). 20 Hz gives 2x margin.
    public int BroadcastIntervalMs { get; set; } = 50;
    /// <summary>IsConnected turns false if no payload arrived within this many seconds.</summary>
    public double ConnectionTimeoutSeconds { get; set; } = 5;
    /// <summary>Overrides contract baseline_windows (demo/test only). 0 = use contract.</summary>
    public int BaselineWindowsOverride { get; set; }
    /// <summary>Operating hours represented by one window step, used to convert window counts to RUL hours.</summary>
    public double HoursPerWindow { get; set; } = 0.05;
    /// <summary>"FeBeam" = FE-beam physics twin (DigitalTwin.Twin); "None" = no physics estimate (PhysicsRulHours null). Fallback while the twin is not commissioned on the real rig.</summary>
    public string PhysicsTwin { get; set; } = "FeBeam";
    /// <summary>Folder holding the published Blazor WASM dashboard (wwwroot). Served from the backend if it exists (one process, one LAN URL).</summary>
    public string DashboardPath { get; set; } = "";
    public UdpOptions Udp { get; set; } = new();
    public ReplayOptions Replay { get; set; } = new();
}

public sealed class UdpOptions
{
    public bool Enabled { get; set; } = true;
    public int Port { get; set; } = 5005;
    public string BindAddress { get; set; } = "0.0.0.0";
}

public sealed class ReplayOptions
{
    public bool Enabled { get; set; }
    /// <summary>RECORDED PAYLOAD FILE. .bin (concatenated 152-byte records) or .csv (rpm + 32 features + 4 disp).
    /// AI-engine will produce AI-engine/models/demo_payloads.bin; set Backend:Replay:Path to it. Empty + Synthetic=true generates a SYNTHETIC degrading run.</summary>
    public string Path { get; set; } = "";
    public bool Synthetic { get; set; }
    public double RateHz { get; set; } = 1;
    public bool Loop { get; set; } = true;
}
