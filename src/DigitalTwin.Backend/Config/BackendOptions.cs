namespace DigitalTwin.Backend.Config;

public sealed class BackendOptions
{
    public const string Section = "Backend";
    public string MachineId { get; set; } = "OH2-PUMP-01";
    /// <summary>Path to model_contract.json (the ONLY source of model metadata).</summary>
    public string ModelContractPath { get; set; } = "AI-engine/models/model_contract.json";
    /// <summary>Re-broadcast period; 100 ms = 10 Hz (spec S9).</summary>
    public int BroadcastIntervalMs { get; set; } = 100;
    /// <summary>IsConnected turns false if no payload arrived within this many seconds.</summary>
    public double ConnectionTimeoutSeconds { get; set; } = 5;
    /// <summary>Overrides contract baseline_windows (demo/test only). 0 = use contract.</summary>
    public int BaselineWindowsOverride { get; set; }
    /// <summary>Operating hours represented by one window step, used to convert window counts to RUL hours.</summary>
    public double HoursPerWindow { get; set; } = 0.05;
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
    /// <summary>.bin (concatenated 152-byte records) or .csv (rpm + 32 features + 4 disp). Empty + Synthetic=true generates a SYNTHETIC degrading run.</summary>
    public string Path { get; set; } = "";
    public bool Synthetic { get; set; }
    public double RateHz { get; set; } = 1;
    public bool Loop { get; set; } = true;
}
