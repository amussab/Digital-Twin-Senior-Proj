using System.Text.Json;

namespace DigitalTwin.Backend.Inference;

public sealed record SubModelSpec(
    string OnnxFile, int EncoderLength, int DecoderLength,
    IReadOnlyList<string> EncoderContColumns, IReadOnlyList<string> EncoderCatColumns,
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> CategoricalEncodings);

/// <summary>Constants for feature engineering. Read from model_contract.json; never hardcoded for the real engine.</summary>
public sealed record FeatureConstants(
    int BaselineWindows, double HiRRef, double HiCurveK,
    double WEnv, double WBp, double WFamily,
    IReadOnlyList<string> ModelInputOrder,
    IReadOnlyList<string> PerChannelFeatureSuffixes,
    double RpmCenter, double RpmHalfRange)
{
    /// <summary>
    /// SIMULATION-ONLY defaults mirroring AI-test/features.py as of 2026-10-04 so the stub can run without
    /// a contract. The real engine refuses to start without the contract.
    /// </summary>
    public static FeatureConstants StubDefaults { get; } = new(
        24, 20.0, 1.0, 0.45, 0.30, 0.25,
        new[] { "hi", "log_env_ratio", "log_bp_ratio", "kurtosis_max", "crest_max", "kurtosis_rise",
                "plane_asymmetry", "ftf_share", "bsf_share", "bpfo_share", "bpfi_share", "family_contrast", "rpm_norm" },
        new[] { "bp_rms", "bp_kurtosis", "bp_crest", "env_rms", "ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag" },
        2675.0, 925.0);
}

/// <summary>Typed view over model_contract.json (shape of AI-test/export_onnx.py contract_version 1).</summary>
public sealed class ModelContract
{
    public required string Directory { get; init; }
    public int ContractVersion { get; init; }
    public string Provenance { get; init; } = "";
    public required FeatureConstants Features { get; init; }
    public required IReadOnlyList<string> ClassLabels { get; init; }
    public required SubModelSpec Nhits { get; init; }
    public required SubModelSpec Tft { get; init; }
    public JsonElement RulCalibration { get; init; }
    public int TrendHistoryWindows { get; init; } = 24;
    public double TrendFloorHi { get; init; } = 0.08;
    public int RequiredHistory => Math.Max(Nhits.EncoderLength, Tft.EncoderLength);

    public static ModelContract Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        var fe = r.GetProperty("feature_engineering");
        var hi = fe.GetProperty("health_index");
        var w = hi.GetProperty("weights");
        var modes = r.GetProperty("operating_modes_rpm").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        var raw = r.GetProperty("raw_feature_order").EnumerateArray().Select(e => e.GetString()!).ToList();
        // per-channel suffixes: names of the first accelerometer, prefix "a1_" stripped
        var perChannel = raw.Take(8).Select(n => n[(n.IndexOf('_') + 1)..]).ToList();

        var feats = new FeatureConstants(
            fe.GetProperty("baseline_windows").GetInt32(),
            hi.GetProperty("r_ref").GetDouble(), hi.GetProperty("curve_k").GetDouble(),
            w.GetProperty("env_rms").GetDouble(), w.GetProperty("bp_rms").GetDouble(), w.GetProperty("family_max").GetDouble(),
            fe.GetProperty("model_input_order").EnumerateArray().Select(e => e.GetString()!).ToList(),
            perChannel, (modes.Min() + modes.Max()) / 2, (modes.Max() - modes.Min()) / 2);

        var proc = r.TryGetProperty("rul_procedure", out var p) ? p : default;
        return new ModelContract
        {
            Directory = Path.GetDirectoryName(Path.GetFullPath(path))!,
            ContractVersion = r.GetProperty("contract_version").GetInt32(),
            Provenance = r.TryGetProperty("data_provenance", out var dp) ? dp.GetString() ?? "" : "",
            Features = feats,
            ClassLabels = r.GetProperty("classes").GetProperty("labels").EnumerateArray().Select(e => e.GetString()!).ToList(),
            Nhits = ParseModel(r.GetProperty("models").GetProperty("nhits")),
            Tft = ParseModel(r.GetProperty("models").GetProperty("tft")),
            RulCalibration = r.TryGetProperty("rul_calibration", out var rc) ? rc.Clone() : default,
            TrendHistoryWindows = proc.ValueKind == JsonValueKind.Object && proc.TryGetProperty("trend_history_windows", out var t) ? t.GetInt32() : 24,
            TrendFloorHi = proc.ValueKind == JsonValueKind.Object && proc.TryGetProperty("trend_floor_hi", out var f) ? f.GetDouble() : 0.08,
        };
    }

    private static SubModelSpec ParseModel(JsonElement m)
    {
        static List<string> Strs(JsonElement e, string name) =>
            e.TryGetProperty(name, out var a) ? a.EnumerateArray().Select(x => x.GetString()!).ToList() : new();
        var enc = new Dictionary<string, IReadOnlyDictionary<string, int>>();
        if (m.TryGetProperty("categorical_encodings", out var ce))
            foreach (var c in ce.EnumerateObject())
                enc[c.Name] = c.Value.EnumerateObject().ToDictionary(k => k.Name, k => k.Value.GetInt32());
        return new SubModelSpec(m.GetProperty("onnx").GetString()!, m.GetProperty("encoder_length").GetInt32(),
            m.GetProperty("decoder_length").GetInt32(), Strs(m, "encoder_cont_columns"), Strs(m, "encoder_cat_columns"), enc);
    }
}
