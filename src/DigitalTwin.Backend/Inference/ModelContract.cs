using System.Text.Json;

namespace DigitalTwin.Backend.Inference;

public sealed record Scaler(double Center, double Scale);

/// <summary>One ONNX sub-model: file, encoder/decoder length, ordered input columns and their scalers (all from the contract).</summary>
public sealed record SubModelSpec(
    string OnnxFile, int EncoderLength, int DecoderLength,
    IReadOnlyList<string> Columns, IReadOnlyList<Scaler> Scalers);

/// <summary>Per-bearing feature-engineering constants (contract feature_engineering / onset). Never hardcoded for the real engine.</summary>
public sealed record FeatureConstants(
    int BaselineWindows, double Eps, double RelFloor,
    double HiRRef, double HiCurveK, double WEnv, double WBp, double WFamily, double HiClipMax,
    double RpmCenter, double RpmScale,
    double OnsetSigmas, int OnsetConsecutive, double OnsetSigmaFloor, int OnsetRunInSkip,
    IReadOnlyList<string> EngineeredOrder)
{
    public static readonly string[] ExpectedOrder =
    {
        "hi", "log_env_ratio", "log_bp_ratio", "kurtosis_max", "crest_max", "kurtosis_rise", "axis_asymmetry",
        "ftf_share", "bsf_share", "bpfo_share", "bpfi_share", "family_contrast",
        "log_ftf_ratio", "log_bsf_ratio", "log_bpfo_ratio", "log_bpfi_ratio",
        "raw_ftf_share", "raw_bsf_share", "raw_bpfo_share", "raw_bpfi_share", "rpm_norm",
    };

    public int Index(string name)
    {
        for (var i = 0; i < EngineeredOrder.Count; i++) if (EngineeredOrder[i] == name) return i;
        throw new KeyNotFoundException(name);
    }

    /// <summary>SIMULATION-ONLY defaults mirroring contract v2 so the stub can run without a contract.</summary>
    public static FeatureConstants StubDefaults { get; } = new(
        24, 1e-9, 1e-6, 20.0, 1.0, 0.45, 0.30, 0.25, 1.5, 2675.0, 925.0, 3.0, 12, 0.01, 6, ExpectedOrder);
}

public sealed record RulCalibration(
    string Method, int HistoryWindows, double BlendW, double UMin,
    IReadOnlyDictionary<string, double> ClassThresholds, double PooledThreshold, double MaxRulFactor,
    IReadOnlyList<double> LoglinCoef, double IntervalRel, int ClassSmoothingWindows)
{
    public double ThresholdFor(string? cls) =>
        cls is null or "healthy" ? PooledThreshold : ClassThresholds.TryGetValue(cls, out var v) ? v : PooledThreshold;
}

/// <summary>Typed view over model_contract.json (contract_version 2, per-bearing hybrid engine).</summary>
public sealed class ModelContract
{
    public required string Directory { get; init; }
    public int ContractVersion { get; init; }
    public string Provenance { get; init; } = "";
    public required FeatureConstants Features { get; init; }
    public required IReadOnlyList<string> ClassLabels { get; init; }
    /// <summary>contract classes.decision.type: "rtf_onset_gate" | "hierarchical_onset_gate" | "" (plain softmax).</summary>
    public string DecisionType { get; init; } = "";
    public required SubModelSpec Nhits { get; init; }
    public required SubModelSpec Tft { get; init; }
    public int NhitsPredictionLength { get; init; }
    public required RulCalibration Rul { get; init; }
    /// <summary>Engineered rows each bearing must hold before the TFT can run (encoder + 1 decoder row).</summary>
    public int RequiredHistory => Tft.EncoderLength + 1;
    /// <summary>Rows the tracker retains (covers the RUL history window and the N-HiTS encoder).</summary>
    public int HistoryCap => Math.Max(Math.Max(RequiredHistory, Nhits.EncoderLength), Rul.HistoryWindows);

    public static ModelContract Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        var fe = r.GetProperty("feature_engineering");
        var hi = fe.GetProperty("health_index"); var w = hi.GetProperty("weights");
        var rpm = fe.GetProperty("rpm_norm"); var on = r.GetProperty("onset");
        var order = fe.GetProperty("engineered_order").EnumerateArray().Select(e => e.GetString()!).ToList();
        if (!order.SequenceEqual(FeatureConstants.ExpectedOrder))
            throw new InvalidDataException("contract engineered_order differs from the order FeatureEngineer implements; the C# port must be updated.");
        var bl = fe.GetProperty("baseline");
        var feats = new FeatureConstants(
            bl.GetProperty("windows").GetInt32(), fe.GetProperty("eps").GetDouble(), bl.GetProperty("rel_floor").GetDouble(),
            hi.GetProperty("r_ref").GetDouble(), hi.GetProperty("curve_k").GetDouble(),
            w.GetProperty("env_rms").GetDouble(), w.GetProperty("bp_rms").GetDouble(), w.GetProperty("family_max").GetDouble(),
            hi.GetProperty("clip_max").GetDouble(), rpm.GetProperty("center").GetDouble(), rpm.GetProperty("scale").GetDouble(),
            on.GetProperty("sigmas").GetDouble(), on.GetProperty("consecutive").GetInt32(), on.GetProperty("sigma_floor").GetDouble(),
            on.GetProperty("run_in_skip_windows").GetInt32(), order);

        var ru = r.GetProperty("rul");
        var cal = new RulCalibration(
            ru.GetProperty("method").GetString()!, ru.GetProperty("history_windows").GetInt32(), ru.GetProperty("blend_w").GetDouble(),
            ru.GetProperty("u_min").GetDouble(),
            ru.GetProperty("class_thresholds").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetDouble()),
            ru.GetProperty("pooled_threshold").GetDouble(), ru.GetProperty("max_rul_factor").GetDouble(),
            ru.GetProperty("loglin_coef").EnumerateArray().Select(e => e.GetDouble()).ToList(),
            ru.TryGetProperty("interval_rel", out var ir) ? ir.GetDouble() : 0.5,
            ru.TryGetProperty("class_smoothing_windows", out var cs) ? cs.GetInt32() : 5);

        var prov = r.GetProperty("provenance");
        var synthetic = prov.TryGetProperty("synthetic", out var s) && s.GetBoolean();
        var provText = $"{(synthetic ? "SYNTHETIC" : "MEASURED public data")}: datasets {string.Join("+", prov.GetProperty("datasets").EnumerateArray().Select(e => e.GetString()))}, run {prov.GetProperty("run").GetString()}, commit {prov.GetProperty("git_commit").GetString()}";
        var models = r.GetProperty("models");
        return new ModelContract
        {
            Directory = Path.GetDirectoryName(Path.GetFullPath(path))!,
            ContractVersion = r.GetProperty("contract_version").GetInt32(),
            Provenance = provText,
            Features = feats,
            ClassLabels = r.GetProperty("classes").GetProperty("labels").EnumerateArray().Select(e => e.GetString()!).ToList(),
            DecisionType = r.GetProperty("classes").TryGetProperty("decision", out var dec) && dec.TryGetProperty("type", out var dt)
                ? dt.GetString() ?? "" : "",
            Nhits = ParseModel(models.GetProperty("nhits")),
            Tft = ParseModel(models.GetProperty("tft")),
            NhitsPredictionLength = models.GetProperty("nhits").GetProperty("prediction_length").GetInt32(),
            Rul = cal,
        };
    }

    private static SubModelSpec ParseModel(JsonElement m)
    {
        var cols = m.GetProperty("columns").EnumerateArray().Select(e => e.GetString()!).ToList();
        var sc = m.GetProperty("scalers");
        var scalers = cols.Select(c => new Scaler(sc.GetProperty(c).GetProperty("center").GetDouble(), sc.GetProperty(c).GetProperty("scale").GetDouble())).ToList();
        return new SubModelSpec(m.GetProperty("onnx").GetString()!, m.GetProperty("encoder_length").GetInt32(),
            m.TryGetProperty("decoder_length", out var d) ? d.GetInt32() : 0, cols, scalers);
    }
}
