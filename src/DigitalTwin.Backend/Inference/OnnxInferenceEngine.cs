using System.Diagnostics;
using DigitalTwin.Backend.Models;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DigitalTwin.Backend.Inference;

/// <summary>
/// Runs the trained TFT (class logits) and N-HiTS (health-index forecast) in-process via ONNX Runtime.
/// C# mirror of AI-engine/aiengine/engine.py (step) + rul.py (estimate, stage, interval). Every constant comes
/// from model_contract.json. N-HiTS is exported with batch 1 (one call per bearing-window).
/// </summary>
public sealed class OnnxInferenceEngine : IInferenceEngine, IDisposable
{
    private readonly ModelContract _c;
    private readonly InferenceSession _nhits, _tft;
    private readonly bool _nhitsHasDecoder, _tftHasDecoder;
    private readonly int[] _tftIdx, _nhIdx;
    private readonly ILogger _log;
    private readonly int _classHealthy;

    public OnnxInferenceEngine(ModelContract contract, ILogger log)
    {
        _c = contract; _log = log;
        var so = new Microsoft.ML.OnnxRuntime.SessionOptions { IntraOpNumThreads = 1 };
        _nhits = new InferenceSession(Path.Combine(contract.Directory, contract.Nhits.OnnxFile), so);
        _tft = new InferenceSession(Path.Combine(contract.Directory, contract.Tft.OnnxFile), so);
        _nhitsHasDecoder = _nhits.InputMetadata.ContainsKey("decoder_cont");
        _tftHasDecoder = _tft.InputMetadata.ContainsKey("decoder_cont");
        _tftIdx = contract.Tft.Columns.Select(c => contract.Features.Index(c)).ToArray();
        _nhIdx = contract.Nhits.Columns.Select(c => contract.Features.Index(c)).ToArray();
        _classHealthy = contract.ClassLabels.ToList().IndexOf("healthy");
        if (_classHealthy < 0) throw new InvalidDataException("contract classes lack 'healthy'");
        WarmUp();
    }

    /// <summary>Runs each ONNX session a few times on zero tensors at startup, so the first real window does not
    /// pay the one-time session/kernel initialisation (measured ~170 ms max on the first call, which would eat the
    /// IS1 &lt;500 ms budget on top of COE's 333 ms). Outputs are discarded; no tracker state is touched.</summary>
    private void WarmUp()
    {
        var sw = Stopwatch.StartNew();
        for (var k = 0; k < 3; k++)
        {
            var L = _c.Tft.EncoderLength; var R = _tftIdx.Length;
            var tf = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("encoder_cont", new DenseTensor<float>(new float[L * R], new[] { 1, L, R })) };
            if (_tftHasDecoder) tf.Add(NamedOnnxValue.CreateFromTensor("decoder_cont", new DenseTensor<float>(new float[R], new[] { 1, 1, R })));
            using (_tft.Run(tf)) { }
            var NL = _c.Nhits.EncoderLength; var NR = _nhIdx.Length;
            var nf = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("encoder_cont", new DenseTensor<float>(new float[NL * NR], new[] { 1, NL, NR })) };
            if (_nhitsHasDecoder) nf.Add(NamedOnnxValue.CreateFromTensor("decoder_cont", new DenseTensor<float>(new float[_c.NhitsPredictionLength * NR], new[] { 1, _c.NhitsPredictionLength, NR })));
            using (_nhits.Run(nf)) { }
        }
        _log.LogInformation("ONNX sessions warmed up in {Ms:F0} ms (3 TFT + 3 N-HiTS dummy runs)", sw.Elapsed.TotalMilliseconds);
    }

    public string Name => "OnnxInferenceEngine";
    public bool IsSimulated => false;
    public string Provenance => _c.Provenance;
    public FeatureConstants Features => _c.Features;
    public int RequiredHistory => _c.RequiredHistory;
    public ModelContract Contract => _c;

    public BearingTracker CreateTracker(int bearing, int baselineWindowsOverride) =>
        new(bearing, new FeatureEngineer(_c.Features), baselineWindowsOverride > 0 ? baselineWindowsOverride : _c.Features.BaselineWindows,
            _c.RequiredHistory, _c.HistoryCap);

    /// <summary>(x - center) / scale in float64, cast to float32 (infer.py::scale).</summary>
    private static float Scale(double x, Scaler s) => (float)((x - s.Center) / s.Scale);

    /// <summary>TFT tensors for the newest window: encoder = previous L rows, decoder = newest row. Returned flattened.</summary>
    internal (float[] Enc, float[] Dec) TftTensors(BearingTracker t)
    {
        var L = _c.Tft.EncoderLength; var R = _tftIdx.Length; var rows = t.Rows;
        var enc = new float[L * R]; var dec = new float[R];
        for (var i = 0; i < L; i++)
        {
            var row = rows[rows.Count - 1 - L + i];
            for (var j = 0; j < R; j++) enc[i * R + j] = Scale(row[_tftIdx[j]], _c.Tft.Scalers[j]);
        }
        var last = rows[^1];
        for (var j = 0; j < R; j++) dec[j] = Scale(last[_tftIdx[j]], _c.Tft.Scalers[j]);
        return (enc, dec);
    }

    private float[] NhitsTensor(BearingTracker t)
    {
        var NL = _c.Nhits.EncoderLength; var R = _nhIdx.Length; var rows = t.Rows;
        var take = Math.Min(NL, rows.Count);
        var enc = new float[NL * R];
        for (var i = 0; i < NL; i++)
        {
            // left-pad with the first available row while the buffer is short (as in evaluation)
            var src = i < NL - take ? rows[rows.Count - take] : rows[rows.Count - take + (i - (NL - take))];
            for (var j = 0; j < R; j++) enc[i * R + j] = Scale(src[_nhIdx[j]], _c.Nhits.Scalers[j]);
        }
        return enc;
    }

    public BearingResult Infer(BearingTracker t)
    {
        var tw = Stopwatch.GetTimestamp();
        var (enc, dec) = TftTensors(t);
        var R = _tftIdx.Length; var L = _c.Tft.EncoderLength;
        var feeds = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_cont", new DenseTensor<float>(enc, new[] { 1, L, R })),
        };
        if (_tftHasDecoder) feeds.Add(NamedOnnxValue.CreateFromTensor("decoder_cont", new DenseTensor<float>(dec, new[] { 1, 1, R })));
        double[] probs;
        using (var res = _tft.Run(feeds))
        {
            var logits = res.First(r => r.Name == "logits").AsEnumerable<float>().Take(_c.ClassLabels.Count).Select(v => (double)v).ToArray();
            probs = Softmax(logits);
        }
        probs = ApplyDecision(probs, t.OnsetConfirmed);
        t.PushProbs(probs, _c.Rul.ClassSmoothingWindows);
        var tTft = Stopwatch.GetTimestamp();

        var nenc = NhitsTensor(t);
        var nfeeds = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_cont", new DenseTensor<float>(nenc, new[] { 1, _c.Nhits.EncoderLength, _nhIdx.Length })),
        };
        if (_nhitsHasDecoder)
        {
            var ndec = new float[_c.NhitsPredictionLength * _nhIdx.Length];
            for (var s = 0; s < _c.NhitsPredictionLength; s++)
                Array.Copy(nenc, (_c.Nhits.EncoderLength - 1) * _nhIdx.Length, ndec, s * _nhIdx.Length, _nhIdx.Length);
            nfeeds.Add(NamedOnnxValue.CreateFromTensor("decoder_cont", new DenseTensor<float>(ndec, new[] { 1, _c.NhitsPredictionLength, _nhIdx.Length })));
        }
        double[] forecast;
        using (var res = _nhits.Run(nfeeds))
            forecast = res.First().AsEnumerable<float>().Select(v => (double)v).ToArray();
        var tNh = Stopwatch.GetTimestamp();

        // RUL threshold class: mean prob over last k windows; healthy if p(healthy) >= 0.5 else argmax fault prob
        var pm = new double[probs.Length];
        foreach (var p in t.Probs) for (var i = 0; i < pm.Length; i++) pm[i] += p[i];
        for (var i = 0; i < pm.Length; i++) pm[i] /= t.Probs.Count;
        string thrCls;
        if (pm[_classHealthy] >= 0.5) thrCls = "healthy";
        else
        {
            var best = -1;
            for (var i = 0; i < pm.Length; i++) if (i != _classHealthy && (best < 0 || pm[i] > pm[best])) best = i;
            thrCls = _c.ClassLabels[best];
        }
        var ts = t.TimeHistory;
        double cad = 0;
        if (ts.Count > 1) cad = FeatureEngineer.Median(Enumerable.Range(1, ts.Count - 1).Select(i => ts[i] - ts[i - 1]).ToArray());
        double? tau = t.OnsetT is { } ot ? t.CurrentT - ot : null;
        var est = RulEstimator.Estimate(_c.Rul, t.HiHistory, forecast, thrCls, tau, t.OnsetHi, Math.Max(cad, 1e-9));
        var stage = RulEstimator.Stage(tau, est);

        var k = Array.IndexOf(probs, probs.Max());
        var hi = t.LatestHi!.Value;
        var details = new BearingDetails(stage, t.OnsetT is not null, thrCls, forecast,
            est * Math.Max(1 - _c.Rul.IntervalRel, 0), est * (1 + _c.Rul.IntervalRel),
            (tTft - tw) * 1000.0 / Stopwatch.Frequency, (tNh - tTft) * 1000.0 / Stopwatch.Frequency,
            enc.Select(v => (double)v).ToArray(), dec.Select(v => (double)v).ToArray());
        return new BearingResult(t.Bearing, BearingStates.Ready, _c.ClassLabels[k], probs[k] * 100, hi, est,
            _c.ClassLabels.Zip(probs).ToDictionary(p => p.First, p => p.Second), t.Fill, t.Required, details);
    }

    /// <summary>Port of engine.py's v2 decision rule (contract classes.decision): before the causal onset is
    /// confirmed the bearing is reported healthy; after it, p(healthy) = 0 and the fault channels are renormalised.</summary>
    private double[] ApplyDecision(double[] probs, bool onsetConfirmed)
    {
        var gate = _c.DecisionType is "rtf_onset_gate" or "hierarchical_onset_gate";
        if (!gate) return probs;
        if (!onsetConfirmed)
        {
            var oneHot = new double[probs.Length];
            oneHot[_classHealthy] = 1.0;
            return oneHot;
        }
        if (_c.DecisionType == "hierarchical_onset_gate") return probs;
        var p = (double[])probs.Clone();
        p[_classHealthy] = 0.0;
        var sum = Math.Max(p.Sum(), 1e-12);
        for (var i = 0; i < p.Length; i++) p[i] /= sum;
        return p;
    }

    private static double[] Softmax(double[] x)
    {
        var max = x.Max(); var e = x.Select(v => Math.Exp(v - max)).ToArray(); var s = e.Sum();
        return e.Select(v => v / s).ToArray();
    }

    public void Dispose() { _nhits.Dispose(); _tft.Dispose(); }
}

/// <summary>Port of aiengine/rul.py (trend_steps, estimate with method "loglin"/"trend"/"onset"/blend, stage_estimate).</summary>
public static class RulEstimator
{
    private const double Eps = 1e-9, HiFloor = 0.02;

    public static double TrendSteps(IReadOnlyList<double> series, double threshold, int nowIndex)
    {
        var n = series.Count;
        if (n < 3) return double.PositiveInfinity;
        var y = series.Select(v => Math.Log(Math.Max(v, HiFloor))).ToArray();
        var x = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        var w = new double[n];
        for (var i = 0; i < n; i++) w[i] = Math.Exp(-1.5 + 1.5 * i / (n - 1));   // exp(linspace(-1.5, 0, n))
        var sw = w.Sum();
        double xm = 0, ym = 0;
        for (var i = 0; i < n; i++) { xm += w[i] * x[i]; ym += w[i] * y[i]; }
        xm /= sw; ym /= sw;
        double vx = 0, cxy = 0;
        for (var i = 0; i < n; i++) { vx += w[i] * (x[i] - xm) * (x[i] - xm); cxy += w[i] * (x[i] - xm) * (y[i] - ym); }
        if (vx <= Eps) return double.PositiveInfinity;
        var slope = cxy / vx;
        if (slope <= 1e-6) return double.PositiveInfinity;
        var level = ym + slope * (nowIndex - xm);
        return Math.Max((Math.Log(Math.Max(threshold, HiFloor)) - level) / slope, 0.0);
    }

    public static double Estimate(RulCalibration cal, IReadOnlyList<double> hiHist, IReadOnlyList<double> forecast, string? cls,
        double? tauHours, double? hiOnset, double cadenceHours)
    {
        var thr = cal.ThresholdFor(cls);
        var hist = hiHist.Skip(Math.Max(0, hiHist.Count - cal.HistoryWindows)).ToList();
        var series = hist.Concat(forecast).ToList();
        var steps = TrendSteps(series, thr, hist.Count - 1);
        var trendH = double.IsFinite(steps) ? steps * cadenceHours : double.PositiveInfinity;

        var onsetH = double.PositiveInfinity; var uRaw = double.NaN;
        var tail = hist.Skip(Math.Max(0, hist.Count - 3)).Concat(forecast.Take(3));
        var hiNow = tail.Average();
        if (tauHours is { } tau0 && hiOnset is { } ho && tau0 > 0)
        {
            uRaw = (hiNow - ho) / Math.Max(thr - ho, 1e-3);
            var u = Math.Clamp(uRaw, cal.UMin, 0.999);
            onsetH = tau0 * (1 - u) / u;
        }
        var spanH = Math.Max(tauHours ?? 0.0, hist.Count * cadenceHours);
        var cap = cal.MaxRulFactor * spanH;
        trendH = Math.Min(trendH, cap); onsetH = Math.Min(onsetH, cap);

        double[]? llf = null;
        if (tauHours is { } tau && tau > 0)
        {
            var u = double.IsFinite(uRaw) ? Math.Clamp(uRaw, -1.0, 2.0) : 0.0;
            llf = new[] { 1.0, Math.Log(tau + cadenceHours), Math.Log(trendH + cadenceHours), u, hiNow / Math.Max(thr, 1e-3) };
        }
        double rul;
        if (cal.Method == "loglin")
        {
            if (llf is not null && cal.LoglinCoef.Count > 0)
            {
                var dot = 0.0; for (var i = 0; i < llf.Length; i++) dot += cal.LoglinCoef[i] * llf[i];
                rul = Math.Min(Math.Exp(Math.Clamp(dot, -20, 20)), cap);
            }
            else rul = trendH;
        }
        else if (cal.Method == "trend") rul = double.IsFinite(trendH) ? trendH : onsetH;
        else if (cal.Method == "onset") rul = double.IsFinite(onsetH) ? onsetH : trendH;
        else if (double.IsFinite(trendH) && double.IsFinite(onsetH)) rul = cal.BlendW * trendH + (1 - cal.BlendW) * onsetH;
        else rul = double.IsFinite(trendH) ? trendH : onsetH;
        if (!double.IsFinite(rul)) rul = cap;
        return Math.Max(rul, 0.0);
    }

    public static int Stage(double? tauHours, double rulHours)
    {
        if (tauHours is not { } tau) return 1;
        var u = tau / Math.Max(tau + rulHours, Eps);
        return (int)Math.Min(6, 2 + Math.Floor(Math.Clamp(u, 0, 1) * 5));
    }
}
