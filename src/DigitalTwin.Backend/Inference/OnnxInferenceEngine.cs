using System.Text.Json;
using DigitalTwin.Backend.Models;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace DigitalTwin.Backend.Inference;

/// <summary>
/// Runs TFT (class logits) and N-HiTS (health-index forecast) in-process. All model metadata comes from
/// model_contract.json. RUL = exponential-trend extrapolation of (recent HI + forecast) to the class threshold,
/// per the contract's rul_procedure. Pending: final contract (per-bearing rebuild) and the exact blend with the
/// population prior; the threshold lookup below tolerates a missing table and falls back to HI = 1.0.
/// </summary>
public sealed class OnnxInferenceEngine : IInferenceEngine, IDisposable
{
    private readonly ModelContract _c;
    private readonly InferenceSession _nhits, _tft;
    private readonly TensorMapper _mapper = new();
    private readonly double _hoursPerWindow;
    private readonly ILogger _log;

    public OnnxInferenceEngine(ModelContract contract, double hoursPerWindow, ILogger log)
    {
        _c = contract; _hoursPerWindow = hoursPerWindow; _log = log;
        _nhits = new InferenceSession(Path.Combine(contract.Directory, contract.Nhits.OnnxFile));
        _tft = new InferenceSession(Path.Combine(contract.Directory, contract.Tft.OnnxFile));
    }

    public string Name => "OnnxInferenceEngine";
    public bool IsSimulated => false;
    public string Provenance => _c.Provenance;
    public FeatureConstants Features => _c.Features;
    public int RequiredHistory => _c.RequiredHistory;

    public BearingResult Infer(BearingTracker t)
    {
        var hist = t.History;

        var tftOut = _tft.Run(_mapper.Build(_c.Tft, hist, _tft.InputMetadata, "tft")).First().AsEnumerable<float>().ToArray();
        var logits = tftOut.TakeLast(_c.ClassLabels.Count).ToArray(); // [1,1,n_classes]
        var probs = Softmax(logits);
        var k = Array.IndexOf(probs, probs.Max());
        var label = _c.ClassLabels[k]; // output-channel order from contract

        var forecast = _nhits.Run(_mapper.Build(_c.Nhits, hist, _nhits.InputMetadata, "nhits")).First().AsEnumerable<float>().ToArray();
        var rul = RulHours(hist.Select(r => r["hi"]).ToList(), forecast, label);

        if (_mapper.Unmapped.Count > 0)
        {
            _log.LogWarning("TensorMapper: contract columns with no engineered counterpart (fed as 0): {Cols}", string.Join(", ", _mapper.Unmapped));
            _mapper.Unmapped.Clear();
        }
        return new BearingResult(t.Bearing, BearingStates.Ready, label, probs[k] * 100, t.Latest!["hi"], rul,
            _c.ClassLabels.Zip(probs).ToDictionary(p => p.First, p => p.Second), t.Fill, t.Required);
    }

    private double? RulHours(List<double> hi, float[] forecast, string label)
    {
        var n = Math.Min(_c.TrendHistoryWindows, hi.Count);
        var series = hi.TakeLast(n).Concat(forecast.Select(f => (double)f)).Select(v => Math.Max(v, _c.TrendFloorHi)).ToArray();
        // least squares of ln(hi) vs step index
        var m = series.Length; var xs = Enumerable.Range(0, m).Select(i => (double)i).ToArray();
        var ys = series.Select(v => Math.Log(v)).ToArray();
        var xm = xs.Average(); var ym = ys.Average();
        var den = xs.Sum(x => (x - xm) * (x - xm));
        if (den == 0) return null;
        var slope = xs.Zip(ys).Sum(p => (p.First - xm) * (p.Second - ym)) / den;
        var current = Math.Max(hi[^1], _c.TrendFloorHi);
        var threshold = Threshold(label);
        if (current >= threshold) return 0;
        if (slope <= 1e-6) return null; // no measurable degradation trend
        return Math.Log(threshold / current) / slope * _hoursPerWindow;
    }

    /// <summary>Class-conditional failure threshold from rul_calibration if present (tolerant lookup), else 1.0.</summary>
    private double Threshold(string label)
    {
        var cal = _c.RulCalibration;
        if (cal.ValueKind != JsonValueKind.Object) return 1.0;
        foreach (var key in new[] { "thresholds", "class_thresholds", "failure_thresholds" })
            if (cal.TryGetProperty(key, out var t) && t.ValueKind == JsonValueKind.Object
                && t.TryGetProperty(label, out var v) && v.ValueKind == JsonValueKind.Number) return v.GetDouble();
        return 1.0;
    }

    private static double[] Softmax(float[] x)
    {
        var max = x.Max(); var e = x.Select(v => Math.Exp(v - max)).ToArray(); var s = e.Sum();
        return e.Select(v => v / s).ToArray();
    }

    public void Dispose() { _nhits.Dispose(); _tft.Dispose(); }
}
