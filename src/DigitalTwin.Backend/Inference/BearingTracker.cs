using DigitalTwin.Backend.Models;

namespace DigitalTwin.Backend.Inference;

/// <summary>Causal onset detector; port of aiengine/labels.py::OnlineOnset.</summary>
public sealed class OnlineOnset
{
    private readonly FeatureConstants _c; private readonly int _baselineWindows;
    private readonly List<double> _base = new();
    private double? _threshold; private int _streak;
    public int N { get; private set; }
    public int? OnsetIndex { get; private set; }

    public OnlineOnset(FeatureConstants c, int baselineWindows) { _c = c; _baselineWindows = baselineWindows; }

    public int? Update(double hi)
    {
        var k = N++;
        if (_threshold is null)
        {
            if (k < _c.OnsetRunInSkip) return OnsetIndex;
            _base.Add(hi);
            if (_base.Count >= _baselineWindows)
            {
                var m = _base.Average();
                var sd = Math.Sqrt(_base.Sum(v => (v - m) * (v - m)) / _base.Count); // numpy std (ddof 0)
                _threshold = m + _c.OnsetSigmas * Math.Max(sd, _c.OnsetSigmaFloor);
            }
            return OnsetIndex;
        }
        if (OnsetIndex is null)
        {
            _streak = hi > _threshold ? _streak + 1 : 0;
            if (_streak >= _c.OnsetConsecutive) OnsetIndex = k - _c.OnsetConsecutive + 1;
        }
        return OnsetIndex;
    }
}

/// <summary>
/// Per-bearing runtime state (mirror of engine.py::_Bearing): commissioning baseline (median of the first
/// baseline_windows windows when none is supplied), engineered-row history, HI/time history, causal onset,
/// class-probability history. Bearing 1 = raw[0:16], bearing 2 = raw[16:32].
/// </summary>
public sealed class BearingTracker
{
    private readonly FeatureEngineer _fe;
    private readonly int _baselineWindows, _required, _cap;
    private readonly List<(float[] Raw, double Rpm, double T)> _pending = new();
    private readonly List<double[]> _eng = new();
    private readonly List<double> _hi = new(), _t = new();
    private readonly Queue<double[]> _probs = new();
    private readonly OnlineOnset _onset;
    private readonly int _hiIdx;
    private double[]? _baseline;

    public BearingTracker(int bearing, FeatureEngineer fe, int baselineWindows, int requiredHistory, int historyCap = 48)
    {
        Bearing = bearing; _fe = fe; _baselineWindows = baselineWindows;
        _required = Math.Max(1, requiredHistory); _cap = Math.Max(historyCap, _required);
        _onset = new OnlineOnset(fe.Constants, baselineWindows);
        _hiIdx = fe.Constants.Index("hi");
    }

    public int Bearing { get; }
    public int Required => _required;
    public int Fill => _eng.Count;
    public int CommissioningFill => _pending.Count;
    /// <summary>Engineered rows (oldest first), contract order, at most HistoryCap.</summary>
    public IReadOnlyList<double[]> Rows => _eng;
    public IReadOnlyList<double> HiHistory => _hi;
    public IReadOnlyList<double> TimeHistory => _t;
    public double[]? Latest => _eng.Count > 0 ? _eng[^1] : null;
    public double? LatestHi => Latest?[_hiIdx];
    public double[]? Baseline => _baseline;
    /// <summary>True once the causal onset is confirmed (Python: b.onset.onset_index is not None).</summary>
    public bool OnsetConfirmed => _onset.OnsetIndex is not null;
    public double? OnsetT { get; private set; }
    public double? OnsetHi { get; private set; }
    public double CurrentT { get; private set; }
    public int FeatureIndex(string name) => _fe.Constants.Index(name);

    public string State => _baseline is null ? BearingStates.Commissioning
        : _eng.Count < _required ? BearingStates.WarmingUp : BearingStates.Ready;

    public void PushProbs(double[] p, int keep) { _probs.Enqueue(p); while (_probs.Count > keep) _probs.Dequeue(); }
    public IReadOnlyCollection<double[]> Probs => _probs;

    public void Update(ReadOnlySpan<float> window32, double rpm, double tHours)
    {
        CurrentT = tHours;
        var raw = window32.Slice((Bearing - 1) * FeatureEngineer.RawCount, FeatureEngineer.RawCount).ToArray();
        if (_baseline is null)
        {
            _pending.Add((raw, rpm, tHours));
            if (_pending.Count < _baselineWindows) return;
            _baseline = _fe.ComputeBaseline(_pending.Select(p => p.Raw).ToList());
            foreach (var p in _pending) Push(_fe.Compute(p.Raw, p.Rpm, _baseline), p.T);
            _pending.Clear();
            return;
        }
        Push(_fe.Compute(raw, rpm, _baseline), tHours);
    }

    private void Push(double[] e, double t)
    {
        _eng.Add(e); _hi.Add(e[_hiIdx]); _t.Add(t);
        if (_eng.Count > _cap) _eng.RemoveAt(0);
        if (_hi.Count > 48) { _hi.RemoveAt(0); _t.RemoveAt(0); }
        var before = _onset.OnsetIndex;
        var on = _onset.Update(e[_hiIdx]);
        if (on is { } o && before is null)
        {
            var back = _onset.N - 1 - o; // windows since the onset window
            OnsetT = back < _t.Count ? _t[_t.Count - 1 - back] : _t[0];
            OnsetHi = back < _hi.Count ? _hi[_hi.Count - 1 - back] : _hi[0];
        }
    }
}
