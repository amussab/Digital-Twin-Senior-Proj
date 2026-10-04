using DigitalTwin.Backend.Models;

namespace DigitalTwin.Backend.Inference;

/// <summary>
/// Per-bearing state: commissioning baseline, then a rolling buffer of engineered feature rows.
/// Bearing 1 = raw[0:16], bearing 2 = raw[16:32].
/// </summary>
public sealed class BearingTracker
{
    private readonly FeatureEngineer _fe;
    private readonly int _baselineWindows, _required;
    private readonly List<float[]> _commissioning = new();
    private readonly List<Dictionary<string, double>> _history = new();
    private double[]? _baseline;

    public BearingTracker(int bearing, FeatureEngineer fe, int baselineWindows, int requiredHistory)
    { Bearing = bearing; _fe = fe; _baselineWindows = baselineWindows; _required = Math.Max(1, requiredHistory); }

    public int Bearing { get; }
    public int Required => _required;
    public int Fill => _history.Count;
    public int CommissioningFill => _commissioning.Count;
    public IReadOnlyList<Dictionary<string, double>> History => _history;
    public Dictionary<string, double>? Latest => _history.Count > 0 ? _history[^1] : null;

    public string State => _baseline is null ? BearingStates.Commissioning
        : _history.Count < _required ? BearingStates.WarmingUp : BearingStates.Ready;

    public void Update(ReadOnlySpan<float> window32, double rpm)
    {
        var raw = window32.Slice((Bearing - 1) * _fe.RawCount, _fe.RawCount);
        if (_baseline is null)
        {
            _commissioning.Add(raw.ToArray());
            if (_commissioning.Count >= _baselineWindows) _baseline = _fe.ComputeBaseline(_commissioning);
            return;
        }
        _history.Add(_fe.Compute(raw.ToArray(), rpm, _baseline));
        var cap = Math.Max(_required, 64);
        if (_history.Count > cap) _history.RemoveAt(0);
    }
}
