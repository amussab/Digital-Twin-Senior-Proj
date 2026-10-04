namespace DigitalTwin.Backend.Inference;

/// <summary>
/// C# port of AI-engine/aiengine/features.py::engineer_arrays for ONE biaxial bearing (16 raw features = X axis 8
/// then Y axis 8, COE C7 order). Stateless; the healthy baseline is passed in. Output order = contract engineered_order.
/// </summary>
public sealed class FeatureEngineer
{
    public const int RawCount = 16;
    private readonly FeatureConstants _c;
    // index of each raw feature inside one axis 8-vector
    private const int Bp = 0, Kurt = 1, Crest = 2, Env = 3;
    private static readonly int[] Fam = { 4, 5, 6, 7 }; // ftf, bsf, bpfo, bpfi

    public FeatureEngineer(FeatureConstants c) => _c = c;
    public FeatureConstants Constants => _c;
    public int BaselineWindows => _c.BaselineWindows;

    /// <summary>Median of every raw column over the commissioning windows, then the relative floor (features.py::_floor).</summary>
    public double[] ComputeBaseline(IReadOnlyList<float[]> windows)
    {
        var b = new double[RawCount];
        for (var j = 0; j < RawCount; j++) b[j] = Median(windows.Select(w => (double)w[j]).ToArray());
        var amax = b.Max(Math.Abs);
        for (var j = 0; j < RawCount; j++) b[j] += _c.RelFloor * (amax + _c.Eps);
        return b;
    }

    public static double Median(double[] v)
    {
        var s = (double[])v.Clone(); Array.Sort(s);
        var n = s.Length;
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    }

    public double[] Compute(float[] raw, double rpm, double[] baseline)
    {
        double eps = _c.Eps;
        double V(int ax, int i) => raw[ax * 8 + i];
        double Ratio(int ax, int i) => V(ax, i) / (baseline[ax * 8 + i] + eps);
        var envR = new[] { Ratio(0, Env), Ratio(1, Env) };
        var bpR = new[] { Ratio(0, Bp), Ratio(1, Bp) };
        var famR = new double[4, 2];
        for (var f = 0; f < 4; f++) for (var a = 0; a < 2; a++) famR[f, a] = Ratio(a, Fam[f]);
        double envMax = Math.Max(envR[0], envR[1]), bpMax = Math.Max(bpR[0], bpR[1]);
        double famMax = double.MinValue; foreach (var v in famR) famMax = Math.Max(famMax, v);

        double kMax = Math.Max(V(0, Kurt), V(1, Kurt)), cMax = Math.Max(V(0, Crest), V(1, Crest));
        double kBase = Math.Max(baseline[Kurt], baseline[8 + Kurt]);

        var loud = envR[1] > envR[0] ? 1 : 0; // numpy argmax: first index on ties
        var pf = new double[4]; for (var f = 0; f < 4; f++) pf[f] = famR[f, loud];
        var tot = pf.Sum() + eps;
        var shares = pf.Select(v => v / tot).ToArray();
        var srt = (double[])shares.Clone(); Array.Sort(srt);

        var rawFam = new double[4];
        for (var f = 0; f < 4; f++) rawFam[f] = Math.Sqrt(V(0, Fam[f]) * V(0, Fam[f]) + V(1, Fam[f]) * V(1, Fam[f]));
        var rtot = rawFam.Sum() + eps;

        double L(double v) => Math.Log(Math.Max(v, eps));
        var o = new double[21];
        o[0] = HealthIndex(envMax, bpMax, famMax);
        o[1] = L(envMax); o[2] = L(bpMax);
        o[3] = kMax; o[4] = cMax; o[5] = Math.Max(kMax - kBase, 0.0);
        o[6] = (envR[0] - envR[1]) / (envR[0] + envR[1] + eps);
        o[7] = shares[0]; o[8] = shares[1]; o[9] = shares[2]; o[10] = shares[3];
        o[11] = srt[3] - srt[2];
        for (var f = 0; f < 4; f++) o[12 + f] = L(Math.Max(famR[f, 0], famR[f, 1]));
        for (var f = 0; f < 4; f++) o[16 + f] = rawFam[f] / rtot;
        o[20] = (rpm - _c.RpmCenter) / _c.RpmScale;
        return o;
    }

    private double HealthIndex(double env, double bp, double fam)
    {
        var fused = Math.Pow(Math.Max(env, 1.0), _c.WEnv) * Math.Pow(Math.Max(bp, 1.0), _c.WBp) * Math.Pow(Math.Max(fam, 1.0), _c.WFamily);
        var hi = Math.Log(1 + _c.HiCurveK * (fused - 1.0)) / Math.Log(1 + _c.HiCurveK * (_c.HiRRef - 1.0));
        return Math.Clamp(hi, 0.0, _c.HiClipMax);
    }
}
