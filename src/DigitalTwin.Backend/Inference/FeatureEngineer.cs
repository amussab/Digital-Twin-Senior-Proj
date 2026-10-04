namespace DigitalTwin.Backend.Inference;

/// <summary>
/// C# port of AI-test/features.py for ONE biaxial bearing (16 raw features = 2 channels x 8). Stateless; the
/// healthy baseline is passed in. All constants come from FeatureConstants (contract). Output is a name -> value
/// map so the tensor mapper picks columns by contract name, not by position.
/// </summary>
public sealed class FeatureEngineer
{
    private const double Eps = 1e-9;
    public const int Channels = 2;
    private readonly FeatureConstants _c;
    private readonly int _perChannel;
    private readonly int _ienv, _ibp, _ikurt, _icrest;
    private readonly int[] _family;

    public FeatureEngineer(FeatureConstants c)
    {
        _c = c;
        _perChannel = c.PerChannelFeatureSuffixes.Count;
        var suffixes = c.PerChannelFeatureSuffixes.ToList();
        int Ix(string s) { var i = suffixes.IndexOf(s); return i >= 0 ? i : throw new InvalidOperationException($"contract raw_feature_order lacks '{s}'"); }
        _ibp = Ix("bp_rms"); _ikurt = Ix("bp_kurtosis"); _icrest = Ix("bp_crest"); _ienv = Ix("env_rms");
        _family = new[] { Ix("ftf_mag"), Ix("bsf_mag"), Ix("bpfo_mag"), Ix("bpfi_mag") };
    }

    public int RawCount => Channels * _perChannel;
    public int BaselineWindows => _c.BaselineWindows;

    /// <summary>Median of each raw column over the commissioning windows (+eps), as in compute_baseline.</summary>
    public double[] ComputeBaseline(IReadOnlyList<float[]> windows)
    {
        var b = new double[RawCount];
        for (var j = 0; j < RawCount; j++)
        {
            var col = windows.Select(w => (double)w[j]).OrderBy(v => v).ToArray();
            var mid = col.Length / 2;
            b[j] = (col.Length % 2 == 1 ? col[mid] : (col[mid - 1] + col[mid]) / 2) + Eps;
        }
        return b;
    }

    public Dictionary<string, double> Compute(float[] raw, double rpm, double[] baseline)
    {
        double R(int ch, int idx) => raw[ch * _perChannel + idx] / baseline[ch * _perChannel + idx];
        var envR = new double[Channels]; var bpR = new double[Channels];
        for (var ch = 0; ch < Channels; ch++) { envR[ch] = R(ch, _ienv); bpR[ch] = R(ch, _ibp); }

        var fam = new double[4, Channels];
        for (var f = 0; f < 4; f++) for (var ch = 0; ch < Channels; ch++) fam[f, ch] = R(ch, _family[f]);

        var envMax = envR.Max(); var bpMax = bpR.Max();
        var famMax = 0.0; foreach (var v in fam) famMax = Math.Max(famMax, v);

        var o = new Dictionary<string, double>
        {
            ["hi"] = HealthIndex(envMax, bpMax, famMax),
            ["log_env_ratio"] = Math.Log(1 + Math.Max(envMax - 1, 0)),
            ["log_bp_ratio"] = Math.Log(1 + Math.Max(bpMax - 1, 0)),
        };
        double kMax = double.MinValue, cMax = double.MinValue, kBase = double.MinValue;
        for (var ch = 0; ch < Channels; ch++)
        {
            kMax = Math.Max(kMax, raw[ch * _perChannel + _ikurt]);
            cMax = Math.Max(cMax, raw[ch * _perChannel + _icrest]);
            kBase = Math.Max(kBase, baseline[ch * _perChannel + _ikurt]);
        }
        o["kurtosis_max"] = kMax; o["crest_max"] = cMax; o["kurtosis_rise"] = Math.Max(kMax - kBase, 0);

        // Plane asymmetry within one bearing: X-channel vs Y-channel envelope ratio.
        o["plane_asymmetry"] = (envR[0] - envR[1]) / (envR[0] + envR[1] + Eps);

        var loudest = 0; for (var ch = 1; ch < Channels; ch++) if (envR[ch] > envR[loudest]) loudest = ch;
        var pf = new double[4]; for (var f = 0; f < 4; f++) pf[f] = fam[f, loudest];
        var total = pf.Sum() + Eps;
        var shares = pf.Select(v => v / total).ToArray();
        o["ftf_share"] = shares[0]; o["bsf_share"] = shares[1]; o["bpfo_share"] = shares[2]; o["bpfi_share"] = shares[3];
        var sorted = shares.OrderBy(v => v).ToArray();
        o["family_contrast"] = sorted[3] - sorted[2];
        o["rpm_norm"] = (rpm - _c.RpmCenter) / _c.RpmHalfRange;
        return o;
    }

    private double HealthIndex(double env, double bp, double fam)
    {
        var fused = Math.Pow(Math.Max(env, 1), _c.WEnv) * Math.Pow(Math.Max(bp, 1), _c.WBp) * Math.Pow(Math.Max(fam, 1), _c.WFamily);
        var num = Math.Log(1 + _c.HiCurveK * (fused - 1));
        var den = Math.Log(1 + _c.HiCurveK * (_c.HiRRef - 1));
        return Math.Clamp(num / den, 0.0, 1.2);
    }
}
