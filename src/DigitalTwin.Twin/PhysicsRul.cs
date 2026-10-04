namespace DigitalTwin.Twin;

public sealed record RulEstimate(double RulHours, double LoHours, double HiHours);

/// <summary>
/// Physics RUL from identified K(t). Fit ln(K/K0) = a + b t over recent points and extrapolate to
/// ln(1 - failureDrop) (25 % stiffness drop, [TEAM DOC]). Interval from slope +/- 1.96 se. Null
/// until enough points exist or unless the slope is significantly negative. Port of physics_rul.py.
/// </summary>
public static class PhysicsRul
{
    public static RulEstimate? Fit(IReadOnlyList<double> t, IReadOnlyList<double> k, double k0, RotorParams p)
    {
        int n = t.Count;
        if (n < p.MinRulPoints) return null;
        double tmin = t.Min(), tmax = t.Max();
        if (tmax - tmin <= 0) return null;
        var y = k.Select(v => Math.Log(v / k0)).ToArray();
        double tm = t.Average(), ym = y.Average();
        double sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++) { sxx += (t[i] - tm) * (t[i] - tm); sxy += (t[i] - tm) * (y[i] - ym); }
        double b = sxy / sxx;
        double a = ym - b * tm;
        if (b >= 0) return null;
        double ss = 0;
        for (int i = 0; i < n; i++) { double r = y[i] - (a + b * t[i]); ss += r * r; }
        double se = Math.Sqrt(ss / Math.Max(n - 2, 1) / sxx);
        if (b + 1.96 * se >= 0) return null;
        double yFail = Math.Log(1.0 - p.FailureDrop);
        double tNow = t[n - 1];
        double Rul(double slope) => Math.Max(tm + (yFail - ym) / slope - tNow, 0.0);
        return new RulEstimate(Rul(b), Rul(b - 1.96 * se), Rul(b + 1.96 * se));
    }
}

internal sealed class BearingTracker
{
    private readonly RotorParams _p;
    public List<double> T { get; } = new();
    public List<double> K { get; } = new();
    public double K0 { get; private set; }

    public BearingTracker(RotorParams p) => _p = p;

    public void SetBaseline(IEnumerable<double> ks)
    {
        var s = ks.OrderBy(v => v).ToArray();
        int n = s.Length;
        K0 = n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    public void Add(double th, double k) { T.Add(th); K.Add(k); }
    public double Drop() => 1.0 - K[^1] / K0;

    public RulEstimate? Rul()
    {
        int skip = Math.Max(T.Count - _p.RulFitPoints, 0);
        return PhysicsRul.Fit(T.Skip(skip).ToArray(), K.Skip(skip).ToArray(), K0, _p);
    }
}
