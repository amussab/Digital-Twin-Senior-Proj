using System.Numerics;

namespace DigitalTwin.Twin;

/// <summary>
/// ME finite-element model of the rig shaft. Euler-Bernoulli beam elements (bending only), 2 degrees
/// of freedom per node (deflection, slope). The mesh puts a node at every bearing, probe and disk and
/// at mid-span, with elements no longer than MaxElementLength. Bearings are springs K at their nodes,
/// disks are point masses with diametral inertia, the shaft overhang past each bearing is included.
/// Steady unbalance response: [K(1 + i eta) - w^2 M] x = F, F = m e w^2 shared equally by the disks.
/// x(t) = Re(X e^{jwt}); phase = arg(X).
/// </summary>
public sealed class FeBeam
{
    public RotorParams P { get; }
    public double[] NodeX { get; }
    private readonly int _n;
    private readonly double[,] _k;
    private readonly double[,] _m;
    private readonly int[] _bearingNode, _probeNode, _diskNode;
    public double MidSpan => (P.BearingPositions[0] + P.BearingPositions[1]) / 2;

    public FeBeam(RotorParams? p = null)
    {
        P = p ?? RotorParams.Default;
        NodeX = BuildMesh(P);
        int ne = NodeX.Length - 1;
        _n = 2 * (ne + 1);
        _k = new double[_n, _n];
        _m = new double[_n, _n];
        double a = Math.PI * P.ShaftD * P.ShaftD / 4;
        double ei = P.E * Math.PI * Math.Pow(P.ShaftD, 4) / 64;
        for (int e = 0; e < ne; e++)
        {
            double le = NodeX[e + 1] - NodeX[e];
            double c = ei / Math.Pow(le, 3), cm = P.Rho * a * le / 420;
            double[,] ke =
            {
                { 12, 6 * le, -12, 6 * le },
                { 6 * le, 4 * le * le, -6 * le, 2 * le * le },
                { -12, -6 * le, 12, -6 * le },
                { 6 * le, 2 * le * le, -6 * le, 4 * le * le },
            };
            double[,] me =
            {
                { 156, 22 * le, 54, -13 * le },
                { 22 * le, 4 * le * le, 13 * le, -3 * le * le },
                { 54, 13 * le, 156, -22 * le },
                { -13 * le, -3 * le * le, -22 * le, 4 * le * le },
            };
            for (int r = 0; r < 4; r++)
                for (int s = 0; s < 4; s++)
                {
                    _k[2 * e + r, 2 * e + s] += c * ke[r, s];
                    _m[2 * e + r, 2 * e + s] += cm * me[r, s];
                }
        }
        _bearingNode = P.BearingPositions.Select(NodeAt).ToArray();
        _probeNode = P.ProbePositions.Select(NodeAt).ToArray();
        _diskNode = P.Disks.Select(d => NodeAt(d.PositionM)).ToArray();
        for (int i = 0; i < P.Disks.Length; i++)
        {
            int j = _diskNode[i];
            _m[2 * j, 2 * j] += P.Disks[i].MassKg;
            _m[2 * j + 1, 2 * j + 1] += P.Disks[i].DiametralInertiaKgM2;
        }
    }

    private static double[] BuildMesh(RotorParams p)
    {
        var key = new List<double> { 0.0, p.ShaftLength, (p.BearingPositions[0] + p.BearingPositions[1]) / 2 };
        key.AddRange(p.BearingPositions);
        key.AddRange(p.ProbePositions);
        key.AddRange(p.Disks.Select(d => d.PositionM));
        var pts = key.Select(v => Math.Round(v, 9)).Distinct().OrderBy(v => v).ToList();
        var x = new List<double> { pts[0] };
        for (int i = 0; i + 1 < pts.Count; i++)
        {
            double a = pts[i], b = pts[i + 1];
            int n = Math.Max(1, (int)Math.Ceiling((b - a) / p.MaxElementLength - 1e-9));
            for (int k = 1; k <= n; k++) x.Add(a + (b - a) * k / n);
        }
        return x.ToArray();
    }

    /// <summary>Index of the node at axial position <paramref name="pos"/> [m] (nearest node).</summary>
    public int NodeAt(double pos)
    {
        int best = 0;
        for (int i = 1; i < NodeX.Length; i++)
            if (Math.Abs(NodeX[i] - pos) < Math.Abs(NodeX[best] - pos)) best = i;
        return best;
    }

    /// <summary>Complex deflection of every node [m] for unbalance <paramref name="unb"/> [kg m].</summary>
    public Complex[] Deflections(double k1, double k2, double rpm, Complex unb)
    {
        double w = 2 * Math.PI * rpm / 60.0;
        var d = new Complex[_n, _n];
        for (int r = 0; r < _n; r++)
            for (int s = 0; s < _n; s++)
                d[r, s] = new Complex(_k[r, s], P.ShaftLossFactor * _k[r, s]) - w * w * _m[r, s];
        var ks = new[] { k1, k2 };
        for (int b = 0; b < 2; b++)
        {
            int dof = 2 * _bearingNode[b];
            d[dof, dof] += new Complex(ks[b], w * P.CBearing[b]);
        }
        var f = new Complex[_n];
        foreach (int j in _diskNode) f[2 * j] += unb * (w * w) / _diskNode.Length;
        var q = SolveComplex(d, f);
        var y = new Complex[NodeX.Length];
        for (int i = 0; i < y.Length; i++) y[i] = q[2 * i];
        return y;
    }

    /// <summary>Complex probe displacements [m] (PROX 1, PROX 3).</summary>
    public Complex[] Response(double k1, double k2, double rpm, Complex unb)
    {
        var y = Deflections(k1, k2, rpm, unb);
        return new[] { y[_probeNode[0]], y[_probeNode[1]] };
    }

    public Complex[] UnitResponse(double k1, double k2, double rpm) => Response(k1, k2, rpm, Complex.One);

    /// <summary>(p1 amp um, p1 phase rad, p2 amp um, p2 phase rad), zero-to-peak.</summary>
    public (double A1, double Ph1, double A2, double Ph2) Probes(double k1, double k2, double rpm, Complex unb)
    {
        var x = Response(k1, k2, rpm, unb);
        return (x[0].Magnitude * 1e6, x[0].Phase, x[1].Magnitude * 1e6, x[1].Phase);
    }

    /// <summary>Static deflection [m] at <paramref name="pos"/> under a load [N] applied there.
    /// Used for the PL^3/48EI hand check.</summary>
    public double StaticDeflection(double k1, double k2, double pos, double load)
    {
        var d = new Complex[_n, _n];
        for (int r = 0; r < _n; r++)
            for (int s = 0; s < _n; s++) d[r, s] = _k[r, s];
        d[2 * _bearingNode[0], 2 * _bearingNode[0]] += k1;
        d[2 * _bearingNode[1], 2 * _bearingNode[1]] += k2;
        var f = new Complex[_n];
        int j = NodeAt(pos);
        f[2 * j] = load;
        return SolveComplex(d, f)[2 * j].Real;
    }

    /// <summary>First critical speed [rpm]: the speed where the probe response peaks.</summary>
    public double FirstCriticalRpm(double k1, double k2, double from = 300, double to = 6000)
    {
        double Amp(double r) { var x = Response(k1, k2, r, Complex.One); return Math.Max(x[0].Magnitude, x[1].Magnitude); }
        double best = from, bv = 0;
        for (double r = from; r <= to; r += 5) { double v = Amp(r); if (v > bv) { bv = v; best = r; } }
        double lo = best - 5, hi = best + 5;
        for (double r = lo; r <= hi; r += 0.5) { double v = Amp(r); if (v > bv) { bv = v; best = r; } }
        return best;
    }

    /// <summary>Gaussian elimination on the banded beam matrix (each node couples only to its
    /// neighbours, so every row has at most 3 non-zero terms each side of the diagonal).</summary>
    private const int HalfBand = 3;

    private static Complex[] SolveComplex(Complex[,] a, Complex[] b)
    {
        int n = b.Length;
        for (int col = 0; col < n; col++)
        {
            if (a[col, col].Magnitude == 0) throw new InvalidOperationException("Singular dynamic stiffness matrix.");
            int last = Math.Min(n - 1, col + HalfBand);
            for (int r = col + 1; r <= last; r++)
            {
                var fct = a[r, col] / a[col, col];
                if (fct == Complex.Zero) continue;
                for (int s = col; s <= last; s++) a[r, s] -= fct * a[col, s];
                b[r] -= fct * b[col];
            }
        }
        var x = new Complex[n];
        for (int r = n - 1; r >= 0; r--)
        {
            var sum = b[r];
            for (int s = r + 1; s <= Math.Min(n - 1, r + HalfBand); s++) sum -= a[r, s] * x[s];
            x[r] = sum / a[r, r];
        }
        return x;
    }
}
