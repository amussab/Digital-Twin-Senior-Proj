using System.Numerics;

namespace DigitalTwin.Twin;

/// <summary>
/// 1D Euler-Bernoulli FE rotor (Hermite beam, 2 DOF/node, lateral plane), steady-state unbalance
/// response. Port of aiengine/twin/fe_beam.py. x(t)=Re(X e^{jwt}); phase = arg(X).
/// </summary>
public sealed class FeBeam
{
    public RotorParams P { get; }
    private readonly int _n;
    private readonly double[,] _k;
    private readonly double[,] _m;

    public FeBeam(RotorParams? p = null)
    {
        P = p ?? RotorParams.Default;
        _n = 2 * (P.NElem + 1);
        _k = new double[_n, _n];
        _m = new double[_n, _n];
        double le = P.Length / P.NElem;
        double a = Math.PI * P.ShaftD * P.ShaftD / 4;
        double i = Math.PI * Math.Pow(P.ShaftD, 4) / 64;
        double ei = P.E * i;
        double c = ei / Math.Pow(le, 3);
        double[,] ke =
        {
            { 12, 6 * le, -12, 6 * le },
            { 6 * le, 4 * le * le, -6 * le, 2 * le * le },
            { -12, -6 * le, 12, -6 * le },
            { 6 * le, 2 * le * le, -6 * le, 4 * le * le },
        };
        double cm = P.Rho * a * le / 420;
        double[,] me =
        {
            { 156, 22 * le, 54, -13 * le },
            { 22 * le, 4 * le * le, 13 * le, -3 * le * le },
            { 54, 13 * le, 156, -22 * le },
            { -13 * le, -3 * le * le, -22 * le, 4 * le * le },
        };
        for (int e = 0; e < P.NElem; e++)
            for (int r = 0; r < 4; r++)
                for (int s = 0; s < 4; s++)
                {
                    _k[2 * e + r, 2 * e + s] += c * ke[r, s];
                    _m[2 * e + r, 2 * e + s] += cm * me[r, s];
                }
        _m[2 * P.DiskNode, 2 * P.DiskNode] += P.DiskMass;
    }

    /// <summary>Complex probe displacements [m] for unbalance <paramref name="unb"/> [kg m].</summary>
    public Complex[] Response(double k1, double k2, double rpm, Complex unb)
    {
        double w = 2 * Math.PI * rpm / 60.0;
        var d = new Complex[_n, _n];
        for (int r = 0; r < _n; r++)
            for (int s = 0; s < _n; s++)
                d[r, s] = _k[r, s] - w * w * _m[r, s];
        var ks = new[] { k1, k2 };
        for (int b = 0; b < 2; b++)
        {
            int dof = 2 * P.BearingNodes[b];
            d[dof, dof] += new Complex(ks[b], w * P.CBearing[b]);
        }
        var f = new Complex[_n];
        f[2 * P.DiskNode] = unb * (w * w);
        var q = SolveComplex(d, f);
        return new[] { q[2 * P.ProbeNodes[0]], q[2 * P.ProbeNodes[1]] };
    }

    public Complex[] UnitResponse(double k1, double k2, double rpm) => Response(k1, k2, rpm, Complex.One);

    /// <summary>(p1 amp um, p1 phase rad, p2 amp um, p2 phase rad)</summary>
    public (double A1, double Ph1, double A2, double Ph2) Probes(double k1, double k2, double rpm, Complex unb)
    {
        var x = Response(k1, k2, rpm, unb);
        return (x[0].Magnitude * 1e6, x[0].Phase, x[1].Magnitude * 1e6, x[1].Phase);
    }

    private static Complex[] SolveComplex(Complex[,] a, Complex[] b)
    {
        int n = b.Length;
        for (int col = 0; col < n; col++)
        {
            int piv = col;
            double best = a[col, col].Magnitude;
            for (int r = col + 1; r < n; r++)
                if (a[r, col].Magnitude > best) { best = a[r, col].Magnitude; piv = r; }
            if (best == 0) throw new InvalidOperationException("Singular dynamic stiffness matrix.");
            if (piv != col)
            {
                for (int s = 0; s < n; s++) (a[col, s], a[piv, s]) = (a[piv, s], a[col, s]);
                (b[col], b[piv]) = (b[piv], b[col]);
            }
            for (int r = col + 1; r < n; r++)
            {
                var fct = a[r, col] / a[col, col];
                if (fct == Complex.Zero) continue;
                for (int s = col; s < n; s++) a[r, s] -= fct * a[col, s];
                b[r] -= fct * b[col];
            }
        }
        var x = new Complex[n];
        for (int r = n - 1; r >= 0; r--)
        {
            var sum = b[r];
            for (int s = r + 1; s < n; s++) sum -= a[r, s] * x[s];
            x[r] = sum / a[r, r];
        }
        return x;
    }
}
