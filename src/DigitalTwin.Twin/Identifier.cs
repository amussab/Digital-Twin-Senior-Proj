using System.Numerics;

namespace DigitalTwin.Twin;

public sealed record IdentResult(double K1, double K2, double RelStd1, double RelStd2, double Cost, bool Converged);

/// <summary>
/// Inverse problem: bearing stiffness K1, K2 from measured probe 1x response. Log-parameterised
/// nonlinear least squares (Levenberg-Marquardt, 2 unknowns, 4 residuals), multi-start. Port of
/// aiengine/twin/identify.py.
/// </summary>
public static class Identifier
{
    private static readonly double[] StartScales = { 1.0, 0.5, 2.0, 0.25, 4.0 };

    public static Complex[] ToComplex(double a1Um, double ph1, double a2Um, double ph2) =>
        new[] { Complex.FromPolarCoordinates(a1Um * 1e-6, ph1), Complex.FromPolarCoordinates(a2Um * 1e-6, ph2) };

    private static double[] Residual(FeBeam beam, double rpm, Complex unb, Complex[] meas, double[] kRef, double t0, double t1)
    {
        var pred = beam.Response(kRef[0] * Math.Exp(t0), kRef[1] * Math.Exp(t1), rpm, unb);
        var r = new double[4];
        for (int i = 0; i < 2; i++)
        {
            var d = (pred[i] - meas[i]) / meas[i].Magnitude;
            r[i] = d.Real;
            r[2 + i] = d.Imaginary;
        }
        return r;
    }

    private static double Sq(double[] r) { double s = 0; foreach (var v in r) s += v * v; return s; }

    private static double[,] Jac(FeBeam beam, double rpm, Complex unb, Complex[] meas, double[] kRef, double t0, double t1)
    {
        const double h = 1e-6;
        var j = new double[4, 2];
        var rp0 = Residual(beam, rpm, unb, meas, kRef, t0 + h, t1);
        var rm0 = Residual(beam, rpm, unb, meas, kRef, t0 - h, t1);
        var rp1 = Residual(beam, rpm, unb, meas, kRef, t0, t1 + h);
        var rm1 = Residual(beam, rpm, unb, meas, kRef, t0, t1 - h);
        for (int i = 0; i < 4; i++)
        {
            j[i, 0] = (rp0[i] - rm0[i]) / (2 * h);
            j[i, 1] = (rp1[i] - rm1[i]) / (2 * h);
        }
        return j;
    }

    private static (double T0, double T1, double Cost, bool Ok) Lm(FeBeam beam, double rpm, Complex unb, Complex[] meas, double[] kRef, double s)
    {
        double t0 = s, t1 = s;
        double lambda = 1e-3;
        double cost = Sq(Residual(beam, rpm, unb, meas, kRef, t0, t1));
        bool ok = false;
        for (int it = 0; it < 200; it++)
        {
            var r = Residual(beam, rpm, unb, meas, kRef, t0, t1);
            var j = Jac(beam, rpm, unb, meas, kRef, t0, t1);
            double a00 = 0, a01 = 0, a11 = 0, g0 = 0, g1 = 0;
            for (int i = 0; i < 4; i++)
            {
                a00 += j[i, 0] * j[i, 0]; a01 += j[i, 0] * j[i, 1]; a11 += j[i, 1] * j[i, 1];
                g0 += j[i, 0] * r[i]; g1 += j[i, 1] * r[i];
            }
            if (Math.Abs(g0) + Math.Abs(g1) < 1e-14) { ok = true; break; }
            bool stepped = false;
            for (int tries = 0; tries < 30; tries++)
            {
                double b00 = a00 + lambda * Math.Max(a00, 1e-12), b11 = a11 + lambda * Math.Max(a11, 1e-12);
                double det = b00 * b11 - a01 * a01;
                if (det == 0) { lambda *= 10; continue; }
                double d0 = -(b11 * g0 - a01 * g1) / det;
                double d1 = -(b00 * g1 - a01 * g0) / det;
                double nc = Sq(Residual(beam, rpm, unb, meas, kRef, t0 + d0, t1 + d1));
                if (nc < cost)
                {
                    bool small = Math.Abs(d0) + Math.Abs(d1) < 1e-12 || (cost - nc) < 1e-18 * Math.Max(cost, 1e-300);
                    t0 += d0; t1 += d1; cost = nc; lambda = Math.Max(lambda / 10, 1e-12); stepped = true;
                    if (small) ok = true;
                    break;
                }
                lambda *= 10;
            }
            if (!stepped) { ok = true; break; }      // no descent direction: at a (local) minimum
            if (ok) break;
        }
        return (t0, t1, cost, ok);
    }

    public static IdentResult Identify(FeBeam beam, double rpm, Complex[] meas, Complex unb, double[]? kStart = null, double? relNoise = null)
    {
        var kRef = kStart ?? beam.P.KNominal;
        double sig = relNoise ?? beam.P.RelNoise;
        (double T0, double T1, double Cost, bool Ok) best = (0, 0, double.PositiveInfinity, false);
        foreach (var sc in StartScales)
        {
            var sol = Lm(beam, rpm, unb, meas, kRef, Math.Log(sc));
            if (sol.Cost < best.Cost) best = sol;
            if (best.Cost < 2e-14) break;
        }
        var j = Jac(beam, rpm, unb, meas, kRef, best.T0, best.T1);
        double a00 = 0, a01 = 0, a11 = 0;
        for (int i = 0; i < 4; i++) { a00 += j[i, 0] * j[i, 0]; a01 += j[i, 0] * j[i, 1]; a11 += j[i, 1] * j[i, 1]; }
        double det = a00 * a11 - a01 * a01;
        double s1 = double.PositiveInfinity, s2 = double.PositiveInfinity;
        if (det > 0)
        {
            double scale = sig * sig / 2.0;
            s1 = Math.Sqrt(Math.Max(a11 / det * scale, 0));
            s2 = Math.Sqrt(Math.Max(a00 / det * scale, 0));
        }
        return new IdentResult(kRef[0] * Math.Exp(best.T0), kRef[1] * Math.Exp(best.T1), s1, s2, best.Cost, best.Ok);
    }
}
