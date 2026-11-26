namespace DigitalTwin.Twin;

/// <summary>
/// Healthy bearing stiffness K0 by hand-calculation method (ME): Hertz point contact for one
/// ball between the two races (Hamrock and Brewe, NASA TM-83084, 1983), the bottom ball carrying
/// the bearing load, K = dQ/d(delta) = 1.5 W / delta. In series with the PLA ring, treated as a
/// solid ring loaded across its bore: k = E * d_bore * width / wall.
/// </summary>
public static class BearingStiffness
{
    // S6700: 11 balls of 1.2 mm, pitch diameter 12.45 mm (measured, COE firmware profile 3).
    public const double BallD = 1.2e-3, PitchD = 12.45e-3, GrooveRatio = 0.52;
    public const double ESteel = 206e9, NuSteel = 0.3;
    // PLA ring from the CAD file adapter_6f.step: OD 17.2 mm, bore 15.3 mm, width 3.4 mm.
    public const double EPla = 3.5e9, RingOd = 17.2e-3, RingBore = 15.3e-3, RingWidth = 3.4e-3;
    public const double G = 9.81;

    public static double DiskInertia(double mass, double radius, double width) =>
        mass * (3 * radius * radius + width * width) / 12.0;

    private static double ContactConstant(double rb, double rRace, double sign, double rGroove, double ePrime)
    {
        double rx = 1.0 / (1.0 / rb + sign / rRace), ry = 1.0 / (1.0 / rb - 1.0 / rGroove), a = ry / rx;
        double k = 1.0339 * Math.Pow(a, 0.636), eps = 1.0003 + 0.5968 / a, phi = 1.5277 + 0.6023 * Math.Log(a);
        double rp = 1.0 / (1.0 / rx + 1.0 / ry);
        return Math.PI * k * ePrime * Math.Sqrt(2 * eps * rp / 9.0) * Math.Pow(phi, -1.5);
    }

    /// <summary>Load-deflection constant Kn [N/m^1.5] of one ball (inner and outer contact in series).</summary>
    public static double BallConstant()
    {
        double ePrime = ESteel / (1 - NuSteel * NuSteel), rb = BallD / 2, rg = GrooveRatio * BallD;
        double ki = ContactConstant(rb, (PitchD - BallD) / 2, +1, rg, ePrime);
        double ko = ContactConstant(rb, (PitchD + BallD) / 2, -1, rg, ePrime);
        return Math.Pow(Math.Pow(ki, -2.0 / 3) + Math.Pow(ko, -2.0 / 3), -1.5);
    }

    /// <summary>Hertz stiffness of the bearing at radial load W [N].</summary>
    public static double Hertz(double load)
    {
        double delta = Math.Pow(load / BallConstant(), 2.0 / 3);
        return 1.5 * load / delta;
    }

    public static double Ring() => EPla * RingBore * RingWidth / ((RingOd - RingBore) / 2);

    public static double InSeries(double a, double b) => 1.0 / (1.0 / a + 1.0 / b);

    /// <summary>Static load on each bearing from the shaft and disk weights (sum of moments).</summary>
    public static double[] BearingLoads(double shaftD, double shaftLength, double rho, double[] bearings, DiskSpec[] disks)
    {
        double ws = rho * Math.PI * shaftD * shaftD / 4 * G;            // N per m of shaft
        double total = ws * shaftLength, moment = ws * shaftLength * shaftLength / 2;
        foreach (var d in disks) { total += d.MassKg * G; moment += d.MassKg * G * d.PositionM; }
        double r2 = (moment - total * bearings[0]) / (bearings[1] - bearings[0]);
        return new[] { total - r2, r2 };
    }

    /// <summary>K0 for both rig bearings (Hertz at each bearing's own load, PLA ring in series).</summary>
    public static double[] RigK0()
    {
        var disks = new[] { new DiskSpec(0.190, 0.800, 0), new DiskSpec(0.355, 0.800, 0) };
        var w = BearingLoads(0.0099, 0.550, 7850.0, new[] { 0.020, 0.530 }, disks);
        return new[] { InSeries(Hertz(w[0]), Ring()), InSeries(Hertz(w[1]), Ring()) };
    }
}
