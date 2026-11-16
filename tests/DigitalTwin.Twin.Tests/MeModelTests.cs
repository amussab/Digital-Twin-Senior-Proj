using System.Numerics;
using DigitalTwin.Twin;
using Xunit;

namespace DigitalTwin.Twin.Tests;

/// <summary>
/// Checks of the ME rig model against hand calculations ([CALCULATED]; no rig data in these tests).
/// Reference numbers were computed independently in Python (numpy dense solver) on 2026-11-16.
/// </summary>
public class MeModelTests
{
    private static double Rel(double a, double b) => Math.Abs(a - b) / Math.Abs(b);

    [Fact]
    public void Static_deflection_matches_PL3_over_48EI()
    {
        // Rigid bearings, 10 N at mid-span: the overhangs carry no load, so the FE answer must equal
        // the simply supported beam formula over the 510 mm span exactly.
        var beam = new FeBeam();
        var p = beam.P;
        double span = p.BearingPositions[1] - p.BearingPositions[0];
        double ei = p.E * Math.PI * Math.Pow(p.ShaftD, 4) / 64;
        double hand = 10 * Math.Pow(span, 3) / (48 * ei);                    // 293.04 um
        double fe = beam.StaticDeflection(1e15, 1e15, beam.MidSpan, 10);
        Assert.True(Rel(fe, hand) < 1e-6, $"FE {fe * 1e6} um vs hand {hand * 1e6} um");
    }

    [Fact]
    public void First_critical_speed_lies_between_hand_bounds()
    {
        // Dunkerley (lower bound) 1473 rpm, Rayleigh with disk masses only (upper bound) 1606 rpm.
        var beam = new FeBeam();
        var k = beam.P.KNominal;
        double crit = beam.FirstCriticalRpm(k[0], k[1]);
        Assert.InRange(crit, 1473, 1606);
        Assert.InRange(crit, 1506, 1512);                                    // Python reference 1509 rpm
    }

    [Fact]
    public void Bearing_stiffness_matches_hand_calculation()
    {
        Assert.True(Rel(BearingStiffness.BallConstant(), 3.6281e9) < 1e-3);  // Kn, N/m^1.5
        Assert.True(Rel(BearingStiffness.Ring(), 191.7e6) < 1e-3);           // PLA ring, N/m
        var w = BearingStiffness.BearingLoads(0.0099, 0.550, 7850, new[] { 0.020, 0.530 }, RotorParams.Default.Disks);
        Assert.True(Rel(w[0], 9.555) < 1e-3 && Rel(w[1], 9.401) < 1e-3, $"loads {w[0]} {w[1]}");
        var k0 = BearingStiffness.RigK0();
        Assert.True(Rel(k0[0], 7.232e6) < 2e-3 && Rel(k0[1], 7.194e6) < 2e-3, $"K0 {k0[0]} {k0[1]}");
    }

    [Fact]
    public void Probe_amplitude_matches_reference_at_1750_rpm()
    {
        // 10 g.mm unbalance shared by the two disks: 17.17 / 17.70 um peak-to-peak (Python reference).
        var beam = new FeBeam();
        var k = beam.P.KNominal;
        var pr = beam.Probes(k[0], k[1], 1750, new Complex(1e-5, 0));
        Assert.True(Rel(2 * pr.A1, 17.17) < 2e-3, $"PROX 1 {2 * pr.A1} um pp");
        Assert.True(Rel(2 * pr.A2, 17.70) < 2e-3, $"PROX 3 {2 * pr.A2} um pp");
    }

    [Fact]
    public void Soft_bearing_stiffness_is_identified()
    {
        // Spec 2 style check on a soft bearing (stock RK4 journal bearing order, 45 kN/m):
        // noise-free probe data at K = 0.8 x nominal, unbalance known.
        var p = RotorParams.Default with { KNominal = new[] { 4.5e4, 4.5e4 } };
        var beam = new FeBeam(p);
        var unb = new Complex(1e-5, 0);
        var meas = beam.Response(0.8 * 4.5e4, 0.8 * 4.5e4, 1000, unb);
        var r = Identifier.Identify(beam, 1000, meas, unb);
        Assert.True(Rel(r.K1, 0.8 * 4.5e4) < 0.01 && Rel(r.K2, 0.8 * 4.5e4) < 0.01, $"K1 {r.K1} K2 {r.K2}");
        Assert.True(r.RelStd1 < p.MaxRelStd, $"RelStd1 {r.RelStd1}");
    }

    [Fact]
    public void Stiff_rig_bearing_is_flagged_as_not_identifiable()
    {
        // With the S6700 (about 7 MN/m) the shaft sets the probe reading: 5 % probe noise gives
        // several hundred percent uncertainty on K, so the twin must not trust a single K estimate.
        var beam = new FeBeam();
        var k = beam.P.KNominal;
        var unb = new Complex(1e-5, 0);
        var r = Identifier.Identify(beam, 1750, beam.Response(k[0], k[1], 1750, unb), unb);
        Assert.True(r.RelStd1 > 1.0, $"RelStd1 {r.RelStd1}");                 // Python reference 6.6
    }

    [Fact]
    public void Twin_commissions_and_holds_K0_on_healthy_data()
    {
        var twin = new FeBeamPhysicsTwin();
        var beam = new FeBeam();
        var k = beam.P.KNominal;
        var unb = Complex.FromPolarCoordinates(1e-5, 0.5);
        var t0 = new DateTimeOffset(2026, 11, 16, 0, 0, 0, TimeSpan.Zero);
        TwinUpdate? u = null;
        for (int i = 0; i < 30; i++)
        {
            var pr = beam.Probes(k[0], k[1], 1750, unb);
            u = twin.Update(1750, pr.A1, pr.Ph1, pr.A2, pr.Ph2, t0.AddMinutes(i));
            if (i < beam.P.NCommission - 1) Assert.True(u.Commissioning);
        }
        Assert.False(u!.Commissioning);
        Assert.NotNull(u.K1);
        Assert.True(Rel(u.K1!.Value, k[0]) < 0.01, $"K1 {u.K1}");
        Assert.Null(u.PhysicsRulHours);                                      // no downward trend, no RUL
    }
}
