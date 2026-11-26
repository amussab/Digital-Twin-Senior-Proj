using System.Numerics;
using DigitalTwin.Backend.Models;
using DigitalTwin.Backend.Services;
using DigitalTwin.Twin;
using Xunit;

namespace DigitalTwin.Backend.Tests;

/// <summary>ME FE twin on the live path, fed with probe readings made by the FE model itself
/// ([CALCULATED] input, no rig data): after commissioning the model must reproduce them.</summary>
public class LiveFeTwinTests
{
    private static LiveWindow Window(double rpm, Complex[] x)
    {
        // node convention: um pk-pk and a phase LAG in degrees, 0..360
        double Pp(Complex c) => 2e6 * c.Magnitude;
        double Lag(Complex c) { var d = -c.Phase * 180 / Math.PI; return d < 0 ? d + 360 : d; }
        return new LiveWindow
        {
            Features = new double[32], Rpm = rpm,
            Bearings = { new LiveBearing(1, "ok", "healthy") },
            Probes = new LiveProbes(rpm, 1000, Pp(x[0]), Lag(x[0]), true, 1000, Pp(x[1]), Lag(x[1]), true),
        };
    }

    [Fact]
    public void Waits_without_probe_data()
    {
        var t = new LiveFeTwin();
        var info = t.Update(new LiveWindow { Features = new double[32], Rpm = 1750 });
        Assert.Equal("waiting", info.Status);
        Assert.InRange(info.ModelCriticalRpm, 1450, 1570);
    }

    [Fact]
    public void Reproduces_model_generated_readings_after_commissioning()
    {
        var t = new LiveFeTwin();
        var beam = new FeBeam();
        var k0 = RotorParams.Default.KNominal;
        var unb = Complex.FromPolarCoordinates(10e-6, 0.7);           // 10 g mm
        var x = beam.Response(k0[0], k0[1], 1750, unb);
        LiveTwinInfo info = null!;
        for (int i = 0; i < RotorParams.Default.NCommission + 5; i++) info = t.Update(Window(1750, x));
        Assert.Equal("running", info.Status);
        Assert.NotNull(info.ResidualPct);
        Assert.True(info.ResidualPct < 1.0, $"residual {info.ResidualPct}%");
        Assert.InRange(info.UnbalanceGmm!.Value, 9.9, 10.1);
        Assert.InRange(info.ModelP1UmPp!.Value, 0.99 * info.MeasP1UmPp!.Value, 1.01 * info.MeasP1UmPp!.Value);
    }
}
