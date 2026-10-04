using System.Numerics;
using System.Text.Json;
using DigitalTwin.Twin;
using Xunit;

namespace DigitalTwin.Twin.Tests;

/// <summary>
/// Parity with the Python reference. Vectors come from AI-engine/reports/twin_golden.json
/// ([SIMULATION], written by `python -m aiengine.twin.demo_twin`).
/// </summary>
public class ParityTests
{
    private static JsonElement Golden()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "AI-engine", "reports", "twin_golden.json")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(dir!.FullName, "AI-engine", "reports", "twin_golden.json"))).RootElement;
    }

    private static double D(JsonElement e, string n) => e.GetProperty(n).GetDouble();
    private static double[] Arr(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static double Rel(double a, double b) => Math.Abs(a - b) / Math.Max(Math.Abs(b), 1e-300);

    [Fact]
    public void Params_match_python()
    {
        var p = Golden().GetProperty("params");
        var d = RotorParams.Default;
        Assert.Equal(D(p, "E"), d.E);
        Assert.Equal(D(p, "shaft_d"), d.ShaftD);
        Assert.Equal(D(p, "length"), d.Length);
        Assert.Equal((int)D(p, "n_elem"), d.NElem);
        Assert.Equal(D(p, "disk_mass"), d.DiskMass);
        Assert.Equal(Arr(p.GetProperty("k_nominal")), d.KNominal);
        Assert.Equal(Arr(p.GetProperty("c_bearing")), d.CBearing);
        Assert.Equal(D(p, "failure_drop"), d.FailureDrop);
        Assert.Equal((int)D(p, "n_commission"), d.NCommission);
        Assert.Equal(D(p, "max_rel_std"), d.MaxRelStd);
        Assert.Equal((int)D(p, "rul_fit_points"), d.RulFitPoints);
        Assert.Equal(Arr(p.GetProperty("bearing_nodes")).Select(x => (int)x), d.BearingNodes);
        Assert.Equal(Arr(p.GetProperty("probe_nodes")).Select(x => (int)x), d.ProbeNodes);
        Assert.Equal((int)D(p, "disk_node"), d.DiskNode);
    }

    [Fact]
    public void Forward_model_matches_python()
    {
        var beam = new FeBeam();
        foreach (var c in Golden().GetProperty("forward").EnumerateArray())
        {
            var exp = Arr(c.GetProperty("probes"));
            var got = beam.Probes(D(c, "k1"), D(c, "k2"), D(c, "rpm"), new Complex(D(c, "unb_re"), D(c, "unb_im")));
            Assert.True(Rel(got.A1, exp[0]) < 1e-6, $"A1 {got.A1} vs {exp[0]}");
            Assert.True(Rel(got.A2, exp[2]) < 1e-6);
            Assert.True(Math.Abs(got.Ph1 - exp[1]) < 1e-6);
            Assert.True(Math.Abs(got.Ph2 - exp[3]) < 1e-6);
        }
    }

    [Fact]
    public void Identification_matches_python()
    {
        var beam = new FeBeam();
        int n = 0;
        foreach (var c in Golden().GetProperty("identify").EnumerateArray())
        {
            var m = Arr(c.GetProperty("meas"));
            var r = Identifier.Identify(beam, D(c, "rpm"), Identifier.ToComplex(m[0], m[1], m[2], m[3]),
                new Complex(D(c, "unb_re"), D(c, "unb_im")));
            Assert.True(Rel(r.K1, D(c, "k1")) < 1e-3, $"K1 {r.K1} vs {D(c, "k1")}");
            Assert.True(Rel(r.K2, D(c, "k2")) < 1e-3, $"K2 {r.K2} vs {D(c, "k2")}");
            Assert.True(Rel(r.RelStd1, D(c, "rel_std1")) < 0.02);
            n++;
        }
        Assert.True(n >= 10);
    }

    [Fact]
    public void Twin_sequence_matches_python()
    {
        var twin = new FeBeamPhysicsTwin();
        var t0 = new DateTimeOffset(2026, 10, 5, 0, 0, 0, TimeSpan.Zero);
        int checkedRul = 0, steps = 0;
        foreach (var s in Golden().GetProperty("sequence").EnumerateArray())
        {
            var m = Arr(s.GetProperty("meas"));
            var u = twin.Update(D(s, "rpm"), m[0], m[1], m[2], m[3], t0.AddHours(D(s, "t_hours")));
            var o = s.GetProperty("out");
            Assert.Equal(o.GetProperty("commissioning").GetBoolean(), u.Commissioning);
            if (u.Commissioning) { Assert.Null(u.K1); continue; }
            Assert.True(Rel(u.K1!.Value, D(o, "k1")) < 1e-3, $"step {steps} K1");
            Assert.True(Rel(u.K2!.Value, D(o, "k2")) < 1e-3, $"step {steps} K2 {u.K2} vs {D(o, "k2")} K1 {u.K1} vs {D(o, "k1")}");
            Assert.True(Math.Abs(u.StiffnessDropPct!.Value - D(o, "drop_pct")) < 0.05, $"step {steps} drop");
            var rul = o.GetProperty("rul_h");
            if (rul.ValueKind == JsonValueKind.Null) Assert.Null(u.PhysicsRulHours);
            else
            {
                Assert.NotNull(u.PhysicsRulHours);
                Assert.True(Rel(u.PhysicsRulHours!.Value, rul.GetDouble()) < 0.01, $"step {steps} RUL {u.PhysicsRulHours} vs {rul.GetDouble()}");
                checkedRul++;
            }
            steps++;
        }
        Assert.True(steps > 100);
        Assert.True(checkedRul > 20);
    }
}
