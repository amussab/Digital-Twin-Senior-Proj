using System.Numerics;

namespace DigitalTwin.Twin;

public sealed record TwinUpdate(
    double? K1, double? K2, double? StiffnessDropPct, double? PhysicsRulHours, bool Commissioning,
    double? RulLoHours = null, double? RulHiHours = null);

/// <summary>
/// INITIAL, REPLACEABLE physics twin (the ME member owns the final FE model). Stateful: the first
/// NCommission windows calibrate the unbalance (weighted linear LS at the nominal stiffness) and
/// the healthy baseline K0 (median); afterwards each window identifies K1/K2 and updates the
/// stiffness drop and the physics RUL. Port of aiengine/twin/physics_rul.py PhysicsTwin.
/// [SIMULATION]-verified only. Not thread-safe; use one instance per rotor.
/// </summary>
public sealed class FeBeamPhysicsTwin
{
    private readonly RotorParams _p;
    private readonly FeBeam _beam;
    private readonly List<(double Rpm, Complex[] Meas, double Th)> _buf = new();
    private readonly BearingTracker[] _trk;
    private Complex? _unb;
    private DateTimeOffset? _t0;

    public FeBeamPhysicsTwin(RotorParams? p = null)
    {
        _p = p ?? RotorParams.Default;
        _beam = new FeBeam(_p);
        _trk = new[] { new BearingTracker(_p), new BearingTracker(_p) };
    }

    public TwinUpdate Update(double rpm, double p1AmpUm, double p1PhaseRad, double p2AmpUm, double p2PhaseRad, DateTimeOffset t)
    {
        _t0 ??= t;
        double th = (t - _t0.Value).TotalHours;
        var meas = Identifier.ToComplex(p1AmpUm, p1PhaseRad, p2AmpUm, p2PhaseRad);
        if (_unb is null)
        {
            _buf.Add((rpm, meas, th));
            if (_buf.Count < _p.NCommission) return new TwinUpdate(null, null, null, null, true);
            _unb = CalibrateUnbalance();
            var ks = _buf.Select(w => Identifier.Identify(_beam, w.Rpm, w.Meas, _unb.Value)).ToList();
            var ok = Enumerable.Range(0, ks.Count).Where(i => Ok(ks[i])).ToList();
            if (ok.Count == 0) ok = Enumerable.Range(0, ks.Count).ToList();
            _trk[0].SetBaseline(ok.Select(i => ks[i].K1));
            _trk[1].SetBaseline(ok.Select(i => ks[i].K2));
            int j = ok[^1];
            _trk[0].Add(_buf[j].Th, ks[j].K1);
            _trk[1].Add(_buf[j].Th, ks[j].K2);
            return Out();
        }
        var res = Identifier.Identify(_beam, rpm, meas, _unb.Value);
        if (!Ok(res)) return Out();       // speed with too little K sensitivity: keep last estimate
        _trk[0].Add(th, res.K1);
        _trk[1].Add(th, res.K2);
        return Out();
    }

    private bool Ok(IdentResult r) => Math.Max(r.RelStd1, r.RelStd2) <= _p.MaxRelStd;

    private Complex CalibrateUnbalance()
    {
        Complex num = Complex.Zero;
        double den = 0;
        foreach (var (rpm, meas, _) in _buf)
        {
            var h = _beam.UnitResponse(_p.KNominal[0], _p.KNominal[1], rpm);
            for (int i = 0; i < 2; i++)
            {
                double w2 = 1.0 / (meas[i].Magnitude * meas[i].Magnitude);
                num += w2 * Complex.Conjugate(h[i]) * meas[i];
                den += w2 * h[i].Magnitude * h[i].Magnitude;
            }
        }
        return num / den;
    }

    private TwinUpdate Out()
    {
        double drop = 100.0 * Math.Max(_trk[0].Drop(), _trk[1].Drop());
        var ruls = _trk.Select(t => t.Rul()).Where(r => r is not null).Select(r => r!).ToList();
        RulEstimate? worst = ruls.Count == 0 ? null : ruls.MinBy(r => r.RulHours);
        return new TwinUpdate(_trk[0].K[^1], _trk[1].K[^1], drop, worst?.RulHours, false, worst?.LoHours, worst?.HiHours);
    }
}
