using System.Numerics;
using DigitalTwin.Backend.Models;
using DigitalTwin.Twin;

namespace DigitalTwin.Backend.Services;

/// <summary>
/// ME FE digital twin on the live rig (owner: ME, A. Elshenawy). Each live window that carries valid
/// 1X probe readings (PROX 1, PROX 3) and a Keyphasor speed is fed to the FE twin
/// (DigitalTwin.Twin.FeBeamPhysicsTwin over the FE model of the RK4 rig). The first NCommission windows
/// calibrate the rig unbalance; after that each window identifies K1/K2 and the page shows the FE model
/// prediction next to the measured 1X amplitude.
/// Time base is the laptop clock (real hours), not a simulated table.
/// With the stiff S6700 bearings K is barely observable from shaft displacement, so the twin is expected
/// to keep K near K0 and give no physics RUL; the residual (measured vs model) is the live check.
/// </summary>
public sealed class LiveFeTwin
{
    private readonly object _gate = new();
    private FeBeamPhysicsTwin _twin = new();
    private readonly double _critRpm;
    private LiveTwinInfo? _last;

    public LiveFeTwin()
    {
        var k0 = _twin.Params.KNominal;
        _critRpm = _twin.Beam.FirstCriticalRpm(k0[0], k0[1]);
    }

    /// <summary>Drops the calibration (unbalance, baseline K). Call when the rig is rebuilt or rebalanced.</summary>
    public void Reset()
    {
        lock (_gate) { _twin = new(); _last = null; }
    }

    public LiveTwinInfo Update(LiveWindow w)
    {
        lock (_gate)
        {
            var k0 = _twin.Params.KNominal;
            var p = w.Probes;
            double? rpm = p?.MeasuredRpm;
            bool usable = p is not null && p.P1X1Valid && p.P3X1Valid && rpm is > 200
                && p.P1X1UmPp is > 0 && p.P3X1UmPp is > 0 && p.P1X1PhaseDeg is not null && p.P3X1PhaseDeg is not null;
            if (!usable)
            {
                string why = p is null ? "no probe data in this window (node firmware without probe values?)"
                    : rpm is null or <= 200 ? "no Keyphasor speed in this window"
                    : "1X probe reading not valid in this window";
                return _last is null
                    ? new LiveTwinInfo("waiting", why, 0, _twin.Params.NCommission, null, null, null, null, null, null, null,
                        null, null, k0[0], k0[1], null, null, _critRpm)
                    : _last with { Status = "holding", Note = why };
            }

            // Node: x1_um_pp is peak-to-peak and x1_phase_deg is a phase LAG from the Keyphasor event.
            // FE model: complex amplitude (zero-peak), x(t) = Re(X e^{jwt}), so phase = -lag.
            double a1 = p!.P1X1UmPp!.Value / 2, a3 = p.P3X1UmPp!.Value / 2;
            double ph1 = -p.P1X1PhaseDeg!.Value * Math.PI / 180, ph3 = -p.P3X1PhaseDeg!.Value * Math.PI / 180;
            var u = _twin.Update(rpm!.Value, a1, ph1, a3, ph3, DateTimeOffset.UtcNow);

            if (u.Commissioning || _twin.Unbalance is not Complex unb)
            {
                _last = new LiveTwinInfo("commissioning",
                    $"learning the rig unbalance from the first {_twin.Params.NCommission} windows",
                    _twin.CommissionCount, _twin.Params.NCommission, rpm, p.P1X1UmPp, p.P3X1UmPp, null, null, null,
                    null, null, null, k0[0], k0[1], null, null, _critRpm);
                return _last;
            }

            // Is K observable at this speed? (relative uncertainty of the identified K, 5 % probe noise)
            var meas = Identifier.ToComplex(a1, ph1, a3, ph3);
            var id = Identifier.Identify(_twin.Beam, rpm.Value, meas, unb);
            double relStd = Math.Max(id.RelStd1, id.RelStd2);
            bool observable = double.IsFinite(relStd) && relStd <= _twin.Params.MaxRelStd;
            double k1 = observable ? u.K1 ?? k0[0] : k0[0], k2 = observable ? u.K2 ?? k0[1] : k0[1];
            var pred = _twin.Beam.Response(k1, k2, rpm.Value, unb);
            double m1 = 2e6 * pred[0].Magnitude, m3 = 2e6 * pred[1].Magnitude;   // um pk-pk
            double r1 = (meas[0] - pred[0]).Magnitude / meas[0].Magnitude;
            double r3 = (meas[1] - pred[1]).Magnitude / meas[1].Magnitude;
            double resid = 100 * (r1 + r3) / 2;
            string note = !observable
                ? $"K cannot be identified from the probes at this speed (uncertainty about {100 * relStd:F0}%): the bearing is much stiffer than the shaft. Model uses the healthy K0."
                : u.PhysicsRulHours is null ? "K identified, not dropping: no physics RUL"
                : "K is dropping: physics RUL from the stiffness trend";
            _last = new LiveTwinInfo("running", note, _twin.Params.NCommission, _twin.Params.NCommission, rpm,
                p.P1X1UmPp, p.P3X1UmPp, m1, m3, resid, observable ? u.K1 : null, observable ? u.K2 : null,
                observable ? u.StiffnessDropPct : null, k0[0], k0[1],
                observable ? u.PhysicsRulHours : null, unb.Magnitude * 1e6, _critRpm);
            return _last;
        }
    }
}
