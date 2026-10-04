"""Physics RUL from identified bearing stiffness K(t), plus the stateful PhysicsTwin.

Degradation d = 1 - K/K0. Failure when d >= failure_drop (25 %) [TEAM DOC].
RUL: least-squares fit of ln(K/K0) = a + b t (exponential stiffness decay, monotone for b<0) over
the most recent points, extrapolated to ln(1 - failure_drop). Interval from the slope standard
error (+/- 1.96 se). Returns None until enough points exist or unless the fitted trend is significantly
decreasing (slope + 1.96 se < 0).
"""
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
from .fe_beam import FeBeam
from .identify import identify, to_complex
from .params import RotorParams, DEFAULT


@dataclass
class RulEstimate:
    rul_hours: float
    lo_hours: float      # optimistic bound is hi, pessimistic is lo (slope +/- 1.96 se)
    hi_hours: float


def fit_rul(t, k, k0, p: RotorParams = DEFAULT, now=None) -> Optional[RulEstimate]:
    """t [h], k [N/m] arrays of recent points (time ordered)."""
    t = np.asarray(t, float)
    y = np.log(np.asarray(k, float) / k0)
    n = len(t)
    if n < p.min_rul_points or np.ptp(t) <= 0:
        return None
    tm, ym = t.mean(), y.mean()
    sxx = float(np.sum((t - tm) ** 2))
    b = float(np.sum((t - tm) * (y - ym)) / sxx)
    a = ym - b * tm
    if b >= 0:
        return None
    res = y - (a + b * t)
    se = math.sqrt(float(np.sum(res ** 2)) / max(n - 2, 1) / sxx)
    if b + 1.96 * se >= 0:       # trend not significantly decreasing at ~95 %: no RUL
        return None
    y_fail = math.log(1.0 - p.failure_drop)
    t_now = t[-1] if now is None else now

    def rul(slope):
        # crossing time on the line through the fitted centroid with the given slope
        t_cross = tm + (y_fail - ym) / slope
        return max(t_cross - t_now, 0.0)

    return RulEstimate(rul(b), rul(b - 1.96 * se), rul(b + 1.96 * se))


class BearingTracker:
    """Per-bearing K(t) history, baseline K0 and RUL."""

    def __init__(self, p: RotorParams = DEFAULT):
        self.p = p
        self.t, self.k = [], []
        self.k0 = None

    def set_baseline(self, ks):
        self.k0 = float(np.median(ks))

    def add(self, t_h, k):
        self.t.append(float(t_h))
        self.k.append(float(k))

    def drop(self, k=None):
        k = self.k[-1] if k is None else k
        return 1.0 - k / self.k0

    def rul(self) -> Optional[RulEstimate]:
        n = self.p.rul_fit_points
        return fit_rul(self.t[-n:], self.k[-n:], self.k0, self.p)


@dataclass
class TwinUpdate:
    k1: Optional[float]
    k2: Optional[float]
    stiffness_drop_pct: Optional[float]      # worst bearing
    physics_rul_hours: Optional[float]       # min over bearings
    commissioning: bool
    rul_lo_hours: Optional[float] = None
    rul_hi_hours: Optional[float] = None


class PhysicsTwin:
    """Stateful twin. Mirrors C# FeBeamPhysicsTwin.

    First n_commission windows: commissioning. The unbalance (complex, kg m) is calibrated by
    weighted linear least squares assuming the NOMINAL bearing stiffness (the known-stiffness
    reference), then each buffered window is identified and K0 = median per bearing.
    """

    def __init__(self, p: RotorParams = DEFAULT):
        self.p = p
        self.beam = FeBeam(p)
        self.buf = []
        self.unb = None
        self.trk = [BearingTracker(p), BearingTracker(p)]

    @property
    def commissioning(self):
        return self.unb is None

    def calibrate_unbalance(self, windows):
        """windows: list of (rpm, meas_complex(2,)). Weighted LS, weights 1/|meas|."""
        num = 0j
        den = 0.0
        kn = self.p.k_nominal
        for rpm, meas in windows:
            h = self.beam.unit_response(kn[0], kn[1], rpm)
            w2 = 1.0 / np.abs(meas) ** 2
            num += np.sum(w2 * np.conj(h) * meas)
            den += np.sum(w2 * np.abs(h) ** 2)
        return complex(num / den)

    def update(self, rpm, a1, ph1, a2, ph2, t_hours) -> TwinUpdate:
        meas = to_complex(a1, ph1, a2, ph2)
        if self.unb is None:
            self.buf.append((float(rpm), meas, float(t_hours)))
            if len(self.buf) < self.p.n_commission:
                return TwinUpdate(None, None, None, None, True)
            self.unb = self.calibrate_unbalance([(r, m) for r, m, _ in self.buf])
            ks = [identify(self.beam, r, m, self.unb) for r, m, _ in self.buf]
            ok = [i for i, x in enumerate(ks) if self._ok(x)] or list(range(len(ks)))
            self.trk[0].set_baseline([ks[i].k1 for i in ok])
            self.trk[1].set_baseline([ks[i].k2 for i in ok])
            # baseline windows are not part of the degradation trend; the last usable one seeds the output
            j = ok[-1]
            self.trk[0].add(self.buf[j][2], ks[j].k1)
            self.trk[1].add(self.buf[j][2], ks[j].k2)
            return self._out(False)
        res = identify(self.beam, rpm, meas, self.unb)
        if not self._ok(res):          # speed with too little K sensitivity: keep last estimate
            return self._out(False)
        self.trk[0].add(t_hours, res.k1)
        self.trk[1].add(t_hours, res.k2)
        return self._out(False)

    def _ok(self, x):
        return max(x.rel_std1, x.rel_std2) <= self.p.max_rel_std

    def _out(self, comm):
        t0, t1 = self.trk
        drops = [t0.drop(), t1.drop()]
        ruls = [t0.rul(), t1.rul()]
        ruls = [r for r in ruls if r is not None]
        worst = min(ruls, key=lambda r: r.rul_hours) if ruls else None
        return TwinUpdate(t0.k[-1], t1.k[-1], 100.0 * max(drops),
                          None if worst is None else worst.rul_hours, comm,
                          None if worst is None else worst.lo_hours,
                          None if worst is None else worst.hi_hours)
