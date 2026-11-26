"""FeaturePhysicsTwin: damage-accumulation digital twin that reads NODE FEATURES ONLY (no proximity amp/phase).

DESIGN NOTE (read before citing any number from this module)
------------------------------------------------------------
What is physics / cited law
  * ISO 281 rating life: L10h = 1e6 / (60 n) * (C/P)^p with p = 3 for ball bearings  [CITED: ISO 281]. Life in hours is
    inversely proportional to speed n [rpm] at fixed load ratio C/P. Used here only through `LifeRef`.
  * Miner linear damage accumulation, dD/dt = r^p / L_ref(rpm), failure at D >= 0.95  [TEAM SPEC M6]. r is a
    dimensionless severity multiplier (r = 1 -> the bearing consumes exactly its rated life; applying the ISO 281
    load-life exponent p = 3 to the defect-energy ratio is a MODELLING CHOICE, see below).
  * Paris-type growth: ln r(t) = a + b t (exponential growth of the defect indicator), fitted by least squares over the most
    recent points with a significance gate (b - 1.96 se > 0), the same structure as physics_rul.fit_rul  [DESIGN CHOICE].

What is assumption
  * C/P (basic dynamic load rating over equivalent load) of the rig bearing is NOT known here: `LifeRef.c_over_p` defaults to
    10.0 and is tagged [ASSUMED]. Supply the catalogue C and the real load to replace it (then tag [CITED]).
  * r_fail = 20 is tied to the health-stage-6 level of aiengine.features (HI_R_REF = 20, where the health index saturates)
    [ASSUMED link; the 20 is a design constant of features.py, not a measured failure amplitude].
  * That the RMS-of-ratios of envelope/order-band energy is a monotone proxy of defect size, and that damage rate scales as
    r^3 on it, are modelling assumptions, not validated on run-to-failure rig data.

Common-mode with the AI (honest limit)
  The twin and the AI RUL head read the SAME 16 node features of the same window. Agreement between their RULs (IS2) therefore
  measures consistency of two readings of one signal and a shared baseline, not independent physical confirmation. They differ
  in model class (log-linear trend + Miner law vs learned head) but not in information. Report IS2 on this twin as such.

r_t (C#-portable: arithmetic + the stored 16-value baseline, no fitted scalers)
  For axis a in {X, Y} with features f (8 per axis, COE C7 order bp_rms, kurtosis, crest, env_rms, ftf, bsf, bpfo, bpfi) and
  baseline b (median of the commissioning windows):
      x_k = f_k / (b_k + EPS)          for k in {env_rms, ftf, bsf, bpfo, bpfi}
      r_a = sqrt( mean_k x_k^2 )       (RMS of the five energy ratios; a healthy window gives about 1)
      r_t = max(r_X, r_Y)
  health_index = clip( ln(max(r_t,1)) / ln(r_fail), 0, 1 ).
RUL outputs: rul_trend (time for the fitted ln r line to reach ln r_fail) and rul_miner (closed form: remaining damage
(0.95 - D) integrated along the fitted exponential, L_ref at the current rpm); rul_hours = min of the two. Prior (no
significant trend): (0.95 - D) * L_ref(rpm) / max(r_t,1)^p, i.e. rated life remaining at the current severity.
Deterministic: no RNG, no wall clock.

ROUND 2 (2026-11-15): BOUNDED DAMAGE LINK. r is a VIBRATION energy ratio, not an ISO 281 load ratio, so applying p = 3 to r directly
(law="power", kept for the record) saturated Miner D at real fault amplitudes (r 5-19 -> r^3 = 125-6859). The default law is now
    g(r) = 1 + k ln(max(r, 1))            equivalent-load factor P_eq/P0 (log-like, bounded growth)   [ASSUMED modelling law]
    dD/dt = g(r)^p / L_ref(rpm),  p = 3 on the LOAD factor  [CITED: ISO 281]
and the RUL extrapolation uses the same law in closed form along the fitted line y = ln r = y0 + b s:
    D_rem = 0.95 - D = ((g0 + k b s)^(p+1) - g0^(p+1)) / ((p+1) k b L_ref)   ->  s = ((g0^(p+1) + (p+1) k b L_ref D_rem)^(1/(p+1)) - g0)/(k b)
k and r_fail are fitted to the SIMULATION GENERATOR on rig_anchor VAL units only (reports/twin_feature_calibration_*.json, loaded by
default). That is a fit to the generator: the law is a modelling assumption and IS2 on it stays common-mode (Rank <= 6).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EPS = 1e-9
ENERGY_IDX = (3, 4, 5, 6, 7)     # env_rms, ftf, bsf, bpfo, bpfi within one axis' 8-vector
D_FAIL = 0.95                    # Miner failure level [TEAM SPEC M6]


@dataclass
class LifeRef:
    """Rated life L_ref(rpm) [h]. Either an rpm->hours table (interpolated, clamped) or ISO 281 L10h from C/P."""
    c_over_p: float = 10.0               # [ASSUMED] catalogue C and real load P not available
    p: float = 3.0                       # [CITED: ISO 281, ball bearings]
    table: Optional[tuple] = None        # ((rpm, hours), ...) sorted by rpm; overrides the formula when given
    provenance: str = "ASSUMED"          # tag of c_over_p / table

    def hours(self, rpm: float) -> float:
        rpm = max(float(rpm), 1.0)
        if self.table:
            xs = [a for a, _ in self.table]
            ys = [b for _, b in self.table]
            return float(np.interp(rpm, xs, ys))
        return 1e6 / (60.0 * rpm) * self.c_over_p ** self.p


FALLBACK_K = 1.0                         # [ASSUMED] used only when no calibration file exists
FALLBACK_R_FAIL = 20.0                   # [ASSUMED] ties to features.HI_R_REF
CAL_DIR = Path(__file__).resolve().parents[2] / "reports"


def load_calibration(path=None) -> dict:
    """Latest reports/twin_feature_calibration_*.json (or `path`) -> {"k", "r_fail", ...}; fallback constants if none."""
    fs = [Path(path)] if path else sorted(CAL_DIR.glob("twin_feature_calibration_*.json"))
    if not fs:
        return {"k": FALLBACK_K, "r_fail": FALLBACK_R_FAIL, "source": "fallback [ASSUMED]"}
    d = json.loads(fs[-1].read_text())
    return {"k": float(d["k"]), "r_fail": float(d["r_fail"]), "source": fs[-1].name}


def _default_k() -> float:
    return load_calibration()["k"]


def _default_r_fail() -> float:
    return load_calibration()["r_fail"]


@dataclass
class FeaturePhysicsParams:
    life: LifeRef = field(default_factory=LifeRef)
    r_fail: float = field(default_factory=_default_r_fail)   # [CALIBRATED on sim val / ASSUMED] stage-6 level of r
    k: float = field(default_factory=_default_k)             # load-factor slope in g(r) = 1 + k ln r  [CALIBRATED on sim val]
    law: str = "bounded"                 # "bounded" (default) | "power" (round-1 g = r, kept for the record)
    n_commission: int = 24               # windows used for the baseline (features.BASELINE_WINDOWS) [DESIGN]
    min_points: int = 8                  # points before a trend RUL is issued [DESIGN, = physics_rul.min_rul_points]
    fit_points: int = 200                # recent points in the trend fit [DESIGN]
    z: float = 1.96                      # significance / interval half-width [DESIGN]


@dataclass
class TwinFeatureUpdate:
    rul_hours: Optional[float]
    lo: Optional[float]                  # pessimistic bound (slope + z se); None for prior/commissioning
    hi: Optional[float]                  # optimistic bound (slope - z se)
    miner_d: float
    health_index: float
    source: str                          # 'trend' | 'prior' | 'commissioning'
    r: float = 1.0
    rul_trend_h: Optional[float] = None
    rul_miner_h: Optional[float] = None


def defect_energy_ratio(raw16, baseline16) -> float:
    """r_t as documented in the module docstring."""
    f = np.asarray(raw16, float).reshape(16)
    b = np.asarray(baseline16, float).reshape(16)
    best = 0.0
    for a in (0, 8):
        s = 0.0
        for k in ENERGY_IDX:
            x = f[a + k] / (b[a + k] + EPS)
            s += x * x
        best = max(best, math.sqrt(s / len(ENERGY_IDX)))
    return best


def load_factor(r: float, p: "FeaturePhysicsParams") -> float:
    r1 = max(r, 1.0)
    return r1 if p.law == "power" else 1.0 + p.k * math.log(r1)


class FeaturePhysicsTwin:
    """Stateful twin for ONE bearing (16 node features per window)."""

    def __init__(self, params: Optional[FeaturePhysicsParams] = None, baseline16=None):
        self.p = params or FeaturePhysicsParams()
        self.base = None if baseline16 is None else np.asarray(baseline16, float).reshape(16).copy()
        self._buf: list = []
        self.t: list = []
        self.y: list = []
        self.d = 0.0
        self._t_prev: Optional[float] = None
        self.counts = {"trend": 0, "prior": 0, "commissioning": 0}

    @property
    def commissioning(self) -> bool:
        return self.base is None

    def set_baseline(self, baseline16) -> None:
        self.base = np.asarray(baseline16, float).reshape(16).copy()

    def reset(self) -> None:
        self.__init__(self.p, None)

    def update(self, features, rpm: float, t_hours: float) -> TwinFeatureUpdate:
        f = np.asarray(features, float).reshape(16)
        life = self.p.life.hours(rpm)
        if self.base is None:
            self._buf.append(f)
            if len(self._buf) >= self.p.n_commission:
                self.base = np.median(np.stack(self._buf), axis=0)
                self._buf = []
            self._t_prev = float(t_hours)
            self.counts["commissioning"] += 1
            return TwinFeatureUpdate(D_FAIL * life, None, None, 0.0, 0.0, "commissioning")
        r = defect_energy_ratio(f, self.base)
        t = float(t_hours)
        pw = self.p.life.p
        if self._t_prev is not None and t > self._t_prev:
            self.d += load_factor(r, self.p) ** pw * (t - self._t_prev) / life
        self._t_prev = t
        self.t.append(t)
        self.y.append(math.log(max(r, EPS)))
        n = self.p.fit_points
        self.t, self.y = self.t[-n:], self.y[-n:]
        hi_ = min(max(math.log(max(r, 1.0)) / math.log(self.p.r_fail), 0.0), 1.0)
        prior = max(D_FAIL - self.d, 0.0) * life / load_factor(r, self.p) ** pw
        fit = self._fit()
        if fit is None:
            self.counts["prior"] += 1
            return TwinFeatureUpdate(prior, None, None, self.d, hi_, "prior", r)
        tm, ym, b, se = fit
        yf = math.log(self.p.r_fail)
        y_now = ym + b * (t - tm)               # fitted level now (smooths the noisy last point)

        def trend(slope):
            return max((yf - y_now) / slope, 0.0)

        def miner(slope):
            rem = max(D_FAIL - self.d, 0.0)
            if self.p.law == "power":
                r0 = math.exp(y_now)
                return math.log1p(pw * slope * life * rem / r0 ** pw) / (pw * slope)
            g0 = 1.0 + self.p.k * max(y_now, 0.0)
            kb = self.p.k * slope
            if kb < 1e-12:                       # k = 0: constant load factor, D grows linearly
                return rem * life / g0 ** pw
            return ((g0 ** (pw + 1) + (pw + 1) * kb * life * rem) ** (1.0 / (pw + 1)) - g0) / kb

        def both(slope):
            return min(trend(slope), miner(slope))

        self.counts["trend"] += 1
        lo = both(b + self.p.z * se)
        hi = both(max(b - self.p.z * se, 1e-12))
        return TwinFeatureUpdate(both(b), lo, hi, self.d, hi_, "trend", r, trend(b), miner(b))

    def _fit(self):
        n = len(self.t)
        if n < self.p.min_points:
            return None
        t = np.asarray(self.t)
        y = np.asarray(self.y)
        if np.ptp(t) <= 0:
            return None
        tm, ym = float(t.mean()), float(y.mean())
        sxx = float(np.sum((t - tm) ** 2))
        b = float(np.sum((t - tm) * (y - ym)) / sxx)
        res = y - (ym + b * (t - tm))
        se = math.sqrt(float(np.sum(res ** 2)) / max(n - 2, 1) / sxx)
        if b - self.p.z * se <= 0:               # not significantly growing at ~95 %: no trend RUL
            return None
        return tm, ym, b, se

    def trend_fraction(self) -> float:
        n = self.counts["trend"] + self.counts["prior"]
        return self.counts["trend"] / n if n else 0.0


class FeaturePhysicsBank:
    """Two bearings from a 32-value vector (slots 0 and 1, 16 each); worst-case (min RUL) summary plus both updates."""

    def __init__(self, params: Optional[FeaturePhysicsParams] = None):
        self.twins = [FeaturePhysicsTwin(params), FeaturePhysicsTwin(params)]

    def update(self, features32, rpm: float, t_hours: float):
        f = np.asarray(features32, float).reshape(32)
        ups = [self.twins[i].update(f[16 * i:16 * i + 16], rpm, t_hours) for i in (0, 1)]
        worst = min(ups, key=lambda u: float("inf") if u.rul_hours is None else u.rul_hours)
        return worst, ups
