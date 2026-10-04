"""N-HiTS health-index forecast -> RUL (hours), with TFT-class-conditioned failure thresholds.

N-HiTS forecasts the observable health index (HI) `prediction_length` steps ahead. RUL is then
solved for, never regressed from a RUL history (which is never observable on a live machine):

  trend      Exponential trend fitted through the recent HI history + the N-HiTS forecast,
             extrapolated to the failure threshold of the TFT-predicted class.
  onset      Time-since-onset scaling. With the causal onset confirmed tau hours ago and the
             forecast-smoothed HI giving the fraction of degradation consumed
             u = (HI - HI_onset) / (thr - HI_onset), RUL = tau * (1 - u) / u.
  blend      w * trend + (1 - w) * onset, w fitted on training bearings.

All estimators are causal (they use only windows up to now) and C#-portable (a weighted
least-squares line and a few divisions). The method and its few parameters are chosen on
training/validation bearings only (train.select_rul) and serialised into model_contract.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

EPS = 1e-9
HI_FLOOR = 0.02


@dataclass
class RULCalibration:
    method: str = "trend"
    history_windows: int = 24
    blend_w: float = 0.5
    u_min: float = 0.05              # floor on the consumed fraction in the onset estimator
    class_thresholds: dict = field(default_factory=dict)
    pooled_threshold: float = 1.0
    max_rul_factor: float = 20.0      # cap: RUL <= factor * (tau or history span)
    loglin_coef: list = field(default_factory=list)   # method "loglin": log RUL = coef . features

    def threshold_for(self, cls: str | None) -> float:
        if cls in (None, "healthy"):
            return self.pooled_threshold
        return float(self.class_thresholds.get(cls, self.pooled_threshold))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _none(v):
    """NaN (e.g. a None that went through a DataFrame) -> None."""
    if v is None:
        return None
    try:
        return None if not np.isfinite(float(v)) else float(v)
    except (TypeError, ValueError):
        return None


def fit_thresholds(units: list[dict], quantile: float = 0.5) -> tuple[dict, float]:
    """units: [{'fault_class', 'hi_end'}] from TRAINING run-to-failure bearings only."""
    by = {}
    for u in units:
        by.setdefault(u["fault_class"], []).append(u["hi_end"])
    allv = [v for vs in by.values() for v in vs]
    pooled = float(np.quantile(allv, quantile)) if allv else 1.0
    th = {c: float(np.quantile(v, quantile)) for c, v in by.items()
          if c in ("outer_race", "inner_race", "ball", "cage")}
    return th, pooled


def end_of_life_hi(hi: np.ndarray, last: int = 3) -> float:
    return float(np.median(np.asarray(hi)[-last:]))


def trend_steps(series: np.ndarray, threshold: float, now_index: int | None = None) -> float:
    """Steps from `now_index` (default: last point) until an exponential trend fitted to
    `series` reaches threshold."""
    y = np.log(np.maximum(np.asarray(series, float), HI_FLOOR))
    n = len(y)
    if n < 3:
        return np.inf
    x = np.arange(n, dtype=float)
    w = np.exp(np.linspace(-1.5, 0.0, n))
    sw = w.sum()
    xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
    vx = (w * (x - xm) ** 2).sum()
    if vx <= EPS:
        return np.inf
    slope = (w * (x - xm) * (y - ym)).sum() / vx
    if slope <= 1e-6:
        return np.inf
    now = x[-1] if now_index is None else float(now_index)
    level = ym + slope * (now - xm)                  # fitted value at "now"
    return max((np.log(max(threshold, HI_FLOOR)) - level) / slope, 0.0)


def estimate(calib: RULCalibration, hi_hist: np.ndarray, forecast: np.ndarray, cls: str | None,
             tau_hours: float | None, hi_onset: float | None, cadence_hours: float) -> dict:
    """RUL (hours) for the current window.

    hi_hist: HI up to and including the current window; forecast: N-HiTS HI forecast for the
    next steps (the forecast's last point sits `prediction_length` steps ahead).
    """
    tau_hours, hi_onset = _none(tau_hours), _none(hi_onset)
    thr = calib.threshold_for(cls)
    hist = np.asarray(hi_hist, float)[-calib.history_windows:]
    fc = np.asarray(forecast, float)
    series = np.concatenate([hist, fc])
    steps = trend_steps(series, thr, now_index=len(hist) - 1)
    trend_h = steps * cadence_hours if np.isfinite(steps) else np.inf

    onset_h = np.inf
    u = np.nan
    u_raw = np.nan
    hi_now = float(np.mean(np.concatenate([hist[-3:], fc[:3]])))
    if tau_hours is not None and hi_onset is not None and tau_hours > 0:
        u_raw = (hi_now - hi_onset) / max(thr - hi_onset, 1e-3)
        u = float(np.clip(u_raw, calib.u_min, 0.999))
        onset_h = tau_hours * (1 - u) / u

    span_h = max(tau_hours or 0.0, len(hist) * cadence_hours)
    cap = calib.max_rul_factor * span_h
    trend_h, onset_h = min(trend_h, cap), min(onset_h, cap)

    llf = None
    if tau_hours is not None and tau_hours > 0:
        llf = loglin_features(tau_hours, trend_h, u_raw, hi_now / max(thr, 1e-3), cadence_hours)
    if calib.method == "loglin":
        if llf is not None and calib.loglin_coef:
            rul = float(np.exp(np.clip(np.dot(calib.loglin_coef, llf), -20, 20)))
            rul = min(rul, cap)
        else:
            rul = trend_h
    elif calib.method == "trend":
        rul = trend_h if np.isfinite(trend_h) else onset_h
    elif calib.method == "onset":
        rul = onset_h if np.isfinite(onset_h) else trend_h
    else:
        if np.isfinite(trend_h) and np.isfinite(onset_h):
            rul = calib.blend_w * trend_h + (1 - calib.blend_w) * onset_h
        else:
            rul = trend_h if np.isfinite(trend_h) else onset_h
    if not np.isfinite(rul):
        rul = cap
    return {"rul_hours": float(max(rul, 0.0)), "rul_trend_hours": float(trend_h),
            "rul_onset_hours": float(onset_h), "u_consumed": float(u) if np.isfinite(u) else None,
            "threshold": thr, "loglin_features": llf}


LOGLIN_FEATURES = ["1", "log(tau_h + cadence_h)", "log(trend_rul_h + cadence_h)", "clip(u_raw, -1, 2)",
                   "hi_now / threshold"]


def loglin_features(tau_h: float, trend_h: float, u_raw: float, hi_rel: float, cad: float) -> list[float]:
    u = float(np.clip(u_raw, -1.0, 2.0)) if np.isfinite(u_raw) else 0.0
    return [1.0, float(np.log(tau_h + cad)), float(np.log(trend_h + cad)), u, float(hi_rel)]


def fit_loglin(X: np.ndarray, y_log: np.ndarray, groups: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Ridge least squares on log RUL, each bearing weighted equally (sum of weights per bearing = 1)."""
    w = np.ones(len(y_log))
    for g in np.unique(groups):
        m = groups == g
        w[m] = 1.0 / m.sum()
    W = np.sqrt(w)[:, None]
    A = X * W
    b = y_log * W[:, 0]
    reg = ridge * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ b)


def stage_estimate(tau_hours: float | None, rul_hours: float) -> int:
    """Predicted health stage: 1 before a confirmed onset, else from the implied fraction
    of degradation consumed u = tau / (tau + RUL) -- the same 5-equal-parts rule as labels."""
    tau_hours = _none(tau_hours)
    if tau_hours is None:
        return 1
    u = tau_hours / max(tau_hours + rul_hours, EPS)
    return int(min(6, 2 + np.floor(np.clip(u, 0, 1) * 5)))


def interval(rul_hours: float, rel_err: float) -> list[float]:
    """Empirical interval from the validation relative-error quantile (contract)."""
    return [float(rul_hours * max(1 - rel_err, 0.0)), float(rul_hours * (1 + rel_err))]
