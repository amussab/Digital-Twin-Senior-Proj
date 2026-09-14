"""Turning an N-HiTS health-index forecast into a remaining-useful-life number.

Why this indirection exists
---------------------------
N-HiTS forecasts the health index, not RUL. RUL is then solved for: the step at
which the forecast health index reaches the bearing's failure threshold. The
reason is stated in data.py -- a forecaster handed RUL as its target learns to
continue a countdown it will never see in service. Forecasting an observable
quantity and solving for the crossing is the standard trend-extrapolation
prognostic formulation, and it is the one the ICS layer doc describes when it
says N-HiTS's slow stack "tracks the multi-day stiffness-decay trend RUL
actually depends on".

Two estimators, combined
------------------------
1.  TREND. Fit an exponential trend through the recent health-index history
    plus the N-HiTS forecast, and extrapolate to the failure threshold. Sharp
    late in life when degradation is clearly under way; near-useless early,
    when the index is flat and its slope is mostly noise.

2.  POPULATION PRIOR. A lookup, learned from training runs only, of typical
    remaining hours given the current health index. Carries the early-life
    estimate where the trend has nothing to work with.

They are blended by trend confidence, which rises with health index and with
how cleanly the trend fits. The blend is what makes an early-life estimate
usable at all -- and `evaluate.py` deliberately also scores the prior on its
own, so the report can state plainly how much N-HiTS's forecast adds over the
population statistic rather than assuming it adds anything.

The failure threshold is CLASS-CONDITIONAL, taken from the TFT's predicted
class. A cage defect never reaches the health index an outer-race defect
reaches at the same true damage, so a single pooled threshold would systematically
over-predict life for weak fault types. This is the point where the two models
stop being independent: the classifier's output selects the regressor's
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import FAULT_CLASSES

# History used alongside the forecast when fitting the trend.
TREND_HISTORY_WINDOWS = 24
# Health index below which the trend is considered uninformative on its own.
TREND_FLOOR_HI = 0.08
_EPS = 1e-9


@dataclass
class RULCalibration:
    """Everything fitted on training runs that RUL estimation needs."""

    hours_per_window: float
    class_thresholds: dict[str, float] = field(default_factory=dict)
    pooled_threshold: float = 1.0
    prior_hi_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prior_rul_hours: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def threshold_for(self, fault_class: str) -> float:
        return self.class_thresholds.get(fault_class, self.pooled_threshold)

    def to_dict(self) -> dict:
        return {
            "hours_per_window": self.hours_per_window,
            "class_thresholds": self.class_thresholds,
            "pooled_threshold": self.pooled_threshold,
            "prior_hi_edges": self.prior_hi_edges.tolist(),
            "prior_rul_hours": self.prior_rul_hours.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RULCalibration":
        return cls(
            hours_per_window=payload["hours_per_window"],
            class_thresholds=payload["class_thresholds"],
            pooled_threshold=payload["pooled_threshold"],
            prior_hi_edges=np.asarray(payload["prior_hi_edges"], dtype=np.float64),
            prior_rul_hours=np.asarray(payload["prior_rul_hours"], dtype=np.float64),
        )


def calibrate(train_frame: pd.DataFrame, hours_per_window: float,
              n_prior_bins: int = 24) -> RULCalibration:
    """Fit failure thresholds and the population prior on TRAINING runs only."""
    faulted = train_frame[train_frame["fault_class"].astype(str) != "healthy"]

    thresholds: dict[str, float] = {}
    end_of_life_hi: list[float] = []
    for fault_class, group in faulted.groupby(
        faulted["fault_class"].astype(str), observed=True
    ):
        finals = group.groupby("run_id")["hi"].last().to_numpy()
        if len(finals):
            thresholds[str(fault_class)] = float(np.median(finals))
            end_of_life_hi.extend(finals.tolist())

    pooled = float(np.median(end_of_life_hi)) if end_of_life_hi else 1.0
    # "healthy" is not a failure mode; if the classifier says healthy we still
    # need some threshold to report a number against, so use the pooled one.
    thresholds.setdefault("healthy", pooled)

    # Population prior: median remaining hours per health-index bin.
    hi = faulted["hi"].to_numpy()
    rul = faulted["rul_hours"].to_numpy()
    valid = np.isfinite(hi) & np.isfinite(rul)
    hi, rul = hi[valid], rul[valid]

    edges = np.quantile(hi, np.linspace(0.0, 1.0, n_prior_bins + 1))
    edges = np.unique(edges)
    bin_idx = np.clip(np.searchsorted(edges, hi, side="right") - 1, 0, len(edges) - 2)
    prior = np.zeros(len(edges) - 1)
    for b in range(len(prior)):
        sel = bin_idx == b
        prior[b] = float(np.median(rul[sel])) if sel.any() else np.nan
    # Fill empty bins by interpolation so the lookup is total over its range.
    if np.isnan(prior).any():
        good = ~np.isnan(prior)
        prior[~good] = np.interp(
            np.flatnonzero(~good), np.flatnonzero(good), prior[good]
        )

    return RULCalibration(
        hours_per_window=hours_per_window,
        class_thresholds=thresholds,
        pooled_threshold=pooled,
        prior_hi_edges=edges,
        prior_rul_hours=prior,
    )


def prior_rul(calib: RULCalibration, hi_now: np.ndarray) -> np.ndarray:
    """Population-prior RUL for a batch of current health-index values."""
    if calib.prior_hi_edges.size < 2:
        return np.full_like(np.asarray(hi_now, dtype=float), np.nan)
    centres = 0.5 * (calib.prior_hi_edges[:-1] + calib.prior_hi_edges[1:])
    return np.interp(np.asarray(hi_now, dtype=float), centres, calib.prior_rul_hours)


def _exponential_trend_steps(series: np.ndarray, threshold: float) -> tuple[float, float]:
    """Steps until `series` reaches `threshold`, and the fit's R^2.

    Fitted on log(health index): bearing degradation compounds, so a
    straight-line fit in log space extrapolates the acceleration that a
    straight-line fit in linear space would miss entirely.
    """
    y = np.log(np.maximum(series, TREND_FLOOR_HI * 0.25))
    x = np.arange(len(y), dtype=float)
    # Recent points weigh more; the early part of the window is context, not
    # evidence about the current rate.
    weights = np.exp(np.linspace(-1.5, 0.0, len(y)))

    w_sum = weights.sum()
    x_mean = float((weights * x).sum() / w_sum)
    y_mean = float((weights * y).sum() / w_sum)
    var_x = float((weights * (x - x_mean) ** 2).sum())
    if var_x <= _EPS:
        return np.inf, 0.0
    slope = float((weights * (x - x_mean) * (y - y_mean)).sum() / var_x)

    residual = y - (y_mean + slope * (x - x_mean))
    ss_res = float((weights * residual**2).sum())
    ss_tot = float((weights * (y - y_mean) ** 2).sum())
    r2 = 0.0 if ss_tot <= _EPS else max(0.0, 1.0 - ss_res / ss_tot)

    if slope <= _EPS:
        return np.inf, r2        # flat or improving: no crossing to predict
    steps = (np.log(max(threshold, _EPS)) - y[-1]) / slope
    return (max(steps, 0.0), r2)


def estimate(
    calib: RULCalibration,
    hi_history: np.ndarray,
    hi_forecast: np.ndarray,
    predicted_class: str,
) -> dict[str, float]:
    """RUL in hours for one window, from its history and N-HiTS forecast.

    Returns the blended estimate plus both components, so the evaluation can
    attribute accuracy rather than just report a total.
    """
    threshold = calib.threshold_for(predicted_class)
    history = np.asarray(hi_history, dtype=float)[-TREND_HISTORY_WINDOWS:]
    series = np.concatenate([history, np.asarray(hi_forecast, dtype=float)])
    hi_now = float(history[-1])

    steps, r2 = _exponential_trend_steps(series, threshold)
    trend_hours = steps * calib.hours_per_window if np.isfinite(steps) else np.inf
    prior_hours = float(prior_rul(calib, np.array([hi_now]))[0])

    # Trend confidence: needs both a health index that has actually left the
    # noise floor and a trend that actually fits.
    level_conf = np.clip((hi_now - TREND_FLOOR_HI) / (threshold - TREND_FLOOR_HI + _EPS), 0.0, 1.0)
    confidence = float(np.clip(level_conf * (0.35 + 0.65 * r2), 0.0, 1.0))
    if not np.isfinite(trend_hours):
        confidence = 0.0
        trend_hours = prior_hours

    blended = confidence * trend_hours + (1.0 - confidence) * prior_hours
    return {
        "rul_hours": float(max(blended, 0.0)),
        "rul_trend_hours": float(trend_hours),
        "rul_prior_hours": float(prior_hours),
        "trend_confidence": confidence,
        "threshold_used": float(threshold),
        "hi_now": hi_now,
    }


def stage_from_hi(calib: RULCalibration, hi: float, predicted_class: str) -> int:
    """Health stage implied by a health-index value.

    The six stages are defined on Miner's damage [spec M6]. The host never sees
    damage, so the stage is read off the health index against the same
    class-conditional failure threshold used for RUL: the index's progress
    toward that threshold stands in for damage's progress toward D = 0.95.
    """
    from config import STAGE_DAMAGE_EDGES, FAILURE_DAMAGE

    threshold = calib.threshold_for(predicted_class)
    implied_damage = FAILURE_DAMAGE * np.clip(hi / (threshold + _EPS), 0.0, 1.2)
    return int(np.searchsorted(STAGE_DAMAGE_EDGES, implied_damage, side="right") + 1)


def class_names() -> list[str]:
    return list(FAULT_CLASSES)
