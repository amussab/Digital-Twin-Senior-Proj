"""Labels (DESIGN.md section 6): causal onset, ground-truth health stage 1-6, observable class.

- Onset (first predicting time): HI exceeds the baseline windows' mu + 3 sigma for 3 consecutive
  windows. Causal: window k is declared only once k, k+1, k+2 are all seen, so the same rule runs
  at runtime (`OnlineOnset`). Search starts after the baseline windows (the baseline has to
  exist before anything can exceed it). sigma has a floor (SIGMA_FLOOR, HI units) -- with a
  median baseline about half the baseline windows have HI exactly 0, so the raw sigma can be ~0
  and one noisy window would trigger. [CLAUDE-SYNTHESIS] engineering guard, documented.
- Ground-truth stage for run-to-failure units: 1 before onset; [t_onset, T_fail] split into 5
  equal-time parts -> stages 2..6. [CLAUDE-SYNTHESIS] a defined labelling, documented as such.
- Observable class: `healthy` until onset, then the unit's fault_class. Seeded-fault records
  (not run-to-failure, faulted from the first window) carry their class throughout. `mixed` /
  `unknown` units are excluded from classification (cls_usable=False) but kept for RUL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import BASELINE_WINDOWS

ONSET_SIGMAS = 3.0
ONSET_CONSECUTIVE = 3
SIGMA_FLOOR = 0.02
CLASSIFIABLE = ("healthy", "outer_race", "inner_race", "ball", "cage")
DEGRADATION_TAIL_EXCLUDED = 0.10       # S7 pre-registered: last 10% of degradation excluded


def onset_threshold(hi_baseline: np.ndarray) -> float:
    hi_baseline = np.asarray(hi_baseline, dtype=float)
    return float(hi_baseline.mean() + ONSET_SIGMAS * max(hi_baseline.std(), SIGMA_FLOOR))


def detect_onset(hi: np.ndarray, search_from: int = BASELINE_WINDOWS) -> int | None:
    """Index of the first window of the first run of 3 consecutive exceedances (or None)."""
    hi = np.asarray(hi, dtype=float)
    thr = onset_threshold(hi[:BASELINE_WINDOWS])
    streak = 0
    for k in range(search_from, len(hi)):
        streak = streak + 1 if hi[k] > thr else 0
        if streak >= ONSET_CONSECUTIVE:
            return k - ONSET_CONSECUTIVE + 1
    return None


class OnlineOnset:
    """Runtime version of detect_onset: feed HI one window at a time."""

    def __init__(self, threshold: float | None = None):
        self.base: list[float] = []
        self.threshold = threshold
        self.streak = 0
        self.n = 0
        self.onset_index: int | None = None

    def update(self, hi: float) -> int | None:
        k = self.n
        self.n += 1
        if self.threshold is None:
            self.base.append(float(hi))
            if len(self.base) >= BASELINE_WINDOWS:
                self.threshold = onset_threshold(np.asarray(self.base))
            return self.onset_index
        if self.onset_index is None:
            self.streak = self.streak + 1 if hi > self.threshold else 0
            if self.streak >= ONSET_CONSECUTIVE:
                self.onset_index = k - ONSET_CONSECUTIVE + 1
        return self.onset_index


def stage_from_fraction(u: np.ndarray) -> np.ndarray:
    """Fraction of degradation elapsed (u in [0,1]) -> stage 2..6; u < 0 -> stage 1."""
    u = np.asarray(u, dtype=float)
    st = 2 + np.floor(np.clip(u, 0, 1) * 5).astype(int)
    st = np.minimum(st, 6)
    return np.where(u < 0, 1, st)


def label_table(eng: pd.DataFrame) -> pd.DataFrame:
    """Add onset/stage/observable-class columns to an engineered window table."""
    eng = eng.sort_values(["unit_id", "window_index"]).reset_index(drop=True)
    n = len(eng)
    onset_idx = np.full(n, -1, dtype=np.int64)
    onset_t = np.full(n, np.nan)
    stage = np.zeros(n, dtype=np.int64)              # 0 = not defined (non-RTF faulted records)
    obs = np.empty(n, dtype=object)
    in_deg = np.zeros(n, dtype=bool)
    in_deg_full = np.zeros(n, dtype=bool)
    usable = np.zeros(n, dtype=bool)

    for uid, idx in eng.groupby("unit_id", sort=False).indices.items():
        g = eng.iloc[idx]
        fc = str(g["fault_class"].iloc[0])
        rtf = bool(g["run_to_failure"].iloc[0])
        t = g["t_hours"].to_numpy(dtype=float)
        usable[idx] = fc in CLASSIFIABLE
        if rtf:
            k = detect_onset(g["hi"].to_numpy())
            if k is None:
                # No detectable onset: treat the last 10% of life as degradation so the unit
                # still contributes, and flag it (onset_index stays -1).
                life = float(g["life_hours"].iloc[0])
                t_on = t[0] + 0.9 * (life - t[0])
            else:
                t_on = t[k]
                onset_idx[idx] = k
            life = float(g["life_hours"].iloc[0])
            onset_t[idx] = t_on
            span = max(life - t_on, 1e-9)
            u = (t - t_on) / span
            stage[idx] = stage_from_fraction(u)
            obs[idx] = np.where(t >= t_on, fc, "healthy")
            in_deg_full[idx] = (t >= t_on) & (t <= life)
            in_deg[idx] = (t >= t_on) & (t <= life - DEGRADATION_TAIL_EXCLUDED * span)
        else:
            stage[idx] = 1 if fc == "healthy" else 0
            obs[idx] = fc
    out = eng.copy()
    out["onset_index"] = onset_idx
    out["onset_t_hours"] = onset_t
    out["health_stage"] = stage
    out["observable_class"] = obs
    out["cls_usable"] = usable
    out["in_degradation_window"] = in_deg
    out["in_degradation_full"] = in_deg_full
    return out
