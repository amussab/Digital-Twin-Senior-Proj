"""Anchor pools from the real rig runs + the measured late-stage profile of the real ball failure trace.

Pools: each real run (settle dropped) is cut into TIME BLOCKS, first 60 % -> train, next 20 % -> val, last 20 % -> test.
A unit of split S draws ONLY pool-S windows. Every pool window is keyed (run_id, order) so disjointness is testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .. import real_data as R
from . import constants as K


@dataclass
class Segment:
    run_id: str
    orders: np.ndarray        # recorder order index of every window (the leakage key with run_id)
    X: np.ndarray             # [n, 32] raw node features (B2X still the recorder's copy of B1Y; policy applied after synthesis)
    severity: float


class Pools:
    """healthy[(rpm)][split] -> [Segment]; fault[(cls, rpm)][split] -> [Segment]."""

    def __init__(self, df=None):
        if df is None:
            df = R.load_runs(K.TRAIN_DIR)
        self.run_table = R.run_table(df)
        self.healthy: dict = {}
        self.fault: dict = {}
        self.runs_used: list[dict] = []
        for r in self.run_table.itertuples():
            g = df[df["run_id"] == r.run_id].sort_values("order")
            key = int(r.rpm_key)
            if key not in R.SPEEDS:
                continue
            if r.label == "ball" and key != 1800:
                continue                                   # ball 3600 run jammed: excluded
            skip = R.settle_windows(key)
            g = g.iloc[skip:]
            n = len(g)
            if n < 20:
                continue
            cuts = {"train": (0, int(round(0.6 * n))), "val": (int(round(0.6 * n)), int(round(0.8 * n))), "test": (int(round(0.8 * n)), n)}
            for s, (a, b) in cuts.items():
                part = g.iloc[a:b]
                seg = Segment(r.run_id, part["order"].to_numpy(int), part[R.FEATURE_COLUMNS].to_numpy(float),
                              0.0 if r.label == "healthy" else float(r.severity))
                tgt = self.healthy.setdefault(key, {}) if r.label == "healthy" else self.fault.setdefault((r.label, key), {})
                tgt.setdefault(s, []).append(seg)
            self.runs_used.append({"run_id": r.run_id, "label": r.label, "rpm_key": key, "severity": float(r.severity),
                                   "post_settle_windows": n, "settle_dropped": skip, "baseline_run": bool(r.is_baseline),
                                   "blocks": {s: b - a for s, (a, b) in cuts.items()}})

    def severity(self, cls: str) -> int:
        sv = {seg.severity for (c, _), d in self.fault.items() if c == cls for segs in d.values() for seg in segs}
        return int(round(float(np.median(list(sv)))))

    def window_keys(self, split: str) -> set:
        keys = set()
        for pool in list(self.healthy.values()) + list(self.fault.values()):
            for seg in pool.get(split, []):
                keys.update((seg.run_id, int(o)) for o in seg.orders)
        return keys


def draw(segments: list[Segment], n: int, rng: np.random.Generator, mean_block: float = K.BLOCK_MEAN):
    """Block bootstrap: concatenate contiguous runs of real windows (geometric block length) -> (X [n,32], run_ids, orders)."""
    w = np.array([len(s.orders) for s in segments], float)
    xs, rid, od = [], [], []
    total = 0
    while total < n:
        s = segments[int(rng.choice(len(segments), p=w / w.sum()))]
        L = int(min(max(1, rng.geometric(1.0 / mean_block)), len(s.orders), n - total))
        a = int(rng.integers(0, len(s.orders) - L + 1))
        xs.append(s.X[a:a + L])
        rid += [s.run_id] * L
        od.append(s.orders[a:a + L])
        total += L
    return np.concatenate(xs)[:n], rid[:n], np.concatenate(od)[:n]


# ---- real ball failure trace -> late-stage profile -------------------------------------------------------------------------------
SEGS = {"seg0": (0, 28), "seg1": (28, 121), "seg2": (121, 144)}


def measure_late_profile(path=None) -> dict:
    """ln(median seg1 / median seg0) [mid] and ln(median seg2 / median seg0) [end] per raw feature (B2X := B2Y policy applied).
    segment 0 = the degrading level that the anchored fault windows stand for; seg2 = disintegrating bearing (windows 121-143).
    [MEASURED on rig, n=1, rpm column invalid]"""
    path = path or K.FAILURE_TRACE
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    F = np.array([r["message"]["features"] for r in rows], float)
    med = {k: np.median(F[a:b], axis=0) for k, (a, b) in SEGS.items()}
    lf = lambda v: np.log(np.maximum(v, K.LOG_FLOOR))                                    # noqa: E731
    mid = R.apply_b2x_policy(lf(med["seg1"]) - lf(med["seg0"]))
    end = R.apply_b2x_policy(lf(med["seg2"]) - lf(med["seg0"]))
    rms = {k: float(med[k][0]) for k in med}
    return {"source": str(path), "tag": "[MEASURED on rig, n=1 ball failure trace, rpm column invalid; segments 0-2 = windows 0-143]",
            "ln_mid": mid.tolist(), "ln_end": end.tolist(), "feature_columns": R.FEATURE_COLUMNS,
            "rms_B1X_median": rms, "rms_ratio_end_over_seg0": rms["seg2"] / rms["seg0"], "rms_ratio_mid_over_seg0": rms["seg1"] / rms["seg0"],
            "env_ratio_end_over_seg0": float(med["seg2"][3] / med["seg0"][3])}


def load_late_profile() -> dict:
    return json.loads(K.LATE_PROFILE_FILE.read_text())


if __name__ == "__main__":
    prof = measure_late_profile()
    K.LATE_PROFILE_FILE.write_text(json.dumps(prof, indent=1))
    print({k: v for k, v in prof.items() if k not in ("ln_mid", "ln_end", "feature_columns")})
