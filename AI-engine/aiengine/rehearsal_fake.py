"""[SIMULATION rehearsal] fake recorder CSVs from the node-matched simulator (synth_coe_node), for an end-to-end dry run.

    python -m aiengine.rehearsal_fake --out <scratch dir> [--tier minimum|full] [--split train|test] [--seed 1] [--prefix F]

Writes one recorder-format CSV per run (same columns as COE/training/training_recorder.py incl. severity, bearing_id, position,
is_baseline) following the protocol in data_coverage.py: per speed a commissioning-baseline run, healthy runs, and
condition x level runs on different 'mounts' (each mount = windows resampled from 3 different simulated runs). Every file
name and bearing_id is prefixed so nothing can be mistaken for rig data. Window ORDER inside a run is a resample (no
temporal structure) and the settle windows are healthy-like windows with an exponentially decaying amplitude: this exercises the
pipeline; it says nothing about rig performance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_coverage as DC
from . import real_data as R
from .config import BEARING_FEATURES, SEED

RECORDER_COLUMNS = ["run_id", "label", "note", "received_at", "rpm", "window_id", "sequence", "device_id", "valid_channels"] \
    + R.FEATURE_COLUMNS + ["severity", "bearing_id", "position", "is_baseline"]


def pool_from_sim(split: str) -> pd.DataFrame:
    """rows = one simulator window with BOTH bearings: feats32, cls1, cls2, bucket1, bucket2, rpm, run."""
    from . import coe_dataset as D
    df = D.generate()
    df = df[(df["split"] == split) & ~df["transition"].astype(bool)]
    b = lambda s: np.where(s <= 0, 0, np.where(s <= 1 / 3, 1, np.where(s <= 2 / 3, 2, 3)))   # noqa: E731
    parts = {}
    for k in (1, 2):
        d = df[df["bearing"] == k].set_index(["bearing_key", "window_index"]).sort_index()
        parts[k] = d
    idx = parts[1].index.intersection(parts[2].index)
    p1, p2 = parts[1].loc[idx], parts[2].loc[idx]
    f = np.concatenate([p1[BEARING_FEATURES].to_numpy(), p2[BEARING_FEATURES].to_numpy()], axis=1)
    out = pd.DataFrame({"run": [i[0] for i in idx], "rpm": p1["rpm"].to_numpy(), "cls1": p1["observable_class"].astype(str).to_numpy(),
                        "cls2": p2["observable_class"].astype(str).to_numpy(), "lv1": b(p1["severity"].to_numpy(float)),
                        "lv2": b(p2["severity"].to_numpy(float))})
    out["lv1"] = np.where(out["cls1"] == "healthy", 0, np.maximum(out["lv1"], 1))
    out["lv2"] = np.where(out["cls2"] == "healthy", 0, np.maximum(out["lv2"], 1))
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(f, columns=R.FEATURE_COLUMNS)], axis=1)


def sample_run(pool, rng, cond, level, position, rpm, n_windows, settle_w):
    p = pool[np.isclose(pool["rpm"], rpm, atol=0.05 * rpm)]
    if cond == "healthy":
        m = (p["cls1"] == "healthy") & (p["cls2"] == "healthy")
    elif position == "b1":
        m = (p["cls1"] == cond) & (p["lv1"] == level) & (p["cls2"] == "healthy")
    elif position == "b2":
        m = (p["cls2"] == cond) & (p["lv2"] == level) & (p["cls1"] == "healthy")
    else:
        m = (p["cls1"] == cond) & (p["lv1"] == level) & (p["cls2"] == cond) & (p["lv2"] == level)
    q = p[m]
    if q.empty:
        return None
    runs = q["run"].unique()
    mount = rng.choice(runs, size=min(3, len(runs)), replace=False)
    q = q[q["run"].isin(mount)]
    X = q[R.FEATURE_COLUMNS].to_numpy()[rng.integers(0, len(q), n_windows)].copy()
    h = p[(p["cls1"] == "healthy") & (p["cls2"] == "healthy")]
    Hh = h[R.FEATURE_COLUMNS].to_numpy()
    for i in range(min(settle_w, n_windows)):                       # settle: healthy-like, decaying amplitude
        X[i] = Hh[rng.integers(0, len(Hh))] * (1.0 + 0.6 * np.exp(-3.0 * i / max(settle_w, 1)))
    return X


def make(out: Path, tier: str, split: str, seed: int, prefix: str, usable_min: float = 2.0) -> int:
    pool = pool_from_sim(split)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_runs, mount_no = 0, 0
    for n in DC.required(tier):
        for j in range(n["runs"] if n["cond"] != "baseline" else 1):
            rpm = n["rpm"]
            pos = ["b1", "b2", "both"][(j + n["severity"]) % 3] if n["cond"] in R.FAULTS else "both"
            sw = R.settle_windows(rpm)
            usable = int(usable_min * 60 / R.window_seconds(rpm))
            cond = "healthy" if n["cond"] == "baseline" else n["cond"]
            nw = sw + (int(60 / R.window_seconds(rpm)) if n["cond"] == "baseline" else usable)
            X = None
            for pp in [pos] + [x for x in ("b1", "b2", "both") if x != pos]:      # fall back if the sim lacks this combination
                X = sample_run(pool, rng, cond, n["severity"], pp, rpm, nw, sw)
                if X is not None:
                    pos = pp
                    break
            if X is None:
                continue
            mount_no += 1
            n_runs += 1
            rid = f"run_{prefix}{n_runs:03d}"
            with open(out / f"{rid}.csv", "w", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(RECORDER_COLUMNS)
                for i, x in enumerate(X):
                    w.writerow([rid, cond, f"[SIMULATION rehearsal] fake {n['cond']}", "2026-01-01T00:00:00", rpm, i + 1, i + 1, "FAKE", 15]
                               + [f"{v:.8g}" for v in x] + [n["severity"], f"{prefix}M{mount_no:02d}", pos, int(n["cond"] == "baseline")])
    return n_runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.rehearsal_fake", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", choices=("minimum", "full"), default="minimum")
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--prefix", default="F")
    a = ap.parse_args(argv)
    n = make(Path(a.out), a.tier, a.split, a.seed, a.prefix)
    print(f"[SIMULATION rehearsal] wrote {n} fake recorder runs to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
