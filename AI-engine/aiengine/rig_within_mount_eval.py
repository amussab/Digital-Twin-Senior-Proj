"""Within-mount, time-blocked evaluation of the rig classifier  [MEASURED on rig, OPTIMISTIC].

Why: with one physical bearing per fault there is no independent held-out bearing, so IS3 cannot be scored (train_real says so).
This script measures how well the 5 classes separate in the data actually recorded, with a leak-free TIME split INSIDE every run:
first 60 % of a run's windows train, a gap of T=25 windows (the classifier's history length) is dropped, the last 40 % test.
The split rule is fixed here BEFORE any fit; every variant is scored exactly once; nothing is tuned on the test part.
The same bearing and mount appear in train and test, so the result is an UPPER bound on cross-bearing performance and is
NOT an IS3 result. Variants: real-only, real + simulated windows (weight 0.3). hp = the grid point train_real selected.

  python -m aiengine.rig_within_mount_eval --data ../COE/training/training_data [--hp 1]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import real_data as R
from . import train_real as T_
from .config import REPORTS_DIR

TRAIN_FRAC, GAP = 0.6, T_.T
H = T_.CLASS_ORDER.index("healthy")


def split_parts(parts):
    tr, te = [], []
    for p in parts:
        n = len(p["Xc"])
        a, b = int(TRAIN_FRAC * n), int(TRAIN_FRAC * n) + GAP
        if a < 5 or n - b < 5:
            continue
        sl = lambda s: {k: (np.asarray(v)[s] if not isinstance(v, list) else v[s]) for k, v in p.items()}
        tr.append(sl(slice(0, a)))
        te.append(sl(slice(b, n)))
    return T_.cat(tr), T_.cat(te)


def score(y, pred):
    from sklearn.metrics import confusion_matrix, f1_score
    labs = list(range(len(T_.CLASS_ORDER)))
    f1 = f1_score(y, pred, labels=labs, average=None, zero_division=0)
    present = sorted(set(y))
    return {"macro_f1": float(np.mean([f1[i] for i in present])),
            "f1": {T_.CLASS_ORDER[i]: float(f1[i]) for i in present},
            "accuracy": float((y == pred).mean()),
            "healthy_false_alarm_rate": float((pred[y == H] != H).mean()) if (y == H).any() else None,
            "missed_fault_rate": float((pred[y != H] == H).mean()) if (y != H).any() else None,
            "confusion_labels": [T_.CLASS_ORDER[i] for i in labs], "confusion": confusion_matrix(y, pred, labels=labs).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hp", type=int, default=1, help="index into train_real.GRID (default 1 = the one train_real selected)")
    ap.add_argument("--sim-weight", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    df = R.load_runs(a.data)
    rt = R.run_table(df)
    baselines, _ = T_.build_baselines(df, rt, R.SETTLE_S, R.BASELINE_S)
    parts, notes = T_.build_samples(df, rt, baselines, R.SETTLE_S, R.BASELINE_S)
    train, test = split_parts(parts)
    print(f"[MEASURED on rig, within-mount] train {len(train['y'])} windows, test {len(test['y'])} windows, classes in test "
          f"{ {T_.CLASS_ORDER[i]: int((test['y'] == i).sum()) for i in sorted(set(test['y']))} }")
    hp = T_.GRID[a.hp]
    out = {"tag": "[MEASURED on rig, WITHIN-MOUNT time-blocked: optimistic upper bound, NOT IS3]", "hp": hp, "train_windows": int(len(train['y'])),
           "test_windows": int(len(test['y'])), "split": f"first {TRAIN_FRAC:.0%} of each run train, gap {GAP}, rest test", "variants": {}, "notes": notes}
    sim = T_.sim_samples()
    for name, use_sim in (("real_only", False), ("real_plus_sim", True)):
        X, y, w = train["Xc"], train["y"], None
        if use_sim:
            X, y = np.concatenate([X, sim["Xc"]]), np.concatenate([y, sim["y"]])
            w = np.concatenate([np.ones(len(train["y"])), np.full(len(sim["y"]), a.sim_weight)])
        P = T_.Clf(hp, a.threads).fit(X, y, w).proba(test["Xc"]).argmax(1)
        out["variants"][name] = score(test["y"], P)
        v = out["variants"][name]
        print(f"{name}: macro-F1 {v['macro_f1']:.3f} acc {v['accuracy']:.3f} per-class "
              f"{ {k: round(x, 3) for k, x in v['f1'].items()} } healthy-false-alarm {v['healthy_false_alarm_rate']} missed {v['missed_fault_rate']}")
    out["seconds"] = round(time.perf_counter() - t0, 1)
    f = Path(REPORTS_DIR) / f"rig_within_mount_{time.strftime('%Y%m%dT%H%M%S')}.json"
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)


if __name__ == "__main__":
    main()
