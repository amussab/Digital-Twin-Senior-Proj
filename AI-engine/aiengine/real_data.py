"""Reader + protocol constants for the COE training recorder's output (shared by train_real, data_coverage, node_eval).

Accepts a directory (or file) holding any mix of
  * recorder CSV exports  (run_id,label,note,received_at,rpm,window_id,sequence,device_id,valid_channels,<32 features>
                           [,severity,bearing_id,position,is_baseline])  -- READ BY HEADER NAME, so old CSVs (no trailing
                           columns) and new CSVs both work;
  * the recorder's own run_*.json (metadata) + run_*.jsonl (one {"received_at","message"} per window) pairs.
When the same run_id appears in both forms the jsonl pair wins (it is the source of the CSV).

Missing metadata: severity healthy -> 0, fault -> NaN (unknown; the stage head skips it); position -> "both"; bearing_id -> "";
is_baseline -> False.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("B1X", "B1Y", "B2X", "B2Y")
FEATS = ("rms", "kurtosis", "crest", "envelope_rms", "ftf", "bsf", "bpfo", "bpfi")
FEATURE_COLUMNS = [f"{c}_{f}" for c in CHANNELS for f in FEATS]
CONDITIONS = ("healthy", "inner_race", "outer_race", "ball", "cage")
FAULTS = CONDITIONS[1:]
POSITIONS = ("both", "b1", "b2")
SEVERITY_NAMES = {0: "healthy", 1: "incipient", 2: "moderate", 3: "severe"}
META_COLUMNS = ["run_id", "label", "severity", "bearing_id", "position", "is_baseline", "note"]
WINDOW_COLUMNS = ["rpm", "window_id", "sequence", "valid_channels", "order"]

# ---- baseline policy (contract 3e-node.1 feature_engineering.baseline.policy) ----------------------------------------------
SETTLE_S = 120.0
BASELINE_S = 60.0
RPM_TOL = 0.05
ENCODER_LEN = 24                 # classifier needs encoder_length + 1 = 25 windows of history
SPEEDS = (1800, 3600)


def window_seconds(rpm: float) -> float:
    """One node window = 20 revolutions = 1200/rpm seconds."""
    return 1200.0 / float(rpm)


def settle_windows(rpm: float, settle_s: float = SETTLE_S) -> int:
    return int(math.ceil(settle_s * float(rpm) / 1200.0))


def baseline_windows(rpm: float, baseline_s: float = BASELINE_S) -> int:
    return max(int(math.ceil(baseline_s * float(rpm) / 1200.0)), 1)


# ---- protocol (see data_coverage.py for the justification) -----------------------------------------------------------------
PROTOCOL = {
    "usable_minutes_per_run": 2.0,          # after the 120 s settle is dropped  => record >= 4 min per run
    "baseline_runs_per_speed": 1,           # >= 3 min (2 min settle + 1 min capture)
    "healthy_runs_per_speed": 3,            # different mounts (bearing_id), not counting the baseline run
    "runs_per_fault_level": 2,              # per condition x level x speed, on DIFFERENT mounts
    "minimum_levels": (2,),                 # tier "minimum": moderate only
    "full_levels": (1, 2, 3),               # tier "full"
}


def _f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _meta_defaults(label: str, sev, bearing_id, position, is_baseline, note) -> dict:
    s = _f(sev)
    if np.isnan(s):
        s = 0.0 if label == "healthy" else np.nan
    pos = str(position or "both").strip().lower()
    return {"label": label, "severity": s, "bearing_id": str(bearing_id or "").strip(),
            "position": pos if pos in POSITIONS else "both",
            "is_baseline": str(is_baseline).strip().lower() in ("1", "true", "yes", "y"), "note": str(note or "")}


def _read_csv(path: Path, rows: dict[str, dict]) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r.get("run_id")
            if not rid:
                continue
            run = rows.setdefault(rid, {"meta": None, "msgs": [], "src": "csv", "file": path.name})
            if run["src"] != "csv":
                continue
            if run["meta"] is None:
                run["meta"] = _meta_defaults(r.get("label", ""), r.get("severity"), r.get("bearing_id"), r.get("position"),
                                             r.get("is_baseline"), r.get("note"))
            try:
                run["msgs"].append({"rpm": float(r["rpm"]), "window_id": int(float(r["window_id"])),
                                    "sequence": int(float(r.get("sequence") or r["window_id"])),
                                    "valid_channels": int(float(r.get("valid_channels") or 15)),
                                    "features": [float(r[c]) for c in FEATURE_COLUMNS]})
            except (KeyError, TypeError, ValueError):
                continue                                   # malformed line: skip, never guess


def _read_jsonl_pairs(directory: Path, rows: dict[str, dict]) -> None:
    for mp in sorted(directory.glob("run_*.json")):
        jp = mp.with_suffix(".jsonl")
        if not jp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except ValueError:
            continue
        label = m.get("label", "")
        run = {"meta": _meta_defaults(label, m.get("severity"), m.get("bearing_id"), m.get("position"),
                                      m.get("is_baseline"), m.get("note")), "msgs": [], "src": "jsonl", "file": jp.name}
        for line in jp.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                msg = d["message"] if isinstance(d, dict) and "message" in d and "features" not in d else d
                if len(msg["features"]) != 32:
                    continue
                run["msgs"].append({"rpm": float(msg["rpm"]), "window_id": int(msg["window_id"]),
                                    "sequence": int(msg.get("sequence", msg["window_id"])),
                                    "valid_channels": int(msg.get("valid_channels", 15)),
                                    "features": [float(v) for v in msg["features"]]})
            except (KeyError, TypeError, ValueError):
                continue
        rows[m.get("run_id", mp.stem)] = run


def load_runs(path) -> pd.DataFrame:
    """-> one row per window (file order within each run): META_COLUMNS + WINDOW_COLUMNS + 32 feature columns.

    Windows with valid_channels != 15 (a channel not valid) and non-finite features are dropped (counted in .attrs).
    """
    path = Path(path)
    runs: dict[str, dict] = {}
    if path.is_dir():
        _read_jsonl_pairs(path, runs)
        csvs = sorted(path.rglob("*.csv"))
    elif path.suffix.lower() == ".csv":
        csvs = [path]
    else:
        raise ValueError(f"{path}: expected a directory or a recorder .csv")
    for c in csvs:
        _read_csv(c, runs)
    recs, dropped = [], 0
    for rid, run in runs.items():
        if run["meta"] is None:
            continue
        for i, m in enumerate(run["msgs"]):
            feats = np.asarray(m["features"], float)
            if (m["valid_channels"] & 15) != 15 or not np.isfinite(feats).all() or m["rpm"] <= 0:
                dropped += 1
                continue
            recs.append({"run_id": rid, **{k: run["meta"][k] for k in META_COLUMNS if k != "run_id"},
                         "rpm": m["rpm"], "window_id": m["window_id"], "sequence": m["sequence"],
                         "valid_channels": m["valid_channels"], "order": i, **dict(zip(FEATURE_COLUMNS, feats))})
    df = pd.DataFrame(recs, columns=META_COLUMNS + WINDOW_COLUMNS + FEATURE_COLUMNS)
    df.attrs["dropped_invalid_windows"] = dropped
    df.attrs["source"] = str(path)
    return df


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: meta + n_windows + median rpm + rpm_key + settle/usable windows (default policy)."""
    if df.empty:
        return pd.DataFrame(columns=META_COLUMNS + ["n_windows", "rpm", "rpm_key"])
    g = df.groupby("run_id", sort=False)
    t = g[["label", "severity", "bearing_id", "position", "is_baseline", "note"]].first()
    t["n_windows"] = g.size()
    t["rpm"] = g["rpm"].median()
    t["rpm_key"] = [rpm_key(r, SPEEDS) for r in t["rpm"]]
    return t.reset_index()


def rpm_key(rpm: float, keys=SPEEDS) -> int:
    """Nearest nominal speed within 5 %, else the rounded rpm itself (a run at an unusual speed gets its own key)."""
    k = min(keys, key=lambda s: abs(s - rpm))
    return int(k) if abs(k - rpm) <= RPM_TOL * k else int(round(rpm))
