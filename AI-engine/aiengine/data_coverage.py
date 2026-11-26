"""Data-sufficiency checker for the recorder's output.

    python -m aiengine.data_coverage <dir-or-csv> [--tier minimum|full] [--settle-s 120]

Prints usable minutes (after dropping the settle windows) per condition x health level x rpm, runs and distinct mounts
(bearing_id), then what is still missing against the recommended protocol.

Protocol and why (numbers are procedure/arithmetic, not tuned on data):
  * One node window = 20 rev = 1200/rpm s: 0.67 s at 1800 rpm, 0.33 s at 3600 rpm.
  * Settle: the first 120 s of every run is dropped (contract policy settle_seconds); the commissioning baseline is the
    next 60 s of a HEALTHY reference run => baseline run >= 3 min per speed.
  * The classifier needs 25 windows of history (encoder 24 + current), so a run needs >= 25 usable windows to give one sample;
    2 usable min = 180 windows at 1800 rpm / 360 at 3600 rpm leaves >150 samples. Windows inside a run are strongly
    correlated, so the limit is the number of independent MOUNTS, not minutes: record >= 4 min per run (2 min settle + 2 min).
  * Splits are by run (bearing swap = new run): a class needs >= 2 runs to have a train and a held-out run, and >= 3 to also
    run grouped cross-validation on the training part. So: healthy >= 3 runs, each fault x level >= 2 runs on different
    mounts per speed. 'minimum' tier = moderate level only (4 faults x 2 runs); 'full' adds incipient + severe.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from . import real_data as R


def usable_table(df: pd.DataFrame, settle_s: float = R.SETTLE_S) -> pd.DataFrame:
    """One row per run with usable windows/minutes after the settle drop."""
    t = R.run_table(df)
    if t.empty:
        return t
    t["settle_windows"] = [R.settle_windows(r, settle_s) for r in t["rpm_key"]]
    t["usable_windows"] = (t["n_windows"] - t["settle_windows"]).clip(lower=0)
    t["usable_min"] = t["usable_windows"] * [R.window_seconds(r) / 60.0 for r in t["rpm_key"]]
    t["raw_min"] = t["n_windows"] * [R.window_seconds(r) / 60.0 for r in t["rpm_key"]]
    t["cond"] = np.where(t["is_baseline"], "baseline", t["label"])
    return t


def coverage_rows(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    t = t.copy()
    t["severity"] = t["severity"].fillna(-1)
    g = t.groupby(["cond", "severity", "rpm_key"])
    return g.agg(runs=("run_id", "nunique"), mounts=("bearing_id", lambda s: len({x for x in s if x})),
                 raw_min=("raw_min", "sum"), usable_min=("usable_min", "sum")).reset_index()


def required(tier: str) -> list[dict]:
    P = R.PROTOCOL
    lv = P["minimum_levels"] if tier == "minimum" else P["full_levels"]
    need = []
    for sp in R.SPEEDS:
        need.append({"cond": "baseline", "severity": 0, "rpm": sp, "runs": P["baseline_runs_per_speed"],
                     "min_each": R.SETTLE_S / 60 + R.BASELINE_S / 60})
        need.append({"cond": "healthy", "severity": 0, "rpm": sp, "runs": P["healthy_runs_per_speed"],
                     "min_each": R.SETTLE_S / 60 + P["usable_minutes_per_run"]})
        for f in R.FAULTS:
            for s in lv:
                need.append({"cond": f, "severity": s, "rpm": sp, "runs": P["runs_per_fault_level"],
                             "min_each": R.SETTLE_S / 60 + P["usable_minutes_per_run"]})
    return need


def missing(t: pd.DataFrame, tier: str = "minimum", settle_s: float = R.SETTLE_S) -> list[dict]:
    out = []
    for n in required(tier):
        if t.empty:
            have = pd.DataFrame()
        else:
            sev = t["severity"].fillna(-1)
            have = t[(t["cond"] == n["cond"]) & (sev == n["severity"]) & (t["rpm_key"] == n["rpm"])]
        if n["cond"] == "baseline":
            ok = have[have["raw_min"] >= n["min_each"] - 0.05] if len(have) else have
        else:
            ok = have[have["usable_min"] >= R.PROTOCOL["usable_minutes_per_run"] - 0.05] if len(have) else have
        mounts = len({b for b in ok["bearing_id"] if b}) if len(ok) else 0
        n_ok = len(ok)
        short = max(0, n["runs"] - n_ok)
        # runs without a bearing_id cannot be shown to be different mounts
        need_mounts = n["cond"] != "baseline" and n_ok >= n["runs"] and mounts < min(n["runs"], n_ok) and mounts > 0
        out.append({**n, "have_runs": n_ok, "have_mounts": mounts, "missing_runs": short,
                    "note": "" if not need_mounts else "runs exist but use fewer distinct bearing_id than runs"})
    return out


def report(df: pd.DataFrame, tier: str = "minimum", settle_s: float = R.SETTLE_S) -> str:
    t = usable_table(df, settle_s)
    lines = ["Data coverage (recorder windows; usable = after the %.0f s settle drop)" % settle_s, ""]
    cov = coverage_rows(t)
    if cov.empty:
        lines.append("  (no windows found)")
    else:
        lines.append(f"{'condition':<10} {'level':>5} {'rpm':>5} {'runs':>4} {'mounts':>6} {'raw min':>8} {'usable min':>10}")
        for r in cov.itertuples():
            lv = "?" if r.severity < 0 else int(r.severity)
            lines.append(f"{r.cond:<10} {lv!s:>5} {r.rpm_key:>5} {r.runs:>4} {r.mounts:>6} {r.raw_min:>8.1f} {r.usable_min:>10.1f}")
    miss = missing(t, tier, settle_s)
    todo = [m for m in miss if m["missing_runs"] > 0 or m["note"]]
    lines += ["", f"Still missing vs the '{tier}' protocol ({sum(m['missing_runs'] for m in todo)} runs, "
                  f"about {sum(m['missing_runs'] * m['min_each'] for m in todo):.0f} min of recording):"]
    if not todo:
        lines.append("  nothing: protocol satisfied")
    for m in todo:
        lv = "" if m["cond"] == "baseline" else f" level {m['severity']}"
        lines.append(f"  {m['cond']}{lv} @ {m['rpm']} rpm: need {m['missing_runs']} more run(s) of >= {m['min_each']:.0f} min"
                     + (f" ({m['note']})" if m["note"] else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.data_coverage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--tier", choices=("minimum", "full"), default="minimum")
    ap.add_argument("--settle-s", type=float, default=R.SETTLE_S)
    a = ap.parse_args(argv)
    df = R.load_runs(a.data)
    print(report(df, a.tier, a.settle_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
