"""Plausibility report: generator feature ranges vs the real fault runs and vs the real ball failure trace.

    python -m aiengine.rig_anchor.plausibility [--data data/rig_anchor/rig_anchor.parquet]
Writes reports/rig_anchor_plausibility.md (+ .json, .png). Compares B1Y (the faulted housing, Y axis) median [p5-p95].
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import real_data as R
from ..config import REPORTS_DIR, ROOT
from . import constants as K
from . import generator as G
from . import pools as P

SHOW = ["B1Y_rms", "B1Y_kurtosis", "B1Y_crest", "B1Y_envelope_rms", "B1Y_ftf", "B1Y_bsf", "B1Y_bpfo", "B1Y_bpfi"]


def _q(x):
    x = np.asarray(x, float)
    return np.percentile(x, [5, 50, 95])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "rig_anchor" / "rig_anchor.parquet"))
    a = ap.parse_args(argv)
    gen = pd.read_parquet(a.data)
    b1 = gen[gen["unit_id"].str.endswith(":B1")]
    real = R.load_runs(K.TRAIN_DIR)
    rt = R.run_table(real)
    trace = np.array([json.loads(l)["message"]["features"] for l in open(K.FAILURE_TRACE, encoding="utf-8")], float)
    prof = P.load_late_profile()
    res = {}
    lines = [f"# rig_anchor plausibility: generator vs real windows", "", f"Provenance: {K.PROVENANCE}.", f"Limits: {K.LIMITS}", "",
             "Cells are median [p5 - p95] of the raw node feature of channel B1Y (faulted housing, Y axis), in g (rms, envelope, bands) or unitless (kurtosis, crest).",
             "Real = post-settle windows of the recorded run(s); generator = all splits, 1800 rpm and 3600 rpm pooled per class unless stated. "
             "'gen anchor' = generator windows with late_w = 0 and g >= 0.98 (the measured fault level, by construction); 'gen end' = last 5 windows of a unit (u > 0.97).", ""]
    classes = [c for c in K.CLASSES]
    for cls in classes:
        rr = rt[(rt["label"] == cls) & ~(rt["is_baseline"] & (cls != "healthy"))]
        rwin = real[real["run_id"].isin(rr["run_id"])]
        # drop settle per run
        parts = []
        for rid, g in rwin.groupby("run_id"):
            key = int(rr[rr["run_id"] == rid]["rpm_key"].iloc[0])
            parts.append(g.sort_values("order").iloc[R.settle_windows(key):])
        rw = pd.concat(parts)
        gc = b1[b1["rig_unit"].isin(b1[b1["fault_class"] == cls]["rig_unit"].unique())] if cls != "healthy" else b1[b1["faulted_bearing"] == 0]
        gc = gc if cls == "healthy" else gc[gc["fault_class"] == cls]
        rows = {"real " + cls: rw[SHOW]}
        if cls == "healthy":
            rows["gen (healthy units)"] = gc[SHOW]
        else:
            rows["gen run-in (u<0)"] = gc[gc["u_true"] < 0][SHOW]
            rows["gen anchor (g>=0.98, w=0)"] = gc[(gc["g"] >= 0.98) & (gc["late_w"] == 0)][SHOW]
            rows["gen end (u>0.97)"] = gc[gc["u_true"] > 0.97][SHOW]
        lines += [f"## {cls}", "| set | n | " + " | ".join(s.replace("B1Y_", "") for s in SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
        for name, d in rows.items():
            cells = []
            for s in SHOW:
                lo, md, hi = _q(d[s])
                cells.append(f"{md:.3g} [{lo:.3g}-{hi:.3g}]")
            lines.append(f"| {name} | {len(d)} | " + " | ".join(cells) + " |")
            res.setdefault(cls, {})[name] = {s: _q(d[s]).tolist() for s in SHOW}
        lines.append("")
    # real failure trace
    lines += ["## real ball failure trace (n=1; rpm column invalid; B1Y)", "| segment | windows | " + " | ".join(s.replace("B1Y_", "") for s in SHOW) + " |",
              "|---|---|" + "---|" * len(SHOW)]
    for name, (lo_i, hi_i) in P.SEGS.items():
        seg = trace[lo_i:hi_i][:, 8:16]
        cells = [f"{np.median(seg[:, j]):.3g} [{np.percentile(seg[:, j], 5):.3g}-{np.percentile(seg[:, j], 95):.3g}]" for j in range(8)]
        lines.append(f"| {name} | {hi_i - lo_i} | " + " | ".join(cells) + " |")
        res.setdefault("failure_trace", {})[name] = np.median(seg, axis=0).tolist()
    # late amplification and ratio checks
    ball = b1[(b1["fault_class"] == "ball")]
    ratio_gen = ball[ball["u_true"] > 0.97]["B1X_rms"].median() / ball[(ball["g"] >= 0.98) & (ball["late_w"] == 0)]["B1X_rms"].median()
    lines += ["", "## Late amplification (measured on the real ball failure trace, B1X rms)",
              f"- measured rms(seg2 = windows 121-143) / rms(seg0 = windows 0-27) = **{prof['rms_ratio_end_over_seg0']:.2f}** "
              f"(seg1/seg0 = {prof['rms_ratio_mid_over_seg0']:.2f}; envelope seg2/seg0 = {prof['env_ratio_end_over_seg0']:.2f}) {prof['tag']}",
              f"- generator ball units: B1X rms at u>0.97 over the anchor level, median ratio = {ratio_gen:.2f} (per-unit late scale lognormal sigma {K.LATE_SIGMA}; shape/anchor jitter blur it).", ""]
    res["late_amp_measured"] = prof["rms_ratio_end_over_seg0"]
    res["late_amp_generator_ball_B1X"] = float(ratio_gen)
    # honest differences
    gh = b1[b1["faulted_bearing"] == 0]
    lines += ["## Where the generator differs from real data (stated plainly)",
              "- Anchor levels are the real fault-run levels BY CONSTRUCTION (resampled windows); agreement there is not a validation.",
              "- Between healthy and anchor (g in (0,1)) the path is log-linear in every feature (assumed; no real mild/incipient data exists, real faults are already 5-19x healthy). "
              "Kurtosis/crest in that region interpolate between the real endpoints; in real windows healthy kurtosis is HIGHER than faulted (near-noise-floor signal), so the generator's kurtosis FALLS with damage until the late bump, the opposite of the textbook 'impulsive early' picture; unverified for the real rig.",
              "- Beyond the anchor, behaviour comes from ONE failure trace (a ball bearing at 3600 rpm with an invalid rpm column) and is applied to all four fault classes; the late shape of inner/outer/cage failures is unknown.",
              "- Window-to-window noise is real (block bootstrap of pool windows); with only 36-73 test-block windows per pool, test units re-use the same few real windows with different nuisance/curves, so test units are NOT independent trajectories.",
              "- Per-unit mount/gain nuisance (sigma 0.20 / 0.10 log) is of the same order as the observed healthy run-to-run spread below (median smaller, max larger; only 2 runs).",
              "- Time axis: 10-min snapshots with a life scale set by ISO 281 at an assumed duty (see rig_anchor_scale_check.md); the real rig windows are contiguous 0.67 s windows.", ""]
    # observed healthy run-to-run spread
    h18 = real[(real["run_id"].isin(["run_20261114_230240", "run_20261114_234701"]))]
    meds = h18.groupby("run_id")[[c for c in R.FEATURE_COLUMNS if c.startswith("B1Y")]].median()
    spread = np.abs(np.log(meds.iloc[0] / meds.iloc[1]))
    lines.append(f"Observed log difference of per-run medians between the two healthy 1800 rpm runs (B1Y, 8 features): median {spread.median():.2f}, max {spread.max():.2f} (n=2 runs, so only an indication) vs MOUNT_SIGMA {K.MOUNT_SIGMA}.")
    res["observed_run_to_run_logdiff"] = {"median": float(spread.median()), "max": float(spread.max())}
    out = REPORTS_DIR / "rig_anchor_plausibility.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        for ax, cls in zip(axs.ravel(), K.FAULTS):
            g1 = b1[(b1["fault_class"] == cls) & (b1["rpm"] == 1800)]
            for uid, d in list(g1.groupby("rig_unit"))[:12]:
                ax.plot(d["u_true"].where(d["u_true"] >= 0, -0.1), d["B1Y_rms"], lw=0.5, alpha=0.5, color="tab:blue")
            rw = real[(real["label"] == cls) & (real["rpm"].round(-2) == 1800)]
            ax.axhline(rw["B1Y_rms"].median(), color="tab:red", ls="--", label="real fault run median (B1Y rms)")
            ax.axhline(real[(real["label"] == "healthy") & (real["rpm"].round(-2) == 1800)]["B1Y_rms"].median(), color="tab:green", ls=":", label="real healthy median")
            ax.axhline(np.median(trace[121:144, 8]), color="k", ls="-.", label="real ball failure trace seg2")
            ax.set_yscale("log")
            ax.set_title(f"{cls} @1800: 12 generator units, B1Y rms vs u")
            ax.set_xlabel("u (true elapsed fraction; -0.1 = run-in)")
            ax.set_ylabel("g")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(K.PROVENANCE, fontsize=8)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "rig_anchor_plausibility.png", dpi=110)
    except Exception as e:                                                  # pragma: no cover
        print("figure skipped:", e)
    print(out)


if __name__ == "__main__":
    main()
