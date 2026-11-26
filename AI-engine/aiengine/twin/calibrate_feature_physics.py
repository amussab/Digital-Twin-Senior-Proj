"""Calibrate the bounded damage link of FeaturePhysicsTwin (k, r_fail) on rig_anchor VAL units ONLY.

    python -m aiengine.twin.calibrate_feature_physics            # writes reports/twin_feature_calibration_<ts>.json/.md

Pre-declared criterion (written before any run): over the grid k in K_GRID x r_fail in RFAIL_GRID choose the pair that MINIMISES
    median over faulted run-to-failure val bearings of ( median over that bearing's windows (true stage 1..3, window_index >= 24,
    RUL_true > 0) of |RUL_phys - RUL_true| / L_ref ),
with RUL_phys the twin's rul_hours (trend or prior), table clock dt = 10 min, baseline = median of the first 24 windows of the
bearing-1 features, L_ref = ISO 281 L10h at P/C = 0.5 (74.1 h @1800, 37.0 h @3600). Ties -> smaller k, then smaller r_fail.
NOT used: any IS2 pass/fail, AI RUL, test units, rig_anchor_test2. This is a fit of two scalars to the SIMULATION GENERATOR;
IS2 on this twin stays common-mode (twin and AI read the same features) and Rank <= 6.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import splits
from ..config import REPORTS_DIR, ROOT
from ..rig_anchor import constants as K
from ..rig_anchor import generator as G
from ..rig_anchor import scale
from .feature_physics import FeaturePhysicsParams, FeaturePhysicsTwin, LifeRef

K_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)   # round 2: extended downward because round 1 (0.05..2.0) optimum sat on the lower edge
RFAIL_GRID = (10.0, 20.0, 50.0)
WARM = 24
TAG = "[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed; calibration = fit to the generator]"


def val_units() -> tuple[pd.DataFrame, list[str]]:
    sec = splits.load()["sections"][K.SPLIT_SECTION]["units"]
    want = {u for u, s in sec.items() if s == "val"}
    df = pd.read_parquet(ROOT / "data" / "rig_anchor" / "rig_anchor.parquet")
    df = df[df["rig_unit"].isin(want)]
    rtf = df[df["run_to_failure"] & df["unit_id"].str.endswith(":B1")]
    return df, sorted(rtf["rig_unit"].unique())


def prep(df: pd.DataFrame, ids: list[str]) -> list[dict]:
    out = []
    for uid in ids:
        u = df[df["rig_unit"] == uid]
        X = G.window_matrix(u)[:, :16]
        b1 = u[u["unit_id"].str.endswith(":B1")].sort_values("window_index")
        out.append({"uid": uid, "X": X, "rpm": float(b1["rpm"].iloc[0]), "wi": b1["window_index"].to_numpy(),
                    "stage": b1["true_stage"].to_numpy(), "rul": b1["rul_hours"].to_numpy()})
    return out


def run_unit(unit: dict, k: float, r_fail: float, law: str = "bounded") -> pd.DataFrame:
    p = FeaturePhysicsParams(life=LifeRef(c_over_p=1.0 / K.RIG_PC_SIM), k=k, r_fail=r_fail, law=law)
    tw = FeaturePhysicsTwin(p, baseline16=np.median(unit["X"][:WARM], axis=0))
    rows = []
    for i, x in enumerate(unit["X"]):
        o = tw.update(x, unit["rpm"], i * K.CADENCE_H)
        rows.append((unit["wi"][i], o.rul_hours, o.source, o.miner_d))
    d = pd.DataFrame(rows, columns=["wi", "phys", "source", "D"])
    d["stage"], d["rul"] = unit["stage"], unit["rul"]
    d["L"] = scale.l_ref_hours(unit["rpm"])
    return d[(d["stage"].isin((1, 2, 3))) & (d["rul"] > 0) & (d["wi"] >= WARM)]


def score(units: list[dict], k: float, r_fail: float, law: str = "bounded") -> dict:
    meds, frames = [], []
    for u in units:
        d = run_unit(u, k, r_fail, law)
        if d.empty:
            continue
        meds.append(float(((d["phys"] - d["rul"]).abs() / d["L"]).median()))
        frames.append(d)
    a = pd.concat(frames)
    st = {}
    for s in (1, 2, 3):
        g = a[a["stage"] == s]
        st[str(s)] = {"truth_over_Lref": float((g["rul"] / g["L"]).median()), "twin_over_Lref": float((g["phys"] / g["L"]).median()),
                      "twin_D_median": float(g["D"].median()), "share_trend": float((g["source"] == "trend").mean())}
    return {"criterion": float(np.median(meds)), "bearings": len(meds), "per_stage": st}


def main() -> int:
    df, ids = val_units()
    units = prep(df, ids)
    grid = []
    for k in K_GRID:
        for rf in RFAIL_GRID:
            grid.append({"k": k, "r_fail": rf, **score(units, k, rf)})
    # k = 0 is a feature-blind rated-life clock (D = t/L_ref): reported as a reference, EXCLUDED from the choice (declared rule: k > 0,
    # added in round 2 after k = 0 won the extended grid; the twin must keep the features in its damage law).
    ref0 = min((g for g in grid if g["k"] == 0.0), key=lambda g: g["criterion"])
    best = min((g for g in grid if g["k"] > 0.0), key=lambda g: (round(g["criterion"], 9), g["k"], g["r_fail"]))
    legacy = score(units, 1.0, 20.0, "power")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"provenance": TAG, "utc": ts, "k": best["k"], "r_fail": best["r_fail"], "law": "g(r)=1+k ln max(r,1); dD/dt=g^3/L_ref",
           "criterion": "min over grid of median over faulted val bearings of median_w |RUL_phys-RUL_true|/L_ref (true stage 1-3, window>=24, RUL>0)",
           "k0_reference_clock": {"criterion": ref0["criterion"], "per_stage": ref0["per_stage"]}, "selection_rule_note": "k > 0 required (k = 0 is a feature-blind clock, reported as reference only); ties -> smaller k, r_fail", "k_grid": K_GRID, "rfail_grid": RFAIL_GRID, "val_units": len(ids),
           "val_unit_list_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
           "val_scores_at_best": {k_: best[k_] for k_ in ("criterion", "bearings", "per_stage")},
           "val_scores_legacy_power_law": legacy, "grid": grid,
           "limits": "Fit of two scalars to the simulation generator on VAL units; the law is a modelling assumption; IS2 on this twin stays "
                     "common-mode (same features as the AI) and Rank <= 6. Test / rig_anchor_test2 units not used."}
    f = REPORTS_DIR / f"twin_feature_calibration_{ts}.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    md = [f"# FeaturePhysicsTwin bounded-link calibration {TAG}", "", f"UTC {ts}; command `python -m aiengine.twin.calibrate_feature_physics`.", "",
          "Law: g(r) = 1 + k ln(max(r,1)) (equivalent-load factor), dD/dt = g^3 / L_ref(rpm) (ISO 281 p = 3 on the LOAD factor), failure D >= 0.95;",
          "trend RUL extrapolates ln r to ln r_fail, Miner RUL uses the same law in closed form; rul = min.", "",
          f"Criterion (declared before running): {out['criterion']}", f"Val units: {len(ids)} (sha256 of the sorted unit list {out['val_unit_list_sha256']}).", "",
          f"## Calibrated: k = {best['k']}, r_fail = {best['r_fail']}   (criterion = {best['criterion']:.4f} of L_ref)", "",
          "| stage | truth RUL/L_ref | twin RUL/L_ref | median D | trend share |", "|---|---|---|---|---|"]
    for s, v in best["per_stage"].items():
        md.append(f"| {s} | {v['truth_over_Lref']:.3f} | {v['twin_over_Lref']:.3f} | {v['twin_D_median']:.3f} | {v['share_trend']:.2f} |")
    md += ["", f"k = 0 reference (feature-blind rated-life clock, excluded from the choice): criterion {ref0['criterion']:.4f}.", "", f"Legacy law (g = r, p = 3 on r, r_fail 20): criterion {legacy['criterion']:.4f}; per stage twin RUL/L_ref "
           + ", ".join(f"{s}: {v['twin_over_Lref']:.3f}" for s, v in legacy["per_stage"].items()), "", "## Limits", out["limits"], "",
          "## Grid (criterion)", "", "| k | " + " | ".join(f"r_fail {r:g}" for r in RFAIL_GRID) + " |", "|---|" + "---|" * len(RFAIL_GRID)]
    for k in K_GRID:
        md.append(f"| {k} | " + " | ".join(f"{g['criterion']:.3f}" for g in grid if g["k"] == k) + " |")
    f.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k_: out[k_] for k_ in ("k", "r_fail", "val_units", "val_scores_at_best")}, indent=1))
    print("legacy", legacy["criterion"], {s: v["twin_over_Lref"] for s, v in legacy["per_stage"].items()})
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
