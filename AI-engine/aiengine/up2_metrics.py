"""UP2 spec metrics (pure functions over prediction frames). Definitions are stated in every report.

S7 / ICS1 (spec metric, absolute hours, DESIGN.md section 7 applied to the generator's TRUE onset):
    degradation window = windows with true onset <= t <= T_fail - 0.1 (T_fail - t_onset)
    MAPE = mean over test run-to-failure bearings of [mean over that bearing's degradation-window
           windows of |RUL_pred_h - RUL_true_h| / RUL_true_h] * 100
  Alternatives, reported alongside, never substituted:
    percent-of-life error = mean |RUL_pred_h - RUL_true_h| / life_h * 100 (points)
    MAPE over the full degradation interval (no 10 % tail exclusion)
    MAPE with the DESIGN.md detected-onset window (in_deg_det)
IS3a macro-F1 over {healthy, outer_race, inner_race, ball, cage}, every scored window of every test bearing.
IS3b stage error = |stage_pred - stage_true| over every scored window of the test run-to-failure bearings
     (stage_true = generator truth); literal max reported, plus fraction within +/-1.
IS3c caught by stage 3: first window where the predicted class equals the bearing's fault for 3
     consecutive windows; caught_by_stage3 = true stage at that window <= 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]
DETECTION_PERSISTENCE = 3


def classification(p: pd.DataFrame) -> dict:
    """p: columns observable_class, pred_class (+ optional dataset)."""
    y, yh = p["observable_class"].astype(str).to_numpy(), p["pred_class"].astype(str).to_numpy()
    out = {"n_windows": int(len(p)), "n_bearings": int(p["unit_id"].nunique()),
           "macro_f1": float(f1_score(y, yh, labels=CLASSES, average="macro", zero_division=0)),
           "accuracy": float((y == yh).mean()),
           "per_class_f1": {c: float(f1_score(y, yh, labels=[c], average="macro", zero_division=0)) for c in CLASSES},
           "per_class_recall": {c: float((yh[y == c] == c).mean()) if (y == c).any() else None for c in CLASSES},
           "per_class_support": {c: int((y == c).sum()) for c in CLASSES},
           "confusion": {"labels": CLASSES, "rows": "true", "cols": "pred",
                         "matrix": confusion_matrix(y, yh, labels=CLASSES).tolist()}}
    f = y != "healthy"
    out["missed_fault_rate"] = float((yh[f] == "healthy").mean()) if f.any() else None
    out["false_alarm_rate"] = float((yh[~f] != "healthy").mean()) if (~f).any() else None
    return out


def caught_by_stage3(p: pd.DataFrame) -> dict:
    """p: per-window rows with unit_id, window_index, fault_class, run_to_failure, health_stage, pred_class."""
    rows = []
    for uid, g in p[p["run_to_failure"]].groupby("unit_id"):
        g = g.sort_values("window_index")
        fc = str(g["fault_class"].iloc[0])
        ok = g["pred_class"].astype(str).to_numpy() == fc
        streak, at = 0, None
        for i, v in enumerate(ok):
            streak = streak + 1 if v else 0
            if streak >= DETECTION_PERSISTENCE:
                at = i
                break
        st = int(g["health_stage"].iloc[at]) if at is not None else None
        rows.append({"unit_id": uid, "fault_class": fc, "caught": at is not None,
                     "caught_at_window": int(g["window_index"].iloc[at]) if at is not None else None,
                     "caught_at_stage": st, "caught_by_stage3": bool(st is not None and st <= 3),
                     "caught_at_u": float(g["u_true"].iloc[at]) if at is not None and "u_true" in g else None})
    fr = pd.DataFrame(rows)
    out = {"rtf_bearings": int(len(fr)),
           "caught_by_stage3": int(fr["caught_by_stage3"].sum()) if len(fr) else 0,
           "caught_by_stage3_frac": float(fr["caught_by_stage3"].mean()) if len(fr) else float("nan"),
           "never_caught": int((~fr["caught"]).sum()) if len(fr) else 0}
    if len(fr):
        out["caught_at_stage_hist"] = {str(k): int(v) for k, v in fr["caught_at_stage"].value_counts(dropna=False).sort_index().items()}
        out["per_class_by_stage3"] = {c: f"{int(g['caught_by_stage3'].sum())}/{len(g)}" for c, g in fr.groupby("fault_class")}
        out["false_early_calls"] = None
    out["per_unit"] = rows
    return out


def stage(p: pd.DataFrame) -> dict:
    """p: rows of run-to-failure test bearings with health_stage (truth) and stage_pred."""
    e = np.abs(p["stage_pred"].to_numpy(int) - p["health_stage"].to_numpy(int))
    hist = pd.Series(e).value_counts().sort_index()
    out = {"n_windows": int(len(e)), "max_abs_error": int(e.max()) if len(e) else None,
           "within1_frac": float((e <= 1).mean()) if len(e) else None, "mae": float(e.mean()) if len(e) else None,
           "exact_frac": float((e == 0).mean()) if len(e) else None,
           "error_hist": {str(int(k)): int(v) for k, v in hist.items()}}
    if len(e):
        bad = p[e >= 2]
        out["windows_error_ge2"] = int(len(bad))
        out["bearings_with_error_ge2"] = int(bad["unit_id"].nunique()) if len(bad) else 0
    return out


def rul(p: pd.DataFrame, col: str = "rul_pred_h") -> dict:
    """p: rows of run-to-failure test bearings with rul_hours, life_hours, u_true, in_deg, in_deg_full, in_deg_det."""
    def mape(window: str):
        per = {}
        for uid, g in p[p[window] & (p["rul_hours"] > 0)].groupby("unit_id"):
            per[uid] = float(np.mean(np.abs(g[col] - g["rul_hours"]) / g["rul_hours"]) * 100)
        return (float(np.mean(list(per.values()))) if per else float("nan")), per
    m, per = mape("in_deg")
    d = p[p["in_deg"] & (p["rul_hours"] > 0)]
    err = (d[col] - d["rul_hours"]).to_numpy()
    rel = np.abs(err) / d["rul_hours"].to_numpy()
    bins = np.linspace(0, 0.9, 10)
    by_u = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = d[(d["u_true"] >= lo) & (d["u_true"] < hi)]
        if len(g):
            r = np.abs(g[col] - g["rul_hours"]) / g["rul_hours"]
            by_u[f"{lo:.1f}-{hi:.1f}"] = {"mape_pct": float(r.mean() * 100), "median_ape_pct": float(r.median() * 100),
                                          "bias_pct": float(((g[col] - g["rul_hours"]) / g["rul_hours"]).mean() * 100),
                                          "n": int(len(g))}
    vals = np.array(list(per.values()))
    out = {"S7_mape_pct": m, "n_bearings": len(per), "n_windows": int(len(d)),
           "per_bearing_mape_pct": per,
           "per_bearing_summary": {"median": float(np.median(vals)), "p90": float(np.quantile(vals, 0.9)),
                                   "max": float(vals.max()), "frac_le_15": float((vals <= 15).mean())} if len(vals) else {},
           "window_ape_pct": {"median": float(np.median(rel) * 100), "p90": float(np.quantile(rel, 0.9) * 100)} if len(rel) else {},
           "rmse_hours": float(np.sqrt(np.mean(err ** 2))) if len(err) else None,
           "alt_percent_of_life_error_pts": float(np.mean(np.abs(err) / d["life_hours"].to_numpy()) * 100) if len(err) else None,
           "alt_mape_full_degradation_pct": mape("in_deg_full")[0],
           "error_vs_life_fraction": by_u}
    if "in_deg_det" in p:
        out["alt_mape_detected_onset_window_pct"] = mape("in_deg_det")[0]
    if "rpm_mode" in p:
        out["per_rpm_mode"] = {str(int(k)): mape_sub(g, col) for k, g in p.groupby("rpm_mode")}
    if "fault_class" in p:
        out["per_fault_class"] = {k: mape_sub(g, col) for k, g in p.groupby("fault_class")}
    return out


def mape_sub(p: pd.DataFrame, col: str) -> float:
    per = [np.mean(np.abs(g[col] - g["rul_hours"]) / g["rul_hours"]) * 100
           for _, g in p[p["in_deg"] & (p["rul_hours"] > 0)].groupby("unit_id")]
    return float(np.mean(per)) if per else float("nan")
