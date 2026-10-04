"""SECONDARY robustness check (coordinator decision 4): leave-one-bearing-out over ALL eligible
XJTU-SY run-to-failure bearings (train + val + test). This is NOT the pre-registered S7 number
(that is scored on the frozen test bearings only, evaluate.py); it shows how much S7 depends on
which bearings happen to be held out.

Per fold (held-out bearing b):
  - TFT retrained on the other XJTU bearings (fixed epochs, no early stopping) -> class for the
    RUL threshold; the held-out bearing's own class is never seen.
  - N-HiTS retrained on the other run-to-failure bearings (fixed epochs).
  - RUL estimator selected INSIDE the fold: the select.grid_rul grid (incl. nested loglin) is
    scored on the OTHER bearings' records, forecast by this fold's N-HiTS (in-sample forecasts:
    the low-capacity estimator grid is chosen on bearings the forecaster saw; stated, not hidden).
    Thresholds come from the other bearings only.
  - The held-out bearing is scored with the pre-registered S7 metric.
Eligible = run-to-failure, >= encoder+horizon windows, onset detected. Mixed-fault bearings are
eligible for RUL (their class is never used for classification training).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import models, rul, select, train
from .config import SEED


def eligible(df: pd.DataFrame, nc: models.NHiTSConfig) -> list[str]:
    x = df[(df["dataset"] == "xjtu_sy") & df["run_to_failure"]]
    g = x.groupby("unit_id").agg(n=("window_index", "size"), onset=("onset_index", "first"))
    need = nc.encoder_length + nc.prediction_length
    return sorted(g[(g["n"] >= need) & (g["onset"] >= 0)].index)


def run_fold(df: pd.DataFrame, uid: str, k: int, nc: models.NHiTSConfig, tc: models.TFTConfig,
             work: Path, epochs: int) -> dict:
    xj = df[df["dataset"] == "xjtu_sy"]
    others = xj[xj["unit_id"] != uid]
    held = xj[xj["unit_id"] == uid]
    fold = work / f"fold_{k:02d}"
    train.seed_all(SEED + k)
    tcf = copy.deepcopy(tc)
    tcf.max_epochs = epochs
    tpath, _ = train.train_tft(others, others.iloc[:0], tcf, fold)
    tft = train.load_model(tpath)
    ncf = copy.deepcopy(nc)
    ncf.max_epochs = epochs
    trn = train.rtf_frames(others)
    npath, _ = train.train_nhits(trn, trn.iloc[:0], ncf, fold)
    nh = train.load_model(npath)
    # estimator selection on the other bearings (in-sample forecasts), thresholds from the others
    recs_o = train.rul_records(nh, others, train.tft_predict(tft, others))
    end_hi = train.end_hi_records(trn)
    recs_o["fold_end_hi"] = json.dumps(end_hi)
    best, _ = select.grid_rul(recs_o)
    th, pooled = rul.fit_thresholds(end_hi, quantile=best["q"])
    calib = rul.RULCalibration(method=best["method"], history_windows=best["history"], blend_w=best["blend_w"],
                               u_min=best["u_min"], class_thresholds=th, pooled_threshold=pooled,
                               loglin_coef=best.get("loglin_coef", []))
    recs_h = train.rul_records(nh, held, train.tft_predict(tft, held))
    sc = train.apply_rul(calib, recs_h)
    mape, _ = train.s7_mape(sc)
    mape_full, _ = train.s7_mape(sc, window="in_deg_full")
    st = sc[sc["health_stage"] >= 1]
    e = np.abs(st["stage_pred"] - st["health_stage"])
    out = {"unit_id": uid, "fault_class": str(held["fault_class"].iloc[0]),
           "frozen_split": str(held["split"].iloc[0]), "S7_mape_pct": mape,
           "mape_full_degradation_pct": mape_full,
           "selected_in_fold": {k2: v for k2, v in best.items() if k2 not in ("per_bearing",)},
           "selection_in_fold_mape_pct": best["lobo_mape"],
           "stage_within1_frac": float((e <= 1).mean()), "stage_mae": float(e.mean()),
           "stage_max_error": int(e.max()), "epochs": epochs}
    sc.drop(columns=["hi_hist", "forecast"]).to_csv(fold / "heldout_rul_preds.csv.gz", index=False)
    train.write_json(fold / "result.json", out)
    return out


def aggregate(work: Path) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(work.glob("fold_*/result.json"))]
    v = np.array([r["S7_mape_pct"] for r in rows], float)
    f = np.isfinite(v)
    return {"folds": rows, "n_folds": len(rows),
            "mean_S7_mape_pct": float(np.mean(v[f])) if f.any() else float("nan"),
            "median_S7_mape_pct": float(np.median(v[f])) if f.any() else float("nan"),
            "frac_bearings_meeting_15pct": float((v[f] <= 15).mean()) if f.any() else float("nan"),
            "note": "SECONDARY robustness table (LOBO over all eligible XJTU-SY run-to-failure bearings, "
                    "incl. the frozen test bearings). Not the pre-registered S7 headline."}
