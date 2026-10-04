"""Model selection on TRAIN + VAL units only (test units are never loaded here).

rul  Leave-one-bearing-out over the run-to-failure train+val bearings: for each held-out bearing,
     N-HiTS is retrained on the others and forecasts the held-out bearing (out-of-fold), and the
     failure thresholds come from the other bearings only. The RUL estimator grid is then scored
     with the S7 metric on all out-of-fold records. Small N makes LOBO the honest choice.
tft  Small grid over TFT settings, each scored by macro-F1 on VAL units.
Results -> reports/model_selection_<what>_<ts>.json (seed, commit, command, counts).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import models, rul, train
from .config import REPORTS_DIR, SEED


def lobo_records(pre: pd.DataFrame, nc: models.NHiTSConfig, tft_model, work: Path,
                 epochs: int = 12) -> pd.DataFrame:
    fitdf = pre[pre["split"].isin(["train", "val"])]
    rtf_units = sorted(fitdf.loc[fitdf["run_to_failure"], "unit_id"].unique())
    preds = train.tft_predict(tft_model, fitdf) if tft_model is not None else None
    out = []
    for k, uid in enumerate(rtf_units):
        trn = train.rtf_frames(fitdf[fitdf["unit_id"] != uid])
        cfg = copy.deepcopy(nc)
        cfg.max_epochs = epochs
        fold = work / f"fold_{k:02d}"
        train.seed_all(SEED + k)
        path, _ = train.train_nhits(trn, trn.iloc[:0], cfg, fold)       # fixed epochs, no val
        nh = train.load_model(path)
        recs = train.rul_records(nh, fitdf[fitdf["unit_id"] == uid], preds)
        recs["fold_end_hi"] = [json.dumps(train.end_hi_records(trn))] * len(recs)
        out.append(recs)
        print(f"  LOBO {k + 1}/{len(rtf_units)} {uid}: {len(recs)} records", flush=True)
    return pd.concat(out, ignore_index=True)


def grid_rul(recs: pd.DataFrame) -> tuple[rul.RULCalibration, list[dict]]:
    table, best = [], None
    for m in ("trend", "onset", "blend"):
        for hw in (12, 24, 48):
            for w in ((0.25, 0.5, 0.75) if m == "blend" else (0.5,)):
                for um in ((0.02, 0.05, 0.1) if m != "trend" else (0.05,)):
                    for q in (0.25, 0.5, 0.75):
                        scored = []
                        for fold_hi, g in recs.groupby("fold_end_hi"):
                            # thresholds fit at quantile q on the fold's own training bearings only
                            th, pooled = rul.fit_thresholds(json.loads(fold_hi), quantile=q)
                            c = rul.RULCalibration(method=m, history_windows=hw, blend_w=w, u_min=um,
                                                   class_thresholds=th, pooled_threshold=pooled)
                            scored.append(train.apply_rul(c, g))
                        s = pd.concat(scored)
                        mape, per = train.s7_mape(s)
                        table.append({"method": m, "history": hw, "blend_w": w, "u_min": um, "q": q,
                                      "lobo_mape": mape, "per_bearing": per})
                        if np.isfinite(mape) and (best is None or mape < best["lobo_mape"]):
                            best = table[-1]
    ll = nested_loglin(recs, best)
    table.append(ll)
    if np.isfinite(ll["lobo_mape"]) and ll["lobo_mape"] < best["lobo_mape"]:
        best = ll
    return best, table


def _fold_scored(recs: pd.DataFrame, hw: int, q: float, method="trend", coef=None) -> pd.DataFrame:
    out = []
    for fold_hi, g in recs.groupby("fold_end_hi"):
        th, pooled = rul.fit_thresholds(json.loads(fold_hi), quantile=q)
        c = rul.RULCalibration(method=method, history_windows=hw, class_thresholds=th, pooled_threshold=pooled,
                               loglin_coef=list(coef) if coef is not None else [])
        out.append(train.apply_rul(c, g))
    return pd.concat(out)


def nested_loglin(recs: pd.DataFrame, base: dict) -> dict:
    """'loglin' estimator scored honestly: for each bearing, coefficients are fitted on the OTHER
    bearings' out-of-fold degradation-window records, then applied to it (nested LOBO)."""
    hw, q = base["history"], base["q"]
    s = _fold_scored(recs, hw, q)
    d = s[s["in_deg"] & (s["rul_true"] > 0) & s["ll_feats"].notna()]
    if d["unit_id"].nunique() < 3:
        return {"method": "loglin", "lobo_mape": float("nan")}
    X = np.array(d["ll_feats"].tolist(), float)
    y = np.log(d["rul_true"].to_numpy())
    g = d["unit_id"].to_numpy()
    preds = []
    for b in np.unique(g):
        coef = rul.fit_loglin(X[g != b], y[g != b], g[g != b])
        sb = _fold_scored(recs[recs["unit_id"] == b], hw, q, "loglin", coef)
        preds.append(sb)
    sc = pd.concat(preds)
    mape, per = train.s7_mape(sc)
    coef_all = rul.fit_loglin(X, y, g)
    return {"method": "loglin", "history": hw, "blend_w": 0.5, "u_min": 0.05, "q": q, "lobo_mape": mape,
            "per_bearing": per, "loglin_coef": coef_all.tolist(), "loglin_features": rul.LOGLIN_FEATURES,
            "note": "nested LOBO: coefficients for each held-out bearing fitted on the others only"}


def run_rul(pre: pd.DataFrame, nc: models.NHiTSConfig, tft_ckpt: Path, work: Path, epochs: int = 12) -> dict:
    tft = train.load_model(tft_ckpt) if tft_ckpt and Path(tft_ckpt).exists() else None
    recs = lobo_records(pre, nc, tft, work, epochs)
    recs.to_pickle(work / "lobo_records.pkl")
    best, table = grid_rul(recs)
    return {"best": best, "grid": table, "nhits_config": nc.to_dict(), "lobo_epochs": epochs,
            "bearings": sorted(recs["unit_id"].unique().tolist())}


def run_tft(pre: pd.DataFrame, grid: list[dict], work: Path) -> list[dict]:
    res = []
    va = pre[pre["split"] == "val"]
    for i, g in enumerate(grid):
        cfg = models.TFTConfig(**g)
        train.seed_all()
        path, meta = train.train_tft(pre[pre["split"] == "train"], va, cfg, work / f"tft_{i:02d}")
        p = train.tft_predict(train.load_model(path), va)
        j = p.merge(va[["unit_id", "window_index", "observable_class", "dataset"]], on=["unit_id", "window_index"])
        y, yh = j["observable_class"].astype(str), j["pred_class"]
        pres = sorted(set(y))
        per_ds = {ds: float(f1_score(gg["observable_class"].astype(str), gg["pred_class"],
                                     labels=sorted(set(gg["observable_class"].astype(str))), average="macro"))
                  for ds, gg in j.groupby("dataset")}
        res.append({"config": {k: v for k, v in g.items()}, "val_macro_f1": float(f1_score(y, yh, labels=pres, average="macro")),
                    "val_per_dataset_f1": per_ds, "epochs": meta["epochs_run"], "val_loss": meta["best_val_loss"],
                    "ckpt": str(path)})
        print(f"  TFT grid {i + 1}/{len(grid)} {g} -> val macro-F1 {res[-1]['val_macro_f1']:.3f}", flush=True)
    return res


def write(what: str, payload: dict, meta: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = REPORTS_DIR / f"model_selection_{what}_{stamp}.json"
    train.write_json(p, {"generated_utc": stamp, "provenance": meta, **payload})
    return p
