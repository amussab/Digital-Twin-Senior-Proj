"""UP2 (v3) training CLI on SYNTHETIC synth_v3 data. All numbers produced are [SYNTHETIC].

    python -m aiengine.up2_train data                 # generate main + robustness sets, freeze run-level splits
    python -m aiengine.up2_train select-gbm           # XGBoost/LightGBM grid, selection on VALIDATION runs only
    python -m aiengine.up2_train nn --epochs 12       # TFT + N-HiTS (pytorch-forecasting) on the same data
    python -m aiengine.up2_train predict              # ONE pass over the frozen TEST runs (+ robustness sets)

Selection rules (pre-declared, validation runs only):
  classifier : max val macro-F1 (5 classes); tie -> fewer trees
  rul        : min val S7 MAPE (bearing-averaged, degradation window)
  z / stage  : min val max-stage-error, then max within-1 fraction, then min MAE
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import up2
from . import up2_metrics as M
from .config import REPORTS_DIR, SEED
from .synth_v3 import generator as gen

RUN_DIR = up2.RUN_DIR
ROBUST_SETS = {
    # name: (n_runs, seed offset, knob overrides) -- every robustness set is fresh runs never used for fitting
    "synth_v3_noise15": (60, 101, {"noise_scale": 1.5}),
    "synth_v3_rpm3000": (60, 202, {"rpm_modes": (3000.0,)}),
    "synth_v3_hard": (60, 303, {"signature": 0.75, "life_sigma": 0.45, "transmission": (0.08, 0.16),
                                "shape_a": (0.25, 0.35), "shape_p": (2.0, 3.0), "eol_sev": (0.75, 1.25)}),
}
EXTRA_TRAIN = ("synth_v3_train2", 480, 404)     # extra TRAINING-only runs (separate dataset id, never val/test)
Z_CUTS = [0.0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12]
CLS_GRID = [{"kind": k, "T": T} for k in ("xgb", "lgbm") for T in (13, 25)]
RUL_GRID = ([{"kind": k, "NL": 48, "levels": False} for k in ("xgb", "lgbm")] +
            [{"kind": k, "NL": NL, "levels": True} for k in ("xgb", "lgbm") for NL in (48, 96)])


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def meta(extra: dict | None = None) -> dict:
    from .train import git_commit, git_dirty
    return {"provenance": "[SYNTHETIC] synth_v3", "seed": SEED, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "command": "python -m aiengine.up2_train " + " ".join(sys.argv[1:]),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}


def load_main(extra: bool = True) -> pd.DataFrame:
    """Frozen-split main set (+ the extra training-only runs, split='train')."""
    df = up2.prepare(gen.DATASET)
    if not extra:
        return df
    name, n, off = EXTRA_TRAIN
    ex = up2.prepare(name, freeze_splits=False, n_runs=n, seed=SEED + off, run_offset=20000)
    ex["split"] = "train"
    out = pd.concat([df, ex], ignore_index=True)
    out.attrs["provenance"] = {**df.attrs.get("provenance", {}),
                               "extra_training_runs": {**ex.attrs.get("provenance", {}), "split": "train only"}}
    return out


def load_robust(name: str) -> pd.DataFrame:
    n, off, kw = ROBUST_SETS[name]
    k = gen.Knobs(**kw)
    return up2.prepare(name, freeze_splits=False, n_runs=n, seed=SEED + off, knobs=k, run_offset=10000 + off * 100)


# --------------------------------------------------------------------------- predictions per head

def cls_features(df, T):
    X, idx = up2.unit_features(df, T, with_decoder=True)
    tr = up2.merge_truth(idx, df, ["observable_class"])
    y = tr["observable_class"].map({c: i for i, c in enumerate(up2.CLASS_ORDER)}).to_numpy()
    return X, idx, y


def rul_features(df, NL, warm, levels=False):
    X, idx = up2.unit_features(df, NL, with_decoder=False, warm=warm, levels=levels)
    return X, idx


def cls_predict(model, X, idx) -> pd.DataFrame:
    P = model.predict_proba(X)
    out = idx.copy()
    for i, c in enumerate(up2.CLASS_ORDER):
        out[f"p_{c}"] = P[:, i]
    out["pred_class"] = [up2.CLASS_ORDER[i] for i in P.argmax(1)]
    return out


def rul_predict(rm, zm, X, idx, z_cut: float = 0.0) -> pd.DataFrame:
    out = idx.copy()
    out["rul_pred_windows"] = np.maximum(np.expm1(rm.predict(X)), 0.0)
    out["rul_pred_h"] = out["rul_pred_windows"] * gen.CADENCE_H
    out["z_pred"] = zm.predict(X)
    out["stage_pred"] = up2.stage_from_z(out["z_pred"].to_numpy(), z_cut)
    return out


TRUTH = ["dataset", "fault_class", "run_to_failure", "observable_class", "health_stage", "rul_hours", "life_hours",
         "u_true", "in_deg", "in_deg_full", "in_deg_det", "rpm_mode", "split", "bearing_key"]


def score(cls_p: pd.DataFrame, rul_p: pd.DataFrame, df: pd.DataFrame) -> dict:
    c = up2.merge_truth(cls_p, df, TRUTH)
    r = up2.merge_truth(rul_p, df, TRUTH)
    rr = r[r["run_to_failure"]]
    joined = c.merge(rul_p[["unit_id", "window_index", "stage_pred"]], on=["unit_id", "window_index"], how="left")
    return {"classification": M.classification(c), "caught": M.caught_by_stage3(joined),
            "stage": M.stage(rr), "rul": M.rul(rr),
            "stage_healthy_bearings": M.stage(r[~r["run_to_failure"]]) if (~r["run_to_failure"]).any() else {}}


# --------------------------------------------------------------------------- commands

def cmd_data(a) -> int:
    df = load_main()
    rep = {"main": {**df.attrs.get("provenance", {}), "split_check": up2.split_check(df)}}
    for name in ROBUST_SETS:
        r = load_robust(name)
        rep[name] = {**r.attrs.get("provenance", {}), "windows": int(len(r)), "runs": int(r["bearing_key"].nunique())}
    print(json.dumps(rep, indent=1, default=str))
    return 0


def _fit_cls(df, cfg, threads):
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    Xtr, itr, ytr = cls_features(tr, cfg["T"])
    Xva, iva, yva = cls_features(va, cfg["T"])
    m, info = up2.fit_classifier(cfg["kind"], Xtr, ytr, Xva, yva, threads=threads)
    p = cls_predict(m, Xva, iva)
    return m, info, p


def _fit_rul(df, cfg, warm, threads):
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    Xtr, itr = rul_features(tr, cfg["NL"], warm, cfg.get("levels", False))
    Xva, iva = rul_features(va, cfg["NL"], warm, cfg.get("levels", False))
    ttr = up2.merge_truth(itr, tr, ["run_to_failure", "u_true", "rul_windows", "z_target"])
    tva = up2.merge_truth(iva, va, ["run_to_failure", "u_true", "rul_windows", "z_target"])
    dtr = (ttr["run_to_failure"] & (ttr["u_true"] >= 0)).to_numpy()
    dva = (tva["run_to_failure"] & (tva["u_true"] >= 0)).to_numpy()
    rm, rinfo = up2.fit_regressor(cfg["kind"], Xtr[dtr], np.log1p(ttr.loc[dtr, "rul_windows"].to_numpy()),
                                  Xva[dva], np.log1p(tva.loc[dva, "rul_windows"].to_numpy()), threads=threads,
                                  wtr=up2.unit_balanced_weights(ttr.loc[dtr, "unit_id"]),
                                  wva=up2.unit_balanced_weights(tva.loc[dva, "unit_id"]))
    zm, zinfo = up2.fit_regressor(cfg["kind"], Xtr, ttr["z_target"].to_numpy(), Xva, tva["z_target"].to_numpy(),
                                  threads=threads)
    return rm, zm, {"rul": rinfo, "z": zinfo}, rul_predict(rm, zm, Xva, iva)


def cmd_select_gbm(a) -> int:
    df = load_main()
    va = df[df["split"] == "val"]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    res = {"cls": [], "rul": []}
    best_c, best_r = None, None
    prev = RUN_DIR / "cls_grid.pkl"
    if a.reuse_cls and prev.exists():
        with open(prev, "rb") as f:
            res["cls"], best_c = pickle.load(f)
        print("reusing classifier grid from", prev, flush=True)
    for cfg in ([] if best_c is not None else CLS_GRID):
        m, info, p = _fit_cls(df, cfg, a.threads)
        s = M.classification(up2.merge_truth(p, va, TRUTH))
        row = {**cfg, **info, "val_macro_f1": s["macro_f1"], "val_per_class_f1": s["per_class_f1"]}
        print("cls", json.dumps(row), flush=True)
        res["cls"].append(row)
        if best_c is None or s["macro_f1"] > best_c[0]:
            best_c = (s["macro_f1"], cfg, m, p)
    with open(prev, "wb") as f:
        pickle.dump((res["cls"], best_c), f)
    warm = best_c[1]["T"]
    for cfg in RUL_GRID:
        rm, zm, info, p = _fit_rul(df, cfg, warm, a.threads)
        rr = up2.merge_truth(p, va, TRUTH)
        rr = rr[rr["run_to_failure"]]
        sr, ss = M.rul(rr), M.stage(rr)
        row = {**cfg, **info, "val_S7_mape_pct": sr["S7_mape_pct"], "val_percent_of_life_pts": sr["alt_percent_of_life_error_pts"],
               "val_stage_max": ss["max_abs_error"], "val_stage_within1": ss["within1_frac"], "val_stage_mae": ss["mae"]}
        print("rul", json.dumps(row), flush=True)
        res["rul"].append(row)
        key_r = sr["S7_mape_pct"]
        if best_r is None or key_r < best_r[0]:
            best_r = (key_r, cfg, rm, zm)
    # z head = the z regressor fitted with the selected RUL config (both share the N-HiTS slot / one window)
    sel = {"cls": best_c[1], "rul": best_r[1], "z": dict(best_r[1])}
    zmodel = best_r[3]
    # z cut (stage 1 vs >= 2 boundary), chosen on VALIDATION: min (max error, #windows error>=2, -within1)
    va_rtf = va[va["run_to_failure"]]
    Xv, iv = rul_features(va_rtf, sel["rul"]["NL"], warm, sel["rul"].get("levels", False))
    zv = up2.merge_truth(iv.assign(z_pred=zmodel.predict(Xv)), va_rtf, ["health_stage"])
    cut_rows = []
    for cut in Z_CUTS:
        e = np.abs(up2.stage_from_z(zv["z_pred"].to_numpy(), cut) - zv["health_stage"].to_numpy())
        cut_rows.append({"z_cut": cut, "max": int(e.max()), "n_ge2": int((e >= 2).sum()), "within1": float((e <= 1).mean())})
    best_cut = sorted(cut_rows, key=lambda r: (r["max"], r["n_ge2"], -r["within1"], -r["z_cut"]))[0]
    sel["z"]["z_cut"] = best_cut["z_cut"]
    res["z_cut_grid"] = cut_rows
    print("z_cut", json.dumps(cut_rows), flush=True)
    for name, obj in (("cls", {"model": best_c[2], "cfg": best_c[1]}), ("rul", {"model": best_r[2], "cfg": best_r[1]}),
                      ("z", {"model": zmodel, "cfg": sel["z"]})):
        with open(RUN_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    out = {"meta": meta({"selection": "validation runs only", "counts": up2.split_check(df)}), "grid": res, "selected": sel}
    path = REPORTS_DIR / f"model_selection_up2_v3_gbm_{now_tag()}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    (RUN_DIR / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selected", json.dumps(sel), "->", path, flush=True)
    return 0


def load_heads():
    def ld(n):
        with open(RUN_DIR / f"{n}.pkl", "rb") as f:
            return pickle.load(f)
    return ld("cls"), ld("rul"), ld("z")


def predict_gbm(df: pd.DataFrame):
    c, r, z = load_heads()
    T, NL = c["cfg"]["T"], r["cfg"]["NL"]
    Xc, ic, _ = cls_features(df, T)
    cp = cls_predict(c["model"], Xc, ic)
    Xr, ir = rul_features(df, NL, warm=T, levels=r["cfg"].get("levels", False))
    rp = rul_predict(r["model"], z["model"], Xr, ir, z["cfg"].get("z_cut", 0.0))
    return cp, rp


def cmd_predict(a) -> int:
    """The single pass over the frozen TEST runs. Saves per-window predictions; up2_eval re-scores them."""
    tag = now_tag()
    df = load_main()
    test = df[df["split"] == "test"]
    out = {"meta": meta({"tag": tag, "touch": "test runs scored once with the selected heads"})}
    sets = {"test": test, **{n: load_robust(n) for n in ROBUST_SETS}}
    pred_dir = REPORTS_DIR / "up2_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, d in sets.items():
        cp, rp = predict_gbm(d)
        f1, f2 = pred_dir / f"up2_v3_gbm_{name}_{tag}.cls.csv.gz", pred_dir / f"up2_v3_gbm_{name}_{tag}.rul.csv.gz"
        cp.to_csv(f1, index=False, float_format="%.6g")
        rp.to_csv(f2, index=False, float_format="%.6g")
        files[name] = {"cls": f1.name, "rul": f2.name}
        s = score(cp, rp, d)
        print(name, "macroF1 %.4f  S7 %.2f  stage max %s within1 %.4f  caught %s/%s" % (
            s["classification"]["macro_f1"], s["rul"]["S7_mape_pct"], s["stage"]["max_abs_error"],
            s["stage"]["within1_frac"], s["caught"]["caught_by_stage3"], s["caught"]["rtf_bearings"]), flush=True)
    out["files"] = files
    (pred_dir / f"up2_v3_gbm_{tag}.index.json").write_text(json.dumps(out, indent=1))
    (RUN_DIR / "latest_predict.json").write_text(json.dumps({"tag": tag, "files": files}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2_train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    s = sub.add_parser("select-gbm")
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--reuse-cls", action="store_true", help="reuse the saved classifier grid (same data, same rule)")
    sub.add_parser("predict")
    a = ap.parse_args(argv)
    return {"data": cmd_data, "select-gbm": cmd_select_gbm, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
