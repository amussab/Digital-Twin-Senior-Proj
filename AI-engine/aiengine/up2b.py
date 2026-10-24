"""UP2 v3b: more training runs, an improved RUL head, and causal post-processing (all [SYNTHETIC], synth_v3 generator).

    python -m aiengine.up2b data       # generate train3 / val2 / test2, register val2 + test2 in splits.json
    python -m aiengine.up2b select     # fit candidates on TRAIN runs, select everything on VALIDATION runs only
    python -m aiengine.up2b predict    # ONE pass: old test (2nd look, disclosed) + fresh test2 (1st look) + robustness
    python -m aiengine.up2b export     # models/v3b_*.onnx + model_contract_v3b.json + golden_vectors_v3b.json

Same generator, schema and knob ranges as v3 (aiengine/synth_v3/generator.py, unchanged). New run sets:
  synth_v3_train3 : TRAIN only (never val/test), new seed, run ids 30000+
  synth_v3_val2   : VALIDATION only, registered in splits.json (whole dataset -> val), run ids 40000+
  synth_v3_test2  : FRESH TEST, registered in splits.json (whole dataset -> test) BEFORE any v3b model was fitted;
                    run ids 50000+; scored exactly once by `predict`.
The v3 test runs (splits.json synth_v3, split test) are never used for fitting/selection; they were scored once
for v3 and are scored again here (second look, disclosed in every report).

Post-processing (part of the deployed engine; mirrored in engine.py and described in model_contract_v3b.json):
  RUL   (per bearing, causal, in WINDOW units, state = previous output r_prev):
          prior = max(r_prev - 1, 0)        (one window elapsed)
          r     = exp(a*log1p(r_raw) + (1-a)*log1p(prior)) - 1     (first output: r = r_raw)
          rul_windows_out = r * (scale_early if stage_out <= 2 else scale_late)   (scales chosen on validation;
                                                                                   r_prev keeps the UNscaled r)
  stage (per bearing, causal, state = previous output s_prev and the last k raw stages):
          m = median of the last k raw stages (k odd; fewer at start -> lower median)
          s = s_prev + clip(m - s_prev, -1, +1)   (first output: s = m)
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits, up2
from . import up2_metrics as M
from .config import CHECKPOINT_DIR, REPORTS_DIR, SEED
from .synth_v3 import generator as gen

RUN_DIR = CHECKPOINT_DIR / "up2_v3b"
TRAIN3 = ("synth_v3_train3", 1600, 505, 30000)     # name, runs, seed offset, run-id offset
VAL2 = ("synth_v3_val2", 120, 606, 40000)
TEST2 = ("synth_v3_test2", 100, 707, 50000)


# --------------------------------------------------------------------------- data

def _gen(spec) -> pd.DataFrame:
    """Generated + labelled table (labelled frame cached next to the generator cache; derived, deterministic)."""
    name, n, off, ro = spec
    lp = up2.cache_path(name).with_suffix(".labelled.parquet")
    pp = up2.cache_path(name).with_suffix(".provenance.json")
    if lp.exists() and pp.exists():
        d = pd.read_parquet(lp)
        d.attrs["provenance"] = json.loads(pp.read_text())
        return d
    raw = up2.generate(name, n_runs=n, seed=SEED + off, run_offset=ro)
    prov = raw.attrs.get("provenance", {})
    # labels/engineering are per unit, so labelling in chunks of runs is exact (and avoids O(units x rows) indexing)
    keys = np.array(sorted(raw["bearing_key"].unique()))
    parts = [up2.label(raw[raw["bearing_key"].isin(set(ch))].reset_index(drop=True)) for ch in np.array_split(keys, max(1, len(keys) // 100))]
    d = pd.concat(parts, ignore_index=True)
    d["split"] = "robust"
    d.attrs["provenance"] = prov
    d.to_parquet(lp, index=False)
    return d


def load_val2() -> pd.DataFrame:
    d = _gen(VAL2)
    splits.register_holdout(d, VAL2[0], "val", note="UP2 v3b extra validation runs (same generator, new seed)")
    d["split"] = splits.assign(d).to_numpy()
    return d


def load_test2() -> pd.DataFrame:
    d = _gen(TEST2)
    splits.register_holdout(d, TEST2[0], "test",
                            note="UP2 v3b FRESH test runs (same generator, new seed); registered before any v3b fit; scored once")
    d["split"] = splits.assign(d).to_numpy()
    return d


def load_dev() -> pd.DataFrame:
    """Train + validation rows only (never a test row): synth_v3 train/val, train2, train3 (train), val2 (val)."""
    from .up2_train import load_main
    m = load_main()                                    # synth_v3 (frozen splits) + synth_v3_train2 (train)
    prov = m.attrs.get("provenance", {})
    m = m[m["split"] != "test"]
    t3 = _gen(TRAIN3)
    t3["split"] = "train"
    v2 = load_val2()
    out = pd.concat([m, t3, v2], ignore_index=True)
    assert not out["split"].eq("test").any()
    out.attrs["provenance"] = {**prov, "train3": t3.attrs.get("provenance", {}), "val2": v2.attrs.get("provenance", {})}
    return out


def load_old_test() -> pd.DataFrame:
    from .up2_train import load_main
    m = up2.prepare(gen.DATASET)
    return m[m["split"] == "test"].copy()


# --------------------------------------------------------------------------- post-processing (reference impl.)

def pp_rul_unit(r_raw: np.ndarray, alpha: float, scale: float) -> np.ndarray:
    """Causal log-EMA with a one-window decrement prior (see module docstring). r_raw in windows, one unit, time order."""
    out = np.empty(len(r_raw))
    prev = None
    for i, r in enumerate(np.maximum(np.asarray(r_raw, float), 0.0)):
        if prev is None or alpha >= 1.0:
            cur = r
        else:
            prior = max(prev - 1.0, 0.0)
            cur = float(np.expm1(alpha * np.log1p(r) + (1.0 - alpha) * np.log1p(prior)))
        prev = cur
        out[i] = scale * cur
    return out


def pp_stage_unit(raw: np.ndarray, k: int, max_step: int | None) -> np.ndarray:
    """Causal median of the last k raw stages, then a +/-max_step rate limit (None = no limit)."""
    raw = np.asarray(raw, int)
    out = np.empty(len(raw), int)
    prev = None
    for i in range(len(raw)):
        w = np.sort(raw[max(0, i - k + 1): i + 1])
        m = int(w[(len(w) - 1) // 2])                   # lower median while the buffer is short / even
        if prev is None or max_step is None:
            cur = m
        else:
            cur = prev + int(np.clip(m - prev, -max_step, max_step))
        prev = cur
        out[i] = cur
    return out


def apply_pp(rp: pd.DataFrame, pp: dict) -> pd.DataFrame:
    """rp: per-window predictions (unit_id, window_index, rul_pred_windows_raw, stage_pred_raw) -> deployed columns
    stage_pred, rul_unscaled (filtered, windows), rul_pred_windows (= rul_unscaled * scale_early if stage_pred <= 2
    else * scale_late), rul_pred_h. Order per window: stage filter, RUL filter, stage-dependent scale."""
    rp = rp.sort_values(["unit_id", "window_index"]).copy()
    rw, st = [], []
    for _, g in rp.groupby("unit_id", sort=False):
        rw.append(pp_rul_unit(g["rul_pred_windows_raw"].to_numpy(), pp["rul_alpha"], 1.0))
        st.append(pp_stage_unit(g["stage_pred_raw"].to_numpy(), pp["stage_k"], pp["stage_max_step"]))
    rp["rul_unscaled"] = np.concatenate(rw) if rw else []
    rp["stage_pred"] = np.concatenate(st) if st else []
    f = np.where(rp["stage_pred"].to_numpy() <= 2, pp.get("rul_scale_early", 1.0), pp.get("rul_scale_late", 1.0))
    rp["rul_pred_windows"] = rp["rul_unscaled"] * f
    rp["rul_pred_h"] = rp["rul_pred_windows"] * gen.CADENCE_H
    return rp


# --------------------------------------------------------------------------- features (cached per dataset)

def _feat_cache(df: pd.DataFrame, NL: int, warm: int, levels: bool, tag: str):
    p = RUN_DIR / "feat" / f"{tag}_NL{NL}_w{warm}_L{int(levels)}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["X"], pd.DataFrame({"unit_id": z["unit_id"], "window_index": z["window_index"]})
    X, idx = up2.unit_features(df, NL, with_decoder=False, warm=warm, levels=levels)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, unit_id=idx["unit_id"].to_numpy(), window_index=idx["window_index"].to_numpy())
    return X, idx


def rul_feats(df: pd.DataFrame, NL: int, warm: int, levels: bool = False, cache: bool = True):
    """Features for every unit of df, cached per dataset (keyed by dataset name + table size)."""
    Xs, idxs = [], []
    for ds, g in df.groupby("dataset", sort=True):
        tag = f"{ds}_{len(g)}_{g['unit_id'].nunique()}"
        X, idx = _feat_cache(g, NL, warm, levels, tag) if cache else up2.unit_features(g, NL, False, warm=warm, levels=levels)
        Xs.append(X)
        idxs.append(idx)
    X = np.concatenate(Xs)
    idx = pd.concat(idxs, ignore_index=True)
    keep = idx.merge(df[["unit_id", "window_index"]].assign(_k=1), on=["unit_id", "window_index"], how="left")["_k"].notna().to_numpy()
    return X[keep], idx[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- RUL / z heads

RUL_TARGETS = {
    # name: (target transform, lightgbm objective, weight rule)
    "log_l2": ("log1p", "regression", "unit"),         # v3 head (geometric-mean estimate)
    "log_l1": ("log1p", "regression_l1", "unit"),      # median in log space
    "raw_mape": ("identity", "regression_l1", "unit_over_y"),   # weighted L1 with 1/y weights == MAPE objective
}


def _rul_xy(X, idx, df):
    t = up2.merge_truth(idx, df, ["run_to_failure", "u_true", "rul_windows", "z_target"])
    d = (t["run_to_failure"] & (t["u_true"] >= 0)).to_numpy()
    return X[d], t.loc[d].reset_index(drop=True)


def fit_rul(Xtr, ttr, Xva, tva, target: str, seed: int, threads: int, params: dict | None = None):
    import lightgbm as lgb
    tf, obj, wr = RUL_TARGETS[target]
    f = (np.log1p if tf == "log1p" else (lambda v: v))
    def w(t):
        u = up2.unit_balanced_weights(t["unit_id"])
        if wr == "unit_over_y":
            u = u / np.maximum(t["rul_windows"].to_numpy(float), 1.0)
            u = u / u.mean()
        return u
    p = {"n_estimators": 3000, "num_leaves": 48, "learning_rate": 0.04, "min_child_samples": 60, **(params or {})}
    m = lgb.LGBMRegressor(objective=obj, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                          random_state=seed, n_jobs=threads, verbose=-1, **p)
    t0 = time.perf_counter()
    m.fit(Xtr, f(ttr["rul_windows"].to_numpy(float)), sample_weight=w(ttr),
          eval_set=[(Xva, f(tva["rul_windows"].to_numpy(float)))], eval_sample_weight=[w(tva)],
          callbacks=[lgb.early_stopping(80, verbose=False)])
    return m, {"target": target, "seed": seed, "best_iteration": int(m.best_iteration_ or p["n_estimators"]),
               "fit_seconds": round(time.perf_counter() - t0, 1), **p}


def rul_windows_from(models: list, target: str, X) -> np.ndarray:
    tf = RUL_TARGETS[target][0]
    pr = np.mean([m.predict(X) for m in models], axis=0)     # average in the TRANSFORMED space (as the ONNX graph)
    return np.maximum(np.expm1(pr) if tf == "log1p" else pr, 0.0)


def fit_z(Xtr, ttr_all, Xva, tva_all, threads: int, params: dict | None = None, seed: int = SEED, healthy_keep: float = 0.3):
    """z regressor on every window of run-to-failure bearings + a seeded subsample of healthy/partner windows."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    keep = ttr_all["run_to_failure"].to_numpy(bool) | (rng.random(len(ttr_all)) < healthy_keep)
    p = {"n_estimators": 3000, "num_leaves": 31, "learning_rate": 0.06, "min_child_samples": 60, **(params or {})}
    m = lgb.LGBMRegressor(objective="regression", subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                          random_state=seed, n_jobs=threads, verbose=-1, **p)
    t0 = time.perf_counter()
    m.fit(Xtr[keep], ttr_all.loc[keep, "z_target"].to_numpy(float),
          eval_set=[(Xva, tva_all["z_target"].to_numpy(float))], callbacks=[lgb.early_stopping(80, verbose=False)])
    return m, {"best_iteration": int(m.best_iteration_ or p["n_estimators"]), "fit_seconds": round(time.perf_counter() - t0, 1),
               "healthy_keep": healthy_keep, **p}


# --------------------------------------------------------------------------- selection (validation runs only)

WARM = 25                     # classifier slot T (v3 cls, reused unchanged): first scored window index = 24
NL = 48
RUL_CANDIDATES = ["log_l2", "log_l1", "raw_mape"]
N_SEEDS = [1, 3]
Z_CUTS = [0.0, -0.03, -0.06, -0.1]
ALPHAS = [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05]
SCALES = [0.85, 0.88, 0.91, 0.94, 0.97, 1.0, 1.03]
STAGE_K = [1, 3, 5, 7, 9]
STAGE_STEP = [None, 1]
TRUTH = ["fault_class", "run_to_failure", "observable_class", "health_stage", "rul_hours", "life_hours",
         "u_true", "in_deg", "in_deg_full", "in_deg_det", "rpm_mode", "split", "bearing_key"]


def _rul_score(rp: pd.DataFrame, truth: pd.DataFrame) -> dict:
    r = up2.merge_truth(rp, truth, TRUTH)
    return M.rul(r[r["run_to_failure"]])


def _stage_score(rp: pd.DataFrame, truth: pd.DataFrame) -> dict:
    r = up2.merge_truth(rp, truth, TRUTH)
    return M.stage(r[r["run_to_failure"]])


def _scale_rows(rp: pd.DataFrame, s_early: float, s_late: float) -> pd.DataFrame:
    """Stage-dependent output scale: s_early while the post-processed stage <= 2, s_late from stage 3 on."""
    out = rp.copy()
    f = np.where(out["stage_pred"].to_numpy() <= 2, s_early, s_late)
    out["rul_pred_windows"] = out["rul_unscaled"] * f
    out["rul_pred_h"] = out["rul_pred_windows"] * gen.CADENCE_H
    return out


def cmd_select(a) -> int:
    from .up2_train import meta, now_tag
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_dev()
    tr, va = dev[dev["split"] == "train"], dev[dev["split"] == "val"]
    log = {"counts": up2.split_check(dev), "rul": [], "z": {}, "pp": {}}
    print("dev", json.dumps(log["counts"]["runs_per_split"]), flush=True)
    Xtr, itr = rul_feats(tr, NL, WARM)
    Xva, iva = rul_feats(va, NL, WARM)
    ttr_all = up2.merge_truth(itr, tr, ["run_to_failure", "u_true", "rul_windows", "z_target"])
    tva_all = up2.merge_truth(iva, va, ["run_to_failure", "u_true", "rul_windows", "z_target"])
    va_rtf = tva_all["run_to_failure"].to_numpy(bool)
    print("features", Xtr.shape, Xva.shape, flush=True)

    # ---- z head + stage post-process
    zm, zinfo = fit_z(Xtr, ttr_all, Xva, tva_all, a.threads)
    log["z"]["fit"] = zinfo
    print("z", json.dumps(zinfo), flush=True)
    zva = zm.predict(Xva)
    base = iva.copy()
    base["z_pred"] = zva
    grid = []
    for cut in Z_CUTS:
        b = base.copy()
        b["stage_pred_raw"] = up2.stage_from_z(zva, cut)
        b["rul_pred_windows_raw"] = 1.0
        b = b[va_rtf]
        for k in STAGE_K:
            for step in STAGE_STEP:
                pp = {"rul_alpha": 1.0, "stage_k": k, "stage_max_step": step}
                s = _stage_score(apply_pp(b, pp), va)
                grid.append({"z_cut": cut, "stage_k": k, "stage_max_step": step, "max": s["max_abs_error"],
                             "n_ge2": s.get("windows_error_ge2", 0), "within1": s["within1_frac"], "mae": s["mae"]})
    grid.sort(key=lambda r: (r["max"], r["n_ge2"], -r["within1"], r["mae"], r["stage_k"]))
    log["pp"]["stage_grid"] = grid
    best_st = grid[0]
    print("stage pp best", json.dumps(best_st), "| raw (k=1, no limit, cut 0):",
          json.dumps([g for g in grid if g["stage_k"] == 1 and g["stage_max_step"] is None and g["z_cut"] == 0.0]), flush=True)

    # ---- RUL head candidates
    Xd, td = _rul_xy(Xtr, itr, tr)
    Xdv, tdv = _rul_xy(Xva, iva, va)
    st_raw = up2.stage_from_z(zva, best_st["z_cut"])
    cands = {}
    for target in RUL_CANDIDATES:
        ms = []
        for si in range(max(N_SEEDS)):
            if si >= 1 and a.ensemble_target not in ("all", target):
                break
            m, info = fit_rul(Xd, td, Xdv, tdv, target, SEED + si, a.threads)
            print("rul fit", json.dumps(info), flush=True)
            ms.append((m, info))
            if si + 1 in N_SEEDS:
                cands[(target, si + 1)] = list(ms)
    rows = []
    for (target, ns), ms in cands.items():
        b = iva.copy()
        b["rul_pred_windows_raw"] = rul_windows_from([m for m, _ in ms], target, Xva)
        b["stage_pred_raw"] = st_raw
        b = b[va_rtf]
        for al in ALPHAS:
            pp = {"rul_alpha": al, "stage_k": best_st["stage_k"], "stage_max_step": best_st["stage_max_step"]}
            p = apply_pp(b, pp)
            for se in SCALES:
                for sl in SCALES:
                    s = _rul_score(_scale_rows(p, se, sl), va)
                    rows.append({"target": target, "n_seeds": ns, "alpha": al, "scale_early": se, "scale_late": sl,
                                 "val_S7": s["S7_mape_pct"], "val_true_onset": s["alt_mape_true_onset_pct"],
                                 "val_pol": s["alt_percent_of_life_error_pts"]})
        raw = [r for r in rows if r["target"] == target and r["n_seeds"] == ns and r["alpha"] == 1.0
               and r["scale_early"] == 1.0 and r["scale_late"] == 1.0][0]
        bst = min((r for r in rows if r["target"] == target and r["n_seeds"] == ns), key=lambda r: r["val_S7"])
        print("rul cand", target, ns, "raw S7 %.2f" % raw["val_S7"], "best pp", json.dumps(bst), flush=True)
    log["rul"] = rows
    # selection: min val S7; ties (within 0.05 pts) -> fewer seeds (smaller ONNX), then alpha closer to 1, scales closer to 1
    best = min(rows, key=lambda r: (round(r["val_S7"] / 0.05), r["n_seeds"], -r["alpha"],
                                     abs(1 - r["scale_early"]) + abs(1 - r["scale_late"])))
    sel = {"cls": {"reused": "checkpoints/up2_v3/cls.pkl (v3 classifier, unchanged)", "T": WARM},
           "rul": {"target": best["target"], "n_seeds": best["n_seeds"], "NL": NL, "levels": False,
                   "transform": RUL_TARGETS[best["target"]][0], "objective": RUL_TARGETS[best["target"]][1],
                   "fits": [i for _, i in cands[(best["target"], best["n_seeds"])]]},
           "z": {"NL": NL, "z_cut": best_st["z_cut"], "fit": zinfo},
           "postprocess": {"rul_alpha": best["alpha"], "rul_scale_early": best["scale_early"], "rul_scale_late": best["scale_late"],
                           "stage_k": best_st["stage_k"], "stage_max_step": best_st["stage_max_step"]},
           "val": {"S7": best["val_S7"], "true_onset": best["val_true_onset"], "pol": best["val_pol"],
                   "stage": {k: best_st[k] for k in ("max", "n_ge2", "within1", "mae")}}}
    with open(RUN_DIR / "heads.pkl", "wb") as f:
        pickle.dump({"rul": [m for m, _ in cands[(best["target"], best["n_seeds"])]], "z": zm, "sel": sel}, f)
    with open(RUN_DIR / "rul_candidates.pkl", "wb") as f:
        pickle.dump({k: [m for m, _ in v] for k, v in cands.items()}, f)
    (RUN_DIR / "selection.json").write_text(json.dumps(sel, indent=1))
    out = {"meta": meta({"selection": "validation runs only (synth_v3 val + synth_v3_val2)"}), "grid": log, "selected": sel}
    path = REPORTS_DIR / f"model_selection_up2_v3b_{now_tag()}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("selected", json.dumps(sel["postprocess"]), json.dumps(sel["val"]), sel["rul"]["target"], sel["rul"]["n_seeds"],
          "->", path, flush=True)
    return 0


# --------------------------------------------------------------------------- prediction (deployed v3b engine logic)

PRED_DIR = REPORTS_DIR / "up2_preds"
FAMILY = "up2_v3b_gbm"


def load_heads():
    with open(RUN_DIR / "heads.pkl", "rb") as f:
        h = pickle.load(f)
    with open(up2.RUN_DIR / "cls.pkl", "rb") as f:
        h["cls"] = pickle.load(f)                 # v3 classifier, unchanged
    return h


def predict(df: pd.DataFrame, h: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .up2_train import cls_features, cls_predict
    h = h or load_heads()
    sel = h["sel"]
    Xc, ic, _ = cls_features(df, h["cls"]["cfg"]["T"])
    cp = cls_predict(h["cls"]["model"], Xc, ic)
    Xr, ir = rul_feats(df, sel["rul"]["NL"], WARM, cache=False)
    rp = ir.copy()
    rp["rul_pred_windows_raw"] = rul_windows_from(h["rul"], sel["rul"]["target"], Xr)
    rp["z_pred"] = h["z"].predict(Xr)
    rp["stage_pred_raw"] = stage_from_z(rp["z_pred"].to_numpy(), sel["z"]["z_cut"], sel["z"].get("z_shift", 0.0))
    rp = apply_pp(rp, sel["postprocess"])
    rp["rul_pred_h_raw"] = rp["rul_pred_windows_raw"] * gen.CADENCE_H
    return cp, rp


def score(cp, rp, df) -> dict:
    from .up2_eval import score_set
    return score_set(cp, rp, df, WARM - 1)


def cmd_predict(a) -> int:
    """The single scoring pass: old test (2nd look) + FRESH test2 (1st and only look) + robustness sets."""
    from .up2_train import ROBUST_SETS, load_robust, meta, now_tag
    prev = sorted(PRED_DIR.glob(f"{FAMILY}_*.index.json"))
    if prev and not a.force:
        raise SystemExit(f"test2 was already scored ({prev[-1].name}); re-scoring is `python -m aiengine.up2_eval`")
    tag = now_tag()
    sets = {"test": load_old_test(), "test2": load_test2(), **{n: load_robust(n) for n in ROBUST_SETS}}
    h = load_heads()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, d in sets.items():
        cp, rp = predict(d, h)
        f1, f2 = PRED_DIR / f"{FAMILY}_{name}_{tag}.cls.csv.gz", PRED_DIR / f"{FAMILY}_{name}_{tag}.rul.csv.gz"
        cp.to_csv(f1, index=False, float_format="%.6g")
        rp.to_csv(f2, index=False, float_format="%.6g")
        files[name] = {"cls": f1.name, "rul": f2.name}
        s = score(cp, rp, d)
        print(name, "macroF1 %.4f  S7 %.2f  stage max %s within1 %.4f  caught %s/%s" % (
            s["classification"]["macro_f1"], s["rul"]["S7_mape_pct"], s["stage"]["max_abs_error"],
            s["stage"]["within1_frac"], s["caught"]["caught_by_stage3"], s["caught"]["rtf_bearings"]), flush=True)
    out = {"meta": meta({"tag": tag, "touch": "old test: 2nd look (scored once for v3); test2: FIRST and only scoring pass",
                         "selection": json.loads((RUN_DIR / "selection.json").read_text())}), "files": files}
    (PRED_DIR / f"{FAMILY}_{tag}.index.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


# --------------------------------------------------------------------------- export (ONNX v3b + contract + golden)

CLS_FILE, RUL_FILE = "v3b_cls.onnx", "v3b_rul.onnx"
CONTRACT_FILE, GOLDEN_FILE = "model_contract_v3b.json", "golden_vectors_v3b.json"


def build_rul_onnx(rul_models: list, transform: str, z_model, NL_: int, R: int, z_cut: float, z_shift: float = 0.0):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from . import up2_export as ux
    from . import up2_window as uw
    agg = uw.aggregator_graph(NL_, R, with_decoder=False, out_name="feat")
    F = uw.n_features(R, False)
    nodes, inits, opsets, outs = list(agg.graph.node), list(agg.graph.initializer), list(agg.opset_import), []
    for i, m in enumerate(rul_models):
        n, ini, o, ops = ux._inline(ux._convert(m, F), f"r{i}_", "feat")
        nodes += n; inits += ini; opsets += ops; outs.append(list(o.values())[0])
    zn, zi, zo, zops = ux._inline(ux._convert(z_model, F), "z_", "feat")
    nodes += zn; inits += zi; opsets += zops
    zv = list(zo.values())[0]
    c = lambda v, n: numpy_helper.from_array(np.array(v, np.float32), n)   # noqa: E731
    nodes.append(helper.make_node("Sum", outs, ["p_sum"]))
    nodes.append(helper.make_node("Mul", ["p_sum", "p_inv_n"], ["p_mean"]))       # mean in the transformed space
    if transform == "log1p":
        nodes += [helper.make_node("Exp", ["p_mean"], ["p_e"]), helper.make_node("Sub", ["p_e", "p_one"], ["p_t"])]
    else:
        nodes.append(helper.make_node("Identity", ["p_mean"], ["p_t"]))
    nodes += [
        helper.make_node("Max", ["p_t", "p_zero"], ["p_rul"]),
        helper.make_node("Add", [zv, "p_shift"], ["p_zs"]),
        helper.make_node("Clip", ["p_zs", "p_zero", "p_one"], ["p_zc"]),
        helper.make_node("Mul", ["p_zc", "p_five"], ["p_z5"]),
        helper.make_node("Floor", ["p_z5"], ["p_fl"]),
        helper.make_node("Add", ["p_fl", "p_two"], ["p_st0"]),
        helper.make_node("Min", ["p_st0", "p_six"], ["p_st1"]),
        helper.make_node("Less", [zv, "p_cut"], ["p_pre"]),
        helper.make_node("Where", ["p_pre", "p_one", "p_st1"], ["p_stage"]),
        helper.make_node("Concat", ["p_rul", zv, "p_stage"], ["rul_head"], axis=1),
    ]
    inits += [c(1.0, "p_one"), c(0.0, "p_zero"), c(5.0, "p_five"), c(2.0, "p_two"), c(6.0, "p_six"),
              c(float(z_cut), "p_cut"), c(float(z_shift), "p_shift"), c(1.0 / len(rul_models), "p_inv_n")]
    outputs = [helper.make_tensor_value_info("rul_head", TensorProto.FLOAT, ["batch", 3])]
    return ux._model(nodes, inits, list(agg.graph.input), outputs, opsets, "v3b_rul")


def postprocess_contract(pp: dict) -> dict:
    return {
        "type": "v3b_stage_median_rul_logema",      # dispatch key for C# RulPostProcessors.Create
        "version": 1,
        "applies_to": "contract v3b direct_v3 heads; per bearing, causal, one update per 'ok' window (after warm-up)",
        "required": True,
        "state_reset": "on engine reset / new bearing stream (same as the window buffers)",
        "order": ["stage", "rul_filter", "rul_scale"],
        "stage": {"type": "median_then_rate_limit", "median_window": int(pp["stage_k"]),
                  "max_step": None if pp["stage_max_step"] is None else int(pp["stage_max_step"]),
                  "input": "s_raw = round(rul_head[2]) (rul_head[2] is an exact integer-valued float)",
                  "procedure": "push s_raw into a FIFO of the last median_window raw stages; m = sorted(fifo)[(len(fifo)-1)//2] "
                               "(lower median while the FIFO is not full); if first output: s = m; else if max_step is null: s = m; "
                               "else s = s_prev + clamp(m - s_prev, -max_step, +max_step); s_prev = s; health_stage = s"},
        "rul_filter": {"type": "log_ema_with_decrement", "alpha": float(pp["rul_alpha"]), "decrement_windows": 1.0,
                       "input": "r_raw = max(rul_head[0], 0) (windows)",
                       "procedure": "if first output or alpha >= 1: r = r_raw; else prior = max(r_prev - 1, 0); "
                                    "r = exp(alpha*ln(1+r_raw) + (1-alpha)*ln(1+prior)) - 1; r_prev = r (UNscaled); "
                                    "compute in float64"},
        "rul_scale": {"type": "stage_dependent", "scale_if_stage_le_2": float(pp["rul_scale_early"]),
                      "scale_if_stage_ge_3": float(pp["rul_scale_late"]),
                      "procedure": "rul_windows = r * (scale_if_stage_le_2 if health_stage <= 2 else scale_if_stage_ge_3) "
                                   "using the post-processed health_stage of the same window; rul_hours = rul_windows * cadence_hours"},
        "outputs": {"rul_hours": "post-processed (deployed)", "health_stage": "post-processed (deployed)",
                    "rul_hours_raw": "rul_head[0] * cadence (diagnostic)", "health_stage_raw": "rul_head[2] (diagnostic)"},
        "reference_impl": ["aiengine/up2b.py::pp_rul_unit, pp_stage_unit, apply_pp (offline)",
                           "aiengine/engine.py::HybridEngine._postprocess (streaming)"],
        "selected_on": "validation runs only (synth_v3 val + synth_v3_val2)",
    }


def cmd_export(a) -> int:
    import onnx
    from . import up2_export as ux
    from .config import MODELS_DIR
    from .train import git_commit
    h = load_heads()
    sel = h["sel"]
    R = len(up2.COLUMNS)
    T, NL_ = h["cls"]["cfg"]["T"], sel["rul"]["NL"]
    m_cls = ux.build_cls(h["cls"]["model"], T, R, ux.importances_per_column(h["cls"]["model"], R))
    m_rul = build_rul_onnx(h["rul"], sel["rul"]["transform"], h["z"], NL_, R, sel["z"]["z_cut"], sel["z"].get("z_shift", 0.0))
    onnx.save(m_cls, str(MODELS_DIR / CLS_FILE))
    onnx.save(m_rul, str(MODELS_DIR / RUL_FILE))
    test2 = load_test2()
    par = parity(MODELS_DIR / CLS_FILE, MODELS_DIR / RUL_FILE, h, test2)
    res = {"cls": {"size_kb": round((MODELS_DIR / CLS_FILE).stat().st_size / 1024, 1), "onnx_inputs": ["encoder_cont", "decoder_cont"],
                   "outputs": ["logits", "variable_weights"], "batch_dim": "dynamic", "parity": par["cls"]},
           "rul": {"size_kb": round((MODELS_DIR / RUL_FILE).stat().st_size / 1024, 1), "onnx_inputs": ["encoder_cont"],
                   "outputs": ["rul_head"], "batch_dim": "dynamic", "parity": par["rul"]}}
    heads = ({"model": h["cls"]["model"], "cfg": h["cls"]["cfg"]}, {"cfg": {"kind": "lgbm", "NL": NL_}},
             {"cfg": {"kind": "lgbm", "z_cut": sel["z"]["z_cut"]}})
    prov = {"synthetic": True, "datasets": ["synth_v3 (train/val)", "synth_v3_train2 (train)", "synth_v3_train3 (train)",
                                            "synth_v3_val2 (val)"], "run": "up2_v3b_gbm", "git_commit": git_commit(),
            "seed": SEED, "selection": sel, "note": "[SYNTHETIC] v3 classifier (unchanged) + v3b RUL/z heads; selected on validation runs only"}
    con = ux.contract(T, NL_, heads, res, prov)
    con["contract_version"] = "3b"
    con["generated_by"] = "AI-engine/aiengine/up2b.py export"
    con["models"]["tft"]["onnx"] = CLS_FILE
    con["models"]["nhits"]["onnx"] = RUL_FILE
    con["models"]["nhits"]["family"] = (f"gradient-boosted trees in the N-HiTS slot: RUL = mean of {len(h['rul'])} LightGBM "
                                        f"({sel['rul']['target']}, transform {sel['rul']['transform']}) + z LightGBM")
    con["rul"]["procedure"] = ("r_raw = rul_head[0] (windows), s_raw = rul_head[2]; apply `postprocess` (required) -> "
                               "health_stage, rul_windows; rul_hours = rul_windows * cadence_hours (median spacing of the last "
                               "<=48 window times). z = rul_head[1]. RUL is meaningful when health_stage >= 2.")
    con["stage"] = (f"raw (computed inside v3b_rul.onnx): rul_head[2] = 1 if z < {sel['z']['z_cut']} else "
                    f"min(6, 2 + floor(5 clip(z + {sel['z'].get('z_shift', 0.0)}, 0, 1))); deployed: postprocess.stage")
    con["postprocess"] = postprocess_contract(sel["postprocess"])
    con["disclosures"] = {
        "class_order": "classes.labels is ALPHABETICAL (ball, cage, healthy, inner_race, outer_race) = the output-channel "
                       "order of logits; it is NOT config.FAULT_CLASSES' order and NOT the order used in report tables "
                       "(healthy, outer_race, inner_race, ball, cage). Always map channels through classes.labels.",
        "provenance": "[SYNTHETIC] every model in this contract was fitted on synth_v3 generator runs only "
                      "(physics-informed generated data), never on team-rig data; public-data (XJTU-SY/IMS/MaFaulDa) "
                      "results belong to contract v2 and are reported separately as [MEASURED on <dataset>].",
        "test_sets": "synth_v3 test (50 runs): scored once for v3, again for v3b (2nd look). synth_v3_test2 (100 fresh "
                     "runs, splits.json): registered before any v3b fit, scored once.",
        "classifier": "v3b reuses the v3 classifier unchanged (v3b_cls.onnx is a re-export of the same trees).",
    }
    (MODELS_DIR / CONTRACT_FILE).write_text(json.dumps(con, indent=2, default=float))
    from .engine import HybridEngine
    eng = HybridEngine(MODELS_DIR, contract_file=CONTRACT_FILE)
    rtf = test2[test2["run_to_failure"]]
    unit_id = rtf.groupby("unit_id").size().idxmin()
    unit = test2[test2["unit_id"] == unit_id].sort_values("window_index")
    golden = eng.golden_vectors(unit, n=12)
    golden["cases_late"] = eng.golden_cases_at(unit, [int(x) for x in np.linspace(len(unit) * 0.3, len(unit) - 1, 16)])
    from .config import BEARING_FEATURES
    golden["inputs_full"] = [{"window_index": int(r.window_index), "rpm": float(r.rpm),
                              "t20_ms": int(round(float(r.t_hours) * 3.6e6)) % (2 ** 32),
                              "features16": [float(getattr(r, c)) for c in BEARING_FEATURES]}
                             for r in unit.itertuples(index=False)]
    golden["provenance"] = "[SYNTHETIC] synth_v3_test2 run"
    golden["postprocess_note"] = ("outputs rul_hours/health_stage are POST-PROCESSED (contract postprocess); "
                                  "rul_hours_raw/health_stage_raw are the bare ONNX heads. Feed every window from window 0.")
    # streaming engine vs offline reference post-process on the same unit
    golden["engine_vs_offline"] = engine_vs_offline(eng, unit, h)
    (MODELS_DIR / GOLDEN_FILE).write_text(json.dumps(golden, indent=1, default=float))
    print(json.dumps({"contract": CONTRACT_FILE, "parity": par, "golden_unit": unit_id, "sizes_kb":
                      {CLS_FILE: res["cls"]["size_kb"], RUL_FILE: res["rul"]["size_kb"]},
                      "engine_vs_offline": golden["engine_vs_offline"]}, indent=1, default=str))
    return 0


def parity(cls_onnx: Path, rul_onnx: Path, h: dict, df: pd.DataFrame) -> dict:
    import onnxruntime as ort
    from . import up2_window as uw
    units = df[df["run_to_failure"]]["unit_id"].drop_duplicates().iloc[:4].tolist() + \
        df[~df["run_to_failure"]]["unit_id"].drop_duplicates().iloc[:2].tolist()
    d = df[df["unit_id"].isin(units)]
    cp, rp = predict(d, h)
    sc = ort.InferenceSession(str(cls_onnx), providers=["CPUExecutionProvider"])
    sr = ort.InferenceSession(str(rul_onnx), providers=["CPUExecutionProvider"])
    T, NL_ = h["cls"]["cfg"]["T"], h["sel"]["rul"]["NL"]
    pc, ro = [], []
    for uid, g in d.groupby("unit_id", sort=True):
        rows = g.sort_values("window_index")[up2.COLUMNS].to_numpy(np.float64)
        bc, _ = uw.blocks(rows, T)
        br, _ = uw.blocks(rows, NL_)
        pc.append(np.exp(sc.run(["logits"], {"encoder_cont": bc[T - 1:, :-1], "decoder_cont": bc[T - 1:, -1:]})[0][:, 0]))
        ro.append(sr.run(["rul_head"], {"encoder_cont": br[T - 1:]})[0])
    pc, ro = np.concatenate(pc), np.concatenate(ro)
    cps, rps = cp.sort_values(["unit_id", "window_index"]), rp.sort_values(["unit_id", "window_index"])
    P = cps[[f"p_{k}" for k in up2.CLASS_ORDER]].to_numpy()
    ref = rps["rul_pred_windows_raw"].to_numpy()
    return {"cls": {"windows": int(len(pc)), "prob_max_abs_diff": float(np.abs(pc - P).max()),
                    "argmax_mismatch": int((pc.argmax(1) != P.argmax(1)).sum())},
            "rul": {"windows": int(len(ro)), "rul_windows_max_rel_diff": float((np.abs(ro[:, 0] - ref) / np.maximum(ref, 1.0)).max()),
                    "z_max_abs_diff": float(np.abs(ro[:, 1] - rps["z_pred"].to_numpy()).max()),
                    "stage_mismatch": int((ro[:, 2].astype(int) != rps["stage_pred_raw"].to_numpy()).sum())}}


def engine_vs_offline(eng, unit: pd.DataFrame, h: dict) -> dict:
    outs = eng.run_unit(unit)
    ok = [(int(w), o) for w, o in zip(unit.sort_values("window_index")["window_index"], outs) if o.get("status") == "ok"]
    _, rp = predict(unit, h)
    rp = rp.set_index("window_index").loc[[w for w, _ in ok]]
    e_r = np.array([o["rul_hours"] for _, o in ok])
    e_s = np.array([o["health_stage"] for _, o in ok])
    return {"windows": len(ok), "rul_hours_max_rel_diff": float((np.abs(e_r - rp["rul_pred_h"].to_numpy()) /
                                                                 np.maximum(rp["rul_pred_h"].to_numpy(), 0.1)).max()),
            "stage_mismatch": int((e_s != rp["stage_pred"].to_numpy()).sum()),
            "note": "engine (streaming, ONNX, contract cadence) vs offline (native trees, generator cadence)"}


def stage_from_z(z: np.ndarray, z_cut: float, z_shift: float = 0.0) -> np.ndarray:
    """1 if z < z_cut else min(6, 2 + floor(5 clip(z + z_shift, 0, 1))). z_shift >= 0 biases the stage up (the
    spec tolerates +/-1, an under-call by 2 stages early in the degradation interval is the failure mode)."""
    z = np.asarray(z, float)
    st = np.minimum(2 + np.floor(np.clip(z + z_shift, 0, 1) * 5), 6)
    return np.where(z < z_cut, 1, st).astype(int)


Z_CUTS_B = [0.0, -0.05, -0.1, -0.15, -0.25]
Z_SHIFTS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


def stage_grid(iva: pd.DataFrame, zva: np.ndarray, va: pd.DataFrame, rtf: np.ndarray) -> list[dict]:
    base = iva.copy()
    base["rul_pred_windows_raw"] = 1.0
    grid = []
    for cut in Z_CUTS_B:
        for sh in Z_SHIFTS:
            b = base.copy()
            b["stage_pred_raw"] = stage_from_z(zva, cut, sh)
            b = b[rtf]
            for k in STAGE_K:
                for step in STAGE_STEP:
                    s = _stage_score(apply_pp(b, {"rul_alpha": 1.0, "stage_k": k, "stage_max_step": step}), va)
                    grid.append({"z_cut": cut, "z_shift": sh, "stage_k": k, "stage_max_step": step, "max": s["max_abs_error"],
                                 "n_ge2": s.get("windows_error_ge2", 0), "bearings_ge2": s.get("bearings_with_error_ge2", 0),
                                 "within1": s["within1_frac"], "mae": s["mae"]})
    grid.sort(key=lambda r: (r["max"], r["n_ge2"], -r["within1"], r["mae"], r["z_shift"], r["stage_k"]))
    return grid


def cmd_select_stage(a) -> int:
    """Stage head re-selection on VALIDATION only: optional smaller z refit (ONNX size), extended (z_cut, z_shift)
    mapping grid x post-process grid. Rewrites heads.pkl / selection.json (stage part only)."""
    from .up2_train import meta, now_tag
    dev = load_dev()
    tr, va = dev[dev["split"] == "train"], dev[dev["split"] == "val"]
    Xva, iva = rul_feats(va, NL, WARM)
    tva_all = up2.merge_truth(iva, va, ["run_to_failure", "u_true", "rul_windows", "z_target", "health_stage"])
    rtf = tva_all["run_to_failure"].to_numpy(bool)
    with open(RUN_DIR / "heads.pkl", "rb") as f:
        h = pickle.load(f)
    zs = {"current": (h["z"], h["sel"]["z"]["fit"])}
    if a.refit_z:
        Xtr, itr = rul_feats(tr, NL, WARM)
        ttr_all = up2.merge_truth(itr, tr, ["run_to_failure", "u_true", "rul_windows", "z_target"])
        zm, info = fit_z(Xtr, ttr_all, Xva, tva_all, a.threads, params={"learning_rate": a.z_lr, "num_leaves": a.z_leaves})
        print("z refit", json.dumps(info), flush=True)
        zs["refit"] = (zm, info)
        del Xtr
    res = {}
    for name, (zm, info) in zs.items():
        zva = zm.predict(Xva)
        g = stage_grid(iva, zva, va, rtf)
        res[name] = {"fit": info, "best": g[0], "grid": g[:40], "leaves": int(info["best_iteration"]) * int(info.get("num_leaves", 31))}
        print(name, "best", json.dumps(g[0]), "leaves", res[name]["leaves"], flush=True)
        b = g[0]
        d = iva.copy()
        d["stage_pred_raw"] = stage_from_z(zva, b["z_cut"], b["z_shift"])
        d["z_pred"] = zva
        d["rul_pred_windows_raw"] = 1.0
        d = apply_pp(d[rtf], {"rul_alpha": 1.0, "stage_k": b["stage_k"], "stage_max_step": b["stage_max_step"]})
        d = up2.merge_truth(d, va, ["health_stage", "u_true"])
        bad = d[np.abs(d["stage_pred"] - d["health_stage"]) >= 2]
        print(" errors>=2:", bad[["unit_id", "window_index", "u_true", "health_stage", "stage_pred_raw", "stage_pred", "z_pred"]]
              .head(25).to_string(), flush=True)
    key = lambda n: (res[n]["best"]["max"], res[n]["best"]["n_ge2"], -res[n]["best"]["within1"], res[n]["leaves"])  # noqa: E731
    pick = min(res, key=key)
    b = res[pick]["best"]
    h["z"] = zs[pick][0]
    h["sel"]["z"] = {"NL": NL, "z_cut": b["z_cut"], "z_shift": b["z_shift"], "fit": zs[pick][1], "chosen_from": list(res)}
    h["sel"]["postprocess"].update({"stage_k": b["stage_k"], "stage_max_step": b["stage_max_step"]})
    h["sel"]["val"]["stage"] = {k: b[k] for k in ("max", "n_ge2", "within1", "mae")}
    with open(RUN_DIR / "heads.pkl", "wb") as f:
        pickle.dump(h, f)
    (RUN_DIR / "selection.json").write_text(json.dumps(h["sel"], indent=1, default=float))
    out = {"meta": meta({"selection": "stage head, validation runs only"}), "results": res, "picked": pick}
    path = REPORTS_DIR / f"model_selection_up2_v3b_stage_{now_tag()}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("picked", pick, json.dumps(b), "->", path, flush=True)
    print("NOTE: RUL post-process scales were selected with the previous stage filter; re-run `select-rulpp` if it changed.")
    return 0


def cmd_select_rulpp(a) -> int:
    """Re-select the RUL candidate + RUL post-process (alpha, stage-dependent scales) on VALIDATION, given the current
    stage head/filter in heads.pkl (the scale depends on the post-processed stage)."""
    from .up2_train import meta, now_tag
    dev = load_dev()
    va = dev[dev["split"] == "val"]
    Xva, iva = rul_feats(va, NL, WARM)
    rtf = up2.merge_truth(iva, va, ["run_to_failure"])["run_to_failure"].to_numpy(bool)
    with open(RUN_DIR / "heads.pkl", "rb") as f:
        h = pickle.load(f)
    with open(RUN_DIR / "rul_candidates.pkl", "rb") as f:
        cands = pickle.load(f)
    sel = h["sel"]
    st_raw = stage_from_z(h["z"].predict(Xva), sel["z"]["z_cut"], sel["z"].get("z_shift", 0.0))
    rows = []
    for (target, ns), ms in cands.items():
        b = iva.copy()
        b["rul_pred_windows_raw"] = rul_windows_from(ms, target, Xva)
        b["stage_pred_raw"] = st_raw
        b = b[rtf]
        for al in ALPHAS:
            p = apply_pp(b, {"rul_alpha": al, "stage_k": sel["postprocess"]["stage_k"],
                             "stage_max_step": sel["postprocess"]["stage_max_step"]})
            for se in SCALES:
                for sl in SCALES:
                    s = _rul_score(_scale_rows(p, se, sl), va)
                    rows.append({"target": target, "n_seeds": ns, "alpha": al, "scale_early": se, "scale_late": sl,
                                 "val_S7": s["S7_mape_pct"], "val_true_onset": s["alt_mape_true_onset_pct"],
                                 "val_pol": s["alt_percent_of_life_error_pts"]})
        raw = [r for r in rows if r["target"] == target and r["n_seeds"] == ns and r["alpha"] == 1.0
               and r["scale_early"] == 1.0 and r["scale_late"] == 1.0][0]
        bst = min((r for r in rows if r["target"] == target and r["n_seeds"] == ns), key=lambda r: r["val_S7"])
        print("rul cand", target, ns, "raw S7 %.2f" % raw["val_S7"], "best pp", json.dumps(bst), flush=True)
    best = min(rows, key=lambda r: (round(r["val_S7"] / 0.05), r["n_seeds"], -r["alpha"],
                                     abs(1 - r["scale_early"]) + abs(1 - r["scale_late"])))
    h["rul"] = cands[(best["target"], best["n_seeds"])]
    sel["rul"].update({"target": best["target"], "n_seeds": best["n_seeds"], "transform": RUL_TARGETS[best["target"]][0],
                       "objective": RUL_TARGETS[best["target"]][1]})
    sel["postprocess"].update({"rul_alpha": best["alpha"], "rul_scale_early": best["scale_early"], "rul_scale_late": best["scale_late"]})
    sel["val"].update({"S7": best["val_S7"], "true_onset": best["val_true_onset"], "pol": best["val_pol"]})
    with open(RUN_DIR / "heads.pkl", "wb") as f:
        pickle.dump(h, f)
    (RUN_DIR / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
    path = REPORTS_DIR / f"model_selection_up2_v3b_rulpp_{now_tag()}.json"
    path.write_text(json.dumps({"meta": meta({"selection": "RUL + post-process, validation runs only"}), "grid": rows,
                                "selected": sel}, indent=1, default=float))
    print("selected", json.dumps(sel["postprocess"]), json.dumps(sel["val"]), best["target"], best["n_seeds"], "->", path, flush=True)
    return 0


def cmd_data(a) -> int:
    dev = load_dev()
    t2 = load_test2()
    rep = {"dev": up2.split_check(dev), "test2": {"runs": int(t2["bearing_key"].nunique()), "windows": int(len(t2)),
                                                  **t2.attrs.get("provenance", {})}}
    print(json.dumps(rep, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2b")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    s = sub.add_parser("select")
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--ensemble-target", default="all", help="target that gets extra seeds: one name or 'all'")
    ss = sub.add_parser("select-stage")
    ss.add_argument("--threads", type=int, default=16)
    ss.add_argument("--refit-z", action="store_true")
    ss.add_argument("--z-lr", type=float, default=0.15)
    ss.add_argument("--z-leaves", type=int, default=31)
    sub.add_parser("select-rulpp")
    pr = sub.add_parser("predict")
    pr.add_argument("--force", action="store_true", help="re-run the scoring pass (must be disclosed)")
    sub.add_parser("export")
    a = ap.parse_args(argv)
    return {"data": cmd_data, "select": cmd_select, "select-stage": cmd_select_stage, "select-rulpp": cmd_select_rulpp, "predict": cmd_predict, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
