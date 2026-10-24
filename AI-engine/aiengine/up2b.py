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
          rul_windows_out = scale * r       (scale chosen on validation; r_prev keeps the UNscaled r)
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
    d = up2.prepare(name, freeze_splits=False, n_runs=n, seed=SEED + off, run_offset=ro)
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
    """rp: per-window predictions (unit_id, window_index, rul_pred_windows_raw, stage_pred_raw) -> deployed columns."""
    rp = rp.sort_values(["unit_id", "window_index"]).copy()
    rw, st = [], []
    for _, g in rp.groupby("unit_id", sort=False):
        rw.append(pp_rul_unit(g["rul_pred_windows_raw"].to_numpy(), pp["rul_alpha"], pp["rul_scale"]))
        st.append(pp_stage_unit(g["stage_pred_raw"].to_numpy(), pp["stage_k"], pp["stage_max_step"]))
    rp["rul_pred_windows"] = np.concatenate(rw) if rw else []
    rp["rul_pred_h"] = rp["rul_pred_windows"] * gen.CADENCE_H
    rp["stage_pred"] = np.concatenate(st) if st else []
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
    t = up2.merge_truth(idx, df, ["run_to_failure", "u_true", "rul_windows", "z_target", "unit_id"])
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
