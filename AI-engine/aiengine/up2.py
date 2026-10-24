"""UP2 (contract v3) data + gradient-boosted heads on the SYNTHETIC synth_v3 rig data.

Everything here is [SYNTHETIC]. Pipeline:
  synth_v3.generator.build -> features.engineer_table (same C#-ported engineering as v2; baseline =
  median of the first 24 windows = the commissioning capture) -> labels (generator truth, plus the
  DESIGN.md section 6 detected-onset labels kept as *_det columns) -> frozen run-level splits
  (splits.json section `synth_v3`, created once, never reshuffled).

Heads (all exported to ONNX, all fed by the tensors the C# host already builds):
  cls  : XGBoost / LightGBM multiclass over {ball, cage, healthy, inner_race, outer_race}
         (alphabetical = output-channel order), TFT slot: encoder_cont [1, L, R] + decoder_cont [1, 1, R]
  rul  : log1p(RUL in windows) regressor, trained on degradation windows (true onset .. failure)
  z    : degradation-position regressor (z = u_true in [0,1] after the true onset, -0.2 before / healthy);
         stage = 1 if z < 0 else min(6, 2 + floor(5 z)) (DESIGN.md section 6 rule on the estimate)
  rul and z share the N-HiTS slot: encoder_cont [1, NL, R] (left-padded), one ONNX with output
  `rul_head` [B, 3] = [rul_windows, z, stage]. Hours = rul_windows * cadence (median window spacing).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from . import labels, splits
from . import up2_window as uw
from .config import CHECKPOINT_DIR, SEED
from .config import UP2_CACHE_DIR as CACHE_DIR
from .synth_v3 import generator as gen

CLASS_ORDER = ["ball", "cage", "healthy", "inner_race", "outer_race"]   # alphabetical = channel order
COLUMNS = list(feat.ENGINEERED)                                         # every engineered input, identity scale
Z_HEALTHY = -0.2
RUN_DIR = CHECKPOINT_DIR / "up2_v3"
DEFAULT_RUNS = 240


# --------------------------------------------------------------------------- data

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def generate(name: str = gen.DATASET, n_runs: int = DEFAULT_RUNS, seed: int = SEED, knobs: gen.Knobs | None = None,
             run_offset: int = 0, force: bool = False) -> pd.DataFrame:
    p = cache_path(name)
    meta_p = p.with_suffix(".provenance.json")
    if p.exists() and not force:
        df = pd.read_parquet(p)
        df.attrs["provenance"] = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        return df
    df = gen.build(n_runs=n_runs, seed=seed, knobs=knobs, dataset=name, run_offset=run_offset)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prov = {**df.attrs["provenance"], "table_hash": gen.table_hash(df), "rows": int(len(df))}
    df.to_parquet(p, index=False)
    meta_p.write_text(json.dumps(prov, indent=1))
    df.attrs["provenance"] = prov
    return df


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered inputs + truth labels + DESIGN.md section 6 detected-onset labels (suffix _det)."""
    prov = df.attrs.get("provenance", {})
    eng = feat.engineer_table(df)
    det = labels.label_table(eng)
    out = det.rename(columns={"observable_class": "observable_class_det", "health_stage": "health_stage_det",
                              "in_degradation_window": "in_deg_det", "in_degradation_full": "in_deg_full_det"})
    u = out["u_true"].to_numpy(float)
    rtf = out["run_to_failure"].to_numpy(bool)
    out["observable_class"] = np.where(rtf & (u >= 0), out["fault_class"], "healthy")
    out["health_stage"] = out["true_stage"].astype(int)
    out["in_deg"] = rtf & (u >= 0) & (u <= 1 - labels.DEGRADATION_TAIL_EXCLUDED)
    out["in_deg_full"] = rtf & (u >= 0)
    out["rul_windows"] = out["rul_hours"] / gen.CADENCE_H
    out["z_target"] = np.where(rtf & (u >= 0), u, Z_HEALTHY)
    out.attrs["provenance"] = prov
    return out


def prepare(name: str = gen.DATASET, freeze_splits: bool = True, **kw) -> pd.DataFrame:
    df = label(generate(name, **kw))
    prov = df.attrs.get("provenance", {})
    if freeze_splits:
        sp = splits.ensure(df)
        df["split"] = splits.assign(df, sp).to_numpy()
    else:
        df["split"] = "robust"
    df.attrs["provenance"] = prov
    return df


def split_check(df: pd.DataFrame) -> dict:
    """Leakage check: every run (bearing_key) maps to exactly one split; no unit in two splits."""
    per_run = df.groupby("bearing_key")["split"].nunique()
    per_unit = df.groupby("unit_id")["split"].nunique()
    runs = df.groupby("split")["bearing_key"].nunique().to_dict()
    cls = (df[df["run_to_failure"]].groupby(["split", "fault_class"])["bearing_key"].nunique()
           .unstack(fill_value=0).to_dict(orient="index"))
    return {"runs_in_multiple_splits": int((per_run > 1).sum()), "units_in_multiple_splits": int((per_unit > 1).sum()),
            "runs_per_split": {k: int(v) for k, v in runs.items()}, "faulted_runs_per_split_class": cls,
            "windows_per_split": {k: int(v) for k, v in df["split"].value_counts().items()}}


# --------------------------------------------------------------------------- tabular features

def unit_features(df: pd.DataFrame, T: int, with_decoder: bool, columns: list[str] = COLUMNS,
                  warm: int | None = None, levels: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Aggregator features for every window with >= `warm` engineered rows of history.
    warm defaults to T (classifier slot: T-1 encoder rows + current). Returns X and the index frame."""
    warm = T if warm is None else warm
    Xs, idx = [], []
    for uid, g in df.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        b, _ = uw.blocks(g[columns].to_numpy(np.float64), T)
        keep = np.arange(len(g)) >= warm - 1
        if not keep.any():
            continue
        Xs.append(uw.compute(b[keep], with_decoder, levels=levels))     # per unit: bounded memory for long T
        idx.append(g.loc[keep, ["unit_id", "window_index"]])
    X = np.concatenate(Xs) if Xs else np.zeros((0, uw.n_features(len(columns), levels)), np.float32)
    return X, pd.concat(idx, ignore_index=True)


def merge_truth(index: pd.DataFrame, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return index.merge(df[["unit_id", "window_index"] + cols], on=["unit_id", "window_index"], how="left")


def class_weights(y: np.ndarray, power: float) -> np.ndarray:
    cnt = np.bincount(y, minlength=len(CLASS_ORDER)).astype(float) + 1
    w = (cnt.sum() / (len(cnt) * cnt)) ** power
    return w[y]


# --------------------------------------------------------------------------- fitting

def fit_classifier(kind: str, Xtr, ytr, Xva, yva, power: float = 0.5, seed: int = SEED, threads: int = 8,
                   params: dict | None = None):
    params = dict(params or {})
    wtr = class_weights(ytr, power)
    t0 = time.perf_counter()
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=params.pop("n_estimators", 600), max_depth=params.pop("max_depth", 6),
                              learning_rate=params.pop("learning_rate", 0.08), subsample=0.8, colsample_bytree=0.6,
                              min_child_weight=5, reg_lambda=1.0, tree_method="hist", objective="multi:softprob",
                              eval_metric="mlogloss", early_stopping_rounds=40, random_state=seed, n_jobs=threads,
                              **params)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[class_weights(yva, power)],
              verbose=False)
        best = int(m.best_iteration) + 1
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=params.pop("n_estimators", 800), num_leaves=params.pop("num_leaves", 48),
                               learning_rate=params.pop("learning_rate", 0.05), subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.6, min_child_samples=40, reg_lambda=1.0, random_state=seed,
                               n_jobs=threads, verbose=-1, **params)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], eval_sample_weight=[class_weights(yva, power)],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        best = int(m.best_iteration_ or m.n_estimators)
    return m, {"kind": kind, "best_iteration": best, "fit_seconds": round(time.perf_counter() - t0, 1),
               "class_weight_power": power}


def fit_regressor(kind: str, Xtr, ytr, Xva, yva, seed: int = SEED, threads: int = 8, wtr=None, wva=None,
                  params: dict | None = None):
    params = dict(params or {})
    t0 = time.perf_counter()
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=params.pop("n_estimators", 1500), max_depth=params.pop("max_depth", 6),
                             learning_rate=params.pop("learning_rate", 0.05), subsample=0.8, colsample_bytree=0.6,
                             min_child_weight=10, reg_lambda=1.0, tree_method="hist", objective="reg:squarederror",
                             early_stopping_rounds=60, random_state=seed, n_jobs=threads, **params)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva] if wva is not None else None,
              verbose=False)
        best = int(m.best_iteration) + 1
    else:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=params.pop("n_estimators", 2000), num_leaves=params.pop("num_leaves", 48),
                              learning_rate=params.pop("learning_rate", 0.03), subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, min_child_samples=40, reg_lambda=1.0, random_state=seed,
                              n_jobs=threads, verbose=-1, **params)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], eval_sample_weight=[wva] if wva is not None else None,
              callbacks=[lgb.early_stopping(60, verbose=False)])
        best = int(m.best_iteration_ or m.n_estimators)
    return m, {"kind": kind, "best_iteration": best, "fit_seconds": round(time.perf_counter() - t0, 1)}


def unit_balanced_weights(units: pd.Series) -> np.ndarray:
    """Each run-to-failure bearing weighs the same in total (the S7 metric is bearing-averaged)."""
    c = units.map(units.value_counts())
    w = 1.0 / c.to_numpy(float)
    return w / w.mean()


def stage_from_z(z: np.ndarray, z_cut: float = 0.0) -> np.ndarray:
    """1 if z < z_cut (before onset) else min(6, 2 + floor(5 clip(z, 0, 1))). z_cut <= 0 is chosen on validation."""
    z = np.asarray(z, float)
    st = np.minimum(2 + np.floor(np.clip(z, 0, 1) * 5), 6)
    return np.where(z < z_cut, 1, st).astype(int)
