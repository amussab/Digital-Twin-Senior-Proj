"""Data preparation, pretraining, fine-tuning and RUL calibration.

Run layout: checkpoints/<run>/{tft.ckpt, nhits.ckpt, rul_calibration.json, run_meta.json}.
Model selection uses training/validation units only; test units are read only by evaluate.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import lightning.pytorch as pl  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from pytorch_forecasting import NHiTS, TemporalFusionTransformer, TimeSeriesDataSet  # noqa: E402

from . import features as feat  # noqa: E402
from . import infer, labels, models, rul, splits  # noqa: E402
from .config import CACHE_DIR, CHECKPOINT_DIR, ROOT, SEED  # noqa: E402
from .datasets import common  # noqa: E402

REAL_DATASETS = ("xjtu_sy", "ims", "mafaulda")
TORCH_THREADS = int(os.environ.get("AIENGINE_THREADS", "12"))


def set_threads(n: int = TORCH_THREADS) -> None:
    torch.set_num_threads(n)


def seed_all(seed: int = SEED) -> None:
    pl.seed_everything(seed, workers=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=ROOT,
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:  # noqa: BLE001
        return True


# --------------------------------------------------------------------------- data

def load_cache(name: str, synthetic_runs: int = 48) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{name}.parquet"
    if name == "synthetic_rig" and not path.exists():
        from .datasets import synthetic_rig
        common.save(synthetic_rig.build(n_runs=synthetic_runs), name)
    if not path.exists():
        return None
    return common.validate(common.load(name))


def available(names=REAL_DATASETS) -> list[str]:
    return [n for n in names if (CACHE_DIR / f"{n}.parquet").exists()]


def prepare(names: list[str]) -> pd.DataFrame:
    """Window tables -> engineered inputs -> labels -> frozen splits (column `split`)."""
    frames = [f for f in (load_cache(n) for n in names) if f is not None]
    if not frames:
        raise FileNotFoundError(f"no caches for {names} in {CACHE_DIR}")
    table = pd.concat(frames, ignore_index=True)
    eng = feat.engineer_table(table)
    lab = labels.label_table(eng)
    sp = splits.ensure(lab)
    lab["split"] = splits.assign(lab, sp).to_numpy()
    return lab


def counts(df: pd.DataFrame) -> dict:
    out = {}
    for (ds, sp), g in df.groupby(["dataset", "split"]):
        out.setdefault(ds, {})[sp] = {"units": int(g["unit_id"].nunique()), "windows": int(len(g))}
    return out


# --------------------------------------------------------------------------- training

def _trainer(run_dir: Path, name: str, max_epochs: int, patience: int, has_val: bool) -> pl.Trainer:
    cbs = []
    if has_val:
        cbs = [EarlyStopping(monitor="val_loss", patience=patience, mode="min", min_delta=1e-4),
               ModelCheckpoint(dirpath=str(run_dir / f"_{name}"), filename="best",
                               monitor="val_loss", mode="min", save_top_k=1)]
    return pl.Trainer(max_epochs=max_epochs, accelerator="cpu", devices=1, gradient_clip_val=0.1,
                      enable_model_summary=False, enable_progress_bar=False,
                      logger=CSVLogger(save_dir=str(run_dir / "logs"), name=name),
                      callbacks=cbs, deterministic=False, log_every_n_steps=20)


def _fit(model, trainer, train_dl, val_dl, run_dir: Path, name: str):
    t0 = time.perf_counter()
    trainer.fit(model, train_dl, val_dl)
    secs = time.perf_counter() - t0
    best = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
    target = run_dir / f"{name}.ckpt"
    if best:
        target.write_bytes(Path(best).read_bytes())
    else:
        trainer.save_checkpoint(str(target))
    vl = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None
    return target, {"train_seconds": round(secs, 1), "epochs_run": int(trainer.current_epoch),
                    "best_val_loss": float(vl) if vl is not None else None}


def tft_frames(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["cls_usable"]].copy()


def train_tft(df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: models.TFTConfig,
              run_dir: Path) -> tuple[Path, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    tr, va = tft_frames(df_train), tft_frames(df_val)
    ds = models.tft_dataset(tr, cfg)
    order = models.class_order(ds)
    w = models.class_weights(tr, order, cfg.class_weight_power)
    model = models.build_tft(ds, cfg, w)
    vds = models.derive(ds, va, "tft", cfg) if len(va) else None
    tdl = ds.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    vdl = vds.to_dataloader(train=False, batch_size=cfg.batch_size * 4, num_workers=0) if vds else None
    trainer = _trainer(run_dir, "tft", cfg.max_epochs, cfg.patience, vdl is not None)
    path, meta = _fit(model, trainer, tdl, vdl, run_dir, "tft")
    meta.update({"class_order": order, "class_weights": w.tolist(), "params": models.count_parameters(model),
                 "train_samples": len(ds), "val_samples": len(vds) if vds else 0, "config": asdict(cfg)})
    return path, meta


def rtf_frames(df: pd.DataFrame) -> pd.DataFrame:
    """N-HiTS trains on run-to-failure units (+ any long non-failing series of the same data)."""
    return df[df["run_to_failure"] | (df["dataset"].isin(["ims"]) & ~df["run_to_failure"])].copy()


def train_nhits(df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: models.NHiTSConfig,
                run_dir: Path) -> tuple[Path, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    need = cfg.encoder_length + cfg.prediction_length
    tr = df_train[df_train.groupby("unit_id")["window_index"].transform("size") >= need]
    va = df_val[df_val.groupby("unit_id")["window_index"].transform("size") >= need]
    ds = models.nhits_dataset(tr, cfg)
    model = models.build_nhits(ds, cfg)
    vds = models.derive(ds, va, "nhits", cfg) if len(va) else None
    tdl = ds.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    vdl = vds.to_dataloader(train=False, batch_size=cfg.batch_size * 4, num_workers=0) if vds else None
    trainer = _trainer(run_dir, "nhits", cfg.max_epochs, cfg.patience, vdl is not None)
    path, meta = _fit(model, trainer, tdl, vdl, run_dir, "nhits")
    meta.update({"params": models.count_parameters(model), "train_samples": len(ds),
                 "val_samples": len(vds) if vds else 0, "config": cfg.to_dict()})
    return path, meta


def finetune(checkpoint: Path, new_frame: pd.DataFrame, kind: str, run_dir: Path,
             val_frame: pd.DataFrame | None = None, lr_factor: float = 0.2,
             freeze_encoder: bool = False, max_epochs: int = 15, patience: int = 4,
             batch_size: int = 128) -> tuple[Path, dict]:
    """Fine-tune a pretrained checkpoint on a new domain.

    The new data is encoded with the PRETRAINING dataset's parameters
    (TimeSeriesDataSet.from_parameters -> same scalers / label encoder / lengths), the learning
    rate is lowered by `lr_factor`, and optionally the early layers are frozen (TFT: variable
    selection + LSTM encoder; N-HiTS: all but the last block).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cls = TemporalFusionTransformer if kind == "tft" else NHiTS
    model = cls.load_from_checkpoint(str(checkpoint), map_location="cpu")
    params = model.dataset_parameters
    cols = (params["time_varying_known_reals"] or []) + (params["time_varying_unknown_reals"] or [])
    target = params["target"]

    def frame(df):
        d = df[["unit_id", "window_index", target] + [c for c in cols if c != target]].copy()
        d["unit_id"] = d["unit_id"].astype(str)
        if kind == "tft":
            d[target] = d[target].astype(str)
        return d

    if kind == "nhits":
        need = params["max_encoder_length"] + params["max_prediction_length"]
        new_frame = new_frame[new_frame.groupby("unit_id")["window_index"].transform("size") >= need]
        if val_frame is not None:
            val_frame = val_frame[val_frame.groupby("unit_id")["window_index"].transform("size") >= need]
    ds = TimeSeriesDataSet.from_parameters(params, frame(new_frame), predict=False)
    tdl = ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    vdl = None
    if val_frame is not None and len(val_frame):
        vds = TimeSeriesDataSet.from_parameters(params, frame(val_frame), predict=False, stop_randomization=True)
        vdl = vds.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=0)
    model.hparams.learning_rate = model.hparams.learning_rate * lr_factor
    frozen = []
    if freeze_encoder:
        prefixes = (("prescalers", "encoder_variable_selection", "decoder_variable_selection",
                     "static_", "lstm_encoder", "input_embeddings")
                    if kind == "tft" else tuple(f"model.blocks.{i}." for i in range(len(model.model.blocks) - 1)))
        for n, p in model.named_parameters():
            if n.startswith(prefixes):
                p.requires_grad_(False)
                frozen.append(n)
    trainer = _trainer(run_dir, f"{kind}_ft", max_epochs, patience, vdl is not None)
    path, meta = _fit(model, trainer, tdl, vdl, run_dir, f"{kind}_ft")
    meta.update({"pretrained_from": str(checkpoint), "lr": model.hparams.learning_rate,
                 "frozen_tensors": len(frozen), "finetune_samples": len(ds)})
    return path, meta


# --------------------------------------------------------------------------- prediction helpers

def load_model(path: Path):
    try:
        return TemporalFusionTransformer.load_from_checkpoint(str(path), map_location="cpu").eval()
    except Exception:  # noqa: BLE001
        return NHiTS.load_from_checkpoint(str(path), map_location="cpu").eval()


def tft_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Per-window class probabilities for every window with >= min_encoder_length history."""
    params = model.dataset_parameters
    cols = params["time_varying_known_reals"] or []
    d = df[df["cls_usable"]] if "cls_usable" in df else df
    if d.empty:
        return pd.DataFrame()
    frame = d[["unit_id", "window_index", "observable_class"] + cols].copy()
    frame["unit_id"] = frame["unit_id"].astype(str)
    frame["observable_class"] = frame["observable_class"].astype(str)
    # The target column is not a model input; a class the model has no output channel for (e.g.
    # `ball`, which has no training bearing) is mapped to a known placeholder so the dataset can be
    # built. Truth is always merged from the original table afterwards, never from this frame.
    known = set(params["target_normalizer"].classes_)
    frame.loc[~frame["observable_class"].isin(known), "observable_class"] = sorted(known)[0]
    # A model trained on post-onset windows only (hierarchical v2) stores a min_prediction_idx > 0;
    # prediction must still cover every window of the unit.
    params = {**params, "min_prediction_idx": int(frame["window_index"].min())}
    ds = TimeSeriesDataSet.from_parameters(params, frame, predict=False, stop_randomization=True)
    order = models.class_order(ds)
    out = model.predict(ds.to_dataloader(train=False, batch_size=2048, num_workers=0),
                        mode="raw", return_index=True, return_x=True)
    logits = out.output["prediction"].detach().cpu().numpy()[:, 0, :]
    probs = infer.softmax(logits) if not np.allclose(logits.sum(-1), 1.0, atol=1e-4) else logits
    res = out.index.copy()
    res["encoder_length"] = out.x["encoder_lengths"].numpy()
    res["pred_class"] = [order[i] for i in probs.argmax(1)]
    res["confidence"] = probs.max(1)
    for i, c in enumerate(order):
        res[f"p_{c}"] = probs[:, i]
    # TimeSeriesDataSet also emits shorter-encoder samples whose decoder is a series' LAST window;
    # keep exactly one prediction per window: the longest encoder.
    res = (res.sort_values(["unit_id", "window_index", "encoder_length"], ascending=[True, True, False])
           .drop_duplicates(["unit_id", "window_index"]).reset_index(drop=True))
    if "healthy" not in order:
        # Hierarchical v2 model (fault-type TFT, no healthy channel): compose with the causal onset
        # gate -> probabilities over fault channels + healthy (DESIGN.md section 6, v2).
        from . import hier
        res = hier.compose(res, hier.gate_table(d), order)
    return res


def is_hierarchical(model) -> bool:
    return "healthy" not in set(model.dataset_parameters["target_normalizer"].classes_)


def nhits_layout(model) -> tuple[list[str], dict, int, int]:
    params = model.dataset_parameters
    ds_cols = params["time_varying_unknown_reals"] or []
    # reals order in the dataset: static reals, known reals, unknown reals (no statics here)
    columns = list(params["time_varying_known_reals"] or []) + list(ds_cols)
    scalers = {}
    for c in columns:
        if c == params["target"]:
            scalers[c] = {"center": 0.0, "scale": 1.0, "kind": "target_identity"}
        else:
            sc = params["scalers"][c]
            scalers[c] = {"center": float(np.ravel(sc.mean_)[0]), "scale": float(np.ravel(sc.scale_)[0]),
                          "kind": "standard"}
    return columns, scalers, params["max_encoder_length"], params["max_prediction_length"]


def tft_layout(model) -> tuple[list[str], dict, int]:
    params = model.dataset_parameters
    columns = list(params["time_varying_known_reals"] or []) + list(params["time_varying_unknown_reals"] or [])
    scalers = {c: {"center": float(np.ravel(params["scalers"][c].mean_)[0]),
                   "scale": float(np.ravel(params["scalers"][c].scale_)[0]), "kind": "standard"}
               for c in columns}
    return columns, scalers, params["max_encoder_length"]


def nhits_wrapper(model):
    return infer.ExportWrapper(model, model.dataset_parameters["max_encoder_length"],
                               model.dataset_parameters["max_prediction_length"], [0.0, 1.0]).eval()


def tft_wrapper(model):
    return infer.ExportWrapper(model, model.dataset_parameters["max_encoder_length"], 1, [0.0, 1.0]).eval()


def nhits_forecasts(model, unit: pd.DataFrame) -> np.ndarray:
    """(n, H) HI forecast made at every window of one unit (causal, left-padded early)."""
    cols, scalers, L, H = nhits_layout(model)
    enc, dec = infer.nhits_tensors(unit[cols].to_numpy(np.float64), cols, scalers, L, H)
    out = infer.torch_run(nhits_wrapper(model), enc, dec)
    return out.reshape(len(unit), -1)


# --------------------------------------------------------------------------- RUL records

def smoothed_class(probs: pd.DataFrame, k: int = 5) -> pd.Series:
    """Causal class for threshold selection: argmax of the mean fault probability over the last
    k windows, if the mean healthy probability is below 0.5; else 'healthy'."""
    pcols = [c for c in probs.columns if c.startswith("p_")]
    roll = probs[pcols].rolling(k, min_periods=1).mean()
    faults = [c for c in pcols if c != "p_healthy"]
    best = roll[faults].idxmax(axis=1).str[2:]
    return best.where(roll.get("p_healthy", 0) < 0.5, "healthy")


def rul_records(nhits_model, df_units: pd.DataFrame, tft_preds: pd.DataFrame | None) -> pd.DataFrame:
    """One record per window of each run-to-failure unit: inputs the RUL estimator needs."""
    recs = []
    tp = None
    if tft_preds is not None and len(tft_preds):
        tp = tft_preds.set_index(["unit_id", "window_index"]).sort_index()
    for uid, unit in df_units[df_units["run_to_failure"]].groupby("unit_id"):
        unit = unit.sort_values("window_index")
        fc = nhits_forecasts(nhits_model, unit)
        hi = unit["hi"].to_numpy(float)
        t = unit["t_hours"].to_numpy(float)
        det = labels.OnlineOnset()
        cls = pd.Series("healthy", index=unit["window_index"].to_numpy())
        if tp is not None and uid in tp.index.get_level_values(0):
            pu = tp.loc[uid]
            sc = smoothed_class(pu)
            cls.loc[sc.index] = sc.to_numpy()
        for i, row in enumerate(unit.itertuples(index=False)):
            on = det.update(hi[i])
            confirmed = on is not None
            tau = (t[i] - t[on]) if confirmed else None
            tw = t[max(0, i - 47):i + 1]
            cad = max(float(np.median(np.diff(tw))) if len(tw) > 1 else 0.0, 1e-9)
            recs.append({
                "unit_id": uid, "dataset": row.dataset, "window_index": int(row.window_index),
                "fault_class": row.fault_class, "t_hours": t[i], "cadence_h": cad,
                "rul_true": float(row.rul_hours), "life_hours": float(row.life_hours),
                "onset_t_hours": float(row.onset_t_hours), "health_stage": int(row.health_stage),
                "in_deg": bool(row.in_degradation_window), "in_deg_full": bool(row.in_degradation_full),
                "tau_h": tau, "hi_onset": float(hi[on]) if confirmed else None,
                "hi_hist": hi[max(0, i - 47):i + 1], "forecast": fc[i],
                "pred_class": str(cls.get(int(row.window_index), "healthy")),
            })
    return pd.DataFrame(recs)


def apply_rul(calib: rul.RULCalibration, recs: pd.DataFrame) -> pd.DataFrame:
    out = recs.copy()
    est = [rul.estimate(calib, r.hi_hist, r.forecast, r.pred_class, r.tau_h, r.hi_onset, r.cadence_h)
           for r in recs.itertuples(index=False)]
    out["rul_pred"] = [e["rul_hours"] for e in est]
    out["rul_trend"] = [e["rul_trend_hours"] for e in est]
    out["rul_onset"] = [e["rul_onset_hours"] for e in est]
    out["ll_feats"] = [e["loglin_features"] for e in est]
    out["stage_pred"] = [rul.stage_estimate(r.tau_h, e["rul_hours"])
                         for r, e in zip(recs.itertuples(index=False), est)]
    return out


def s7_mape(scored: pd.DataFrame, col="rul_pred", window="in_deg") -> tuple[float, dict]:
    """PRE-REGISTERED S7 metric (DESIGN.md section 7): bearing-averaged MAPE over the
    degradation window [t_onset, T_fail - 0.1 (T_fail - t_onset)]."""
    per = {}
    for uid, g in scored[scored[window]].groupby("unit_id"):
        g = g[g["rul_true"] > 0]
        if len(g):
            per[uid] = float(np.mean(np.abs(g[col] - g["rul_true"]) / g["rul_true"]) * 100)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def fit_rul(recs_fit: pd.DataFrame, end_hi: list[dict], grid: bool = True) -> tuple[rul.RULCalibration, dict]:
    """Choose the RUL estimator on fitting records (training/validation bearings only)."""
    th, pooled = rul.fit_thresholds(end_hi)
    best, table = None, []
    methods = ["trend", "onset", "blend"] if grid else ["trend"]
    for m in methods:
        for hw in ([12, 24, 48] if grid else [24]):
            for w in ([0.25, 0.5, 0.75] if m == "blend" else [0.5]):
                for umin in ([0.02, 0.05, 0.1] if m != "trend" else [0.05]):
                    c = rul.RULCalibration(method=m, history_windows=hw, blend_w=w, u_min=umin,
                                           class_thresholds=th, pooled_threshold=pooled)
                    score, _ = s7_mape(apply_rul(c, recs_fit))
                    table.append({"method": m, "history": hw, "blend_w": w, "u_min": umin, "mape": score})
                    if np.isfinite(score) and (best is None or score < best[0]):
                        best = (score, c)
    calib = best[1] if best else rul.RULCalibration(class_thresholds=th, pooled_threshold=pooled)
    return calib, {"grid": table, "selected": calib.to_dict(), "fit_mape": best[0] if best else None}


def end_hi_records(df: pd.DataFrame) -> list[dict]:
    return [{"fault_class": str(g["fault_class"].iloc[0]),
             "hi_end": rul.end_of_life_hi(g.sort_values("window_index")["hi"].to_numpy())}
            for _, g in df[df["run_to_failure"]].groupby("unit_id")]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def command_line() -> str:
    return " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
