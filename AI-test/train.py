"""Training for both ICS models, plus the RUL calibration fitted alongside."""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import data as data_mod
import features as feat_mod
import rul as rul_mod
import synth
from config import ARTIFACT_DIR, DATA_DIR, RunConfig
from models import build_nhits, build_tft, count_parameters

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_FILE = DATA_DIR / "synthetic_runs.parquet"
ENGINEERED_FILE = DATA_DIR / "engineered.parquet"
SPLITS_FILE = DATA_DIR / "splits.json"
CALIB_FILE = ARTIFACT_DIR / "rul_calibration.json"
NHITS_CKPT = ARTIFACT_DIR / "nhits.ckpt"
TFT_CKPT = ARTIFACT_DIR / "tft.ckpt"
META_FILE = ARTIFACT_DIR / "training_meta.json"

# Windows: dataloader worker processes re-import the module and cost more in
# spawn overhead than they save on a dataset this size.
NUM_WORKERS = 0

# Set ICS_AI_PROGRESS=1 to get Lightning's per-step progress bar back.
SHOW_PROGRESS = os.environ.get("ICS_AI_PROGRESS", "") == "1"


def _seed(cfg: RunConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))


def generate(cfg: RunConfig, force: bool = False) -> pd.DataFrame:
    """Create (or reuse) the synthetic corpus and its engineered features."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ENGINEERED_FILE.exists() and not force:
        print(f"  reusing {ENGINEERED_FILE.name} (pass --force to regenerate)")
        return pd.read_parquet(ENGINEERED_FILE)

    print(f"  generating {cfg.synth.n_runs} synthetic run-to-failure histories...")
    raw = synth.generate_dataset(cfg.synth)
    engineered, _ = feat_mod.engineer_dataset(raw)
    engineered = data_mod.apply_labelling_policy(engineered)

    raw.to_parquet(DATASET_FILE, index=False)
    engineered.to_parquet(ENGINEERED_FILE, index=False)

    splits = data_mod.split_runs(engineered, cfg)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))

    n_windows = len(engineered)
    hours = n_windows * cfg.synth.hours_per_window
    print(f"  {n_windows:,} windows across {cfg.synth.n_runs} runs "
          f"({hours:,.0f} h of simulated machine life)")
    print(f"  split by run -> train {len(splits['train'])}, "
          f"val {len(splits['val'])}, test {len(splits['test'])}")
    return engineered


def load_splits() -> dict[str, list[int]]:
    return json.loads(SPLITS_FILE.read_text())


def _trainer(cfg: RunConfig, max_epochs: int, name: str, monitor: str) -> pl.Trainer:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_model_summary=False,
        # Off by default: the bar emits a line per step, which buries the
        # actual results when this is run non-interactively or piped to a log.
        enable_progress_bar=SHOW_PROGRESS,
        log_every_n_steps=10,
        logger=CSVLogger(save_dir=str(ARTIFACT_DIR / "logs"), name=name),
        callbacks=[
            EarlyStopping(monitor=monitor, patience=6, mode="min", min_delta=1e-4),
            ModelCheckpoint(
                dirpath=str(ARTIFACT_DIR / "logs" / name),
                filename="best",
                monitor=monitor,
                mode="min",
                save_top_k=1,
            ),
        ],
        deterministic=False,
    )


def train_nhits(engineered: pd.DataFrame, splits: dict, cfg: RunConfig) -> dict:
    """Train the health-index forecaster (ICS1's RUL path)."""
    print("\n[N-HiTS] health-index forecaster -> ICS1 (RUL)")
    frames = {
        name: data_mod.subset(engineered, ids) for name, ids in splits.items()
    }
    training, others = data_mod.build_nhits_datasets(
        frames["train"], {"val": frames["val"]}, cfg.nhits
    )
    train_loader = training.to_dataloader(
        train=True, batch_size=cfg.nhits.batch_size, num_workers=NUM_WORKERS
    )
    val_loader = others["val"].to_dataloader(
        train=False, batch_size=cfg.nhits.batch_size * 2, num_workers=NUM_WORKERS
    )

    model = build_nhits(training, cfg.nhits)
    print(f"  {count_parameters(model):,} trainable parameters, "
          f"encoder {cfg.nhits.encoder_length} -> horizon {cfg.nhits.prediction_length}")

    trainer = _trainer(cfg, cfg.nhits.max_epochs, "nhits", "val_loss")
    started = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    elapsed = time.perf_counter() - started

    best = trainer.checkpoint_callback.best_model_path
    torch.save(torch.load(best, weights_only=False), NHITS_CKPT)
    val_loss = float(trainer.callback_metrics.get("val_loss", float("nan")))
    print(f"  trained in {elapsed:.1f}s, best val MAE {val_loss:.4f} "
          f"health-index units")
    return {
        "parameters": count_parameters(model),
        "train_seconds": elapsed,
        "val_mae_health_index": val_loss,
        "epochs_run": trainer.current_epoch + 1,
    }


def train_tft(engineered: pd.DataFrame, splits: dict, cfg: RunConfig) -> dict:
    """Train the per-window fault classifier (ICS2)."""
    print("\n[TFT] per-window fault classifier -> ICS2")
    frames = {
        name: data_mod.subset(engineered, ids) for name, ids in splits.items()
    }
    training, others = data_mod.build_tft_datasets(
        frames["train"], {"val": frames["val"]}, cfg.tft
    )
    train_loader = training.to_dataloader(
        train=True, batch_size=cfg.tft.batch_size, num_workers=NUM_WORKERS
    )
    val_loader = others["val"].to_dataloader(
        train=False, batch_size=cfg.tft.batch_size * 2, num_workers=NUM_WORKERS
    )

    order = data_mod.class_order(training)
    weights = data_mod.class_weights(frames["train"], order)
    print("  output channel order: " + ", ".join(
        f"{i}={name}" for i, name in enumerate(order)))
    print("  class weights: " + ", ".join(
        f"{name} {w:.2f}" for name, w in zip(order, weights)))

    model = build_tft(training, cfg.tft, weights)
    print(f"  {count_parameters(model):,} trainable parameters, "
          f"encoder {cfg.tft.encoder_length} -> 1 window")

    trainer = _trainer(cfg, cfg.tft.max_epochs, "tft", "val_loss")
    started = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    elapsed = time.perf_counter() - started

    best = trainer.checkpoint_callback.best_model_path
    torch.save(torch.load(best, weights_only=False), TFT_CKPT)
    val_loss = float(trainer.callback_metrics.get("val_loss", float("nan")))
    print(f"  trained in {elapsed:.1f}s, best val weighted cross-entropy {val_loss:.4f}")
    return {
        "parameters": count_parameters(model),
        "train_seconds": elapsed,
        "val_cross_entropy": val_loss,
        "epochs_run": trainer.current_epoch + 1,
        "class_order": order,
        "class_weights": weights.tolist(),
    }


def calibrate_rul(engineered: pd.DataFrame, splits: dict, cfg: RunConfig) -> dict:
    """Fit failure thresholds and the population prior on training runs only."""
    print("\n[calibration] RUL thresholds and population prior")
    train_frame = data_mod.subset(engineered, splits["train"])
    calib = rul_mod.calibrate(train_frame, cfg.synth.hours_per_window)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(calib.to_dict(), indent=2))
    print("  class-conditional failure thresholds (health index):")
    for name, value in sorted(calib.class_thresholds.items()):
        print(f"    {name:<12} {value:.3f}")
    return calib.to_dict()


def run(cfg: RunConfig, force: bool = False, models: str = "both") -> dict:
    _seed(cfg)
    print("=" * 74)
    print("ICS AI testbench -- training")
    print("=" * 74)

    engineered = generate(cfg, force=force)
    if not SPLITS_FILE.exists():
        SPLITS_FILE.write_text(json.dumps(data_mod.split_runs(engineered, cfg), indent=2))
    splits = load_splits()

    meta: dict = {
        "seed": cfg.seed,
        "data_is_synthetic": True,
        "n_runs": cfg.synth.n_runs,
        "n_windows": int(len(engineered)),
        "hours_per_window": cfg.synth.hours_per_window,
        "splits": {k: len(v) for k, v in splits.items()},
    }

    if models in ("both", "nhits"):
        meta["nhits"] = train_nhits(engineered, splits, cfg)
    if models in ("both", "tft"):
        meta["tft"] = train_tft(engineered, splits, cfg)
    meta["rul_calibration"] = calibrate_rul(engineered, splits, cfg)

    META_FILE.write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nArtifacts written to {ARTIFACT_DIR}")
    return meta
