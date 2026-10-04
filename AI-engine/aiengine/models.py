"""TimeSeriesDataSet + model builders for N-HiTS (HI forecast -> RUL) and TFT (fault class).

pytorch-forecasting supplies the architectures only (no pretrained weights). "Pretrained" in
this project = our own pretraining on public datasets, then fine-tuning (train.finetune).

Model inputs never include dataset identity (that would let the classifier shortcut by source);
speed enters only through rpm_norm. Group id = unit_id (one bearing in one run/record).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_forecasting import NHiTS, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.metrics import MAE, CrossEntropy

from . import features as feat


@dataclass
class TFTConfig:
    min_encoder_length: int = 2
    max_encoder_length: int = 6
    hidden_size: int = 32
    attention_head_size: int = 2
    hidden_continuous_size: int = 16
    dropout: float = 0.15
    learning_rate: float = 3e-3
    batch_size: int = 256
    max_epochs: int = 25
    patience: int = 5
    inputs: list = field(default_factory=lambda: list(feat.TFT_INPUTS))
    class_weight_power: float = 1.0       # 1 = inverse frequency, 0 = unweighted


@dataclass
class NHiTSConfig:
    encoder_length: int = 24
    prediction_length: int = 12
    hidden_size: int = 64
    n_blocks: tuple = (1, 1, 1)
    dropout: float = 0.1
    learning_rate: float = 2e-3
    batch_size: int = 256
    max_epochs: int = 25
    patience: int = 5
    covariates: list = field(default_factory=lambda: list(feat.NHITS_COVARIATES))

    def to_dict(self):
        d = asdict(self)
        d["n_blocks"] = list(self.n_blocks)
        return d


class WeightedCrossEntropy(CrossEntropy):
    """Cross-entropy with per-class weights (indexed in the encoder's output-channel order)."""

    def __init__(self, weights=None, **kwargs):
        super().__init__(**kwargs)
        self._w = None if weights is None else torch.as_tensor(np.asarray(weights), dtype=torch.float32)

    def loss(self, y_pred, target):
        w = None if self._w is None else self._w.to(y_pred.device)
        flat = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), target.view(-1).long(),
                               weight=w, reduction="none")
        return flat.view(-1, target.size(-1))


def _frame(df: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    keep = ["unit_id", "window_index", target] + [c for c in cols if c != target]
    out = df[keep].copy()
    out["unit_id"] = out["unit_id"].astype(str)
    for c in cols:
        if c != target:
            out[c] = out[c].astype(np.float32)
    return out


def tft_dataset(train_df: pd.DataFrame, cfg: TFTConfig) -> TimeSeriesDataSet:
    df = _frame(train_df, cfg.inputs, "observable_class")
    df["observable_class"] = df["observable_class"].astype(str)
    return TimeSeriesDataSet(
        df, time_idx="window_index", target="observable_class", group_ids=["unit_id"],
        min_encoder_length=cfg.min_encoder_length, max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=1, max_prediction_length=1,
        # "known" = available at the decoder step: the current window is measured before it is
        # classified, so its own features must be seen (unknown reals are encoder-only).
        time_varying_known_reals=list(cfg.inputs),
        target_normalizer=NaNLabelEncoder(add_nan=False),
        allow_missing_timesteps=False, add_relative_time_idx=False,
        add_target_scales=False, add_encoder_length=False,
    )


def nhits_dataset(train_df: pd.DataFrame, cfg: NHiTSConfig) -> TimeSeriesDataSet:
    df = _frame(train_df, [feat.NHITS_TARGET] + list(cfg.covariates), feat.NHITS_TARGET)
    df[feat.NHITS_TARGET] = df[feat.NHITS_TARGET].astype(np.float32)
    return TimeSeriesDataSet(
        df, time_idx="window_index", target=feat.NHITS_TARGET, group_ids=["unit_id"],
        min_encoder_length=cfg.encoder_length, max_encoder_length=cfg.encoder_length,
        min_prediction_length=cfg.prediction_length, max_prediction_length=cfg.prediction_length,
        time_varying_unknown_reals=[feat.NHITS_TARGET] + list(cfg.covariates),
        # Identity: HI is a bounded index on a fixed scale, and its absolute level is exactly
        # what rul.py extrapolates against a failure threshold.
        target_normalizer=TorchNormalizer(method="identity"),
        allow_missing_timesteps=False, add_relative_time_idx=False,
        add_target_scales=False, add_encoder_length=False,
    )


def derive(template: TimeSeriesDataSet, df: pd.DataFrame, kind: str, cfg) -> TimeSeriesDataSet:
    if kind == "tft":
        d = _frame(df, cfg.inputs, "observable_class")
        d["observable_class"] = d["observable_class"].astype(str)
    else:
        d = _frame(df, [feat.NHITS_TARGET] + list(cfg.covariates), feat.NHITS_TARGET)
    return TimeSeriesDataSet.from_dataset(template, d, predict=False, stop_randomization=True)


def class_order(ds: TimeSeriesDataSet) -> list[str]:
    """Output-channel order = the label encoder's (alphabetical). Never config order."""
    mapping = dict(ds.target_normalizer.classes_)
    return [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def class_weights(train_df: pd.DataFrame, order: list[str], power: float = 1.0) -> np.ndarray:
    counts = train_df["observable_class"].astype(str).value_counts().reindex(order).fillna(0).to_numpy(float) + 1
    w = (counts.sum() / (len(order) * counts)) ** power
    return (w / w.mean()).astype(np.float32)


def build_tft(ds: TimeSeriesDataSet, cfg: TFTConfig, weights) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        ds, learning_rate=cfg.learning_rate, hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size, hidden_continuous_size=cfg.hidden_continuous_size,
        dropout=cfg.dropout, output_size=len(class_order(ds)), loss=WeightedCrossEntropy(weights),
        log_interval=-1, log_val_interval=-1, reduce_on_plateau_patience=3,
    )


def build_nhits(ds: TimeSeriesDataSet, cfg: NHiTSConfig) -> NHiTS:
    return NHiTS.from_dataset(
        ds, learning_rate=cfg.learning_rate, hidden_size=cfg.hidden_size,
        n_blocks=list(cfg.n_blocks), dropout=cfg.dropout, loss=MAE(),
        backcast_loss_ratio=0.2, log_interval=-1, log_val_interval=-1,
        reduce_on_plateau_patience=3,
    )


def scaler_params(ds: TimeSeriesDataSet) -> dict[str, dict]:
    """center/scale of every real input, in encoder_cont column order (for C# / engine)."""
    out = {}
    target = ds.target if isinstance(ds.target, str) else None
    for name in ds.reals:
        if name == target:
            out[name] = {"center": 0.0, "scale": 1.0, "kind": "target_identity"}
            continue
        sc = ds.scalers.get(name)
        if sc is None:
            out[name] = {"center": 0.0, "scale": 1.0, "kind": "identity"}
        else:
            out[name] = {"center": float(np.ravel(sc.mean_)[0]), "scale": float(np.ravel(sc.scale_)[0]),
                         "kind": "standard"}
    return out


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
