"""Model construction for the two ICS-owned models.

N-HiTS  -> ICS1, continuous RUL path   (via the health-index forecast, rul.py)
TFT     -> ICS2, per-window fault classification

Both come from `pytorch-forecasting`, which is the library the ICS layer doc
names. Note for anyone writing this up: the library supplies the *architectures*
and trains them from initialisation. It does not ship pretrained weights for
either model, so "pretrained, fine-tuned" is not an accurate description of
what this pipeline does -- see the README's "Wording that needs fixing" note
before repeating that phrasing in the report.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_forecasting import NHiTS, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, CrossEntropy

from config import FAULT_CLASSES, NHiTSConfig, TFTConfig


class WeightedCrossEntropy(CrossEntropy):
    """Cross-entropy with per-class weights.

    pytorch-forecasting's CrossEntropy has no weight argument, and the label
    distribution here is heavily skewed toward "healthy" by the labelling
    policy in data.py. Unweighted, the classifier maximises accuracy by
    under-calling faults -- precisely the failure mode Ch.2.3 identifies as the
    ethical risk (a false "healthy" reading is worse than a false alarm).
    Overriding `loss` is the library's documented extension point.
    """

    def __init__(self, weights: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        self._weights = (
            None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        )

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = None if self._weights is None else self._weights.to(y_pred.device)
        flat = F.cross_entropy(
            y_pred.view(-1, y_pred.size(-1)),
            target.view(-1).long(),
            weight=weight,
            reduction="none",
        )
        # Same reshape the library's own CrossEntropy.loss uses.
        return flat.view(-1, target.size(-1))


def build_nhits(dataset: TimeSeriesDataSet, cfg: NHiTSConfig) -> NHiTS:
    """Health-index forecaster.

    MAE rather than the library's default MASE: the target is a bounded index
    on a fixed scale, so MASE's scaling by the in-sample naive error adds noise
    without adding meaning, and MAE keeps the loss directly readable as
    health-index units.
    """
    return NHiTS.from_dataset(
        dataset,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        n_blocks=list(cfg.n_blocks),
        dropout=cfg.dropout,
        loss=MAE(),
        backcast_loss_ratio=0.2,
        log_interval=-1,
        log_val_interval=-1,
        reduce_on_plateau_patience=3,
    )


def build_tft(
    dataset: TimeSeriesDataSet, cfg: TFTConfig, weights: np.ndarray | None = None
) -> TemporalFusionTransformer:
    """Per-window fault classifier.

    output_size is the number of classes and the loss is cross-entropy, which
    is pytorch-forecasting's supported classification configuration. The
    variable-selection network and interpretable attention are unaffected by
    the change of head -- and those are the whole reason TFT owns this path
    instead of N-HiTS: a fault call a technician can inspect, per the ICS layer
    doc's section 4.
    """
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        hidden_continuous_size=cfg.hidden_continuous_size,
        dropout=cfg.dropout,
        output_size=len(FAULT_CLASSES),
        loss=WeightedCrossEntropy(weights),
        log_interval=-1,
        log_val_interval=-1,
        reduce_on_plateau_patience=3,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
