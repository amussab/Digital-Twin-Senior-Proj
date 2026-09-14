"""Dataset assembly: labelling policy, run-level splits, TimeSeriesDataSets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer

from config import FAULT_CLASSES, NHiTSConfig, RunConfig, TFTConfig
from features import MODEL_INPUTS, TARGET_CLASS, TARGET_HI

# A fault is labelled as such only once it is physically present in the signal.
# Below this health stage the bearing's features are indistinguishable from a
# healthy one, so labelling those windows with the eventual fault type would be
# asking the classifier to predict the future from noise -- and would inflate
# every metric computed against those labels.
DETECTABLE_FROM_STAGE = 2


def apply_labelling_policy(data: pd.DataFrame) -> pd.DataFrame:
    """Add `observable_class`, the label the classifier is actually trained on.

    `fault_class` stays as the run's ground-truth eventual failure mode and is
    used for reporting which runs were caught in time. `observable_class` is
    what a technician could legitimately expect the system to say *at that
    window*: "healthy" until the defect is physically detectable, the true
    class from then on.
    """
    out = data.copy()
    detectable = out["health_stage"].to_numpy() >= DETECTABLE_FROM_STAGE
    truth = out["fault_class"].astype(str).to_numpy()
    out["observable_class"] = np.where(detectable, truth, "healthy")
    out["observable_class"] = pd.Categorical(
        out["observable_class"], categories=FAULT_CLASSES
    )
    out["fault_detectable"] = detectable
    return out


def split_runs(data: pd.DataFrame, cfg: RunConfig) -> dict[str, list[int]]:
    """Split by run, never by window.

    Windows from one bearing's history are strongly autocorrelated; a random
    window-level split would put near-duplicate rows on both sides of the
    boundary and report an accuracy the system will not reproduce in service.
    Splitting whole run-to-failure histories is the only honest option.
    Stratified by fault class so every split contains every class.
    """
    rng = np.random.default_rng(cfg.seed)
    per_run = (
        data.groupby("run_id")["fault_class"].first().astype(str).reset_index()
    )
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for fault_class, group in per_run.groupby("fault_class"):
        # .copy(): pandas 3 hands back read-only views, which rng.shuffle rejects.
        run_ids = group["run_id"].to_numpy().copy()
        rng.shuffle(run_ids)
        n = len(run_ids)
        n_test = max(1, int(round(n * cfg.test_fraction)))
        n_val = max(1, int(round(n * cfg.val_fraction)))
        # A class must always keep at least one training run. Give ground from
        # val first, then test: a class the model never saw in training would
        # poison every metric it appears in, whereas a class missing from val
        # only costs some early-stopping signal.
        while n_test + n_val >= n and n_val > 0:
            n_val -= 1
        while n_test + n_val >= n and n_test > 0:
            n_test -= 1
        if n_test == 0:
            print(f"  ! class '{fault_class}' has only {n} run(s) -- no test "
                  f"runs held out for it; its metrics will be missing.")
        splits["test"].extend(int(r) for r in run_ids[:n_test])
        splits["val"].extend(int(r) for r in run_ids[n_test : n_test + n_val])
        splits["train"].extend(int(r) for r in run_ids[n_test + n_val :])

    return {name: sorted(ids) for name, ids in splits.items()}


def subset(data: pd.DataFrame, run_ids: list[int]) -> pd.DataFrame:
    return data[data["run_id"].isin(run_ids)].reset_index(drop=True)


def _common_kwargs(cfg_model, target: str) -> dict:
    return dict(
        time_idx="window_index",
        target=target,
        group_ids=["run_id"],
        min_encoder_length=cfg_model.encoder_length,
        max_encoder_length=cfg_model.encoder_length,
        min_prediction_length=cfg_model.prediction_length,
        max_prediction_length=cfg_model.prediction_length,
        allow_missing_timesteps=False,
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
    )


def build_nhits_datasets(
    frame_train: pd.DataFrame, frame_other: dict[str, pd.DataFrame], cfg: NHiTSConfig
) -> tuple[TimeSeriesDataSet, dict[str, TimeSeriesDataSet]]:
    """N-HiTS forecasts the health index forward.

    It is NOT trained on RUL directly. A forecasting model is given the
    target's own history as encoder input, and RUL's history is a linear
    countdown -- a model handed that would learn to subtract rather than to
    read degradation, and would have nothing to read at inference time, since
    RUL is never observed in service. Forecasting the observable health index
    and solving for the threshold crossing (see rul.py) keeps the model doing
    work it can actually do on a live machine.
    """
    covariates = [c for c in MODEL_INPUTS if c != TARGET_HI]
    training = TimeSeriesDataSet(
        frame_train,
        **_common_kwargs(cfg, TARGET_HI),
        time_varying_unknown_reals=[TARGET_HI] + covariates,
        static_categoricals=["operating_mode"],
        # Identity, not the library's automatic per-encoder normaliser. The
        # health index is already a bounded index on a fixed scale, and its
        # absolute level is exactly the quantity rul.py extrapolates against a
        # failure threshold. Rescaling it per sample would make each forecast
        # internally consistent but mutually incomparable.
        target_normalizer=TorchNormalizer(method="identity"),
    )
    others = {
        name: TimeSeriesDataSet.from_dataset(
            training, frame, predict=False, stop_randomization=True
        )
        for name, frame in frame_other.items()
    }
    return training, others


def build_tft_datasets(
    frame_train: pd.DataFrame, frame_other: dict[str, pd.DataFrame], cfg: TFTConfig
) -> tuple[TimeSeriesDataSet, dict[str, TimeSeriesDataSet]]:
    """TFT classifies one window at a time.

    Decoder length is 1 because ICS2 is a per-window decision, not a forecast.
    The target is categorical and the loss is cross-entropy, which is
    pytorch-forecasting's supported classification path -- TFT's variable
    selection and attention weights survive it unchanged, and those are the
    reason TFT owns this path rather than N-HiTS (ICS layer doc, section 4).
    """
    training = TimeSeriesDataSet(
        frame_train,
        **_common_kwargs(cfg, "observable_class"),
        time_varying_unknown_reals=MODEL_INPUTS,
        static_categoricals=["operating_mode"],
        target_normalizer=NaNLabelEncoder(add_nan=False),
    )
    others = {
        name: TimeSeriesDataSet.from_dataset(
            training, frame, predict=False, stop_randomization=True
        )
        for name, frame in frame_other.items()
    }
    return training, others


def class_order(dataset: TimeSeriesDataSet) -> list[str]:
    """The label order the model's output channels actually use.

    CRITICAL, and not the same as config.FAULT_CLASSES. NaNLabelEncoder assigns
    indices by sorting the labels alphabetically, so output channel 0 is
    "ball", not "healthy". Anything that maps a channel index back to a label
    -- evaluation, replay, the ONNX contract the .NET host reads, and the class
    weights fed to the loss -- must take the order from here. Assuming the
    config's order instead silently permutes every class and is not visible in
    the training loss.
    """
    encoder = dataset.target_normalizer
    mapping = dict(encoder.classes_)
    return [label for label, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def class_weights(frame: pd.DataFrame, order: list[str]) -> np.ndarray:
    """Inverse-frequency weights over the five classes, in model-output order.

    The labelling policy makes "healthy" dominant: every run starts healthy and
    faulted runs only earn their true label after stage 2. Unweighted training
    on that distribution produces a classifier that is accurate and useless.

    `order` must come from class_order() -- cross-entropy indexes this array by
    output channel, so a differently-ordered array weights the wrong classes.
    """
    counts = frame["observable_class"].value_counts().reindex(order).fillna(0.0)
    counts = counts.to_numpy(dtype=np.float64) + 1.0
    weights = counts.sum() / (len(order) * counts)
    return weights.astype(np.float32)
