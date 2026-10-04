"""Window-table schema shared by every dataset adapter (DESIGN.md section 3)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import dsp
from ..config import BANDPASS_HZ, BEARING_FEATURES, CACHE_DIR, ENVELOPE_RATE_HZ, FFT_SIZE

META_COLUMNS = [
    "dataset", "unit_id", "bearing_key", "window_index", "t_hours", "rpm",
    "fault_class", "run_to_failure", "life_hours", "rul_hours", "severity",
]
TRAILER_COLUMNS = ["fs_hz", "band_hz", "orders"]
SCHEMA = META_COLUMNS + BEARING_FEATURES + TRAILER_COLUMNS
VALID_CLASSES = {"healthy", "outer_race", "inner_race", "ball", "cage", "mixed", "unknown"}


def pipeline(fs_hz: float, orders: dict[str, float]) -> dsp.PipelineConfig:
    """COE Components 3-7 configured for one data source (same band/envelope/FFT as the rig)."""
    return dsp.PipelineConfig(
        sample_rate_hz=fs_hz,
        bandpass_hz=BANDPASS_HZ,
        envelope_rate_hz=ENVELOPE_RATE_HZ,
        fft_size=FFT_SIZE,
        bearing_orders=dict(orders),
    )


@dataclass
class UnitMeta:
    dataset: str
    unit_id: str
    bearing_key: str
    fault_class: str
    run_to_failure: bool
    life_hours: float = float("nan")
    severity: str = ""


def row(meta: UnitMeta, window_index: int, t_hours: float, rpm: float,
        feats16: np.ndarray, cfg: dsp.PipelineConfig) -> dict:
    """One schema row for one bearing-window."""
    rul = meta.life_hours - t_hours if meta.run_to_failure else float("nan")
    out = {
        "dataset": meta.dataset, "unit_id": meta.unit_id, "bearing_key": meta.bearing_key,
        "window_index": int(window_index), "t_hours": float(t_hours), "rpm": float(rpm),
        "fault_class": meta.fault_class, "run_to_failure": bool(meta.run_to_failure),
        "life_hours": float(meta.life_hours), "rul_hours": float(rul), "severity": meta.severity,
    }
    out.update({name: float(v) for name, v in zip(BEARING_FEATURES, feats16)})
    out["fs_hz"] = float(cfg.sample_rate_hz)
    out["band_hz"] = f"{cfg.bandpass_hz[0]:.0f}-{cfg.bandpass_hz[1]:.0f}"
    out["orders"] = json.dumps({k: round(v, 4) for k, v in cfg.bearing_orders.items()})
    return out


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the schema: columns, types, classes, consecutive window indices per unit."""
    missing = [c for c in SCHEMA if c not in frame.columns]
    if missing:
        raise ValueError(f"window table missing columns: {missing}")
    frame = frame[SCHEMA].copy()
    bad = set(frame["fault_class"].unique()) - VALID_CLASSES
    if bad:
        raise ValueError(f"invalid fault_class values: {bad}")
    for col in BEARING_FEATURES:
        frame[col] = frame[col].astype("float32")
    frame = frame.sort_values(["unit_id", "window_index"]).reset_index(drop=True)
    for unit, g in frame.groupby("unit_id"):
        idx = g["window_index"].to_numpy()
        if not np.array_equal(idx, np.arange(len(idx))):
            raise ValueError(f"unit {unit}: window_index must be 0..n-1 consecutive")
    if not np.isfinite(frame[BEARING_FEATURES].to_numpy()).all():
        raise ValueError("non-finite feature values")
    return frame


def save(frame: pd.DataFrame, name: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    validate(frame).to_parquet(CACHE_DIR / f"{name}.parquet", index=False)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / f"{name}.parquet")
