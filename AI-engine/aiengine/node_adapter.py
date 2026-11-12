"""Adapter between the real COE node's JSON stream and the deployed AI post-processing.

The node (COE/firmware, COE/inference) sends one JSON object per 20-revolution window:

    {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": 1800, "window_id": 17,
     "sequence": 17, "features": [32 floats], ...}

`features` = B1X, B1Y, B2X, B2Y x (rms, kurtosis, crest, envelope_rms, ftf, bsf, bpfo, bpfi) in g. B2X is a COPY of
B1Y (4th sensor uncabled). That is exactly the model contract's raw order (bearing 1 = [0:16], bearing 2 = [16:32]).

    node_json_to_raw(msg) -> float32[32]            (validation only; no scaling, no reordering)
    NodeEngine(contract_file="model_contract_v3e_node.json")  (v3d_node: model_contract_v3d_node.json).predict(msg) -> per-bearing result dict

`NodeEngine` does NOT re-implement anything: it drives `aiengine.engine.HybridEngine` (features.py engineered
features with the 24-window median baseline, classifier ONNX, causal onset detector, RUL/stage heads and the
contract's cross_bearing_gate), the same code the C# host is pinned to by golden vectors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from .config import MODELS_DIR
from .engine import HybridEngine

SCHEMA_VERSION = 1
ALL_CHANNELS_MASK = 0b1111          # B1X, B1Y, B2X, B2Y
N_FEATURES = 32
REV_PER_WINDOW = 20                 # [firmware pipeline_config.h COE_WINDOW_REVOLUTIONS]


def node_json_to_raw(message, require_all_channels: bool = True) -> np.ndarray:
    """Validate one node message (dict or JSON str/bytes) and return its 32 features as float32 in contract order.

    Raises ValueError on: wrong schema_version, unit other than 'g', not all channels valid (unless
    require_all_channels=False), wrong feature count, non-finite values.
    """
    msg = _as_dict(message)
    if msg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {msg.get('schema_version')!r} (expected {SCHEMA_VERSION})")
    unit = msg.get("acceleration_unit")
    if unit != "g":
        raise ValueError(f"acceleration_unit must be 'g', got {unit!r}")
    vc = msg.get("valid_channels")
    if not isinstance(vc, (int, np.integer)) or isinstance(vc, bool):
        raise ValueError("valid_channels missing or not an integer bit mask")
    if require_all_channels and (int(vc) & ALL_CHANNELS_MASK) != ALL_CHANNELS_MASK:
        raise ValueError(f"valid_channels={vc}: not all four channels valid (need {ALL_CHANNELS_MASK})")
    feats = msg.get("features")
    if not isinstance(feats, (list, tuple)) or len(feats) != N_FEATURES:
        raise ValueError(f"features must be a list of {N_FEATURES} numbers")
    x = np.asarray(feats, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN/inf")
    return x.astype(np.float32)


def _as_dict(message) -> dict:
    if isinstance(message, (str, bytes, bytearray)):
        message = json.loads(message)
    if not isinstance(message, dict):
        raise ValueError("node message must be a JSON object")
    return message


def window_time_ms(msg: dict, rpm: float) -> int:
    """t20 time base for the engine (cadence/RUL hours only; classification does not use it): the message's own
    `t20_ms` if present, else window_id/sequence x one nominal 20-revolution window (the node has no wall clock field)."""
    if "t20_ms" in msg:
        return int(msg["t20_ms"]) & 0xFFFFFFFF
    wid = int(msg.get("window_id", msg.get("sequence", 0)))
    return int(round(wid * REV_PER_WINDOW * 60.0 / max(float(rpm), 1.0) * 1000.0)) & 0xFFFFFFFF


class NodeEngine:
    """Streaming predictor over node messages. One instance per node/session; call `predict` once per window, in order."""

    def __init__(self, contract_file: str = "model_contract_v3e_node.json", model_dir: Path = MODELS_DIR,
                 threads: int = 1, baselines: Optional[dict] = None, require_all_channels: bool = True):
        self.engine = HybridEngine(model_dir, threads=threads, baselines=baselines, contract_file=contract_file)
        self.contract = self.engine.c
        self.require_all_channels = require_all_channels
        self.n = 0

    def reset(self) -> None:
        self.engine.reset()
        self.n = 0

    def predict(self, message) -> dict:
        """One window -> {"window_id", "rpm", "bearings": [{bearing, status, fault_class, class_probs, onset_detected,
        transmitted_from_partner, health_stage, ...}, ...], "latency_ms": end-to-end host time incl. validation}.

        `status` is "warming_up" until the commissioning baseline (24 windows) and the classifier buffer
        (encoder_length + 1 windows) are filled; fault_class/class_probs exist only when status == "ok".
        """
        t0 = time.perf_counter()
        msg = _as_dict(message)
        raw = node_json_to_raw(msg, self.require_all_channels)
        rpm = float(msg.get("rpm", 0.0))
        if not np.isfinite(rpm) or rpm <= 0:
            raise ValueError("rpm missing or invalid")
        out = self.engine.process(rpm, raw, window_time_ms(msg, rpm))
        if "error" in out:
            raise ValueError(out["error"])
        self.n += 1
        lat = (time.perf_counter() - t0) * 1000.0
        return {"window_id": msg.get("window_id"), "sequence": msg.get("sequence"), "rpm": rpm,
                "bearings": out["bearings"], "latency_ms": lat}

    def predict_many(self, messages: Iterable) -> Iterator[dict]:
        for m in messages:
            yield self.predict(m)


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
