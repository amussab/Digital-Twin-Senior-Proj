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
from datetime import datetime, timezone
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


DEFAULT_POLICY = {"settle_seconds": 120.0, "baseline_seconds": 60.0}
RPM_TOL = 0.05                      # rpm within 5 % of a stored baseline's rpm = same operating point


def policy_windows(rpm: float, policy: dict) -> tuple[int, int]:
    """(settle_windows, baseline_windows) from seconds and the node cadence: one window = 20 rev = 1200/rpm seconds."""
    wps = float(rpm) / (REV_PER_WINDOW * 60.0)
    return int(np.ceil(policy["settle_seconds"] * wps)), max(int(np.ceil(policy["baseline_seconds"] * wps)), 1)


class NodeEngine:
    """Streaming predictor over node messages. One instance per node/session; call `predict` once per window, in order.

    Baseline policy (contract `feature_engineering.baseline.policy`, or `policy=` / `settle_windows=` / `baseline_windows=`):
    skip `settle` windows after start / rpm change (rig warm-up), take the per-feature median of the next `baseline`
    windows, keyed per rpm. Contracts without a policy block (and policy=None) keep the legacy HybridEngine behaviour
    (median of the first 24 windows). `save_baseline` / `load_baseline` persist a commissioning baseline as JSON.
    """

    def __init__(self, contract_file: str = "model_contract_v3e_node.json", model_dir: Path = MODELS_DIR,
                 threads: int = 1, baselines: Optional[dict] = None, require_all_channels: bool = True,
                 policy="contract", settle_windows: Optional[int] = None,
                 baseline_windows: Optional[int] = None):
        self.engine = HybridEngine(model_dir, threads=threads, baselines=baselines, contract_file=contract_file)
        self.contract = self.engine.c
        self.require_all_channels = require_all_channels
        if policy == "contract":
            policy = self.contract.get("feature_engineering", {}).get("baseline", {}).get("policy")
        self.policy = dict(policy) if policy else None
        self.fixed_windows = (settle_windows, baseline_windows)
        if settle_windows is not None or baseline_windows is not None:
            self.policy = self.policy or dict(DEFAULT_POLICY)
        self.saved: dict[int, dict] = {}          # rpm key -> {"b1": [16], "b2": [16], ...}
        self.n = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self.key: Optional[int] = None            # rpm key of the active baseline / warm-up
        self.phase = "legacy" if self.policy is None else "idle"
        self.seen = 0                             # windows since (re)start at this rpm
        self.collect: list = []                   # [(msg, raw)] baseline windows (replayed once the baseline exists)
        self.need = (0, 0)

    def reset(self) -> None:
        self.engine.reset()
        self.n = 0
        self._reset_state()

    # ---- baseline persistence -------------------------------------------------------------------------------
    def save_baseline(self, path) -> None:
        if self.policy is None:
            raise ValueError("contract has no baseline policy")
        data = {"contract_version": self.contract.get("contract_version"), "policy": self.policy,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "baselines": {str(k): v for k, v in self.saved.items()}}
        Path(path).write_text(json.dumps(data, indent=1))

    def load_baseline(self, path) -> None:
        d = json.loads(Path(path).read_text())
        if d.get("contract_version") != self.contract.get("contract_version"):
            raise ValueError(f"baseline file contract {d.get('contract_version')!r} != {self.contract.get('contract_version')!r}")
        self.saved = {int(k): v for k, v in d["baselines"].items()}
        if self.policy is None:
            self.policy = dict(DEFAULT_POLICY)
        self.engine.baselines = {}
        self.engine.reset()
        self._reset_state()

    def _lookup(self, rpm: float) -> Optional[int]:
        best = min(self.saved, key=lambda k: abs(k - rpm), default=None)
        return best if best is not None and abs(best - rpm) <= RPM_TOL * rpm else None

    def _activate(self, key: int) -> None:
        sv = self.saved[key]
        self.engine.baselines = {1: np.asarray(sv["b1"]), 2: np.asarray(sv["b2"])}
        self.engine.reset()
        self.key, self.phase = key, "run"

    def _warm(self, msg, rpm, why):
        return {"window_id": msg.get("window_id"), "sequence": msg.get("sequence"), "rpm": rpm, "baseline_phase": why,
                "bearings": [{"bearing": k, "status": "warming_up", "fault_class": "warming_up", "reason": why}
                             for k in (1, 2)], "latency_ms": 0.0}

    def predict(self, message) -> dict:
        """One window -> {"window_id", "rpm", "bearings": [{bearing, status, fault_class, class_probs, onset_detected,
        transmitted_from_partner, health_stage, ...}, ...], "latency_ms": end-to-end host time incl. validation}.

        `status` is "warming_up" (fault_class "warming_up") until settle + baseline windows and the classifier buffer
        (encoder_length + 1 windows) are done; class_probs exist only when status == "ok".
        """
        t0 = time.perf_counter()
        msg = _as_dict(message)
        raw = node_json_to_raw(msg, self.require_all_channels)
        rpm = float(msg.get("rpm", 0.0))
        if not np.isfinite(rpm) or rpm <= 0:
            raise ValueError("rpm missing or invalid")
        if self.phase != "legacy":
            if self.key is not None and abs(rpm - self.key) > RPM_TOL * self.key:
                self.engine.baselines, self.key, self.phase = {}, None, "idle"      # rpm change: new baseline / buffers
                self.engine.reset()
                self.seen, self.collect = 0, []
            if self.phase == "idle":
                k = self._lookup(rpm)
                if k is not None:
                    self._activate(k)
                else:
                    self.key, self.phase, self.seen, self.collect = int(round(rpm)), "settle", 0, []
                    sw, bw = self.fixed_windows
                    a, b = policy_windows(rpm, self.policy)
                    self.need = (a if sw is None else sw, b if bw is None else bw)
            if self.phase == "settle":
                self.seen += 1
                if self.seen <= self.need[0]:
                    self.n += 1
                    return self._warm(msg, rpm, f"settling {self.seen}/{self.need[0]}")
                self.phase = "collect"
            if self.phase == "collect":
                self.collect.append((msg, raw))
                if len(self.collect) < self.need[1]:
                    self.n += 1
                    return self._warm(msg, rpm, f"baseline {len(self.collect)}/{self.need[1]}")
                from . import features as feat
                R = np.stack([r for _, r in self.collect]).astype(np.float64)
                self.saved[self.key] = {"rpm": self.key, "windows": len(self.collect), "settle_windows": self.need[0],
                                        "b1": feat._floor(np.median(R[:, :16], axis=0)).tolist(),
                                        "b2": feat._floor(np.median(R[:, 16:], axis=0)).tolist()}
                buf, self.collect = self.collect, []
                self._activate(self.key)
                for m, r in buf[:-1]:            # fill classifier buffer with the (healthy-by-procedure) baseline windows
                    self.engine.process(rpm, r, window_time_ms(m, rpm))
        out = self.engine.process(rpm, raw, window_time_ms(msg, rpm))
        if "error" in out:
            raise ValueError(out["error"])
        for b in out["bearings"]:
            if b.get("status") == "warming_up":
                b["fault_class"] = "warming_up"
        self.n += 1
        lat = (time.perf_counter() - t0) * 1000.0
        return {"window_id": msg.get("window_id"), "sequence": msg.get("sequence"), "rpm": rpm,
                "bearings": out["bearings"], "latency_ms": lat}

    def predict_many(self, messages: Iterable) -> Iterator[dict]:
        for m in messages:
            yield self.predict(m)


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
