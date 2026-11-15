"""Live runner for the COE receiver: importable by COE/inference/model_hook.py (no COE files are touched here).

    from aiengine.node_live import setup, predict
    model = setup()                       # NodeEngine, contract v3e_node (env NODE_CONTRACT), optional env NODE_BASELINE file
    line  = predict(window, model)        # one-line string per window

`window` = receiver dict: features (32), rpm, window_id, valid_channels, received_at (epoch s or ISO str).
Env: NODE_TIME_BASE (realtime|table; table = simulated rig_anchor replay, RUL/physics shown), NODE_CONTRACT (default v3e_node), NODE_BASELINE (JSON saved by NodeEngine.save_baseline; loaded if it exists, and
re-saved whenever a new rpm baseline is captured), NODE_SETTLE_S / NODE_BASELINE_S (override 120 / 60 s).
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from .node_adapter import NodeEngine


def setup() -> NodeEngine:
    contract = os.environ.get("NODE_CONTRACT", "v3e_node")
    pol = {"settle_seconds": float(os.environ.get("NODE_SETTLE_S", 120)),
           "baseline_seconds": float(os.environ.get("NODE_BASELINE_S", 60))}
    tb = os.environ.get("NODE_TIME_BASE", "realtime")      # "table" ONLY for the simulated-trajectory replay (rig_anchor); live node = realtime
    eng = NodeEngine(contract_file=f"model_contract_{contract}.json", policy=pol, time_base=tb)
    p = os.environ.get("NODE_BASELINE")
    eng._baseline_path = p
    eng._saved_keys = set()
    if p and Path(p).exists():
        eng.load_baseline(p)
        eng._saved_keys = set(eng.saved)
    return eng


def _recv_epoch(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def predict_full(window: dict, model: NodeEngine):
    """Like predict() but returns (engine_output_dict | None, one_line_string)."""
    msg = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": window.get("valid_channels", 15),
           "rpm": window["rpm"], "window_id": window.get("window_id", 0), "features": list(window["features"])}
    try:
        out = model.predict(msg)
    except ValueError as e:
        return None, f"window {msg['window_id']}: rejected ({e})"
    p = getattr(model, "_baseline_path", None)
    if p and set(model.saved) != getattr(model, "_saved_keys", set()):
        model.save_baseline(p)
        model._saved_keys = set(model.saved)
    rt = _recv_epoch(window.get("received_at"))
    lat = f"{out['latency_ms']:.1f} ms" + (f" (e2e {1000 * (time.time() - rt):.0f} ms)" if rt else "")
    parts = []
    for b in out["bearings"]:
        if b.get("status") != "ok":
            parts.append(f"B{b['bearing']} warming_up {b.get('reason', '')}")
            continue
        c = b["fault_class"]
        extra = ""
        if b.get("rul_valid"):
            extra = f" stage={b.get('health_stage')} AIrul={b['ai_rul_hours']:.1f}h phys={b['physics_rul_hours']:.1f}h"
        parts.append(f"B{b['bearing']} {c} {b['class_probs'].get(c, 0):.2f} onset={'Y' if b.get('onset_detected') else 'n'}{extra}")
    if out["bearings"][0].get("status") != "ok" and "baseline_phase" in out:
        return out, f"warming_up {out['baseline_phase']} | {lat}"
    return out, f"w{msg['window_id']} {msg['rpm']:.0f}rpm | " + " | ".join(parts) + f" | {lat}"


def predict(window: dict, model: NodeEngine) -> str:
    return predict_full(window, model)[1]
