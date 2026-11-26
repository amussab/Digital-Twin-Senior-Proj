"""Real-window checks of v6_node through NodeEngine (same procedures as v5_real, v6 contract/baseline). [MEASURED on rig] but NOT independent:
the real windows are the anchor source of rig_anchor. Reported, never gated."""
from __future__ import annotations

import json

from . import v5_real as R5
from .config import MODELS_DIR

CONTRACT = "model_contract_v6_node.json"


def _engine6(contract: str):
    from .node_adapter import NodeEngine
    ne = NodeEngine(contract_file=contract, model_dir=MODELS_DIR)
    b = json.loads((MODELS_DIR / "baseline_v6_node.json").read_text())
    ne.saved = {int(k): v for k, v in b["baselines"].items()}
    return ne


def run(frozen: dict, heads: dict, fz: dict) -> dict:
    orig = R5._engine
    R5._engine = _engine6                                   # v5_real resolves _engine at call time: reuse its procedures unchanged
    try:
        trace = R5.failure_trace(frozen["baseline_control"], CONTRACT)
        real = R5.real_runs(CONTRACT)
        lat = R5.latency(CONTRACT)
    finally:
        R5._engine = orig
    trace = {k.replace("v5_", "v6_"): v for k, v in trace.items()}
    trace["samples"] = [{("v6_w" if k == "v5_w" else k): v for k, v in s.items()} for s in trace["samples"]]
    return {"real_windows": real, "failure_trace": trace, "latency": lat}
