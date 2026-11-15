"""HYBRID RIG DEMO engine (`NODE_CONTRACT=hybrid_rig`): fault class from the real-data model, stage + RUL from the simulation-anchored heads.

    class / baseline policy / onset / health index  <- v4_real  (trained on real node windows; realtime NodeEngine)
    stage, AI RUL, physics RUL, Miner D             <- NODE_RUL_CONTRACT (default v6_node; v7_node swaps in by name) on a TABLE clock

The RUL heads were trained on 10-minute simulated snapshots, live windows arrive every 1200/rpm s. Each live window is therefore mapped to
`NODE_RUL_TIMESCALE` table snapshots (DEMO time-scale, default 1.0 = one 10-minute snapshot per live window), so the heads see the
trajectory shape they were trained on; RUL is in TABLE hours, not wall-clock hours. Stage / RUL / physics RUL are issued only when v4_real
calls the bearing faulted (not healthy / warming_up) after the baseline policy, and RUL additionally needs stage >= 2 (as trained); else n/a.
The RUL engine shares v4_real's commissioning baseline (same raw median), so it never self-baselines on a faulted bearing.

PROVENANCE (non-visual, carried in payload / logs): rul_source = RUL_SOURCE below. The sim heads are NOT validated on rig data.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .config import MODELS_DIR
from .node_adapter import NodeEngine, _as_dict

RUL_SOURCE = "sim-trained head (hybrid_rig), not validated on rig"
CLS_CONTRACT = "v4_real"
DEFAULT_RUL_CONTRACT = "v6_node"


class HybridNodeEngine:
    def __init__(self, cls_contract: str = CLS_CONTRACT, rul_contract: str | None = None, timescale: float | None = None,
                 policy="contract", model_dir: Path = MODELS_DIR, threads: int = 1):
        self.cls_name = cls_contract
        self.rul_name = rul_contract or os.environ.get("NODE_RUL_CONTRACT", DEFAULT_RUL_CONTRACT)
        self.timescale = float(timescale if timescale is not None else os.environ.get("NODE_RUL_TIMESCALE", 1.0))
        self.cls = NodeEngine(f"model_contract_{cls_contract}.json", model_dir=model_dir, threads=threads, policy=policy)
        # RUL engine: table clock; policy 'contract' is irrelevant because the baseline is injected from the class engine.
        self.rul = NodeEngine(f"model_contract_{self.rul_name}.json", model_dir=model_dir, threads=threads, policy="contract",
                              time_base="table", table_step=self.timescale)
        self.contract = self.cls.contract
        self.n = 0

    # persistence / policy are the class engine's (the RUL engine borrows its baseline)
    @property
    def saved(self): return self.cls.saved
    @property
    def policy(self): return self.cls.policy
    def save_baseline(self, path): self.cls.save_baseline(path)
    def load_baseline(self, path):
        self.cls.load_baseline(path)
        self._share_baseline()

    def reset(self):
        self.cls.reset(); self.rul.reset(); self.n = 0

    def _share_baseline(self):
        self.rul.saved = {k: dict(v) for k, v in self.cls.saved.items()}

    def meta(self) -> dict:
        return {"rul_source": RUL_SOURCE, "rul_contract": self.rul_name, "class_contract": self.cls_name,
                "rul_timescale": self.timescale, "rul_time_unit": "table hours (10-min simulated snapshot per live window x timescale)"}

    def predict(self, message) -> dict:
        t0 = time.perf_counter()
        msg = _as_dict(message)
        out = self.cls.predict(msg)
        rul_b = {}
        if self.cls.phase in ("run", "legacy") and self.cls.key is not None:
            if self.cls.key not in self.rul.saved:
                self._share_baseline()
            r = self.rul.predict(msg)
            rul_b = {b["bearing"]: b for b in r["bearings"]}
        for b in out["bearings"]:
            self._merge(b, rul_b.get(b["bearing"]))
        out.update(self.meta())
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        self.n += 1
        return out

    def _merge(self, b: dict, r: dict | None) -> None:
        faulted = b.get("status") == "ok" and b.get("fault_class") not in (None, "healthy", "warming_up")
        rok = r is not None and r.get("status") == "ok"
        stage = int(r["health_stage"]) if rok and "health_stage" in r else None
        valid = bool(faulted and rok and stage is not None and stage >= 2)
        b["health_stage_v4_real"] = b.get("health_stage")
        b["health_stage"] = stage if (faulted and rok) else None
        b["time_base"] = "hybrid_table"
        b["rul_class_agrees"] = (r.get("fault_class") == b.get("fault_class")) if rok else None
        b["rul_head_class"] = r.get("fault_class") if rok else None
        for k in ("ai_rul_head_hours", "physics_rul_head_hours", "miner_d", "physics_health_index", "physics_source"):
            b[k] = r.get(k) if rok else None
        if not rok:
            b["miner_d"] = None
        b["ai_rul_hours"] = b["ai_rul_head_hours"] if valid else None
        b["physics_rul_hours"] = b["physics_rul_head_hours"] if valid else None
        if not valid:
            b["miner_d"] = b["miner_d"] if faulted and rok else None
        b["rul_valid"] = valid
        b["rul_source"] = RUL_SOURCE
        b["rul_note"] = ("hybrid_rig: class from %s; stage/RUL from %s on a table clock (x%g), table hours" % (self.cls_name, self.rul_name, self.timescale)
                         if valid else "n/a: RUL/stage issued only for a bearing classified faulted, after baseline, at stage >= 2")

    def predict_many(self, messages):
        for m in messages:
            yield self.predict(m)
