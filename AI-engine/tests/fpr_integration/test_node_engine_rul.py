"""NodeEngine v5/v6-style contract: AI RUL + physics twin fields, time bases, v3e/v4_real unchanged. [SIMULATION; rig-anchored]"""
import numpy as np
import pytest

from aiengine.node_adapter import NodeEngine
from aiengine.config import MODELS_DIR

C = "model_contract_v5_node.json"
pytestmark = pytest.mark.skipif(not (MODELS_DIR / C).exists(), reason="v5_node contract missing")


def msg(i, rpm=1800, scale=1.0, rng=None):
    base = np.array([0.04, 4.0, 6.0, 0.03, 0.001, 0.001, 0.001, 0.001] * 4) * scale
    f = base * (rng.normal(1, 0.03, 32) if rng is not None else 1.0)
    return {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": rpm, "window_id": i, "features": list(f)}


def run(e, n=60, scale_late=1.0):
    rng = np.random.default_rng(0)
    out = None
    for i in range(n):
        out = e.predict(msg(i, scale=scale_late if i > 40 else 1.0, rng=rng))
    return out


def test_table_time_base_has_physics_fields_and_gate():
    e = NodeEngine(C, policy=None, time_base="table")
    o = run(e, 60, scale_late=6.0)["bearings"][0]
    assert o["status"] == "ok" and o["time_base"] == "table"
    for k in ("ai_rul_hours", "ai_rul_head_hours", "physics_rul_hours", "miner_d", "physics_health_index", "physics_source", "rul_valid", "rul_note"):
        assert k in o
    assert o["ai_rul_head_hours"] is not None and o["physics_rul_head_hours"] is not None
    assert o["rul_valid"] == (o["ai_rul_hours"] is not None)
    if not o["rul_valid"]:
        assert o["ai_rul_hours"] is None and o["physics_rul_hours"] is None and o["rul_note"].startswith("n/a")


def test_healthy_stream_rul_is_na():
    e = NodeEngine(C, policy=None, time_base="table")
    o = run(e, 60)["bearings"][1]
    assert o["fault_class"] == "healthy" and o["ai_rul_hours"] is None and o["rul_valid"] is False


def test_realtime_time_base_is_na_but_has_health_index():
    e = NodeEngine(C, policy=None)            # default realtime
    o = run(e, 60, scale_late=6.0)["bearings"][0]
    assert o["time_base"] == "realtime" and o["ai_rul_hours"] is None and o["physics_rul_hours"] is None
    assert o["miner_d"] is None and o["physics_health_index"] is not None and o["rul_note"].startswith("n/a: live node")


def test_table_clock_independent_of_replay_timing():
    a, b = NodeEngine(C, policy=None, time_base="table"), NodeEngine(C, policy=None, time_base="table")
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    for i in range(55):
        m1, m2 = msg(i, 1800, 4.0 if i > 30 else 1.0, rng1), msg(i, 1800, 4.0 if i > 30 else 1.0, rng2)
        m2["t20_ms"] = i * 7919          # arbitrary replay clock must not matter
        o1, o2 = a.predict(m1), b.predict(m2)
    x, y = o1["bearings"][0], o2["bearings"][0]
    assert x["ai_rul_head_hours"] == y["ai_rul_head_hours"] and x["physics_rul_head_hours"] == y["physics_rul_head_hours"]


def test_older_contracts_unchanged():
    e = NodeEngine(policy=None)               # v3e_node default
    o = run(e, 40)["bearings"][0]
    assert "ai_rul_hours" not in o and "physics_rul_hours" not in o


def test_table_needs_cadence_contract():
    with pytest.raises(ValueError):
        NodeEngine(time_base="table")         # v3e has no rul.table_cadence_hours
