"""hybrid_rig: v4_real class + sim-head stage/RUL/physics on a table clock; v4_real behaviour unchanged; replay of recorded real runs.
[MEASURED host, replay of rig data] (same-bearing recordings, not independent)."""
import json

import numpy as np
import pytest

from aiengine.config import MODELS_DIR
from aiengine.hybrid_replay import DATA, load_runs, replay, summarise
from aiengine.node_adapter import NodeEngine
from aiengine.node_hybrid import RUL_SOURCE, HybridNodeEngine

BASE = MODELS_DIR / "baseline_v4_real.json"
pytestmark = pytest.mark.skipif(not (BASE.exists() and (MODELS_DIR / "model_contract_v6_node.json").exists() and DATA.exists()),
                                reason="v4_real/v6_node models or recorded rig data missing")


def _runs():
    return {m["run_id"]: (m, x) for m, x in load_runs(DATA)}


def pick(label):
    return next((m, x) for m, x in _runs().values() if m["label"] == label)


def test_class_identical_to_v4_real():
    m, msgs = pick("inner_race")
    h = HybridNodeEngine(); h.load_baseline(BASE)
    v = NodeEngine("model_contract_v4_real.json"); v.load_baseline(BASE)
    for msg in msgs[:120]:
        a, b = h.predict(msg)["bearings"], v.predict(msg)["bearings"]
        for x, y in zip(a, b):
            assert x["fault_class"] == y["fault_class"] and x["status"] == y["status"]
            assert x.get("class_probs") == y.get("class_probs")


def test_healthy_run_has_no_rul_or_stage_and_provenance_fields():
    m, msgs = pick("healthy")
    outs = replay(msgs, BASE, limit=150)
    s = summarise(m, outs)
    assert s["B1"]["rul_issued_pct"] == 0 and s["B2"]["rul_issued_pct"] == 0
    for o in outs:
        assert o["rul_source"] == RUL_SOURCE and o["rul_contract"] == "v6_node" and o["rul_timescale"] == 1.0 and o["class_contract"] == "v4_real"
        for b in o["bearings"]:
            assert b["rul_valid"] is False and b["ai_rul_hours"] is None and b["physics_rul_hours"] is None and b["health_stage"] is None


def test_fault_run_issues_stage_and_rul_with_physics_after_warmup():
    m, msgs = pick("outer_race")
    outs = replay(msgs, BASE, limit=200)
    b1 = [o["bearings"][0] for o in outs if o["bearings"][0]["status"] == "ok"]
    assert b1 and any(b["rul_valid"] for b in b1)
    v = next(b for b in b1 if b["rul_valid"])
    assert v["fault_class"] != "healthy" and v["health_stage"] >= 2 and v["ai_rul_hours"] > 0 and v["physics_rul_hours"] is not None
    assert v["miner_d"] is not None and v["time_base"] == "hybrid_table" and v["rul_source"] == RUL_SOURCE
    # warm-up windows: nothing issued
    for o in outs:
        if o["bearings"][0]["status"] != "ok":
            assert o["bearings"][0]["ai_rul_hours"] is None


def test_timescale_scales_table_clock_only():
    m, msgs = pick("inner_race")
    a = replay(msgs, BASE, timescale=1.0, limit=100)[-1]["bearings"][0]
    b = replay(msgs, BASE, timescale=2.0, limit=100)[-1]["bearings"][0]
    assert a["fault_class"] == b["fault_class"]
    assert HybridNodeEngine(timescale=2.0).timescale == 2.0
    with pytest.raises(ValueError):
        HybridNodeEngine(timescale=0)


def test_rul_contract_is_a_setting(monkeypatch):
    monkeypatch.setenv("NODE_RUL_CONTRACT", "v5_node")
    if not (MODELS_DIR / "model_contract_v5_node.json").exists():
        pytest.skip("v5 missing")
    assert HybridNodeEngine().rul_name == "v5_node"


def test_node_live_setup_hybrid(monkeypatch):
    from aiengine import node_live
    monkeypatch.setenv("NODE_CONTRACT", "hybrid_rig"); monkeypatch.delenv("NODE_BASELINE", raising=False)
    e = node_live.setup()
    assert isinstance(e, HybridNodeEngine) and e.meta()["rul_source"] == RUL_SOURCE


def test_replay_all_real_runs_numbers(tmp_path):
    """Honest replay numbers over every recorded run (same bearings as training: optimistic)."""
    rows = [summarise(m, replay(x, BASE)) for m, x in _runs().values()]
    (tmp_path / "r.json").write_text(json.dumps(rows))
    for r in rows:
        assert r["B2"]["rul_issued_pct"] < 25                      # healthy housing: little to no RUL issued
        if r["label"] == "healthy":
            assert r["B1"]["class_accuracy"] > 0.97 and r["B1"]["rul_issued_pct"] == 0
        else:
            assert r["B1"]["class_accuracy"] > 0.9 and r["B1"]["rul_issued_pct"] > 90
            assert r["B1"]["ai_rul_h"] is not None and r["B1"]["ai_rul_h"]["min"] >= 0
