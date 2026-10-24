"""COE domain-gap adaptation tests: simulator generalisation, plan/split, calibration maps, gate, exported contract.

Run: AI-engine/.venv/Scripts/python.exe -m pytest AI-engine/tests/coe_adapt -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aiengine import coe_adapt as A
from aiengine import coe_dataset as D
from aiengine.coe.sources import SimulatedRigSource
from aiengine.config import MODELS_DIR


def _codes(src):
    return [(b.codes.copy(), b.edges_s.copy()) for b in src.blocks()]


@pytest.mark.parametrize("fb", [1, 2])
def test_dual_source_reproduces_single_fault_source(fb):
    a = _codes(SimulatedRigSource("outer_race", 0.7, 3600.0, seed=11, n_blocks=40, faulted_bearing=fb))
    b = _codes(D.DualFaultRigSource({fb: ("outer_race", 0.7)}, 3600.0, seed=11, n_blocks=40))
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_dual_source_healthy_and_second_fault():
    a = _codes(SimulatedRigSource("healthy", 0.0, 1750.0, seed=3, n_blocks=20))
    b = _codes(D.DualFaultRigSource({}, 1750.0, seed=3, n_blocks=20))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    c = _codes(D.DualFaultRigSource({1: ("outer_race", 0.7), 2: ("inner_race", 0.7)}, 1750.0, seed=3, n_blocks=20))
    assert not all(np.array_equal(x[0], y[0]) for x, y in zip(a, c))


def test_plan_deterministic_and_split_by_run():
    p1, p2 = D.plan(60), D.plan(60)
    assert [s.to_dict() for s in p1] == [s.to_dict() for s in p2]
    assert {s.split for s in p1} == {"train", "val", "test"}
    assert len({s.run_id for s in p1}) == 60


def test_simulate_run_labels():
    spec = [s for s in D.plan(40) if s.scenario == "b1"][0]
    res = D.simulate_run(spec, healthy_windows=26, fault_windows=4)
    t = D.to_table(res, healthy_windows=26)
    b1, b2 = t[t["bearing"] == 1], t[t["bearing"] == 2]
    assert (b1["observable_class"].iloc[:27] == "healthy").all() and (b1["observable_class"].iloc[27:] == spec.b1[0]).all()
    assert (b2["observable_class"] == "healthy").all() and b2["partner_faulty"].iloc[-1]
    assert t["transition"].sum() == 2


def test_quantile_relu_form_equals_interp():
    rng = np.random.default_rng(0)
    coe, syn = rng.lognormal(0, 1, (4000, len(A.COLS))), rng.normal(0, 1, (4000, len(A.COLS)))
    coe[:, 3] = 1.0                                           # degenerate column
    qm = A.fit_quantile(coe, syn)
    x = rng.lognormal(0, 1.3, (50, 5, len(A.COLS))).astype(np.float32)
    assert np.abs(A.apply_quantile_relu(x, qm) - A.apply_quantile(x, qm)).max() < 1e-4


def test_gate_relabels_weaker_bearing_only():
    eng = pd.DataFrame({"unit_id": ["r:B1", "r:B2"], "window_index": [0, 0], "bearing_key": ["r", "r"],
                        "bearing": [1, 2], **{c: [0.0, 0.0] for c in A.COLS if c.startswith("log_")}})
    eng.loc[0, "log_bpfo_ratio"], eng.loc[1, "log_bpfo_ratio"] = np.log(1 + 20.0), np.log(1 + 5.0)
    pred = pd.DataFrame({"unit_id": ["r:B1", "r:B2"], "window_index": [0, 0], "pred_class": ["outer_race", "outer_race"]})
    g = A.gate(pred, eng, 0.5, "family")
    assert list(g["pred_class"]) == ["outer_race", "healthy"]
    assert list(A.gate(pred, eng, 0.0)["pred_class"]) == ["outer_race", "outer_race"]


def test_split_registered_in_splits_json():
    from aiengine import splits
    sec = splits.load()["sections"].get(D.DATASET)
    if sec is None:
        pytest.skip("synth_coe not generated yet")
    runs = pd.Series(sec["units"]).groupby(lambda u: u.rsplit(":", 1)[0]).nunique()
    assert (runs == 1).all() and set(sec["units"].values()) == {"train", "val", "test"}


_CON = MODELS_DIR / "model_contract_v3b_coe.json"


@pytest.mark.skipif(not _CON.exists(), reason="not exported yet")
def test_contract_keeps_v3b_interface():
    new, old = json.loads(_CON.read_text()), json.loads((MODELS_DIR / "model_contract_v3b.json").read_text())
    for k in ("onnx_inputs", "outputs", "encoder_length", "decoder_length", "columns", "scalers"):
        assert new["models"]["tft"][k] == old["models"]["tft"][k]
    assert new["models"]["nhits"] == old["models"]["nhits"]
    assert new["postprocess"] == old["postprocess"] and new["classes"] == old["classes"]
    assert new["models"]["tft"]["parity"]["argmax_mismatch"] == 0


@pytest.mark.skipif(not _CON.exists(), reason="not exported yet")
def test_golden_rig_stream_replays():
    from aiengine.engine import HybridEngine
    gold = json.loads((MODELS_DIR / "golden_vectors_v3b_coe.json").read_text())["rig_stream"]
    eng = HybridEngine(MODELS_DIR, contract_file=_CON.name)
    for inp, exp in zip(gold["inputs"], gold["outputs"]):
        o = eng.process(inp["rpm"], np.asarray(inp["features32"]), inp["t20_ms"])["bearings"]
        for b, e in zip(o, exp):
            assert b["status"] == e["status"] and b.get("fault_class") == e.get("fault_class")
            if "class_probs" in e:
                assert max(abs(b["class_probs"][c] - e["class_probs"][c]) for c in e["class_probs"]) < 1e-4
