"""v7 tests: generator PROFILE option is backwards compatible and does what it says; L-split registry; ONNX parity of the exported v7_node heads."""
from __future__ import annotations

import json

import numpy as np
import pytest

from aiengine.config import MODELS_DIR
from aiengine.rig_anchor import constants as K
from aiengine.rig_anchor import constants_profiles as CP
from aiengine.rig_anchor import generator as G
from aiengine.rig_anchor import pools as P
from aiengine.rig_anchor import registry_l as RL


@pytest.fixture(scope="module")
def pools():
    return P.Pools()


def _spec(cls="outer_race", split="val", i=3, rpm=1800):
    return {"unit_id": f"{K.DATASET}:u{i:04d}", "split": split, "fault_class": cls, "rpm": rpm}


def test_default_profile_is_identity(pools):
    s = _spec()
    a = G.generate_unit(s, pools)
    b = G.generate_unit(s, pools, profile=None)
    c = G.generate_unit(s, pools, profile={})
    d = G.generate_unit(s, pools, profile=CP.HARD)
    for o in (b, c, d):
        assert a.equals(o)


def test_profile_changes_only_what_it_declares(pools):
    s = _spec()
    base = G.generate_unit(s, pools)
    narrow = G.generate_unit(s, pools, profile={"life_sigma": 0.10})
    assert base["shape_a"].iloc[0] == narrow["shape_a"].iloc[0]          # same draws, only life scaled
    assert base["anchor_u"].iloc[0] == narrow["anchor_u"].iloc[0]


def test_life_sigma_controls_life_spread(pools):
    def spread(sig):
        L = []
        for i in range(60):
            s = _spec(i=i, rpm=1800)
            g = G.generate_unit(s, pools, profile={"life_sigma": sig})
            L.append(np.log(g[g["unit_id"].str.endswith(":B1")]["life_hours"].iloc[0]))
        return float(np.std(L))
    assert spread(0.10) < 0.5 * spread(0.35)


def test_l_registry_plan():
    plan = RL.plan_units_l()
    ids = [p["unit_id"] for p in plan]
    assert len(ids) == len(set(ids))
    assert sum(p["split"] == "test" for p in plan) >= 80
    assert all(i.startswith("rigL:") for i in ids)
    assert not {p["unit_id"] for p in plan} & {p["unit_id"] for p in G.plan_units()}      # new unit ids
    assert all(not (p["fault_class"] == "ball" and p["rpm"] != 1800) for p in plan)
    assert RL.plan_units_l() == plan                                                     # deterministic


def test_l_test_shape_is_wider_than_train():
    L = CP.PROFILES["L"]
    for k in ("a", "p"):
        lo, hi = L["shape_train"][k]
        tlo, thi = RL.SHAPE_TEST_L[k]
        assert tlo < lo and thi > hi


def test_onnx_parity_if_exported():
    f = MODELS_DIR / "model_contract_v7_node.json"
    if not f.exists():
        pytest.skip("v7_node not exported yet")
    par = json.loads(f.read_text())["models"]["tft"]["parity"]
    assert par["cls_argmax_mismatch"] == 0
    assert par["cls_prob_max_abs_diff"] < 1e-4
    assert par["rul_windows_max_rel_diff"] < 1e-3
    assert par["stage_mismatch"] == 0
