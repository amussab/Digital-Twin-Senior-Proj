"""v5_node tests: metric reuse, post-process equivalence (engine streaming vs reference), ONNX parity + determinism (skip if not exported)."""
import json

import numpy as np
import pandas as pd
import pytest

from aiengine import up2_metrics as M
from aiengine import up2b, v5_node as V
from aiengine.config import MODELS_DIR


def test_metrics_are_reused_not_redefined():
    assert V.M is M
    assert not hasattr(V, "rul_mape") and not hasattr(V, "macro_f1")
    # a known frame: constant 10 % over-prediction -> S7 = 10 %
    n = 20
    p = pd.DataFrame({"unit_id": "u", "rul_hours": np.arange(n, 0, -1) * 1.0, "life_hours": 20.0, "u_true": np.linspace(0, .95, n),
                      "in_deg_det": True, "in_deg": True, "in_deg_full": True})
    p["rul_pred_h"] = p["rul_hours"] * 1.1
    assert abs(M.rul(p)["S7_mape_pct"] - 10.0) < 1e-9


def test_feature_set_is_v3e_node():
    from aiengine import features as feat
    assert V.COLS == list(feat.V3E_COLUMNS) and V.COLS[0] == "hi"


def test_postprocess_engine_equals_reference():
    from aiengine.engine import HybridEngine
    pp = {"stage_k": 5, "stage_max_step": 1, "rul_alpha": 0.25, "rul_scale_early": 0.85, "rul_scale_late": 1.15}
    rng = np.random.default_rng(1)
    rr = np.abs(rng.normal(200, 80, 60))
    ss = rng.integers(1, 7, 60)
    rp = pd.DataFrame({"unit_id": "u", "window_index": np.arange(60), "rul_pred_windows_raw": rr, "stage_pred_raw": ss})
    ref = V.postprocess(rp, pp)
    eng = HybridEngine.__new__(HybridEngine)
    eng.pp = up2b.postprocess_contract(pp)
    b = type("B", (), {})()
    from collections import deque
    b.pp_r, b.pp_s, b.pp_hist = None, None, deque()
    got = [eng._postprocess(b, float(r), int(s)) for r, s in zip(rr, ss)]
    assert np.allclose([g[0] for g in got], ref["rul_pred_windows"].to_numpy())
    assert [g[1] for g in got] == ref["stage_pred"].tolist()


def test_cadence_matches_table():
    from aiengine.synth_v3 import generator as gen
    assert abs(gen.CADENCE_H - V.K.CADENCE_H) < 1e-12


needs = pytest.mark.skipif(not (MODELS_DIR / V.CONTRACT).exists() or not V.RUN_DIR.joinpath("heads.pkl").exists(), reason="v5_node not exported")


@needs
def test_onnx_parity_on_val_units():
    fz, _ = V.load_frozen()
    heads = V.load_heads()
    tab = V.engineer(V.load_table())
    va = V.FeatSet(V.split_of(tab, "val"))
    par = V.parity(heads, fz, va, MODELS_DIR / V.CLS_ONNX, MODELS_DIR / V.RUL_ONNX, max_units=3)
    assert par["cls_prob_max_abs_diff"] < 1e-4 and par["cls_argmax_mismatch"] == 0
    assert par["rul_windows_max_rel_diff"] < 5e-3 and par["z_max_abs_diff"] < 5e-3
    assert par["stage_mismatch"] <= 2


@needs
def test_contract_loads_in_nodeengine_and_golden_replays():
    from aiengine.engine import HybridEngine
    con = json.loads((MODELS_DIR / V.CONTRACT).read_text())
    assert con["contract_version"] == V.CONTRACT_VERSION and con["models"]["nhits"]["encoder_length"] == V.NL
    g = json.loads((MODELS_DIR / V.GOLDEN).read_text())
    eng = HybridEngine(MODELS_DIR, contract_file=V.CONTRACT)
    ins = g["inputs"]
    byw = {c["window_index"]: c for c in g["cases"]}
    for r in ins:
        o = eng.process(r["rpm"], np.concatenate([r["features16"], r["features16"]]), r["t20_ms"])["bearings"][0]
        if r["window_index"] in byw and o.get("status") == "ok":
            e = byw[r["window_index"]]["output"]
            assert abs(o["rul_hours"] - e["rul_hours"]) <= 1e-3 * max(abs(e["rul_hours"]), 1.0)
            assert o["health_stage"] == e["health_stage"]


@needs
def test_determinism_of_frozen_scoring():
    fz, _ = V.load_frozen()
    heads = V.load_heads()
    tab = V.engineer(V.load_table())
    va = V.FeatSet(V.split_of(tab, "val").query("rig_unit in @UNITS", local_dict={"UNITS": sorted(tab[tab.split == "val"].rig_unit.unique())[:4]}))
    a = V.full_scores(va, V.predict_all(heads, fz, va))
    b = V.full_scores(va, V.predict_all(heads, fz, va))
    assert a["S7"]["S7_mape_pct"] == b["S7"]["S7_mape_pct"] and a["IS3_classification"]["macro_f1"] == b["IS3_classification"]["macro_f1"]
