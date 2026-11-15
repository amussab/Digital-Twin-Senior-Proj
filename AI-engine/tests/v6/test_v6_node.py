"""v6_node tests: aggregator parity, fresh-split integrity (registered, disjoint, test-pool only), metric reuse, ONNX parity + golden replay (skip if not exported)."""
import json

import numpy as np
import pandas as pd
import pytest

from aiengine import splits, up2_metrics as M
from aiengine import v5_node as V5, v6_agg as AG, v6_data as D, v6_node as V
from aiengine.config import MODELS_DIR
from aiengine.rig_anchor import constants as K
from aiengine.rig_anchor import pools as P


def test_metrics_are_reused_not_redefined():
    assert V.M is M and V.V5.M is M
    assert not hasattr(V, "rul_mape") and not hasattr(V, "macro_f1")


def test_aggregator_graph_equals_numpy_reference():
    rng = np.random.default_rng(0)
    for T in (48, 144):
        b = rng.normal(0.3, 0.4, (7, T, V.R)).astype(np.float32)
        b[:, :, 0] = np.abs(b[:, :, 0])
        got = AG.compute(b)
        ref = AG.reference(b)
        assert got.shape == (7, AG.n_features(V.R))
        assert np.allclose(got, ref, atol=1e-4, rtol=1e-4)
        assert len(AG.feature_names(V.COLS)) == got.shape[1]


def test_fresh_split_registered_and_disjoint():
    sp = splits.load()["sections"]
    assert D.SECTION in sp and D.verify()
    fresh = {u for u, s in sp[D.SECTION]["units"].items() if s == "test"}
    extra = {u for u, s in sp[D.SECTION]["units"].items() if s == "train"}
    old = set(sp[K.SPLIT_SECTION]["units"])
    ctrl = set(sp[D.SECTION]["control_units"])
    assert len(fresh) >= 60 and len(extra) >= 360
    assert not (fresh & old) and not (extra & old) and not (ctrl & old) and not (fresh & extra) and not (ctrl & (fresh | extra))
    comp = sp[D.SECTION]["composition"]["test"]
    assert all(k.startswith("ball@1800") or "ball" not in k for k in comp)          # ball only at 1800 rpm
    assert {k.split("@")[0] for k in comp} == set(K.CLASSES)


def test_fresh_units_draw_only_from_test_pool_and_load_refuses_train_val_leak():
    pools = P.Pools()
    allowed = pools.window_keys("test")
    pl = pd.DataFrame(D.plan("test"))
    spec = pl.iloc[3].to_dict()
    from aiengine.rig_anchor import generator as G
    df = G.generate_unit(spec, pools)
    keys = set(zip(df["f_run"], df["f_order"].astype(int))) | set(zip(df["h_run"], df["h_order"].astype(int)))
    assert keys <= allowed
    assert not (keys & pools.window_keys("train")) and not (keys & pools.window_keys("val"))


def test_plan_is_deterministic_and_balanced():
    a, b = D.plan("test"), D.plan("test")
    assert a == b and len(a) == 80
    assert [p["fault_class"] for p in a[:5]] == list(K.CLASSES)


needs = pytest.mark.skipif(not (MODELS_DIR / V.CONTRACT).exists() or not V.RUN_DIR.joinpath("heads.pkl").exists(), reason="v6_node not exported")


@needs
def test_onnx_parity_on_val_units():
    fz, _ = V.load_frozen()
    heads = V.load_heads()
    tab = V.load_trval()
    par = V.parity(heads, fz, tab[tab["split"] == "val"], MODELS_DIR / V.CLS_ONNX, MODELS_DIR / V.RUL_ONNX, max_units=3)
    assert par["cls_prob_max_abs_diff"] < 1e-4 and par["cls_argmax_mismatch"] == 0
    assert par["rul_windows_max_rel_diff"] < 5e-3 and par["z_max_abs_diff"] < 5e-3
    assert par["stage_mismatch"] <= 2


@needs
def test_contract_loads_in_engine_and_golden_replays():
    from aiengine.engine import HybridEngine
    con = json.loads((MODELS_DIR / V.CONTRACT).read_text())
    fz, _ = V.load_frozen()
    assert con["contract_version"] == V.CONTRACT_VERSION and con["models"]["nhits"]["encoder_length"] == fz["frozen"]["rul_NL"]
    g = json.loads((MODELS_DIR / V.GOLDEN).read_text())
    eng = HybridEngine(MODELS_DIR, contract_file=V.CONTRACT)
    byw = {c["window_index"]: c for c in g["cases"]}
    n = 0
    for r in g["inputs"]:
        o = eng.process(r["rpm"], np.concatenate([r["features16"], r["features16"]]), r["t20_ms"])["bearings"][0]
        if r["window_index"] in byw and o.get("status") == "ok":
            e = byw[r["window_index"]]["output"]
            assert abs(o["rul_hours"] - e["rul_hours"]) <= 1e-3 * max(abs(e["rul_hours"]), 1.0)
            n += 1
    assert n > 0
