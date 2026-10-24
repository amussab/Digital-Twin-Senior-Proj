"""UP2 (contract v3) tests: generator determinism, split leakage, metric functions, ONNX parity.

Run: AI-engine/.venv/Scripts/python.exe -m pytest AI-engine/tests/up2 -q
ONNX parity tests skip only when the v3 models have not been exported yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiengine import up2
from aiengine import up2_metrics as M
from aiengine import up2_window as uw
from aiengine.config import MODELS_DIR, ROOT
from aiengine.synth_v3 import generator as gen


# --------------------------------------------------------------------------- generator

def test_generator_deterministic():
    a, b = gen.build(n_runs=6, seed=11), gen.build(n_runs=6, seed=11)
    assert gen.table_hash(a) == gen.table_hash(b)
    c = gen.build(n_runs=6, seed=12)
    assert gen.table_hash(a) != gen.table_hash(c)


def test_generator_labels_consistent():
    df = gen.build(n_runs=10, seed=5)
    r = df[df["run_to_failure"]]
    for _, g in r.groupby("unit_id"):
        g = g.sort_values("window_index")
        assert g["rul_hours"].iloc[-1] == pytest.approx(0.0)
        assert (np.diff(g["true_stage"].to_numpy()) >= 0).all()          # stages never go back
        assert g["true_stage"].iloc[0] == 1 and g["true_stage"].iloc[-1] == 6
        on = int(g["true_onset_index"].iloc[0])
        assert (g["u_true"].to_numpy()[:on] < 0).all() and g["u_true"].iloc[on] == pytest.approx(0.0)
    # every run: exactly one faulted bearing, or none for healthy runs
    per_run = df.groupby("bearing_key")["run_to_failure"].apply(lambda s: s.groupby(df.loc[s.index, "unit_id"]).first().sum())
    assert per_run.max() <= 1
    assert df.attrs["provenance"]["synthetic"] is True


def test_scenario_payload_roundtrip(tmp_path):
    from aiengine import payload
    from aiengine.synth_v3 import scenario
    out = tmp_path / "demo.bin"
    info = scenario.build("inner_race", out, n_healthy=60, n_deg=100)
    raw = out.read_bytes()
    assert len(raw) == info["n_payloads"] * payload.PAYLOAD_SIZE == 160 * 152
    w = payload.decode(raw[:152])
    assert w.features.shape == (32,) and abs(w.rpm - 1750) < 20
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["synthetic"] and meta["bearing_1"]["fault_class"] == "inner_race"
    assert meta["bearing_1"]["truth_per_window"]["health_stage"][-1] == 6


# --------------------------------------------------------------------------- splits

def test_split_no_leakage():
    sec = json.loads((ROOT / "splits.json").read_text())["sections"]["synth_v3"]["units"]
    runs = {}
    for uid, s in sec.items():
        runs.setdefault(uid.rsplit(":", 1)[0], set()).add(s)
    assert all(len(v) == 1 for v in runs.values()), "a run has bearings in different splits"
    n = pd.Series({k: next(iter(v)) for k, v in runs.items()}).value_counts()
    assert n["test"] >= 40 and n["train"] > n["test"]


def test_robustness_sets_disjoint_ids():
    # robustness sets use run_offset >= 10000 so their unit ids can never collide with frozen-split units
    df = gen.build(n_runs=2, seed=1, run_offset=10100, dataset="synth_v3_noise15")
    assert df["unit_id"].str.startswith("synth_v3_noise15:run101").all()


# --------------------------------------------------------------------------- metrics

def _rul_frame(pred_factor: float):
    n = 50
    u = np.linspace(0, 1, n)
    life = 100.0
    t = 20 + u * 80
    rul = life - t
    return pd.DataFrame({"unit_id": "a", "window_index": np.arange(n), "rul_hours": rul, "life_hours": life,
                         "u_true": u, "in_deg": u <= 0.9, "in_deg_det": (u >= 0.1) & (u <= 0.9), "in_deg_full": True, "rul_pred_h": rul * pred_factor})


def test_mape_definition():
    r = M.rul(_rul_frame(1.1))
    assert r["S7_mape_pct"] == pytest.approx(10.0, abs=1e-9) and r["alt_mape_true_onset_pct"] == pytest.approx(10.0)
    assert M.rul(_rul_frame(1.0))["S7_mape_pct"] == pytest.approx(0.0)


def test_stage_and_caught():
    p = pd.DataFrame({"unit_id": "a", "window_index": range(6), "health_stage": [1, 2, 2, 3, 4, 6],
                      "stage_pred": [1, 1, 3, 3, 4, 4]})
    s = M.stage(p)
    assert s["max_abs_error"] == 2 and s["within1_frac"] == pytest.approx(5 / 6)
    c = pd.DataFrame({"unit_id": "a", "window_index": range(6), "fault_class": "ball", "run_to_failure": True,
                      "health_stage": [1, 2, 2, 3, 4, 6], "pred_class": ["healthy", "ball", "ball", "ball", "ball", "ball"]})
    r = M.caught_by_stage3(c)
    assert r["caught_by_stage3"] == 1 and r["per_unit"][0]["caught_at_stage"] == 3


def test_classification_macro_f1():
    p = pd.DataFrame({"unit_id": "a", "observable_class": ["healthy", "ball", "cage", "inner_race", "outer_race"] * 2,
                      "pred_class": ["healthy", "ball", "cage", "inner_race", "outer_race"] * 2})
    assert M.classification(p)["macro_f1"] == pytest.approx(1.0)


def test_stage_from_z():
    assert up2.stage_from_z(np.array([-0.3, 0.0, 0.19, 0.2, 0.99, 1.0, 1.4])).tolist() == [1, 2, 2, 3, 6, 6, 6]


# --------------------------------------------------------------------------- ONNX

def test_aggregator_matches_numpy():
    rng = np.random.default_rng(0)
    T, R = 13, 21
    x = rng.normal(size=(5, T, R)).astype(np.float32)
    f = uw.compute(x, with_decoder=True).reshape(5, len(uw.AGGS), R)
    W = uw.weight_matrix(T).astype(np.float64)
    xd = x.astype(np.float64)
    ref_lin = np.einsum("kt,btr->bkr", W, xd)
    np.testing.assert_allclose(f[:, :len(uw.LINEAR_AGGS)], ref_lin, atol=1e-4)
    np.testing.assert_allclose(f[:, -2], xd.std(axis=1), atol=1e-4)
    np.testing.assert_allclose(f[:, -1], xd.max(axis=1), atol=1e-6)
    np.testing.assert_allclose(f[:, 0], xd[:, -1], atol=1e-6)          # "last" = current row


def _v3_exported():
    return (MODELS_DIR / "model_contract_v3.json").exists() and (MODELS_DIR / "golden_vectors_v3.json").exists()


@pytest.mark.skipif(not _v3_exported(), reason="v3 models not exported yet")
def test_v3_golden_parity():
    """Re-run the reference engine (ONNX Runtime, contract v3) on the golden inputs; outputs must match."""
    from aiengine.engine import HybridEngine
    from aiengine.config import BEARING_FEATURES
    g = json.loads((MODELS_DIR / "golden_vectors_v3.json").read_text())
    eng = HybridEngine(MODELS_DIR, contract_file="model_contract_v3.json")
    outs = {}
    for w in g["inputs"]:
        f16 = np.asarray(w["features16"], np.float64)
        o = eng.process(w["rpm"], np.concatenate([f16, f16]), w["t20_ms"])["bearings"][0]
        outs[w["window_index"]] = o
    assert g["cases"], "golden file has no cases"
    for c in g["cases"]:
        o = outs[c["window_index"]]
        assert o["fault_class"] == c["output"]["fault_class"]
        assert o["health_stage"] == c["output"]["health_stage"]
        for k, v in c["output"]["class_probs"].items():
            assert abs(o["class_probs"][k] - v) < 1e-5
        assert o["rul_hours"] == pytest.approx(c["output"]["rul_hours"], rel=1e-4, abs=1e-6)


@pytest.mark.skipif(not _v3_exported(), reason="v3 models not exported yet")
def test_v3_contract_shape():
    c = json.loads((MODELS_DIR / "model_contract_v3.json").read_text())
    assert c["contract_version"] == 3 and c["provenance"]["synthetic"] is True
    assert c["classes"]["labels"] == sorted(c["classes"]["labels"])        # alphabetical channel order
    assert c["rul"]["method"] == "direct_v3"
    par = c["models"]["tft"]["parity"]
    assert par["cls_argmax_mismatch"] == 0 and par["cls_prob_max_abs_diff"] < 1e-5
    assert par["stage_mismatch"] == 0 and par["rul_windows_max_rel_diff"] < 1e-4
