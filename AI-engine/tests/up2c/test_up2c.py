"""UP2 v3c tests: physics features (fast identifier parity, streaming == offline replay, prior), feature layout,
ONNX/contract/golden (skip until exported), test3 registration.

Run: AI-engine/.venv/Scripts/python.exe -m pytest AI-engine/tests/up2c -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aiengine import up2, up2c
from aiengine import up2_window as uw
from aiengine import up2c_physics as P
from aiengine.config import MODELS_DIR, ROOT
from aiengine.twin.fe_beam import FeBeam
from aiengine.twin.identify import identify, to_complex


def _noisy_windows(n, rpm0, seed=0):
    rng = np.random.default_rng(seed)
    b = FeBeam()
    rpm = rpm0 * (1 + rng.uniform(-0.002, 0.002, n))
    k1 = 5e4 * (1 - 0.25 * rng.uniform(0, 1, n))
    k2 = 5e4 * (1 - 0.25 * rng.uniform(0, 1, n))
    unb = 1e-5 * np.exp(0.5j)
    D = np.array([b.probes(k1[j], k2[j], rpm[j], unb) for j in range(n)])
    D[:, [0, 2]] *= 1 + rng.uniform(-0.05, 0.05, (n, 2))
    D[:, [1, 3]] += rng.uniform(-0.05, 0.05, (n, 2))
    return b, rpm, D, unb


def test_condensed_response_exact():
    b = FeBeam()
    rpm = np.array([1750.0, 3600.0, 2900.0])
    h0, Pm, G, g = P.condensed(b, rpm)
    for i, (k1, k2) in enumerate([(5e4, 4e4), (3.9e4, 5e4), (4.4e4, 4.6e4)]):
        r = h0[i] - Pm[i] @ np.linalg.solve(G[i] + np.diag([1 / k1, 1 / k2]), g[i])
        ref = b.response(k1, k2, rpm[i], 1.0 + 0j)
        assert np.abs(r - ref).max() / np.abs(ref).max() < 1e-8


@pytest.mark.parametrize("rpm0", [1750.0, 3600.0])
def test_batch_identify_matches_stock_identifier(rpm0):
    b, rpm, D, unb = _noisy_windows(12, rpm0, seed=int(rpm0))
    R = P.batch_identify(rpm, D[:, 0], D[:, 1], D[:, 2], D[:, 3], unb, b)
    S = [identify(b, rpm[j], to_complex(*D[j]), unb) for j in range(len(rpm))]
    for j, s in enumerate(S):
        assert abs(R["k1"][j] / s.k1 - 1) < 1e-4 and abs(R["k2"][j] / s.k2 - 1) < 1e-4
        assert abs(R["s1"][j] / s.rel_std1 - 1) < 1e-3 and abs(R["s2"][j] / s.rel_std2 - 1) < 1e-3


def _toy_run(n=45, rtf=True):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"run_id": 777, "rpm": 1750.0 * (1 + rng.uniform(-0.002, 0.002, n)),
                         "t_hours": np.arange(n) / 6.0, "window_index": np.arange(n),
                         "damage": np.linspace(0, 1, n) ** 2 if rtf else np.zeros(n),
                         "run_to_failure": rtf, "faulted_bearing": 2})


def test_streaming_equals_offline_replay():
    g = _toy_run()
    r = P.run_features(("k", g, 1))
    pf = P.PhysicsFeatures()
    t = g["t_hours"].to_numpy()
    stream = np.array([pf.update(float(g["rpm"].iloc[i]), *map(float, r["disp"][i]), float(t[i])) for i in range(len(g))])
    off = r["feats"].astype(float)
    assert stream.shape == off.shape == (len(g), 2, 4)
    np.testing.assert_allclose(stream[:, :, [0, 3]], off[:, :, [0, 3]], atol=1e-4)
    np.testing.assert_allclose(stream[:, :, [1, 2]], off[:, :, [1, 2]], rtol=1e-4)


def test_prior_before_trend_and_coverage():
    g = _toy_run(30, rtf=False)
    f = P.run_features(("k", g, 1))["feats"]
    assert np.isfinite(f).all()
    # commissioning windows: no drop, nominal K, prior RUL, zero confidence
    np.testing.assert_allclose(f[:19, :, 0], 0.0)
    np.testing.assert_allclose(f[:19, :, 2], [[P.prior_rul(1750.0, t)] * 2 for t in g["t_hours"].iloc[:19]], rtol=1e-5)
    assert (f[:19, :, 3] == 0).all()
    # healthy rig: no significant stiffness trend -> prior everywhere (confidence 0), so a value always exists
    prior = f[:, :, 3] == 0
    assert prior.mean() > 0.9
    assert P.prior_rul(1750.0, 1000.0) == pytest.approx(P.PRIOR_FLOOR * P.L_POP[1750])
    assert P.prior_rul(3600.0, 6.0) == pytest.approx(P.L_POP[3600] - 6.0)


def test_old_feature_index_selects_v3b_layout():
    rng = np.random.default_rng(0)
    R, Rp = len(up2.COLUMNS), len(up2c.columns())
    blk = rng.normal(size=(5, 48, Rp)).astype(np.float32)
    full = uw.compute(blk, with_decoder=False)
    old = uw.compute(np.ascontiguousarray(blk[:, :, :R]), with_decoder=False)
    np.testing.assert_allclose(full[:, up2c.old_feature_index(R, Rp)], old, rtol=1e-6, atol=1e-6)


def test_test3_registered_as_holdout():
    sp = json.loads((ROOT / "splits.json").read_text())["sections"]
    sec = sp.get("synth_v3_test3")
    assert sec is not None and set(sec["units"].values()) == {"test"}
    assert not set(sec["units"]) & {u for k, s in sp.items() if k != "synth_v3_test3" for u in s["units"]}


CON = MODELS_DIR / "model_contract_v3c.json"


@pytest.mark.skipif(not CON.exists(), reason="v3c not exported yet")
def test_contract_and_onnx_parity():
    c = json.loads(CON.read_text())
    assert c["contract_version"] == "3c"
    assert c["models"]["nhits"]["columns"] == up2c.columns()
    pf = c["physics_features"]
    assert pf["columns"] == P.PHYS_COLUMNS and pf["rul_fit_points"] == P.RUL_FIT_POINTS
    assert c["postprocess"]["type"] == "v3b_stage_median_rul_logema" and c["postprocess"]["required"]
    par = c["models"]["nhits"]["parity"]
    assert par["rul_windows_max_rel_diff"] < 1e-4 and par["stage_mismatch"] == 0
    assert (MODELS_DIR / "v3b_rul.onnx").exists() and (MODELS_DIR / "model_contract_v3b.json").exists()   # v3b intact


@pytest.mark.skipif(not (MODELS_DIR / "golden_vectors_v3c.json").exists(), reason="v3c golden not exported yet")
def test_golden_engine_matches_offline():
    g = json.loads((MODELS_DIR / "golden_vectors_v3c.json").read_text())
    eo = g["engine_vs_offline"]
    assert eo["stage_mismatch"] == 0 and eo["rul_hours_max_rel_diff"] < 1e-3
    assert eo["phys_max_abs_diff_vs_offline_batch"] < 1e-2
    assert len(g["inputs"]) == len(g["physics"]) and all(len(i["displacement4"]) == 4 for i in g["inputs"])
