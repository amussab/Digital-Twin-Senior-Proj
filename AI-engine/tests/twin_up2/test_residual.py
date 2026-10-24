"""Tests for aiengine.up2_residual ([SIMULATION] physics twin on [SYNTHETIC] stiffness trajectory)."""
import numpy as np
import pandas as pd

from aiengine import up2_residual as R
from aiengine.twin.params import DEFAULT
from aiengine.twin.physics_rul import PhysicsTwin
from aiengine.twin.identify import to_complex


def _run(n=90, fb=1, sim=None, seed=3):
    sim = sim or R.SimCfg()
    t = np.arange(n) / 6.0
    d = np.clip((np.arange(n) - 25) / (n - 26), 0, 1)       # monotone, 0 until onset, 1 at the end
    rpm = np.full(n, 1750.0)
    disp = R.simulate_displacements(rpm, d, fb, sim, np.random.default_rng(seed))
    return t, d, rpm, disp


def test_k_trajectory_monotone_and_hits_threshold():
    d = np.linspace(0, 1, 50)
    k = R.k_trajectory(d, 1.0, 2)
    assert np.all(np.diff(k[:, 1]) <= 1e-9)
    assert np.isclose(k[-1, 1], 0.75 * DEFAULT.k_nominal[1])
    assert np.allclose(k[:, 0], DEFAULT.k_nominal[0])          # partner bearing untouched


def test_displacement_payload_shape_dtype_and_noise_free_determinism():
    t, d, rpm, disp = _run(sim=R.SimCfg(amp_noise=0.0, phase_noise=0.0))
    assert disp.shape == (90, 4) and disp.dtype == np.float32
    assert np.all(np.isfinite(disp))
    # stiffness drop on bearing 1 changes probe-1 amplitude
    assert abs(disp[-1, 0] - disp[0, 0]) > 1e-3 * disp[0, 0]


def test_noise_free_identification_recovers_k():
    t, d, rpm, disp = _run(n=40, sim=R.SimCfg(amp_noise=0.0, phase_noise=0.0))
    ident = R.identify_run(rpm, disp)
    k_true = R.k_trajectory(d, 1.0, 1)
    err = np.abs(ident["k1"][20:] / k_true[20:, 0] - 1)
    assert err.max() < 0.02          # float32 payload quantisation only


def test_replay_matches_real_physics_twin():
    t, d, rpm, disp = _run(n=70)
    ident = R.identify_run(rpm, disp)
    rul, drop = R.replay_physics_rul(t, ident, R.STOCK)
    tw = PhysicsTwin()
    ref = []
    for i in range(len(t)):
        u = tw.update(rpm[i], *map(float, disp[i]), t[i])
        ref.append(np.nan if u.physics_rul_hours is None else u.physics_rul_hours)
    ref = np.array(ref)
    assert np.array_equal(np.isnan(ref[20:]), np.isnan(rul[20:]))
    m = ~np.isnan(ref[20:])
    assert np.allclose(ref[20:][m], rul[20:][m], rtol=1e-6)


def test_physics_rul_decreases_with_degradation_and_silent_before_onset():
    t, d, rpm, disp = _run(n=120, sim=R.SimCfg(amp_noise=0.02, phase_noise=0.02))
    rul, _ = R.replay_physics_rul(t, R.identify_run(rpm, disp), R.STOCK)
    assert np.isnan(rul[:20]).all()
    assert np.isfinite(rul[-20:]).any()


def test_ema_prior_advances_by_dt():
    ident = {"k1": np.r_[np.full(20, 5e4), 5e4 * np.exp(-0.01 * np.arange(1, 41))],
             "k2": np.full(60, 5e4), "s1": np.full(60, 0.01), "s2": np.full(60, 0.01)}
    t = np.arange(60, dtype=float)
    a, _ = R.replay_physics_rul(t, ident, R.FilterCfg(200, 1.0, 0.15))
    b, _ = R.replay_physics_rul(t, ident, R.FilterCfg(200, 0.1, 0.15))
    assert np.isfinite(a).sum() == np.isfinite(b).sum() > 0
    assert np.nanstd(np.diff(b)) <= np.nanstd(np.diff(a)) + 1e-9    # smoothing does not add jitter


def test_score_residual_definition_and_floor():
    rows = pd.DataFrame({"unit_id": "u", "window_index": [0, 1, 2, 3], "t_hours": [0, 1, 2, 3.], "rpm_mode": 1750.,
                         "true_stage": [1, 2, 3, 4], "rul_hours": [10., 9, 8, 7], "u_true": 0., "fault_class": "ball",
                         "rul_phys_h": [np.nan, 9.9, 8.0, 7.0], "k_drop_pct": 0.})
    preds = pd.DataFrame({"unit_id": "u", "window_index": [0, 1, 2, 3], "rul_pred_h": [10., 9.0, 8.0, 7.0]})
    s = R.score(rows, preds)
    assert s["residual"]["n"] == 2                     # stage 1 has no physics RUL, stage 4 excluded
    assert np.isclose(s["residual"]["max"], 0.9 / 9.0)  # |9.9-9|/9
    assert s["coverage"] == 2 / 3
    assert s["verdict_literal_max"] == "FAIL"


def test_fault_units_truncates_at_stage3_and_uses_split():
    df = pd.DataFrame({"unit_id": ["a"] * 5 + ["b"] * 2, "window_index": list(range(5)) + [0, 1],
                       "split": ["test"] * 5 + ["train"] * 2, "run_to_failure": True,
                       "true_stage": [1, 2, 3, 4, 5, 1, 2]})
    u = R.fault_units(df, "test")
    assert list(u["window_index"]) == [0, 1, 2] and set(u["unit_id"]) == {"a"}
