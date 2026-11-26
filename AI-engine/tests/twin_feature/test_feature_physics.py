import math
import json
import numpy as np
from aiengine.twin.feature_physics import (FeaturePhysicsTwin, FeaturePhysicsParams, LifeRef, defect_energy_ratio, load_calibration)

BASE = np.array([1.0, 3.0, 4.0, 0.5, 0.1, 0.1, 0.1, 0.1] * 2)


def feats(r, rng=None, noise=0.0):
    f = BASE.copy()
    for a in (0, 8):
        for k in (3, 4, 5, 6, 7):
            f[a + k] *= r
    if rng is not None and noise:
        f = f * np.exp(rng.normal(0, noise, 16))
    return f


def run(rfun, T=100.0, dt=0.5, rpm=1800, noise=0.0, seed=0, params=None):
    tw = FeaturePhysicsTwin(params)
    rng = np.random.default_rng(seed)
    out = []
    t = 0.0
    for _ in range(24):
        tw.update(feats(1.0, rng, noise), rpm, t)
        t += dt
    while t < T:
        out.append((t, tw.update(feats(rfun(t), rng, noise), rpm, t)))
        t += dt
    return tw, out


def big_life():
    return FeaturePhysicsParams(life=LifeRef(c_over_p=1e3), law="power", r_fail=20.0)   # Miner never binds


def test_ratio_baseline_is_one():
    assert abs(defect_energy_ratio(BASE, BASE) - 1.0) < 1e-6
    assert abs(defect_energy_ratio(feats(5.0), BASE) - 5.0) < 1e-6


def test_flat_healthy_prior_small_d():
    tw, out = run(lambda t: 1.0, noise=0.03, seed=1, params=FeaturePhysicsParams(law="power", r_fail=20.0))
    assert out[-1][1].source == "prior" and tw.trend_fraction() < 0.1
    assert out[-1][1].miner_d < 0.05
    assert out[-1][1].health_index < 0.2


def test_ramp_monotone_rul_and_d():
    p = FeaturePhysicsParams(life=LifeRef(c_over_p=3.0), law="power", r_fail=20.0)
    tw, out = run(lambda t: math.exp(0.02 * t), T=120, params=p)
    tr = [(t, u) for t, u in out if u.source == "trend"]
    assert len(tr) > 50
    d = [u.miner_d for _, u in out]
    assert all(b >= a for a, b in zip(d, d[1:])) and d[-1] > d[0]
    ruls = [u.rul_hours for _, u in tr[10:]]
    assert ruls[-1] < ruls[0]
    assert all(b <= a + 1e-6 for a, b in zip(ruls, ruls[1:]))


def test_known_exponential_recovers_rul():
    b = 0.03
    tf = math.log(20.0) / b
    tw, out = run(lambda t: math.exp(b * t), T=0.8 * tf, params=big_life())
    t, u = out[-1]
    assert u.source == "trend"
    assert abs(u.rul_hours - (tf - t)) / (tf - t) < 0.05
    assert u.lo <= u.rul_hours + 1e-9 <= u.hi + 1e-9


def test_noise_robust():
    b = 0.03
    tf = math.log(20.0) / b
    tw, out = run(lambda t: math.exp(b * t), T=0.8 * tf, noise=0.05, seed=3, params=big_life())
    t, u = out[-1]
    assert u.source == "trend" and abs(u.rul_hours - (tf - t)) / (tf - t) < 0.3
    tw2, _ = run(lambda t: 1.0, noise=0.05, seed=4, params=FeaturePhysicsParams(law="power", r_fail=20.0))
    assert tw2.trend_fraction() < 0.2


def test_lref_speed_scaling():
    L = LifeRef(c_over_p=10.0)
    assert abs(L.hours(1800) / L.hours(3600) - 2.0) < 1e-9
    assert abs(L.hours(1800) - 1e6 / (60 * 1800) * 1000) < 1e-6
    Lt = LifeRef(table=((1000, 100.0), (3000, 50.0)))
    assert abs(Lt.hours(2000) - 75.0) < 1e-9 and Lt.hours(5000) == 50.0
    p1 = FeaturePhysicsParams(life=LifeRef(c_over_p=3.0), law="power", r_fail=20.0)
    _, o1 = run(lambda t: 1.0, T=20, rpm=1800, params=p1)
    _, o2 = run(lambda t: 1.0, T=20, rpm=3600, params=p1)
    assert abs(o2[-1][1].miner_d / o1[-1][1].miner_d - 2.0) < 1e-6


def test_miner_reaches_095_at_analytic_time():
    r = 2.0
    p = FeaturePhysicsParams(life=LifeRef(c_over_p=3.0), law="power", r_fail=20.0)
    life = p.life.hours(1800)
    t_fail = 0.95 * life / r ** 3
    tw = FeaturePhysicsTwin(p, baseline16=BASE)
    dt = t_fail / 1000
    tw.update(feats(r), 1800, 0.0)
    crossed = None
    for i in range(1, 1200):
        t = i * dt
        u = tw.update(feats(r), 1800, t)
        if crossed is None and u.miner_d >= 0.95:
            crossed = t
    assert abs(crossed - t_fail) / t_fail < 2e-3


def test_deterministic():
    a = run(lambda t: math.exp(0.02 * t), noise=0.05, seed=7)[1]
    b = run(lambda t: math.exp(0.02 * t), noise=0.05, seed=7)[1]
    assert [(u.rul_hours, u.miner_d, u.source) for _, u in a] == [(u.rul_hours, u.miner_d, u.source) for _, u in b]


# ---- round 2: bounded damage link -------------------------------------------------------------------------------
def bparams(k=1.0, r_fail=20.0, c=2.0):
    return FeaturePhysicsParams(life=LifeRef(c_over_p=c), law="bounded", k=k, r_fail=r_fail)


def test_bounded_d_growth_at_r20():
    k, c = 1.0, 2.0
    p = bparams(k, 20.0, c)
    life = p.life.hours(1800)
    tw = FeaturePhysicsTwin(p, baseline16=BASE)
    tw.update(feats(20.0), 1800, 0.0)
    u = tw.update(feats(20.0), 1800, 1.0)
    g = 1 + k * math.log(20.0)
    assert abs(u.miner_d - g ** 3 / life) < 1e-7
    assert g ** 3 < 100           # bounded: far below the legacy r^3 = 8000
    p2 = FeaturePhysicsParams(life=p.life, law="power", r_fail=20.0)
    tw2 = FeaturePhysicsTwin(p2, baseline16=BASE)
    tw2.update(feats(20.0), 1800, 0.0)
    assert tw2.update(feats(20.0), 1800, 1.0).miner_d > 50 * u.miner_d


def test_bounded_rul_positive_monotone_ramp():
    p = bparams(0.8, 20.0, 4.0)
    tw, out = run(lambda t: math.exp(0.02 * t), T=140, dt=1.0, params=p)
    tr = [u for _, u in out if u.source == "trend" and u.miner_d < 0.95]
    assert len(tr) > 40
    ruls = [u.rul_hours for u in tr]
    assert all(x > 0 for x in ruls)
    assert all(b <= a + 1e-6 for a, b in zip(ruls, ruls[1:]))


def test_bounded_miner_closed_form_matches_integration():
    # constant-slope exponential, Miner binding: closed-form miner RUL vs numerical integration of g^3/L
    k, b, y0, life = 0.7, 0.05, math.log(3.0), 30.0
    g0 = 1 + k * y0
    rem = 0.95
    s = ((g0 ** 4 + 4 * k * b * life * rem) ** 0.25 - g0) / (k * b)
    n = 200000
    ds = s / n
    D = sum((1 + k * (y0 + b * (i + 0.5) * ds)) ** 3 for i in range(n)) * ds / life
    assert abs(D - rem) < 1e-4


def test_params_loaded_from_calibration_file(tmp_path):
    f = tmp_path / "twin_feature_calibration_x.json"
    f.write_text(json.dumps({"k": 0.123, "r_fail": 33.0}))
    c = load_calibration(f)
    assert c["k"] == 0.123 and c["r_fail"] == 33.0
    d = load_calibration()                      # repo default: latest calibration file or the declared fallback
    p = FeaturePhysicsParams()
    assert p.k == d["k"] and p.r_fail == d["r_fail"] and p.law == "bounded"
