"""[SIMULATION] tests for the initial physics twin. Rank 6 evidence at best, not rig data.

Run: cd AI-engine && python -m pytest aiengine/twin/tests -q   (or: python aiengine/twin/tests/test_twin.py)
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aiengine.twin import FeBeam, identify, to_complex, PhysicsTwin, fit_rul, DEFAULT  # noqa: E402
from aiengine.twin.verification import s2_verification, rul_summary  # noqa: E402


def test_forward_sanity():
    b = FeBeam()
    p = DEFAULT
    # static check: very low speed, response is tiny and finite; stiffer bearing => smaller 1750 rpm amplitude
    a_soft = b.probes(4e4, 4e4, 1750.0)[0]
    a_stiff = b.probes(1.2e5, 1.2e5, 1750.0)[0]
    assert a_stiff < a_soft
    # shaft first natural frequency with rigid-ish supports is of the right order (beam theory check)
    w = np.sqrt(np.linalg.eigvals(np.linalg.solve(b.M, b.K + _kb(b, 1e9))).real.clip(0))
    w1 = np.sort(w)[0] / (2 * np.pi)
    # analytic check: pinned-pinned beam, point mass at a=0.16, b=0.28 from the supports, plus
    # ~0.4 of the shaft mass: k = 3 E I L / (a^2 b^2), f = sqrt(k / m_eff) / 2 pi  (~41.7 Hz)
    pr = b.p
    I = np.pi * pr.shaft_d ** 4 / 64
    L, a_, b_ = 0.44, 0.16, 0.28
    k = 3 * pr.E * I * L / (a_ ** 2 * b_ ** 2)
    m_eff = pr.disk_mass + 0.4 * pr.rho * np.pi * pr.shaft_d ** 2 / 4 * L
    f_an = np.sqrt(k / m_eff) / (2 * np.pi)
    assert abs(w1 - f_an) / f_an < 0.05


def _kb(b, k):
    K = np.zeros_like(b.K)
    for d in b.bdof:
        K[d, d] = k
    return K


def test_s2_noise_free_exact():
    rows = [r for r in s2_verification(n_trials=2) if r["noise"] == "none"]
    assert max(r["err_max_pct"] for r in rows) < 0.5


def test_s2_noisy_1750rpm_within_10pct():
    """S2 target <= 10 % (mean over noise trials), at the well-conditioned 1750 rpm mode [SIMULATION]."""
    rows = [r for r in s2_verification(n_trials=30) if r["noise"] != "none" and r["rpm"] == 1750.0]
    for r in rows:
        print(f"[SIMULATION] {r['case']:<14} 1750 rpm  mean {r['err_mean_pct']:.2f}%  p95 {r['err_p95_pct']:.2f}%")
        assert r["err_mean_pct"] <= 10.0


def test_s2_3600rpm_is_weakly_observable():
    """Documented limit: at 3600 rpm (above the bearing mode) K is poorly observable; the twin gates it."""
    rows = [r for r in s2_verification(n_trials=30) if r["noise"] != "none" and r["rpm"] == 3600.0]
    assert min(r["predicted_rel_std_pct"] for r in rows) > DEFAULT.max_rel_std * 100


def test_physics_rul_synthetic_decay():
    s = rul_summary(seeds=range(100, 106))
    for c in s["checkpoints"]:
        print(f"[SIMULATION] RUL at {c['life_fraction']*100:.0f}% life: mean err {c['err_mean_pct']:.1f}%")
    by = {c["life_fraction"]: c for c in s["checkpoints"]}
    assert by[0.7]["err_mean_pct"] < 25.0      # loose: the 1-sigma noise-limited regime
    # noise-free is an algorithm check, should be near-exact
    from aiengine.twin.verification import rul_run
    for r in rul_run(noise=False):
        assert r["rul_err_pct"] < 1.0


def test_no_rul_without_trend():
    t = np.arange(30.0)
    assert fit_rul(t[:3], [5e4] * 3, 5e4) is None
    assert fit_rul(t, 5e4 * np.ones(30), 5e4) is None    # flat
    assert fit_rul(t, 5e4 * (1 + 0.01 * np.sin(t)), 5e4) is None


def test_twin_commissioning_flag():
    tw = PhysicsTwin()
    b = FeBeam()
    unb = DEFAULT.unbalance_nominal * np.exp(1j * DEFAULT.unbalance_phase_nominal)
    m = b.probes(5e4, 5e4, 1750.0, unb)
    for i in range(DEFAULT.n_commission - 1):
        u = tw.update(1750.0, *m, float(i))
        assert u.commissioning and u.k1 is None
    u = tw.update(1750.0, *m, 99.0)
    assert not u.commissioning and abs(u.k1 - 5e4) / 5e4 < 1e-3 and abs(u.stiffness_drop_pct) < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
