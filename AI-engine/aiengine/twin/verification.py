"""[SIMULATION] verification harness for S2 (K identification error) and physics RUL.

Every number produced here is Rank 6 evidence at best: forward-model synthetic data
(the same model family that does the identification, i.e. an "inverse crime" test) plus noise.
It is NOT a measurement on the RK4 rig.
"""
import numpy as np
from .fe_beam import FeBeam
from .identify import identify, to_complex
from .physics_rul import PhysicsTwin
from .params import DEFAULT

AMP_NOISE = 0.05      # +/-5 % uniform multiplicative amplitude noise
PHASE_NOISE = 0.05    # +/-0.05 rad (~2.9 deg) uniform additive phase noise
N_TRIALS = 40

# (name, K1/K0, K2/K0)
CASES = [("healthy", 1.0, 1.0), ("K1 -10%", 0.9, 1.0), ("K2 -15%", 1.0, 0.85),
         ("both -20%", 0.8, 0.8), ("asym -25%/-5%", 0.75, 0.95), ("K1 -30%", 0.7, 1.0)]
RPMS = (1750.0, 3600.0)


def noisy(beam, k1, k2, rpm, unb, rng, amp_noise=AMP_NOISE, ph_noise=PHASE_NOISE):
    a1, f1, a2, f2 = beam.probes(k1, k2, rpm, unb)
    a1 *= 1 + rng.uniform(-amp_noise, amp_noise)
    a2 *= 1 + rng.uniform(-amp_noise, amp_noise)
    f1 += rng.uniform(-ph_noise, ph_noise)
    f2 += rng.uniform(-ph_noise, ph_noise)
    return a1, f1, a2, f2


def s2_verification(seed=20261004, n_trials=N_TRIALS):
    p = DEFAULT
    beam = FeBeam(p)
    unb = p.unbalance_nominal * np.exp(1j * p.unbalance_phase_nominal)   # known reference
    rng = np.random.default_rng(seed)
    rows = []
    for name, f1, f2 in CASES:
        k1t, k2t = p.k_nominal[0] * f1, p.k_nominal[1] * f2
        for rpm in RPMS:
            for noisy_case in (False, True):
                errs = []
                stds = []
                for _ in range(n_trials if noisy_case else 1):
                    m = noisy(beam, k1t, k2t, rpm, unb, rng) if noisy_case else beam.probes(k1t, k2t, rpm, unb)
                    r = identify(beam, rpm, to_complex(*m), unb)
                    errs.append(max(abs(r.k1 - k1t) / k1t, abs(r.k2 - k2t) / k2t) * 100)
                    stds.append(max(r.rel_std1, r.rel_std2) * 100)
                rows.append(dict(case=name, k1_true=k1t, k2_true=k2t, rpm=rpm,
                                 noise="+/-5% amp, +/-0.05 rad phase" if noisy_case else "none",
                                 n_trials=len(errs), err_mean_pct=float(np.mean(errs)),
                                 err_p95_pct=float(np.percentile(errs, 95)), err_max_pct=float(np.max(errs)),
                                 predicted_rel_std_pct=float(np.mean(stds))))
    return rows


def rul_run(seed=7, decay_per_hour=0.0016, dt_h=1.0, rpm=1750.0, noise=True):
    """Synthetic exponential K decay on bearing 1 (bearing 2 healthy). Returns checkpoints."""
    p = DEFAULT
    rng = np.random.default_rng(seed)
    beam = FeBeam(p)
    unb = p.unbalance_nominal * np.exp(1j * p.unbalance_phase_nominal)
    twin = PhysicsTwin(p)
    k0 = p.k_nominal[0]
    t_fail = np.log(1 - p.failure_drop) / -decay_per_hour       # true time K1 hits the 25 % drop
    t = 0.0
    out = []
    checkpoints = {0.5, 0.7, 0.85}
    seen = set()
    last_u = None
    while t < t_fail * 0.95:
        k1 = k0 * np.exp(-decay_per_hour * max(t - p.n_commission * dt_h, 0.0))
        m = noisy(beam, k1, p.k_nominal[1], rpm, unb, rng) if noise else beam.probes(k1, p.k_nominal[1], rpm, unb)
        u = twin.update(rpm, *m, t)
        last_u = u
        t_fail_abs = p.n_commission * dt_h + t_fail
        frac = t / t_fail_abs
        for c in checkpoints:
            if frac >= c and c not in seen and u.physics_rul_hours is not None:
                seen.add(c)
                truth = t_fail_abs - t
                out.append(dict(life_fraction=c, t_hours=t, rul_est_h=u.physics_rul_hours, rul_true_h=truth,
                                rul_err_pct=abs(u.physics_rul_hours - truth) / truth * 100,
                                interval_h=[u.rul_lo_hours, u.rul_hi_hours],
                                drop_est_pct=u.stiffness_drop_pct))
        t += dt_h
    return out


def rul_summary(seeds=range(100, 110), **kw):
    """Run rul_run over several seeds; per life-fraction mean/max RUL error [%]."""
    runs = [rul_run(seed=s, **kw) for s in seeds]
    summ = []
    for c in (0.5, 0.7, 0.85):
        errs = [float(r["rul_err_pct"]) for run in runs for r in run if r["life_fraction"] == c]
        if errs:
            summ.append(dict(life_fraction=c, n_runs=len(errs), err_mean_pct=float(np.mean(errs)),
                             err_max_pct=float(np.max(errs))))
    return dict(seeds=list(seeds), decay_per_hour=kw.get("decay_per_hour", 0.0016), checkpoints=summ)
