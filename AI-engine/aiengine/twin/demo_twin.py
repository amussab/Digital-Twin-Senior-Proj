"""Demo: S2 verification table + physics-RUL check + golden vectors for the C# port.

    cd AI-engine && python -m aiengine.twin.demo_twin

Writes reports/twin_s2_verification.json and reports/twin_golden.json.
ALL numbers are [SIMULATION] (Rank 6): forward-model synthetic data + noise, not rig data.
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from .fe_beam import FeBeam
from .identify import identify, to_complex
from .params import DEFAULT, as_dict
from .physics_rul import PhysicsTwin
from .verification import s2_verification, rul_summary, noisy, CASES, AMP_NOISE, PHASE_NOISE, N_TRIALS

SEED = 20261004
REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_golden(seed=SEED):
    p = DEFAULT
    beam = FeBeam(p)
    rng = np.random.default_rng(seed + 1)
    unb = p.unbalance_nominal * np.exp(1j * p.unbalance_phase_nominal)
    fwd, ident = [], []
    for _, f1, f2 in CASES:
        for rpm in (1750.0, 3600.0, 2400.0):
            k1, k2 = p.k_nominal[0] * f1, p.k_nominal[1] * f2
            fwd.append(dict(k1=k1, k2=k2, rpm=rpm, unb_re=unb.real, unb_im=unb.imag,
                            probes=list(beam.probes(k1, k2, rpm, unb))))
        for noise in (False, True):
            m = noisy(beam, p.k_nominal[0] * f1, p.k_nominal[1] * f2, 1750.0, unb, rng) if noise \
                else beam.probes(p.k_nominal[0] * f1, p.k_nominal[1] * f2, 1750.0, unb)
            r = identify(beam, 1750.0, to_complex(*m), unb)
            ident.append(dict(rpm=1750.0, unb_re=unb.real, unb_im=unb.imag, meas=list(m),
                              k1=r.k1, k2=r.k2, rel_std1=r.rel_std1, rel_std2=r.rel_std2))
    # twin sequence: 20 commissioning + decaying K1, with a few 3600 rpm windows (gated out)
    twin = PhysicsTwin(p)
    seq = []
    decay = 0.0016
    for i in range(160):
        t_h = float(i)
        k1 = p.k_nominal[0] * np.exp(-decay * max(i - p.n_commission, 0))
        rpm = 3600.0 if i in (25, 60, 90) else 1750.0
        m = noisy(beam, k1, p.k_nominal[1], rpm, unb, rng)
        u = twin.update(rpm, *m, t_h)
        seq.append(dict(t_hours=t_h, rpm=rpm, meas=list(m),
                        out=dict(k1=u.k1, k2=u.k2, drop_pct=u.stiffness_drop_pct,
                                 rul_h=u.physics_rul_hours, commissioning=u.commissioning)))
    return dict(label="[SIMULATION] golden vectors exported by Python for the C# parity test",
                seed=seed + 1, params=as_dict(p), forward=fwd, identify=ident, sequence=seq)


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = s2_verification(SEED)
    rul = rul_summary()
    print("[SIMULATION] S2: K identification error, max over the two bearings, known unbalance")
    print(f"{'case':<15}{'rpm':>6}  {'noise':<32}{'mean%':>7}{'p95%':>7}{'max%':>7}{'pred sd%':>9}")
    for r in rows:
        print(f"{r['case']:<15}{r['rpm']:>6.0f}  {r['noise']:<32}{r['err_mean_pct']:>7.2f}"
              f"{r['err_p95_pct']:>7.2f}{r['err_max_pct']:>7.2f}{r['predicted_rel_std_pct']:>9.1f}")
    print("\n[SIMULATION] physics RUL (K1 exponential decay, 1750 rpm, 1 window/h)")
    for c in rul["checkpoints"]:
        print(f"  at {c['life_fraction']*100:.0f}% of life: mean err {c['err_mean_pct']:.1f}%  "
              f"max {c['err_max_pct']:.1f}%  (n={c['n_runs']})")
    s2 = {
        "label": "[SIMULATION] Rank 6 evidence: forward-model synthetic data + noise. NOT a rig measurement. "
                 "Same model family generates and identifies (inverse crime).",
        "spec": "S2: K identification error <= 10 % vs known-stiffness reference",
        "seed": SEED, "git_commit": _commit(),
        "command": "cd AI-engine && python -m aiengine.twin.demo_twin",
        "noise_model": dict(amp_uniform_rel=AMP_NOISE, phase_uniform_rad=PHASE_NOISE, n_trials=N_TRIALS),
        "params": as_dict(DEFAULT),
        "cases": rows, "physics_rul": rul,
    }
    (REPORTS / "twin_s2_verification.json").write_text(json.dumps(s2, indent=2))
    (REPORTS / "twin_golden.json").write_text(json.dumps(build_golden(), indent=1))
    print(f"\nwrote {REPORTS / 'twin_s2_verification.json'} and twin_golden.json")


if __name__ == "__main__":
    sys.exit(main())
