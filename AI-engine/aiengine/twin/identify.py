"""Inverse problem: identify bearing stiffness K1, K2 from measured probe 1x amp/phase."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .fe_beam import FeBeam


@dataclass
class IdentResult:
    k1: float
    k2: float
    rel_std1: float      # 1-sigma relative uncertainty of K1 (linearised, noise model rel_noise)
    rel_std2: float
    cost: float          # weighted residual sum of squares at optimum
    converged: bool


def to_complex(a1_um, ph1, a2_um, ph2):
    return np.array([a1_um * np.exp(1j * ph1), a2_um * np.exp(1j * ph2)]) * 1e-6


def _resid(theta, beam, rpm, unb, meas, k_ref):
    pred = beam.response(k_ref[0] * np.exp(theta[0]), k_ref[1] * np.exp(theta[1]), rpm, unb)
    r = (pred - meas) / np.abs(meas)          # relative complex residual, avoids phase wrap
    return np.concatenate([r.real, r.imag])


START_SCALES = (1.0, 0.5, 2.0, 0.25, 4.0)    # multi-start, mirrored in the C# port


def identify(beam: FeBeam, rpm, meas, unb, k_start=None, rel_noise=None) -> IdentResult:
    """meas: complex (2,) probe displacement [m]; unb: complex unbalance [kg m].

    Log-parameterised: K = K_ref * exp(theta). Multi-start (resonances create local minima);
    the first start reaching a negligible residual wins, else the best of all.
    """
    p = beam.p
    k_ref = np.array(k_start if k_start is not None else p.k_nominal, float)
    sig = p.rel_noise if rel_noise is None else rel_noise
    best = None
    for sc in START_SCALES:
        s = float(np.log(sc))
        sol = least_squares(_resid, [s, s], args=(beam, rpm, unb, meas, k_ref),
                            method="trf", jac="3-point", xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-14:
            break
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * sig ** 2 / 2.0   # each real/imag part has sigma ~ sig/sqrt(2)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std = np.array([np.inf, np.inf])
    return IdentResult(float(k_ref[0] * np.exp(best.x[0])), float(k_ref[1] * np.exp(best.x[1])),
                       float(std[0]), float(std[1]), float(2 * best.cost), bool(best.success))
