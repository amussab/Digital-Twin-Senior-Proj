"""v3c per-window PHYSICS FEATURES from the 4 payload displacement floats ([SIMULATION] FE twin).  C#-portable.

Input per rig window: rpm, p1_amp_um, p1_phase_rad, p2_amp_um, p2_phase_rad (payload bytes 132..147), t_h (hours
since the stream started).  State per rig: the twin's commissioning buffer / unbalance / baselines K0, the twin's
BearingTracker (gated K history) and an inverse-variance EMA per bearing.  Output per bearing b in {1, 2}
(PHYS_COLUMNS, appended in this order to the 21 engineered columns of the N-HiTS-slot rows):

  phys_k_drop  = 1 - r_ema_b                         stiffness-drop fraction (smoothed)
  phys_k_est   = K0_b * r_ema_b / 1000               bearing stiffness estimate [kN/m]
  phys_rul_h   = trend RUL of bearing b if the twin's fit_rul returns one (clipped to [0, RUL_CLIP x L_pop]),
                 else the POPULATION-LIFE PRIOR  max(L_pop(rpm) - t_h, PRIOR_FLOOR x L_pop(rpm))   [hours]
  phys_conf    = lo/hi of the trend RUL interval (in [0,1], 1 = tight) when the trend RUL is used, else 0 (= prior)

Algorithm (one call per window, in order; identical to aiengine/twin/physics_rul.PhysicsTwin except where marked NEW):
  1. meas = to_complex(a1, ph1, a2, ph2)  (um -> m)
  2. commissioning (first N_COMM = 20 windows): buffer; at the 20th window calibrate the unbalance (twin
     calibrate_unbalance: weighted LS at nominal K), identify every buffered window, ok = max(rel_std1, rel_std2)
     <= MAX_REL_STD (else all), K0_b = median(K_b over ok); BearingTracker seeded with the last ok window (twin).
     NEW: EMA state seeded with every ok commissioning window, in order (step 4 rule).
     Outputs during commissioning: k_drop = 0, k_est = K_nominal/1000, rul = prior, conf = 0.
  3. later windows: (K1, K2, s1, s2) = identify(...) (twin objective, unchanged); if max(s1, s2) <= MAX_REL_STD
     add (t_h, K_b) to BearingTracker b (twin gating).
  4. NEW, every window (no gate, weights instead): w = 1 / max(s_b, S_FLOOR)^2;
     num_b = LAMBDA*num_b + w*K_b/K0_b;  den_b = LAMBDA*den_b + w;  r_ema_b = num_b/den_b.
  5. trend RUL = BearingTracker b .rul() = twin fit_rul over the last RUL_FIT_POINTS gated points (exp trend of K,
     significance gate, failure at 25 % drop) -> (rul, lo, hi) or None.  Per bearing (the twin's own output is
     the min over bearings; the AI RUL is per bearing, so the per-bearing value is used).
Constants live in PHYS_SPEC (written to model_contract_v3c.json -> physics_features).

Fast batch path (training data only): `batch_identify` minimises the SAME objective as twin/identify.py (relative
complex residual, log-parameterised K, the same 5 starts, best cost wins) with a Levenberg-Marquardt loop on the
exactly condensed FE model (Sherman-Morrison-Woodbury on the 2 bearing DOFs; the response is algebraically identical
to FeBeam.response).  Parity with the stock identifier is a test (tests/up2c).  C# keeps Identifier.Identify.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .twin.fe_beam import FeBeam
from .twin.identify import START_SCALES, to_complex
from .twin.params import DEFAULT
from .twin.physics_rul import BearingTracker

PHYS_COLUMNS = ["phys_k_drop", "phys_k_est", "phys_rul_h", "phys_conf"]
N_COMM = DEFAULT.n_commission
MAX_REL_STD = DEFAULT.max_rel_std
RUL_FIT_POINTS = 32            # twin agent's validation-tuned trend window (reports/up2_residual_20261024T072601Z.md)
LAMBDA = 1.0 - 1.0 / 24.0      # EMA forgetting factor (~24 windows = 4 h effective memory) [design choice]
S_FLOOR = 0.01                 # floor on rel_std in the EMA weight [design choice]
PRIOR_FLOOR = 0.1              # prior never below 10 % of the population life [design choice]
RUL_CLIP = 3.0                 # trend RUL clipped to 3 x population life [design choice]
RPM_SPLIT = 2675.0             # rpm below -> 1750 population, above -> 3600 population
# Population life [h] = median failure time since start of the TRAIN run-to-failure bearings (synth_v3_train3,
# 644 / 636 bearings), computed by `population_life()`; [SYNTHETIC] and to be replaced by the bearing's rated life
# (ISO 281 L10 at the duty point) on the real rig.
L_POP = {1750: 72.917, 3600: 46.5}


def phys_spec() -> dict:
    return {"columns": PHYS_COLUMNS, "n_commission": N_COMM, "max_rel_std": MAX_REL_STD,
            "rul_fit_points": RUL_FIT_POINTS, "ema_lambda": LAMBDA, "ema_s_floor": S_FLOOR,
            "prior_floor_frac": PRIOR_FLOOR, "rul_clip_frac": RUL_CLIP, "rpm_split": RPM_SPLIT,
            "population_life_hours": {str(k): v for k, v in L_POP.items()},
            "failure_drop": DEFAULT.failure_drop, "k_nominal": list(DEFAULT.k_nominal)}


def l_pop(rpm: float) -> float:
    return L_POP[1750] if rpm < RPM_SPLIT else L_POP[3600]


def prior_rul(rpm: float, t_h: float) -> float:
    L = l_pop(rpm)
    return max(L - t_h, PRIOR_FLOOR * L)


def population_life(df) -> dict:
    """Median failure time [h] since run start of run-to-failure bearings, per rpm mode (TRAIN data only)."""
    r = df[df["run_to_failure"]].sort_values("window_index").groupby("unit_id").first()
    L = r["t_hours"] + r["rul_hours"]
    return {int(k): float(v) for k, v in L.groupby(r["rpm_mode"]).median().items()}


# ------------------------------------------------------------------------------------------ condensed FE model

def condensed(beam: FeBeam, rpm: np.ndarray, batch: int = 512):
    """Per-window condensed matrices (unit unbalance): h0 (N,2), P (N,2,2), G (N,2,2), g (N,2).
    probe response(k1,k2) per unit unbalance = h0 - P (diag(1/k) + G)^-1 g   (exact SMW condensation)."""
    p = beam.p
    rpm = np.asarray(rpm, float)
    n = len(rpm)
    h0 = np.empty((n, 2), complex); P = np.empty((n, 2, 2), complex)
    G = np.empty((n, 2, 2), complex); g = np.empty((n, 2), complex)
    nd = beam.ndof
    rhs_b = np.zeros((nd, 2))
    rhs_b[beam.bdof[0], 0] = 1.0
    rhs_b[beam.bdof[1], 1] = 1.0
    for s in range(0, n, batch):
        w = 2 * np.pi * rpm[s:s + batch] / 60.0
        m = len(w)
        D = (beam.K[None] - (w ** 2)[:, None, None] * beam.M[None]).astype(complex)
        for dof, c in zip(beam.bdof, p.c_bearing):
            D[:, dof, dof] += 1j * w * c
        R = np.zeros((m, nd, 3), complex)
        R[:, beam.ddof, 0] = w ** 2
        R[:, :, 1:] = rhs_b[None]
        X = np.linalg.solve(D, R)
        h0[s:s + m] = X[:, beam.pdof, 0]
        g[s:s + m] = X[:, beam.bdof, 0]
        P[s:s + m] = X[:, beam.pdof, 1:]
        G[s:s + m] = X[:, beam.bdof, 1:]
    return h0, P, G, g


def _inv2(M):
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    det = a * d - b * c
    out = np.empty_like(M)
    out[..., 0, 0] = d / det; out[..., 0, 1] = -b / det
    out[..., 1, 0] = -c / det; out[..., 1, 1] = a / det
    return out


def _resid_jac(theta, kref, cm, unb, meas):
    """theta (N,S,2) -> r (N,S,4), J (N,S,4,2), relative complex residual exactly as twin/identify._resid."""
    h0, P, G, g = cm
    k = kref * np.exp(theta)                                   # (N,S,2)
    M = G[:, None] + np.eye(2)[None, None] * (1.0 / k)[..., None, :]   # diag(1/k) + G
    Mi = _inv2(M)
    A = np.einsum("nsij,nj->nsi", Mi, g)
    pred = unb[:, None, None] * (h0[:, None] - np.einsum("nij,nsj->nsi", P, A))
    am = np.abs(meas)[:, None]
    rc = (pred - meas[:, None]) / am
    PMi = np.einsum("nij,nsjk->nsik", P, Mi)                   # (N,S,2,2)
    dpred = -unb[:, None, None, None] * PMi * (A / k)[..., None, :]   # column i: P Mi e_i * A_i/k_i
    dr = dpred / am[..., None]
    r = np.concatenate([rc.real, rc.imag], axis=-1)
    J = np.concatenate([dr.real, dr.imag], axis=-2)
    return r, J


def batch_identify(rpm, a1, ph1, a2, ph2, unb, beam: FeBeam | None = None, iters: int = 60, cm=None) -> dict:
    """Vectorised identification over windows sharing one calibrated unbalance (complex) or per-window unbalances.
    Same objective, parameterisation, starts and uncertainty formula as twin/identify.identify."""
    beam = beam or FeBeam(DEFAULT)
    p = beam.p
    rpm = np.asarray(rpm, float)
    n = len(rpm)
    meas = np.stack([np.asarray(a1, float) * np.exp(1j * np.asarray(ph1, float)),
                     np.asarray(a2, float) * np.exp(1j * np.asarray(ph2, float))], axis=1) * 1e-6
    unb = np.broadcast_to(np.asarray(unb, complex), (n,)).copy()
    cm = cm if cm is not None else condensed(beam, rpm)
    kref = np.asarray(p.k_nominal, float)[None, None]
    S = len(START_SCALES)
    theta = np.repeat(np.log(np.asarray(START_SCALES, float))[None, :, None], 2, axis=2) * np.ones((n, 1, 1))
    mu = np.full((n, S), 1e-3)
    r, J = _resid_jac(theta, kref, cm, unb, meas)
    cost = (r ** 2).sum(-1)
    for _ in range(iters):
        JTJ = np.einsum("nski,nskj->nsij", J, J)
        JTr = np.einsum("nski,nsk->nsi", J, r)
        Ad = JTJ + mu[..., None, None] * (np.eye(2) * JTJ)  # Marquardt scaling
        Ad = Ad + 1e-30 * np.eye(2)
        det = Ad[..., 0, 0] * Ad[..., 1, 1] - Ad[..., 0, 1] * Ad[..., 1, 0]
        dx0 = -(Ad[..., 1, 1] * JTr[..., 0] - Ad[..., 0, 1] * JTr[..., 1]) / det
        dx1 = -(-Ad[..., 1, 0] * JTr[..., 0] + Ad[..., 0, 0] * JTr[..., 1]) / det
        step = np.stack([dx0, dx1], -1)
        step = np.clip(np.nan_to_num(step), -2.0, 2.0)
        th_new = np.clip(theta + step, -12.0, 12.0)
        r2, J2 = _resid_jac(th_new, kref, cm, unb, meas)
        c2 = (r2 ** 2).sum(-1)
        acc = np.isfinite(c2) & (c2 < cost)
        theta = np.where(acc[..., None], th_new, theta)
        r = np.where(acc[..., None], r2, r)
        J = np.where(acc[..., None, None], J2, J)
        cost = np.where(acc, c2, cost)
        mu = np.where(acc, np.maximum(mu * 0.3, 1e-9), np.minimum(mu * 10.0, 1e9))
    best = np.argmin(np.where(np.isfinite(cost), cost, np.inf), axis=1)
    ix = np.arange(n)
    th, Jb, cb = theta[ix, best], J[ix, best], cost[ix, best]
    JTJ = np.einsum("nki,nkj->nij", Jb, Jb)
    det = JTJ[:, 0, 0] * JTJ[:, 1, 1] - JTJ[:, 0, 1] * JTJ[:, 1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        v1 = JTJ[:, 1, 1] / det * p.rel_noise ** 2 / 2.0
        v2 = JTJ[:, 0, 0] / det * p.rel_noise ** 2 / 2.0
    s1 = np.where(det > 0, np.sqrt(np.clip(v1, 0, None)), np.inf)
    s2 = np.where(det > 0, np.sqrt(np.clip(v2, 0, None)), np.inf)
    k = np.asarray(p.k_nominal, float)[None] * np.exp(th)
    return {"k1": k[:, 0], "k2": k[:, 1], "s1": s1, "s2": s2, "cost": cb}


def calibrate_unbalance_cm(cm, a1, ph1, a2, ph2, p=DEFAULT) -> complex:
    """Twin PhysicsTwin.calibrate_unbalance on the condensed model (weighted LS at nominal K, weights 1/|meas|^2)."""
    h0, P, G, g = cm
    kn = np.asarray(p.k_nominal, float)
    M = G + np.eye(2)[None] * (1.0 / kn)[None, None, :]
    h = h0 - np.einsum("nij,nj->ni", P, np.einsum("nij,nj->ni", _inv2(M), g))
    meas = np.stack([np.asarray(a1) * np.exp(1j * np.asarray(ph1)), np.asarray(a2) * np.exp(1j * np.asarray(ph2))], 1) * 1e-6
    w2 = 1.0 / np.abs(meas) ** 2
    return complex(np.sum(w2 * np.conj(h) * meas) / np.sum(w2 * np.abs(h) ** 2))


# ------------------------------------------------------------------------------------------ feature replay

@dataclasses.dataclass
class _Ema:
    num: float = 0.0
    den: float = 0.0

    def push(self, ratio: float, s: float) -> None:
        w = 1.0 / max(s, S_FLOOR) ** 2
        if not (np.isfinite(ratio) and np.isfinite(w)):
            return
        self.num = LAMBDA * self.num + w * ratio
        self.den = LAMBDA * self.den + w

    def value(self) -> float:
        return self.num / self.den if self.den > 0 else 1.0


def replay_features(t_h, rpm, ident: dict, rul_fit_points: int = RUL_FIT_POINTS) -> np.ndarray:
    """Streaming feature computation given every window's identification (computed with the unbalance calibrated
    on the first N_COMM windows).  Returns (n, 2, 4) = per window, per bearing, PHYS_COLUMNS.  This is the
    reference the C# port must reproduce (the streaming class `PhysicsFeatures` below is the same logic)."""
    t_h = np.asarray(t_h, float); rpm = np.asarray(rpm, float)
    n = len(t_h)
    out = np.zeros((n, 2, 4))
    pr = dataclasses.replace(DEFAULT, rul_fit_points=rul_fit_points)
    kn = np.asarray(DEFAULT.k_nominal, float)
    trk = [BearingTracker(pr), BearingTracker(pr)]
    ema = [_Ema(), _Ema()]
    K = np.stack([ident["k1"], ident["k2"]], 1)
    Sd = np.stack([ident["s1"], ident["s2"]], 1)
    ok = np.maximum(Sd[:, 0], Sd[:, 1]) <= MAX_REL_STD
    for i in range(n):
        if i < N_COMM - 1 or n < N_COMM:
            for b in range(2):
                out[i, b] = (0.0, kn[b] / 1000.0, prior_rul(rpm[i], t_h[i]), 0.0)
            continue
        if i == N_COMM - 1:
            idx = [j for j in range(N_COMM) if ok[j]] or list(range(N_COMM))
            for b in range(2):
                trk[b].set_baseline(K[idx, b])
                trk[b].add(t_h[idx[-1]], K[idx[-1], b])
                for j in idx:
                    ema[b].push(K[j, b] / trk[b].k0, Sd[j, b])
        else:
            if ok[i]:
                for b in range(2):
                    trk[b].add(t_h[i], K[i, b])
            for b in range(2):
                ema[b].push(K[i, b] / trk[b].k0, Sd[i, b])
        for b in range(2):
            r = ema[b].value()
            est = trk[b].rul()
            L = l_pop(rpm[i])
            if est is None:
                rul, conf = prior_rul(rpm[i], t_h[i]), 0.0
            else:
                rul = min(max(est.rul_hours, 0.0), RUL_CLIP * L)
                conf = float(np.clip(est.lo_hours / est.hi_hours, 0.0, 1.0)) if est.hi_hours > 0 else 0.0
            out[i, b] = (1.0 - r, trk[b].k0 * r / 1000.0, rul, conf)
    return out


class PhysicsFeatures:
    """Streaming form (one call per rig window) used by the v3c engine and the golden vectors.
    Uses the STOCK twin identifier (twin/identify.identify), i.e. exactly what the C# twin runs."""

    def __init__(self):
        from .twin.physics_rul import PhysicsTwin
        self.twin = PhysicsTwin(dataclasses.replace(DEFAULT, rul_fit_points=RUL_FIT_POINTS))
        self.buf = []
        self.ident = {"k1": [], "k2": [], "s1": [], "s2": []}
        self.t, self.rpm = [], []

    def update(self, rpm, a1, ph1, a2, ph2, t_h) -> np.ndarray:
        """-> (2, 4) features for this window (bearing 1, bearing 2)."""
        from .twin.identify import identify
        self.t.append(float(t_h)); self.rpm.append(float(rpm))
        tw = self.twin
        meas = to_complex(a1, ph1, a2, ph2)
        if tw.unb is None:
            self.buf.append((float(rpm), meas))
            if len(self.buf) < N_COMM:
                kn = DEFAULT.k_nominal
                return np.array([[0.0, kn[b] / 1000.0, prior_rul(rpm, t_h), 0.0] for b in range(2)])
            tw.unb = tw.calibrate_unbalance(self.buf)
            res = [identify(tw.beam, r, m, tw.unb) for r, m in self.buf]
        else:
            res = [identify(tw.beam, float(rpm), meas, tw.unb)]
        for x in res:
            for k, v in zip(("k1", "k2", "s1", "s2"), (x.k1, x.k2, x.rel_std1, x.rel_std2)):
                self.ident[k].append(v)
        # replay is O(n) per call; fine for golden vectors / tests (the C# port keeps incremental state)
        return replay_features(self.t, self.rpm, {k: np.asarray(v) for k, v in self.ident.items()})[-1]


# ------------------------------------------------------------------------------------------ per-run driver

def simulate_run(g, seed: int) -> np.ndarray:
    """(n, 4) float32 payload displacement floats for one RUN (window order), K(t) tied to the faulted bearing's
    generator damage via the twin agent's machinery (aiengine/up2_residual.py, imported read-only): nominal SimCfg
    = documented sensor model +/-5 % amplitude, +/-0.05 rad phase (uniform), exact FE plant."""
    from .up2_residual import SimCfg, simulate_displacements
    rng = np.random.default_rng([seed, int(g["run_id"].iloc[0]), 0])
    rpm = g["rpm"].to_numpy(float)
    rtf = bool(g["run_to_failure"].iloc[0])
    fb = int(g["faulted_bearing"].iloc[0]) if rtf else 1
    dmg = g["damage"].to_numpy(float) if rtf else np.zeros(len(g))
    return simulate_displacements(rpm, dmg, fb, SimCfg(), rng)


def run_features(args) -> dict:
    """One run: (run_key, faulted-bearing table rows in window order, seed) -> displacement + identification +
    features for bearing 1 and 2."""
    key, g, seed = args
    disp = simulate_run(g, seed)
    beam = FeBeam(DEFAULT)
    rpm = g["rpm"].to_numpy(float)
    t = g["t_hours"].to_numpy(float) - float(g["t_hours"].iloc[0])
    cm = condensed(beam, rpm)
    d = disp.astype(float)
    if len(rpm) >= N_COMM:
        sl = slice(0, N_COMM)
        unb = calibrate_unbalance_cm(tuple(x[sl] for x in cm), d[sl, 0], d[sl, 1], d[sl, 2], d[sl, 3])
    else:
        unb = DEFAULT.unbalance_nominal * np.exp(1j * DEFAULT.unbalance_phase_nominal)
    ident = batch_identify(rpm, d[:, 0], d[:, 1], d[:, 2], d[:, 3], unb, beam, cm=cm)
    feats = replay_features(t, rpm, ident)
    return {"key": key, "window_index": g["window_index"].to_numpy(), "disp": disp, "feats": feats.astype(np.float32),
            "k1": ident["k1"], "k2": ident["k2"], "s1": ident["s1"], "s2": ident["s2"], "unb": unb}
