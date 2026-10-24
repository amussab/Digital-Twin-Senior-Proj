"""IS2 physics-AI RUL residual on the SYNTHETIC synth_v3 runs (UP2).  [SIMULATION] + [SYNTHETIC]

    AI-engine/.venv/Scripts/python.exe -m aiengine.up2_residual [--preds-index PATH] [--set test]
    -> reports/up2_residual_<ts>.json + .md

SPEC IS2 (Project architecture/overview/Specifications.md, row IS2/N2), quoted exactly:
    "Physics-AI residual: max relative error between the FE-identified physics RUL and the AI RUL
     <=5%, health stage 1->3"

Metric (ours, stated because the spec leaves it open): per window
    residual = |RUL_physics - RUL_AI| / RUL_AI          (both in hours, same failure definition,
                                                          same time base)
over windows of the faulted bearing of each run-to-failure test run whose TRUE health stage is 1, 2 or 3
(generator truth, DESIGN.md section 6) and where BOTH a physics RUL and an AI RUL exist.  The literal reading
of the spec ("max") is the verdict statistic; p95 / median / mean and the share of windows <= 5 % are reported
next to it.  Coverage (share of stage-1..3 windows in which the physics twin issued an RUL at all) is reported
because a twin that stays silent has no residual.

HOW THE PHYSICS RUL IS PRODUCED HERE (the synthetic runs have no displacement signal, so one is built):
  generator damage d(u) in [0,1] (monotone, d=1 at the failure window)  ->  true bearing stiffness
  K(t) = K0 * (1 - 0.25 * d(u)**m)   (m=1 nominal; 25 % drop = the failure criterion [TEAM DOC])
  -> the FE beam (aiengine/twin/fe_beam.py) forward-computes the 4 payload displacement floats (2 probes x
  amplitude/phase, float32) at the window's rpm with the commissioned unbalance -> measurement noise
  -> the REAL identifier (twin/identify.py) -> the REAL PhysicsTwin gating / baseline / BearingTracker / fit_rul.

CIRCULARITY (state this whenever the number is quoted): the stiffness trajectory is derived from the SAME
synthetic damage state that drives the AI's input features and its RUL label.  The residual therefore measures
how well two estimators that both read one latent damage variable agree; it does NOT show that a real bearing's
stiffness drop and a real vibration-based RUL agree.  It is [SIMULATION] evidence (Rank 6 at best), and an
"inverse crime" (the forward model equals the identification model) unless the plant-mismatch sweep row is used.
Because the AI RUL itself carries its own error against truth, even a perfect physics RUL cannot score below
|truth - RUL_AI| / RUL_AI; that floor is reported.

Tuning rule: the physics RUL filter (trend window length, causal EMA, identification gate) is tuned on VALIDATION
runs against the TRUTH RUL only (never against the AI predictions, never on test); the test runs are scored once.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import up2
from .config import REPORTS_DIR, SEED
from .synth_v3 import generator as gen
from .twin.fe_beam import FeBeam
from .twin.identify import identify, to_complex
from .twin.params import DEFAULT, RotorParams
from .twin.physics_rul import BearingTracker, PhysicsTwin

PRED_DIR = REPORTS_DIR / "up2_preds"
SPEC_TEXT = ("Physics-AI residual: max relative error between the FE-identified physics RUL and the AI RUL "
             "<=5%, health stage 1->3")
TARGET = 0.05
N_COMM = DEFAULT.n_commission


# --------------------------------------------------------------------------------------- configs

@dataclasses.dataclass(frozen=True)
class SimCfg:
    """What generates the displacement signal (plant side)."""
    name: str = "nominal"
    amp_noise: float = 0.05        # +/- uniform multiplicative amplitude noise (twin's documented sensor model)
    phase_noise: float = 0.05      # +/- uniform additive phase noise [rad]
    curve_m: float = 1.0           # K drop = 0.25 * d**m   (m != 1: stiffness-curve mismatch)
    e_scale: float = 1.0           # plant shaft modulus / twin's assumed (model mismatch)
    disk_scale: float = 1.0        # plant disk mass / twin's assumed (model mismatch)
    note: str = ""


@dataclasses.dataclass(frozen=True)
class FilterCfg:
    """Physics-RUL post-processing (twin side).  STOCK = the twin exactly as committed."""
    fit_points: int = DEFAULT.rul_fit_points   # trend-fit window (windows)
    ema_alpha: float = 1.0                     # 1.0 = no smoothing; <1 = causal EMA of the RUL, prior advanced by dt
    max_rel_std: float = DEFAULT.max_rel_std   # identification gate
    model: str = "exp"                         # "exp": ln K linear in t (stock); "quad": ln K quadratic in t
    weighted: bool = False                     # weight points by 1/rel_std^2 (the identifier's own confidence)
    sig_gate: bool = True                      # stock: no RUL unless slope + 1.96 se < 0
    law_a: float = 0.3                         # model "law": linear share of the assumed damage law D(u)=a u+(1-a) u^p
    law_p: float = 2.5                         # model "law": exponent of the accelerating part


STOCK = FilterCfg()
LN_FAIL = float(np.log(1.0 - DEFAULT.failure_drop))


N_T0, N_R, R_MIN, R_MAX = 24, 32, 0.25, 300.0
_R_GRID = np.geomspace(R_MIN, R_MAX, N_R)


def law_rul(t, k, sd, k0, flt: FilterCfg, min_points: int = DEFAULT.min_rul_points):
    """Degradation-law RUL (NEW code, not in the C# twin).  Physics-of-failure style: the stiffness drop follows
    q(t) = 0.25 * D((t - t0)/L), D(u) = a u + (1-a) u^p for u in [0,1] (0 before onset t0), failure (q = 0.25) at
    t0 + L.  The shape (a, p) is an ASSUMED prior (tuned on validation runs), onset t0 and failure time are
    unknown.  Grid posterior (flat in t0, log-flat in RUL, Gaussian likelihood with the identifier's own rel_std);
    returns the posterior MEDIAN of the RUL [h]."""
    t = np.asarray(t, float)[-flt.fit_points:]
    q = 1.0 - np.asarray(k, float)[-flt.fit_points:] / k0
    sd = np.maximum(np.asarray(sd, float)[-flt.fit_points:], 5e-3)
    if len(t) < min_points or np.ptp(t) <= 0:
        return None
    tn = t[-1]
    t0 = np.linspace(t[0], tn, N_T0)[:, None, None]                    # onset candidates (n_t0,1,1)
    r = _R_GRID[None, :, None]                                          # RUL candidates (1,n_r,1)
    L = (tn - t0) + r                                                   # life since onset
    u = np.clip((t[None, None, :] - t0) / L, 0.0, 1.0)
    model = DEFAULT.failure_drop * (flt.law_a * u + (1 - flt.law_a) * u ** flt.law_p)
    sse = (((q[None, None, :] - model) / sd[None, None, :]) ** 2).sum(axis=2)       # (n_t0, n_r)
    post = np.exp(-0.5 * (sse - sse.min()))
    pr = post.sum(axis=0)                                               # marginal over r
    c = np.cumsum(pr) / pr.sum()
    return float(_R_GRID[min(int(np.searchsorted(c, 0.5)), N_R - 1)])


def fit_rul_v2(t, k, sd, k0, flt: FilterCfg, min_points: int = DEFAULT.min_rul_points):
    """Alternative physics-RUL trend fit (NEW code, not in the C# twin).  Weighted polynomial fit of ln(K/K0) on
    the last `fit_points` points, level/slope/curvature taken AT NOW, extrapolated to ln(0.75).  Returns hours or None."""
    if flt.model == "law":
        return law_rul(t, k, sd, k0, flt, min_points)
    t = np.asarray(t, float)[-flt.fit_points:]
    y = np.log(np.asarray(k, float)[-flt.fit_points:] / k0)
    sd = np.asarray(sd, float)[-flt.fit_points:]
    if len(t) < min_points or np.ptp(t) <= 0:
        return None
    w = 1.0 / np.maximum(sd, 1e-3) if flt.weighted else np.ones_like(t)      # polyfit weights are 1/sigma
    tau = t - t[-1]
    deg = 2 if flt.model == "quad" else 1
    coef, cov = np.polyfit(tau, y, deg, w=w, cov="unscaled")
    if deg == 2:
        c, b, a0 = coef
    else:
        b, a0 = coef
        c = 0.0
    if flt.sig_gate:
        se_b = float(np.sqrt(max(cov[-2, -2], 0.0)))
        if b + 1.96 * se_b >= 0:
            return None
    if a0 <= LN_FAIL:
        return 0.0
    if deg == 2 and c < 0:                       # accelerating decay: first positive root of a0 + b t + c t^2 = LN_FAIL
        disc = b * b - 4 * c * (a0 - LN_FAIL)
        roots = [(-b + sg * np.sqrt(disc)) / (2 * c) for sg in (1, -1)] if disc >= 0 else []
        roots = [r for r in roots if r > 0]
        if roots:
            return float(min(roots))
    if b >= 0:
        return None
    return float((LN_FAIL - a0) / b)

SWEEP = [
    SimCfg("nominal", note="documented sensor model +/-5 % amp, +/-0.05 rad; exact FE plant; K drop linear in damage"),
    SimCfg("noise10", 0.10, 0.10, note="identification input noise doubled (+/-10 % amp, +/-0.10 rad)"),
    SimCfg("noise15", 0.15, 0.15, note="identification input noise tripled (+/-15 % amp, +/-0.15 rad)"),
    SimCfg("curve_m0.7", curve_m=0.7, note="stiffness curve mismatch: K drop = 0.25*d^0.7 (earlier, faster drop)"),
    SimCfg("curve_m1.5", curve_m=1.5, note="stiffness curve mismatch: K drop = 0.25*d^1.5 (later, steeper drop)"),
    SimCfg("plant_mismatch", e_scale=1.05, disk_scale=1.05,
           note="plant shaft E +5 % and disk mass +5 % vs the twin's assumed values (FE model-form error)"),
    SimCfg("hard_combo", 0.10, 0.10, curve_m=1.5, e_scale=1.05, disk_scale=1.05,
           note="noise10 + curve_m1.5 + plant_mismatch together"),
]


# --------------------------------------------------------------------------------------- simulation

def _plant(sim: SimCfg) -> FeBeam:
    if sim.e_scale == 1.0 and sim.disk_scale == 1.0:
        return FeBeam(DEFAULT)
    return FeBeam(dataclasses.replace(DEFAULT, E=DEFAULT.E * sim.e_scale, disk_mass=DEFAULT.disk_mass * sim.disk_scale))


def k_trajectory(damage: np.ndarray, m: float, faulted_bearing: int, k_nom=DEFAULT.k_nominal) -> np.ndarray:
    """(n, 2) true stiffness [N/m].  Faulted bearing: K0*(1-0.25 d^m).  Partner bearing: nominal."""
    d = np.clip(np.asarray(damage, float), 0.0, 1.0)
    k = np.tile(np.asarray(k_nom, float), (len(d), 1))
    k[:, faulted_bearing - 1] *= 1.0 - DEFAULT.failure_drop * d ** m
    return k


def simulate_displacements(rpm, damage, faulted_bearing, sim: SimCfg, rng, unb=None) -> np.ndarray:
    """(n, 4) float32 [a1_um, ph1, a2_um, ph2]: the payload's 4 displacement floats."""
    beam = _plant(sim)
    unb = DEFAULT.unbalance_nominal * np.exp(1j * DEFAULT.unbalance_phase_nominal) if unb is None else unb
    ks = k_trajectory(damage, sim.curve_m, faulted_bearing)
    out = np.empty((len(ks), 4))
    for i, (k, r) in enumerate(zip(ks, rpm)):
        a1, f1, a2, f2 = beam.probes(k[0], k[1], float(r), unb)
        a1 *= 1 + rng.uniform(-sim.amp_noise, sim.amp_noise)
        a2 *= 1 + rng.uniform(-sim.amp_noise, sim.amp_noise)
        f1 += rng.uniform(-sim.phase_noise, sim.phase_noise)
        f2 += rng.uniform(-sim.phase_noise, sim.phase_noise)
        out[i] = (a1, f1, a2, f2)
    return out.astype(np.float32)


def calibrate_unbalance(rpm, disp, p: RotorParams = DEFAULT) -> complex:
    twin = PhysicsTwin(p)
    meas = [to_complex(*map(float, d)) for d in disp[:N_COMM]]
    return twin.calibrate_unbalance([(float(r), m) for r, m in zip(rpm[:N_COMM], meas)])


def _identify_chunk(args) -> dict:
    rpm, disp, unb = args
    beam = FeBeam(DEFAULT)
    res = [identify(beam, float(r), to_complex(*map(float, d)), unb) for r, d in zip(rpm, disp)]
    return {"k1": [x.k1 for x in res], "k2": [x.k2 for x in res], "s1": [x.rel_std1 for x in res],
            "s2": [x.rel_std2 for x in res]}


def _join(parts: list[dict]) -> dict:
    return {k: np.concatenate([np.asarray(q[k], float) for q in parts]) for k in ("k1", "k2", "s1", "s2")}


def identify_run(rpm, disp, p: RotorParams = DEFAULT) -> dict:
    """Real unbalance calibration + real identifier over every window (serial).
    Returns arrays of the raw identification (filter-independent)."""
    unb = calibrate_unbalance(rpm, disp, p)
    return _join([_identify_chunk((rpm, disp, unb))])


def replay_physics_rul(t_h: np.ndarray, ident: dict, flt: FilterCfg) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of PhysicsTwin.update() on pre-computed identifications (so filter sweeps do not re-identify;
    parity with the real PhysicsTwin is a test).  Uses the real BearingTracker / fit_rul.
    Returns (physics_rul_hours with NaN where the twin issues none, stiffness_drop_pct)."""
    p = dataclasses.replace(DEFAULT, rul_fit_points=flt.fit_points, max_rel_std=flt.max_rel_std)
    n = len(t_h)
    rul = np.full(n, np.nan)
    drop = np.full(n, np.nan)
    trk = [BearingTracker(p), BearingTracker(p)]
    ok = np.maximum(ident["s1"], ident["s2"]) <= p.max_rel_std
    if n < N_COMM:
        return rul, drop
    idx = [i for i in range(N_COMM) if ok[i]] or list(range(N_COMM))
    trk[0].set_baseline(ident["k1"][idx])
    trk[1].set_baseline(ident["k2"][idx])
    j = idx[-1]
    trk[0].add(t_h[j], ident["k1"][j])
    trk[1].add(t_h[j], ident["k2"][j])
    ema, t_prev = None, None
    v2 = flt.model != "exp" or flt.weighted or not flt.sig_gate
    hist = [([], [], []), ([], [], [])]            # (t, k, sd) per bearing, for the v2 fit
    if v2:
        for b, (kk, ss) in enumerate(((ident["k1"], ident["s1"]), (ident["k2"], ident["s2"]))):
            hist[b][0].append(t_h[j]); hist[b][1].append(kk[j]); hist[b][2].append(ss[j])
    for i in range(N_COMM, n):
        if ok[i]:
            trk[0].add(t_h[i], ident["k1"][i])
            trk[1].add(t_h[i], ident["k2"][i])
        if v2:
            if ok[i]:
                for b, (kk, ss) in enumerate(((ident["k1"], ident["s1"]), (ident["k2"], ident["s2"]))):
                    hist[b][0].append(t_h[i]); hist[b][1].append(kk[i]); hist[b][2].append(ss[i])
            ests = [r for r in (fit_rul_v2(h[0], h[1], h[2], trk[b].k0, flt) for b, h in enumerate(hist)) if r is not None]
        else:
            ests = [e.rul_hours for e in (trk[0].rul(), trk[1].rul()) if e is not None]
        drop[i] = 100.0 * max(trk[0].drop(), trk[1].drop())
        if ests:
            raw = min(ests)
            if ema is None or flt.ema_alpha >= 1.0:
                ema = raw
            else:
                prior = max(ema - (t_h[i] - t_prev), 0.0)
                ema = prior + flt.ema_alpha * (raw - prior)
            t_prev = t_h[i]
            rul[i] = ema
    return rul, drop


# --------------------------------------------------------------------------------------- parallel driver

CHUNK = 12


CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_up2" / "up2_residual"


def cached_identifications(units: pd.DataFrame, sim: SimCfg, tag: str, workers: int | None = None) -> dict:
    """run_identifications with an on-disk cache keyed by (tag, sim, table content): identification is the slow
    part (about 0.3 s/window at 3600 rpm) and does not depend on the physics-RUL filter."""
    import hashlib
    key = hashlib.sha1((tag + json.dumps(dataclasses.asdict(sim), sort_keys=True) + str(len(units)) +
                        str(units["damage"].sum())).encode()).hexdigest()[:12]
    f = CACHE / f"{tag}_{sim.name}_{key}.npz"
    if f.exists():
        z = np.load(f)
        out: dict = {}
        for k in z.files:
            uid, name = k.rsplit("|", 1)
            out.setdefault(uid, {})[name] = z[k]
        return out
    out = run_identifications(units, sim, workers)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **{f"{u}|{n}": v for u, d in out.items() for n, v in d.items()})
    return out


def run_identifications(units: pd.DataFrame, sim: SimCfg, workers: int | None = None) -> dict:
    """units: faulted-bearing rows, stage<=3 windows only.  -> {unit_id: ident dict}.
    Displacements are simulated in the parent (deterministic per run), identification is chunked over workers
    (each window is identified independently once the unbalance is calibrated)."""
    sims, jobs, order = {}, [], []
    for uid, g in units.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        rng = np.random.default_rng([SEED, int(g["run_id"].iloc[0]), 0])
        rpm = g["rpm"].to_numpy(float)
        disp = simulate_displacements(rpm, g["damage"].to_numpy(float), int(g["faulted_bearing"].iloc[0]), sim, rng)
        unb = calibrate_unbalance(rpm, disp)
        for i in range(0, len(rpm), CHUNK):
            jobs.append((rpm[i:i + CHUNK], disp[i:i + CHUNK], unb))
            order.append(uid)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_identify_chunk, jobs, chunksize=1))
    out: dict = {}
    for uid, part in zip(order, parts):
        out.setdefault(uid, []).append(part)
    return {uid: _join(v) for uid, v in out.items()}


# --------------------------------------------------------------------------------------- data

def fault_units(df: pd.DataFrame, split: str, max_stage: int = 3) -> pd.DataFrame:
    """Faulted-bearing rows of run-to-failure runs in `split`, windows up to the end of true stage `max_stage`.
    The twin is causal, so truncating after stage 3 does not change any stage 1..3 output."""
    u = df[(df["split"] == split) & df["run_to_failure"] & (df["true_stage"] <= max_stage)]
    return u.sort_values(["unit_id", "window_index"]).reset_index(drop=True)


def load_preds(index_path: Path, name: str) -> tuple[pd.DataFrame, dict]:
    idx = json.loads(Path(index_path).read_text())
    rp = pd.read_csv(PRED_DIR / idx["files"][name]["rul"])
    return rp, idx


def latest_index(family: str = "gbm") -> Path:
    files = sorted(PRED_DIR.glob(f"up2_v3_{family}_*.index.json"))
    if not files:
        raise SystemExit(f"no up2_v3_{family}_*.index.json in {PRED_DIR}")
    return files[-1]


# --------------------------------------------------------------------------------------- scoring

def physics_table(units: pd.DataFrame, idents: dict, flt: FilterCfg) -> pd.DataFrame:
    rows = []
    for uid, g in units.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        rul, drop = replay_physics_rul(g["t_hours"].to_numpy(float), idents[uid], flt)
        rows.append(g[["unit_id", "window_index", "t_hours", "rpm_mode", "true_stage", "rul_hours", "u_true",
                       "fault_class"]].assign(rul_phys_h=rul, k_drop_pct=drop))
    return pd.concat(rows, ignore_index=True)


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "max": float(x.max()), "p95": float(np.percentile(x, 95)), "p90": float(np.percentile(x, 90)),
            "median": float(np.median(x)), "mean": float(x.mean()), "frac_le_5pct": float((x <= TARGET).mean())}


def score(tab: pd.DataFrame, preds: pd.DataFrame | None) -> dict:
    """tab: physics_table output.  preds: AI rul predictions (unit_id, window_index, rul_pred_h) or None
    (then only physics-vs-truth numbers are produced; used for validation tuning)."""
    t = tab[tab["true_stage"] <= 3].copy()
    t = t[t["rul_hours"] > 0]
    if preds is not None:
        t = t.merge(preds[["unit_id", "window_index", "rul_pred_h"]], on=["unit_id", "window_index"], how="inner")
    out = {"n_windows_stage1to3": int(len(t))}
    have = t["rul_phys_h"].notna()
    out["coverage"] = float(have.mean()) if len(t) else None
    out["coverage_by_stage"] = {str(s): float(g["rul_phys_h"].notna().mean()) for s, g in t.groupby("true_stage")}
    out["coverage_by_rpm"] = {str(int(r)): float(g["rul_phys_h"].notna().mean()) for r, g in t.groupby("rpm_mode")}
    h = t[have].copy()
    h["ape_truth"] = (h["rul_phys_h"] - h["rul_hours"]).abs() / h["rul_hours"]
    out["physics_vs_truth"] = _stats(h["ape_truth"].to_numpy())
    out["physics_vs_truth_by_stage"] = {str(s): _stats(g["ape_truth"].to_numpy()) for s, g in h.groupby("true_stage")}
    out["physics_vs_truth_by_rpm"] = {str(int(r)): _stats(g["ape_truth"].to_numpy()) for r, g in h.groupby("rpm_mode")}
    if preds is not None:
        h = h[h["rul_pred_h"] > 0].copy()
        h["res"] = (h["rul_phys_h"] - h["rul_pred_h"]).abs() / h["rul_pred_h"]
        h["floor"] = (h["rul_hours"] - h["rul_pred_h"]).abs() / h["rul_pred_h"]     # perfect-physics residual
        h["ai_ape"] = (h["rul_pred_h"] - h["rul_hours"]).abs() / h["rul_hours"]
        out["residual"] = _stats(h["res"].to_numpy())
        per_run = h.groupby("unit_id")["res"].mean()
        out["residual"]["mean_of_run_means"] = float(per_run.mean()) if len(per_run) else None
        out["residual"]["n_runs"] = int(per_run.shape[0])
        out["residual_by_stage"] = {str(s): _stats(g["res"].to_numpy()) for s, g in h.groupby("true_stage")}
        out["residual_by_rpm"] = {str(int(r)): _stats(g["res"].to_numpy()) for r, g in h.groupby("rpm_mode")}
        out["residual_by_fault"] = {k: _stats(g["res"].to_numpy()) for k, g in h.groupby("fault_class")}
        out["floor_perfect_physics"] = _stats(h["floor"].to_numpy())
        out["ai_vs_truth"] = _stats(h["ai_ape"].to_numpy())
        out["verdict_literal_max"] = "PASS" if out["residual"]["max"] <= TARGET else "FAIL"
        out["verdict_alt_p95"] = "PASS" if out["residual"]["p95"] <= TARGET else "FAIL"
    return out


# --------------------------------------------------------------------------------------- tuning (validation only)

TUNE_GRID = ([FilterCfg(fp, a, g) for fp in (16, 32, 64, 128, 200) for a in (1.0, 0.3) for g in (0.15, 0.30)] +
             [FilterCfg(fp, 1.0, g, m, True, False) for fp in (24, 48, 96) for g in (0.15, 0.60) for m in ("exp", "quad")] +
             [FilterCfg(fp, a, g, "law", True, False, la, lp) for fp in (60, 120, 200) for a in (1.0, 0.3)
              for g in (0.15, 0.60) for la, lp in ((0.3, 1.5), (0.3, 2.0), (0.3, 2.5), (0.3, 3.0))])


def tune_on_validation(val_units: pd.DataFrame, val_idents: dict) -> tuple[FilterCfg, list[dict]]:
    """Objective: mean |physics - TRUTH| / truth over validation stage-2..3 windows where an RUL is issued, among
    configs that issue an RUL in >= 40 % of stage-2..3 windows (a silent twin must not win).  No AI predictions,
    no test data."""
    rows = []
    for f in TUNE_GRID:
        tab = physics_table(val_units, val_idents, f)
        t = tab[(tab["true_stage"].isin([2, 3])) & (tab["rul_hours"] > 0)]
        cov = float(t["rul_phys_h"].notna().mean())
        ape = ((t["rul_phys_h"] - t["rul_hours"]).abs() / t["rul_hours"]).dropna()
        rows.append({**dataclasses.asdict(f), "coverage_stage2to3": cov, "mean_ape_vs_truth": float(ape.mean()),
                     "median_ape_vs_truth": float(ape.median())})
    ok = [r for r in rows if r["coverage_stage2to3"] >= 0.40]
    best = min(ok or rows, key=lambda r: r["mean_ape_vs_truth"])
    return FilterCfg(best["fit_points"], best["ema_alpha"], best["max_rel_std"], best["model"], best["weighted"],
                     best["sig_gate"], best["law_a"], best["law_p"]), rows


# --------------------------------------------------------------------------------------- main

def _git() -> dict:
    def g(*a):
        try:
            return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": g("rev-parse", "--short", "HEAD"), "dirty": bool(g("status", "--porcelain", "--", "aiengine"))}


def fmt_pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.1f} %"


def write_md(rep: dict, path: Path) -> None:
    nom = rep["nominal_tuned"]
    L = []
    L.append(f"# IS2 physics-AI RUL residual  ({rep['meta']['utc']})")
    L.append("")
    L.append("**[SIMULATION] + [SYNTHETIC].**  Not a measurement on the rig.  Spec text (quoted): "
             f"\"{SPEC_TEXT}\"")
    L.append("")
    L.append("## Verdict")
    L.append("")
    r = nom["residual"]
    L.append(f"| Reading | Value | Target | Verdict |\n|---|---|---|---|")
    L.append(f"| IS2 literal: max per-window residual, stages 1-3, nominal, tuned filter | {fmt_pct(r['max'])} | <= 5 % | **{nom['verdict_literal_max']}** |")
    L.append(f"| p95 residual (alternative reading) | {fmt_pct(r['p95'])} | <= 5 % | {nom['verdict_alt_p95']} |")
    L.append(f"| median / mean residual | {fmt_pct(r['median'])} / {fmt_pct(r['mean'])} | - | - |")
    L.append(f"| share of windows with residual <= 5 % | {fmt_pct(r['frac_le_5pct'])} | - | - |")
    L.append(f"| physics coverage (stage 1-3 windows with a physics RUL) | {fmt_pct(nom['coverage'])} | - | - |")
    L.append("")
    L.append("Metric: `residual = |RUL_physics - RUL_AI| / RUL_AI` per window, hours, faulted bearing of each "
             f"run-to-failure `{rep['meta']['set']}` run, TRUE stage in {{1,2,3}}, windows where both RULs exist "
             f"(n = {r['n']} windows, {r['n_runs']} runs).  AI predictions: `{rep['meta']['preds_index']}` "
             f"({rep['meta']['preds_provenance']}).")
    L.append("")
    L.append("### Limitations to state alongside the number")
    L.append("")
    L.append("1. **Circularity.** The stiffness trajectory K(t) fed to the FE twin is derived from the same synthetic "
             "damage curve that drives the generator's features and RUL label.  Agreement here is agreement of two "
             "estimators of one latent variable, not evidence about real bearings.")
    L.append("2. **Floor set by the AI.** The AI RUL has its own error vs truth, so even a perfect physics RUL gives a "
             f"residual of {fmt_pct(nom['floor_perfect_physics']['median'])} median / "
             f"{fmt_pct(nom['floor_perfect_physics']['max'])} max (row 'perfect physics' below).  A <= 5 % IS2 would need the AI "
             "to be within ~5 % of truth, which S7 (MAPE 15 %) does not require.")
    L.append("3. **Stage 1 has no trend.** Before onset K is constant; the trend-based physics RUL is (correctly) silent "
             "there.  Coverage by stage is in the JSON.")
    L.append("4. **3600 rpm.** K is weakly observable above the bearing mode (twin README); the identification gate drops "
             "most such windows, so 3600 rpm runs contribute little or no physics RUL (coverage by rpm below).")
    L.append("5. Identification and FE model are the twin's initial, replaceable version with ASSUMED geometry "
             "(`twin/params.py`).")
    L.append("")
    L.append("## Sweep (test runs scored once per row; tuned filter unless stated)")
    L.append("")
    L.append("| Setting | What changed | coverage | n | max | p95 | median | mean | <=5 % share | phys-vs-truth median |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rep["sweep"]:
        s = row["score"]
        rs = s["residual"]
        L.append(f"| {row['name']} | {row['note']} | {fmt_pct(s['coverage'])} | {rs['n']} | {fmt_pct(rs['max'])} | "
                 f"{fmt_pct(rs['p95'])} | {fmt_pct(rs['median'])} | {fmt_pct(rs['mean'])} | {fmt_pct(rs['frac_le_5pct'])} | "
                 f"{fmt_pct(s['physics_vs_truth'].get('median'))} |")
    s = rep["stock_nominal"]
    rs = s["residual"]
    L.append(f"| nominal, STOCK filter | twin as committed (fit 200, no EMA, gate 0.15) | {fmt_pct(s['coverage'])} | {rs['n']} | "
             f"{fmt_pct(rs['max'])} | {fmt_pct(rs['p95'])} | {fmt_pct(rs['median'])} | {fmt_pct(rs['mean'])} | "
             f"{fmt_pct(rs['frac_le_5pct'])} | {fmt_pct(s['physics_vs_truth'].get('median'))} |")
    fl = nom["floor_perfect_physics"]
    L.append(f"| perfect physics (RUL_physics := truth) | lower bound of the residual given this AI | 100.0 % | {fl['n']} | "
             f"{fmt_pct(fl['max'])} | {fmt_pct(fl['p95'])} | {fmt_pct(fl['median'])} | {fmt_pct(fl['mean'])} | "
             f"{fmt_pct(fl['frac_le_5pct'])} | 0 |")
    L.append("")
    L.append("## Nominal, by stage and rpm (residual vs AI)")
    L.append("")
    L.append("| group | n | max | p95 | median | mean | <=5 % share |\n|---|---|---|---|---|---|---|")
    for k, v in {**{f"stage {k}": v for k, v in nom["residual_by_stage"].items()},
                 **{f"{k} rpm": v for k, v in nom["residual_by_rpm"].items()}}.items():
        if v.get("n"):
            L.append(f"| {k} | {v['n']} | {fmt_pct(v['max'])} | {fmt_pct(v['p95'])} | {fmt_pct(v['median'])} | "
                     f"{fmt_pct(v['mean'])} | {fmt_pct(v['frac_le_5pct'])} |")
    L.append("")
    L.append("Coverage by rpm: " + ", ".join(f"{k} rpm {fmt_pct(v)}" for k, v in nom["coverage_by_rpm"].items())
             + ".  Coverage by stage: " + ", ".join(f"stage {k} {fmt_pct(v)}" for k, v in nom["coverage_by_stage"].items()) + ".")
    L.append("")
    L.append(f"AI vs truth on the same windows: median {fmt_pct(nom['ai_vs_truth']['median'])}, mean "
             f"{fmt_pct(nom['ai_vs_truth']['mean'])}; physics vs truth: median {fmt_pct(nom['physics_vs_truth']['median'])}, "
             f"mean {fmt_pct(nom['physics_vs_truth']['mean'])}.")
    L.append("")
    L.append("## Filter tuned on VALIDATION runs only (objective: physics vs TRUTH RUL, stages 2-3)")
    L.append("")
    t = rep["tuning"]
    L.append(f"Chosen: {t['chosen']}.  Stock: {t['stock']}.  Best 8 of {len(t['grid'])} grid points by mean APE (plus the stock row):")
    L.append("")
    L.append("| fit_points | ema_alpha | gate | model(a,p) | weighted | sig_gate | coverage(st2-3) | mean APE vs truth | median APE vs truth |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    top = sorted(t["grid"], key=lambda r_: r_["mean_ape_vs_truth"])[:8]
    stock_row = [r_ for r_ in t["grid"] if all(r_[k] == v for k, v in t["stock"].items())]
    for r_ in top + [x for x in stock_row if x not in top]:
        L.append(f"| {r_['fit_points']} | {r_['ema_alpha']} | {r_['max_rel_std']} | {r_['model']}({r_['law_a']},{r_['law_p']}) | "
                 f"{r_['weighted']} | {r_['sig_gate']} | {fmt_pct(r_['coverage_stage2to3'])} | "
                 f"{fmt_pct(r_['mean_ape_vs_truth'])} | {fmt_pct(r_['median_ape_vs_truth'])} |")
    L.append("")
    L.append("## C# port impact")
    L.append("")
    L.append(rep["csharp_note"])
    L.append("")
    L.append("## Provenance")
    L.append("")
    m = rep["meta"]
    L.append(f"- command: `{m['command']}`; commit {m['git']['commit']} (aiengine dirty: {m['git']['dirty']}); seed {m['seed']}")
    L.append(f"- data: {m['data_provenance']}; split `{m['set']}`: {m['n_units']} faulted-bearing units")
    L.append(f"- predictions: `{m['preds_index']}` ({m['preds_provenance']}); test predictions were made once by the ML agent, "
             "this script only reads them")
    L.append("- all numbers: [SIMULATION] physics twin driven by a [SYNTHETIC] damage-derived stiffness trajectory")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def csharp_note(chosen: FilterCfg) -> str:
    if chosen == STOCK:
        return "None.  The tuned configuration equals the committed twin (fit window 200, no smoothing, gate 0.15), so `src/DigitalTwin.Twin` is unchanged."
    parts = []
    if chosen.model == "law":
        parts.append(f"a new RUL estimator replacing `fit_rul`: degradation-law grid posterior `law_rul` in aiengine/up2_residual.py "
                     f"(assumed damage law D(u)={chosen.law_a}u+{1 - chosen.law_a:.1f}u^{chosen.law_p}, onset grid {N_T0} x RUL grid {N_R} "
                     f"({R_MIN}..{R_MAX} h, geometric), Gaussian likelihood with the identifier's rel_std, posterior median; "
                     f"~40 lines of C#, no new dependencies).  Identification, gating logic and golden K vectors unchanged")
    elif (chosen.model, chosen.weighted, chosen.sig_gate) != (STOCK.model, STOCK.weighted, STOCK.sig_gate):
        parts.append(f"a new trend fit replacing `fit_rul` (see `fit_rul_v2` in aiengine/up2_residual.py: model={chosen.model}, "
                     f"weighted by 1/rel_std (np.polyfit w=1/sigma)={chosen.weighted}, significance gate kept={chosen.sig_gate}; "
                     "level/slope/curvature taken at the newest point, extrapolated to ln(0.75); a weighted polynomial fit is "
                     "~25 lines of C#; the RUL interval would need re-deriving)")
    if chosen.fit_points != STOCK.fit_points:
        parts.append(f"`RotorParams.rul_fit_points` {STOCK.fit_points} -> {chosen.fit_points} (one constant)")
    if chosen.max_rel_std != STOCK.max_rel_std:
        parts.append(f"`RotorParams.max_rel_std` {STOCK.max_rel_std} -> {chosen.max_rel_std} (one constant)")
    if chosen.ema_alpha < 1.0:
        parts.append(f"a causal RUL smoothing stage after `PhysicsTwin` output: when a raw RUL exists and state exists, "
                     f"`prior = max(ema - (t_h - t_prev), 0); ema = prior + {chosen.ema_alpha}*(raw - prior)`, else "
                     "`ema = raw`; `t_prev = t_h` on every raw RUL; None stays None.  Identification, fit_rul, gating logic and "
                     "the golden vectors are otherwise unchanged")
    return ("Needed in `src/DigitalTwin.Twin` (identification and `fit_rul` unchanged): " + "; ".join(parts) +
            ".  `twin_golden.json` stays valid for K identification; the RUL golden must be regenerated if constants change.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds-index", type=Path, default=None, help="predictions index.json (default: latest up2_v3_gbm_*)")
    ap.add_argument("--set", default="test", help="prediction set key in the index (default test)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    a = ap.parse_args(argv)
    idx_path = a.preds_index or latest_index("gbm")
    preds, idx = load_preds(idx_path, a.set)
    if a.set == "test":
        df = up2.prepare(gen.DATASET)
        split = "test"
    else:
        from .up2_train import load_robust
        df = load_robust(a.set)
        split = "robust"
    val_df = up2.prepare(gen.DATASET)

    # 1) tune on validation (nominal plant), truth only
    nominal = SWEEP[0]
    vu = fault_units(val_df, "val")
    v_id = cached_identifications(vu, nominal, "val", a.workers)
    chosen, grid = tune_on_validation(vu, v_id)
    print("tuned on validation:", chosen)

    # 2) score the chosen filter once per test setting, and the stock filter at nominal
    tu = fault_units(df, split)
    units = tu["unit_id"].nunique()
    sweep, nominal_ident = [], None
    for sim in ([nominal] if a.no_sweep else SWEEP):
        ident = cached_identifications(tu, sim, a.set, a.workers)
        if sim.name == "nominal":
            nominal_ident = ident
        sweep.append({"name": sim.name, "note": sim.note, "sim": dataclasses.asdict(sim),
                      "score": score(physics_table(tu, ident, chosen), preds)})
        print(sim.name, sweep[-1]["score"]["residual"]["max"], sweep[-1]["score"]["residual"]["median"])
    stock = score(physics_table(tu, nominal_ident, STOCK), preds)
    nom = sweep[0]["score"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rep = {"meta": {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": ts, "spec": SPEC_TEXT,
                    "command": "python -m aiengine.up2_residual" + (f" --preds-index {idx_path.name}" if a.preds_index else ""),
                    "git": _git(), "seed": SEED, "set": a.set, "n_units": int(units), "preds_index": idx_path.name,
                    "preds_provenance": idx.get("meta", {}).get("provenance", "unknown") + f", predictions tag {idx.get('meta', {}).get('tag')}",
                    "data_provenance": df.attrs.get("provenance", {}).get("provenance", "[SYNTHETIC] synth_v3") +
                    f", table_hash {df.attrs.get('provenance', {}).get('table_hash', 'n/a')}",
                    "provenance": "[SIMULATION] physics twin on a [SYNTHETIC] damage-derived stiffness trajectory",
                    "metric": "|RUL_phys - RUL_AI| / RUL_AI per window, true stage 1..3, faulted bearing, both RULs present"},
           "tuning": {"chosen": dataclasses.asdict(chosen), "stock": dataclasses.asdict(STOCK), "grid": grid,
                      "objective": "validation, stage 2-3, mean |phys-truth|/truth, coverage>=40%"},
           "nominal_tuned": nom, "stock_nominal": stock, "sweep": sweep, "csharp_note": csharp_note(chosen)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"up2_residual_{ts}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    write_md(rep, a.out_dir / f"up2_residual_{ts}.md")
    print(f"IS2 literal max residual {fmt_pct(nom['residual']['max'])} -> {nom['verdict_literal_max']}; "
          f"p95 {fmt_pct(nom['residual']['p95'])}; median {fmt_pct(nom['residual']['median'])}; coverage {fmt_pct(nom['coverage'])}")
    print("wrote", a.out_dir / f"up2_residual_{ts}.md")


if __name__ == "__main__":
    main()
