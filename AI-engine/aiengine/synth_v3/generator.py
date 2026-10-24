"""SYNTHETIC two-bearing rig generator, v3 (UP2). Emits COE-C7 feature windows (16 per bearing,
32 per rig window = the 152-byte payload's feature block) for whole runs.

ALWAYS SYNTHETIC. No seeded-fault or run-to-failure data from the team rig exists yet. This is a
physics-INFORMED stand-in (behaviour taken from the bearing-diagnostics literature, magnitudes
chosen by us), used to exercise and score the UP2 pipeline. Every number derived from it is
tagged [SYNTHETIC].

One run = one rig = two bearings (bearing 1 -> features[0:16], bearing 2 -> features[16:32],
each X's 8 then Y's 8 in COE C7 order). In a fault run ONE bearing carries the seeded defect and
runs to failure; the partner bearing is healthy and only sees attenuated transmitted vibration.
Healthy runs give two healthy bearings that never fail.

Behaviour modelled (per bearing, per axis):
  * latent damage d(u) = a*u + (1-a)*u**p over the degradation interval, u = elapsed fraction
    (a, p jittered per run), times a per-run end-of-life severity factor;
  * degradation duration scales with 1750/rpm (life in revolutions roughly fixed, ISO 281
    style) and a per-run lognormal rate factor -> run-to-run life variation that is
    observable from the trend (the "rate" knob);
  * defect-order band energy (FTF/BSF/BPFO/BPFI) grows with severity on the fault's own family,
    with leakage into the others; ball faults are cage-modulated (FTF leakage), cage faults
    leak into BSF -> deliberate class overlap;
  * band-pass RMS and envelope RMS grow with severity; a late-stage knee adds broadband energy
    that blurs the family signature;
  * kurtosis / crest rise early (impulsive spalls) then fall back as damage spreads;
  * load-zone axis asymmetry (outer race: Y loaded), speed scaling of amplitudes, multiplicative
    lognormal noise, per-axis sensor-sensitivity drift, rare transient impacts.
Labels carried with every window (generator truth): true onset, elapsed fraction u, latent
damage, health stage 1-6 (DESIGN.md section 6 rule applied to the TRUE onset: stage 1 before
onset, then 5 equal-time parts of [onset, failure] -> stages 2..6), RUL hours.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import BEARING_FEATURES, OPERATING_RPM, SEED
from ..datasets.common import SCHEMA, validate
from . import PROVENANCE

DATASET = "synth_v3"
FAULTS = ["outer_race", "inner_race", "ball", "cage"]
FAMILIES = ("ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag")
CADENCE_H = 10.0 / 60.0         # one 20-rev window kept per 10-minute snapshot (as synthetic_rig / IMS)
EXTRA_COLUMNS = ["true_onset_index", "u_true", "damage", "true_stage", "run_id", "faulted_bearing",
                 "rpm_mode", "noise_scale"]


@dataclass
class Knobs:
    """Generator difficulty knobs (all reported in every output's provenance).

    2026-10-24 (ml-engineer), decided on VALIDATION runs before any test score existed: with the wider
    first-draft ranges (shape_a 0.25-0.35, shape_p 2.0-3.0, eol_sev 0.75-1.25) the best validation RUL MAPE
    was ~17-18 %, because the end-of-life level and curve shape were nearly unobservable nuisances. They were
    narrowed so that the per-run degradation RATE is observable from the trend ("realistic-but-learnable");
    the first-draft ranges are kept as the `synth_v3_hard` robustness set so the sensitivity is reported.
    """
    noise_scale: float = 1.0          # multiplies every lognormal noise sigma
    life_ref_h: float = 50.0          # median degradation duration at 1750 rpm (hours)
    life_sigma: float = 0.35          # lognormal sigma of the per-run degradation-rate factor
    life_clip_windows: tuple = (80, 720)
    healthy_windows: tuple = (60, 200)  # pre-onset run-in per fault run
    healthy_run_windows: tuple = (250, 700)
    shape_a: tuple = (0.28, 0.32)     # linear share of the damage curve (narrowed 2026-10-24, see note)
    shape_p: tuple = (2.3, 2.7)       # power of the accelerating part (narrowed 2026-10-24)
    eol_sev: tuple = (0.85, 1.15)     # end-of-life severity factor (failure is not one fixed HI)
    transmission: tuple = (0.04, 0.10)  # fraction of the faulted bearing's growth seen by the partner
    leak: float = 0.12                # generic leakage of the fault family into the other families
    signature: float = 1.0            # multiplies every fault-family gain (lower = harder)
    drift_sigma: float = 0.04         # per-axis linear sensitivity drift over the run (fraction)
    transient_rate: float = 0.004     # probability per window of a transient impact
    rpm_modes: tuple = tuple(OPERATING_RPM)
    rpm_jitter: float = 0.002

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# Base healthy levels (per axis), [CLAUDE-SYNTHESIS] magnitudes in the same units as synthetic_rig.
BASE = {"bp_rms": 0.45, "env_rms": 0.11, "kurt": 3.0, "crest": 4.1, "fam": 0.012}
G_BP, G_ENV = 3.5, 5.5          # calibrated 2026-10-24 to XJTU-SY end-of-life env ratio (median 6.4)
# Per-class signature: own family, family gain, leakage overrides, bp/env gain, kurtosis bump
# height, axis gains (x, y), extra noise on the own family.
CLASS = {
    "outer_race": dict(fam="bpfo_mag", gain=34.0, leak={}, amp=1.0, kurt=5.5, axis=(0.6, 1.0), fam_noise=0.0),
    "inner_race": dict(fam="bpfi_mag", gain=24.0, leak={"bpfo_mag": 0.10}, amp=0.8, kurt=4.5, axis=(0.85, 0.85),
                       fam_noise=0.05),
    "ball": dict(fam="bsf_mag", gain=17.0, leak={"ftf_mag": 0.35}, amp=0.7, kurt=4.0, axis=(0.8, 0.8),
                 fam_noise=0.22),
    "cage": dict(fam="ftf_mag", gain=13.0, leak={"bsf_mag": 0.30}, amp=0.55, kurt=2.0, axis=(0.9, 0.9),
                 fam_noise=0.10),
}
NOISE = {"bp_rms": 0.06, "env_rms": 0.08, "fam": 0.12, "kurt": 0.07, "crest": 0.05}


def stage_from_u(u: np.ndarray) -> np.ndarray:
    """DESIGN.md section 6 rule on the TRUE elapsed fraction: u < 0 -> 1, else 2 + floor(5u), max 6."""
    u = np.asarray(u, float)
    st = np.minimum(2 + np.floor(np.clip(u, 0, 1) * 5).astype(int), 6)
    return np.where(u < 0, 1, st)


def _run_params(rng: np.random.Generator, fault: str, k: Knobs) -> dict:
    mode = float(rng.choice(np.asarray(k.rpm_modes, float)))
    p = {"fault": fault, "rpm_mode": mode, "faulted_bearing": int(rng.integers(1, 3))}
    if fault == "healthy":
        p["n_healthy"] = int(rng.integers(k.healthy_run_windows[0], k.healthy_run_windows[1] + 1))
        p["n_deg"] = 0
    else:
        p["n_healthy"] = int(rng.integers(k.healthy_windows[0], k.healthy_windows[1] + 1))
        life_h = k.life_ref_h * (1750.0 / mode) * float(np.exp(rng.normal(0, k.life_sigma)))
        p["n_deg"] = int(np.clip(round(life_h / CADENCE_H), *k.life_clip_windows))
    p.update(a=float(rng.uniform(*k.shape_a)), pw=float(rng.uniform(*k.shape_p)),
             eol=float(rng.uniform(*k.eol_sev)), trans=float(rng.uniform(*k.transmission)),
             bump_c=float(rng.uniform(0.35, 0.50)), bump_w=float(rng.uniform(0.18, 0.28)))
    return p


def _axis_features(rng, n, s, d, bump, knee, cls, axis_gain, healthy_machine, rpm, k: Knobs,
                   own: bool, trans: float) -> dict:
    """8 COE features for one axis. s = severity (0..~1.15), d = latent damage, own = defect is on
    this bearing (else transmitted: scaled growth, no impulsiveness)."""
    ns = k.noise_scale
    speed = (rpm / 1750.0) ** 0.8                    # vibration amplitude grows with speed
    t = np.linspace(0, 1, n)
    drift = 1.0 + rng.normal(0, k.drift_sigma) * t   # sensor sensitivity drift
    g = healthy_machine
    noisy = lambda v, sg: v * np.exp(rng.normal(0, sg * ns, n))   # noqa: E731
    out = {}
    if cls is None:
        amp, kurt_h, fam_name, fam_gain, leak_over, fam_noise = 0.0, 0.0, None, 0.0, {}, 0.0
    else:
        c = CLASS[cls]
        amp, kurt_h, fam_name = c["amp"], c["kurt"], c["fam"]
        fam_gain, leak_over, fam_noise = c["gain"] * k.signature, c["leak"], c["fam_noise"]
    w = axis_gain * (1.0 if own else trans)
    se = s * w
    out["bp_rms"] = noisy(BASE["bp_rms"] * g["bp"] * speed * drift * (1 + G_BP * amp * se) * knee, NOISE["bp_rms"])
    out["env_rms"] = noisy(BASE["env_rms"] * g["env"] * speed * drift * (1 + G_ENV * amp * se ** 1.1) * knee,
                           NOISE["env_rms"])
    imp = (kurt_h * axis_gain * bump) if own else 0.0
    late = 1.0 * d if own else 0.0
    out["bp_kurtosis"] = noisy(BASE["kurt"] * g["kurt"] + imp + late, NOISE["kurt"])
    out["bp_crest"] = noisy(BASE["crest"] * g["crest"] + 0.55 * imp + 0.5 * late, NOISE["crest"])
    for fam in FAMILIES:
        base = BASE["fam"] * g[fam] * speed * drift
        if fam_name is None:
            gain = 1.0
            sg = NOISE["fam"]
        elif fam == fam_name:
            gain = 1 + fam_gain * se ** 1.25
            sg = NOISE["fam"] + fam_noise
        else:
            gain = 1 + leak_over.get(fam, k.leak) * fam_gain * se
            sg = NOISE["fam"]
        spread = 1 + 3.0 * (knee - 1.0) if own else 1.0   # late-stage distributed damage blurs the signature
        out[fam] = noisy(base * gain * spread, sg)
    return out


def generate_run(run_id: int, fault: str, rng: np.random.Generator, k: Knobs,
                 overrides: dict | None = None) -> pd.DataFrame:
    """overrides: fix run parameters (e.g. faulted_bearing, rpm_mode, n_healthy, n_deg) for demo scenarios."""
    p = _run_params(rng, fault, k)
    p.update(overrides or {})
    nh, nd = p["n_healthy"], p["n_deg"]
    n = nh + nd
    healthy = fault == "healthy"
    i = np.arange(n)
    if healthy:
        u = np.full(n, -1.0)
    else:
        u = (i - nh) / max(nd - 1, 1)                # 0 at true onset, 1 at the failure window
        u = np.where(i < nh, -1.0, u)
    uc = np.clip(u, 0, 1)
    d = np.where(u < 0, 0.0, p["a"] * uc + (1 - p["a"]) * uc ** p["pw"])
    s = d * p["eol"]
    bump = np.exp(-(((d - p["bump_c"]) / p["bump_w"]) ** 2)) * (d > 0)
    knee = 1.0 + 0.8 * np.clip((d - 0.75) / 0.25, 0, 1) ** 2
    rpm = np.clip(p["rpm_mode"] * (1 + rng.normal(0, k.rpm_jitter, n)), p["rpm_mode"] * 0.99, p["rpm_mode"] * 1.01)
    t = i * CADENCE_H
    life = float(t[-1])
    true_stage = stage_from_u(u)
    frames = []
    for bearing in (1, 2):
        own = (not healthy) and bearing == p["faulted_bearing"]
        cls = None if healthy else fault
        g = {"bp": rng.lognormal(0, 0.2), "env": rng.lognormal(0, 0.2), "kurt": 1 + rng.normal(0, 0.03),
             "crest": 1 + rng.normal(0, 0.03), **{f: rng.lognormal(0, 0.25) for f in FAMILIES}}
        axes = {}
        for ai, axis in enumerate(("x", "y")):
            ag = CLASS[fault]["axis"][ai] if not healthy else 1.0
            ag = ag * float(np.exp(rng.normal(0, 0.05)))
            feats = _axis_features(rng, n, s, d, bump, knee if own else np.ones(n), cls, ag, g, rpm, k,
                                   own, p["trans"])
            axes[axis] = feats
        f16 = np.stack([axes[c.split("_", 1)[0]][c.split("_", 1)[1]] for c in BEARING_FEATURES], axis=1)
        # transient impacts (both axes of this bearing): one-window kurtosis/crest/bp spikes
        hit = rng.random(n) < k.transient_rate
        if hit.any():
            m = hit.sum()
            for ax0 in (0, 8):
                f16[hit, ax0 + 1] += rng.uniform(3, 10, m)
                f16[hit, ax0 + 2] += rng.uniform(1.5, 5, m)
                f16[hit, ax0 + 0] *= rng.uniform(1.1, 1.4, m)
        uid = f"{DATASET}:run{run_id:04d}:B{bearing}"
        fr = pd.DataFrame(f16.astype(np.float32), columns=BEARING_FEATURES)
        fr.insert(0, "dataset", DATASET)
        fr.insert(1, "unit_id", uid)
        fr.insert(2, "bearing_key", f"{DATASET}:run{run_id:04d}")
        fr.insert(3, "window_index", i)
        fr.insert(4, "t_hours", t)
        fr.insert(5, "rpm", rpm)
        fr.insert(6, "fault_class", fault if own else "healthy")
        fr.insert(7, "run_to_failure", bool(own))
        fr.insert(8, "life_hours", life if own else np.nan)
        fr.insert(9, "rul_hours", (life - t) if own else np.nan)
        fr.insert(10, "severity", f"eol{p['eol']:.2f}" if own else "")
        fr["fs_hz"] = 30_000.0
        fr["band_hz"] = "2000-8000"
        fr["orders"] = json.dumps({"FTF": 0.38, "BSF": 1.98, "BPFO": 3.05, "BPFI": 4.95})
        fr["true_onset_index"] = nh if own else -1
        fr["u_true"] = u if own else -1.0
        fr["damage"] = d if own else 0.0
        fr["true_stage"] = true_stage if own else 1
        fr["run_id"] = run_id
        fr["faulted_bearing"] = 0 if healthy else p["faulted_bearing"]
        fr["rpm_mode"] = p["rpm_mode"]
        fr["noise_scale"] = k.noise_scale
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)


def build(n_runs: int = 240, seed: int = SEED, knobs: Knobs | None = None, healthy_fraction: float = 0.2,
          dataset: str = DATASET, run_offset: int = 0) -> pd.DataFrame:
    """Deterministic for (n_runs, seed, knobs). Classes balanced: healthy_fraction healthy runs,
    the rest split equally over the 4 faults."""
    k = knobs or Knobs()
    rng = np.random.default_rng(seed)
    n_h = int(round(n_runs * healthy_fraction))
    assign = ["healthy"] * n_h
    while len(assign) < n_runs:
        assign.append(FAULTS[(len(assign) - n_h) % 4])
    rng.shuffle(assign)
    frames = [generate_run(run_offset + r, f, rng, k) for r, f in enumerate(assign)]
    df = pd.concat(frames, ignore_index=True)
    if dataset != DATASET:
        df["unit_id"] = df["unit_id"].str.replace(DATASET, dataset, n=1)
        df["bearing_key"] = df["bearing_key"].str.replace(DATASET, dataset, n=1)
        df["dataset"] = dataset
    extra = df[["unit_id", "window_index"] + EXTRA_COLUMNS]
    base = validate(df[SCHEMA])
    out = base.merge(extra, on=["unit_id", "window_index"], how="left", validate="one_to_one")
    out.attrs["provenance"] = provenance(n_runs, seed, k, dataset)
    return out


def provenance(n_runs: int, seed: int, k: Knobs, dataset: str = DATASET) -> dict:
    return {"provenance": PROVENANCE, "synthetic": True, "dataset": dataset, "generator": "aiengine/synth_v3/generator.py",
            "n_runs": n_runs, "seed": seed, "knobs": k.to_dict(), "cadence_hours": CADENCE_H}


def table_hash(df: pd.DataFrame) -> str:
    """Content hash of the feature block + labels (determinism checks, report provenance)."""
    cols = ["unit_id", "window_index", "rpm", "rul_hours"] + BEARING_FEATURES
    h = hashlib.sha256(pd.util.hash_pandas_object(df[cols], index=False).values.tobytes())
    return h.hexdigest()[:16]
