"""Rig-anchored healthy -> failure trajectories in the real 32-feature node space.

[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]

x_t = exp((1-g_t) ln h_t + g_t ln f_t) per raw feature, h_t / f_t = real healthy / real fault windows (block-bootstrap, pool of the
unit's split only), g in [0,1] up to the anchor (u_anchor: where the MEASURED fault level sits), then for g>1 ("late", w in [0,1]) the
measured late-stage amplification of the real ball failure trace is applied as a piecewise-linear ln-profile (0 -> mid at w=.5 -> end at w=1).
Kurtosis / crest follow the same real endpoints in the g<=1 part (log-linear between real windows: ASSUMPTION, healthy kurtosis is high
because the signal is near the noise floor) and the MEASURED bump/fall (mid x1.5, end x1.2 of segment 0) in the late part.
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from .. import real_data as R
from ..config import BEARING_FEATURES, RIG_BEARING_ORDERS
from ..datasets.common import SCHEMA
from ..synth_v3.generator import stage_from_u
from . import constants as K
from . import pools as P
from . import scale

EXTRA_COLUMNS = ["rig_unit", "unit_seq", "u_true", "damage", "g", "late_w", "true_stage", "faulted_bearing", "split", "provenance",
                 "anchor_u", "shape_a", "shape_p", "late_scale", "unit_rpm", "pool_split",
                 "h_run", "h_order", "f_run", "f_order"]
RAW_COLUMNS = list(R.FEATURE_COLUMNS)
OUT_COLUMNS = SCHEMA + EXTRA_COLUMNS[:-4] + RAW_COLUMNS + EXTRA_COLUMNS[-4:]


def unit_seed(unit_id: str) -> int:
    return (K.MASTER_SEED * 1000003 + zlib.crc32(unit_id.encode())) % (2 ** 32)


def plan_units() -> list[dict]:
    """Deterministic unit plan (no RNG): balanced over 5 classes and over speeds. Ball only at 1800."""
    plan, k = [], 0
    for split in K.SPLITS:
        n = K.UNITS_PER_SPLIT[split]
        for i in range(n):
            cls = K.CLASSES[i % 5]
            sp = K.SPEEDS_BY_CLASS[cls]
            rpm = sp[(i // 5) % len(sp)]
            plan.append({"unit_id": f"{K.DATASET}:u{k:04d}", "split": split, "fault_class": cls, "rpm": rpm})
            k += 1
    return plan


def _profile(w: np.ndarray, mid: np.ndarray, end: np.ndarray) -> np.ndarray:
    """piecewise-linear ln-ratio at late fraction w in [0,1]: [n] -> [n,32]"""
    w = np.clip(w, 0, 1)[:, None]
    return np.where(w <= 0.5, 2 * w * mid, mid + (2 * w - 1) * (end - mid))


def _shape_ranges(split: str) -> dict:
    return K.SHAPE_TEST if split == "test" else K.SHAPE_TRAIN


def generate_unit(spec: dict, pools: P.Pools, late: dict | None = None, shape_override: dict | None = None) -> pd.DataFrame:
    """-> per-bearing rows (2 per window) with OUT_COLUMNS. Deterministic for the spec's unit_id."""
    late = late or P.load_late_profile()
    uid, split, cls, rpm = spec["unit_id"], spec["split"], spec["fault_class"], int(spec["rpm"])
    rng = np.random.default_rng(unit_seed(uid))
    healthy = cls == "healthy"
    ranges = shape_override or _shape_ranges(split)
    a = float(rng.uniform(*ranges["a"]))
    pw = float(rng.uniform(*ranges["p"]))
    late_scale = float(np.exp(rng.normal(0, K.LATE_SIGMA[split])))
    if healthy:
        n_h, n_d = int(rng.integers(K.HEALTHY_UNIT_WINDOWS[0], K.HEALTHY_UNIT_WINDOWS[1] + 1)), 0
        sev, u_anchor = 0, np.nan
    else:
        sev = pools.severity(cls)
        u_anchor = float(K.U_ANCHOR[sev] + rng.uniform(-K.U_ANCHOR_JITTER, K.U_ANCHOR_JITTER))
        n_h = int(rng.integers(K.RUNIN_WINDOWS[0], K.RUNIN_WINDOWS[1] + 1))
        life_h = scale.median_life_hours(rpm) * float(np.exp(rng.normal(0, K.LIFE_SIGMA)))
        n_d = int(np.clip(round(life_h / K.CADENCE_H), *K.LIFE_CLIP_WINDOWS))
    n = n_h + n_d
    i = np.arange(n)
    # ---- damage curve and anchor placement ----
    u = np.full(n, -1.0) if healthy else np.where(i < n_h, -1.0, (i - n_h) / max(n_d - 1, 1))
    uc = np.clip(u, 0, 1)
    d = np.where(u < 0, 0.0, a * uc + (1 - a) * uc ** pw)
    if healthy:
        g, w = np.zeros(n), np.zeros(n)
    else:
        da = a * u_anchor + (1 - a) * u_anchor ** pw
        g = np.where(u < 0, 0.0, np.minimum(d / da, 1.0))
        w = np.where(u < 0, 0.0, np.clip((d - da) / (1.0 - da), 0, 1))
    # ---- real windows ----
    h_seg = pools.healthy[rpm][split]
    Hx, h_run, h_ord = P.draw(h_seg, n, rng)
    if healthy:
        Fx, f_run, f_ord = Hx, h_run, h_ord
    else:
        Fx, f_run, f_ord = P.draw(pools.fault[(cls, rpm)][split], n, rng)
    lh, lf = np.log(np.maximum(Hx, K.LOG_FLOOR)), np.log(np.maximum(Fx, K.LOG_FLOOR))
    logx = (1 - g)[:, None] * lh + g[:, None] * lf
    if not healthy:
        logx = logx + late_scale * _profile(w, np.asarray(late["ln_mid"]), np.asarray(late["ln_end"]))
    # ---- nuisance: per-channel gain, mount offset, drift (log domain) ----
    nuis = np.zeros(32)
    gain = rng.normal(0, K.GAIN_SIGMA, 4)
    for c in range(4):
        for f in range(8):
            if f in K.AMPLITUDE_IDX:
                nuis[c * 8 + f] = gain[c] + rng.normal(0, K.MOUNT_SIGMA)
            else:
                nuis[c * 8 + f] = rng.normal(0, K.MOUNT_SIGMA_SHAPE)
    drift = np.zeros((n, 32))
    t01 = np.linspace(0, 1, n)
    for c in range(4):
        dr = np.log1p(rng.normal(0, K.DRIFT_SIGMA) * t01)
        for f in K.AMPLITUDE_IDX:
            drift[:, c * 8 + f] = dr
    X = np.exp(logx + nuis[None, :] + drift)
    X = R.apply_b2x_policy(X)                                   # B2X := B2Y (policy, after synthesis)
    # ---- labels ----
    t = i * K.CADENCE_H
    life = float(t[-1])
    true_stage = stage_from_u(u) if not healthy else np.ones(n, int)
    frames = []
    for bearing in (1, 2):
        own = (not healthy) and bearing == 1
        sl = slice(0, 16) if bearing == 1 else slice(16, 32)
        fr = pd.DataFrame(X[:, sl].astype(np.float32), columns=BEARING_FEATURES)
        fr.insert(0, "dataset", K.DATASET)
        fr.insert(1, "unit_id", f"{uid}:B{bearing}")
        fr.insert(2, "bearing_key", uid)
        fr.insert(3, "window_index", i)
        fr.insert(4, "t_hours", t)
        fr.insert(5, "rpm", float(rpm))
        fr.insert(6, "fault_class", cls if own else "healthy")
        fr.insert(7, "run_to_failure", bool(own))
        fr.insert(8, "life_hours", life if own else np.nan)
        fr.insert(9, "rul_hours", (life - t) if own else np.nan)
        fr.insert(10, "severity", f"sev{sev}_estimated" if own else "")
        fr["fs_hz"] = K.FS_HZ
        fr["band_hz"] = K.BAND_HZ
        fr["orders"] = json.dumps(RIG_BEARING_ORDERS)
        fr["rig_unit"] = uid
        fr["unit_seq"] = int(uid.split("u")[-1])
        fr["u_true"] = u if own else -1.0
        fr["damage"] = d if own else 0.0
        fr["g"] = g if own else 0.0
        fr["late_w"] = w if own else 0.0
        fr["true_stage"] = true_stage if own else 1
        fr["faulted_bearing"] = 1 if not healthy else 0
        fr["split"] = split
        fr["provenance"] = K.PROVENANCE
        fr["anchor_u"] = u_anchor
        fr["shape_a"], fr["shape_p"], fr["late_scale"] = a, pw, late_scale
        fr["unit_rpm"] = rpm
        fr["pool_split"] = split
        for j, col in enumerate(RAW_COLUMNS):
            fr[col] = X[:, j].astype(np.float32)
        fr["h_run"], fr["h_order"], fr["f_run"], fr["f_order"] = h_run, h_ord, f_run, f_ord
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)[OUT_COLUMNS]


def build(pools: P.Pools | None = None, plan: list[dict] | None = None, shape_override: dict | None = None) -> pd.DataFrame:
    pools = pools or P.Pools()
    late = P.load_late_profile()
    plan = plan or plan_units()
    df = pd.concat([generate_unit(s, pools, late, shape_override) for s in plan], ignore_index=True)
    df.attrs["provenance"] = {"provenance": K.PROVENANCE, "limits": K.LIMITS, "seed": K.MASTER_SEED, "units": len(plan),
                              "cadence_hours": K.CADENCE_H, "baseline_windows": K.BASELINE_WINDOWS, "b2x_policy": R.B2X_POLICY}
    return df


def window_matrix(unit_rows: pd.DataFrame) -> np.ndarray:
    """one row per window (bearing-1 rows) -> [n,32] raw node features (replayable)."""
    b1 = unit_rows[unit_rows["unit_id"].str.endswith(":B1")].sort_values("window_index")
    return b1[RAW_COLUMNS].to_numpy(float)
