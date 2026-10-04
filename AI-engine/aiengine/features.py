"""Per-bearing engineered model inputs (DESIGN.md section 4). C#-portable by construction.

Every transform here is arithmetic on ONE bearing-window's 16 raw COE features (X's 8 then
Y's 8, COE C7 order) plus that bearing's stored 16-value healthy baseline. No fitted objects,
no pandas in the per-window path (`engineer_arrays` is pure numpy and is what the C# host
mirrors; `model_contract.json` carries every constant).

Baselines (per unit):
- run-to-failure units (XJTU-SY, IMS, synthetic_rig): median of the first 24 windows;
- MaFaulDa: median over the speed-matched `normal` record at the same bearing position
  (nearest rpm, never the unit itself);
- rig: the commissioning capture (spec M5) -- supplied by the host at runtime.
`compute_baselines` takes a *baseline provider* so any of these policies can be plugged in.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import BEARING_FEATURES

BASELINE_WINDOWS = 24
EPS = 1e-9
# Relative floor added to each baseline value: a baseline of exactly 0 (an order band with no
# energy) would make every later ratio infinite.
BASELINE_REL_FLOOR = 1e-6

# Health-index constants (same formulation as AI-test/features.py, re-expressed per bearing).
HI_R_REF = 20.0
HI_CURVE_K = 1.0
HI_WEIGHTS = {"env_rms": 0.45, "bp_rms": 0.30, "family_max": 0.25}
HI_CLIP_MAX = 1.5

RPM_CENTER = 2675.0
RPM_SCALE = 925.0

FAMILIES = ("ftf", "bsf", "bpfo", "bpfi")
# Index of each raw feature inside one axis' 8-vector (COE C7 order).
IDX = {"bp_rms": 0, "bp_kurtosis": 1, "bp_crest": 2, "env_rms": 3,
       "ftf_mag": 4, "bsf_mag": 5, "bpfo_mag": 6, "bpfi_mag": 7}

ENGINEERED = [
    "hi",
    "log_env_ratio", "log_bp_ratio",
    "kurtosis_max", "crest_max", "kurtosis_rise",
    "axis_asymmetry",
    "ftf_share", "bsf_share", "bpfo_share", "bpfi_share", "family_contrast",
    "log_ftf_ratio", "log_bsf_ratio", "log_bpfo_ratio", "log_bpfi_ratio",
    "raw_ftf_share", "raw_bsf_share", "raw_bpfo_share", "raw_bpfi_share",
    "rpm_norm",
]

# Default model inputs (selected on validation data, see reports/model_selection_*.json).
TFT_INPUTS = list(ENGINEERED)
NHITS_TARGET = "hi"
NHITS_COVARIATES = ["log_env_ratio", "log_bp_ratio", "kurtosis_max", "kurtosis_rise",
                    "family_contrast"]


def health_index(env_ratio: np.ndarray, bp_ratio: np.ndarray, fam_ratio: np.ndarray) -> np.ndarray:
    """Weighted geometric mean of amplitude ratios (clipped >= 1), log-compressed."""
    fused = (np.maximum(env_ratio, 1.0) ** HI_WEIGHTS["env_rms"]
             * np.maximum(bp_ratio, 1.0) ** HI_WEIGHTS["bp_rms"]
             * np.maximum(fam_ratio, 1.0) ** HI_WEIGHTS["family_max"])
    hi = np.log1p(HI_CURVE_K * (fused - 1.0)) / np.log1p(HI_CURVE_K * (HI_R_REF - 1.0))
    return np.clip(hi, 0.0, HI_CLIP_MAX)


def engineer_arrays(raw16: np.ndarray, rpm: np.ndarray, baseline16: np.ndarray) -> np.ndarray:
    """(n,16) raw + (n,) rpm + (16,) baseline -> (n, len(ENGINEERED)) float64.

    The reference implementation the C# host mirrors line by line.
    """
    raw = np.asarray(raw16, dtype=np.float64).reshape(-1, 16)
    rpm = np.asarray(rpm, dtype=np.float64).reshape(-1)
    base = np.asarray(baseline16, dtype=np.float64).reshape(16)
    ax = [raw[:, 0:8], raw[:, 8:16]]          # X, Y
    bx = [base[0:8], base[8:16]]

    def ratio(name: str) -> np.ndarray:        # (n, 2): per axis
        i = IDX[name]
        return np.stack([ax[a][:, i] / (bx[a][i] + EPS) for a in (0, 1)], axis=1)

    env_r, bp_r = ratio("env_rms"), ratio("bp_rms")
    fam_r = np.stack([ratio(f + "_mag") for f in FAMILIES], axis=0)    # (4, n, 2)
    env_max, bp_max = env_r.max(axis=1), bp_r.max(axis=1)
    fam_max = fam_r.max(axis=(0, 2))

    kurt = np.stack([ax[0][:, 1], ax[1][:, 1]], axis=1)
    crest = np.stack([ax[0][:, 2], ax[1][:, 2]], axis=1)
    kurt_base = max(bx[0][1], bx[1][1])

    # Baseline-normalised family shares on the axis with the larger envelope ratio.
    loud = env_r.argmax(axis=1)
    rows = np.arange(raw.shape[0])
    per_fam = np.stack([fam_r[f][rows, loud] for f in range(4)], axis=1)   # (n, 4)
    shares = per_fam / (per_fam.sum(axis=1, keepdims=True) + EPS)
    srt = np.sort(shares, axis=1)

    # Raw (un-baselined) family shares, energy summed over both axes.
    raw_fam = np.stack([np.sqrt(ax[0][:, IDX[f + "_mag"]] ** 2 + ax[1][:, IDX[f + "_mag"]] ** 2)
                        for f in FAMILIES], axis=1)
    raw_shares = raw_fam / (raw_fam.sum(axis=1, keepdims=True) + EPS)

    e_x, e_y = env_r[:, 0], env_r[:, 1]
    cols = [
        health_index(env_max, bp_max, fam_max),
        np.log(np.maximum(env_max, EPS)),
        np.log(np.maximum(bp_max, EPS)),
        kurt.max(axis=1),
        crest.max(axis=1),
        np.maximum(kurt.max(axis=1) - kurt_base, 0.0),
        (e_x - e_y) / (e_x + e_y + EPS),
        shares[:, 0], shares[:, 1], shares[:, 2], shares[:, 3],
        srt[:, -1] - srt[:, -2],
        *[np.log(np.maximum(fam_r[f].max(axis=1), EPS)) for f in range(4)],
        raw_shares[:, 0], raw_shares[:, 1], raw_shares[:, 2], raw_shares[:, 3],
        (rpm - RPM_CENTER) / RPM_SCALE,
    ]
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------- baselines

def baseline_first_windows(unit: pd.DataFrame, n: int = BASELINE_WINDOWS) -> np.ndarray:
    head = unit.sort_values("window_index").head(n)
    return _floor(np.median(head[BEARING_FEATURES].to_numpy(dtype=np.float64), axis=0))


def _floor(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    return b + BASELINE_REL_FLOOR * (np.abs(b).max() + EPS)


def mafaulda_position(unit_id: str) -> str:
    return unit_id.rsplit(":", 1)[-1]


def is_mafaulda_normal(unit_id: str) -> bool:
    body = unit_id.split(":", 1)[1] if ":" in unit_id else unit_id
    return body.startswith("normal")


def make_mafaulda_provider(table: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
    """Speed-matched `normal` record at the same bearing position (nearest median rpm).

    A normal unit is never its own baseline (that would make it trivially perfect); it uses
    the nearest *other* normal record. Baselines play the role of commissioning data, which
    is always available on the deployed machine, so the normal pool is not split-restricted;
    no label information flows through a baseline.
    """
    maf = table[table["dataset"] == "mafaulda"]
    normals = {}
    for uid, g in maf.groupby("unit_id"):
        if is_mafaulda_normal(uid):
            normals[uid] = (mafaulda_position(uid), float(g["rpm"].median()),
                            np.median(g[BEARING_FEATURES].to_numpy(dtype=np.float64), axis=0))

    def provider(unit: pd.DataFrame) -> np.ndarray:
        uid = unit["unit_id"].iloc[0]
        pos, rpm = mafaulda_position(uid), float(unit["rpm"].median())
        cands = [(abs(r - rpm), b) for k, (p, r, b) in normals.items() if p == pos and k != uid]
        if not cands:
            cands = [(abs(r - rpm), b) for k, (p, r, b) in normals.items() if k != uid]
        if not cands:
            return baseline_first_windows(unit)
        return _floor(min(cands, key=lambda c: c[0])[1])

    return provider


def default_provider(table: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
    maf = make_mafaulda_provider(table) if (table["dataset"] == "mafaulda").any() else None

    def provider(unit: pd.DataFrame) -> np.ndarray:
        if unit["dataset"].iloc[0] == "mafaulda" and maf is not None:
            return maf(unit)
        return baseline_first_windows(unit)

    return provider


def compute_baselines(table: pd.DataFrame,
                      provider: Callable[[pd.DataFrame], np.ndarray] | None = None
                      ) -> dict[str, np.ndarray]:
    provider = provider or default_provider(table)
    return {uid: provider(g) for uid, g in table.groupby("unit_id", sort=True)}


def engineer_table(table: pd.DataFrame, baselines: dict[str, np.ndarray] | None = None,
                   provider=None) -> pd.DataFrame:
    """Add ENGINEERED columns to a window table (all units)."""
    table = table.sort_values(["unit_id", "window_index"]).reset_index(drop=True)
    baselines = baselines or compute_baselines(table, provider)
    out = np.zeros((len(table), len(ENGINEERED)))
    raw = table[BEARING_FEATURES].to_numpy(dtype=np.float64)
    rpm = table["rpm"].to_numpy(dtype=np.float64)
    for uid, idx in table.groupby("unit_id", sort=False).indices.items():
        out[idx] = engineer_arrays(raw[idx], rpm[idx], baselines[uid])
    eng = pd.DataFrame(out.astype(np.float32), columns=ENGINEERED, index=table.index)
    return pd.concat([table, eng], axis=1)
