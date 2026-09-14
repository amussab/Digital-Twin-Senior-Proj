"""Host-side feature engineering, shared by training and inference.

DESIGN CONSTRAINT: every transform in this module must be re-implementable in
C# in a few dozen lines, because the production host is ASP.NET Core and these
transforms run there, in-process, ahead of the ONNX session. That rules out
anything stateful, fitted, or library-specific -- no scalers pickled out of
scikit-learn, no pandas-only operations inside the per-window path. What a
window needs is: its own 32 features, the machine's stored healthy baseline,
and arithmetic.

`export_onnx.py` writes the exact constants used here into the model contract
JSON so the C# side has no magic numbers of its own to drift from.

The commissioning baseline
--------------------------
Spec M5 calls for a healthy baseline capture. That capture is what gives this
module its per-channel reference levels, and it is what removes per-sensor
calibration gain and mounting variation from the model's input. One baseline
per installed machine, captured once, stored by the host, re-used for every
window after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ACCEL_PLANE,
    FEATURE_NAMES,
    N_ACCEL,
)

# Windows of healthy operation averaged into the commissioning baseline.
BASELINE_WINDOWS = 24

# Health-index shaping constants. R_REF is the amplitude-ratio-to-baseline that
# maps to HI = 1.0; it is a scale choice, not a failure threshold. The actual
# per-class failure thresholds are learned from data in rul.py.
HI_R_REF = 20.0
HI_CURVE_K = 1.0
# Weights of the three amplitude ratios fused into the health index. They sum
# to 1 and combine as a weighted geometric mean, so no single ratio spiking can
# dominate the index.
HI_WEIGHTS = {"env_rms": 0.45, "bp_rms": 0.30, "family_max": 0.25}

_EPS = 1e-9

# Columns the models actually consume. Curated rather than "all 32 raw + all
# derived": the TFT's variable-selection network costs compute per input, and
# ICS2's 200 ms/window budget is the thing being protected.
MODEL_INPUTS = [
    "hi",                  # fused health index
    "log_env_ratio",       # loudest envelope channel vs. baseline
    "log_bp_ratio",        # loudest band-pass channel vs. baseline
    "kurtosis_max",        # peak impulsiveness across channels
    "crest_max",
    "kurtosis_rise",       # kurtosis above its own baseline
    "plane_asymmetry",     # which end of the shaft is degrading
    "ftf_share",           # order-family dominance: the classification signal
    "bsf_share",
    "bpfo_share",
    "bpfi_share",
    "family_contrast",     # how sharply one family leads the other three
    "rpm_norm",
]

TARGET_HI = "hi"
TARGET_CLASS = "fault_class"


def _cols(suffix: str) -> list[str]:
    return [f"a{i + 1}_{suffix}" for i in range(N_ACCEL)]


def compute_baseline(frame: pd.DataFrame) -> dict[str, float]:
    """Commissioning baseline from a machine's first healthy windows.

    Uses the median, not the mean, so a single bad window during commissioning
    cannot shift the reference for the machine's whole service life.
    """
    head = frame.head(BASELINE_WINDOWS)
    return {name: float(head[name].median()) + _EPS for name in FEATURE_NAMES}


def health_index(ratios_env: np.ndarray, ratios_bp: np.ndarray,
                 ratios_family: np.ndarray) -> np.ndarray:
    """Fused health index in [0, ~1.2] from amplitude ratios to baseline.

    Log-compressed because bearing amplitude growth is multiplicative: without
    compression the index sits near zero for most of the bearing's life and
    then leaps, which is exactly the shape a forecaster handles worst.
    """
    r_env = np.maximum(ratios_env, 1.0)
    r_bp = np.maximum(ratios_bp, 1.0)
    r_fam = np.maximum(ratios_family, 1.0)
    fused = (
        r_env ** HI_WEIGHTS["env_rms"]
        * r_bp ** HI_WEIGHTS["bp_rms"]
        * r_fam ** HI_WEIGHTS["family_max"]
    )
    numerator = np.log1p(HI_CURVE_K * (fused - 1.0))
    denominator = np.log1p(HI_CURVE_K * (HI_R_REF - 1.0))
    return np.clip(numerator / denominator, 0.0, 1.2)


def engineer(frame: pd.DataFrame, baseline: dict[str, float]) -> pd.DataFrame:
    """Add the derived model inputs to one run's rows, in place-safe fashion."""
    out = frame.copy()

    env_cols, bp_cols = _cols("env_rms"), _cols("bp_rms")
    kurt_cols, crest_cols = _cols("bp_kurtosis"), _cols("bp_crest")

    env_ratio = out[env_cols].to_numpy() / np.array([baseline[c] for c in env_cols])
    bp_ratio = out[bp_cols].to_numpy() / np.array([baseline[c] for c in bp_cols])

    family_suffixes = ["ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag"]
    family_stack = []          # (n_windows, n_accel) per family, baseline-normalised
    for suffix in family_suffixes:
        cols = _cols(suffix)
        ratio = out[cols].to_numpy() / np.array([baseline[c] for c in cols])
        family_stack.append(ratio)
    family_arr = np.stack(family_stack, axis=0)        # (4 fam, n_win, n_accel)

    env_max = env_ratio.max(axis=1)
    bp_max = bp_ratio.max(axis=1)
    family_max = family_arr.max(axis=(0, 2))

    out["hi"] = health_index(env_max, bp_max, family_max)
    out["log_env_ratio"] = np.log1p(np.maximum(env_max - 1.0, 0.0))
    out["log_bp_ratio"] = np.log1p(np.maximum(bp_max - 1.0, 0.0))

    kurt = out[kurt_cols].to_numpy()
    crest = out[crest_cols].to_numpy()
    out["kurtosis_max"] = kurt.max(axis=1)
    out["crest_max"] = crest.max(axis=1)
    kurt_base = np.array([baseline[c] for c in kurt_cols]).max()
    out["kurtosis_rise"] = np.maximum(kurt.max(axis=1) - kurt_base, 0.0)

    # Plane asymmetry: a real bearing fault is local to one end of the shaft,
    # so the two measurement planes diverge. A speed change or a mass-unbalance
    # problem raises both planes together and leaves this near zero -- which is
    # the point of carrying it.
    plane1 = [i for i in range(N_ACCEL) if ACCEL_PLANE[i] == 1]
    plane2 = [i for i in range(N_ACCEL) if ACCEL_PLANE[i] == 2]
    e1 = env_ratio[:, plane1].mean(axis=1)
    e2 = env_ratio[:, plane2].mean(axis=1)
    out["plane_asymmetry"] = (e1 - e2) / (e1 + e2 + _EPS)

    # Order-family shares on whichever channel is loudest overall. These four
    # numbers are the physical basis for telling the fault classes apart: an
    # outer-race defect puts its energy at BPFO, an inner-race defect at BPFI,
    # and so on [COE Component 7 / bearing kinematics].
    loudest = env_ratio.argmax(axis=1)
    rows = np.arange(family_arr.shape[1])
    per_family = np.stack(
        [family_arr[f][rows, loudest] for f in range(len(family_suffixes))], axis=1
    )
    total = per_family.sum(axis=1, keepdims=True) + _EPS
    shares = per_family / total
    for idx, suffix in enumerate(family_suffixes):
        out[suffix.replace("_mag", "_share")] = shares[:, idx]

    # How decisively the leading family leads the runner-up. Near zero means
    # the four families are rising together, which is broadband noise or a
    # non-bearing problem, not a localisable defect.
    ordered = np.sort(shares, axis=1)
    out["family_contrast"] = ordered[:, -1] - ordered[:, -2]

    # Both supported modes normalised to a small symmetric range, so speed is
    # available to the model without dominating the input scale.
    out["rpm_norm"] = (out["rpm"].to_numpy() - 2675.0) / 925.0

    return out


def engineer_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Apply per-run baselining across the whole corpus.

    Baselines are computed per run because each run represents a separately
    installed machine with its own sensors, mounting and calibration.
    """
    frames, baselines = [], {}
    for run_id, run in data.groupby("run_id", sort=True):
        run = run.sort_values("window_index")
        baseline = compute_baseline(run)
        baselines[int(run_id)] = baseline
        frames.append(engineer(run, baseline))
    return pd.concat(frames, ignore_index=True), baselines
