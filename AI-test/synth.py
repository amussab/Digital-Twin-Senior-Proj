"""Synthetic run-to-failure generator.

WHAT THIS IS, PLAINLY: no fault-seeded or run-to-failure data from the team's
own rig exists yet -- that is an ME-owned dependency still pending. This module
fabricates feature vectors with the *shape and ordering* COE's Component 7
produces, following published bearing-degradation behaviour, so the model
pipeline, the latency path, the ONNX export and the spec-check harness can all
be exercised end to end before real data lands.

Numbers produced here are synthetic by construction. Every report this
testbench writes stamps that on its face. Nothing generated here may be quoted
as a measured result.

Degradation behaviour modelled
------------------------------
1.  Miner's cumulative damage D(t) accumulates with an incubation phase then a
    propagation phase, reaching D = 0.95 at end of life [spec M6].
2.  Band-pass and envelope RMS grow monotonically with D, with a late-life
    knee.
3.  Kurtosis and crest factor rise as discrete impacts appear, peak in
    mid-life, then fall back as the defect spreads and the signal becomes
    broadband again. This non-monotonic bump is the well-documented behaviour
    that makes RMS-only trending insufficient, and it is why the feature set
    carries all four statistics rather than amplitude alone.
    [Randall & Antoni, MSSP 2011, doi:10.1016/j.ymssp.2010.07.017]
4.  The faulted bearing's own order family (BPFO / BPFI / BSF / FTF) grows far
    faster than the other three, which pick up only leakage and sidebands.
    This is the signal the classifier has to separate.
5.  Cage (FTF) faults are generated deliberately weak and slow. Cage defects
    are genuinely the hardest of the four to detect, and a testbench that made
    all four classes equally easy would report a macro-F1 that means nothing.
6.  Per-run, per-channel calibration gains vary. Without this a model can
    memorise absolute levels instead of learning degradation shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ACCEL_PLANE,
    FAULT_CLASSES,
    FAULT_TO_ORDER_FAMILY,
    FEATURE_NAMES_PER_ACCEL,
    N_ACCEL,
    OPERATING_RPM,
    RPM_TOLERANCE,
    STAGE_DAMAGE_EDGES,
    FAILURE_DAMAGE,
    SynthConfig,
)

# Healthy-state reference levels, in the units Component 2 calibrates to
# (m/s^2 for acceleration statistics, dimensionless for kurtosis/crest, and
# arbitrary-but-consistent spectral magnitude for the order families).
BASE_BP_RMS = 0.45          # m/s^2 band-pass RMS on a healthy bearing
BASE_ENV_RMS = 0.11         # m/s^2 envelope RMS
BASE_KURTOSIS = 3.0         # Gaussian reference
BASE_CREST = 4.1
BASE_FAMILY_MAG = 0.012     # order-family noise floor

# Growth from healthy to end of life, as a multiple of the healthy level.
EOL_BP_RMS_RATIO = 6.5
EOL_ENV_RMS_RATIO = 11.0
EOL_FAMILY_RATIO = 42.0     # the faulted family's own growth
CROSS_FAMILY_LEAKAGE = 0.16  # how much the other three families follow along

# Vibration reaching the far measurement plane through the shaft and housing.
OFF_PLANE_TRANSMISSION = 0.34

# Class-specific severity scaling. Cage faults stay weak; outer-race faults are
# the loudest and earliest.
CLASS_SEVERITY = {
    "outer_race": 1.00,
    "inner_race": 0.82,
    "ball": 0.66,
    "cage": 0.40,
}


def stage_from_damage(damage: np.ndarray) -> np.ndarray:
    """Map Miner's D onto the six health stages [spec M6]."""
    return np.searchsorted(STAGE_DAMAGE_EDGES, damage, side="right") + 1


def _damage_curve(n: int, rng: np.random.Generator) -> np.ndarray:
    """Miner's D from install to D = 0.95, incubation then propagation."""
    u = np.linspace(0.0, 1.0, n)
    alpha = rng.uniform(0.12, 0.34)       # linear share (steady wear)
    k = rng.uniform(3.0, 7.0)             # propagation exponent
    shape = alpha * u + (1.0 - alpha) * u**k
    return FAILURE_DAMAGE * shape


def _impulsiveness_bump(damage: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rise-peak-decay profile for kurtosis and crest factor, in [0, 1]."""
    centre = rng.uniform(0.38, 0.52)
    width = rng.uniform(0.22, 0.32)
    bump = np.exp(-(((damage - centre) / width) ** 2))
    # Late life adds a broadband floor: the signal gets rougher even as the
    # discrete impacts stop dominating.
    floor = 0.25 * np.clip((damage - 0.6) / 0.35, 0.0, 1.0)
    return np.clip(bump + floor, 0.0, 1.4)


def _late_knee(damage: np.ndarray) -> np.ndarray:
    """Extra amplitude growth concentrated in the final stages."""
    return 1.0 + 0.9 * np.clip((damage - 0.72) / 0.23, 0.0, 1.0) ** 2


def generate_run(
    run_id: int,
    fault_class: str,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One bearing's life history, one row per stored 20-revolution window."""
    n = int(rng.integers(cfg.min_windows, cfg.max_windows + 1))
    healthy = fault_class == "healthy"

    if healthy:
        # A healthy bearing still wears. It just never reaches the knee: damage
        # creeps up but stays inside stage 1, below the first stage boundary,
        # so healthy runs never contaminate the stage-error metric.
        damage = np.linspace(0.0, rng.uniform(0.010, 0.045), n)
    else:
        damage = _damage_curve(n, rng)

    stage = stage_from_damage(damage)
    severity = damage / FAILURE_DAMAGE                     # normalised 0..1
    severity = severity * (1.0 if healthy else CLASS_SEVERITY[fault_class])
    bump = _impulsiveness_bump(damage, rng)
    knee = _late_knee(damage)

    # Operating mode is fixed for a run; speed jitters inside the +/-1% band.
    mode = float(rng.choice(OPERATING_RPM))
    rpm = mode * (1.0 + rng.normal(0.0, 0.0025, n))
    rpm = np.clip(rpm, mode * (1 - RPM_TOLERANCE), mode * (1 + RPM_TOLERANCE))

    faulted_plane = int(rng.integers(1, 3))                # 1 or 2
    family = FAULT_TO_ORDER_FAMILY[fault_class]
    family_index = {"FTF": 4, "BSF": 5, "BPFO": 6, "BPFI": 7}

    columns: dict[str, np.ndarray] = {}

    for accel in range(N_ACCEL):
        # Per-run, per-channel calibration spread and mounting variation.
        gain = rng.normal(1.0, 0.055)
        transmission = (
            1.0 if (healthy or ACCEL_PLANE[accel] == faulted_plane)
            else OFF_PLANE_TRANSMISSION
        )
        s = severity * transmission
        prefix = f"a{accel + 1}_"

        def noisy(values: np.ndarray, sigma: float) -> np.ndarray:
            """Multiplicative lognormal measurement noise."""
            return values * np.exp(rng.normal(0.0, sigma, n))

        bp_rms = BASE_BP_RMS * gain * (1.0 + (EOL_BP_RMS_RATIO - 1.0) * s) * knee
        env_rms = BASE_ENV_RMS * gain * (1.0 + (EOL_ENV_RMS_RATIO - 1.0) * s**1.15) * knee

        columns[prefix + "bp_rms"] = noisy(bp_rms, 0.075)
        columns[prefix + "env_rms"] = noisy(env_rms, 0.085)
        columns[prefix + "bp_kurtosis"] = noisy(
            BASE_KURTOSIS + 9.5 * bump * transmission * (0.0 if healthy else 1.0),
            0.06,
        )
        columns[prefix + "bp_crest"] = noisy(
            BASE_CREST + 5.2 * bump * transmission * (0.0 if healthy else 1.0),
            0.055,
        )

        for fam, local_idx in family_index.items():
            name = prefix + FEATURE_NAMES_PER_ACCEL[local_idx]
            if healthy or fam != family:
                # Non-faulted families follow the overall level weakly.
                growth = 1.0 + CROSS_FAMILY_LEAKAGE * (EOL_FAMILY_RATIO - 1.0) * s
            else:
                growth = 1.0 + (EOL_FAMILY_RATIO - 1.0) * s**1.25
            columns[name] = noisy(BASE_FAMILY_MAG * gain * growth, 0.11)

    # 1x shaft displacement, the FE digital twin's input [COE Component 8].
    # Grows mildly with damage as clearance opens up; phase drifts slowly.
    disp = {}
    for plane in (1, 2):
        on_fault = (not healthy) and plane == faulted_plane
        base_amp = rng.uniform(6.0, 14.0)
        growth = 1.0 + (1.6 if on_fault else 0.35) * severity
        disp[f"p{plane}_amp_um"] = base_amp * growth * np.exp(rng.normal(0, 0.05, n))
        phase0 = rng.uniform(-np.pi, np.pi)
        drift = np.cumsum(rng.normal(0.0, 0.012, n)) + 0.55 * severity
        disp[f"p{plane}_phase_rad"] = np.arctan2(
            np.sin(phase0 + drift), np.cos(phase0 + drift)
        )

    window_index = np.arange(n, dtype=np.int64)
    hours_elapsed = window_index * cfg.hours_per_window
    # True RUL: hours until D reaches 0.95. Right-censored for healthy runs,
    # which never reach it -- left as NaN rather than filled with a guess.
    rul_hours = (n - 1 - window_index) * cfg.hours_per_window
    if healthy:
        rul_hours = np.full(n, np.nan)

    frame = pd.DataFrame(
        {
            "run_id": run_id,
            "window_index": window_index,
            "hours_elapsed": hours_elapsed,
            "rpm": rpm,
            # String, not int: pytorch-forecasting rejects numeric columns used
            # as categoricals. It is a mode label, not a quantity -- the actual
            # measured speed travels in `rpm`.
            "operating_mode": f"{int(mode)}",
            "fault_class": fault_class,
            "faulted_plane": 0 if healthy else faulted_plane,
            "damage": damage,
            "health_stage": stage,
            "rul_hours": rul_hours,
            **columns,
            **disp,
        }
    )
    # t20 timestamp, milliseconds since node startup, wrapping at uint32.
    frame["t20_ms"] = (
        (hours_elapsed * 3_600_000.0).astype(np.int64) % (2**32)
    ).astype(np.int64)
    return frame


def generate_dataset(cfg: SynthConfig) -> pd.DataFrame:
    """Build the full synthetic corpus."""
    rng = np.random.default_rng(cfg.seed)
    faulted_classes = [c for c in FAULT_CLASSES if c != "healthy"]

    n_healthy = max(1, int(round(cfg.n_runs * cfg.healthy_fraction)))
    assignments = ["healthy"] * n_healthy
    while len(assignments) < cfg.n_runs:
        assignments.append(faulted_classes[len(assignments) % len(faulted_classes)])
    rng.shuffle(assignments)

    runs = [
        generate_run(run_id, fault_class, cfg, rng)
        for run_id, fault_class in enumerate(assignments)
    ]
    data = pd.concat(runs, ignore_index=True)
    data["fault_class"] = pd.Categorical(data["fault_class"], categories=FAULT_CLASSES)
    return data
