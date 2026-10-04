"""SYNTHETIC rig stand-in, ported from AI-test/synth.py, emitted in the common window schema.

ALWAYS SYNTHETIC. No fault-seeded or run-to-failure data from the team's rig exists yet. This
fabricates COE-C7-shaped feature vectors following published bearing-degradation behaviour so
the pipeline (features -> labels -> splits -> models -> export -> engine) can be exercised
end to end. Nothing produced here may be quoted as a measurement.

One synthetic run = one rig = TWO bearings (rig plane 1 = bearing 1 = accelerometers X,Y;
plane 2 = bearing 2). One of them carries the seeded fault (run-to-failure unit); the other is
healthy and sees only transmitted vibration (unit with run_to_failure=False). Healthy runs
give two healthy units. Orders are the rig's [TEAM DOC].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BEARING_FEATURES, FEATURE_NAMES_PER_ACCEL, OPERATING_RPM, SEED
from ..geometry import BEARING_ORDERS
from .common import UnitMeta, pipeline, row, validate

DATASET = "synthetic_rig"
ORDERS = BEARING_ORDERS["rig"]
FAULTS = ["outer_race", "inner_race", "ball", "cage"]
FAMILY_OF = {"outer_race": "bpfo_mag", "inner_race": "bpfi_mag", "ball": "bsf_mag", "cage": "ftf_mag"}

BASE = {"bp_rms": 0.45, "env_rms": 0.11, "kurt": 3.0, "crest": 4.1, "fam": 0.012}
EOL_BP, EOL_ENV, EOL_FAM, LEAK = 6.5, 11.0, 42.0, 0.16
OFF_PLANE = 0.34
SEVERITY = {"outer_race": 1.0, "inner_race": 0.82, "ball": 0.66, "cage": 0.40}
FAILURE_DAMAGE = 0.95


def _damage(n, rng):
    u = np.linspace(0, 1, n)
    a, k = rng.uniform(0.12, 0.34), rng.uniform(3.0, 7.0)
    return FAILURE_DAMAGE * (a * u + (1 - a) * u ** k)


def _bump(d, rng):
    c, w = rng.uniform(0.38, 0.52), rng.uniform(0.22, 0.32)
    return np.clip(np.exp(-(((d - c) / w) ** 2)) + 0.25 * np.clip((d - 0.6) / 0.35, 0, 1), 0, 1.4)


def generate_run(run_id: int, fault: str, rng: np.random.Generator,
                 min_windows=220, max_windows=620, hours_per_window=10 / 60) -> pd.DataFrame:
    n = int(rng.integers(min_windows, max_windows + 1))
    healthy = fault == "healthy"
    damage = np.linspace(0, rng.uniform(0.01, 0.045), n) if healthy else _damage(n, rng)
    sev = damage / FAILURE_DAMAGE * (1.0 if healthy else SEVERITY[fault])
    bump = _bump(damage, rng)
    knee = 1.0 + 0.9 * np.clip((damage - 0.72) / 0.23, 0, 1) ** 2
    mode = float(rng.choice(OPERATING_RPM))
    rpm = np.clip(mode * (1 + rng.normal(0, 0.0025, n)), mode * 0.99, mode * 1.01)
    faulted_bearing = int(rng.integers(1, 3))
    t = np.arange(n) * hours_per_window
    life = float(t[-1])
    cfg = pipeline(30_000.0, ORDERS)

    rows = []
    for bearing in (1, 2):
        on = (not healthy) and bearing == faulted_bearing
        trans = 1.0 if (healthy or on) else OFF_PLANE
        s = sev * trans
        feats = {}
        for axis in ("x", "y"):
            gain = rng.normal(1.0, 0.055)
            noisy = lambda v, sg: v * np.exp(rng.normal(0, sg, n))   # noqa: E731
            imp = 0.0 if healthy else 1.0
            feats[f"{axis}_bp_rms"] = noisy(BASE["bp_rms"] * gain * (1 + (EOL_BP - 1) * s) * knee, 0.075)
            feats[f"{axis}_env_rms"] = noisy(BASE["env_rms"] * gain * (1 + (EOL_ENV - 1) * s ** 1.15) * knee, 0.085)
            feats[f"{axis}_bp_kurtosis"] = noisy(BASE["kurt"] + 9.5 * bump * trans * imp, 0.06)
            feats[f"{axis}_bp_crest"] = noisy(BASE["crest"] + 5.2 * bump * trans * imp, 0.055)
            for fam in ("ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag"):
                if healthy or fam != FAMILY_OF[fault]:
                    g = 1 + LEAK * (EOL_FAM - 1) * s
                else:
                    g = 1 + (EOL_FAM - 1) * s ** 1.25
                feats[f"{axis}_{fam}"] = noisy(BASE["fam"] * gain * g, 0.11)
        f16 = np.stack([feats[c] for c in BEARING_FEATURES], axis=1)
        meta = UnitMeta(
            dataset=DATASET, unit_id=f"{DATASET}:run{run_id:03d}:B{bearing}",
            bearing_key=f"{DATASET}:run{run_id:03d}",      # both bearings of a run split together
            fault_class=fault if on else "healthy",
            run_to_failure=bool(on), life_hours=life if on else float("nan"),
        )
        for i in range(n):
            rows.append(row(meta, i, float(t[i]), float(rpm[i]), f16[i], cfg))
    return pd.DataFrame(rows)


def build(n_runs: int = 48, seed: int = SEED, healthy_fraction: float = 0.18,
          min_windows: int = 220, max_windows: int = 620) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_h = max(1, int(round(n_runs * healthy_fraction)))
    assign = ["healthy"] * n_h
    while len(assign) < n_runs:
        assign.append(FAULTS[len(assign) % 4])
    rng.shuffle(assign)
    frames = [generate_run(i, f, rng, min_windows, max_windows) for i, f in enumerate(assign)]
    return validate(pd.concat(frames, ignore_index=True))


assert FEATURE_NAMES_PER_ACCEL[4:] == ["ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag"]
