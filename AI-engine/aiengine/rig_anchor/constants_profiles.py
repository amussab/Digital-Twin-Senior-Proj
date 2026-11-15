"""Optional generator PROFILES for rig_anchor (v7 research). The default (profile=None) is exactly constants.py: nothing here changes it.

[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape + life variance assumed]
A profile overrides only: life_sigma (run-to-run lognormal life/rate factor, the one assumed quantity nobody measured),
the train/val shape ranges, and optionally the test shape ranges / late sigma. Everything else stays as constants.py.
"""
from __future__ import annotations

from . import constants as K

HARD = {"life_sigma": K.LIFE_SIGMA, "shape_train": K.SHAPE_TRAIN, "shape_test": K.SHAPE_TEST}      # == default behaviour

NARROW = {"a": (0.28, 0.32), "p": (2.3, 2.7)}      # synth_v3 narrowed knobs (UP2 precedent)

PROFILES = {
    "hard": HARD,
    "ls025": {"life_sigma": 0.25},
    "ls015": {"life_sigma": 0.15},
    "ls010": {"life_sigma": 0.10},
    "ls035_narrow": {"life_sigma": 0.35, "shape_train": {"a": (0.28, 0.32), "p": (2.3, 2.7)}, "shape_test": {"a": (0.28, 0.32), "p": (2.3, 2.7)}},
    "ls025_narrow": {"life_sigma": 0.25, "shape_train": {"a": (0.28, 0.32), "p": (2.3, 2.7)}, "shape_test": {"a": (0.28, 0.32), "p": (2.3, 2.7)}},
    "ls015_narrow": {"life_sigma": 0.15, "shape_train": {"a": (0.28, 0.32), "p": (2.3, 2.7)}, "shape_test": {"a": (0.28, 0.32), "p": (2.3, 2.7)}},
    # v7 sweep: life variance + the other per-unit nuisances that are also unobservable from one rate/progress reading
    "ls010_narrow": {"life_sigma": 0.10, "shape_train": NARROW, "shape_test": NARROW},
    "ls010_narrow_l05": {"life_sigma": 0.10, "shape_train": NARROW, "shape_test": NARROW, "late_sigma": 0.05},
    "ls010_narrow_l05_j01": {"life_sigma": 0.10, "shape_train": NARROW, "shape_test": NARROW, "late_sigma": 0.05, "u_jitter": 0.01},
    "N_ls035": {"life_sigma": 0.35, "shape_train": NARROW, "shape_test": NARROW, "late_sigma": 0.05, "u_jitter": 0.01},
    "N_ls025": {"life_sigma": 0.25, "shape_train": NARROW, "shape_test": NARROW, "late_sigma": 0.05, "u_jitter": 0.01},
    "N_ls015": {"life_sigma": 0.15, "shape_train": NARROW, "shape_test": NARROW, "late_sigma": 0.05, "u_jitter": 0.01},
    # PRE-DECLARED learnable profile used by v7_node (selected on VALIDATION; rule + evidence in reports/s7_research_*.md)
    "L": {"life_sigma": 0.25, "shape_train": NARROW, "shape_test": {"a": (0.20, 0.40), "p": (2.0, 3.0)}, "late_sigma": 0.05, "u_jitter": 0.01},
    "floor_probe": {"life_sigma": 0.03, "shape_train": {"a": (0.30, 0.30), "p": (2.5, 2.5)}, "shape_test": {"a": (0.30, 0.30), "p": (2.5, 2.5)},
                    "late_sigma": 0.02, "u_jitter": 0.0},
}


def get(name: str | None) -> dict | None:
    if name in (None, "hard"):
        return None
    return PROFILES[name]
