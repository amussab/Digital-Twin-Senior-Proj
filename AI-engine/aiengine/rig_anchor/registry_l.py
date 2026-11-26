"""Registered unit splits of the 'learnable' (L) profile units (v7): section rig_anchor_L in splits.json, written BEFORE any fit.

[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape + life variance assumed]
Plan = deterministic enumeration (no RNG), balanced over the 5 classes and the speeds (ball only 1800), unit ids 'rigL:uNNNN' (new ids =>
new per-unit random streams; the pools are the same real-run time blocks as rig_anchor: train 60 % / val 20 % / test 20 %).
Profiles used (declared here, frozen):
  train / val : PROFILE_L (life_sigma, narrowed shape a/p, late_sigma, u_jitter; chosen on VALIDATION of the train/val units, see reports/s7_research_*)
  test        : same life / late / jitter, but shape ranges WIDER than train (OOD shape): SHAPE_TEST_L (moderate). The full registered
                wide ranges (constants.SHAPE_TEST) are generated as a second, ROBUSTNESS test set from the same unit ids (not gated).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .. import splits as S
from . import constants as K

SECTION = "rig_anchor_L"
SEED_NOTE = 20261116
UNITS = {"train": 2000, "val": 100, "test": 100}
SHAPE_TEST_L = {"a": (0.20, 0.40), "p": (2.0, 3.0)}        # = the registered rig_anchor TRAIN family, 5x wider in a / 2.5x in p than the L train/val shapes


def plan_units_l() -> list[dict]:
    plan, k = [], 0
    for split in K.SPLITS:
        for i in range(UNITS[split]):
            cls = K.CLASSES[i % 5]
            sp = K.SPEEDS_BY_CLASS[cls]
            plan.append({"unit_id": f"rigL:u{k:04d}", "split": split, "fault_class": cls, "rpm": sp[(i // 5) % len(sp)]})
            k += 1
    return plan


def section(profile_name: str, profile: dict) -> dict:
    plan = plan_units_l()
    by = {}
    for p in plan:
        by.setdefault(p["split"], {}).setdefault(f"{p['fault_class']}@{p['rpm']}", 0)
        by[p["split"]][f"{p['fault_class']}@{p['rpm']}"] += 1
    return {"units": {p["unit_id"]: p["split"] for p in plan},
            "unit_meta": {p["unit_id"]: {"fault_class": p["fault_class"], "rpm": p["rpm"]} for p in plan},
            "counts": dict(UNITS), "composition": by, "profile_name": profile_name,
            "profile_train_val": json.loads(json.dumps(profile)), "shape_test_L_primary": SHAPE_TEST_L,
            "shape_test_wide_robustness": K.SHAPE_TEST, "seed_note": SEED_NOTE,
            "policy": "plan_units_l (deterministic); pool windows split by time blocks of the real runs exactly as rig_anchor; test units are NEW ids; "
                      "test scored once with frozen choices; test shape OOD vs train.",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": K.PROVENANCE.replace("curve shape assumed", "curve shape + life variance assumed") +
                    " Registered before any L-profile model is fitted."}


def register(profile_name: str, profile: dict) -> str:
    sp = S.load()
    if SECTION in sp["sections"]:
        return "exists"
    sp["sections"][SECTION] = section(profile_name, profile)
    S.save(sp)
    return "registered"


def verify() -> bool:
    sp = S.load()["sections"].get(SECTION)
    plan = plan_units_l()
    return sp is not None and sp["units"] == {p["unit_id"]: p["split"] for p in plan}
