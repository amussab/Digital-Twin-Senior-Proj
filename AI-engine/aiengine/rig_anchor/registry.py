"""Register the unit-level train/val/test assignment in AI-engine/splits.json (section 'rig_anchor') BEFORE anything is fitted.

Idempotent: an existing section is never rewritten (verify() checks it still equals the deterministic plan).
Only this section is added; other sections are left byte-for-byte equal in content.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .. import splits as S
from . import constants as K
from .generator import plan_units
from .pools import Pools


def section() -> dict:
    plan = plan_units()
    counts: dict = {}
    for p in plan:
        counts[p["split"]] = counts.get(p["split"], 0) + 1
    by = {}
    for p in plan:
        by.setdefault(p["split"], {}).setdefault(f"{p['fault_class']}@{p['rpm']}", 0)
        by[p["split"]][f"{p['fault_class']}@{p['rpm']}"] += 1
    pools = Pools()
    return {
        "units": {p["unit_id"]: p["split"] for p in plan},
        "unit_meta": {p["unit_id"]: {"fault_class": p["fault_class"], "rpm": p["rpm"]} for p in plan},
        "counts": counts, "composition": by,
        "policy": "aiengine.rig_anchor.generator.plan_units (deterministic enumeration; balanced over 5 classes and speeds; ball only 1800); "
                  "pool windows split by TIME BLOCKS inside each real run: train first 60 %, val next 20 %, test last 20 % (after the settle drop); "
                  "units of split S use ONLY pool-S windows; test shape parameters drawn from a WIDER range than train (shape OOD)",
        "pool_blocks": {"fractions": K.POOL_FRACTIONS, "runs": pools.runs_used},
        "shape_ranges": {"train_val": K.SHAPE_TRAIN, "test": K.SHAPE_TEST},
        "seed": K.MASTER_SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": K.PROVENANCE + " " + K.LIMITS + " Registered before any model is fitted; test scored once.",
    }


def register() -> str:
    sp = S.load()
    if K.SPLIT_SECTION in sp["sections"]:
        return "exists"
    sp["sections"][K.SPLIT_SECTION] = section()
    S.save(sp)
    return "registered"


def verify() -> bool:
    sp = S.load()["sections"].get(K.SPLIT_SECTION)
    plan = plan_units()
    return sp is not None and sp["units"] == {p["unit_id"]: p["split"] for p in plan} and \
        sp["unit_meta"] == {p["unit_id"]: {"fault_class": p["fault_class"], "rpm": p["rpm"]} for p in plan}


if __name__ == "__main__":
    print(register(), verify())
