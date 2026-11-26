"""v6_node data: fresh unit registration (splits.json section `rig_anchor_test2`) + generation. [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]

Registered BEFORE any v6 fit:
  * `test`  : 80 FRESH test units (ids u1000..u1079), composition like the registered v5 test split (5 classes x speeds, ball only 1800), drawn ONLY
              from the TEST pool blocks (last 20 % of every real run), shape ranges = K.SHAPE_TEST (same OOD ranges). New unit ids => new per-unit seeds
              (unit_seed = f(MASTER_SEED, crc32(unit_id))). They are NOT new bearings: they re-use the same real anchor windows as the old test units.
  * `train` : 400 EXTRA train units (u2000..u2399) from the train pool blocks (more simulated trajectories of the same 5 anchor bearings).
  * `control_wrong_shape` : 80 units (u3000..u3079), test composition/pool, near-linear damage curves (a 0.85-0.95, p 1.0-1.2), NEW seeds.
The generator (rig_anchor/generator.py) is called unchanged through its public API (generate_unit with explicit unit specs).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import splits as S
from .config import ROOT
from .rig_anchor import constants as K
from .rig_anchor import generator as G
from .rig_anchor import pools as P

SECTION = "rig_anchor_test2"
DATA_DIR = ROOT / "data" / "rig_anchor"
DATA = DATA_DIR / "rig_anchor_v6.parquet"
DATA_WRONG = DATA_DIR / "rig_anchor_v6_wrong.parquet"
BLOCKS = {"test": (1000, 80), "train": (2000, 400), "wrong": (3000, 80)}
WRONG_SHAPE = {"a": (0.85, 0.95), "p": (1.0, 1.2)}


def plan(kind: str) -> list[dict]:
    """Deterministic (no RNG) balanced plan, same enumeration rule as rig_anchor.generator.plan_units."""
    start, n = BLOCKS[kind]
    split = "train" if kind == "train" else "test"
    out = []
    for i in range(n):
        cls = K.CLASSES[i % 5]
        sp = K.SPEEDS_BY_CLASS[cls]
        out.append({"unit_id": f"{K.DATASET}:u{start + i:04d}", "split": split, "fault_class": cls, "rpm": sp[(i // 5) % len(sp)]})
    return out


def section() -> dict:
    units, meta = {}, {}
    for kind in ("test", "train"):
        for p in plan(kind):
            units[p["unit_id"]] = p["split"]
            meta[p["unit_id"]] = {"fault_class": p["fault_class"], "rpm": p["rpm"]}
    ctrl = {p["unit_id"]: {"fault_class": p["fault_class"], "rpm": p["rpm"]} for p in plan("wrong")}
    comp = {}
    for u, m in meta.items():
        comp.setdefault(units[u], {}).setdefault(f"{m['fault_class']}@{m['rpm']}", 0)
        comp[units[u]][f"{m['fault_class']}@{m['rpm']}"] += 1
    return {"units": units, "unit_meta": meta, "control_units": ctrl, "composition": comp,
            "counts": {"test": sum(v == "test" for v in units.values()), "train": sum(v == "train" for v in units.values()),
                       "control_wrong_shape": len(ctrl)},
            "policy": "FRESH test units for v6_node (ids u1000-u1079), drawn only from the TEST pool blocks of rig_anchor.pools (last 20 % of each real run, "
                      "the SAME real windows as the v5 test units); same shape OOD ranges as the registered rig_anchor test split; new per-unit seeds via new "
                      "unit ids. NOT new bearings. Extra train units u2000-u2399 from the train pool blocks. control_wrong_shape: u3000-u3079.",
            "shape_ranges": {"train": K.SHAPE_TRAIN, "test": K.SHAPE_TEST, "control_wrong_shape": WRONG_SHAPE},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": K.PROVENANCE + " " + K.LIMITS + " Registered before any v6 model is fitted; the fresh test units are scored once."}


def register() -> str:
    sp = S.load()
    if SECTION in sp["sections"]:
        return "exists"
    sp["sections"][SECTION] = section()
    S.save(sp)
    return "registered"


def verify() -> bool:
    cur = S.load()["sections"].get(SECTION)
    new = section()
    return cur is not None and all(cur[k] == new[k] for k in ("units", "unit_meta", "control_units"))


def build(out_main: Path = DATA, out_wrong: Path = DATA_WRONG) -> dict:
    if not verify():
        raise SystemExit("rig_anchor_test2 missing/changed in splits.json: refusing to generate")
    pools, late = P.Pools(), P.load_late_profile()
    tr = pd.concat([G.generate_unit(s, pools, late) for s in plan("train")], ignore_index=True)
    te = pd.concat([G.generate_unit(s, pools, late) for s in plan("test")], ignore_index=True)
    tr["v6_set"], te["v6_set"] = "train_extra", "fresh_test"
    out_main.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([tr, te], ignore_index=True).to_parquet(out_main, index=False)
    wr = pd.concat([G.generate_unit(s, pools, late, shape_override=WRONG_SHAPE) for s in plan("wrong")], ignore_index=True)
    wr["v6_set"] = "control_wrong_shape"
    wr.to_parquet(out_wrong, index=False)
    return {"train_extra_rows": len(tr), "fresh_test_rows": len(te), "wrong_rows": len(wr)}


if __name__ == "__main__":
    print(register(), verify())
    print(json.dumps(build()))
