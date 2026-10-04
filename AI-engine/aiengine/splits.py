"""Frozen bearing-level splits (DESIGN.md section 5) -> AI-engine/splits.json.

Never window-level. Each dataset gets its own section, created once and then frozen: re-running
never re-shuffles an existing section (pass force=True to rebuild deliberately, which must be
recorded). Test units are never used for training, early stopping, model selection or
threshold calibration.

Policies
- xjtu_sy: by bearing, stratified by operating condition (BearingC_k -> condition C):
  per condition 1 test, 1 val, rest train.
- ims: the fine-tuning demonstration domain. Kept out of pretraining entirely. Within IMS:
  run-to-failure bearings -> one held out as `ft_test`, the rest `ft_train`; non-failing
  bearings -> `ft_train`.
- mafaulda: units grouped by (position, fault type, severity) = bearing_key; within a group,
  records are ordered by rpm and cut into contiguous speed blocks; whole blocks go to one split,
  so test speeds of a group are never seen for that group in training. The test/val block index
  rotates across groups so every speed range is represented in test.
- synthetic_rig: by run (both bearings together), stratified by the run's fault class.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import ROOT, SEED

SPLITS_FILE = ROOT / "splits.json"
N_SPEED_BLOCKS = 5


def load() -> dict:
    if SPLITS_FILE.exists():
        return json.loads(SPLITS_FILE.read_text())
    return {"version": 1, "seed": SEED, "sections": {}}


def save(splits: dict) -> None:
    SPLITS_FILE.write_text(json.dumps(splits, indent=1, sort_keys=True))


def unit_split_map(splits: dict) -> dict[str, str]:
    out = {}
    for sec in splits["sections"].values():
        out.update(sec["units"])
    return out


def _xjtu_condition(unit_id: str) -> str:
    m = re.search(r"Bearing(\d)_(\d)", unit_id)
    return m.group(1) if m else "?"


def _split_xjtu(units: pd.DataFrame, rng) -> dict[str, str]:
    out = {}
    for cond, g in units.groupby(units["unit_id"].map(_xjtu_condition)):
        ids = sorted(g["unit_id"])
        rng.shuffle(ids)
        for i, u in enumerate(ids):
            out[u] = "test" if i == 0 else ("val" if i == 1 else "train")
    return out


def _split_ims(units: pd.DataFrame, rng) -> dict[str, str]:
    out = {}
    rtf = sorted(units.loc[units["run_to_failure"], "unit_id"])
    rng.shuffle(rtf)
    for i, u in enumerate(rtf):
        out[u] = "ft_test" if i == 0 else "ft_train"
    for u in units.loc[~units["run_to_failure"], "unit_id"]:
        out[u] = "ft_train"
    return out


def _split_mafaulda(units: pd.DataFrame, rng) -> dict[str, str]:
    out = {}
    groups = sorted(units["bearing_key"].unique())
    offset = int(rng.integers(0, N_SPEED_BLOCKS))
    for gi, key in enumerate(groups):
        g = units[units["bearing_key"] == key].sort_values("rpm")
        ids = g["unit_id"].tolist()
        blocks = np.array_split(np.arange(len(ids)), min(N_SPEED_BLOCKS, len(ids)))
        nb = len(blocks)
        test_b = (gi + offset) % nb
        val_b = (test_b + 2) % nb if nb >= 3 else None
        for b, members in enumerate(blocks):
            split = "test" if b == test_b else ("val" if b == val_b else "train")
            if nb < 3 and b != test_b:
                split = "train"
            for m in members:
                out[ids[m]] = split
    return out


def _split_by_key(units: pd.DataFrame, rng, val=0.15, test=0.2) -> dict[str, str]:
    out = {}
    keys = units.groupby("bearing_key").agg(
        cls=("fault_class", lambda s: "|".join(sorted(set(s) - {"healthy"})) or "healthy"))
    for _, g in keys.groupby("cls"):
        ks = sorted(g.index)
        rng.shuffle(ks)
        n = len(ks)
        nt = max(1, int(round(n * test))) if n >= 3 else 0
        nv = max(1, int(round(n * val))) if n >= 3 else 0
        for i, k in enumerate(ks):
            s = "test" if i < nt else ("val" if i < nt + nv else "train")
            for u in units.loc[units["bearing_key"] == k, "unit_id"]:
                out[u] = s
    return out


POLICIES = {"xjtu_sy": _split_xjtu, "ims": _split_ims, "mafaulda": _split_mafaulda,
            "synthetic_rig": _split_by_key}


def ensure(table: pd.DataFrame, force: bool = False) -> dict:
    """Create sections for datasets present in `table` that have none yet. Never reshuffles."""
    splits = load()
    changed = False
    units = (table.groupby("unit_id")
             .agg(dataset=("dataset", "first"), bearing_key=("bearing_key", "first"),
                  fault_class=("fault_class", "first"), run_to_failure=("run_to_failure", "first"),
                  rpm=("rpm", "median"), n=("window_index", "size"))
             .reset_index())
    for ds, g in units.groupby("dataset"):
        sec = splits["sections"].get(ds)
        missing = set(g["unit_id"]) - set(sec["units"]) if sec else set(g["unit_id"])
        if sec and not missing and not force:
            continue
        if sec and missing and not force:
            raise RuntimeError(
                f"splits.json section '{ds}' is frozen but {len(missing)} new unit(s) appeared "
                f"(e.g. {sorted(missing)[:3]}). Rebuild deliberately with force=True.")
        rng = np.random.default_rng(splits.get("seed", SEED) + sum(map(ord, ds)))
        mapping = POLICIES.get(ds, _split_by_key)(g, rng)
        counts = pd.Series(mapping).value_counts().to_dict()
        splits["sections"][ds] = {
            "policy": POLICIES.get(ds, _split_by_key).__name__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": {k: int(v) for k, v in counts.items()},
            "units": dict(sorted(mapping.items())),
        }
        changed = True
    if changed:
        save(splits)
    return splits


def assign(table: pd.DataFrame, splits: dict | None = None) -> pd.Series:
    m = unit_split_map(splits or load())
    s = table["unit_id"].map(m)
    if s.isna().any():
        raise RuntimeError(f"{s.isna().sum()} rows have units without a split; run splits.ensure")
    return s
