"""Render one simulated unit to recorder-format jsonl for COE/inference/replay_node.py.

    python -m aiengine.rig_anchor.render_unit rig_anchor:u0007 --out unit.jsonl [--commissioning]

One line per window: {"message": {...rpm, window_id, sequence, features[32]...}, "received_at": ...}.
NOTE: table window_index 0 == first replayed window unless --commissioning is used, which PREPENDS 180 (settle) + 90 (baseline) HEALTHY
resampled windows at 1800 rpm (360+180 at 3600; the contract policy: settle 120 s dropped, median over the next 60 s) drawn with the
unit's own nuisance, so a NodeEngine baseline can be built by the real policy; the replay offset is printed and stored in the first line's
"sim_meta". Replay cadence is 1200/rpm s per window; the table's t_hours (10-min snapshots) is the simulated clock, not the replay clock.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import real_data as R
from . import constants as K
from . import generator as G
from . import pools as P


def render(unit_id: str, out: str, commissioning: bool = False) -> dict:
    plan = {p["unit_id"]: p for p in G.plan_units()}
    if unit_id not in plan:
        raise SystemExit(f"unknown unit {unit_id}")
    spec = plan[unit_id]
    pools = P.Pools()
    df = G.generate_unit(spec, pools)
    X = G.window_matrix(df)
    rpm = int(spec["rpm"])
    pre = np.zeros((0, 32))
    if commissioning:
        npre = R.settle_windows(rpm) + R.baseline_windows(rpm)
        rng = np.random.default_rng(G.unit_seed(unit_id) + 1)
        Hx, _, _ = P.draw(pools.healthy[rpm][spec["split"]], npre, rng)
        # same nuisance as the unit: ratio of the unit's first healthy window to its own source window is not recoverable; use unit run-in median level
        first = X[:min(K.BASELINE_WINDOWS, len(X))]
        scale = np.median(first, axis=0) / np.maximum(np.median(Hx, axis=0), K.LOG_FLOOR)
        pre = R.apply_b2x_policy(Hx * scale)
    allX = np.concatenate([pre, X])
    t0 = datetime(2026, 11, 16, 9, 0, 0, tzinfo=timezone(timedelta(hours=3)))
    dt = 1200.0 / rpm
    with open(out, "w", encoding="utf-8") as f:
        for k, row in enumerate(allX):
            msg = {"acceleration_unit": "g", "bearing_profile": 1, "device_id": "rig-anchor-sim", "features": [float(v) for v in row],
                   "rpm": rpm, "schema_version": 1, "sequence": k, "session_id": "0000riganchor000", "valid_channels": 15, "window_id": k}
            if k == 0:
                msg["sim_meta"] = {"provenance": K.PROVENANCE, "unit_id": unit_id, "fault_class": spec["fault_class"], "split": spec["split"],
                                   "replay_offset_windows": len(pre), "table_cadence_hours": K.CADENCE_H}
            f.write(json.dumps({"message": msg, "received_at": (t0 + timedelta(seconds=k * dt)).isoformat(timespec="milliseconds")}) + "\n")
    return {"unit": unit_id, "windows": len(allX), "prepended": len(pre), "class": spec["fault_class"], "rpm": rpm}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("unit_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commissioning", action="store_true")
    a = ap.parse_args(argv)
    print(K.PROVENANCE, render(a.unit_id, a.out, a.commissioning))


if __name__ == "__main__":
    main()
