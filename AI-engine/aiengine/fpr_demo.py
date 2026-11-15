"""FPR demo assets: render rig_anchor VAL units to node-format jsonl (commissioning windows prepended) for the live replay.

    python -m aiengine.fpr_demo render            # writes AI-engine/demo/fpr_<class>_<unit>.jsonl (committed, small) + demo_manifest.json
    python -m aiengine.fpr_demo check [--contract v7_node] [file]   # offline NodeEngine(policy, table time base) run -> class/stage/RUL/physics per window

Provenance: [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]. The units are VALIDATION units
(never test). The replay clock is the node cadence (1200/rpm s per window); the RUL clock is the 10-minute snapshot clock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT
from .rig_anchor import constants as K

DEMO_DIR = ROOT / "demo"
DEMO_UNITS = {"inner_race": "rig_anchor:u0241", "outer_race": "rig_anchor:u0242", "cage": "rig_anchor:u0244"}


def cmd_render(a) -> int:
    from .rig_anchor import render_unit
    DEMO_DIR.mkdir(exist_ok=True)
    man = {"provenance": K.PROVENANCE, "limits": K.LIMITS, "split": "val", "files": {}}
    for cls, uid in DEMO_UNITS.items():
        f = DEMO_DIR / f"fpr_{cls}_{uid.split(':')[1]}.jsonl"
        info = render_unit.render(uid, str(f), commissioning=True)
        man["files"][f.name] = info
        print(f.name, info, f.stat().st_size // 1024, "KB")
    (DEMO_DIR / "demo_manifest.json").write_text(json.dumps(man, indent=1))
    return 0


def cmd_check(a) -> int:
    from .node_adapter import NodeEngine, read_jsonl
    f = Path(a.file) if a.file else next(iter(sorted(DEMO_DIR.glob("fpr_*.jsonl"))))
    e = NodeEngine(f"model_contract_{a.contract}.json", time_base="table")
    last = None
    for k, line in enumerate(read_jsonl(f)):
        o = e.predict(line["message"])
        b = o["bearings"][0]
        if b.get("status") == "ok" and k % a.every == 0:
            print(k, b["fault_class"], "stage", b["health_stage"], "AI", b["ai_rul_hours"], "phys", b["physics_rul_hours"], "D", round(b["miner_d"], 2), b["physics_source"])
        last = b
    print("last:", {k: last.get(k) for k in ("fault_class", "health_stage", "ai_rul_hours", "physics_rul_hours", "miner_d")})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.fpr_demo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render")
    c = sub.add_parser("check")
    c.add_argument("file", nargs="?")
    c.add_argument("--contract", default="v7_node")
    c.add_argument("--every", type=int, default=40)
    a = ap.parse_args(argv)
    return {"render": cmd_render, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
