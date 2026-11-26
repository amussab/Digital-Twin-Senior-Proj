"""Dry-run monitor: print what the dashboard is being fed, per bearing (class, rpm, stage, RUL), from the backend.

    python scripts/rig_dryrun.py [--backend http://127.0.0.1:5080] [--every 2] [--count 0]

Reads GET /api/live/latest (the same Live block the dashboard shows) and prints one line per new window. Standard library only.
RUL is a SIMULATION-trained head in the hybrid_rig demo (table hours); it is n/a for a healthy bearing or stage < 2.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def get(url: str):
    with urllib.request.urlopen(url, timeout=4) as r:
        return json.loads(r.read().decode())


def fmt(x, p=1):
    return "n/a" if x is None else f"{x:.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:5080")
    ap.add_argument("--every", type=float, default=2.0, help="poll period, s")
    ap.add_argument("--count", type=int, default=0, help="stop after N new windows (0 = until Ctrl+C)")
    a = ap.parse_args()
    try:
        h = get(a.backend + "/api/health")
        print("backend:", h)
    except Exception as e:  # noqa: BLE001
        print("BACKEND NOT REACHABLE:", e)
        return 2
    last, n = None, 0
    while True:
        try:
            L = get(a.backend + "/api/live/latest")
        except Exception as e:  # noqa: BLE001
            print("poll failed:", e)
            L = None
        if L and L.get("seq") != last:
            last = L["seq"]
            n += 1
            parts = []
            for b in L.get("bearings", []):
                parts.append(f"B{b['bearing']}: {b.get('faultClass')} stage={b.get('healthStage')} RUL={fmt(b.get('aiRulHours'))} h"
                             f" phys={fmt(b.get('physicsRulHours'))} h [{b.get('status')}]")
            print(f"seq {L['seq']:>5} rpm {L.get('rpm'):>6.0f} base={L.get('baselinePhase')} inf={fmt(L.get('inferenceMs'))} ms | " + " | ".join(parts)
                  + f" | {L.get('contract')} {L.get('rulContract') or ''}")
            if a.count and n >= a.count:
                return 0
        elif L is None or last is None:
            print("no live window yet (node not posting, or still warming up)")
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
