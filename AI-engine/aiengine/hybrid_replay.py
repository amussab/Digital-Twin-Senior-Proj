"""Replay recorded real rig runs (COE/training/training_data/run_*.jsonl + .json meta) through the hybrid_rig engine.

    python -m aiengine.hybrid_replay [--data DIR] [--baseline models/baseline_v4_real.json] [--out reports/hybrid_replay.json]

[MEASURED host, replay of rig data]. Recordings are same-bearing (one bearing per fault), NOT independent held-out bearings; the
baseline file was captured from the healthy baseline runs of the same rig day. RUL/stage heads are simulation-trained: numbers show
what the demo displays, not accuracy. Faulted bearing = B1 (position b1); B2 is the always-healthy housing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, REPORTS_DIR
from .node_hybrid import HybridNodeEngine

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "COE" / "training" / "training_data"


def load_runs(data: Path = DATA):
    for meta in sorted(Path(data).glob("run_*.json")):
        jl = meta.with_suffix(".jsonl")
        if not jl.exists():
            continue
        m = json.loads(meta.read_text())
        msgs = []
        for line in jl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                msgs.append(d["message"] if "message" in d and "features" not in d else d)
        yield m, msgs


def replay(msgs, baseline: Path | None = None, rul_contract: str | None = None, timescale: float | None = None, limit: int | None = None) -> list[dict]:
    eng = HybridNodeEngine(rul_contract=rul_contract, timescale=timescale)
    if baseline:
        eng.load_baseline(baseline)
    return [eng.predict(m) for m in (msgs[:limit] if limit else msgs)]


def summarise(meta: dict, outs: list[dict]) -> dict:
    label = meta["label"]
    pos = meta.get("position") or "b1"
    bi = 1 if pos in ("b1", "both") else 2
    ok = [o for o in outs if o["bearings"][0].get("status") == "ok"]
    res = {"run_id": meta["run_id"], "label": label, "severity": meta.get("severity"), "windows": len(outs), "windows_ok": len(ok)}
    for k in (1, 2):
        bs = [o["bearings"][k - 1] for o in ok]
        cc = Counter(b["fault_class"] for b in bs)
        n = max(len(bs), 1)
        want = label if (k == bi or pos == "both") else "healthy"
        res[f"B{k}"] = {"expected": want, "class_counts": dict(cc), "class_accuracy": cc.get(want, 0) / n,
                        "false_alarm_pct": 100.0 * sum(v for c, v in cc.items() if c != "healthy") / n if want == "healthy" else None,
                        "rul_issued_pct": 100.0 * sum(b["rul_valid"] for b in bs) / n,
                        "stage_counts": dict(Counter(b["health_stage"] for b in bs)),
                        "ai_rul_h": _rng([b["ai_rul_hours"] for b in bs]), "physics_rul_h": _rng([b["physics_rul_hours"] for b in bs]),
                        "ai_rul_head_h_ungated": _rng([b["ai_rul_head_hours"] for b in bs]),
                        "miner_d": _rng([b["miner_d"] for b in bs])}
    return res


def _rng(v):
    v = [x for x in v if x is not None]
    return None if not v else {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--baseline", default=str(MODELS_DIR / "baseline_v4_real.json"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "hybrid_replay.json"))
    a = ap.parse_args()
    rows = []
    for meta, msgs in load_runs(Path(a.data)):
        rows.append(summarise(meta, replay(msgs, Path(a.baseline))))
        r = rows[-1]
        print(r["run_id"], r["label"], f"B1 acc {r['B1']['class_accuracy']:.3f}", f"B2 FA {r['B2']['false_alarm_pct']:.1f}%" if r["B2"]["false_alarm_pct"] is not None else "",
              "stages", r["B1"]["stage_counts"], "RULissued%", round(r["B1"]["rul_issued_pct"], 1), "RUL", r["B1"]["ai_rul_h"])
    Path(a.out).write_text(json.dumps({"tag": "[MEASURED host, replay of rig data]; same-bearing recordings, not independent; sim-trained RUL/stage heads", "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
