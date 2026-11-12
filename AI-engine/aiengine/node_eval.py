"""Replay a recorded node stream through the deployed engine and summarise it.

    python -m aiengine.node_eval <file.jsonl|file.csv> [--contract v3e_node] [--rpm-override N] [--out reports/x.json]

Inputs: host_receiver JSONL (one node message per line, or {"received_at":..,"message":{..}}) or the
COE/training/training_recorder CSV (run_id,label,note,received_at,rpm,window_id,sequence,device_id,valid_channels, then
B1X_rms .. B2Y_bpfi); CSV rows are grouped by run_id, one fresh engine (own 24-window baseline) per run.
Prints per-bearing class distribution after warm-up, first onset window, host latency p50/p95, writes a JSON report to
reports/node_eval_<name>_<ts>.json. Provenance of every number = the input file (tag it when quoting).
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import REPORTS_DIR
from .node_adapter import NodeEngine

CHANNELS = ("B1X", "B1Y", "B2X", "B2Y")
FEATS = ("rms", "kurtosis", "crest", "envelope_rms", "ftf", "bsf", "bpfo", "bpfi")
FEATURE_COLUMNS = [f"{c}_{f}" for c in CHANNELS for f in FEATS]


def read_runs(path: Path) -> dict[str, dict]:
    """-> {run_key: {"label": str|None, "messages": [dict]}} in file order."""
    path = Path(path)
    runs: dict[str, dict] = {}
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": int(float(r["valid_channels"])),
                     "rpm": float(r["rpm"]), "window_id": int(float(r["window_id"])), "sequence": int(float(r["sequence"])),
                     "features": [float(r[c]) for c in FEATURE_COLUMNS]}
                run = runs.setdefault(r["run_id"], {"label": r.get("label"), "messages": []})
                run["messages"].append(m)
    else:
        msgs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                msgs.append(d["message"] if "message" in d and "features" not in d else d)
        runs[path.stem] = {"label": None, "messages": msgs}
    return runs


def evaluate_run(messages: list[dict], contract: str, rpm_override: float | None, legacy: bool = False) -> dict:
    eng = NodeEngine(contract_file=f"model_contract_{contract}.json", policy=None if legacy else "contract")
    dist = {1: {}, 2: {}}
    onset = {1: None, 2: None}
    lat, skipped = [], 0
    for i, m in enumerate(messages):
        if rpm_override is not None:
            m = {**m, "rpm": rpm_override}
        try:
            out = eng.predict(m)
        except ValueError:
            skipped += 1
            continue
        lat.append(out["latency_ms"])
        for b in out["bearings"]:
            k = b["bearing"]
            if b.get("status") != "ok":
                continue
            c = b["fault_class"]
            dist[k][c] = dist[k].get(c, 0) + 1
            if onset[k] is None and b.get("onset_detected"):
                onset[k] = i
    res = {"windows": len(messages), "skipped_invalid": skipped,
           "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))} if lat else None,
           "bearings": {}}
    for k in (1, 2):
        n = sum(dist[k].values())
        res["bearings"][str(k)] = {"windows_after_warmup": n, "class_counts": dist[k],
                                   "class_fraction": {c: v / n for c, v in dist[k].items()} if n else {},
                                   "first_onset_window_index": onset[k]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.node_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--contract", default="v3e_node", help="contract name without model_contract_ / .json")
    ap.add_argument("--rpm-override", type=float, default=None)
    ap.add_argument("--legacy-baseline", action="store_true", help="ignore the contract baseline policy (24-window median)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    path = Path(a.file)
    runs = read_runs(path)
    report = {"file": str(path), "contract": a.contract, "rpm_override": a.rpm_override,
              "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": {}}
    for key, run in runs.items():
        r = evaluate_run(run["messages"], a.contract, a.rpm_override, a.legacy_baseline)
        r["label"] = run["label"]
        report["runs"][key] = r
        print(f"== {key} (label={run['label']}) windows={r['windows']} skipped={r['skipped_invalid']} latency={r['latency_ms']}")
        for k, b in r["bearings"].items():
            fr = ", ".join(f"{c} {100 * v:.1f}%" for c, v in sorted(b["class_fraction"].items(), key=lambda kv: -kv[1]))
            print(f"   B{k}: n={b['windows_after_warmup']} {fr or '(still warming up)'}; first onset idx {b['first_onset_window_index']}")
    out = Path(a.out) if a.out else REPORTS_DIR / f"node_eval_{path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
