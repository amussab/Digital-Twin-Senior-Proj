"""Real-window checks of v7_node through NodeEngine. [MEASURED on rig] but NOT independent: the real windows are the anchor source
of rig_anchor. Reported, never gated. 1 bearing per fault class; severities ESTIMATED; ball has one 1800 rpm run only."""
from __future__ import annotations

import json

import numpy as np

from . import real_data as R
from . import up2b
from .config import MODELS_DIR
from .rig_anchor import constants as K

REAL_TAG = "[MEASURED on rig; anchor-source windows, NOT independent]"
TRACE_TAG = "[MEASURED on rig, n=1 ball failure trace, rpm column invalid -> 3600]"
STAGE_OF_SEV = {0: 1, 1: 3, 2: 4, 3: 6}


def _engine(contract: str):
    from .node_adapter import NodeEngine
    ne = NodeEngine(contract_file=contract, model_dir=MODELS_DIR)
    b = json.loads((MODELS_DIR / "baseline_v7_node.json").read_text())
    ne.saved = {int(k): v for k, v in b["baselines"].items()}
    return ne


def _msg(w, rpm, feats, t20=None) -> dict:
    m = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": rpm, "window_id": int(w),
         "features": list(map(float, feats))}
    if t20 is not None:
        m["t20_ms"] = int(t20)
    return m


def real_runs(contract: str = "model_contract_v7_node.json") -> dict:
    df = R.load_runs(K.TRAIN_DIR)
    rt = R.run_table(df)
    ne = _engine(contract)
    per_run, agg = [], {}
    for r in rt.itertuples():
        if int(r.rpm_key) not in ne.saved:
            continue
        g = df[df["run_id"] == r.run_id].sort_values("order")
        skip = R.settle_windows(r.rpm_key) + (R.baseline_windows(r.rpm_key) if r.is_baseline else 0)
        g = g.iloc[skip:]
        ne.reset()
        cnt = {1: {}, 2: {}}
        stg = {1: [], 2: []}
        for w in g.itertuples():
            o = ne.predict(_msg(w.window_id, w.rpm, [getattr(w, c) for c in R.FEATURE_COLUMNS]))
            for b in o["bearings"]:
                if b.get("status") == "ok":
                    cnt[b["bearing"]][b["fault_class"]] = cnt[b["bearing"]].get(b["fault_class"], 0) + 1
                    stg[b["bearing"]].append(b["health_stage"])
        sev = None if np.isnan(r.severity) else int(r.severity)
        for k in (1, 2):
            faulty = r.label != "healthy" and r.position in ("both", f"b{k}")
            truth = r.label if faulty else "healthy"
            n = sum(cnt[k].values())
            if not n:
                continue
            per_run.append({"run_id": r.run_id, "rpm": int(r.rpm_key), "bearing": k, "truth": truth,
                            "severity_estimated": sev if faulty else 0, "windows": n, "predicted": cnt[k],
                            "frac_correct": cnt[k].get(truth, 0) / n, "median_stage": float(np.median(stg[k])),
                            "expected_stage_from_severity": STAGE_OF_SEV.get(sev if faulty else 0)})
            a = agg.setdefault(truth, {"windows": 0, "correct": 0})
            a["windows"] += n
            a["correct"] += cnt[k].get(truth, 0)
    h = agg.get("healthy", {"windows": 0, "correct": 0})
    return {"tag": REAL_TAG, "per_run_bearing": per_run,
            "per_truth_class_window_recall": {k: v["correct"] / v["windows"] for k, v in agg.items()},
            "windows_by_truth": {k: v["windows"] for k, v in agg.items()},
            "healthy_false_alarm_rate_all_real_healthy_bearing_windows": 1 - h["correct"] / h["windows"] if h["windows"] else None,
            "note": "healthy = H-01 post-baseline windows (both bearings) + bearing 2 of the single-housing fault runs (housing 2 always healthy)"}


def failure_trace(lin: dict, contract: str = "model_contract_v7_node.json") -> dict:
    ne = _engine(contract)
    ne.reset()
    lines = [json.loads(l) for l in K.FAILURE_TRACE.read_text().splitlines() if l.strip()][:144]
    last = len(lines) - 1
    rows = []
    for i, rec in enumerate(lines):
        o = ne.predict(_msg(i, 3600, rec["message"]["features"], t20=i * 1000))     # 1 s per window -> hours*3600 = windows
        b = o["bearings"][0]
        ok = b.get("status") == "ok"
        rows.append({"i": i, "hi": b.get("health_index"), "rul_w": b["rul_hours"] * 3600.0 if ok else None,
                     "stage": b.get("health_stage") if ok else None, "cls": b.get("fault_class")})
    ok = [r for r in rows if r["rul_w"] is not None and r["i"] < last]
    true = np.array([last - r["i"] for r in ok], float)
    pred = np.array([r["rul_w"] for r in ok])
    hi = np.array([r["hi"] if r["hi"] is not None else 0.0 for r in rows], float)
    m6 = np.array([hi[max(0, r["i"] - 5):r["i"] + 1].mean() for r in ok])
    braw = np.maximum(np.expm1(lin["a"] + lin["b"] * m6), 0.0)
    base = up2b.pp_rul_unit(braw, lin["rul_alpha"], 1.0)

    def mape(p, sel):
        return float(np.mean(np.abs(p[sel] - true[sel]) / true[sel]) * 100)
    allw = np.ones(len(ok), bool)
    cut = true >= 0.1 * last
    return {"tag": TRACE_TAG, "windows_total": len(lines), "scored_windows": len(ok), "first_scored_window": ok[0]["i"],
            "true_rul_definition": "windows-to-last-window (window 143 = end), 1 window = 20 rev at 3600 rpm = 0.333 s",
            "v7_mape_pct_all": mape(pred, allw), "v7_mape_pct_excl_last10pct": mape(pred, cut),
            "baseline_mape_pct_all": mape(base, allw), "baseline_mape_pct_excl_last10pct": mape(base, cut),
            "v7_median_pred_over_true": float(np.median(pred / np.maximum(true, 1))),
            "v7_class_counts": {str(c): sum(1 for r in rows if r["cls"] == c) for c in {r["cls"] for r in rows}},
            "v7_stage_counts": {str(s): sum(1 for r in rows if r["stage"] == s) for s in sorted({r["stage"] for r in rows if r["stage"]})},
            "samples": [{"i": r["i"], "true_w": last - r["i"], "v7_w": r["rul_w"], "stage": r["stage"]} for r in rows[24::12]],
            "caveat": "not gated; n=1 trace; the generator's late amplification is MEASURED from this same trace, so this is not independent; "
                      "if the trivial baseline matches v5 the number measures the assumed curve"}


def latency(contract: str = "model_contract_v7_node.json", n: int = 400) -> dict:
    """NodeEngine host latency [MEASURED host], legacy baseline (first 24 windows), replay of a real run's windows."""
    from .node_adapter import NodeEngine
    df = R.load_runs(K.TRAIN_DIR)
    rid = df.groupby("run_id").size().idxmax()
    g = df[df["run_id"] == rid].sort_values("order").iloc[:n + 80]
    ne = NodeEngine(contract_file=contract, model_dir=MODELS_DIR, policy=None)
    lat, eng = [], []
    for i, w in enumerate(g.itertuples()):
        o = ne.predict(_msg(w.window_id, w.rpm, [getattr(w, c) for c in R.FEATURE_COLUMNS], t20=i * 600000))
        if o["bearings"][0].get("status") == "ok":
            lat.append(o["latency_ms"])
            eng.append(o["bearings"][0]["latency_ms"]["total"])
    a = np.array(lat)
    return {"tag": "[MEASURED host]", "windows": int(len(a)), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
            "engine_only_per_bearing_p95_ms": float(np.percentile(eng, 95)), "spec_S8_ms": 200,
            "note": "NodeEngine.predict end-to-end per window (both bearings, validation + features + ONNX + post-process), 1 thread, this laptop"}


def run(frozen: dict, heads: dict, fz: dict) -> dict:
    return {"real_windows": real_runs(), "failure_trace": failure_trace(frozen["baseline_control"]), "latency": latency()}
