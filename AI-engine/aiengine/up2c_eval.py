"""UP2 v3c spec check: ONE command that re-scores the SAVED test3 predictions (no retraining, no new test choices),
measures latency live and writes the table.  [SYNTHETIC] synth_v3 + [SIMULATION] FE twin.

    AI-engine/.venv/Scripts/python.exe -m aiengine.up2c_eval [--no-latency]
    -> reports/up2c_spec_check_<ts>.json + .md

IS2 (Specifications.md row IS2/N2, quoted): "Physics-AI residual: max relative error between the FE-identified physics
RUL and the AI RUL <=5%, health stage 1->3".
  LITERAL metric (primary, verdict): per window  residual = |RUL_physics - RUL_AI| / RUL_AI,  max over every scored
  window (window_index >= 24, the AI warm-up) of every run-to-failure test3 bearing (the faulted bearing) whose TRUE
  generator stage is 1, 2 or 3 and RUL_true > 0.  RUL_physics = phys_rul_h of that bearing (the twin's stiffness-trend
  RUL when it exists, else the documented population-life prior, so coverage is 100 % of those windows by
  construction); RUL_AI = the deployed post-processed AI RUL (hours).  Reported: max (verdict), p95, median, mean,
  share <= 5 %, coverage, share of windows where the physics value is the prior.
  Readings shown side by side:
    v3c (physics-informed AI, consumes RUL_physics as an input)  -> CONSISTENCY between a physics-driven estimator and
        an AI that reads it; NOT independent validation.
    ablation / v3b (AI without physics inputs)                   -> INDEPENDENT cross-check (the old 143 % reading).
  ABLATION (report 2026-09-03 framing): S7 MAPE v3c vs the same model without physics features, relative improvement
  (S7_ablation - S7_v3c) / S7_ablation, target >= 15 %.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import up2
from .config import MODELS_DIR, REPORTS_DIR, SPEC_TARGETS
from .up2_eval import fmt, onnx_latency, score_set, verdict
from .up2_train import TRUTH, meta, now_tag

PRED_DIR = REPORTS_DIR / "up2_preds"
FAMILY = "up2_v3c_gbm"
IS2_TARGET = 0.05
ABL_TARGET = 0.15
OLD_RESIDUAL = REPORTS_DIR / "up2_residual_20261024T072601Z.json"
V3B_HISTORY = REPORTS_DIR / "up2_spec_check_20261024T114947Z.json"
SPEC_TEXT = ("Physics-AI residual: max relative error between the FE-identified physics RUL and the AI RUL <=5%, "
             "health stage 1->3")


def latest_index() -> dict:
    files = sorted(PRED_DIR.glob(f"{FAMILY}_*.index.json"))
    if not files:
        raise SystemExit("no v3c predictions: run `python -m aiengine.up2c predict` first")
    return json.loads(files[-1].read_text()) | {"index_file": files[-1].name}


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "max": float(x.max()), "p95": float(np.percentile(x, 95)), "median": float(np.median(x)),
            "mean": float(x.mean()), "share_le_5pct": float((x <= IS2_TARGET).mean())}


def is2(rp: pd.DataFrame, t3: pd.DataFrame, min_window: int) -> dict:
    t = t3[t3["run_to_failure"] & t3["true_stage"].between(1, 3) & (t3["rul_hours"] > 0) & (t3["window_index"] >= min_window)]
    base = len(t)
    m = t.merge(rp[["unit_id", "window_index", "rul_pred_h"]], on=["unit_id", "window_index"], how="left")
    have = m["rul_pred_h"].notna() & m["phys_rul_h"].notna()
    h = m[have & (m["rul_pred_h"] > 0)].copy()
    h["res"] = (h["phys_rul_h"] - h["rul_pred_h"]).abs() / h["rul_pred_h"]
    h["prior"] = h["phys_conf"] <= 0
    out = {"definition": "|RUL_phys - RUL_AI| / RUL_AI per window; faulted bearings of test3 run-to-failure runs; "
                         f"true stage 1..3; window_index >= {min_window}; RUL_true > 0",
           "n_windows_stage1to3": int(base), "coverage": float(len(h) / base) if base else None,
           "share_physics_is_prior": float(h["prior"].mean()) if len(h) else None,
           "n_bearings": int(h["unit_id"].nunique()), "residual": _stats(h["res"].to_numpy()),
           "by_stage": {str(s): _stats(g["res"].to_numpy()) for s, g in h.groupby("true_stage")},
           "by_source": {("prior" if k else "trend"): _stats(g["res"].to_numpy()) for k, g in h.groupby("prior")},
           "by_rpm": {str(int(k)): _stats(g["res"].to_numpy()) for k, g in h.groupby("rpm_mode")},
           "physics_vs_truth": _stats(((h["phys_rul_h"] - h["rul_hours"]).abs() / h["rul_hours"]).to_numpy()),
           "ai_vs_truth": _stats(((h["rul_pred_h"] - h["rul_hours"]).abs() / h["rul_hours"]).to_numpy())}
    out["verdict_literal_max"] = "PASS" if out["residual"].get("max", np.inf) <= IS2_TARGET else "FAIL"
    return out


def latencies(t3: pd.DataFrame) -> dict:
    from . import up2_window as uw
    from .up2c import columns, NL
    out = {}
    con = MODELS_DIR / "model_contract_v3c.json"
    if not con.exists():
        return {"note": "v3c not exported yet"}
    c = json.loads(con.read_text())
    unit = t3[t3["run_to_failure"]].groupby("unit_id").size().idxmax()
    g = t3[t3["unit_id"] == unit].sort_values("window_index")
    br, _ = uw.blocks(g[columns()].to_numpy(np.float64), NL)
    T = c["models"]["tft"]["encoder_length"] + 1
    bc, _ = uw.blocks(g[up2.COLUMNS].to_numpy(np.float64), T)
    out["v3c_cls_onnx (= v3b classifier)"] = onnx_latency(MODELS_DIR / c["models"]["tft"]["onnx"],
        lambda i: {"encoder_cont": bc[T + i % (len(bc) - T): T + i % (len(bc) - T) + 1, :-1],
                   "decoder_cont": bc[T + i % (len(bc) - T): T + i % (len(bc) - T) + 1, -1:]})
    out["v3c_rul_onnx"] = onnx_latency(MODELS_DIR / c["models"]["nhits"]["onnx"], lambda i: {"encoder_cont": br[i % len(br): i % len(br) + 1]})
    # full engine step incl. the stock twin identification of the window (physics features), first 160 windows of a rig
    import time
    from .up2c_engine import PhysicsHybridEngine, run_rig
    eng = PhysicsHybridEngine(MODELS_DIR, contract_file=con.name)
    run = t3[t3["bearing_key"] == g["bearing_key"].iloc[0]]
    ts, last = [], time.perf_counter()
    for *_, o in run_rig(eng, run, n=160):
        now = time.perf_counter()
        if all(b.get("status") == "ok" for b in o["bearings"]):
            ts.append((now - last) * 1000)
        last = now
    a = np.asarray(ts[5:])
    out["v3c_engine_rig_window"] = {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                                    "max_ms": float(a.max()), "n": int(len(a)),
                                    "note": "one RIG window = physics update (stock twin identify; reference replay is O(n), "
                                            "C# keeps incremental state) + both bearings' features+cls+RUL+post-process"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2c_eval")
    ap.add_argument("--no-latency", action="store_true")
    a = ap.parse_args(argv)
    from .up2c import WARM, attach_physics, load_test3
    idx = latest_index()
    t3 = attach_physics(load_test3(), "synth_v3_test3")
    cp = pd.read_csv(PRED_DIR / idx["files"]["cls"])
    mw = WARM - 1
    rep = {"meta": meta({"tool": "aiengine.up2c_eval", "index": idx["index_file"], "scored_utc": idx["meta"].get("utc")}),
           "spec_is2": SPEC_TEXT, "data": {"test3_runs": int(t3["bearing_key"].nunique()), "windows": int(len(t3)),
                                           "table_hash": t3.attrs.get("provenance", {}).get("table_hash")},
           "models": {}, "is2": {}}
    preds = {k: pd.read_csv(PRED_DIR / idx["files"][k]) for k in ("v3c", "ablation", "v3b")}
    for k, rp in preds.items():
        rep["models"][k] = score_set(cp, rp, t3, mw)
        rep["is2"][k] = is2(rp, t3, mw)
    s7 = {k: rep["models"][k]["rul"]["S7_mape_pct"] for k in preds}
    rel = (s7["ablation"] - s7["v3c"]) / s7["ablation"]
    rep["ablation"] = {"S7_v3c": s7["v3c"], "S7_ablation": s7["ablation"], "S7_v3b": s7["v3b"],
                       "relative_improvement_vs_ablation": rel, "relative_improvement_vs_v3b": (s7["v3b"] - s7["v3c"]) / s7["v3b"],
                       "verdict": "PASS" if rel >= ABL_TARGET else "FAIL", "target": ">= 15 % relative MAPE improvement"}
    if OLD_RESIDUAL.exists():
        o = json.loads(OLD_RESIDUAL.read_text())["nominal_tuned"]
        rep["is2_history"] = {"source": OLD_RESIDUAL.name, "max": o["residual"]["max"], "p95": o["residual"]["p95"],
                              "median": o["residual"]["median"], "coverage": o["coverage"],
                              "what": "twin stock trend RUL only (no prior) vs v3 GBM AI, old synth_v3 test"}
    if V3B_HISTORY.exists():
        hb = json.loads(V3B_HISTORY.read_text())["test"].get("v3b:test2", {})
        rep["v3b_test2_history"] = {"S7": hb.get("rul", {}).get("S7_mape_pct"), "F1": hb.get("classification", {}).get("macro_f1"),
                                    "stage_max": hb.get("stage", {}).get("max_abs_error"),
                                    "caught": f"{hb.get('caught', {}).get('caught_by_stage3')}/{hb.get('caught', {}).get('rtf_bearings')}",
                                    "source": V3B_HISTORY.name}
    rep["latency"] = {} if a.no_latency else latencies(t3)
    T = SPEC_TARGETS
    rows = []
    for k, label in (("v3c", "v3c physics-informed (candidate)"), ("ablation", "v3c without physics (ablation)"), ("v3b", "v3b frozen (deployed)")):
        s = rep["models"][k]
        rows += [
            {"spec": "S7 / ICS1", "model": label, "op": "<=", "target": T["S7_rul_mape_pct"], "measured": s["rul"]["S7_mape_pct"], "unit": "%"},
            {"spec": "IS3a", "model": label, "op": ">=", "target": T["IS3_macro_f1"], "measured": s["classification"]["macro_f1"], "unit": ""},
            {"spec": "IS3b", "model": label, "op": "<=", "target": T["IS3_stage_error_max"], "measured": s["stage"]["max_abs_error"], "unit": "stages"},
            {"spec": "IS3c", "model": label, "op": "==", "target": s["caught"]["rtf_bearings"], "measured": s["caught"]["caught_by_stage3"],
             "unit": f"/{s['caught']['rtf_bearings']}"},
            {"spec": "IS2 literal max", "model": label + (" [CONSISTENCY: AI consumes physics]" if k == "v3c" else " [INDEPENDENT]"),
             "op": "<=", "target": IS2_TARGET, "measured": rep["is2"][k]["residual"]["max"], "unit": "(fraction)"},
        ]
    rows.append({"spec": "IS2 ablation (Sept-3 reading)", "model": "v3c vs ablation", "op": ">=", "target": ABL_TARGET,
                 "measured": rel, "unit": "(relative S7 gain)"})
    for key, v in rep["latency"].items():
        if isinstance(v, dict) and "p95_ms" in v:
            spec = "S8 / ICS2" if "cls" in key else ("IS1 (ICS leg)" if "engine" in key else "RUL head")
            tgt = T["S8_tft_latency_ms"] if spec != "IS1 (ICS leg)" else T["IS1_e2e_latency_ms"]
            rows.append({"spec": spec, "model": key, "op": "<", "target": tgt, "measured": v["p95_ms"], "unit": "ms p95"})
    for r in rows:
        r["verdict"] = verdict(r["op"], r["target"], r["measured"])
        r["provenance"] = "[SYNTHETIC] synth_v3_test3 FRESH (1st look)" + (" + [SIMULATION] twin" if "IS2" in r["spec"] else "") \
            if "ms" not in r["unit"] else "[MEASURED] live on this host (synthetic inputs)"
    rep["spec_table"] = rows
    ts = now_tag()
    jp, mp = REPORTS_DIR / f"up2c_spec_check_{ts}.json", REPORTS_DIR / f"up2c_spec_check_{ts}.md"
    jp.write_text(json.dumps(rep, indent=1, default=float))
    write_md(rep, mp)
    print("[SYNTHETIC] v3c spec check on FRESH synth_v3_test3 (saved predictions re-scored)")
    for r in rows:
        print(f"{r['spec']:28s} {r['op']:>2s} {fmt(r['target']):>7s} {fmt(r['measured']):>10s} {r['unit']:18s} {r['verdict']:5s} {r['model']}")
    print("->", jp, "\n->", mp)
    return 0


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f} %"


def write_md(rep: dict, path: Path) -> None:
    L = ["# UP2 v3c (physics-informed RUL) spec check -- [SYNTHETIC] synth_v3_test3 + [SIMULATION] FE twin", "",
         "**Provenance: synthetic generator runs (aiengine/synth_v3) and a simulated FE twin with ASSUMED geometry; not a "
         "measurement of the team rig.** Displacements come from the twin with K(t) tied to the generator damage, so "
         "physics and accelerometer features share one latent damage variable (circularity: the physics benefit is an upper "
         "bound until rig data exists).", "",
         f"- generated {rep['meta']['utc']}, commit `{rep['meta']['git_commit']}` (dirty={rep['meta']['git_dirty']}), seed {rep['meta']['seed']}",
         f"- command: `python -m aiengine.up2c_eval`; predictions `{rep['meta']['index']}` (made once, {rep['meta']['scored_utc']})",
         f"- test3: {rep['data']['test3_runs']} FRESH runs, {rep['data']['windows']} bearing-windows, hash `{rep['data']['table_hash']}`; "
         "registered in splits.json before any v3c fit; selection on synth_v3_val2 only", "",
         "## Spec table", "", "| spec | model | target | measured | verdict | provenance |", "|---|---|---|---|---|---|"]
    for r in rep["spec_table"]:
        L.append(f"| {r['spec']} | {r['model']} | {r['op']} {fmt(r['target'])} | {fmt(r['measured'])} {r['unit']} | **{r['verdict']}** | {r['provenance']} |")
    L += ["", "## IS2", "", f"Spec text: \"{rep['spec_is2']}\"", "", f"Definition: {rep['is2']['v3c']['definition']}.", "",
          "| AI | reading | max (verdict) | p95 | median | mean | share <= 5 % | coverage | physics = prior | n |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, rd in (("v3c", "consistency (AI consumes physics)"), ("ablation", "independent"), ("v3b", "independent")):
        s = rep["is2"][k]; r = s["residual"]
        L.append(f"| {k} | {rd} | {pct(r['max'])} ({s['verdict_literal_max']}) | {pct(r['p95'])} | {pct(r['median'])} | {pct(r['mean'])} | "
                 f"{pct(r['share_le_5pct'])} | {pct(s['coverage'])} | {pct(s['share_physics_is_prior'])} | {r['n']} |")
    if "is2_history" in rep:
        hh = rep["is2_history"]
        L.append(f"| v3 GBM (history) | independent, trend-only twin, old test | {pct(hh['max'])} (FAIL) | {pct(hh['p95'])} | {pct(hh['median'])} | - | - | {pct(hh['coverage'])} | 0 % | - |")
    L += ["", "By stage / physics source (v3c):", "", "```", json.dumps({k: rep["is2"]["v3c"][k] for k in ("by_stage", "by_source", "by_rpm")}, indent=1), "```", "",
          f"Physics RUL vs truth (same windows): median {pct(rep['is2']['v3c']['physics_vs_truth']['median'])}; AI (v3c) vs truth: median "
          f"{pct(rep['is2']['v3c']['ai_vs_truth']['median'])}.", "",
          "**What each reading means for the safety argument.** The independent reading (ablation/v3b vs physics) is a "
          "genuine cross-check: two estimators built from different sensors (accelerometers vs proximity probes) must agree, so "
          "a large residual flags a problem in one of them. With the physics-informed AI (v3c) the residual measures how "
          "closely an AI that READS the physics RUL tracks it: a small value shows consistency, not independent validation, "
          "and cannot catch an error shared by both (e.g. a wrong stiffness model). The literal number is reported for both; "
          "neither is substituted for the other.", "",
          "## Ablation (report 2026-09-03 framing)", "", "```", json.dumps(rep["ablation"], indent=1), "```", "",
          "## v3b history (fresh test2, scored earlier)", "", "```", json.dumps(rep.get("v3b_test2_history", {}), indent=1), "```", "",
          "## Per-model detail", ""]
    for k, s in rep["models"].items():
        ru = s["rul"]
        L.append(f"- **{k}**: S7 {ru['S7_mape_pct']:.2f} % ({ru['n_bearings']} bearings), true-onset {ru['alt_mape_true_onset_pct']:.2f} %, "
                 f"%-of-life {ru['alt_percent_of_life_error_pts']:.2f} pts, per rpm {ru.get('per_rpm_mode')}; macro-F1 "
                 f"{s['classification']['macro_f1']:.4f}; stage max {s['stage']['max_abs_error']} (within1 {s['stage']['within1_frac']:.4f}); "
                 f"caught {s['caught']['caught_by_stage3']}/{s['caught']['rtf_bearings']}")
    L += ["", "## Latency (live, this host, 1 thread)", "", "```", json.dumps(rep["latency"], indent=1), "```"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
