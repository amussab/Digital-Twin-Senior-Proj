"""UP2 spec check: one command that regenerates/loads the SYNTHETIC data, re-scores the SAVED per-window
predictions (no retraining, no new look at test choices), measures latency live, and prints/writes the spec table.

    AI-engine/.venv/Scripts/python.exe -m aiengine.up2_eval            # from AI-engine/
    -> reports/up2_spec_check_<ts>.json + .md

Rows: spec -> target -> measured -> PASS/FAIL -> provenance -> model. Every synth_v3 number is [SYNTHETIC].
The previous public-data results (PPR, v2) are carried from their stored report as [MEASURED on ...].
Metric definitions: aiengine/up2_metrics.py (module docstring), repeated in the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import up2
from . import up2_metrics as M
from .config import MODELS_DIR, REPORTS_DIR, SPEC_TARGETS
from .up2_train import ROBUST_SETS, TRUTH, load_main, load_robust, meta, now_tag

PRED_DIR = REPORTS_DIR / "up2_preds"
REAL_V2_REPORT = REPORTS_DIR / "spec_check_20261005T012112Z_real_v2.json"


def latest_index(family: str) -> dict | None:
    files = sorted(PRED_DIR.glob(f"up2_v3_{family}_*.index.json"))
    return json.loads(files[-1].read_text()) | {"index_file": files[-1].name} if files else None


def load_preds(idx: dict, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = idx["files"][name]
    return pd.read_csv(PRED_DIR / f["cls"]), pd.read_csv(PRED_DIR / f["rul"])


def score_set(cp: pd.DataFrame, rp: pd.DataFrame, df: pd.DataFrame, min_window: int | None = None) -> dict:
    if min_window is not None:     # common scored windows across model families (all heads warm)
        cp, rp = cp[cp["window_index"] >= min_window], rp[rp["window_index"] >= min_window]
    c = up2.merge_truth(cp, df, TRUTH)
    r = up2.merge_truth(rp, df, TRUTH)
    rr = r[r["run_to_failure"]]
    j = c.merge(rp[["unit_id", "window_index", "stage_pred"]], on=["unit_id", "window_index"], how="left")
    return {"classification": M.classification(c), "caught": M.caught_by_stage3(j), "stage": M.stage(rr),
            "rul": M.rul(rr)}


def onnx_latency(path: Path, feeds_fn, n: int = 300) -> dict:
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    ts = []
    for i in range(n + 20):
        f = feeds_fn(i)
        f = {k: v for k, v in f.items() if k in {x.name for x in s.get_inputs()}}
        t0 = time.perf_counter()
        s.run(None, f)
        if i >= 20:
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
            "n": int(len(a)), "threads": 1, "batch": 1}


def latencies(df: pd.DataFrame) -> dict:
    from . import up2_window as uw
    out = {}
    unit = df[df["run_to_failure"]].groupby("unit_id").size().idxmax()
    g = df[df["unit_id"] == unit].sort_values("window_index")
    rows = g[up2.COLUMNS].to_numpy(np.float64)
    con3 = MODELS_DIR / "model_contract_v3.json"
    if con3.exists():
        c = json.loads(con3.read_text())
        T = c["models"]["tft"]["encoder_length"] + 1
        NL = c["models"]["nhits"]["encoder_length"]
        bc, _ = uw.blocks(rows, T)
        br, _ = uw.blocks(rows, NL)
        out["v3_cls_onnx"] = onnx_latency(MODELS_DIR / c["models"]["tft"]["onnx"],
                                          lambda i: {"encoder_cont": bc[T + i % (len(bc) - T): T + i % (len(bc) - T) + 1, :-1],
                                                     "decoder_cont": bc[T + i % (len(bc) - T): T + i % (len(bc) - T) + 1, -1:]})
        out["v3_rul_onnx"] = onnx_latency(MODELS_DIR / c["models"]["nhits"]["onnx"],
                                          lambda i: {"encoder_cont": br[i % len(br): i % len(br) + 1]})
        from .engine import HybridEngine
        out["v3_engine_step"] = HybridEngine(MODELS_DIR, contract_file=con3.name).benchmark(g, n=200)
    con_nn = MODELS_DIR / "model_contract_v3_nn.json"
    if con_nn.exists():
        from . import infer
        c = json.loads(con_nn.read_text())
        mt = c["models"]["tft"]
        enc, dec = infer.tft_tensors(g[mt["columns"]].to_numpy(np.float64), mt["columns"], mt["scalers"], mt["encoder_length"])
        out["v3_nn_tft_onnx"] = onnx_latency(MODELS_DIR / mt["onnx"],
                                             lambda i: {"encoder_cont": enc[i % len(enc)][None], "decoder_cont": dec[i % len(dec)][None]})
        from .engine import HybridEngine
        out["v3_nn_engine_step"] = HybridEngine(MODELS_DIR, contract_file=con_nn.name).benchmark(g, n=200)
    out["host"] = f"{platform.processor() or platform.machine()} / {platform.system()} {platform.release()}"
    out["unit"] = unit
    return out


def verdict(op: str, target: float, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    ok = {"<=": v <= target, "<": v < target, ">=": v >= target, "==": v == target}[op]
    return "PASS" if ok else "FAIL"


def spec_rows(s: dict, lat: dict, model: str, prov: str = "[SYNTHETIC] synth_v3 test runs") -> list[dict]:
    T = SPEC_TARGETS
    rul, st, cl, ca = s["rul"], s["stage"], s["classification"], s["caught"]
    rows = [
        ("S7 / ICS1", "RUL MAPE, absolute hours (spec metric, DESIGN.md s7 pre-registered window)", "<=", T["S7_rul_mape_pct"], rul["S7_mape_pct"], "%"),
        ("S7-strict", "STRICTER variant (synthetic only): same MAPE from the TRUE onset", "<=", T["S7_rul_mape_pct"],
         rul["alt_mape_true_onset_pct"], "%"),
        ("S7-alt", "ALTERNATIVE (not the spec metric): percent-of-life error", "<=", T["S7_rul_mape_pct"],
         rul["alt_percent_of_life_error_pts"], "pts"),
        ("IS3a", "macro-F1 over {healthy, OD, ID, BD, CF}", ">=", T["IS3_macro_f1"], cl["macro_f1"], ""),
        ("IS3b", "health-stage error, literal max over all scored windows", "<=", T["IS3_stage_error_max"],
         st["max_abs_error"], "stages"),
        ("IS3c", "every injected fault caught by stage 3 (bearings)", "==", ca["rtf_bearings"], ca["caught_by_stage3"],
         f"/{ca['rtf_bearings']}"),
    ]
    out = [{"spec": a, "what": b, "op": op, "target": t, "measured": v, "unit": u, "verdict": verdict(op, t, v),
            "provenance": prov, "model": model} for a, b, op, t, v, u in rows]
    return out


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def write_md(rep: dict, path: Path) -> None:
    L = ["# UP2 spec check (contract v3) -- [SYNTHETIC] synth_v3 data", "",
         "**Provenance: every synth_v3 number below is [SYNTHETIC]: physics-informed generated data "
         "(aiengine/synth_v3), NOT a measurement of the team rig or of any public dataset.** "
         "The public-data (PPR v2) rows are [MEASURED on XJTU-SY / IMS / MaFaulDa], carried from their stored report.", "",
         f"- generated: {rep['meta']['utc']}  commit `{rep['meta']['git_commit']}` (dirty={rep['meta']['git_dirty']})  seed {rep['meta']['seed']}",
         f"- command: `{rep['meta']['command']}`",
         f"- data: {rep['data']['main']['n_runs']} runs, table hash `{rep['data']['main'].get('table_hash')}`, "
         f"split by run (splits.json section synth_v3): {rep['data']['split_check']['runs_per_split']}",
         f"- predictions re-scored from: {rep['pred_index'].get('gbm')} (deployed GBM heads), {rep['pred_index'].get('nn')} (TFT + N-HiTS)",
         f"- scored windows: window_index >= {rep['min_window']} (every head warm: classifier needs its encoder history)", "",
         "## Spec table", "", "| spec | what | target | measured | verdict | model | provenance |", "|---|---|---|---|---|---|---|"]
    for r in rep["spec_table"]:
        L.append(f"| {r['spec']} | {r['what']} | {r['op']} {fmt(r['target'])} | {fmt(r['measured'])} {r['unit']} | "
                 f"**{r['verdict']}** | {r['model']} | {r['provenance']} |")
    L += ["", "## Metric definitions", "", "```", M.__doc__.strip(), "```", ""]
    for fam in ("gbm", "nn"):
        s = rep["test"].get(fam)
        if not s:
            continue
        L += [f"## Test detail: {rep['families'][fam]}", ""]
        cl = s["classification"]
        L.append("Per-class F1: " + ", ".join(f"{k} {v:.3f}" for k, v in cl["per_class_f1"].items()) +
                 f"; accuracy {cl['accuracy']:.3f}; windows {cl['n_windows']} on {cl['n_bearings']} bearings")
        L += ["", "Confusion (rows true, cols pred; order " + ", ".join(cl["confusion"]["labels"]) + "):", "", "```"]
        L += [" ".join(f"{x:7d}" for x in row) for row in cl["confusion"]["matrix"]]
        L += ["```", ""]
        st = s["stage"]
        L.append(f"Stage error histogram {st['error_hist']} (n={st['n_windows']}); within +/-1 {st['within1_frac']:.4f}; "
                 f"MAE {st['mae']:.3f}; windows with error >= 2: {st.get('windows_error_ge2')} on {st.get('bearings_with_error_ge2')} bearings")
        ca = s["caught"]
        L.append(f"Caught by stage 3: {ca['caught_by_stage3']}/{ca['rtf_bearings']}; caught-at-stage histogram {ca.get('caught_at_stage_hist')}; "
                 f"per class {ca.get('per_class_by_stage3')}; never caught {ca['never_caught']}")
        ru = s["rul"]
        L.append(f"RUL: S7 {ru['S7_mape_pct']:.2f} % over {ru['n_bearings']} bearings / {ru['n_windows']} windows; per-bearing "
                 f"median {ru['per_bearing_summary'].get('median', float('nan')):.2f}, p90 {ru['per_bearing_summary'].get('p90', float('nan')):.2f}, "
                 f"max {ru['per_bearing_summary'].get('max', float('nan')):.2f}, share of bearings <= 15 %: {ru['per_bearing_summary'].get('frac_le_15', float('nan')):.2f}; "
                 f"RMSE {ru['rmse_hours']:.2f} h; full-interval MAPE {ru['alt_mape_full_degradation_pct']:.2f} %; "
                 f"true-onset (stricter) MAPE {ru.get('alt_mape_true_onset_pct', float('nan')):.2f} %; per rpm {ru.get('per_rpm_mode')}; per class {ru.get('per_fault_class')}")
        L += ["", "RUL error vs true elapsed fraction of the degradation interval:", "", "| u | MAPE % | median APE % | bias % | n |", "|---|---|---|---|---|"]
        for k, v in ru["error_vs_life_fraction"].items():
            L.append(f"| {k} | {v['mape_pct']:.2f} | {v['median_ape_pct']:.2f} | {v['bias_pct']:+.2f} | {v['n']} |")
        L.append("")
    L += ["## Robustness (fresh runs never used for fitting or selection; informational, no verdict)", "",
          "| set | knobs changed | model | macro-F1 | stage max | within+/-1 | caught by st.3 | S7 MAPE % | %-of-life pts |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, by in rep["robustness"].items():
        for fam, s in by.items():
            L.append(f"| {name} | {ROBUST_SETS[name][2]} | {fam} | {s['classification']['macro_f1']:.3f} | {s['stage']['max_abs_error']} | "
                     f"{s['stage']['within1_frac']:.4f} | {s['caught']['caught_by_stage3']}/{s['caught']['rtf_bearings']} | "
                     f"{s['rul']['S7_mape_pct']:.2f} | {s['rul']['alt_percent_of_life_error_pts']:.2f} |")
    L += ["", "## Latency (live, this host, ONNX Runtime, 1 thread, batch 1)", "", "```", json.dumps(rep["latency"], indent=1), "```", "",
          "## Previous public-data results (PPR v2) -- [MEASURED on XJTU-SY / IMS / MaFaulDa], stored report", "",
          "```", json.dumps(rep.get("public_v2", {}), indent=1), "```", "",
          "## Generator difficulty knobs", "", "```", json.dumps(rep["data"]["main"].get("knobs", {}), indent=1), "```"]
    path.write_text("\n".join(L), encoding="utf-8")


def public_v2() -> dict:
    if not REAL_V2_REPORT.exists():
        return {}
    r = json.loads(REAL_V2_REPORT.read_text())
    m = r.get("metrics", r)
    v = r.get("verdicts") or m.get("verdicts")
    return {"source": REAL_V2_REPORT.name, "provenance": "[MEASURED on XJTU-SY / IMS / MaFaulDa public bearing data]",
            "verdicts": v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2_eval")
    ap.add_argument("--no-latency", action="store_true")
    a = ap.parse_args(argv)
    df = load_main()
    test = df[df["split"] == "test"]
    gi, ni = latest_index("gbm"), latest_index("nn")
    if gi is None:
        raise SystemExit("no saved GBM predictions: run `python -m aiengine.up2_train predict` first")
    sel = json.loads((up2.RUN_DIR / "selection.json").read_text()) if (up2.RUN_DIR / "selection.json").exists() else {}
    min_window = int(sel.get("cls", {}).get("T", 25)) - 1
    fams = {"gbm": f"DEPLOYED v3 heads: {sel.get('cls', {}).get('kind', '?')} classifier + {sel.get('rul', {}).get('kind', '?')} RUL/{sel.get('z', {}).get('kind', '?')} stage (gradient-boosted trees)",
            "nn": "TFT (class) + N-HiTS (HI forecast -> RUL, rul.py) -- spec-named models, trained on the same runs"}
    rep = {"meta": meta({"tool": "aiengine.up2_eval"}), "families": fams, "min_window": min_window,
           "pred_index": {"gbm": gi.get("index_file"), "nn": ni.get("index_file") if ni else None},
           "data": {"main": df.attrs.get("provenance", {}), "split_check": up2.split_check(df)},
           "test": {}, "robustness": {}}
    for fam, idx in (("gbm", gi), ("nn", ni)):
        if idx is None:
            continue
        cp, rp = load_preds(idx, "test")
        rep["test"][fam] = score_set(cp, rp, test, min_window)
        for name in ROBUST_SETS:
            if name in idx["files"]:
                r = load_robust(name)
                cp, rp = load_preds(idx, name)
                rep["robustness"].setdefault(name, {})[fam] = score_set(cp, rp, r, min_window)
    rep["latency"] = {} if a.no_latency else latencies(test)
    table = spec_rows(rep["test"]["gbm"], rep["latency"], "v3 GBM (deployed)")
    if "nn" in rep["test"]:
        table += spec_rows(rep["test"]["nn"], rep["latency"], "TFT + N-HiTS (synth_v3)")
    lat = rep["latency"]
    for key, model in (("v3_cls_onnx", "v3 GBM classifier ONNX (deployed)"), ("v3_nn_tft_onnx", "TFT ONNX (synth_v3)")):
        if key in lat:
            table.append({"spec": "S8 / ICS2", "what": "classification latency per window, p95 (1 thread, batch 1)", "op": "<",
                          "target": SPEC_TARGETS["S8_tft_latency_ms"], "measured": lat[key]["p95_ms"], "unit": "ms",
                          "verdict": verdict("<", SPEC_TARGETS["S8_tft_latency_ms"], lat[key]["p95_ms"]),
                          "provenance": "[MEASURED] live on this host (synthetic inputs)", "model": model})
    for key, model in (("v3_engine_step", "v3 engine step (features+cls+RUL)"), ("v3_nn_engine_step", "TFT+N-HiTS engine step")):
        if key in lat and lat[key]:
            table.append({"spec": "IS1 (ICS leg)", "what": "ICS host leg per bearing-window, p95", "op": "<",
                          "target": SPEC_TARGETS["IS1_e2e_latency_ms"], "measured": lat[key]["p95_ms"], "unit": "ms",
                          "verdict": verdict("<", SPEC_TARGETS["IS1_e2e_latency_ms"], lat[key]["p95_ms"]) + " (leg only)",
                          "provenance": "[MEASURED] live on this host", "model": model})
    rep["spec_table"] = table
    rep["public_v2"] = public_v2()
    ts = now_tag()
    jp, mp = REPORTS_DIR / f"up2_spec_check_{ts}.json", REPORTS_DIR / f"up2_spec_check_{ts}.md"
    jp.write_text(json.dumps(rep, indent=1, default=float))
    write_md(rep, mp)
    print("[SYNTHETIC] UP2 spec check (synth_v3 test runs; robustness and public-data rows in the report)")
    print(f"{'spec':12s} {'target':>10s} {'measured':>12s}  verdict  model")
    for r in table:
        print(f"{r['spec']:12s} {r['op']:>2s} {fmt(r['target']):>7s} {fmt(r['measured']):>10s} {r['unit']:6s} {r['verdict']:8s} {r['model']}  {r['provenance']}")
    print("->", jp, "\n->", mp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
