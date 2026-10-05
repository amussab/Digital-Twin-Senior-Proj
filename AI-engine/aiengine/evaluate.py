"""Spec scoring on held-out TEST units (DESIGN.md sections 1 and 7).

S7   RUL MAPE, PRE-REGISTERED metric (bearing-averaged over the degradation window
     [t_onset, T_fail - 0.1 (T_fail - t_onset)]); extras reported, never substituted:
     RMSE (h), PHM-2012 score, percent-of-life error, MAPE over the full degradation window.
S8   TFT single-window latency p50/p95/max (batch 1), PyTorch and ONNX Runtime.
IS3  macro-F1 over {healthy, outer_race, inner_race, ball, cage}, per-class F1, confusion,
     stage error (fraction within +/-1, mean, max), caught-by-stage-3.
IS1  ICS leg only (features + TFT + N-HiTS + RUL for one window) -- the network/dashboard legs
     are measured by the backend.
Every report carries provenance on its face: datasets (real vs SYNTHETIC), counts, seed, git
commit, command line.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

from . import features as feat
from . import infer, rul, train
from .config import MODELS_DIR, REPORTS_DIR, SEED, SPEC_TARGETS

CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]
DETECTION_PERSISTENCE = 3
STAGE_WITHIN1_TARGET = 0.95     # AI-test convention for "stage error <= 1" (fraction of windows)


# --------------------------------------------------------------------------- classification

def classification_metrics(preds: pd.DataFrame, truth: pd.DataFrame) -> dict:
    j = preds.merge(truth[["unit_id", "window_index", "observable_class", "fault_class", "dataset",
                           "health_stage", "run_to_failure"]], on=["unit_id", "window_index"])
    if j.empty:
        return {"n": 0}
    y, p = j["observable_class"].astype(str).to_numpy(), j["pred_class"].to_numpy()
    present = [c for c in CLASSES if (y == c).any()]
    trained = [c for c in CLASSES if f"p_{c}" in preds.columns]
    # Testable = classes with BOTH training support (a model output channel) and test support.
    testable = [c for c in present if c in trained]
    m = np.isin(y, testable)
    out = {
        "n": int(len(j)),
        "macro_f1": float(f1_score(y[m], p[m], labels=testable, average="macro", zero_division=0)),
        "macro_f1_testable_classes": testable,
        "macro_f1_n": int(m.sum()),
        "macro_f1_all5": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
        "trained_classes": trained,
        "untestable_note": {c: ("test support %d windows, 0 training support (no output channel)" % int((y == c).sum())
                                if (y == c).any() else "0 test support" + ("" if c in trained else ", 0 training support"))
                            for c in CLASSES if c not in testable},
        "classes_present": present,
        "accuracy": float((y == p).mean()),
        "per_class_f1": {c: float(f1_score(y, p, labels=[c], average="macro", zero_division=0)) for c in present},
        "per_class_support": {c: int((y == c).sum()) for c in present},
        "confusion": {"labels": CLASSES, "matrix": confusion_matrix(y, p, labels=CLASSES).tolist()},
    }
    f = y != "healthy"
    if f.any():
        out["missed_fault_rate"] = float((p[f] == "healthy").mean())
    if (~f).any():
        out["false_alarm_rate"] = float((p[~f] != "healthy").mean())
    out["per_dataset"] = {}
    for ds, g in j.groupby("dataset"):
        yy, pp = g["observable_class"].astype(str).to_numpy(), g["pred_class"].to_numpy()
        pres = [c for c in CLASSES if (yy == c).any()]
        out["per_dataset"][ds] = {
            "n": int(len(g)), "classes_present": pres,
            "macro_f1": float(f1_score(yy, pp, labels=pres, average="macro", zero_division=0)),
            "accuracy": float((yy == pp).mean()),
            "recall": {c: float((pp[yy == c] == c).mean()) for c in pres},
            "macro_f1_all5": float(f1_score(yy, pp, labels=CLASSES, average="macro", zero_division=0)),
            "per_class_f1": {c: float(f1_score(yy, pp, labels=[c], average="macro", zero_division=0)) for c in pres},
            "confusion": {"labels": CLASSES, "matrix": confusion_matrix(yy, pp, labels=CLASSES).tolist()},
        }
    return out


def caught_by_stage3(preds: pd.DataFrame, truth: pd.DataFrame) -> dict:
    j = preds.merge(truth[["unit_id", "window_index", "fault_class", "health_stage", "run_to_failure",
                           "dataset"]], on=["unit_id", "window_index"])
    rows = []
    for uid, g in j.groupby("unit_id"):
        fc = str(g["fault_class"].iloc[0])
        if fc not in CLASSES or fc == "healthy":
            continue
        g = g.sort_values("window_index")
        ok = (g["pred_class"].to_numpy() == fc)
        streak, at = 0, None
        for i, v in enumerate(ok):
            streak = streak + 1 if v else 0
            if streak >= DETECTION_PERSISTENCE:
                at = i
                break
        rtf = bool(g["run_to_failure"].iloc[0])
        stage = int(g["health_stage"].iloc[at]) if (at is not None and rtf) else None
        rows.append({"unit_id": uid, "dataset": g["dataset"].iloc[0], "fault_class": fc,
                     "run_to_failure": rtf, "caught": at is not None,
                     "caught_at_window": int(g["window_index"].iloc[at]) if at is not None else None,
                     "caught_at_stage": stage,
                     "caught_by_stage3": (stage is not None and stage <= 3) if rtf else None})
    fr = pd.DataFrame(rows)
    out = {"per_unit": rows}
    if len(fr):
        r = fr[fr["run_to_failure"]]
        out["rtf_units"] = int(len(r))
        out["caught_by_stage3_frac"] = float(r["caught_by_stage3"].astype(bool).mean()) if len(r) else float("nan")
        s = fr[~fr["run_to_failure"]]
        out["seeded_records"] = int(len(s))
        out["seeded_caught_frac"] = float(s["caught"].mean()) if len(s) else float("nan")
    return out


# --------------------------------------------------------------------------- RUL

def phm2012_score(pred: np.ndarray, true: np.ndarray) -> float:
    """PHM-2012 challenge score: Er = 100 (ActRUL - RUL)/ActRUL; A = exp(-ln(0.5) Er/5) for
    Er <= 0, exp(+ln(0.5) Er/20) for Er > 0; score = mean A (1 = perfect)."""
    er = 100 * (true - pred) / np.maximum(true, 1e-9)
    a = np.where(er <= 0, np.exp(-np.log(0.5) * er / 5), np.exp(np.log(0.5) * er / 20))
    return float(np.mean(a))


def rul_metrics(scored: pd.DataFrame) -> dict:
    mape, per = train.s7_mape(scored)
    mape_full, per_full = train.s7_mape(scored, window="in_deg_full")
    d = scored[scored["in_deg"] & (scored["rul_true"] > 0)]
    out = {"S7_mape_pct": mape, "per_bearing_mape_pct": per, "mape_full_degradation_pct": mape_full,
           "per_bearing_mape_full_pct": per_full, "n_windows": int(len(d)), "n_bearings": len(per)}
    if len(d):
        err = d["rul_pred"].to_numpy() - d["rul_true"].to_numpy()
        out["rmse_hours"] = float(np.sqrt(np.mean(err ** 2)))
        out["phm2012_score"] = phm2012_score(d["rul_pred"].to_numpy(), d["rul_true"].to_numpy())
        life = d["life_hours"].to_numpy()
        out["percent_of_life_error_pts"] = float(np.mean(np.abs(err) / life) * 100)
        for col, key in (("rul_trend", "trend_only"), ("rul_onset", "onset_only")):
            alt = d.assign(rul_pred=np.where(np.isfinite(d[col]), d[col], d["rul_pred"]))
            out[f"S7_mape_{key}_pct"] = train.s7_mape(alt)[0]
        out["per_dataset"] = {ds: train.s7_mape(g)[0] for ds, g in scored.groupby("dataset")}
    st = scored[scored["health_stage"] >= 1]
    if len(st):
        e = np.abs(st["stage_pred"].to_numpy() - st["health_stage"].to_numpy())
        out["stage_within1_frac"] = float((e <= 1).mean())
        out["stage_mae"] = float(e.mean())
        out["stage_max_error"] = int(e.max())
        out["stage_n"] = int(len(e))
    return out


# --------------------------------------------------------------------------- latency

def _pct(a: list[float]) -> dict:
    a = np.asarray(a)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()), "mean_ms": float(a.mean()), "n": int(len(a))}


def latency(tft_model, nhits_model, sample: pd.DataFrame, n: int = 300, threads: int = 1) -> dict:
    """Single-window (batch 1) latency. torch threads = `threads` to mimic one host core."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    out = {"torch_threads": threads}
    try:
        cols, sc, L = train.tft_layout(tft_model)
        enc, dec = infer.tft_tensors(sample[cols].to_numpy(np.float64), cols, sc, L)
        w = train.tft_wrapper(tft_model)
        times = []
        with torch.no_grad():
            for i in range(n + 20):
                e, d = torch.as_tensor(enc[i % len(enc)][None]), torch.as_tensor(dec[i % len(dec)][None])
                t0 = time.perf_counter()
                w(e, d)
                if i >= 20:
                    times.append((time.perf_counter() - t0) * 1000)
        out["tft_torch"] = _pct(times)
        ncols, nsc, NL, NH = train.nhits_layout(nhits_model)
        ne, nd = infer.nhits_tensors(sample[ncols].to_numpy(np.float64), ncols, nsc, NL, NH)
        nw = train.nhits_wrapper(nhits_model)
        times = []
        with torch.no_grad():
            for i in range(n + 20):
                t0 = time.perf_counter()
                nw(torch.as_tensor(ne[i % len(ne)][None]), torch.as_tensor(nd[i % len(nd)][None]))
                if i >= 20:
                    times.append((time.perf_counter() - t0) * 1000)
        out["nhits_torch"] = _pct(times)
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            for name, path, a, b in (("tft_onnx", MODELS_DIR / "tft.onnx", enc, dec),
                                     ("nhits_onnx", MODELS_DIR / "nhits.onnx", ne, nd)):
                if not path.exists():
                    continue
                s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
                names = [i.name for i in s.get_inputs()]
                times = []
                for i in range(n + 20):
                    feed = {"encoder_cont": a[i % len(a)][None], "decoder_cont": b[i % len(b)][None]}
                    feed = {k: v for k, v in feed.items() if k in names}
                    t0 = time.perf_counter()
                    s.run(None, feed)
                    if i >= 20:
                        times.append((time.perf_counter() - t0) * 1000)
                out[name] = _pct(times)
        except ImportError:
            pass
        # Full ICS engine step (features + both models + RUL), via the reference engine.
        try:
            from .engine import HybridEngine
            if (MODELS_DIR / "model_contract.json").exists():
                eng = HybridEngine(MODELS_DIR, threads=threads)
                out["engine_step"] = eng.benchmark(sample, n=min(n, 200))
        except Exception as exc:  # noqa: BLE001
            out["engine_step_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        torch.set_num_threads(prev)
    return out


# --------------------------------------------------------------------------- report

def verdicts(m: dict) -> list[dict]:
    T = SPEC_TARGETS
    lat = m.get("latency", {})
    s8 = lat.get("tft_onnx", lat.get("tft_torch", {})).get("p95_ms", float("nan"))
    eng = lat.get("engine_step", {}).get("p95_ms", float("nan"))
    rows = [
        ("S7", "RUL MAPE on held-out degradation window (N-HiTS)", "<=", T["S7_rul_mape_pct"],
         m.get("rul", {}).get("S7_mape_pct", float("nan")), "%"),
        ("S8", "TFT classification latency per window, p95 single-window", "<", T["S8_tft_latency_ms"], s8, "ms"),
        ("IS3a", "macro-F1 over the testable classes (train AND test support; primary)", ">=", T["IS3_macro_f1"],
         m.get("classification", {}).get("macro_f1", float("nan")), ""),
        ("IS3a-5", "macro-F1 over all 5 classes (a class with no output channel scores F1 0)", ">=", T["IS3_macro_f1"],
         m.get("classification", {}).get("macro_f1_all5", float("nan")), ""),
        ("IS3b", "stage error <= 1, literal reading: MAX abs stage error over test windows", "<=",
         T["IS3_stage_error_max"], m.get("rul", {}).get("stage_max_error", float("nan")), ""),
        ("IS3c", "run-to-failure faults caught (3 consecutive correct) by stage 3", ">=", 1.0,
         m.get("detection", {}).get("caught_by_stage3_frac", float("nan")), ""),
        ("IS1 (ICS leg)", "features + TFT + N-HiTS + RUL per window, p95", "<", T["IS1_e2e_latency_ms"], eng, "ms"),
    ]
    out = []
    for sid, desc, op, target, val, unit in rows:
        if val is None or not np.isfinite(float(val)):
            status = "NOT MEASURED"
        else:
            val = float(val)
            ok = {"<=": val <= target, "<": val < target, ">=": val >= target}[op]
            status = "PASS" if ok else "FAIL"
        out.append({"spec": sid, "requirement": desc, "op": op, "target": target, "measured": val,
                    "unit": unit, "status": status})
    return out


def provenance_banner(datasets: list[str]) -> str:
    real = [d for d in datasets if d != "synthetic_rig"]
    if not real:
        return ("> **SYNTHETIC DATA ONLY.** Every number below comes from `synthetic_rig`, generated "
                "data. It proves the pipeline runs end to end; it is NOT a measurement of any real "
                "bearing and must not be quoted as one.")
    names = {"xjtu_sy": "XJTU-SY", "ims": "IMS", "mafaulda": "MaFaulDa"}
    s = (f"> **MEASURED on {' / '.join(names.get(d, d) for d in real)} public bearing datasets, not the team rig.** "
         f"[MEASURED on {', '.join(real)}] -- scored on held-out test bearings never used for training, "
         "early stopping, model selection or calibration. These are measurements of THOSE datasets' "
         "bearings; no rig data exists yet.")
    if "synthetic_rig" in datasets:
        s += " Rows tagged `synthetic_rig` are SYNTHETIC."
    return s


def write_report(metrics: dict, meta: dict, tag: str = "") -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"spec_check_{stamp}{('_' + tag) if tag else ''}"
    vs = verdicts(metrics)
    payload = {"generated_utc": stamp, "provenance": meta, "verdicts": vs, "metrics": metrics}
    jp = REPORTS_DIR / f"{name}.json"
    train.write_json(jp, payload)

    def f(v, unit=""):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.3f}{unit}" if unit == "" else f"{v:.1f} {unit}"

    L = [f"# AI-engine spec check ({stamp})", "", provenance_banner(meta["datasets"]), ""]
    if metrics.get("disclosure"):
        L += [f"> **{metrics['disclosure']}**", ""]
    L += [
         f"- Seed `{meta['seed']}`, git `{meta['git_commit']}`{' (+uncommitted changes)' if meta.get('git_dirty') else ''}",
         f"- Command: `{meta['command']}`", f"- Run dir: `{meta['run']}`",
         f"- Host: {meta['host']}", "", "## Verdicts", "",
         "| Spec | Requirement | Target | Measured | Verdict |", "|---|---|---|---|---|"]
    for v in vs:
        L.append(f"| {v['spec']} | {v['requirement']} | {v['op']} {v['target']} {v['unit']} | "
                 f"{f(v['measured'], v['unit'])} | **{v['status']}** |")
    r = metrics.get("rul", {})
    L += ["", "## S7 -- RUL detail (pre-registered metric, DESIGN.md section 7)", "",
          f"- S7 MAPE (bearing-averaged, degradation window minus last 10%): **{f(r.get('S7_mape_pct'), '%')}**",
          f"- MAPE over the full degradation window: {f(r.get('mape_full_degradation_pct'), '%')}",
          f"- RMSE: {f(r.get('rmse_hours'), 'h')}; PHM-2012 score: {f(r.get('phm2012_score'))}; "
          f"percent-of-life error: {f(r.get('percent_of_life_error_pts'), 'pts')}",
          f"- Estimator components: trend-only {f(r.get('S7_mape_trend_only_pct'), '%')}, "
          f"onset-only {f(r.get('S7_mape_onset_only_pct'), '%')}",
          f"- Stage (IS3b): max abs error {r.get('stage_max_error', 'n/a')}, mean abs error {f(r.get('stage_mae'))}, "
          f"fraction within +/-1 {f(r.get('stage_within1_frac'))} over {r.get('stage_n')} windows", "", "| Test bearing | MAPE (S7 window) | MAPE (full window) |",
          "|---|---|---|"]
    for k, v in r.get("per_bearing_mape_pct", {}).items():
        L.append(f"| {k} | {v:.1f} % | {r.get('per_bearing_mape_full_pct', {}).get(k, float('nan')):.1f} % |")
    c = metrics.get("classification", {})
    L += ["", "## IS3 -- classification", "",
          f"- Test windows: {c.get('n')}. **Primary IS3a macro-F1 over the testable classes "
          f"{c.get('macro_f1_testable_classes')} ({c.get('macro_f1_n')} windows): {f(c.get('macro_f1'))}**",
          f"- macro-F1 over all 5 classes: {f(c.get('macro_f1_all5'))}. Classes the model was trained on: "
          f"{c.get('trained_classes')}. Not testable: {c.get('untestable_note')}",
          f"- Missed-fault rate {f(c.get('missed_fault_rate'))}, false-alarm rate {f(c.get('false_alarm_rate'))}",
          "", "| Class | F1 | Support |", "|---|---|---|"]
    for k, v in c.get("per_class_f1", {}).items():
        L.append(f"| {k} | {v:.3f} | {c['per_class_support'][k]} |")
    L += ["", "Confusion (rows = true, cols = predicted, order " + ", ".join(CLASSES) + "):", "", "```"]
    for lab, rowv in zip(CLASSES, c.get("confusion", {}).get("matrix", [])):
        L.append(f"{lab:>11} " + " ".join(f"{x:>7d}" for x in rowv))
    L += ["```", "", "Per dataset (source-confounding check):", "",
          "| Dataset | n | classes | macro-F1 | recall per class |", "|---|---|---|---|---|"]
    for ds, v in c.get("per_dataset", {}).items():
        rec = ", ".join(f"{k} {x:.2f}" for k, x in v["recall"].items())
        L.append(f"| {ds} | {v['n']} | {len(v['classes_present'])} | {v['macro_f1']:.3f} | {rec} |")
    for ds, v in c.get("per_dataset", {}).items():
        if "confusion" in v:
            L += ["", f"Confusion, {ds} only (rows = true, cols = predicted, order " + ", ".join(CLASSES) + "):", "", "```"]
            for lab, rowv in zip(CLASSES, v["confusion"]["matrix"]):
                L.append(f"{lab:>11} " + " ".join(f"{x:>7d}" for x in rowv))
            L.append("```")
    sc = metrics.get("source_check")
    if sc:
        L += ["", "Training-class sources (windows per class per dataset, train split):", "", "```",
              json.dumps(sc, indent=1), "```"]
    d = metrics.get("detection", {})
    L += ["", "## Detection", "",
          f"- Run-to-failure test faults caught by stage 3: {f(d.get('caught_by_stage3_frac'))} over {d.get('rtf_units')} unit(s)",
          f"- Seeded-fault test records called correctly (3 consecutive windows): {f(d.get('seeded_caught_frac'))} "
          f"over {d.get('seeded_records')} record(s)"]
    lat = metrics.get("latency", {})
    L += ["", f"## Latency (single window, batch 1, {lat.get('torch_threads')} thread)", "",
          "| Path | p50 | p95 | max |", "|---|---|---|---|"]
    for k in ("tft_torch", "tft_onnx", "nhits_torch", "nhits_onnx", "engine_step"):
        if k in lat:
            L.append(f"| {k} | {lat[k]['p50_ms']:.2f} ms | {lat[k]['p95_ms']:.2f} ms | {lat[k]['max_ms']:.2f} ms |")
    for k, v in metrics.get("extra_tracks", {}).items():
        L += ["", f"## Track: {k}", "", "```", json.dumps(v, indent=1, default=str)[:4000], "```"]
    L += ["", "## Counts", "", "```", json.dumps(meta.get("counts", {}), indent=1), "```", "",
          f"Full JSON: `reports/{jp.name}`"]
    mp = REPORTS_DIR / f"{name}.md"
    text = "\n".join(L) + "\n"
    mp.write_text(text, encoding="utf-8")
    (REPORTS_DIR / "latest.md").write_text(text, encoding="utf-8")
    return jp, mp


def source_check(df_train: pd.DataFrame) -> dict:
    d = df_train[df_train["cls_usable"]]
    return {c: {ds: int(n) for ds, n in g.groupby("dataset").size().items()}
            for c, g in d.groupby("observable_class")}


def host_info() -> str:
    return f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, torch {torch.__version__}"


def score(run_dir: Path, df: pd.DataFrame, test_split: str = "test", tft_ckpt=None, nhits_ckpt=None,
          calib_file=None, latency_n: int = 300) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    tft = train.load_model(Path(tft_ckpt or run_dir / "tft.ckpt"))
    nh = train.load_model(Path(nhits_ckpt or run_dir / "nhits.ckpt"))
    calib = rul.RULCalibration.from_dict(json.loads(Path(calib_file or run_dir / "rul_calibration.json").read_text()))
    test = df[df["split"] == test_split]
    preds = train.tft_predict(tft, test)
    m = {"classification": classification_metrics(preds, test), "detection": caught_by_stage3(preds, test)}
    recs = train.rul_records(nh, test, preds)
    scored = train.apply_rul(calib, recs) if len(recs) else pd.DataFrame()
    m["rul"] = rul_metrics(scored) if len(scored) else {}
    if latency_n:
        sample = test[test["unit_id"] == test.groupby("unit_id").size().idxmax()].sort_values("window_index")
        m["latency"] = latency(tft, nh, sample, n=latency_n)
    return m, preds, scored
