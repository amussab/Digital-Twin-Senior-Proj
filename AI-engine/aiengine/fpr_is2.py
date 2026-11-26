"""FPR IS2 harness: physics-AI residual (wording v2) on rig_anchor simulated units.  [SIMULATION; rig-anchored]

    python -m aiengine.fpr_is2 predict --split val  [--contract v5_node]   # deployed NodeEngine (AI heads + feature physics twin), saves preds
    python -m aiengine.fpr_is2 select  [--contract v5_node]                # VALIDATION only: thresholds (up2d_is2.choose rule), controls,
                                                                           #   writes reports/fpr_is2_frozen_<contract>_<ts>.json  (= pre-registration)
    python -m aiengine.fpr_is2 test --split rig_anchor_test2[:test] --contract v6_node [--data file.parquet]
                                                                           # ONE scoring pass; refused without a valid frozen file, refused a 2nd time
    python -m aiengine.fpr_is2 eval --split val|<section>[:split] [--contract c]   # re-score SAVED predictions (never touches a model)

--split: `val` / `train` / `test` = those unit splits of the registered section `rig_anchor` (data/rig_anchor/rig_anchor.parquet);
`SECTION` or `SECTION:split` = any other section registered in splits.json (default split `test`, data data/rig_anchor/SECTION.parquet
unless --data). The units must be registered in splits.json BEFORE use.

Metric = IS2 wording v2 of aiengine.up2d_is2 (its pure functions are REUSED, not copied):
  e_w = |RUL_phys,w - RUL_AI,w| / L_ref  for the FAULTED bearing of each run-to-failure unit, TRUE stage 1..3, window_index >= 24 (AI
  warm-up), RUL_true > 0; per bearing median / p95; MED = median over bearings of the median, P95 = 95th pct over bearings of the p95;
  PASS iff MED <= T_MED and P95 <= T_P95, thresholds FROZEN from validation by up2d_is2.choose (1.1 x max(AI value, exact-AI floor),
  rounded up to 0.01) and all six wrong-AI controls FAIL on validation.
  L_ref = ISO 281 L10h at the simulation-equivalent duty P/C = 0.5 (rig_anchor.scale; 74.1 h @1800, 37.0 h @3600 rpm) [ASSUMED duty].

HONEST LIMITS (printed into every report):
  * COMMON MODE: the physics twin and the AI RUL head read the SAME node features of the same window (and the same commissioning
    baseline). Agreement measures consistency of two readings of one signal, not independent physical confirmation.
  * The data are simulated trajectories (endpoints resampled from rig windows, 1 bearing per fault, curve shape assumed) [SIMULATION].
  * The AI RUL used is the head's post-processed output BEFORE the dashboard display gate (stage >= 2 and faulted); the twin's RUL is
    its trend RUL when the fit is significant, else the rated-life prior (share reported).
  * A pass proves nothing about real bearings; v1 literal (max <= 5 %) is reported next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits, up2d_is2 as U
from .config import REPORTS_DIR, ROOT
from .rig_anchor import constants as K
from .rig_anchor import generator as G
from .rig_anchor import scale

PRED_DIR = REPORTS_DIR / "fpr_is2_preds"
DEFAULT_CONTRACT = "v5_node"
WARM = U.WARM_MIN_WINDOW
TAG = K.PROVENANCE
LIMITS_TEXT = ("COMMON MODE: twin and AI read the same node features and the same baseline, so agreement is consistency of two readings of "
               "one signal, not independent confirmation. " + K.LIMITS +
               " L_ref uses an ASSUMED simulation-equivalent duty (P/C = 0.5). The thresholds are tied to this simulated population.")


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- units / splits

def resolve_split(spec: str, data: str | None = None) -> dict:
    """-> {"section", "split", "data"}; the section must be registered in splits.json."""
    if spec in K.SPLITS:
        section, split = K.SPLIT_SECTION, spec
    elif ":" in spec:
        section, split = spec.split(":", 1)
    else:
        section, split = spec, "test"
    sec = splits.load()["sections"].get(section)
    if sec is None or "units" not in sec:
        raise SystemExit(f"split section {section!r} is not registered in splits.json: refusing")
    path = Path(data) if data else ROOT / "data" / "rig_anchor" / ("rig_anchor.parquet" if section == K.SPLIT_SECTION else f"{section}.parquet")
    return {"section": section, "split": split, "data": path, "units": sec["units"]}


def load_units(r: dict) -> pd.DataFrame:
    """Rows of the registered units of the chosen split (both bearings)."""
    if not r["data"].exists():
        raise SystemExit(f"data file missing: {r['data']} (git-ignored; rebuild with python -m aiengine.rig_anchor.build or pass --data)")
    df = pd.read_parquet(r["data"])
    want = {u for u, s in r["units"].items() if s == r["split"]}
    df = df[df["rig_unit"].isin(want)]
    if df.empty:
        raise SystemExit(f"no rows of split {r['split']!r} of section {r['section']!r} in {r['data']}")
    if (df["rig_unit"].map(r["units"]) != df["split"]).any() and r["section"] == K.SPLIT_SECTION:
        raise SystemExit("data split column disagrees with the registered split: refusing")
    return df


# --------------------------------------------------------------------------- prediction (deployed engine, table time base)

def predict_units(df: pd.DataFrame, contract: str = DEFAULT_CONTRACT, units: int | None = None) -> pd.DataFrame:
    """Run the deployed NodeEngine (ONNX heads + post-processing + FeaturePhysicsTwin, legacy 24-window baseline, table time base)
    over the faulted run-to-failure units. One row per window of the faulted bearing (bearing 1)."""
    from .node_adapter import NodeEngine
    eng = NodeEngine(f"model_contract_{contract}.json", policy=None, time_base="table")
    rtf = df[df["run_to_failure"] & df["unit_id"].str.endswith(":B1")]
    ids = sorted(rtf["rig_unit"].unique())[: units or None]
    rows = []
    for uid in ids:
        u = df[df["rig_unit"] == uid]
        X = G.window_matrix(u)
        b1 = u[u["unit_id"].str.endswith(":B1")].sort_values("window_index")
        rpm = float(b1["rpm"].iloc[0])
        eng.reset()
        for i, x in enumerate(X):
            o = eng.predict({"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": rpm, "window_id": i,
                             "features": [float(v) for v in x]})
            b = o["bearings"][0]
            if b.get("status") != "ok":
                continue
            rows.append({"unit_id": f"{uid}:B1", "rig_unit": uid, "window_index": int(b1["window_index"].iloc[i]),
                         "rul_pred_h": b["ai_rul_head_hours"], "phys_rul_h": b["physics_rul_head_hours"],
                         "phys_source": b["physics_source"], "miner_d": b["miner_d"], "phys_hi": b["physics_health_index"],
                         "stage_pred": b["health_stage"], "pred_class": b["fault_class"], "rul_valid": bool(b["rul_valid"])})
    return pd.DataFrame(rows)


def scored(df: pd.DataFrame, preds: pd.DataFrame, stages=(1, 2, 3), min_window: int = WARM) -> pd.DataFrame:
    """Join predictions with the truth; same selection as up2d_is2.scored_windows, plus the columns its pure functions read."""
    t = df[df["run_to_failure"] & df["unit_id"].str.endswith(":B1")][["unit_id", "window_index", "rpm", "true_stage", "rul_hours", "life_hours"]]
    t = t[t["true_stage"].isin(stages) & (t["rul_hours"] > 0) & (t["window_index"] >= min_window)]
    m = t.merge(preds, on=["unit_id", "window_index"], how="inner")
    m = m[m["rul_pred_h"].notna() & m["phys_rul_h"].notna()].copy()
    m["rpm_mode"] = m["rpm"].astype(float)
    m["L_ref"] = [scale.l_ref_hours(r) for r in m["rpm_mode"]]
    m["phys_conf"] = (m["phys_source"] == "trend").astype(float)
    m["diff_h"] = (m["phys_rul_h"] - m["rul_pred_h"]).abs()
    m["e_lref"] = m["diff_h"] / m["L_ref"]
    m["e_v1"] = m["diff_h"] / m["rul_pred_h"].where(m["rul_pred_h"] > 0)
    return m.reset_index(drop=True)


def life_scale(df: pd.DataFrame) -> dict:
    """simulated life / L_ref of the run-to-failure units (the scale check, recomputed on the scored population)."""
    b = df[df["run_to_failure"] & df["unit_id"].str.endswith(":B1")]
    rows = []
    for uid, g in b.groupby("unit_id"):
        rpm = float(g["rpm"].iloc[0])
        on = g[g["u_true"] >= 0]["t_hours"].min()
        life = float(g["life_hours"].iloc[0])
        lr = scale.l_ref_hours(rpm)
        rows.append({"life_over_Lref": life / lr, "onset_to_failure_over_Lref": (life - on) / lr, "rpm": rpm})
    x = pd.DataFrame(rows)
    return {"units": int(len(x)), "life_over_Lref_median": float(x["life_over_Lref"].median()),
            "onset_to_failure_over_Lref_median": float(x["onset_to_failure_over_Lref"].median()),
            "onset_to_failure_over_Lref_min_max": [float(x["onset_to_failure_over_Lref"].min()), float(x["onset_to_failure_over_Lref"].max())],
            "L_ref_h": {str(int(r)): scale.l_ref_hours(r) for r in sorted(x["rpm"].unique())}, "duty": "P/C = %.2f [ASSUMED]" % K.RIG_PC_SIM}


def summarize(m: pd.DataFrame, t_med: float | None, t_p95: float | None) -> dict:
    """All IS2 numbers on one scored frame (pure; used on val, test and by the tests)."""
    v2 = U.is2_v2(m, t_med, t_p95)
    floor = U.is2_v2(U.perfect_ai(m), t_med, t_p95)
    out = {"v2": v2, "v1_literal": U.is2_v1(m), "exact_AI_floor": {k: floor[k] for k in ("MED", "P95", "window_median", "window_p95")},
           "twin": {"share_trend": float((m["phys_source"] == "trend").mean()), "share_prior": float((m["phys_source"] != "trend").mean()),
                    "sources": m["phys_source"].value_counts().to_dict()},
           "sweep": U.sweep(m), "flag_auroc": U.flag_auroc(m)}
    if t_med is not None and t_p95 is not None:
        out["controls"] = U.controls(m, t_med, t_p95)
        out["controls_all_fail"] = all(c["verdict"] == "FAIL" for c in out["controls"])
        out["exact_AI_floor"]["verdict"] = floor.get("verdict")
    return out


# --------------------------------------------------------------------------- saved predictions

def save_preds(preds: pd.DataFrame, contract: str, split_name: str, extra: dict) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = now_tag()
    f = PRED_DIR / f"fpr_is2_{contract}_{split_name}_{tag}.csv.gz"
    preds.to_csv(f, index=False, float_format="%.6g")
    idx = {"file": f.name, "contract": contract, "split": split_name, "tag": tag, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
           "rows": int(len(preds)), "provenance": TAG, **extra}
    (PRED_DIR / f"fpr_is2_{contract}_{split_name}_{tag}.index.json").write_text(json.dumps(idx, indent=1, default=float))
    return f


def latest_preds(contract: str, split_name: str) -> tuple[dict, pd.DataFrame]:
    fs = sorted(PRED_DIR.glob(f"fpr_is2_{contract}_{split_name}_*.index.json"))
    if not fs:
        raise SystemExit(f"no saved predictions for {contract}/{split_name}: run `python -m aiengine.fpr_is2 predict --split ...`")
    idx = json.loads(fs[-1].read_text())
    return idx, pd.read_csv(PRED_DIR / idx["file"])


def split_name(r: dict) -> str:
    return r["split"] if r["section"] == K.SPLIT_SECTION else f"{r['section']}_{r['split']}"


def latest_frozen(contract: str) -> tuple[dict, Path]:
    fs = sorted(REPORTS_DIR.glob(f"fpr_is2_frozen_{contract}_*.json"))
    if not fs:
        raise SystemExit(f"refused: no frozen thresholds for {contract}; run `python -m aiengine.fpr_is2 select --contract {contract}` on validation first")
    return json.loads(fs[-1].read_text()), fs[-1]


# --------------------------------------------------------------------------- commands

def cmd_predict(a) -> int:
    r = resolve_split(a.split, a.data)
    if r["split"] == "test" and r["section"] == K.SPLIT_SECTION:
        raise SystemExit("refused: use `test` (single-pass, frozen thresholds) for the registered test split")
    df = load_units(r)
    t0 = time.perf_counter()
    p = predict_units(df, a.contract, a.units)
    f = save_preds(p, a.contract, split_name(r), {"section": r["section"], "seconds": round(time.perf_counter() - t0, 1)})
    print(a.contract, split_name(r), "->", f.name, len(p), "rows", f"{time.perf_counter() - t0:.0f} s")
    return 0


def cmd_select(a) -> int:
    r = resolve_split("val", a.data)
    df = load_units(r)
    idx, preds = latest_preds(a.contract, "val")
    m = scored(df, preds)
    base = summarize(m, None, None)
    floor = U.is2_v2(U.perfect_ai(m), None, None)
    ch = U.choose([base["v2"]], [floor])
    s = summarize(m, ch["T_MED"], ch["T_P95"])
    valid = bool(s["controls_all_fail"] and s["v2"]["verdict"] == "PASS")
    out = {"contract": a.contract, "status": "FROZEN (pre-registration; written BEFORE any test unit of the final contract is scored)",
           "provenance": TAG, "limits": LIMITS_TEXT, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "val_preds": {k: idx[k] for k in ("file", "sha256", "rows")}, "rule": ch["rule"], "T_MED": ch["T_MED"], "T_P95": ch["T_P95"],
           "val_MED_max": ch["val_MED_max"], "val_P95_max": ch["val_P95_max"], "controls_all_fail_on_val": s["controls_all_fail"],
           "val_verdict": s["v2"]["verdict"], "valid_for_test": valid, "val": s, "life_scale_val": life_scale(df),
           "note_if_invalid": None if valid else "controls did not all FAIL on validation and/or validation did not pass: the metric/thresholds "
                                                  "are not a meaningful test; test mode refuses unless --score-anyway (then flagged in the report)"}
    f = REPORTS_DIR / f"fpr_is2_frozen_{a.contract}_{now_tag()}.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    f.with_suffix(".md").write_text(render(out, "val"))
    print(json.dumps({k: out[k] for k in ("contract", "T_MED", "T_P95", "val_MED_max", "val_P95_max", "val_verdict", "controls_all_fail_on_val", "valid_for_test")}, indent=1))
    for c in s["controls"]:
        print("control", c)
    print("v1 literal", s["v1_literal"]["max"], s["v1_literal"]["verdict"], "| twin trend share", round(s["twin"]["share_trend"], 3), "| exact-AI floor", s["exact_AI_floor"])
    print("->", f)
    return 0


def cmd_eval(a) -> int:
    r = resolve_split(a.split, a.data)
    df = load_units(r)
    idx, preds = latest_preds(a.contract, split_name(r))
    fz, fp = latest_frozen(a.contract)
    s = summarize(scored(df, preds), fz["T_MED"], fz["T_P95"])
    out = {"contract": a.contract, "split": split_name(r), "preds": idx["file"], "frozen": fp.name, "provenance": TAG, "result": s,
           "life_scale": life_scale(df)}
    print(render(out | {"T_MED": fz["T_MED"], "T_P95": fz["T_P95"], "val": s, "val_verdict": s["v2"]["verdict"],
                        "controls_all_fail_on_val": None, "val_preds": {"file": idx["file"]}, "life_scale_val": out["life_scale"],
                        "rule": fz["rule"], "utc": ""}, split_name(r)))
    return 0


def cmd_test(a) -> int:
    r = resolve_split(a.split, a.data)
    if r["section"] == K.SPLIT_SECTION and r["split"] != "test":
        raise SystemExit("test mode scores a test split only")
    marker = REPORTS_DIR / f"fpr_is2_test_done_{a.contract}_{r['section']}_{r['split']}.json"
    if marker.exists():
        raise SystemExit(f"refused: this contract/section/split was already scored once ({marker.name}); re-scoring = `eval` on the saved predictions")
    fz, fp = latest_frozen(a.contract)
    if not fz.get("valid_for_test") and not a.score_anyway:
        raise SystemExit(f"refused: {fp.name} is not valid_for_test ({fz.get('note_if_invalid')}); --score-anyway to score and flag it")
    df = load_units(r)
    t0 = time.perf_counter()
    preds = predict_units(df, a.contract)
    f = save_preds(preds, a.contract, split_name(r), {"section": r["section"], "frozen": fp.name, "touch": "FIRST and only scoring pass"})
    marker.write_text(json.dumps({"frozen": fp.name, "frozen_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(), "preds": f.name,
                                  "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
    s = summarize(scored(df, preds), fz["T_MED"], fz["T_P95"])
    out = {"contract": a.contract, "status": "TEST scored once", "split": split_name(r), "provenance": TAG, "limits": LIMITS_TEXT,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frozen_file": fp.name,
           "frozen_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(), "flag": None if fz.get("valid_for_test") else "SCORED ANYWAY: val controls did not all fail",
           "T_MED": fz["T_MED"], "T_P95": fz["T_P95"], "val_preds": {"file": f.name}, "val_verdict": s["v2"]["verdict"],
           "controls_all_fail_on_val": fz.get("controls_all_fail_on_val"), "val": s, "rule": fz["rule"], "life_scale_val": life_scale(df),
           "utc_seconds": round(time.perf_counter() - t0, 1)}
    g = REPORTS_DIR / f"fpr_is2_{a.contract}_{split_name(r)}_{now_tag()}"
    g.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    g.with_suffix(".md").write_text(render(out, split_name(r)))
    print(json.dumps({"verdict": s["v2"]["verdict"], "MED": s["v2"]["MED"], "P95": s["v2"]["P95"], "T": [fz["T_MED"], fz["T_P95"]],
                      "controls_all_fail": s["controls_all_fail"], "v1_max": s["v1_literal"]["max"], "report": g.name}, indent=1, default=float))
    return 0


# --------------------------------------------------------------------------- report text

def render(o: dict, split: str) -> str:
    s, v2 = o["val"], o["val"]["v2"]
    L = [f"# FPR IS2 (wording v2) on rig_anchor simulated units: contract {o['contract']}, split {split}", "",
         f"Provenance: {TAG}", "", f"Limits: {LIMITS_TEXT}", ""]
    if o.get("flag"):
        L += [f"**FLAG: {o['flag']}**", ""]
    L += ["## Frozen thresholds (from VALIDATION only)",
          f"T_MED = {o['T_MED']}, T_P95 = {o['T_P95']} (x L_ref). Rule: {o.get('rule')}. Preds: {o.get('val_preds', {}).get('file')}.", "",
          "## Result", "| quantity | value | threshold | verdict |", "|---|---|---|---|",
          f"| MED (median over bearings of per-bearing median e) | {v2['MED']:.4f} | {o['T_MED']} | {'PASS' if v2['MED'] <= o['T_MED'] else 'FAIL'} |",
          f"| P95 (95th pct over bearings of per-bearing p95 e) | {v2['P95']:.4f} | {o['T_P95']} | {'PASS' if v2['P95'] <= o['T_P95'] else 'FAIL'} |",
          f"| **IS2 v2** | n_bearings {v2['n_bearings']}, n_windows {v2['n_windows']} | both | **{v2.get('verdict')}** |", ""]
    f = s["exact_AI_floor"]
    L += ["## Context (what the number means)",
          f"- Exact-AI floor (AI := true RUL, same windows): MED {f['MED']:.4f}, P95 {f['P95']:.4f}. A perfect AI still disagrees with this twin by that much; "
          "thresholds are never tighter than this floor.",
          f"- Twin issues a TREND RUL in {100 * s['twin']['share_trend']:.1f} % of scored windows; the rest is the rated-life prior (sources {s['twin']['sources']}). "
          "Where the twin is the prior, agreement mostly measures the AI against the rated-life assumption.",
          f"- Residual-flag AUROC for 'AI wrong by > 0.20 L_ref': {s['flag_auroc'].get('auroc')}.",
          f"- Twin vs truth (median |.|/L_ref) {v2['phys_vs_truth_lref_median']:.3f}; AI vs truth {v2['ai_vs_truth_lref_median']:.3f}.",
          f"- Scale (simulated life vs ISO 281 L_ref): {o['life_scale_val']}", "",
          "## Literal v1 (max over windows of |RUL_phys - RUL_AI| / RUL_AI <= 5 %)"]
    v1 = s["v1_literal"]
    L += [f"max {v1['max']:.3f}, p95 {v1['p95']:.3f}, median {v1['median']:.3f}, share of windows <= 5 %: {v1['share_le_5pct']:.3f} -> **{v1['verdict']}**", ""]
    if "controls" in s:
        L += ["## Negative controls (wrong AI RULs on the same windows; must all FAIL)", "| control | MED | P95 | verdict |", "|---|---|---|---|"]
        L += [f"| {c['control']} | {c['MED']:.4f} | {c['P95']:.4f} | {c['verdict']} |" for c in s["controls"]]
        L += ["", f"All controls FAIL: **{s['controls_all_fail']}**", ""]
    L += ["## Honest reading",
          "- Twin and AI read the same features (common mode): a pass says the two estimators are consistent on this simulated population, not that either is right about a real bearing.",
          "- Data are simulated, rig-anchored; the damage curve is assumed; the thresholds are tied to this population and to the ASSUMED duty used for L_ref.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.fpr_is2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "select", "test", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--contract", default=DEFAULT_CONTRACT, help="contract name without prefix/suffix (v5_node now, v6_node when it lands)")
        p.add_argument("--data", default=None)
        if name != "select":
            p.add_argument("--split", required=True)
        if name == "predict":
            p.add_argument("--units", type=int, default=None)
        if name == "test":
            p.add_argument("--score-anyway", action="store_true")
    a = ap.parse_args(argv)
    return {"predict": cmd_predict, "select": cmd_select, "test": cmd_test, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
