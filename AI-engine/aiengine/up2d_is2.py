"""UP2 IS2 wording v2: data, single scoring pass and the pure metric functions.  [SYNTHETIC] + [SIMULATION].

    python -m aiengine.up2d_is2 data              # generate + REGISTER synth_v3_val4 (val) and synth_v3_test4 (test) in
                                                  # splits.json, then their physics features (FE twin). Nothing is scored.
    python -m aiengine.up2d_is2 predict --set val4|val2   # v3b (deployed) + v3c (research) AI RUL on a VALIDATION set
    python -m aiengine.up2d_is2 select            # threshold choice from VALIDATION only -> reports/up2d_is2_select_<ts>.json
    python -m aiengine.up2d_is2 predict --set test4       # ONE scoring pass on test4 (refused unless FROZEN thresholds exist)
    python -m aiengine.up2d_eval                  # re-score the saved predictions: v2 wording + v1 literal + sweep

IS2 wording v2 (Specifications.md, coach feedback 2026-10-25) -- the metric implemented by `is2_v2` below:
  quantity     e_w = |RUL_phys,w - RUL_AI,w| / L_ref           per window w, both RULs in hours, same bearing
               RUL_phys = FE-twin physics RUL (stiffness-trend RUL, else the documented rated-life prior: aiengine/
               up2c_physics.py phys_rul_h); RUL_AI = deployed AI RUL (v3b, post-processed).
  normaliser   L_ref = reference (rated) life of the bearing at its duty speed [h]; on synthetic data the TRAIN-run
               median life L_pop(rpm) (72.9 h at 1750 rpm, 46.5 h at 3600 rpm, up2c_physics.L_POP), on the rig the
               ISO 281 L10 at the duty point. Fixed per bearing, so a late window with RUL -> 0 does not blow up e_w.
  windows      faulted bearing of every run-to-failure run, TRUE health stage 1..3, after the AI warm-up
               (window_index >= 24), RUL_true > 0.
  aggregation  per bearing b: med_b = median_w e_w, p95_b = 95th percentile_w e_w; across bearings:
               MED = median_b med_b, P95 = 95th percentile_b p95_b.
  thresholds   MED <= T_MED and P95 <= T_P95 (both must hold); values fixed from validation (FROZEN below).
v1 literal (kept, still reported): max_w |RUL_phys - RUL_AI| / RUL_AI <= 5 %.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import splits
from .config import REPORTS_DIR, SEED

PRED_DIR = REPORTS_DIR / "up2_preds"
FAMILY = "up2d_is2"
VAL4 = ("synth_v3_val4", 80, 909, 70000)      # name, runs, seed offset, run-id offset (disjoint from 0..69999)
TEST4 = ("synth_v3_test4", 100, 1010, 80000)
WARM_MIN_WINDOW = 24                          # AI warm-up: first scored window index (up2b.WARM - 1)

# ---- FROZEN thresholds (set by `select` on validation only; written here BEFORE test4 is scored) ----------------
# Frozen 2026-10-25 from reports/up2d_is2_select_20261024T225211Z.json (validation: synth_v3_val4 + synth_v3_val2),
# committed BEFORE synth_v3_test4 was scored. Disclosure: the rule was refined twice ON VALIDATION (margin 1.2 / 0.05 steps
# -> 1.1 / 0.01 steps + perfect-AI floor term) after the validation negative control "AI = 0.5 L_ref constant" PASSED the
# first rule; the superseded selection files are in reports/_superseded/. test4 had not been touched at any point.
FROZEN = {"T_MED": 0.18, "T_P95": 0.63, "source": "reports/up2d_is2_select_20261024T225211Z.json"}

V1_TEXT = ("Physics-AI residual: max relative error between the FE-identified physics RUL and the AI RUL <=5%, "
           "health stage 1->3")
V1_TARGET = 0.05


# --------------------------------------------------------------------------- data (registration first)

def _register(spec, split: str, note: str) -> pd.DataFrame:
    from .up2b import _gen
    d = _gen(spec)
    splits.register_holdout(d, spec[0], split, note=note)
    d["split"] = splits.assign(d).to_numpy()
    return d


def load_val4() -> pd.DataFrame:
    return _register(VAL4, "val", "UP2 IS2-v2 threshold-selection runs (same generator, new seed); never used to fit "
                                  "any model; used ONLY to choose the reworded IS2 thresholds")


def load_test4() -> pd.DataFrame:
    return _register(TEST4, "test", "UP2 IS2-v2 FRESH test runs (same generator, new seed); registered before any IS2-v2 "
                                    "threshold choice; scored once (v3b deployed + v3c research in the same pass)")


def load_set(name: str) -> pd.DataFrame:
    from . import up2b
    from .up2c import attach_physics
    raw = {"val4": load_val4, "test4": load_test4, "val2": up2b.load_val2}[name]()
    full = {"val4": VAL4[0], "test4": TEST4[0], "val2": "synth_v3_val2"}[name]
    return attach_physics(raw, full)


def cmd_data(a) -> int:
    import time
    from .up2c import compute_physics, phys_path
    for spec, loader in ((VAL4, load_val4), (TEST4, load_test4)):
        t0 = time.perf_counter()
        d = loader()
        p = phys_path(spec[0])
        if not p.exists():
            compute_physics(d, a.workers).to_parquet(p, index=False)
        print(spec[0], "runs", d["bearing_key"].nunique(), "windows", len(d), "rtf bearings",
              d[d["run_to_failure"]]["unit_id"].nunique(), f"{time.perf_counter() - t0:.0f} s",
              d.attrs.get("provenance", {}).get("table_hash"), flush=True)
    return 0


# --------------------------------------------------------------------------- prediction

def predict_set(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """v3b deployed AI RUL (+ classifier) and v3c research RUL on df (df carries the physics columns)."""
    from . import up2b, up2c
    hb = up2b.load_heads()
    h = up2c.load_heads()
    cp, rp_b = up2b.predict(df, hb)
    return {"cls": cp, "v3b": rp_b, "v3c": up2c.predict_rul(df, h["v3c"], hb)}


def cmd_predict(a) -> int:
    from .up2_train import meta, now_tag
    name = a.set
    prev = sorted(PRED_DIR.glob(f"{FAMILY}_{name}_*.index.json"))
    if name == "test4":
        if FROZEN is None:
            raise SystemExit("refused: thresholds are not FROZEN in aiengine/up2d_is2.py (run `select` on validation first)")
        if prev and not a.force:
            raise SystemExit(f"test4 was already scored ({prev[-1].name}); re-scoring is `python -m aiengine.up2d_eval`")
    tag = now_tag()
    df = load_set(name)
    out = predict_set(df)
    files = {}
    for k, f in out.items():
        p = PRED_DIR / f"{FAMILY}_{name}_{k}_{tag}.csv.gz"
        f.to_csv(p, index=False, float_format="%.6g")
        files[k] = p.name
    idx = {"meta": meta({"tool": "aiengine.up2d_is2 predict", "set": name, "tag": tag, "frozen": FROZEN,
                         "table_hash": df.attrs.get("provenance", {}).get("table_hash"),
                         "touch": "FIRST and only scoring pass" if name == "test4" else "validation"}),
           "files": files}
    (PRED_DIR / f"{FAMILY}_{name}_{tag}.index.json").write_text(json.dumps(idx, indent=1, default=float))
    print(name, "->", files, flush=True)
    return 0


def latest_preds(name: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    files = sorted(PRED_DIR.glob(f"{FAMILY}_{name}_*.index.json"))
    if not files:
        raise FileNotFoundError(f"no saved {name} predictions: run `python -m aiengine.up2d_is2 predict --set {name}`")
    idx = json.loads(files[-1].read_text()) | {"index_file": files[-1].name}
    return idx, {k: pd.read_csv(PRED_DIR / v) for k, v in idx["files"].items()}


# --------------------------------------------------------------------------- metric (pure functions)

def l_ref(rpm_mode) -> np.ndarray:
    from .up2c_physics import l_pop
    return np.array([l_pop(float(r)) for r in np.asarray(rpm_mode, float)])


def scored_windows(truth: pd.DataFrame, rp: pd.DataFrame, min_window: int = WARM_MIN_WINDOW,
                   stages=(1, 2, 3)) -> pd.DataFrame:
    """Faulted bearing of each run-to-failure run, TRUE stage in `stages`, window_index >= min_window, RUL_true > 0,
    joined with the AI RUL (rul_pred_h). Requires truth columns phys_rul_h, rul_hours, rpm_mode, true_stage."""
    t = truth[truth["run_to_failure"] & truth["true_stage"].isin(stages) & (truth["rul_hours"] > 0)
              & (truth["window_index"] >= min_window)]
    m = t.merge(rp[["unit_id", "window_index", "rul_pred_h"]], on=["unit_id", "window_index"], how="inner")
    m = m[m["rul_pred_h"].notna() & m["phys_rul_h"].notna()].copy()
    m["L_ref"] = l_ref(m["rpm_mode"])
    m["diff_h"] = (m["phys_rul_h"] - m["rul_pred_h"]).abs()
    m["e_lref"] = m["diff_h"] / m["L_ref"]                         # v2 quantity
    m["e_v1"] = m["diff_h"] / m["rul_pred_h"].where(m["rul_pred_h"] > 0)   # v1 literal quantity
    return m


def per_bearing(m: pd.DataFrame, col: str = "e_lref") -> pd.DataFrame:
    g = m.groupby("unit_id")[col]
    return pd.DataFrame({"n": g.size(), "med": g.median(), "p95": g.quantile(0.95), "max": g.max(),
                         "rpm_mode": m.groupby("unit_id")["rpm_mode"].first()})


def is2_v2(m: pd.DataFrame, t_med: float | None, t_p95: float | None) -> dict:
    """Reworded IS2: MED = median over bearings of per-bearing median e; P95 = 95th pct over bearings of per-bearing p95."""
    pb = per_bearing(m)
    MED, P95 = float(pb["med"].median()), float(pb["p95"].quantile(0.95))
    out = {"n_windows": int(len(m)), "n_bearings": int(len(pb)), "MED": MED, "P95": P95,
           "worst_bearing_p95": float(pb["p95"].max()), "window_max": float(m["e_lref"].max()),
           "window_p95": float(m["e_lref"].quantile(0.95)), "window_median": float(m["e_lref"].median()),
           "share_bearings_p95_le_T": float((pb["p95"] <= t_p95).mean()) if t_p95 is not None else None,
           "by_stage": {str(int(s)): {"n": int(len(g)), "median": float(g["e_lref"].median()),
                                      "p95": float(g["e_lref"].quantile(0.95))} for s, g in m.groupby("true_stage")},
           "by_rpm": {str(int(r)): {"bearings": int(len(g)), "MED": float(g["med"].median()), "P95": float(g["p95"].quantile(0.95))}
                      for r, g in pb.groupby("rpm_mode")},
           "share_physics_is_prior": float((m["phys_conf"] <= 0).mean()) if "phys_conf" in m else None,
           "ai_vs_truth_lref_median": float(((m["rul_pred_h"] - m["rul_hours"]).abs() / m["L_ref"]).median()),
           "phys_vs_truth_lref_median": float(((m["phys_rul_h"] - m["rul_hours"]).abs() / m["L_ref"]).median())}
    if t_med is not None and t_p95 is not None:
        out["T_MED"], out["T_P95"] = t_med, t_p95
        out["verdict"] = "PASS" if (MED <= t_med and P95 <= t_p95) else "FAIL"
    return out


def is2_v1(m: pd.DataFrame) -> dict:
    x = m["e_v1"].dropna().to_numpy()
    return {"text": V1_TEXT, "n": int(len(x)), "max": float(x.max()), "p95": float(np.percentile(x, 95)),
            "median": float(np.median(x)), "share_le_5pct": float((x <= V1_TARGET).mean()),
            "verdict": "PASS" if x.max() <= V1_TARGET else "FAIL"}


def sweep(m: pd.DataFrame) -> list[dict]:
    """Alternative readings on the same windows (NOT the spec metric): which normaliser/aggregation drives the number."""
    rows = []
    for col, norm in (("e_v1", "RUL_AI (v1)"), ("e_lref", "L_ref (v2)")):
        x = m[col].dropna()
        pb = per_bearing(m.dropna(subset=[col]), col)
        rows.append({"normaliser": norm, "window_max": float(x.max()), "window_p95": float(x.quantile(0.95)),
                     "window_median": float(x.median()), "bearing_med_median": float(pb["med"].median()),
                     "bearing_p95_p95": float(pb["p95"].quantile(0.95)), "bearing_p95_max": float(pb["p95"].max())})
    tr = m["rul_hours"]
    x = (m["diff_h"] / tr)
    rows.append({"normaliser": "RUL_true", "window_max": float(x.max()), "window_p95": float(x.quantile(0.95)),
                 "window_median": float(x.median()), "bearing_med_median": None, "bearing_p95_p95": None, "bearing_p95_max": None})
    return rows


def perfect_ai(m: pd.DataFrame) -> pd.DataFrame:
    """Same windows with RUL_AI := RUL_true: the disagreement an exact AI would still show against this physics RUL
    (the physics estimator's own information floor)."""
    f = m.copy()
    f["rul_pred_h"] = f["rul_hours"]
    f["diff_h"] = (f["phys_rul_h"] - f["rul_pred_h"]).abs()
    f["e_lref"] = f["diff_h"] / f["L_ref"]
    f["e_v1"] = f["diff_h"] / f["rul_pred_h"]
    return f


def controls(m: pd.DataFrame, t_med: float, t_p95: float) -> list[dict]:
    """Negative controls (NOT models): deliberately wrong AI RULs on the same windows. A spec that passes these is too loose."""
    def sub(rul):
        f = m.copy(); f["rul_pred_h"] = rul
        f["diff_h"] = (f["phys_rul_h"] - f["rul_pred_h"]).abs(); f["e_lref"] = f["diff_h"] / f["L_ref"]
        r = is2_v2(f, t_med, t_p95)
        return {"MED": r["MED"], "P95": r["P95"], "verdict": r["verdict"]}
    rng = np.random.default_rng(SEED)
    shuf = m.groupby("unit_id")["rul_pred_h"].transform(lambda x: x.iloc[::-1].to_numpy())
    return [{"control": k, **sub(v)} for k, v in (
        ("AI x 1.5 (optimistic bias)", m["rul_pred_h"] * 1.5),
        ("AI x 0.6 (pessimistic bias)", m["rul_pred_h"] * 0.6),
        ("AI + 0.3 L_ref", m["rul_pred_h"] + 0.3 * m["L_ref"]),
        ("AI time-reversed within bearing", shuf),
        ("AI = 0.5 L_ref constant", 0.5 * m["L_ref"]),
        ("AI = truth x lognormal(0, 0.5) noise", m["rul_hours"] * np.exp(rng.normal(0, 0.5, len(m)))),
    )]


def flag_auroc(m: pd.DataFrame, bad: float = 0.20) -> dict:
    """Does a large disagreement flag a bad AI estimate? AUROC of e_lref for |RUL_AI - RUL_true| > bad x L_ref."""
    from sklearn.metrics import roc_auc_score
    y = ((m["rul_pred_h"] - m["rul_hours"]).abs() / m["L_ref"] > bad).to_numpy()
    if y.all() or not y.any():
        return {"bad_def": bad, "auroc": None}
    return {"bad_def": f"|AI - truth| > {bad:.2f} L_ref", "share_bad": float(y.mean()),
            "auroc": float(roc_auc_score(y, m["e_lref"].to_numpy()))}


# --------------------------------------------------------------------------- threshold selection (validation only)

ROUND = 0.01
MARGIN = 1.10


def choose(stats: list[dict], floors: list[dict], margin: float = MARGIN) -> dict:
    """Pre-declared rule (validation only): T = ceil_to_ROUND(margin x max over validation sets of max(deployed-AI value,
    perfect-AI floor)). The floor term means the spec never demands closer agreement than the TRUE RUL itself has with
    the physics RUL; the margin covers set-to-set variation (val4 vs val2). Negative controls must all FAIL (checked)."""
    med = max(max(s["MED"] for s in stats), max(f["MED"] for f in floors))
    p95 = max(max(s["P95"] for s in stats), max(f["P95"] for f in floors))
    up = lambda v: float(round(np.ceil(margin * v / ROUND - 1e-9) * ROUND, 4))   # noqa: E731
    return {"T_MED": up(med), "T_P95": up(p95),
            "rule": f"ceil to {ROUND:.2f} of {margin} x max over validation sets of max(v3b value, perfect-AI floor)",
            "val_MED_max": med, "val_P95_max": p95}


def cmd_select(a) -> int:
    from .up2_train import meta, now_tag
    res = {}
    for name in ("val4", "val2"):
        truth = load_set(name)
        idx, pr = latest_preds(name)
        res[name] = {"index": idx["index_file"]}
        for k in ("v3b", "v3c"):
            m = scored_windows(truth, pr[k])
            res[name][k] = {"v2": is2_v2(m, None, None), "v1": is2_v1(m), "sweep": sweep(m),
                            "flag_auroc": flag_auroc(m)}
        res[name]["floor_perfect_ai"] = is2_v2(perfect_ai(scored_windows(truth, pr["v3b"])), None, None)
    ch = choose([res[n]["v3b"]["v2"] for n in ("val4", "val2")], [res[n]["floor_perfect_ai"] for n in ("val4", "val2")])
    for n in ("val4", "val2"):
        res[n]["controls"] = controls(scored_windows(load_set(n), latest_preds(n)[1]["v3b"]), ch["T_MED"], ch["T_P95"])
    ch["controls_all_fail_on_validation"] = all(c["verdict"] == "FAIL" for n in ("val4", "val2") for c in res[n]["controls"])
    ch["floor_perfect_ai"] = {n: {"MED": res[n]["floor_perfect_ai"]["MED"], "P95": res[n]["floor_perfect_ai"]["P95"]}
                              for n in ("val4", "val2")}
    out = {"meta": meta({"tool": "aiengine.up2d_is2 select", "data": "VALIDATION ONLY (synth_v3_val4, synth_v3_val2)"}),
           "chosen": ch, "results": res}
    p = REPORTS_DIR / f"up2d_is2_select_{now_tag()}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(ch, indent=1))
    for n in res:
        for k in ("v3b", "v3c"):
            v = res[n][k]["v2"]
            print(n, k, "MED %.4f P95 %.4f worst-bearing p95 %.4f window max %.4f | v1 max %.3f" %
                  (v["MED"], v["P95"], v["worst_bearing_p95"], v["window_max"], res[n][k]["v1"]["max"]))
    for n in ("val4", "val2"):
        for c in res[n]["controls"]:
            print(n, "control", c)
        print(n, "flag", res[n]["v3b"]["flag_auroc"])
    print("->", p)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2d_is2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("predict")
    p.add_argument("--set", required=True, choices=["val4", "val2", "test4"])
    p.add_argument("--force", action="store_true", help="re-run a scoring pass (must be disclosed)")
    sub.add_parser("select")
    a = ap.parse_args(argv)
    return {"data": cmd_data, "predict": cmd_predict, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
