"""Contract v7_node: RUL + 5-class + stage heads on NODE features, trained on the rig_anchor_L (learnable-profile) simulated trajectories.

Data tag (every number): [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape + life variance assumed].
No real run-to-failure data exists for 4 of 5 fault classes. Test units are held-out SIMULATED TRAJECTORIES, not held-out bearings.

    python -m aiengine.v7_node select     # val-only grid, writes the FROZEN choices report (pre-registration) + pickles
    python -m aiengine.v7_node export     # ONNX heads + model_contract_v7_node.json + golden vectors + parity (val only)
    python -m aiengine.v7_node test       # ONE scoring pass on the registered test units + controls + real checks + report

Features: aiengine.features.V3E_COLUMNS (16 baseline-relative columns, the v3e_node / v4_real classifier set) of one bearing, baseline =
median of the unit's first 24 windows (legacy form; NodeEngine's settle+capture policy is carried in the contract as for v4_real).
Heads (gradient-boosted XGBoost, up2_window aggregator inside the ONNX graph, as v3b):
  rul   : log1p(RUL in table windows) regressor on degradation windows (true onset .. failure) of the faulted bearing
  z     : degradation position regressor (all windows; healthy / transmitted partner = -0.2)  ->  stage 1..6 (up2.stage_from_z)
  cls   : 5-class XGBoost in the TFT slot (+ optional cross-bearing gate, tau chosen on val)
Post-process (C#-portable, contract `postprocess`, selected on val): stage median + rate limit, RUL log-EMA, stage-dependent scale.
Metrics are aiengine.up2_metrics (S7 MAPE / IS3 macro-F1 / stage error / caught by stage 3): reused, not redefined.
Units of RUL: TABLE windows of 10 min (one simulated snapshot per window); rul_hours = rul_windows x cadence_hours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import coe_adapt as A
from . import features as feat
from . import labels, splits, up2
from . import up2b
from . import up2_metrics as M
from . import up2_window as uw
from .config import BEARING_FEATURES, CHECKPOINT_DIR, MODELS_DIR, REPORTS_DIR, ROOT, SEED
from .rig_anchor import constants as K
from .rig_anchor import registry_l as registry
from .rig_anchor import constants_profiles as CP

COLS = list(feat.V3E_COLUMNS)
R = len(COLS)
T_CLS, NL = 25, 48
TAG = K.PROVENANCE.replace("curve shape assumed", "curve shape + life variance assumed")
LIMITS = K.LIMITS
DATA_DIR = ROOT / "data" / "rig_anchor_L"
DATA = DATA_DIR / "trval.parquet"
CACHE = ROOT / "scratch_v7"
RUN_DIR = CHECKPOINT_DIR / "v7_node"
CONTRACT, CLS_ONNX, RUL_ONNX, GOLDEN, BASELINE = ("model_contract_v7_node.json", "v7_node_cls.onnx", "v7_node_rul.onnx",
                                                   "golden_vectors_v7_node.json", "baseline_v7_node.json")
CONTRACT_VERSION = "7-node.1:v7_node"
CLASS_ORDER = up2.CLASS_ORDER
HI_COL = 0                                           # COLS[0] == "hi"
MEAN6_HI = uw.AGGS.index("mean6") * R + HI_COL       # column of the baseline-control feature in the aggregate vector

CLS_GRID = [dict(max_depth=d, learning_rate=lr) for d in (4, 6) for lr in (0.05, 0.1)]
GATE_TAUS = (0.0, 0.3, 0.5, 0.75, 1.0)
RUL_GRID = [dict(max_depth=d, levels=lv) for d in (4, 6) for lv in (False, True)]
Z_GRID = [dict(max_depth=d) for d in (4, 6)]
Z_CUTS = (0.0, -0.04, -0.08, -0.12)
STAGE_K = (1, 5, 9, 15)
STAGE_STEP = (None, 1)
RUL_ALPHA = (1.0, 0.5, 0.25, 0.1)
SCALE_EARLY = (0.7, 0.85, 1.0, 1.2)
SCALE_LATE = (0.85, 1.0, 1.15)

TRUTH = ["unit_id", "window_index", "fault_class", "run_to_failure", "observable_class", "health_stage", "rul_hours", "life_hours",
         "u_true", "in_deg", "in_deg_full", "in_deg_det", "rpm_mode", "split", "bearing_key", "bearing", "rul_windows", "z_target"]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def meta(extra: dict | None = None) -> dict:
    from .train import git_commit, git_dirty
    return {"provenance": TAG, "limits": LIMITS, "seed": SEED, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "command": "python -m aiengine.v7_node " + " ".join(sys.argv[1:]),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}


# --------------------------------------------------------------------------- data

def load_table(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    sp = splits.load()["sections"].get(registry.SECTION)
    if sp is None or not registry.verify() or (df["rig_unit"].map(sp["units"]) != df["split"]).any():
        raise SystemExit("rig_anchor_L split missing/changed in splits.json: refusing to fit or score")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered inputs (extended, baseline = median of the first 24 windows of each bearing-unit) + truth labels (as up2.label)."""
    eng = feat.engineer_table(df, extended=True)
    det = labels.label_table(eng)
    out = det.rename(columns={"observable_class": "observable_class_det", "health_stage": "health_stage_det",
                              "in_degradation_window": "in_deg_det", "in_degradation_full": "in_deg_full_det"})
    u = out["u_true"].to_numpy(float)
    rtf = out["run_to_failure"].to_numpy(bool)
    out["observable_class"] = np.where(rtf & (u >= 0), out["fault_class"], "healthy")
    out["health_stage"] = out["true_stage"].astype(int)
    out["in_deg"] = rtf & (u >= 0) & (u <= 1 - labels.DEGRADATION_TAIL_EXCLUDED)
    out["in_deg_full"] = rtf & (u >= 0)
    out["rul_windows"] = out["rul_hours"] / K.CADENCE_H
    out["z_target"] = np.where(rtf & (u >= 0), u, up2.Z_HEALTHY)
    out["bearing"] = out["unit_id"].str.rsplit(":B", n=1).str[1].astype(int)
    out["rpm_mode"] = out["rpm"].astype(float)
    return out


class FeatSet:
    """Aggregate features of every window with >= 25 rows of history: classifier (T=25 + decoder) and RUL/z (NL=48, with HI levels)."""

    def __init__(self, tab: pd.DataFrame):
        self.tab = tab
        Xc, ic = up2.unit_features(tab, T_CLS, True, columns=COLS, warm=T_CLS)
        Xr, ir = up2.unit_features(tab, NL, False, columns=COLS, warm=T_CLS, levels=True)
        assert ic.equals(ir)
        self.Xc, self.Xr = Xc, Xr
        self.T = up2.merge_truth(ic, tab, [c for c in TRUTH if c not in ("unit_id", "window_index")])
        self.F = uw.n_features(R, False)

    def xr(self, levels: bool) -> np.ndarray:
        return self.Xr if levels else self.Xr[:, :self.F]

    def sub(self, mask: np.ndarray) -> "FeatSet":
        o = object.__new__(FeatSet)
        o.tab, o.Xc, o.Xr, o.T, o.F = self.tab, self.Xc[mask], self.Xr[mask], self.T[mask].reset_index(drop=True), self.F
        return o


def _fs_chunk(df: pd.DataFrame):
    fs = FeatSet(engineer(df))
    return fs.tab, fs.Xc, fs.Xr, fs.T


def featset(path: Path, split: str, workers: int = 4, chunk_units: int = 40) -> "FeatSet":
    """Registry-checked, chunk-parallel (per-unit independent) engineering + features of one split; pickled next to the parquet (git-ignored)."""
    cp = path.with_name(f"{path.stem}_{split}_fs.pkl")
    if cp.exists() and cp.stat().st_mtime > path.stat().st_mtime:
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    from concurrent.futures import ProcessPoolExecutor
    df = load_table(path)
    df = df[df["split"] == split]
    units = sorted(df["rig_unit"].unique())
    chunks = [df[df["rig_unit"].isin(units[i:i + chunk_units])] for i in range(0, len(units), chunk_units)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_fs_chunk, chunks))
    o = object.__new__(FeatSet)
    o.tab = pd.concat([p[0] for p in parts], ignore_index=True)
    o.Xc = np.concatenate([p[1] for p in parts])
    o.Xr = np.concatenate([p[2] for p in parts])
    o.T = pd.concat([p[3] for p in parts], ignore_index=True)
    o.F = uw.n_features(R, False)
    with open(cp, "wb") as fh:
        pickle.dump(o, fh, protocol=4)
    return o


def split_of(tab: pd.DataFrame, name: str) -> pd.DataFrame:
    return tab[tab["split"] == name]


# --------------------------------------------------------------------------- heads

def y_cls(T: pd.DataFrame) -> np.ndarray:
    return T["observable_class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()


def fit_cls(tr: FeatSet, va: FeatSet, hp: dict, threads: int):
    return up2.fit_classifier("xgb", tr.Xc, y_cls(tr.T), va.Xc, y_cls(va.T), threads=threads, params=dict(hp))


def deg_mask(T: pd.DataFrame) -> np.ndarray:
    return (T["run_to_failure"] & (T["u_true"] >= 0) & (T["rul_windows"] > 0)).to_numpy(bool)


def fit_rul(tr: FeatSet, va: FeatSet, cfg: dict, threads: int):
    dtr, dva = deg_mask(tr.T), deg_mask(va.T)
    Xtr, Xva = tr.xr(cfg["levels"]), va.xr(cfg["levels"])
    return up2.fit_regressor("xgb", Xtr[dtr], np.log1p(tr.T.loc[dtr, "rul_windows"].to_numpy()), Xva[dva],
                             np.log1p(va.T.loc[dva, "rul_windows"].to_numpy()), threads=threads,
                             wtr=up2.unit_balanced_weights(tr.T.loc[dtr, "unit_id"]), wva=up2.unit_balanced_weights(va.T.loc[dva, "unit_id"]),
                             params={"max_depth": cfg["max_depth"]})


def fit_z(tr: FeatSet, va: FeatSet, cfg: dict, levels: bool, threads: int):
    return up2.fit_regressor("xgb", tr.xr(levels), tr.T["z_target"].to_numpy(), va.xr(levels), va.T["z_target"].to_numpy(),
                             threads=threads, params={"max_depth": cfg["max_depth"]})


def fit_hi_baseline(tr: FeatSet) -> dict:
    """Trivial control: log1p(RUL windows) = a + b * HI(mean of the last 6 windows) (OLS, degradation windows of the train units)."""
    d = deg_mask(tr.T)
    x = tr.Xr[d, MEAN6_HI].astype(float)
    y = np.log1p(tr.T.loc[d, "rul_windows"].to_numpy())
    b, a = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b), "feature": "mean of the last 6 rows of the health index hi (aggregate 'mean6:hi')"}


def hi_baseline_predict(lin: dict, S: FeatSet) -> np.ndarray:
    return np.maximum(np.expm1(lin["a"] + lin["b"] * S.Xr[:, MEAN6_HI].astype(float)), 0.0)


def cls_probs(model, S: FeatSet) -> np.ndarray:
    return model.predict_proba(S.Xc)


def apply_gate(P: np.ndarray, S: FeatSet, tau: float) -> pd.DataFrame:
    pr = A.frame(S.T[["unit_id", "window_index"]], P)
    return A.gate(pr, S.tab, tau, "family")


def head_raw(rm, zm, S: FeatSet, levels: bool, z_cut: float) -> pd.DataFrame:
    X = S.xr(levels)
    out = S.T[["unit_id", "window_index"]].copy()
    out["rul_pred_windows_raw"] = np.maximum(np.expm1(rm.predict(X)), 0.0)
    out["z_pred"] = zm.predict(X)
    out["stage_pred_raw"] = up2.stage_from_z(out["z_pred"].to_numpy(), z_cut)
    return out


def postprocess(raw: pd.DataFrame, pp: dict) -> pd.DataFrame:
    rp = up2b.apply_pp(raw, pp)
    rp["rul_pred_h"] = rp["rul_pred_windows"] * K.CADENCE_H
    return rp


def score_rul_frame(S: FeatSet, rp: pd.DataFrame, col: str = "rul_pred_h") -> dict:
    r = S.T.merge(rp[["unit_id", "window_index", col] + (["stage_pred"] if "stage_pred" in rp else [])], on=["unit_id", "window_index"])
    return M.rul(r[r["run_to_failure"]], col=col)


def score_stage_frame(S: FeatSet, rp: pd.DataFrame) -> dict:
    r = S.T.merge(rp[["unit_id", "window_index", "stage_pred"]], on=["unit_id", "window_index"])
    return M.stage(r[r["run_to_failure"]])


def stage_key(st: dict) -> tuple:
    return (st["max_abs_error"], st["windows_error_ge2"], -st["within1_frac"], st["mae"])


def cls_scores(S: FeatSet, pred: pd.DataFrame) -> dict:
    """IS3 classification parts on every scored window (both bearings of every unit)."""
    c = S.T.merge(pred[["unit_id", "window_index", "pred_class"]], on=["unit_id", "window_index"])
    out = {"classification": M.classification(c), "caught": M.caught_by_stage3(c)}
    out["healthy_alarms"] = healthy_alarms(c)
    out["per_rpm_macro_f1"] = {str(int(k)): M.classification(g)["macro_f1"] for k, g in c.groupby("rpm_mode")}
    return out


def healthy_alarms(c: pd.DataFrame) -> dict:
    """False-alarm rate (window predicted a fault while the bearing is observably healthy), split by where the healthy window comes from."""
    h = c[c["observable_class"] == "healthy"].copy()
    h["alarm"] = h["pred_class"] != "healthy"
    kind = np.where(h["fault_class"] == "healthy", np.where(h["run_to_failure"], "pre_onset_faulted_bearing", "healthy_unit_or_partner"), "x")
    h["kind"] = np.where(h["run_to_failure"], "pre_onset_faulted_bearing",
                         np.where(h["unit_id"].map(h.groupby("bearing_key")["fault_class"].first()) == "healthy", "healthy_unit", "partner_of_faulted"))
    # partner_of_faulted: bearing 2 of a faulted unit; fault_class of that unit's bearing 1 is the fault (looked up from c)
    unit_fault = c[c["run_to_failure"]].groupby("bearing_key")["fault_class"].first()
    h["kind"] = np.where(h["run_to_failure"], "pre_onset_faulted_bearing",
                         np.where(h["bearing_key"].isin(unit_fault.index), "partner_of_faulted_bearing", "healthy_unit"))
    out = {"windows": int(len(h)), "false_alarm_rate": float(h["alarm"].mean()) if len(h) else None}
    for k, g in h.groupby("kind"):
        out[k] = {"windows": int(len(g)), "false_alarm_rate": float(g["alarm"].mean())}
    # alarm EVENT: 3 consecutive fault predictions on a healthy bearing-unit (the persistence rule of caught_by_stage3)
    ev = 0
    n_units = 0
    for uid, g in c[(c["observable_class"] == "healthy") & ~c["run_to_failure"]].groupby("unit_id"):
        n_units += 1
        a = (g.sort_values("window_index")["pred_class"] != "healthy").to_numpy()
        run = np.convolve(a.astype(int), np.ones(M.DETECTION_PERSISTENCE, int), "valid") if len(a) >= M.DETECTION_PERSISTENCE else np.array([0])
        ev += int((run >= M.DETECTION_PERSISTENCE).any())
    out["healthy_bearing_units"] = n_units
    out["healthy_bearing_units_with_3_consecutive_alarms"] = ev
    return out


# --------------------------------------------------------------------------- select (val only) -> frozen choices

def cmd_select(a) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    frozen_files = sorted(REPORTS_DIR.glob("v7_node_frozen_*.json"))
    if frozen_files and not a.force:
        raise SystemExit(f"choices already frozen ({frozen_files[-1].name}); not re-selecting (use --force before any test pass)")
    if (REPORTS_DIR / "v7_node_test_done.json").exists():
        raise SystemExit("test already scored: selection is closed")
    t0 = time.perf_counter()
    sets = {s: featset(DATA, s) for s in ("train", "val")}
    tr, va = sets["train"], sets["val"]
    counts = {s: {"units": int(S.tab["rig_unit"].nunique()), "feature_rows": int(len(S.T))} for s, S in sets.items()}
    print("features", counts, f"{time.perf_counter() - t0:.0f}s", flush=True)
    th = a.threads
    # ---- classifier + gate ----
    cls_rows, best_c = [], None
    for hp in CLS_GRID:
        m, info = fit_cls(tr, va, hp, th)
        P = cls_probs(m, va)
        for tau in GATE_TAUS:
            pred = apply_gate(P, va, tau)
            s = cls_scores(va, pred)
            row = {"hp": hp, "gate_tau": tau, "best_iteration": info["best_iteration"], "val_macro_f1": s["classification"]["macro_f1"],
                   "val_false_alarm": s["classification"]["false_alarm_rate"], "val_missed_fault": s["classification"]["missed_fault_rate"],
                   "val_caught_by_stage3": f"{s['caught']['caught_by_stage3']}/{s['caught']['rtf_bearings']}"}
            cls_rows.append(row)
            print("cls", json.dumps(row), flush=True)
            key = (row["val_macro_f1"], -info["best_iteration"])
            if best_c is None or key > best_c[0]:
                best_c = (key, hp, tau, m, info)
    # ---- RUL regressor ----
    rul_rows, best_r = [], None
    zdefault = None
    for cfg in RUL_GRID:
        rm, info = fit_rul(tr, va, cfg, th)
        zm0, _ = fit_z(tr, va, Z_GRID[0], cfg["levels"], th) if zdefault is None or zdefault[0] != cfg["levels"] else (zdefault[1], None)
        zdefault = (cfg["levels"], zm0)
        raw = head_raw(rm, zm0, va, cfg["levels"], 0.0)
        rp = postprocess(raw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": 1.0, "rul_scale_early": 1.0, "rul_scale_late": 1.0})
        s7 = score_rul_frame(va, rp)["S7_mape_pct"]
        row = {**cfg, "best_iteration": info["best_iteration"], "val_S7_mape_pct_unfiltered": s7}
        rul_rows.append(row)
        print("rul", json.dumps(row), flush=True)
        if best_r is None or s7 < best_r[0]:
            best_r = (s7, cfg, rm, info)
    cfg_r = best_r[1]
    # ---- z / stage head (same feature set as the selected RUL head) + stage post-process + z_cut ----
    stage_rows, best_z = [], None
    vrt = va.sub(va.T["run_to_failure"].to_numpy(bool))
    for zc in Z_GRID:
        zm, zinfo = fit_z(tr, va, zc, cfg_r["levels"], th)
        for cut in Z_CUTS:
            raw = head_raw(best_r[2], zm, vrt, cfg_r["levels"], cut)
            for k in STAGE_K:
                for step in STAGE_STEP:
                    rp = postprocess(raw, {"stage_k": k, "stage_max_step": step, "rul_alpha": 1.0, "rul_scale_early": 1.0, "rul_scale_late": 1.0})
                    st = score_stage_frame(vrt, rp)
                    row = {"z_head": zc, "z_cut": cut, "stage_k": k, "stage_max_step": step, "val_stage_max": st["max_abs_error"],
                           "val_stage_n_ge2": st["windows_error_ge2"], "val_stage_within1": st["within1_frac"], "val_stage_mae": st["mae"]}
                    stage_rows.append(row)
                    key = stage_key(st)
                    if best_z is None or key < best_z[0]:
                        best_z = (key, zc, cut, k, step, zm, zinfo)
    print("stage best", json.dumps({"z_head": best_z[1], "z_cut": best_z[2], "k": best_z[3], "step": best_z[4], "key": best_z[0]}), flush=True)
    _, zc, cut, k, step, zm, zinfo = best_z
    # ---- RUL post-process (alpha, scales) on val S7, stage from the selected stage post-process ----
    raw = head_raw(best_r[2], zm, vrt, cfg_r["levels"], cut)
    pp_rows, best_p = [], None
    for al in RUL_ALPHA:
        for se in SCALE_EARLY:
            for sl in SCALE_LATE:
                pp = {"stage_k": k, "stage_max_step": step, "rul_alpha": al, "rul_scale_early": se, "rul_scale_late": sl}
                s7 = score_rul_frame(vrt, postprocess(raw, pp))["S7_mape_pct"]
                pp_rows.append({**pp, "val_S7_mape_pct": s7})
                if best_p is None or s7 < best_p[0]:
                    best_p = (s7, pp)
    pp_sel = best_p[1]
    # ---- trivial baseline: own alpha on val (same grid, scale 1), same S7 scoring ----
    lin = fit_hi_baseline(tr)
    braw = vrt.T[["unit_id", "window_index"]].copy()
    braw["rul_pred_windows_raw"] = hi_baseline_predict(lin, vrt)
    braw["stage_pred_raw"] = 1
    base_rows, best_b = [], None
    for al in RUL_ALPHA:
        bp = {"stage_k": 1, "stage_max_step": None, "rul_alpha": al, "rul_scale_early": 1.0, "rul_scale_late": 1.0}
        s7 = score_rul_frame(vrt, postprocess(braw, bp))["S7_mape_pct"]
        base_rows.append({"alpha": al, "val_S7_mape_pct": s7})
        if best_b is None or s7 < best_b[0]:
            best_b = (s7, al)
    lin["rul_alpha"] = best_b[1]
    # ---- val numbers of the selected configuration (all heads together, as deployed) ----
    rp_val = postprocess(head_raw(best_r[2], zm, va, cfg_r["levels"], cut), pp_sel)
    Pv = cls_probs(best_c[3], va)
    val = {"S7_mape_pct": score_rul_frame(va, rp_val)["S7_mape_pct"], "S7_baseline_mape_pct": best_b[0],
           "stage": {k_: v for k_, v in score_stage_frame(va, rp_val).items() if k_ != "error_hist"},
           "is3": {"macro_f1": cls_scores(va, apply_gate(Pv, va, best_c[2]))["classification"]["macro_f1"],
                   "caught": {k_: v for k_, v in cls_scores(va, apply_gate(Pv, va, best_c[2]))["caught"].items() if k_ != "per_unit"}}}
    with open(RUN_DIR / "heads.pkl", "wb") as f:
        pickle.dump({"cls": best_c[3], "rul": best_r[2], "z": zm, "lin": lin}, f)
    frozen = {
        "meta": meta({"phase": "PRE-REGISTRATION: frozen choices written before the test units are scored",
                      "data_counts": counts, "data_file": str(DATA.name), "elapsed_s": round(time.perf_counter() - t0, 1),
                      "n_threads": th}),
        "feature_columns": COLS, "T_cls": T_CLS, "NL_rul": NL,
        "frozen": {"classifier_hp": best_c[1], "classifier_best_iteration": best_c[4]["best_iteration"], "gate_tau": best_c[2],
                   "gate_signal": "family", "rul_head": cfg_r, "rul_best_iteration": best_r[3]["best_iteration"],
                   "z_head": zc, "z_best_iteration": zinfo["best_iteration"], "z_cut": cut, "postprocess": pp_sel,
                   "baseline_control": lin},
        "selection_rule": {"classifier": "max val macro-F1 (all windows, both bearings), tie -> fewer trees", "rul": "min val S7 MAPE (unfiltered)",
                           "stage": "min val (max stage error, #windows error>=2, -within1, MAE)",
                           "rul_post": "min val S7 MAPE over alpha x scale_early x scale_late", "baseline": "OLS fit on train; alpha by val S7"},
        "val_grid": {"classifier": cls_rows, "rul": rul_rows, "stage": stage_rows, "rul_postprocess": pp_rows, "baseline_alpha": base_rows},
        "val_selected": val,
    }
    path = REPORTS_DIR / f"v7_node_frozen_{now_tag()}.json"
    path.write_text(json.dumps(frozen, indent=1, default=float))
    print("FROZEN ->", path.name, json.dumps(frozen["frozen"], default=float))
    print("VAL", json.dumps(val, default=float))
    return 0


def load_frozen() -> tuple[dict, Path]:
    fs = sorted(REPORTS_DIR.glob("v7_node_frozen_*.json"))
    if not fs:
        raise SystemExit("run `select` first")
    return json.loads(fs[-1].read_text()), fs[-1]


def load_heads() -> dict:
    with open(RUN_DIR / "heads.pkl", "rb") as f:
        return pickle.load(f)


def predict_all(heads: dict, fz: dict, S: FeatSet) -> dict:
    """Frozen deployed behaviour on one feature set: classifier (+gate), post-processed RUL and stage, trivial-baseline RUL."""
    f = fz["frozen"]
    pred = apply_gate(cls_probs(heads["cls"], S), S, f["gate_tau"])
    rp = postprocess(head_raw(heads["rul"], heads["z"], S, f["rul_head"]["levels"], f["z_cut"]), f["postprocess"])
    lin = heads["lin"]
    braw = S.T[["unit_id", "window_index"]].copy()
    braw["rul_pred_windows_raw"] = hi_baseline_predict(lin, S)
    braw["stage_pred_raw"] = 1
    bp = postprocess(braw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": lin["rul_alpha"], "rul_scale_early": 1.0, "rul_scale_late": 1.0})
    return {"cls": pred, "rp": rp, "base": bp}


def full_scores(S: FeatSet, p: dict) -> dict:
    cs = cls_scores(S, p["cls"])
    rr = score_rul_frame(S, p["rp"])
    br = score_rul_frame(S, p["base"])
    r = S.T.merge(p["rp"][["unit_id", "window_index", "stage_pred"]], on=["unit_id", "window_index"])
    st = M.stage(r[r["run_to_failure"]])
    stage_h = M.stage(r[~r["run_to_failure"]]) if (~r["run_to_failure"]).any() else {}
    return {"S7": {k: v for k, v in rr.items() if k != "per_bearing_mape_true_onset_pct"}, "S7_baseline": {k: v for k, v in br.items() if k != "per_bearing_mape_true_onset_pct"},
            "IS3_classification": cs["classification"], "IS3_caught": {k: v for k, v in cs["caught"].items() if k != "per_unit"},
            "IS3_caught_per_unit": cs["caught"]["per_unit"], "IS3_stage": st, "stage_healthy_bearings": stage_h,
            "healthy_alarms": cs["healthy_alarms"], "per_rpm_macro_f1": cs["per_rpm_macro_f1"]}


# --------------------------------------------------------------------------- export (val data only for parity / golden)

def build_models(heads: dict, fz: dict):
    from . import up2_export as X
    f = fz["frozen"]
    m_cls = X.build_cls(heads["cls"], T_CLS, R, X.importances_per_column(heads["cls"], R))
    m_rul = X.build_rul(heads["rul"], heads["z"], NL, R, levels=f["rul_head"]["levels"], z_cut=float(f["z_cut"]))
    return m_cls, m_rul


def onnx_head_outputs(S: FeatSet, onnx_cls: Path, onnx_rul: Path, max_units: int = 6):
    """ONNX (deployed) vs native for the first max_units units of S. Returns (diffs dict)."""
    import onnxruntime as ort
    sc = ort.InferenceSession(str(onnx_cls), providers=["CPUExecutionProvider"])
    sr = ort.InferenceSession(str(onnx_rul), providers=["CPUExecutionProvider"])
    units = S.T["unit_id"].drop_duplicates().iloc[:max_units]
    out = {"p": [], "rul": [], "z": [], "st": []}
    for uid in units:
        g = S.tab[S.tab["unit_id"] == uid].sort_values("window_index")
        rows = g[COLS].to_numpy(np.float64)
        bc, _ = uw.blocks(rows, T_CLS)
        br, _ = uw.blocks(rows, NL)
        for k in range(T_CLS - 1, len(g)):
            lg = sc.run(["logits"], {"encoder_cont": bc[k:k + 1, :-1], "decoder_cont": bc[k:k + 1, -1:]})[0]
            o = sr.run(["rul_head"], {"encoder_cont": br[k:k + 1]})[0][0]
            out["p"].append(np.exp(lg[0, 0])); out["rul"].append(o[0]); out["z"].append(o[1]); out["st"].append(o[2])
    return units, {k: np.asarray(v) for k, v in out.items()}


def parity(heads: dict, fz: dict, S: FeatSet, onnx_cls: Path, onnx_rul: Path, max_units: int = 6) -> dict:
    f = fz["frozen"]
    units, o = onnx_head_outputs(S, onnx_cls, onnx_rul, max_units)
    m = S.T["unit_id"].isin(units).to_numpy()
    Sm = S.sub(m)
    P = cls_probs(heads["cls"], Sm)
    raw = head_raw(heads["rul"], heads["z"], Sm, f["rul_head"]["levels"], f["z_cut"])
    rel = np.abs(o["rul"] - raw["rul_pred_windows_raw"].to_numpy()) / np.maximum(raw["rul_pred_windows_raw"].to_numpy(), 1.0)
    return {"units": int(len(units)), "windows": int(len(o["p"])), "cls_prob_max_abs_diff": float(np.abs(o["p"] - P).max()),
            "cls_argmax_mismatch": int((o["p"].argmax(1) != P.argmax(1)).sum()), "rul_windows_max_rel_diff": float(rel.max()),
            "z_max_abs_diff": float(np.abs(o["z"] - raw["z_pred"].to_numpy()).max()),
            "stage_mismatch": int((o["st"].astype(int) != raw["stage_pred_raw"].to_numpy()).sum()),
            "reference": "aggregator via ORT + native xgboost predict (float32 features); units are VAL units"}


def contract_dict(fz: dict, par: dict, sizes: dict, onnx_paths: tuple) -> dict:
    from .train import git_commit
    con = json.loads((MODELS_DIR / "model_contract_v4_real.json").read_text())      # same layout the NodeEngine/C# host already loads
    f = fz["frozen"]
    pp = f["postprocess"]
    con["contract_version"] = CONTRACT_VERSION
    con["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con["generated_by"] = "AI-engine/aiengine/v7_node.py export (layout of model_contract_v4_real.json; heads, inputs, postprocess replaced)"
    con["provenance"] = {"synthetic": True, "tag": TAG, "limits": LIMITS, "datasets": ["rig_anchor_L (train units; val units for selection)"],
                         "note": f"{TAG} gradient-boosted heads trained on rig_anchor_L train units only; selected on val units; test units scored once "
                                 "(reports/v7_node_*.json). NOT a model of real run-to-failure data.",
                         "seed": SEED, "git_commit": git_commit(), "command": "python -m aiengine.v7_node select/export/test",
                         "frozen_choices_file": fz["_file"], "frozen": f}
    ident = {c: {"center": 0.0, "scale": 1.0, "kind": "identity"} for c in COLS}
    t, n = con["models"]["tft"], con["models"]["nhits"]
    t.update({"onnx": CLS_ONNX, "family": "gradient-boosted trees (xgb) in the TFT slot, trained on rig_anchor_L simulated trajectories " + TAG,
              "encoder_length": T_CLS - 1, "columns": COLS, "scalers": ident, "warmup_windows": T_CLS, "size_kb": sizes["cls_kb"], "parity": par,
              "present_classes": list(CLASS_ORDER)})
    n.update({"onnx": RUL_ONNX, "family": "gradient-boosted trees (rul + z regressors, xgb) in the N-HiTS slot " + TAG, "encoder_length": NL,
              "prediction_length": 1, "columns": COLS, "scalers": ident, "size_kb": sizes["rul_kb"],
              "output": "rul_head [B,3] = [rul_windows, z, stage]; rul_windows in TABLE windows (one simulated 10-min snapshot per window); "
                        f"stage = 1 if z < {f['z_cut']} else min(6, 2 + floor(5 clip(z, 0, 1)))",
              "in_graph_features": {"aggregates": uw.AGGS, "hi_levels": uw.HI_LEVELS if f["rul_head"]["levels"] else [], "reference_impl": "aiengine/up2_window.py"},
              "parity": par})
    con["rul"].update({"method": "direct_v3",
                       "procedure": "rul_hours = postprocess(rul_head[0]) * cadence_hours (median spacing of the last <=48 window times); stage = postprocess(rul_head[2]). "
                                    "RUL is meaningful when stage >= 2 and the bearing is the faulted one. CADENCE: the head is trained on windows spaced "
                                    f"{K.CADENCE_H * 60:.0f} min apart ({TAG}); hosts must feed one window per snapshot with t20_ms spaced accordingly.",
                       "table_cadence_hours": K.CADENCE_H, "interval_rel": 0.25})
    con["stage"] = (f"rul_head[2] from the z head: 1 if z < {f['z_cut']} else min(6, 2 + floor(5 clip(z,0,1))); deployed: postprocess.stage "
                    f"(median {pp['stage_k']}, max_step {pp['stage_max_step']})")
    con["postprocess"] = up2b.postprocess_contract(pp)
    con["postprocess"]["selected_on"] = "rig_anchor_L val units only"
    if f["gate_tau"] > 0:
        con["cross_bearing_gate"] = {"tau": float(f["gate_tau"]), "signal": "family", "selected_on": "rig_anchor_L val units only",
                                     "family_of_class": {c: f"log_{fam}_ratio" for c, fam in A.FAMILY_OF.items()}}
    else:
        con.pop("cross_bearing_gate", None)
    con["real_data"] = {"note": "no real windows are used for fitting v7_node; real runs are used only as the anchor pools of rig_anchor_L "
                                "and for reported checks (not independent: they are the anchor source)"}
    con["disclosures"] = dict(con.get("disclosures", {}))
    con["disclosures"]["provenance"] = con["provenance"]["note"]
    con["disclosures"]["limits"] = LIMITS
    return con


def pick_golden_unit(tab: pd.DataFrame) -> str:
    va = tab[(tab["split"] == "val") & tab["run_to_failure"] & (tab["rpm"] == 1800)]
    sizes = va.groupby("rig_unit").size()
    return str(sizes.index[(sizes - sizes.median()).abs().argmin()])


def cmd_export(a) -> int:
    import onnx
    from .engine import HybridEngine
    fz, path = load_frozen()
    fz["_file"] = path.name
    heads = load_heads()
    va = featset(DATA, "val")
    tab = va.tab
    m_cls, m_rul = build_models(heads, fz)
    onnx.save(m_cls, str(MODELS_DIR / CLS_ONNX))
    onnx.save(m_rul, str(MODELS_DIR / RUL_ONNX))
    par = parity(heads, fz, va, MODELS_DIR / CLS_ONNX, MODELS_DIR / RUL_ONNX)
    sizes = {"cls_kb": round((MODELS_DIR / CLS_ONNX).stat().st_size / 1024, 1), "rul_kb": round((MODELS_DIR / RUL_ONNX).stat().st_size / 1024, 1)}
    con = contract_dict(fz, par, sizes, (CLS_ONNX, RUL_ONNX))
    (MODELS_DIR / CONTRACT).write_text(json.dumps(con, indent=2, default=float))
    base = json.loads((MODELS_DIR / "baseline_v4_real.json").read_text())
    base["contract_version"] = CONTRACT_VERSION
    base["note"] = "rig commissioning baselines [MEASURED on rig] copied from baseline_v4_real.json; same policy"
    (MODELS_DIR / BASELINE).write_text(json.dumps(base, indent=1))
    # golden vectors from a VAL unit (test units are never touched here)
    uid = pick_golden_unit(tab)
    eng = HybridEngine(MODELS_DIR, contract_file=CONTRACT)
    u1 = tab[tab["unit_id"] == f"{uid}:B1"].sort_values("window_index")
    u2 = tab[tab["unit_id"] == f"{uid}:B2"].sort_values("window_index")
    golden = eng.golden_vectors(u1, n=12)
    golden["cases_late"] = eng.golden_cases_at(u1, [int(x) for x in np.linspace(len(u1) * 0.5, len(u1) - 1, 8)])
    eng.reset()
    rig_in, rig_out = [], []
    keys = ("bearing", "status", "fault_class", "transmitted_from_partner", "class_probs", "health_stage", "rul_hours", "rul_hours_raw", "health_stage_raw")
    for r1, r2 in zip(u1.itertuples(index=False), u2.itertuples(index=False)):
        f32 = np.concatenate([[getattr(r1, c) for c in BEARING_FEATURES], [getattr(r2, c) for c in BEARING_FEATURES]])
        t_ms = int(round(float(r1.t_hours) * 3.6e6)) % (2 ** 32)
        o = eng.process(float(r1.rpm), f32, t_ms)
        rig_in.append({"window_index": int(r1.window_index), "rpm": float(r1.rpm), "t20_ms": t_ms, "features32": f32.tolist()})
        rig_out.append([{k: v for k, v in b.items() if k in keys} for b in o["bearings"]])
    golden["rig_stream"] = {"unit": uid, "scenario": "bearing 1 = faulted bearing of a VAL unit, bearing 2 = partner (B2X := B2Y policy applied in the table)",
                            "inputs": rig_in, "outputs": rig_out}
    golden["provenance"] = f"{TAG} VAL unit {uid}"
    (MODELS_DIR / GOLDEN).write_text(json.dumps(golden, indent=1, default=float))
    print(json.dumps({"parity": par, "sizes_kb": sizes, "golden_unit": uid}, indent=1))
    return 0


# --------------------------------------------------------------------------- test (ONE pass) + controls + real checks

def train_1800_cls(frozen: dict, tr: FeatSet, va: FeatSet, th: int):
    """Cross-speed control: the frozen classifier hyper-parameters, fitted on 1800-rpm train units only (no early stopping on test)."""
    m1 = tr.T["rpm_mode"].to_numpy() == 1800
    v1 = va.T["rpm_mode"].to_numpy() == 1800
    return up2.fit_classifier("xgb", tr.Xc[m1], y_cls(tr.T[m1]), va.Xc[v1], y_cls(va.T[v1]), threads=th, params=dict(frozen["classifier_hp"]))[0]


TEST_SETS = {"test_L": "PRIMARY: L profile, test shape range SHAPE_TEST_L (OOD vs train)",
             "test_wide": "robustness: same unit ids, L profile but the full registered wide shape ranges (a .05-.60, p 1.5-4)",
             "test_hard": "robustness: same unit ids, HARD profile (life_sigma .35, late .35, jitter .04, wide shapes) = the rig_anchor test conditions",
             "test_wrong": "control: same unit ids, L profile, near-linear damage curves (a .85-.95, p 1.0-1.2)"}


def cmd_data(a) -> int:
    """Register rig_anchor_L (idempotent) and build the git-ignored parquets: trval + the four test sets (test sets are only WRITTEN here)."""
    from .rig_anchor import generator as G
    from .rig_anchor import pools as P
    print("split:", registry.register("L", CP.PROFILES["L"]), "verified:", registry.verify(), flush=True)
    pl = P.Pools()
    plan = registry.plan_units_l()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    L = CP.PROFILES["L"]
    tv = [p for p in plan if p["split"] != "test"]
    te = [p for p in plan if p["split"] == "test"]
    G.build(pl, tv, profile=L).to_parquet(DATA, index=False)
    prof = {"test_L": (dict(L, shape_test=registry.SHAPE_TEST_L), None),
            "test_wide": (dict(L, shape_test=K.SHAPE_TEST), None),
            "test_hard": ({"life_sigma": K.LIFE_SIGMA, "late_sigma": K.LATE_SIGMA["test"], "u_jitter": K.U_ANCHOR_JITTER, "shape_test": K.SHAPE_TEST}, None),
            "test_wrong": (L, {"a": (0.85, 0.95), "p": (1.0, 1.2)})}
    for name, (pf, over) in prof.items():
        G.build(pl, te, profile=pf, shape_override=over).to_parquet(DATA_DIR / f"{name}.parquet", index=False)
    print("built", [p.name for p in DATA_DIR.glob("*.parquet")])
    return 0


def score_set(heads, fz, name: str) -> tuple:
    S = featset(DATA_DIR / f"{name}.parquet", "test")
    return S, full_scores(S, predict_all(heads, fz, S))


def short(sc: dict) -> dict:
    return {"S7_mape_pct": sc["S7"]["S7_mape_pct"], "S7_true_onset_pct": sc["S7"]["alt_mape_true_onset_pct"], "S7_n_bearings": sc["S7"]["n_bearings"],
            "S7_per_bearing_summary": sc["S7"]["per_bearing_summary"], "S7_baseline_mape_pct": sc["S7_baseline"]["S7_mape_pct"],
            "IS3_macro_f1": sc["IS3_classification"]["macro_f1"], "IS3_stage_max": sc["IS3_stage"]["max_abs_error"],
            "IS3_stage_within1": sc["IS3_stage"]["within1_frac"], "IS3_caught": sc["IS3_caught"]}


def cmd_test(a) -> int:
    marker = REPORTS_DIR / "v7_node_test_done.json"
    if marker.exists():
        raise SystemExit(f"test already scored once ({marker.name}); not re-scoring")
    from . import v7_real
    fz, fpath = load_frozen()
    fz["_file"] = fpath.name
    f = fz["frozen"]
    heads = load_heads()
    t0 = time.perf_counter()
    tr, va = (featset(DATA, s) for s in ("train", "val"))
    frozen_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
    print("scoring TEST once with", fpath.name, frozen_sha[:12], flush=True)
    te, test = score_set(heads, fz, "test_L")
    counts = {"train": {"units": int(tr.tab["rig_unit"].nunique())}, "val": {"units": int(va.tab["rig_unit"].nunique())},
              "test": {"units": int(te.tab["rig_unit"].nunique()), "feature_rows": int(len(te.T)),
                       "faulted_run_to_failure_bearings": int(te.T[te.T["run_to_failure"]]["unit_id"].nunique())}}
    side = {}
    for nm in ("test_wide", "test_hard", "test_wrong"):
        _, sc = score_set(heads, fz, nm)
        side[nm] = {"description": TEST_SETS[nm], **short(sc)}
    m1800 = train_1800_cls(f, tr, va, a.threads)
    te3 = te.sub(te.T["rpm_mode"].to_numpy() == 3600)
    pr = apply_gate(m1800.predict_proba(te3.Xc), te3, f["gate_tau"])
    c3 = cls_scores(te3, pr)
    present = [c for c, n in c3["classification"]["per_class_support"].items() if n]
    pcf = c3["classification"]["per_class_f1"]
    cross = {"train": "train units at 1800 rpm only (frozen hyper-parameters, early stopping on val 1800)", "test": "test units at 3600 rpm (ball has none)",
             "macro_f1_over_classes_present": float(np.mean([pcf[c] for c in present])), "per_class_f1": pcf, "classes_present": present,
             "macro_f1_up2_definition_5_classes": c3["classification"]["macro_f1"], "false_alarm_rate": c3["classification"]["false_alarm_rate"],
             "missed_fault_rate": c3["classification"]["missed_fault_rate"], "windows": c3["classification"]["n_windows"]}
    real = v7_real.run(f, heads, fz)
    s7lim = 15.0
    out = {"meta": meta({"touch": "registered fresh test units scored once (4 pre-declared variants of the same unit ids)", "data_counts": counts,
                         "frozen_file": fpath.name, "frozen_sha256": frozen_sha, "test_seconds": round(time.perf_counter() - t0, 1)}),
           "frozen": f, "val_selected": fz["val_selected"], "test": test,
           "gates": {"S7_mape_le_15": bool(test["S7"]["S7_mape_pct"] <= s7lim), "IS3_macro_f1_ge_0.85": bool(test["IS3_classification"]["macro_f1"] >= 0.85),
                     "IS3_stage_error_le_1": bool(test["IS3_stage"]["max_abs_error"] <= 1),
                     "IS3_caught_by_stage3_all": bool(test["IS3_caught"]["caught_by_stage3"] == test["IS3_caught"]["rtf_bearings"])},
           "controls": {"trivial_baseline_log_linear_on_HI": {"S7_mape_pct": test["S7_baseline"]["S7_mape_pct"], "fit": f["baseline_control"],
                                                              "per_bearing_summary": test["S7_baseline"]["per_bearing_summary"]},
                        "wrong_shape": side["test_wrong"]},
           "side_by_side_same_units": side, "test_set_definitions": TEST_SETS,
           "reported_not_gated": {"cross_speed_1800_to_3600": cross, "per_rpm_macro_f1": test["per_rpm_macro_f1"], "real_checks": real}}
    marker.write_text(json.dumps({"frozen": fpath.name, "utc": out["meta"]["utc"]}, indent=1))
    tag = now_tag()
    (REPORTS_DIR / f"v7_node_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    from . import v7_report
    (REPORTS_DIR / f"v7_node_{tag}.md").write_text(v7_report.render(out))
    print(json.dumps({"gates": out["gates"], "S7": test["S7"]["S7_mape_pct"], "baseline": test["S7_baseline"]["S7_mape_pct"],
                      "F1": test["IS3_classification"]["macro_f1"], "stage_max": test["IS3_stage"]["max_abs_error"],
                      "caught": test["IS3_caught"]["caught_by_stage3"], "side": {k: (v["S7_mape_pct"], v["IS3_macro_f1"]) for k, v in side.items()},
                      "report": f"v7_node_{tag}"}, default=float, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.v7_node")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    s = sub.add_parser("select")
    s.add_argument("--threads", type=int, default=6)
    s.add_argument("--force", action="store_true")
    sub.add_parser("export")
    t = sub.add_parser("test")
    t.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    return {"data": cmd_data, "select": cmd_select, "export": cmd_export, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
