"""Contract v6_node: RUL + 5-class + stage heads on NODE features, trained on MORE rig_anchor simulated trajectories, with LONGER-context
trajectory features for the RUL / stage heads. Successor of v5_node (v5's FAILED test result stays in reports/v5_node_*.json forever).

Data tag (every number): [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed].
Test units are held-out SIMULATED TRAJECTORIES (not bearings) and re-use the same real anchor windows (test pool blocks) as the v5 test units.

    python -m aiengine.v6_node select   # val-only grids -> FROZEN pre-registration json + heads pickle
    python -m aiengine.v6_node export   # ONNX heads + contract + golden vectors + baseline + parity (val data only)
    python -m aiengine.v6_node test     # ONE scoring pass on the FRESH test units (rig_anchor_test2) + controls + real checks + report

Design motivation (disclosed): v5's design was motivated by its own validation split; v6's design (more units, long-context features,
wider post-process grid, classifier class-weight/context grid) was motivated by the DIAGNOSTICS of the v5 test pass (S7 23.4 %, stage error 2,
2 late catches, wrong early prediction on the real BA-1 trace). Therefore the OLD test units are not used again for anything; the arbiter is the
FRESH test set registered in splits.json section `rig_anchor_test2` BEFORE any v6 fit. Selection uses the rig_anchor VAL units only.
Metrics are aiengine.up2_metrics (S7 MAPE / IS3 macro-F1 / stage error / caught by stage 3): reused, not redefined.
Units of RUL: TABLE windows of 10 min; rul_hours = rul_windows x cadence_hours.
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
from . import splits, up2, up2b
from . import up2_metrics as M
from . import v5_node as V5
from . import v6_agg as AG
from . import v6_data as D
from .config import BEARING_FEATURES, CHECKPOINT_DIR, MODELS_DIR, REPORTS_DIR, SEED
from .rig_anchor import constants as K
from .rig_anchor import registry

COLS = V5.COLS
R = len(COLS)
WARM = 49                                           # every head is scored from the same window (>= the largest classifier context)
TAG, LIMITS = K.PROVENANCE, K.LIMITS
RUN_DIR = CHECKPOINT_DIR / "v6_node"
CONTRACT, CLS_ONNX, RUL_ONNX, GOLDEN, BASELINE = ("model_contract_v6_node.json", "v6_node_cls.onnx", "v6_node_rul.onnx",
                                                   "golden_vectors_v6_node.json", "baseline_v6_node.json")
CONTRACT_VERSION = "6-node.1:v6_node"
CLASS_ORDER = up2.CLASS_ORDER
MARKER = REPORTS_DIR / "v6_node_test_done.json"

# ---- grids (all reported; selection on VAL only) ----
TC_GRID = (25, 49)                                  # classifier context (rows, current included)
CLS_GRID = [dict(max_depth=d, learning_rate=0.1, power=pw) for d in (4, 6) for pw in (0.5, 1.0)]
GATE_TAUS = (0.0, 0.5)
NL_GRID = (48, 144, 288)
RUL_GRID = [dict(max_depth=d, mono=m, tail=t) for d in (4, 6) for m in (False, True) for t in (False, True)]   # tail: train only u <= 0.92 (the S7 window excludes the last 10 %)
Z_GRID = [dict(max_depth=d) for d in (4, 6)]
Z_CUTS = (0.0, -0.04, -0.08, -0.12)
STAGE_K = (1, 5, 9, 15, 21)
STAGE_STEP = (None, 1)
RUL_ALPHA = (1.0, 0.5, 0.25, 0.1, 0.05)
SCALE_EARLY = (0.6, 0.7, 0.85, 1.0, 1.2)
SCALE_LATE = (0.7, 0.8, 0.9, 1.0, 1.1)
FALSE_ALARM_CAP = 0.03


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def meta(extra: dict | None = None) -> dict:
    from .train import git_commit, git_dirty
    return {"provenance": TAG, "limits": LIMITS, "seed": SEED, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "command": "python -m aiengine.v6_node " + " ".join(sys.argv[1:]),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}


# --------------------------------------------------------------------------- data

def _verify_splits() -> dict:
    sp = splits.load()["sections"]
    if K.SPLIT_SECTION not in sp or not registry.verify() or D.SECTION not in sp or not D.verify():
        raise SystemExit("rig_anchor / rig_anchor_test2 split missing or changed in splits.json: refusing to fit or score")
    return sp


def load_trval() -> pd.DataFrame:
    """train (v5 240 units + 400 registered extra units) and val (v5 60 units). The v5 TEST units and the v6 fresh test units are NOT loaded."""
    sp = _verify_splits()
    cache = RUN_DIR / "tab_trval.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    old = pd.read_parquet(V5.DATA)
    if (old["rig_unit"].map(sp[K.SPLIT_SECTION]["units"]) != old["split"]).any():
        raise SystemExit("v5 table disagrees with the registered split")
    old = old[old["split"].isin(["train", "val"])]
    new = pd.read_parquet(D.DATA)
    new = new[new["v6_set"] == "train_extra"].drop(columns=["v6_set"])
    if (new["rig_unit"].map(sp[D.SECTION]["units"]) != "train").any():
        raise SystemExit("extra train units disagree with the registered split")
    tab = V5.engineer(pd.concat([old, new], ignore_index=True))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_pickle(cache)
    return tab


def load_fresh(path: Path = D.DATA, which: str = "fresh_test") -> pd.DataFrame:
    sp = _verify_splits()
    df = pd.read_parquet(path)
    df = df[df["v6_set"] == which].drop(columns=["v6_set"])
    ref = sp[D.SECTION]["units"] if which != "control_wrong_shape" else {u: "test" for u in sp[D.SECTION]["control_units"]}
    if (df["rig_unit"].map(ref) != "test").any():
        raise SystemExit("fresh/control units disagree with the registered split")
    return V5.engineer(df)


class FS:
    """Aggregate features for every window with >= WARM rows of history: classifier (context Tc, decoder) and RUL/z (context NL, v6_agg)."""

    def __init__(self, tab: pd.DataFrame, Tc: int, NL: int, tag: str = ""):
        self.tab, self.Tc, self.NL = tab, Tc, NL
        self.Xc, ic = self._cached(tag, "c", Tc, lambda: up2.unit_features(tab, Tc, True, columns=COLS, warm=WARM))
        self.Xr, ir = self._cached(tag, "r", NL, lambda: AG.unit_features(tab, NL, COLS, WARM))
        assert ic.equals(ir)
        self.T = up2.merge_truth(ic, tab, [c for c in V5.TRUTH if c not in ("unit_id", "window_index")])
        self.F = AG.n_features(R)

    @staticmethod
    def _cached(tag, kind, n, fn):
        if tag:
            p = RUN_DIR / "feat" / f"{tag}_{kind}{n}.npz"
            if p.exists():
                z = np.load(p, allow_pickle=True)
                return z["X"], pd.DataFrame({"unit_id": z["unit_id"], "window_index": z["window_index"]})
        X, idx = fn()
        if tag:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, X=X, unit_id=idx["unit_id"].to_numpy(), window_index=idx["window_index"].to_numpy())
        return X, idx

    def sub(self, mask: np.ndarray) -> "FS":
        o = object.__new__(FS)
        o.tab, o.Tc, o.NL, o.F = self.tab, self.Tc, self.NL, self.F
        o.Xc, o.Xr, o.T = self.Xc[mask], self.Xr[mask], self.T[mask].reset_index(drop=True)
        return o


def mono_constraints() -> tuple:
    names = AG.feature_names(COLS)
    mono = []
    for n in names:
        agg, col = n.split(":")
        if agg.startswith("frac_above") or (col in ("hi", "log_env_ratio", "log_bp_ratio") and agg in ("last", "mean3", "mean6", "mean24", "mean96", "mean", "max")):
            mono.append(-1)
        else:
            mono.append(0)
    return tuple(mono)


# --------------------------------------------------------------------------- heads

def fit_cls(tr: FS, va: FS, hp: dict, threads: int):
    p = dict(hp)
    power = p.pop("power")
    return up2.fit_classifier("xgb", tr.Xc, V5.y_cls(tr.T), va.Xc, V5.y_cls(va.T), power=power, threads=threads, params=p)


def fit_rul(tr: FS, va: FS, cfg: dict, threads: int):
    dtr, dva = V5.deg_mask(tr.T), V5.deg_mask(va.T)
    if cfg.get("tail"):
        dtr, dva = dtr & (tr.T["u_true"].to_numpy() <= 0.92), dva & (va.T["u_true"].to_numpy() <= 0.92)
    params = {"max_depth": cfg["max_depth"]}
    if cfg.get("mono"):
        params["monotone_constraints"] = mono_constraints()
    return up2.fit_regressor("xgb", tr.Xr[dtr], np.log1p(tr.T.loc[dtr, "rul_windows"].to_numpy()), va.Xr[dva],
                             np.log1p(va.T.loc[dva, "rul_windows"].to_numpy()), threads=threads,
                             wtr=up2.unit_balanced_weights(tr.T.loc[dtr, "unit_id"]), wva=up2.unit_balanced_weights(va.T.loc[dva, "unit_id"]),
                             params=params)


def fit_z(tr: FS, va: FS, cfg: dict, threads: int):
    return up2.fit_regressor("xgb", tr.Xr, tr.T["z_target"].to_numpy(), va.Xr, va.T["z_target"].to_numpy(), threads=threads,
                             params={"max_depth": cfg["max_depth"]})


def head_raw(rm, zm, S: FS, z_cut: float) -> pd.DataFrame:
    out = S.T[["unit_id", "window_index"]].copy()
    out["rul_pred_windows_raw"] = np.maximum(np.expm1(rm.predict(S.Xr)), 0.0)
    out["z_pred"] = zm.predict(S.Xr)
    out["stage_pred_raw"] = up2.stage_from_z(out["z_pred"].to_numpy(), z_cut)
    return out


def hi_col(S: FS) -> int:
    return AG.AGGS.index("mean6") * R                  # mean6:hi


def fit_hi_baseline(tr: FS) -> dict:
    d = V5.deg_mask(tr.T)
    x = tr.Xr[d, hi_col(tr)].astype(float)
    y = np.log1p(tr.T.loc[d, "rul_windows"].to_numpy())
    b, a = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b), "feature": "mean of the last 6 rows of the health index hi (aggregate 'mean6:hi')"}


def hi_baseline_predict(lin: dict, S: FS) -> np.ndarray:
    return np.maximum(np.expm1(lin["a"] + lin["b"] * S.Xr[:, hi_col(S)].astype(float)), 0.0)


def cls_probs(model, S: FS) -> np.ndarray:
    return model.predict_proba(S.Xc)


class _TF:                                              # tiny adapter: V5 scoring helpers only read .T / .tab
    def __init__(self, S: FS):
        self.T, self.tab = S.T, S.tab


def score_rul(S: FS, rp: pd.DataFrame) -> dict:
    return V5.score_rul_frame(_TF(S), rp)


def score_stage(S: FS, rp: pd.DataFrame) -> dict:
    return V5.score_stage_frame(_TF(S), rp)


def cls_scores(S: FS, pred: pd.DataFrame) -> dict:
    return V5.cls_scores(_TF(S), pred)


def apply_gate(P: np.ndarray, S: FS, tau: float) -> pd.DataFrame:
    return V5.apply_gate(P, _TF(S), tau)


def predict_all(heads: dict, fz: dict, S: FS) -> dict:
    f = fz["frozen"]
    pred = apply_gate(cls_probs(heads["cls"], S), S, f["gate_tau"])
    rp = V5.postprocess(head_raw(heads["rul"], heads["z"], S, f["z_cut"]), f["postprocess"])
    lin = heads["lin"]
    braw = S.T[["unit_id", "window_index"]].copy()
    braw["rul_pred_windows_raw"] = hi_baseline_predict(lin, S)
    braw["stage_pred_raw"] = 1
    bp = V5.postprocess(braw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": lin["rul_alpha"], "rul_scale_early": 1.0, "rul_scale_late": 1.0})
    return {"cls": pred, "rp": rp, "base": bp}


def full_scores(S: FS, p: dict) -> dict:
    ts = _TF(S)
    cs = V5.cls_scores(ts, p["cls"])
    rr, br = score_rul(S, p["rp"]), score_rul(S, p["base"])
    r = S.T.merge(p["rp"][["unit_id", "window_index", "stage_pred"]], on=["unit_id", "window_index"])
    st = M.stage(r[r["run_to_failure"]])
    drop = ("per_bearing_mape_true_onset_pct",)
    return {"S7": {k: v for k, v in rr.items() if k not in drop}, "S7_baseline": {k: v for k, v in br.items() if k not in drop},
            "IS3_classification": cs["classification"], "IS3_caught": {k: v for k, v in cs["caught"].items() if k != "per_unit"},
            "IS3_caught_per_unit": cs["caught"]["per_unit"], "IS3_stage": st,
            "stage_healthy_bearings": M.stage(r[~r["run_to_failure"]]) if (~r["run_to_failure"]).any() else {},
            "healthy_alarms": cs["healthy_alarms"], "per_rpm_macro_f1": cs["per_rpm_macro_f1"]}


# --------------------------------------------------------------------------- Bayes-style floor (oracle knowing the unit's shape + true onset)

def oracle_floor(tr: FS, S: FS, threads: int, va_for_stop: FS | None = None) -> dict:
    """S7 of a PRIVILEGED oracle: the same RUL regressor as the deployed head (same features, same unit-balanced log1p target, same scoring) but ALSO
    given the unit's hidden generator parameters (damage-curve a and p, late-amplification scale, anchor level u_anchor, and the true life in windows).
    It still sees the noisy engineered features only. It is NOT deployable; its S7 estimates how much S7 error remains when the unit's life/shape is KNOWN,
    i.e. the part due to feature noise alone (a lower-bound-style floor for any honest model). Fitted on TRAIN units, early-stopped on the given val set
    (the val set itself when S is the val set), scored with up2_metrics.rul."""
    def priv(X, FSx):
        t = FSx.tab.groupby("unit_id")[["shape_a", "shape_p", "late_scale", "anchor_u", "life_hours"]].first()
        t["life_hours"] = t["life_hours"] / K.CADENCE_H
        u = FSx.T["unit_id"].map(lambda x: x.rsplit(":B", 1)[0] + ":B1")
        pr = t.reindex(u).to_numpy(float)
        return np.concatenate([X, pr], axis=1)
    dtr = V5.deg_mask(tr.T)
    Xtr = priv(tr.Xr, tr)
    vs = va_for_stop or S
    dva = V5.deg_mask(vs.T)
    m, _ = up2.fit_regressor("xgb", Xtr[dtr], np.log1p(tr.T.loc[dtr, "rul_windows"].to_numpy()), priv(vs.Xr, vs)[dva],
                             np.log1p(vs.T.loc[dva, "rul_windows"].to_numpy()), threads=threads,
                             wtr=up2.unit_balanced_weights(tr.T.loc[dtr, "unit_id"]), wva=up2.unit_balanced_weights(vs.T.loc[dva, "unit_id"]), params={"max_depth": 6})
    raw = S.T[["unit_id", "window_index"]].copy()
    raw["rul_pred_windows_raw"] = np.maximum(np.expm1(m.predict(priv(S.Xr, S))), 0.0)
    raw["stage_pred_raw"] = 1
    out = {}
    for al in (1.0, 0.5):
        r = score_rul(S, V5.postprocess(raw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": al, "rul_scale_early": 1.0, "rul_scale_late": 1.0}))
        out[f"alpha{al}"] = {"S7_mape_pct": r["S7_mape_pct"], "per_bearing_summary": r["per_bearing_summary"]}
    best = min(out.values(), key=lambda v: v["S7_mape_pct"])
    return {"S7_mape_pct": best["S7_mape_pct"], "per_bearing_summary": best["per_bearing_summary"], "by_alpha": out,
            "definition": "privileged oracle: deployed-head regressor + hidden generator parameters (a, p, late_scale, anchor_u, life); noisy features only; NOT deployable"}


# --------------------------------------------------------------------------- select (VAL only) -> frozen pre-registration

def cmd_select(a) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    frozen_files = sorted(REPORTS_DIR.glob("v6_node_frozen_*.json"))
    if frozen_files and not a.force and not a.smoke:
        raise SystemExit(f"choices already frozen ({frozen_files[-1].name}); not re-selecting (use --force before any test pass)")
    if MARKER.exists():
        raise SystemExit("v6 test already scored: selection is closed")
    t0 = time.perf_counter()
    tab = load_trval()
    ttr, tva = tab[tab["split"] == "train"], tab[tab["split"] == "val"]
    global TC_GRID, CLS_GRID, GATE_TAUS, NL_GRID, RUL_GRID, Z_GRID, Z_CUTS, STAGE_K, RUL_ALPHA, SCALE_EARLY, SCALE_LATE
    pref = "v6_node_frozen_"
    if a.smoke:                                              # code-path check only: tiny data, tiny grids, NOT a result, never frozen
        ttr = ttr[ttr["rig_unit"].isin(sorted(ttr["rig_unit"].unique())[::8])]
        tva = tva[tva["rig_unit"].isin(sorted(tva["rig_unit"].unique())[::3])]
        TC_GRID, CLS_GRID, GATE_TAUS, NL_GRID = (25,), CLS_GRID[:1], (0.0,), (48, 144)
        RUL_GRID, Z_GRID, Z_CUTS, STAGE_K, RUL_ALPHA, SCALE_EARLY, SCALE_LATE = RUL_GRID[:2], Z_GRID[:1], (0.0,), (1, 5), (1.0, 0.5), (0.7,), (1.0,)
        pref = "v6_node_SMOKE_"
    counts = {"train": {"units": int(ttr["rig_unit"].nunique())}, "val": {"units": int(tva["rig_unit"].nunique())}}
    th = a.threads
    # ---------------- classifier + gate (context Tc x hp x class-weight power x gate tau)
    cls_rows, best_c = [], None
    for Tc in TC_GRID:
        S_tr, S_va = FS(ttr, Tc, NL_GRID[0], "" if a.smoke else "tr"), FS(tva, Tc, NL_GRID[0], "" if a.smoke else "va")
        for hp in CLS_GRID:
            m, info = fit_cls(S_tr, S_va, hp, th)
            P = cls_probs(m, S_va)
            for tau in GATE_TAUS:
                s = cls_scores(S_va, apply_gate(P, S_va, tau))
                row = {"Tc": Tc, "hp": hp, "gate_tau": tau, "best_iteration": info["best_iteration"], "val_macro_f1": s["classification"]["macro_f1"],
                       "val_false_alarm": s["classification"]["false_alarm_rate"], "val_missed_fault": s["classification"]["missed_fault_rate"],
                       "val_per_class_f1": s["classification"]["per_class_f1"],
                       "val_caught_by_stage3": f"{s['caught']['caught_by_stage3']}/{s['caught']['rtf_bearings']}",
                       "val_caught_n": s["caught"]["caught_by_stage3"]}
                cls_rows.append(row)
                print("cls", json.dumps({k: v for k, v in row.items() if k != "val_per_class_f1"}), flush=True)
                key = (row["val_false_alarm"] <= FALSE_ALARM_CAP, row["val_macro_f1"], row["val_caught_n"], -info["best_iteration"])
                if best_c is None or key > best_c[0]:
                    best_c = (key, Tc, hp, tau, m, info)
        del S_tr, S_va
    # ---------------- RUL regressor: context NL x depth x monotone
    rul_rows, best_r = [], None
    for NL in NL_GRID:
        S_tr, S_va = FS(ttr, TC_GRID[0], NL, "" if a.smoke else "tr"), FS(tva, TC_GRID[0], NL, "" if a.smoke else "va")
        zbase = None
        for cfg in RUL_GRID:
            rm, info = fit_rul(S_tr, S_va, cfg, th)
            if zbase is None:
                zbase = fit_z(S_tr, S_va, Z_GRID[0], th)[0]
            raw = head_raw(rm, zbase, S_va, 0.0)
            s7 = {}
            for al in (1.0, 0.5):
                s7[al] = score_rul(S_va, V5.postprocess(raw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": al, "rul_scale_early": 1.0, "rul_scale_late": 1.0}))["S7_mape_pct"]
            row = {"NL": NL, **cfg, "best_iteration": info["best_iteration"], "val_S7_mape_pct_alpha1": s7[1.0], "val_S7_mape_pct_alpha0.5": s7[0.5],
                   "val_S7_selection": min(s7.values())}
            rul_rows.append(row)
            print("rul", json.dumps(row), flush=True)
            if best_r is None or row["val_S7_selection"] < best_r[0]:
                best_r = (row["val_S7_selection"], NL, cfg, rm, info)
        del S_tr, S_va
    _, NLs, cfg_r, rm, rinfo = best_r
    # ---------------- z / stage head on the selected context + stage post-process; RUL post-process
    S_tr, S_va = FS(ttr, TC_GRID[0], NLs, "" if a.smoke else "tr"), FS(tva, TC_GRID[0], NLs, "" if a.smoke else "va")
    vrt = S_va.sub(S_va.T["run_to_failure"].to_numpy(bool))
    stage_rows, best_z = [], None
    for zc in Z_GRID:
        zm, zinfo = fit_z(S_tr, S_va, zc, th)
        for cut in Z_CUTS:
            raw = head_raw(rm, zm, vrt, cut)
            for k in STAGE_K:
                for step in STAGE_STEP:
                    rp = V5.postprocess(raw, {"stage_k": k, "stage_max_step": step, "rul_alpha": 1.0, "rul_scale_early": 1.0, "rul_scale_late": 1.0})
                    st = score_stage(vrt, rp)
                    stage_rows.append({"z_head": zc, "z_cut": cut, "stage_k": k, "stage_max_step": step, "val_stage_max": st["max_abs_error"],
                                       "val_stage_n_ge2": st["windows_error_ge2"], "val_stage_within1": st["within1_frac"], "val_stage_mae": st["mae"]})
                    key = V5.stage_key(st)
                    if best_z is None or key < best_z[0]:
                        best_z = (key, zc, cut, k, step, zm, zinfo)
    print("stage best", json.dumps({"z_head": best_z[1], "z_cut": best_z[2], "k": best_z[3], "step": best_z[4], "key": best_z[0]}), flush=True)
    _, zc, cut, k, step, zm, zinfo = best_z
    raw = head_raw(rm, zm, vrt, cut)
    pp_rows, best_p = [], None
    for al in RUL_ALPHA:
        for se in SCALE_EARLY:
            for sl in SCALE_LATE:
                pp = {"stage_k": k, "stage_max_step": step, "rul_alpha": al, "rul_scale_early": se, "rul_scale_late": sl}
                s7 = score_rul(vrt, V5.postprocess(raw, pp))["S7_mape_pct"]
                pp_rows.append({**pp, "val_S7_mape_pct": s7})
                if best_p is None or s7 < best_p[0]:
                    best_p = (s7, pp)
    pp_sel = best_p[1]
    lin = fit_hi_baseline(S_tr)
    braw = vrt.T[["unit_id", "window_index"]].copy()
    braw["rul_pred_windows_raw"] = hi_baseline_predict(lin, vrt)
    braw["stage_pred_raw"] = 1
    base_rows, best_b = [], None
    for al in RUL_ALPHA:
        s7 = score_rul(vrt, V5.postprocess(braw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": al, "rul_scale_early": 1.0, "rul_scale_late": 1.0}))["S7_mape_pct"]
        base_rows.append({"alpha": al, "val_S7_mape_pct": s7})
        if best_b is None or s7 < best_b[0]:
            best_b = (s7, al)
    lin["rul_alpha"] = best_b[1]
    floor = oracle_floor(S_tr, S_va, th)
    # ---------------- val numbers of the deployed configuration (classifier context may differ from the RUL one: scored separately, same rows)
    _, Tcs, hpc, tau, cm, cinfo = best_c
    Sc_va = FS(tva, Tcs, NLs, "" if a.smoke else "va")
    heads = {"cls": cm, "rul": rm, "z": zm, "lin": lin}
    fz_tmp = {"frozen": {"gate_tau": tau, "z_cut": cut, "postprocess": pp_sel}}
    val_all = full_scores(Sc_va, predict_all(heads, fz_tmp, Sc_va))
    val = {"S7_mape_pct": val_all["S7"]["S7_mape_pct"], "S7_baseline_mape_pct": val_all["S7_baseline"]["S7_mape_pct"],
           "S7_per_bearing_summary": val_all["S7"]["per_bearing_summary"], "oracle_floor_S7_mape_pct": floor["S7_mape_pct"],
           "stage": {k_: v for k_, v in val_all["IS3_stage"].items() if k_ != "error_hist"},
           "is3": {"macro_f1": val_all["IS3_classification"]["macro_f1"], "false_alarm_rate": val_all["IS3_classification"]["false_alarm_rate"],
                   "per_class_f1": val_all["IS3_classification"]["per_class_f1"], "caught": val_all["IS3_caught"]},
           "healthy_alarms": val_all["healthy_alarms"]}
    with open(RUN_DIR / ("smoke_heads.pkl" if a.smoke else "heads.pkl"), "wb") as f:
        pickle.dump(heads, f)
    frozen = {
        "meta": meta({"phase": "PRE-REGISTRATION: frozen choices written before the FRESH test units (rig_anchor_test2) are scored",
                      "data_counts": counts, "elapsed_s": round(time.perf_counter() - t0, 1), "n_threads": th,
                      "fresh_test_registered_in": "splits.json section rig_anchor_test2 (commit 48c6643, before any v6 fit)"}),
        "feature_columns": COLS, "warm_windows": WARM,
        "frozen": {"classifier_hp": hpc, "classifier_T": Tcs, "classifier_best_iteration": cinfo["best_iteration"], "gate_tau": tau,
                   "gate_signal": "family", "rul_head": cfg_r, "rul_NL": NLs, "rul_best_iteration": rinfo["best_iteration"],
                   "z_head": zc, "z_best_iteration": zinfo["best_iteration"], "z_cut": cut, "postprocess": pp_sel, "baseline_control": lin,
                   "aggregator": {"module": "aiengine/v6_agg.py", "aggs": AG.AGGS, "hi_levels": AG.HI_LEVELS, "env_levels": AG.ENV_LEVELS}},
        "selection_rule": {"classifier": f"max val macro-F1 among configs with val false-alarm <= {FALSE_ALARM_CAP}, tie -> more windows caught by stage 3 -> fewer trees",
                           "rul": "min val S7 MAPE (unfiltered, no scale; min over alpha in {1.0, 0.5})",
                           "stage": "min val (max stage error, #windows error>=2, -within1, MAE)",
                           "rul_post": "min val S7 MAPE over alpha x scale_early x scale_late", "baseline": "OLS fit on train; alpha by val S7"},
        "val_grid": {"classifier": cls_rows, "rul": rul_rows, "stage": stage_rows, "rul_postprocess": pp_rows, "baseline_alpha": base_rows},
        "oracle_floor_val": floor, "val_selected": val,
    }
    path = (RUN_DIR / f"{pref}{now_tag()}.json") if a.smoke else REPORTS_DIR / f"{pref}{now_tag()}.json"
    path.write_text(json.dumps(frozen, indent=1, default=float))
    print("FROZEN ->", path.name, json.dumps(frozen["frozen"], default=float))
    print("VAL", json.dumps(val, default=float))
    return 0


def load_frozen() -> tuple[dict, Path]:
    fs = sorted(REPORTS_DIR.glob("v6_node_frozen_*.json"))
    if not fs:
        raise SystemExit("run `select` first")
    return json.loads(fs[-1].read_text()), fs[-1]


def load_heads() -> dict:
    with open(RUN_DIR / "heads.pkl", "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- export (val data only for parity / golden)

def build_models(heads: dict, fz: dict):
    from onnx import helper, numpy_helper, TensorProto
    from . import up2_export as X
    f = fz["frozen"]
    m_cls = X.build_cls(heads["cls"], f["classifier_T"], R, X.importances_per_column(heads["cls"], R))
    NL = f["rul_NL"]
    agg = AG.aggregator_graph(NL, R, out_name="feat")
    F = AG.n_features(R)
    rn, ri, routs, rops = X._inline(X._convert(heads["rul"], F), "r_", "feat")
    zn, zi, zouts, zops = X._inline(X._convert(heads["z"], F), "z_", "feat")
    rv, zv = list(routs.values())[0], list(zouts.values())[0]
    post = [helper.make_node("Exp", [rv], ["p_e"]), helper.make_node("Sub", ["p_e", "p_one"], ["p_em1"]), helper.make_node("Max", ["p_em1", "p_zero"], ["p_rul"]),
            helper.make_node("Clip", [zv, "p_zero", "p_one"], ["p_zc"]), helper.make_node("Mul", ["p_zc", "p_five"], ["p_z5"]),
            helper.make_node("Floor", ["p_z5"], ["p_fl"]), helper.make_node("Add", ["p_fl", "p_two"], ["p_st0"]), helper.make_node("Min", ["p_st0", "p_six"], ["p_st1"]),
            helper.make_node("Less", [zv, "p_cut"], ["p_pre"]), helper.make_node("Where", ["p_pre", "p_one", "p_st1"], ["p_stage"]),
            helper.make_node("Concat", ["p_rul", zv, "p_stage"], ["rul_head"], axis=1)]
    c = lambda v, n: numpy_helper.from_array(np.array(v, np.float32), n)   # noqa: E731
    inits = list(agg.graph.initializer) + ri + zi + [c(1.0, "p_one"), c(0.0, "p_zero"), c(5.0, "p_five"), c(2.0, "p_two"), c(6.0, "p_six"), c(float(f["z_cut"]), "p_cut")]
    outputs = [helper.make_tensor_value_info("rul_head", TensorProto.FLOAT, ["batch", 3])]
    m_rul = X._model(list(agg.graph.node) + rn + zn + post, inits, list(agg.graph.input), outputs, list(agg.opset_import) + rops + zops, "v6_rul")
    return m_cls, m_rul


def onnx_head_outputs(S_tab: pd.DataFrame, f: dict, onnx_cls: Path, onnx_rul: Path, max_units: int = 6):
    import onnxruntime as ort
    from . import up2_window as uw
    sc = ort.InferenceSession(str(onnx_cls), providers=["CPUExecutionProvider"])
    sr = ort.InferenceSession(str(onnx_rul), providers=["CPUExecutionProvider"])
    units = S_tab["unit_id"].drop_duplicates().iloc[:max_units]
    out = {"p": [], "rul": [], "z": [], "st": []}
    Tc, NL = f["classifier_T"], f["rul_NL"]
    for uid in units:
        g = S_tab[S_tab["unit_id"] == uid].sort_values("window_index")
        rows = g[COLS].to_numpy(np.float64)
        bc, _ = uw.blocks(rows, Tc)
        br, _ = uw.blocks(rows, NL)
        for k in range(WARM - 1, len(g)):
            lg = sc.run(["logits"], {"encoder_cont": bc[k:k + 1, :-1], "decoder_cont": bc[k:k + 1, -1:]})[0]
            o = sr.run(["rul_head"], {"encoder_cont": br[k:k + 1]})[0][0]
            out["p"].append(np.exp(lg[0, 0])); out["rul"].append(o[0]); out["z"].append(o[1]); out["st"].append(o[2])
    return units, {k: np.asarray(v) for k, v in out.items()}


def parity(heads: dict, fz: dict, va_tab: pd.DataFrame, onnx_cls: Path, onnx_rul: Path, max_units: int = 6) -> dict:
    f = fz["frozen"]
    units, o = onnx_head_outputs(va_tab, f, onnx_cls, onnx_rul, max_units)
    Sm = FS(va_tab[va_tab["unit_id"].isin(units)], f["classifier_T"], f["rul_NL"])
    P = cls_probs(heads["cls"], Sm)
    raw = head_raw(heads["rul"], heads["z"], Sm, f["z_cut"])
    # units are iterated in sorted order inside FS; align the ONNX rows (iterated in first-appearance order) by key
    order = {u: i for i, u in enumerate(units)}
    key = Sm.T["unit_id"].map(order).to_numpy() * 10 ** 6 + Sm.T["window_index"].to_numpy()
    idx = np.argsort(key, kind="stable")
    P, rr, zz, ss = P[idx], raw["rul_pred_windows_raw"].to_numpy()[idx], raw["z_pred"].to_numpy()[idx], raw["stage_pred_raw"].to_numpy()[idx]
    rel = np.abs(o["rul"] - rr) / np.maximum(rr, 1.0)
    return {"units": int(len(units)), "windows": int(len(o["p"])), "cls_prob_max_abs_diff": float(np.abs(o["p"] - P).max()),
            "cls_argmax_mismatch": int((o["p"].argmax(1) != P.argmax(1)).sum()), "rul_windows_max_rel_diff": float(rel.max()),
            "z_max_abs_diff": float(np.abs(o["z"] - zz).max()), "stage_mismatch": int((o["st"].astype(int) != ss).sum()),
            "reference": "aggregator via ORT + native xgboost predict (float32 features); units are VAL units"}


def contract_dict(fz: dict, par: dict, sizes: dict) -> dict:
    from .train import git_commit
    con = json.loads((MODELS_DIR / "model_contract_v5_node.json").read_text())      # same layout the NodeEngine/C# host already loads
    f = fz["frozen"]
    pp = f["postprocess"]
    Tc, NL = f["classifier_T"], f["rul_NL"]
    con["contract_version"] = CONTRACT_VERSION
    con["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con["generated_by"] = "AI-engine/aiengine/v6_node.py export (layout of model_contract_v5_node.json; heads, inputs, postprocess replaced)"
    con["provenance"] = {"synthetic": True, "tag": TAG, "limits": LIMITS,
                         "datasets": ["rig_anchor train units (240 v5 + 400 registered extra); val units for selection"],
                         "note": f"{TAG} gradient-boosted heads trained on rig_anchor train units only; selected on val units; FRESH test units (splits.json "
                                 "rig_anchor_test2) scored once (reports/v6_node_*.json). NOT a model of real run-to-failure data.",
                         "seed": SEED, "git_commit": git_commit(), "command": "python -m aiengine.v6_node select/export/test",
                         "frozen_choices_file": fz["_file"], "frozen": f}
    ident = {c: {"center": 0.0, "scale": 1.0, "kind": "identity"} for c in COLS}
    t, n = con["models"]["tft"], con["models"]["nhits"]
    t.update({"onnx": CLS_ONNX, "family": "gradient-boosted trees (xgb) in the TFT slot, trained on rig_anchor simulated trajectories " + TAG,
              "encoder_length": Tc - 1, "columns": COLS, "scalers": ident, "warmup_windows": Tc, "size_kb": sizes["cls_kb"], "parity": par,
              "present_classes": list(CLASS_ORDER)})
    n.update({"onnx": RUL_ONNX, "family": "gradient-boosted trees (rul + z regressors, xgb) in the N-HiTS slot, long-context aggregator " + TAG,
              "encoder_length": NL, "prediction_length": 1, "columns": COLS, "scalers": ident, "size_kb": sizes["rul_kb"],
              "output": "rul_head [B,3] = [rul_windows, z, stage]; rul_windows in TABLE windows (one simulated 10-min snapshot per window); "
                        f"stage = 1 if z < {f['z_cut']} else min(6, 2 + floor(5 clip(z, 0, 1)))",
              "in_graph_features": {"aggregates": AG.AGGS, "hi_levels": AG.HI_LEVELS, "env_levels": AG.ENV_LEVELS, "reference_impl": "aiengine/v6_agg.py"},
              "parity": par})
    con["rul"].update({"method": "direct_v3",
                       "procedure": "rul_hours = postprocess(rul_head[0]) * cadence_hours (median spacing of the last <=48 window times); stage = postprocess(rul_head[2]). "
                                    "RUL is meaningful when stage >= 2 and the bearing is the faulted one. CADENCE: the head is trained on windows spaced "
                                    f"{K.CADENCE_H * 60:.0f} min apart ({TAG}) with a {NL}-window context (left-padded with the oldest row while the buffer fills); "
                                    "hosts must feed one window per snapshot with t20_ms spaced accordingly.",
                       "table_cadence_hours": K.CADENCE_H, "interval_rel": 0.25})
    con["stage"] = (f"rul_head[2] from the z head: 1 if z < {f['z_cut']} else min(6, 2 + floor(5 clip(z,0,1))); deployed: postprocess.stage "
                    f"(median {pp['stage_k']}, max_step {pp['stage_max_step']})")
    con["postprocess"] = up2b.postprocess_contract(pp)
    con["postprocess"]["selected_on"] = "rig_anchor val units only"
    if f["gate_tau"] > 0:
        con["cross_bearing_gate"] = {"tau": float(f["gate_tau"]), "signal": "family", "selected_on": "rig_anchor val units only",
                                     "family_of_class": {c: f"log_{fam}_ratio" for c, fam in A.FAMILY_OF.items()}}
    else:
        con.pop("cross_bearing_gate", None)
    con["real_data"] = {"note": "no real windows are used for fitting v6_node; real runs are used only as the anchor pools of rig_anchor "
                                "and for reported checks (not independent: they are the anchor source)"}
    con["disclosures"] = dict(con.get("disclosures", {}))
    con["disclosures"]["provenance"] = con["provenance"]["note"]
    con["disclosures"]["limits"] = LIMITS
    return con


def cmd_export(a) -> int:
    import onnx
    from .engine import HybridEngine
    fz, path = load_frozen()
    fz["_file"] = path.name
    heads = load_heads()
    tab = load_trval()
    va = tab[tab["split"] == "val"]
    m_cls, m_rul = build_models(heads, fz)
    onnx.save(m_cls, str(MODELS_DIR / CLS_ONNX))
    onnx.save(m_rul, str(MODELS_DIR / RUL_ONNX))
    par = parity(heads, fz, va, MODELS_DIR / CLS_ONNX, MODELS_DIR / RUL_ONNX)
    sizes = {"cls_kb": round((MODELS_DIR / CLS_ONNX).stat().st_size / 1024, 1), "rul_kb": round((MODELS_DIR / RUL_ONNX).stat().st_size / 1024, 1)}
    con = contract_dict(fz, par, sizes)
    (MODELS_DIR / CONTRACT).write_text(json.dumps(con, indent=2, default=float))
    base = json.loads((MODELS_DIR / "baseline_v4_real.json").read_text())
    base["contract_version"] = CONTRACT_VERSION
    base["note"] = "rig commissioning baselines [MEASURED on rig] copied from baseline_v4_real.json; same policy"
    (MODELS_DIR / BASELINE).write_text(json.dumps(base, indent=1))
    uid = V5.pick_golden_unit(tab)
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

def cmd_test(a) -> int:
    if MARKER.exists():
        raise SystemExit(f"v6 test already scored once ({MARKER.name}); not re-scoring")
    from . import v6_real
    fz, fpath = load_frozen()
    fz["_file"] = fpath.name
    f = fz["frozen"]
    heads = load_heads()
    t0 = time.perf_counter()
    frozen_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
    onnx_sha = {n: hashlib.sha256((MODELS_DIR / n).read_bytes()).hexdigest()[:16] for n in (CLS_ONNX, RUL_ONNX, CONTRACT)}
    print("scoring FRESH TEST once with", fpath.name, frozen_sha[:12], flush=True)
    tte = load_fresh()
    te = FS(tte, f["classifier_T"], f["rul_NL"])
    counts = {"fresh_test": {"units": int(tte["rig_unit"].nunique()), "feature_rows": int(len(te.T)),
                             "faulted_run_to_failure_bearings": int(te.T[te.T["run_to_failure"]]["unit_id"].nunique())}}
    p = predict_all(heads, fz, te)
    test = full_scores(te, p)
    # oracle floor on the fresh test (descriptive; NOT used for any choice)
    ttr = load_trval()
    # ---- controls: wrong-shape (new seeds), trivial baseline, cross-speed
    wtab = load_fresh(D.DATA_WRONG, "control_wrong_shape")
    wte = FS(wtab, f["classifier_T"], f["rul_NL"])
    wsc = full_scores(wte, predict_all(heads, fz, wte))
    wrong = {"S7_mape_pct": wsc["S7"]["S7_mape_pct"], "S7_baseline_mape_pct": wsc["S7_baseline"]["S7_mape_pct"],
             "S7_per_bearing_summary": wsc["S7"]["per_bearing_summary"], "IS3_macro_f1": wsc["IS3_classification"]["macro_f1"],
             "IS3_stage_max": wsc["IS3_stage"]["max_abs_error"], "IS3_stage_within1": wsc["IS3_stage"]["within1_frac"], "IS3_caught": wsc["IS3_caught"],
             "units": int(wtab["rig_unit"].nunique()),
             "note": "80 NEW units (u3000-u3079, new seeds) with test composition/pool and near-linear damage curves (a 0.85-0.95, p 1.0-1.2) instead of "
                     "the assumed accelerating family; frozen heads, no refit"}
    # v5 frozen heads on the SAME fresh units (apples to apples; the old v5 test number is NOT re-scored)
    v5_fresh = v5_on_fresh(tte)
    # cross-speed control: frozen hp, 1800-rpm train units only
    S_trc = FS(ttr[ttr["split"] == "train"], f["classifier_T"], f["rul_NL"], "tr")
    S_vac = FS(ttr[ttr["split"] == "val"], f["classifier_T"], f["rul_NL"], "va")
    m1 = S_trc.T["rpm_mode"].to_numpy() == 1800
    v1 = S_vac.T["rpm_mode"].to_numpy() == 1800
    hp = dict(f["classifier_hp"]); power = hp.pop("power")
    m1800 = up2.fit_classifier("xgb", S_trc.Xc[m1], V5.y_cls(S_trc.T[m1]), S_vac.Xc[v1], V5.y_cls(S_vac.T[v1]), power=power, threads=a.threads, params=hp)[0]
    te3 = te.sub(te.T["rpm_mode"].to_numpy() == 3600)
    c3 = cls_scores(te3, apply_gate(m1800.predict_proba(te3.Xc), te3, f["gate_tau"]))
    present = [c for c, n in c3["classification"]["per_class_support"].items() if n]
    pcf = c3["classification"]["per_class_f1"]
    cross = {"train": "train units at 1800 rpm only (frozen hyper-parameters, early stopping on val 1800)", "test": "fresh test units at 3600 rpm (ball has none)",
             "macro_f1_over_classes_present": float(np.mean([pcf[c] for c in present])), "per_class_f1": pcf, "classes_present": present,
             "macro_f1_up2_definition_5_classes": c3["classification"]["macro_f1"], "false_alarm_rate": c3["classification"]["false_alarm_rate"],
             "missed_fault_rate": c3["classification"]["missed_fault_rate"], "windows": c3["classification"]["n_windows"]}
    floor = oracle_floor(S_trc, te, a.threads, va_for_stop=S_vac)
    real = v6_real.run(f, heads, fz)
    v5rep = json.loads(sorted(REPORTS_DIR.glob("v5_node_2*.json"))[-1].read_text())
    out = {"meta": meta({"touch": "FRESH test units (rig_anchor_test2) scored once", "data_counts": counts, "frozen_file": fpath.name, "frozen_sha256": frozen_sha,
                         "onnx_sha256_16": onnx_sha, "test_seconds": round(time.perf_counter() - t0, 1)}),
           "frozen": f, "val_selected": fz["val_selected"], "test": test,
           "gates": {"S7_mape_le_15": bool(test["S7"]["S7_mape_pct"] <= 15.0), "IS3_macro_f1_ge_0.85": bool(test["IS3_classification"]["macro_f1"] >= 0.85),
                     "IS3_stage_error_le_1": bool(test["IS3_stage"]["max_abs_error"] <= 1),
                     "IS3_caught_by_stage3_all": bool(test["IS3_caught"]["caught_by_stage3"] == test["IS3_caught"]["rtf_bearings"])},
           "controls": {"trivial_baseline_log_linear_on_HI": {"S7_mape_pct": test["S7_baseline"]["S7_mape_pct"], "fit": f["baseline_control"],
                                                              "per_bearing_summary": test["S7_baseline"]["per_bearing_summary"]},
                        "wrong_shape": wrong, "oracle_floor_fresh_test": floor},
           "v5_vs_v6": {"v5_registered_test_scored_once": {"units": 60, "S7": v5rep["test"]["S7"]["S7_mape_pct"], "F1": v5rep["test"]["IS3_classification"]["macro_f1"],
                                                            "stage_max": v5rep["test"]["IS3_stage"]["max_abs_error"], "caught": v5rep["test"]["IS3_caught"]["caught_by_stage3"],
                                                            "report": "reports/" + sorted(REPORTS_DIR.glob("v5_node_2*.json"))[-1].name},
                        "v5_frozen_heads_on_fresh_units": v5_fresh,
                        "v6_fresh_test": {"S7": test["S7"]["S7_mape_pct"], "F1": test["IS3_classification"]["macro_f1"], "stage_max": test["IS3_stage"]["max_abs_error"],
                                          "caught": test["IS3_caught"]["caught_by_stage3"]}},
           "reported_not_gated": {"cross_speed_1800_to_3600": cross, "per_rpm_macro_f1": test["per_rpm_macro_f1"], "real_checks": real}}
    MARKER.write_text(json.dumps({"frozen": fpath.name, "utc": out["meta"]["utc"]}, indent=1))
    tag = now_tag()
    (REPORTS_DIR / f"v6_node_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    from . import v6_report
    (REPORTS_DIR / f"v6_node_{tag}.md").write_text(v6_report.render(out, fz))
    print(json.dumps({"gates": out["gates"], "S7": test["S7"]["S7_mape_pct"], "baseline": test["S7_baseline"]["S7_mape_pct"],
                      "F1": test["IS3_classification"]["macro_f1"], "stage_max": test["IS3_stage"]["max_abs_error"],
                      "caught": test["IS3_caught"]["caught_by_stage3"], "report": f"v6_node_{tag}"}, default=float, indent=1))
    return 0


def v5_on_fresh(tte: pd.DataFrame) -> dict:
    fz5, p5 = V5.load_frozen()
    with open(CHECKPOINT_DIR / "v5_node" / "heads.pkl", "rb") as fh:
        h5 = pickle.load(fh)
    S5 = V5.FeatSet(tte)
    sc = V5.full_scores(S5, V5.predict_all(h5, fz5, S5))
    return {"note": "v5 frozen heads (reports/" + p5.name + ") scored on the SAME fresh units; a new set, NOT a re-score of the v5 registered test units",
            "S7": sc["S7"]["S7_mape_pct"], "F1": sc["IS3_classification"]["macro_f1"], "stage_max": sc["IS3_stage"]["max_abs_error"],
            "caught": f"{sc['IS3_caught']['caught_by_stage3']}/{sc['IS3_caught']['rtf_bearings']}", "S7_baseline": sc["S7_baseline"]["S7_mape_pct"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.v6_node")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--force", action="store_true")
    s.add_argument("--smoke", action="store_true")
    sub.add_parser("export")
    t = sub.add_parser("test")
    t.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    return {"select": cmd_select, "export": cmd_export, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
