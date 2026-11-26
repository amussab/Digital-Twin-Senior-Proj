"""Phase-1 research for S7 (read: reports/s7_research_*.md). TRAIN/VAL units only; test units are never built here.
[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape + life variance assumed]

    python -m aiengine.v7_research job <tag> [--profile P] [--noise NU] [--threads N] [--full]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import up2
from . import v5_node as V5
from . import v7_node as V7
from .rig_anchor import constants as K
from .rig_anchor import constants_profiles as CP
from .rig_anchor import generator as G
from .rig_anchor import pools as P
from .rig_anchor import scale as SC

EXTRA = ["t_hours", "onset_index", "onset_t_hours", "shape_a", "shape_p", "late_scale", "unit_rpm", "anchor_u"]
TAG = "[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape + life variance assumed]"


def noisy_pools(nu: float) -> P.Pools:
    """Pools whose window-to-window variation inside each real run is shrunk (log domain) by factor nu around the run median (nu=1: real)."""
    pl = P.Pools()
    if nu == 1.0:
        return pl
    for d in list(pl.healthy.values()) + list(pl.fault.values()):
        for segs in d.values():
            for s in segs:
                lx = np.log(np.maximum(s.X, K.LOG_FLOOR))
                m = np.median(lx, axis=0, keepdims=True)
                s.X = np.exp(m + nu * (lx - m))
    return pl


def prep(profile, nu):
    import pickle
    cp = V7.CACHE / f"prep_{profile}_{nu}.pkl"
    if cp.exists():
        return pickle.load(open(cp, "rb"))
    t0 = time.perf_counter()
    df = G.build(noisy_pools(nu), [p for p in G.plan_units() if p["split"] in ("train", "val")], profile=CP.get(profile))
    tab = V5.engineer(df)
    out = {}
    for s in ("train", "val"):
        tb = tab[tab["split"] == s]
        X, idx = up2.unit_features(tb, V7.NL, False, columns=V7.COLS, warm=V5.T_CLS, levels=True)
        T = up2.merge_truth(idx, tb, [c for c in V5.TRUTH + EXTRA if c not in ("unit_id", "window_index")])
        out[s] = (X, T)
    print(f"prep {profile} nu={nu}: {time.perf_counter() - t0:.0f}s", flush=True)
    pickle.dump(out, open(cp, "wb"), protocol=4)
    return out


def sub(X, T):
    o = type("S", (), {})()
    o.Xr, o.T, o.F = X, T, X.shape[1]
    o.xr = lambda levels: X
    return o


def true_onset_t(T: pd.DataFrame) -> pd.Series:
    return T.assign(_t=np.where(T["u_true"] >= 0, T["t_hours"], np.nan)).groupby("unit_id")["_t"].min()


def add_clock(X, T, kind):
    t = T["t_hours"].to_numpy(float)
    if kind == "true":                                   # ORACLE: hours since the generator's true onset
        ton = T["unit_id"].map(true_onset_t(T)).to_numpy(float)
    else:                                                # causal detected onset (labels.detect_onset on the HI)
        ton = T["onset_t_hours"].where(T["onset_index"] >= 0).to_numpy(float)
    e = np.where(np.isnan(ton), -1.0, t - ton)
    return np.concatenate([X, e[:, None].astype(np.float32)], 1)


def fit_eval(tr, va, X_tr, X_va, threads, depth=6):
    dtr, dva = V5.deg_mask(tr.T), V5.deg_mask(va.T)
    m, info = up2.fit_regressor("xgb", X_tr[dtr], np.log1p(tr.T.loc[dtr, "rul_windows"].to_numpy()), X_va[dva],
                                np.log1p(va.T.loc[dva, "rul_windows"].to_numpy()), threads=threads,
                                wtr=up2.unit_balanced_weights(tr.T.loc[dtr, "unit_id"]), wva=up2.unit_balanced_weights(va.T.loc[dva, "unit_id"]),
                                params={"max_depth": depth})
    raw = va.T[["unit_id", "window_index"]].copy()
    raw["rul_pred_windows_raw"] = np.maximum(np.expm1(m.predict(X_va)), 0.0)
    raw["stage_pred_raw"] = 1
    rp = V5.postprocess(raw, {"stage_k": 1, "stage_max_step": None, "rul_alpha": 1.0, "rul_scale_early": 1.0, "rul_scale_late": 1.0})
    r = V5.score_rul_frame(va, rp)
    return m, rp, r, info


def prior_floor(va_T: pd.DataFrame, sigma: float, n_mc=40000, seed=7) -> dict:
    """Clock-only floor: a predictor that knows the TRUE onset time and the generating life distribution (median_life(rpm), lognormal sigma,
    clip 80..720 windows) but sees NO feature. For each (rpm, k windows since onset) the MAPE-optimal RUL is the 1/RUL-weighted median of
    (n_d - 1 - k) over the life distribution conditional on n_d - 1 > k. Scored on the val S7 windows (detected-onset window), bearing-averaged."""
    rng = np.random.default_rng(seed)
    d = va_T[va_T["in_deg_det"] & (va_T["rul_hours"] > 0) & va_T["run_to_failure"]].copy()
    ton = true_onset_t(va_T)
    d["k"] = np.round((d["t_hours"] - d["unit_id"].map(ton)) / K.CADENCE_H).astype(int)
    d["p"] = np.nan
    for rpm, g in d.groupby("unit_rpm"):
        lh = SC.median_life_hours(float(rpm)) * np.exp(rng.normal(0, sigma, n_mc))
        nd = np.clip(np.round(lh / K.CADENCE_H), *K.LIFE_CLIP_WINDOWS)
        cache = {}
        pred = []
        for k in g["k"].to_numpy():
            if k not in cache:
                r = nd - 1 - k
                r = r[r > 0]
                if len(r) < 50:
                    cache[k] = 1.0
                else:
                    o = np.argsort(r)
                    w = (1.0 / r[o]).cumsum()
                    cache[k] = float(r[o][np.searchsorted(w, w[-1] / 2)])
            pred.append(cache[k])
        d.loc[g.index, "p"] = np.asarray(pred) * K.CADENCE_H
    pers = {uid: float(np.mean(np.abs(g["p"] - g["rul_hours"]) / g["rul_hours"]) * 100) for uid, g in d.groupby("unit_id")}
    return {"S7_like_mape_pct": float(np.mean(list(pers.values()))), "n_bearings": len(pers)}


def decompose(va, rp):
    r = va.T.merge(rp[["unit_id", "window_index", "rul_pred_h"]], on=["unit_id", "window_index"])
    r = r[r["run_to_failure"]]
    s = r[r["in_deg_det"] & (r["rul_hours"] > 0)].copy()
    s["ape"] = np.abs(s["rul_pred_h"] - s["rul_hours"]) / s["rul_hours"] * 100
    s["bias"] = (s["rul_pred_h"] - s["rul_hours"]) / s["rul_hours"] * 100
    out = {}
    per = s.groupby("unit_id").agg(mape=("ape", "mean"), life_h=("life_hours", "first"), rpm=("unit_rpm", "first"), cls=("fault_class", "first"),
                                   late=("late_scale", "first"), bias=("bias", "mean"), n=("ape", "size"))
    ton = true_onset_t(va.T)
    per["deg_h"] = per["life_h"] - per.index.map(ton)
    per["lf"] = per["deg_h"] / per["rpm"].map(lambda x: SC.median_life_hours(float(x)))
    per["abs_ln_lf"] = np.abs(np.log(per["lf"].clip(lower=1e-3)))
    out["per_bearing_mape_summary"] = {"mean": float(per["mape"].mean()), "median": float(per["mape"].median()), "p90": float(per["mape"].quantile(0.9))}
    out["corr_unit_mape_vs_abs_ln_lifefactor"] = float(per[["mape", "abs_ln_lf"]].corr().iloc[0, 1])
    out["corr_unit_mape_vs_abs_ln_late_scale"] = float(np.corrcoef(per["mape"], np.abs(np.log(per["late"])))[0, 1])
    out["corr_unit_bias_vs_ln_lifefactor"] = float(np.corrcoef(per["bias"], np.log(per["lf"].clip(lower=1e-3)))[0, 1])
    per["lf_bin"] = pd.cut(per["lf"], [0, 0.6, 0.8, 1.25, 1.6, 100])
    out["by_life_factor_bin"] = {str(k): {"units": int(len(g)), "mape": float(g["mape"].mean()), "bias": float(g["bias"].mean())}
                                 for k, g in per.groupby("lf_bin", observed=True)}
    out["by_rpm"] = {str(k): float(g["mape"].mean()) for k, g in per.groupby("rpm")}
    out["by_class"] = {str(k): float(g["mape"].mean()) for k, g in per.groupby("cls")}
    ub = pd.cut(s["u_true"], [-0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    out["window_mape_by_true_u"] = {str(k): {"mape": float(g["ape"].mean()), "bias": float(g["bias"].mean()), "n": int(len(g))} for k, g in s.groupby(ub, observed=True)}
    out["window_mape_by_true_stage"] = {str(k): {"mape": float(g["ape"].mean()), "bias": float(g["bias"].mean()), "n": int(len(g))} for k, g in s.groupby("health_stage")}
    out["share_of_S7_windows_u_lt_0.2"] = float((s["u_true"] < 0.2).mean())
    ton_d = va.T.assign(_t=np.where(va.T["onset_index"] >= 0, va.T["onset_t_hours"], np.nan)).groupby("unit_id")["_t"].first()
    lag = (ton_d - ton).dropna()
    lagf = lag / per["deg_h"].reindex(lag.index)
    out["onset_detection_lag_frac_of_degradation"] = {"median": float(lagf.median()), "p90": float(lagf.quantile(0.9)), "max": float(lagf.max()),
                                                      "undetected_units": int(len(per) - len(lag))}
    out["n_units"] = int(len(per))
    out["deg_windows_per_unit_median"] = float(s.groupby("unit_id").size().median())
    return out


def job(tag, profile, nu, threads, full):
    t0 = time.perf_counter()
    sets = prep(profile, nu)
    (Xtr, Ttr), (Xva, Tva) = sets["train"], sets["val"]
    tr, va = sub(Xtr, Ttr), sub(Xva, Tva)
    sigma = (CP.get(profile) or {}).get("life_sigma", K.LIFE_SIGMA)
    res = {"tag": tag, "profile": profile or "hard", "noise_factor": nu, "life_sigma": sigma, "provenance": TAG}
    m, rp, r, info = fit_eval(tr, va, Xtr, Xva, threads)
    res["recipe_v5_levels_d6"] = {"val_S7_pct": r["S7_mape_pct"], "val_S7_true_onset_pct": r["alt_mape_true_onset_pct"], "best_iteration": info["best_iteration"],
                                  "per_bearing": r["per_bearing_summary"], "pol_pts": r["alt_percent_of_life_error_pts"]}
    print(tag, "recipe", json.dumps(res["recipe_v5_levels_d6"]), flush=True)
    res["clock_only_prior_floor"] = prior_floor(Tva, sigma)
    print(tag, "prior floor", res["clock_only_prior_floor"], flush=True)
    if full:
        res["decomposition"] = decompose(va, rp)
        res["error_vs_life_fraction"] = r["error_vs_life_fraction"]
        for kind in ("true", "det"):
            m2, rp2, r2, i2 = fit_eval(tr, va, add_clock(Xtr, Ttr, kind), add_clock(Xva, Tva, kind), threads)
            res[f"recipe_plus_clock_{kind}_onset"] = {"val_S7_pct": r2["S7_mape_pct"], "val_S7_true_onset_pct": r2["alt_mape_true_onset_pct"],
                                                      "note": "ORACLE (true onset time)" if kind == "true" else "causal: hours since detected onset appended as one feature (needs host state)"}
            print(tag, kind, json.dumps(res[f"recipe_plus_clock_{kind}_onset"]), flush=True)
    res["elapsed_s"] = round(time.perf_counter() - t0)
    (V7.CACHE / f"res_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("tag")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    return job(a.tag, a.profile, a.noise, a.threads, a.full)


if __name__ == "__main__":
    raise SystemExit(main())
