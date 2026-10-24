"""Close the synth_v3 -> COE-pipeline domain gap of the classifier [SIMULATION]/[SYNTHETIC].

Candidates (all selected on COE-val ONLY; COE-test is scored once by `coe_adapt_eval test`):
  base      deployed v3b classifier as-is (models/v3b_cls.onnx)
  affine    per-column affine calibration of the 21 engineered inputs (COE scale -> synth_v3 scale), fitted on the
            HEALTHY windows of COE-train vs synth_v3-train (median / IQR matching). It is exactly the contract's existing
            `scalers` (x - center) / scale, so C# needs NO change. v3b trees unchanged.
  quantile  per-column monotone piecewise-linear quantile map (21 knots, COE-train pooled -> synth_v3-train pooled),
            embedded IN the classifier ONNX graph ahead of the aggregator, so C# needs NO change. v3b trees unchanged.
  mix_w*    XGBoost classifier (same aggregator, T=25, same hyper-parameters as v3) retrained on synth_v3 train (+train2)
            + COE-train, COE weight share w; early stopping on COE-val.
  +gate     optional cross-bearing transmission gate (post-process over the two bearings of ONE payload window):
            a bearing predicted faulty is re-labelled healthy when its excess defect-family ratio is below
            tau x its partner's for the same family (the defect is on the partner; this bearing only sees the
            transmitted vibration). Needs both bearings -> a C# change (specified in the contract).
Pre-declared selection rule (COE-val): max macro-F1 (5 classes, per bearing-window); eligible only if synth_v3-val
macro-F1 >= SYNTH_FLOOR (the replay demo stream must not regress); tie (|dF1| < 0.002) -> lower transmitted false-fault rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as feat
from . import up2
from . import up2_window as uw

T_CLS = 25
COLS = list(up2.COLUMNS)
SYNTH_FLOOR = 0.93
FAMILY_OF = {"outer_race": "bpfo", "inner_race": "bpfi", "ball": "bsf", "cage": "ftf"}
QUANTILES = np.linspace(0.0, 1.0, 21)


# --------------------------------------------------------------------------- data

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered inputs exactly as the engine computes them (baseline = median of the first 24 windows = commissioning)."""
    out = feat.engineer_table(df)
    out["observable_class"] = out["observable_class"].astype(str)
    return out


def blocks_for(df: pd.DataFrame, T: int = T_CLS, columns=COLS, stride: int = 1):
    """[n, T, R] blocks (left-padded) for every row with a full T-row history (every `stride`-th), plus the index."""
    B, idx = [], []
    for uid, g in df.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        b, full = uw.blocks(g[columns].to_numpy(np.float64), T)
        sel = np.flatnonzero(full)[::stride]
        B.append(b[sel].astype(np.float32))
        idx.append(g.iloc[sel][["unit_id", "window_index"]])
    return np.concatenate(B).astype(np.float32), pd.concat(idx, ignore_index=True)


# --------------------------------------------------------------------------- calibration maps

def fit_affine(coe_rows: np.ndarray, syn_rows: np.ndarray) -> dict:
    """Per column: y = (x - center) / scale with scale = IQR_coe / IQR_syn, center = med_coe - med_syn * scale.
    Fitted on HEALTHY windows only (the commissioning-like state both domains share). Degenerate IQR -> identity."""
    sc = {}
    for j, c in enumerate(COLS):
        xc, xs = coe_rows[:, j], syn_rows[:, j]
        qc, qs = np.percentile(xc, [25, 50, 75]), np.percentile(xs, [25, 50, 75])
        ic, is_ = qc[2] - qc[0], qs[2] - qs[0]
        if c == "rpm_norm" or ic < 1e-9 or is_ < 1e-9:
            sc[c] = {"center": 0.0, "scale": 1.0, "kind": "identity"}
            continue
        s = ic / is_
        sc[c] = {"center": float(qc[1] - qs[1] * s), "scale": float(s), "kind": "affine_coe_to_synth"}
    return sc


def apply_affine(blocks: np.ndarray, sc: dict) -> np.ndarray:
    c = np.array([sc[k]["center"] for k in COLS], np.float64)
    s = np.array([sc[k]["scale"] for k in COLS], np.float64)
    return ((blocks.astype(np.float64) - c) / s).astype(np.float32)


def fit_quantile(coe_rows: np.ndarray, syn_rows: np.ndarray) -> dict:
    """Per column monotone piecewise-linear map through (q_coe(p), q_syn(p)), p = 0, 0.05, ..., 1; constant outside.
    Ties in the COE knots are merged (strictly increasing x-knots). rpm_norm is left identity."""
    out = {}
    for j, c in enumerate(COLS):
        if c == "rpm_norm":
            out[c] = {"x": [-1e6, 1e6], "y": [-1e6, 1e6]}
            continue
        xk, yk = np.quantile(coe_rows[:, j], QUANTILES), np.quantile(syn_rows[:, j], QUANTILES)
        keep = np.concatenate([[True], np.diff(xk) > 1e-9 * max(1.0, np.abs(xk).max())])
        xk, yk = xk[keep], np.maximum.accumulate(yk[keep])
        if len(xk) < 2:
            xk, yk = np.array([xk[0], xk[0] + 1.0]), np.array([yk[0], yk[0]])
        out[c] = {"x": xk.tolist(), "y": yk.tolist()}
    return out


def apply_quantile(blocks: np.ndarray, qm: dict) -> np.ndarray:
    """Reference implementation (numpy interp = the ONNX relu-sum form, see quantile_graph_nodes)."""
    y = np.empty_like(blocks, dtype=np.float64)
    for j, c in enumerate(COLS):
        y[..., j] = np.interp(blocks[..., j].astype(np.float64), qm[c]["x"], qm[c]["y"])
    return y.astype(np.float32)


def quantile_tensors(qm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Padded relu-sum coefficients: y = y0 + sum_i d_i * relu(xc - k_i), xc = clip(x, lo, hi), per column.
    Returns lo [R], hi [R], y0 [R], knots [K, R], deltas [K, R] (padding: knot = lo, delta = 0)."""
    R = len(COLS)
    K = max(len(qm[c]["x"]) - 1 for c in COLS)
    lo, hi, y0 = np.zeros(R), np.zeros(R), np.zeros(R)
    kn, de = np.zeros((K, R)), np.zeros((K, R))
    for j, c in enumerate(COLS):
        x, y = np.asarray(qm[c]["x"]), np.asarray(qm[c]["y"])
        s = np.diff(y) / np.diff(x)
        lo[j], hi[j], y0[j] = x[0], x[-1], y[0]
        kn[:, j] = x[0]
        kn[:len(s), j] = x[:-1]
        de[:len(s), j] = np.concatenate([[s[0]], np.diff(s)])
    return lo, hi, y0, kn, de


def apply_quantile_relu(blocks: np.ndarray, qm: dict) -> np.ndarray:
    lo, hi, y0, kn, de = quantile_tensors(qm)
    x = np.clip(blocks.astype(np.float64), lo, hi)
    return (y0 + (np.maximum(x[..., None, :] - kn, 0.0) * de).sum(-2)).astype(np.float32)


# --------------------------------------------------------------------------- classifier runs

def onnx_probs(onnx_path, blocks: np.ndarray, batch: int = 8192) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(blocks), batch):
        b = blocks[i:i + batch]
        lg = s.run(["logits"], {"encoder_cont": b[:, :-1], "decoder_cont": b[:, -1:]})[0]
        out.append(np.exp(lg[:, 0, :].astype(np.float64)))
    P = np.concatenate(out)
    return P / P.sum(1, keepdims=True)


def tree_probs(model, blocks: np.ndarray) -> np.ndarray:
    X = uw.compute(blocks, with_decoder=True)
    return model.predict_proba(X)


def frame(idx: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    out = idx.copy()
    for i, c in enumerate(up2.CLASS_ORDER):
        out[f"p_{c}"] = P[:, i]
    out["pred_class"] = [up2.CLASS_ORDER[i] for i in P.argmax(1)]
    return out


# --------------------------------------------------------------------------- cross-bearing gate

def gate(pred: pd.DataFrame, eng: pd.DataFrame, tau: float, signal: str = "family") -> pd.DataFrame:
    """Cross-bearing transmission gate on per-bearing predictions of the SAME rig windows.

    For bearing k with pre-gate class c != healthy: x_k = exp(e_k) - 1, x_p = exp(e_p) - 1 with e = the engineered
    log ratio of the CURRENT window (signal 'family': log_<fam(c)>_ratio, signal 'env': log_env_ratio) of bearing k and
    of its partner p. If x_p > 0 and x_k < tau * x_p -> class healthy. Both bearings are decided from the PRE-gate
    classes (simultaneous). tau <= 0 disables the gate.
    """
    out = pred.copy()
    if tau <= 0:
        out["gated"] = False
        return out
    key = ["bearing_key", "window_index"]
    e = eng[["unit_id", "window_index", "bearing_key", "bearing"] + [c for c in COLS if c.startswith("log_")]]
    m = out.merge(e, on=["unit_id", "window_index"], how="left")
    part = m[key + ["bearing"] + [c for c in COLS if c.startswith("log_")]].copy()
    part["bearing"] = 3 - part["bearing"]
    m = m.merge(part, on=key + ["bearing"], how="left", suffixes=("", "_partner"))
    cls = m["pred_class"].to_numpy()
    gated = np.zeros(len(m), bool)
    for c, fam in FAMILY_OF.items():
        col = f"log_{fam}_ratio" if signal == "family" else "log_env_ratio"
        sel = cls == c
        if not sel.any():
            continue
        xk = np.expm1(m.loc[sel, col].to_numpy(float))
        xp = np.expm1(m.loc[sel, col + "_partner"].to_numpy(float))
        g = np.isfinite(xp) & (xp > 0) & (xk < tau * xp)
        gated[np.flatnonzero(sel)[g]] = True
    out["gated"] = gated
    out.loc[gated, "pred_class"] = "healthy"
    return out


# --------------------------------------------------------------------------- metrics

def coe_metrics(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Per-class F1 / confusion (aiengine.up2_metrics.classification) + per-bearing domain-gap rates.

    transmitted_false_fault_rate: healthy bearing whose PARTNER is faulted, predicted non-healthy (the COE-demo bug);
    clean_false_fault_rate: healthy bearing with a healthy partner, predicted non-healthy;
    missed_fault_rate: faulted bearing predicted healthy; wrong_class_rate: faulted, predicted another fault class.
    `b2_*` = bearing 2 only (the demo symptom: bearing 1 faulted, bearing 2 healthy)."""
    from . import up2_metrics as M
    cols = ["observable_class", "bearing", "partner_faulty", "transition", "scenario", "rpm_nominal", "severity"]
    p = pred.merge(truth[["unit_id", "window_index"] + cols], on=["unit_id", "window_index"], how="left")
    p = p[~p["transition"].astype(bool)]
    y, yh = p["observable_class"].to_numpy(), p["pred_class"].to_numpy()
    h = y == "healthy"
    pf = p["partner_faulty"].astype(bool).to_numpy()
    b2 = p["bearing"].to_numpy() == 2
    rate = lambda m: (float((yh[m] != "healthy").mean()) if m.any() else None)   # noqa: E731
    out = {"classification": M.classification(p)}
    out["transmitted_false_fault_rate"] = rate(h & pf)
    out["clean_false_fault_rate"] = rate(h & ~pf)
    out["b2_transmitted_false_fault_rate"] = rate(h & pf & b2)
    out["b1_transmitted_false_fault_rate"] = rate(h & pf & ~b2)
    f = ~h
    out["missed_fault_rate"] = float((yh[f] == "healthy").mean()) if f.any() else None
    out["wrong_class_rate"] = float(((yh[f] != "healthy") & (yh[f] != y[f])).mean()) if f.any() else None
    per_b = {}
    for b in (1, 2):
        q = p[p["bearing"] == b]
        roles = np.where(q["observable_class"] != "healthy", "faulted",
                         np.where(q["partner_faulty"].astype(bool), "healthy_partner_faulted", "healthy_partner_healthy"))
        tab = pd.crosstab(pd.Series(roles, name="role"), pd.Series(np.where(q["pred_class"] == "healthy", "pred_healthy",
                          np.where(q["pred_class"] == q["observable_class"], "pred_correct_fault", "pred_other_fault")),
                          name="pred"))
        per_b[str(b)] = {r: {c: int(v) for c, v in row.items()} for r, row in tab.to_dict(orient="index").items()}
    out["per_bearing_roles"] = per_b
    out["macro_f1_by_rpm"] = {str(k): float(M.classification(g)["macro_f1"]) for k, g in p.groupby("rpm_nominal")}
    out["recall_by_severity"] = {f"{k:g}": float((g["pred_class"] == g["observable_class"]).mean())
                                 for k, g in p[p["observable_class"] != "healthy"].groupby("severity")}
    out["n_windows"] = int(len(p))
    return out


def short(m: dict) -> dict:
    c = m["classification"]
    return {"macro_f1": round(c["macro_f1"], 4), "accuracy": round(c["accuracy"], 4),
            "per_class_f1": {k: round(v, 4) for k, v in c["per_class_f1"].items()},
            "b2_transmitted_false_fault_rate": m["b2_transmitted_false_fault_rate"],
            "transmitted_false_fault_rate": m["transmitted_false_fault_rate"],
            "clean_false_fault_rate": m["clean_false_fault_rate"], "missed_fault_rate": m["missed_fault_rate"],
            "wrong_class_rate": m["wrong_class_rate"]}
