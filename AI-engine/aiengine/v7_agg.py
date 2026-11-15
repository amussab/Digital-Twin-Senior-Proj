"""v7 window aggregator (copy of v6_agg at 2026-11-15, so v6 may keep evolving) (ONE implementation, as an ONNX graph; numpy reference for the parity test).

Extends aiengine.up2_window (v3/v5): same input contract (`encoder_cont` float32[B, T, R] = the last T engineered rows, left-padded with the
oldest row as the host does), richer and LONGER-context trajectory features for the RUL / z heads:

  linear aggregates (one MatMul): last, mean of the last 3 / 6 / 24 / 96 rows, mean of all rows, mean of the first quarter,
      least-squares slope over all rows, over the last 24 and over the last 96 rows        (windows clamp to T)
  nonlinear: std over all rows, max over all rows
  level occupancy (fraction of the T rows above a level) for the health index `hi` and for `log_env_ratio`:
      over a long block this is the elapsed time since the signal crossed each level, spacing of levels = growth rate.
Everything is arithmetic on the engineered rows: C#-portable (no fitted object). The classifier slot keeps aiengine.up2_window unchanged.
"""
from __future__ import annotations

import numpy as np

LIN = [("last", None), ("mean3", 3), ("mean6", 6), ("mean24", 24), ("mean96", 96), ("mean", 0), ("mean_q1", -1),
       ("slope", 0), ("slope24", 24), ("slope96", 96)]
LIN_NAMES = [n for n, _ in LIN]
NONLIN = ["std", "max"]
AGGS = LIN_NAMES + NONLIN
HI_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
ENV_LEVELS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
LEVEL_SPECS = [(0, HI_LEVELS), (1, ENV_LEVELS)]          # (column index in COLS, levels)
OPSET = 15


def n_features(R: int, levels: bool = True) -> int:
    return len(AGGS) * R + (sum(len(l) for _, l in LEVEL_SPECS) if levels else 0)


def feature_names(columns: list[str], levels: bool = True) -> list[str]:
    n = [f"{a}:{c}" for a in AGGS for c in columns]
    if levels:
        n += [f"frac_above_{v}:{columns[ci]}" for ci, lv in LEVEL_SPECS for v in lv]
    return n


def weight_matrix(T: int) -> np.ndarray:
    W = np.zeros((len(LIN), T), np.float64)
    for r, (name, k) in enumerate(LIN):
        if name == "last":
            W[r, -1] = 1.0
        elif name.startswith("mean") and name not in ("mean", "mean_q1"):
            kk = min(k, T)
            W[r, -kk:] = 1.0 / kk
        elif name == "mean":
            W[r, :] = 1.0 / T
        elif name == "mean_q1":
            q = max(1, T // 4)
            W[r, :q] = 1.0 / q
        elif name.startswith("slope"):
            kk = T if k == 0 else min(k, T)
            x = np.arange(kk, dtype=float)
            xc = x - x.mean()
            W[r, -kk:] = xc / max((xc ** 2).sum(), 1e-12)
    return W.astype(np.float32)


def reference(blocks: np.ndarray, levels: bool = True) -> np.ndarray:
    """numpy reference of the graph (float64 math): blocks [B, T, R] -> [B, F]."""
    b = np.asarray(blocks, np.float64)
    W = weight_matrix(b.shape[1]).astype(np.float64)
    lin = np.einsum("kt,btr->bkr", W, b)
    std = b.std(axis=1, keepdims=True)
    mx = b.max(axis=1, keepdims=True)
    out = np.concatenate([lin, std, mx], axis=1).reshape(len(b), -1)
    if levels:
        occ = [(b[:, :, ci][:, :, None] > np.asarray(lv)[None, None, :]).mean(axis=1) for ci, lv in LEVEL_SPECS]
        out = np.concatenate([out] + occ, axis=1)
    return out


def aggregator_graph(T: int, R: int, out_name: str = "feat", levels: bool = True):
    from onnx import TensorProto, helper, numpy_helper
    inputs = [helper.make_tensor_value_info("encoder_cont", TensorProto.FLOAT, ["batch", T, R])]
    src = "encoder_cont"
    inits = [numpy_helper.from_array(weight_matrix(T), "agg_W"), numpy_helper.from_array(np.array([0, -1], np.int64), "agg_shape")]
    nodes = [
        helper.make_node("MatMul", ["agg_W", src], ["agg_lin"]),
        helper.make_node("ReduceMean", [src], ["agg_mu"], axes=[1], keepdims=1),
        helper.make_node("Sub", [src, "agg_mu"], ["agg_c"]),
        helper.make_node("Mul", ["agg_c", "agg_c"], ["agg_c2"]),
        helper.make_node("ReduceMean", ["agg_c2"], ["agg_var"], axes=[1], keepdims=1),
        helper.make_node("Sqrt", ["agg_var"], ["agg_std"]),
        helper.make_node("ReduceMax", [src], ["agg_max"], axes=[1], keepdims=1),
        helper.make_node("Concat", ["agg_lin", "agg_std", "agg_max"], ["agg_all"], axis=1),
        helper.make_node("Reshape", ["agg_all", "agg_shape"], ["agg_flat" if levels else out_name]),
    ]
    if levels:
        inits.append(numpy_helper.from_array(np.array([2], np.int64), "agg_ax2"))
        occs = []
        for j, (ci, lv) in enumerate(LEVEL_SPECS):
            inits += [numpy_helper.from_array(np.array([ci], np.int64), f"agg_s{j}"), numpy_helper.from_array(np.array([ci + 1], np.int64), f"agg_e{j}"),
                      numpy_helper.from_array(np.asarray(lv, np.float32).reshape(1, 1, -1), f"agg_lv{j}")]
            nodes += [helper.make_node("Slice", [src, f"agg_s{j}", f"agg_e{j}", "agg_ax2"], [f"agg_col{j}"]),
                      helper.make_node("Greater", [f"agg_col{j}", f"agg_lv{j}"], [f"agg_gt{j}"]),
                      helper.make_node("Cast", [f"agg_gt{j}"], [f"agg_gtf{j}"], to=TensorProto.FLOAT),
                      helper.make_node("ReduceMean", [f"agg_gtf{j}"], [f"agg_occ{j}"], axes=[1], keepdims=0)]
            occs.append(f"agg_occ{j}")
        nodes.append(helper.make_node("Concat", ["agg_flat"] + occs, [out_name], axis=1))
    outputs = [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["batch", n_features(R, levels)])]
    g = helper.make_graph(nodes, "v6_window_aggregator", inputs, outputs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    return m


_SESSIONS: dict = {}


def compute(blocks_: np.ndarray, batch: int = 32768, levels: bool = True) -> np.ndarray:
    import onnxruntime as ort
    B, T, R = blocks_.shape
    key = (T, R, levels)
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        _SESSIONS[key] = ort.InferenceSession(aggregator_graph(T, R, levels=levels).SerializeToString(), so, providers=["CPUExecutionProvider"])
    s = _SESSIONS[key]
    out = [s.run(["feat"], {"encoder_cont": blocks_[i:i + batch].astype(np.float32)})[0] for i in range(0, B, batch)]
    return np.concatenate(out) if out else np.zeros((0, n_features(R, levels)), np.float32)


def unit_features(tab, T: int, columns: list[str], warm: int, levels: bool = True):
    """Features of every window with >= warm rows of history, per unit (bounded memory). -> X, index frame"""
    import pandas as pd
    from . import up2_window as uw
    Xs, idx = [], []
    for uid, g in tab.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        b, _ = uw.blocks(g[columns].to_numpy(np.float64), T)
        keep = np.arange(len(g)) >= warm - 1
        Xs.append(compute(b[keep], levels=levels))
        idx.append(g.loc[keep, ["unit_id", "window_index"]])
    return np.concatenate(Xs), pd.concat(idx, ignore_index=True)
