"""Window aggregator for the UP2 (v3) gradient-boosted heads: ONE implementation, as an ONNX graph.

The C# host already builds, per bearing-window, the tensor `encoder_cont` float32[1, T, R] of the
last T engineered rows (contract `columns` order, contract scalers; v3 uses identity scalers) and,
for the classifier slot, `decoder_cont` float32[1, 1, R] = the current row. The v3 heads compute
their tabular features INSIDE the ONNX graph from those tensors, so the host needs no new feature
code. Training features are produced by running this same graph (onnxruntime, batched), so the
features the trees were fit on and the deployed features are computed by identical float32 ops.

Aggregates over the T-row block X (oldest first, current row last), per column:
  linear (one MatMul with a constant [K, T] matrix):
    last, mean of the last 6 rows, mean of all rows, mean of the first quarter,
    least-squares slope over all rows, slope over the last half
  nonlinear: std over all rows (population), max over all rows
  level-occupancy (column 0 = the health index `hi` only, when `levels` is on): fraction of the T rows with
    hi > level, for HI_LEVELS. Over a long block this is an elapsed-time estimate (how long ago the HI
    crossed each level) and the spacing between levels gives the degradation rate.
Feature vector = [K+2 aggregates] x [R columns], aggregate-major, then the level fractions (see `feature_names`).
"""

from __future__ import annotations

import numpy as np

LINEAR_AGGS = ["last", "mean6", "mean", "mean_q1", "slope", "slope_h2"]
NONLIN_AGGS = ["std", "max"]
AGGS = LINEAR_AGGS + NONLIN_AGGS
HI_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
OPSET = 17


def n_features(R: int, levels: bool) -> int:
    return len(AGGS) * R + (len(HI_LEVELS) if levels else 0)


def weight_matrix(T: int) -> np.ndarray:
    W = np.zeros((len(LINEAR_AGGS), T), np.float64)
    W[0, -1] = 1.0
    k = min(6, T)
    W[1, -k:] = 1.0 / k
    W[2, :] = 1.0 / T
    q = max(1, T // 4)
    W[3, :q] = 1.0 / q
    x = np.arange(T, dtype=float)
    xc = x - x.mean()
    W[4] = xc / max((xc ** 2).sum(), 1e-12)
    h = max(2, T // 2)
    xh = np.arange(h, dtype=float)
    xhc = xh - xh.mean()
    W[5, -h:] = xhc / max((xhc ** 2).sum(), 1e-12)
    return W.astype(np.float32)


def feature_names(columns: list[str], levels: bool = False) -> list[str]:
    return [f"{a}:{c}" for a in AGGS for c in columns] + ([f"frac_above_{v}:{columns[0]}" for v in HI_LEVELS] if levels else [])


def aggregator_graph(T: int, R: int, with_decoder: bool, out_name: str = "feat", levels: bool = False):
    """ONNX graph: encoder_cont [B, Te, R] (+ decoder_cont [B, 1, R]) -> feat [B, (K+2)*R].
    With a decoder, the block is concat(encoder, decoder) along time, so T = Te + 1."""
    from onnx import TensorProto, helper, numpy_helper
    Te = T - 1 if with_decoder else T
    inputs = [helper.make_tensor_value_info("encoder_cont", TensorProto.FLOAT, ["batch", Te, R])]
    nodes = []
    src = "encoder_cont"
    if with_decoder:
        inputs.append(helper.make_tensor_value_info("decoder_cont", TensorProto.FLOAT, ["batch", 1, R]))
        nodes.append(helper.make_node("Concat", ["encoder_cont", "decoder_cont"], ["agg_x"], axis=1))
        src = "agg_x"
    W = weight_matrix(T)
    inits = [numpy_helper.from_array(W, "agg_W"),
             numpy_helper.from_array(np.array([0, -1], np.int64), "agg_shape")]
    nodes += [
        helper.make_node("MatMul", ["agg_W", src], ["agg_lin"]),                       # [B, K, R]
        helper.make_node("ReduceMean", [src], ["agg_mu"], axes=[1], keepdims=1),       # [B, 1, R]
        helper.make_node("Sub", [src, "agg_mu"], ["agg_c"]),
        helper.make_node("Mul", ["agg_c", "agg_c"], ["agg_c2"]),
        helper.make_node("ReduceMean", ["agg_c2"], ["agg_var"], axes=[1], keepdims=1),
        helper.make_node("Sqrt", ["agg_var"], ["agg_std"]),
        helper.make_node("ReduceMax", [src], ["agg_max"], axes=[1], keepdims=1),
        helper.make_node("Concat", ["agg_lin", "agg_std", "agg_max"], ["agg_all"], axis=1),  # [B, K+2, R]
        helper.make_node("Reshape", ["agg_all", "agg_shape"], ["agg_flat" if levels else out_name]),  # [B, (K+2)*R]
    ]
    if levels:
        inits += [numpy_helper.from_array(np.array([0], np.int64), "agg_c0s"),
                  numpy_helper.from_array(np.array([1], np.int64), "agg_c0e"),
                  numpy_helper.from_array(np.array([2], np.int64), "agg_ax2"),
                  numpy_helper.from_array(np.asarray(HI_LEVELS, np.float32).reshape(1, 1, -1), "agg_lv")]
        nodes += [
            helper.make_node("Slice", [src, "agg_c0s", "agg_c0e", "agg_ax2"], ["agg_hi"]),         # [B, T, 1]
            helper.make_node("Greater", ["agg_hi", "agg_lv"], ["agg_gt"]),                         # [B, T, nL]
            helper.make_node("Cast", ["agg_gt"], ["agg_gtf"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMean", ["agg_gtf"], ["agg_occ"], axes=[1], keepdims=0),        # [B, nL]
            helper.make_node("Concat", ["agg_flat", "agg_occ"], [out_name], axis=1),
        ]
    outputs = [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["batch", n_features(R, levels)])]
    g = helper.make_graph(nodes, "v3_window_aggregator", inputs, outputs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    return m


def blocks(rows: np.ndarray, T: int, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(n, R) rows of one bearing -> blocks [n, T, R] ending at every row (left-padded with row 0,
    as the C# N-HiTS slot does) and a mask of rows that have a full T-row history (TFT slot rule)."""
    rows = np.asarray(rows, np.float32)
    n = len(rows)
    pad = np.concatenate([np.repeat(rows[:1], T - 1, axis=0), rows], axis=0)
    idx = np.arange(n)[:, None] + np.arange(T)[None, :]
    full = np.arange(n) >= (T - 1 + start)
    return pad[idx], full


_SESSIONS: dict = {}


def compute(blocks_: np.ndarray, with_decoder: bool, batch: int = 65536, levels: bool = False) -> np.ndarray:
    """Run the aggregator graph (onnxruntime) on [B, T, R] blocks -> [B, F] float32 features."""
    import onnxruntime as ort
    B, T, R = blocks_.shape
    key = (T, R, with_decoder, levels)
    if key not in _SESSIONS:
        m = aggregator_graph(T, R, with_decoder, levels=levels)
        _SESSIONS[key] = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    s = _SESSIONS[key]
    out = []
    for i in range(0, B, batch):
        b = blocks_[i:i + batch].astype(np.float32)
        feed = {"encoder_cont": b[:, :-1, :], "decoder_cont": b[:, -1:, :]} if with_decoder else {"encoder_cont": b}
        out.append(s.run(["feat"], feed)[0])
    return np.concatenate(out) if out else np.zeros((0, n_features(R, levels)), np.float32)
