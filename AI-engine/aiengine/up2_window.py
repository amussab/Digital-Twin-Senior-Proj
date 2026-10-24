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
Feature vector = [K+2 aggregates] x [R columns], aggregate-major (see `feature_names`).
"""

from __future__ import annotations

import numpy as np

LINEAR_AGGS = ["last", "mean6", "mean", "mean_q1", "slope", "slope_h2"]
NONLIN_AGGS = ["std", "max"]
AGGS = LINEAR_AGGS + NONLIN_AGGS
OPSET = 17


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


def feature_names(columns: list[str]) -> list[str]:
    return [f"{a}:{c}" for a in AGGS for c in columns]


def aggregator_graph(T: int, R: int, with_decoder: bool, out_name: str = "feat"):
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
        helper.make_node("Reshape", ["agg_all", "agg_shape"], [out_name]),             # [B, (K+2)*R]
    ]
    outputs = [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["batch", len(AGGS) * R])]
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


def compute(blocks_: np.ndarray, with_decoder: bool, batch: int = 65536) -> np.ndarray:
    """Run the aggregator graph (onnxruntime) on [B, T, R] blocks -> [B, F] float32 features."""
    import onnxruntime as ort
    B, T, R = blocks_.shape
    key = (T, R, with_decoder)
    if key not in _SESSIONS:
        m = aggregator_graph(T, R, with_decoder)
        _SESSIONS[key] = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    s = _SESSIONS[key]
    out = []
    for i in range(0, B, batch):
        b = blocks_[i:i + batch].astype(np.float32)
        feed = {"encoder_cont": b[:, :-1, :], "decoder_cont": b[:, -1:, :]} if with_decoder else {"encoder_cont": b}
        out.append(s.run(["feat"], feed)[0])
    return np.concatenate(out) if out else np.zeros((0, len(AGGS) * R), np.float32)
