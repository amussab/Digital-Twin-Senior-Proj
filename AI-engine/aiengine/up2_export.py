"""Export the UP2 v3 gradient-boosted heads to ONNX + models/model_contract_v3.json + golden vectors.

Files (flat in AI-engine/models/, v2 files untouched):
  v3_cls.onnx   TFT slot.   inputs encoder_cont f32[B, T-1, R], decoder_cont f32[B, 1, R]
                outputs logits f32[B, 1, 5] (= log class probabilities; softmax(logits) = probabilities),
                        variable_weights f32[1, R] (constant: normalised gain importance per input column)
  v3_rul.onnx   N-HiTS slot. input encoder_cont f32[B, NL, R] (left-padded with the oldest row, as the host does)
                output rul_head f32[B, 3] = [rul_windows, z, stage]
                  rul_windows = remaining life in windows (hours = rul_windows * cadence_hours, cadence = median
                                spacing of the last <=48 window timestamps, exactly what the host already computes)
                  z           = estimated elapsed fraction of the degradation interval (< 0 = before onset)
                  stage       = 1 if z < z_cut else min(6, 2 + floor(5 * clip(z, 0, 1)))   (z_cut <= 0, from validation)
Both graphs start with the same window aggregator (aiengine/up2_window.py) followed by the converted tree
ensemble(s) (onnxmltools), so the host only builds the tensors it already builds.
Scalers are identity (center 0, scale 1): trees are scale-invariant, so no fitted object is in the per-window path.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import features as feat
from . import labels, up2
from . import up2_window as uw
from .config import BEARING_FEATURES, MODELS_DIR, SEED
from .payload import DISPLACEMENT_NAMES, PAYLOAD_SIZE
from .synth_v3 import generator as gen

OPSET = 17
ML_OPSET = 3
CLS_FILE, RUL_FILE, CONTRACT_FILE, GOLDEN_FILE = "v3_cls.onnx", "v3_rul.onnx", "model_contract_v3.json", "golden_vectors_v3.json"


def _convert(model, n_features: int):
    from onnxmltools.convert.common.data_types import FloatTensorType
    init = [("input", FloatTensorType([None, n_features]))]
    name = type(model).__name__
    if name.startswith("XGB"):
        import onnxmltools
        m = onnxmltools.convert_xgboost(model, initial_types=init, target_opset=OPSET)
    else:
        import onnxmltools
        kw = {"zipmap": False} if "Classifier" in name else {}
        m = onnxmltools.convert_lightgbm(model, initial_types=init, target_opset=OPSET, **kw)
    return m


def _inline(model: onnx.ModelProto, prefix: str, input_from: str):
    """Nodes/initializers of a converted model with every name prefixed; its graph input is wired to `input_from`.
    Returns (nodes, initializers, {original output name: prefixed name}, opset imports)."""
    g = onnx.compose.add_prefix(model, prefix).graph
    nodes = [helper.make_node("Identity", [input_from], [g.input[0].name])] + list(g.node)
    outs = {o.name[len(prefix):]: o.name for o in g.output}
    return nodes, list(g.initializer), outs, list(model.opset_import)


def _model(nodes, inits, inputs, outputs, opsets, name):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits)
    ops = {}
    for o in opsets:
        ops[o.domain] = max(ops.get(o.domain, 0), o.version)
    ops[""] = max(ops.get("", 0), OPSET)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid(d, v) for d, v in ops.items()],
                          producer_name="aiengine.up2_export")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def build_cls(model, T: int, R: int, importances: np.ndarray) -> onnx.ModelProto:
    agg = uw.aggregator_graph(T, R, with_decoder=True, out_name="feat")
    tree = _convert(model, len(uw.AGGS) * R)
    tn, ti, touts, tops = _inline(tree, "c_", "feat")
    prob = touts["probabilities"]
    vw = np.asarray(importances, np.float32).reshape(1, R)
    post = [
        helper.make_node("Max", [prob, "c_eps"], ["c_pclip"]),
        helper.make_node("Log", ["c_pclip"], ["c_logp"]),
        helper.make_node("Unsqueeze", ["c_logp", "c_ax1"], ["logits"]),
        helper.make_node("Identity", ["c_vw"], ["variable_weights"]),
    ]
    inits = list(agg.graph.initializer) + ti + [
        numpy_helper.from_array(np.array(1e-12, np.float32), "c_eps"),
        numpy_helper.from_array(np.array([1], np.int64), "c_ax1"),
        numpy_helper.from_array(vw, "c_vw")]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1, len(up2.CLASS_ORDER)]),
               helper.make_tensor_value_info("variable_weights", TensorProto.FLOAT, [1, R])]
    return _model(list(agg.graph.node) + tn + post, inits, list(agg.graph.input), outputs,
                  list(agg.opset_import) + tops, "v3_cls")


def build_rul(rul_model, z_model, NL: int, R: int, levels: bool = False, z_cut: float = 0.0) -> onnx.ModelProto:
    agg = uw.aggregator_graph(NL, R, with_decoder=False, out_name="feat", levels=levels)
    F = uw.n_features(R, levels)
    rn, ri, routs, rops = _inline(_convert(rul_model, F), "r_", "feat")
    zn, zi, zouts, zops = _inline(_convert(z_model, F), "z_", "feat")
    rv, zv = list(routs.values())[0], list(zouts.values())[0]
    post = [
        helper.make_node("Exp", [rv], ["p_e"]),
        helper.make_node("Sub", ["p_e", "p_one"], ["p_em1"]),
        helper.make_node("Max", ["p_em1", "p_zero"], ["p_rul"]),                  # [B, 1]
        helper.make_node("Clip", [zv, "p_zero", "p_one"], ["p_zc"]),
        helper.make_node("Mul", ["p_zc", "p_five"], ["p_z5"]),
        helper.make_node("Floor", ["p_z5"], ["p_fl"]),
        helper.make_node("Add", ["p_fl", "p_two"], ["p_st0"]),
        helper.make_node("Min", ["p_st0", "p_six"], ["p_st1"]),
        helper.make_node("Less", [zv, "p_cut"], ["p_pre"]),
        helper.make_node("Where", ["p_pre", "p_one", "p_st1"], ["p_stage"]),
        helper.make_node("Concat", ["p_rul", zv, "p_stage"], ["rul_head"], axis=1),
    ]
    c = lambda v, n: numpy_helper.from_array(np.array(v, np.float32), n)   # noqa: E731
    inits = list(agg.graph.initializer) + ri + zi + [c(1.0, "p_one"), c(0.0, "p_zero"), c(5.0, "p_five"),
                                                    c(2.0, "p_two"), c(6.0, "p_six"), c(float(z_cut), "p_cut")]
    outputs = [helper.make_tensor_value_info("rul_head", TensorProto.FLOAT, ["batch", 3])]
    return _model(list(agg.graph.node) + rn + zn + post, inits, list(agg.graph.input), outputs,
                  list(agg.opset_import) + rops + zops, "v3_rul")


def importances_per_column(model, R: int) -> np.ndarray:
    imp = np.asarray(getattr(model, "feature_importances_", np.ones(len(uw.AGGS) * R)), float)
    per = imp.reshape(len(uw.AGGS), R).sum(axis=0)
    return per / max(per.sum(), 1e-12)


def parity(cls_onnx: Path, rul_onnx: Path, heads, df) -> dict:
    """ONNX Runtime (deployed graphs) vs Python (aggregator via ORT + native xgboost/lightgbm predict)."""
    import onnxruntime as ort
    from .up2_train import cls_features, cls_predict, rul_features, rul_predict
    c, r, z = heads
    T, NL = c["cfg"]["T"], r["cfg"]["NL"]
    units = df["unit_id"].drop_duplicates().iloc[:6]
    d = df[df["unit_id"].isin(units)]
    Xc, ic, _ = cls_features(d, T)
    ref_c = cls_predict(c["model"], Xc, ic)
    Xr, ir = rul_features(d, NL, warm=T, levels=r["cfg"].get("levels", False))
    ref_r = rul_predict(r["model"], z["model"], Xr, ir, z["cfg"].get("z_cut", 0.0))
    sc = ort.InferenceSession(str(cls_onnx), providers=["CPUExecutionProvider"])
    sr = ort.InferenceSession(str(rul_onnx), providers=["CPUExecutionProvider"])
    pc, prul, pz, pst = [], [], [], []
    for uid, g in d.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        rows = g[up2.COLUMNS].to_numpy(np.float64)
        bc, _ = uw.blocks(rows, T)
        br, _ = uw.blocks(rows, NL)
        for k in range(T - 1, len(g)):
            lg = sc.run(["logits"], {"encoder_cont": bc[k:k + 1, :-1], "decoder_cont": bc[k:k + 1, -1:]})[0]
            pc.append(np.exp(lg[0, 0]))
            o = sr.run(["rul_head"], {"encoder_cont": br[k:k + 1]})[0][0]
            prul.append(o[0]); pz.append(o[1]); pst.append(o[2])
    pc = np.asarray(pc)
    P = ref_c[[f"p_{k}" for k in up2.CLASS_ORDER]].to_numpy()
    rel = np.abs(np.asarray(prul) - ref_r["rul_pred_windows"].to_numpy()) / np.maximum(ref_r["rul_pred_windows"].to_numpy(), 1.0)
    return {"windows": int(len(pc)), "cls_prob_max_abs_diff": float(np.abs(pc - P).max()),
            "cls_argmax_mismatch": int((pc.argmax(1) != P.argmax(1)).sum()),
            "rul_windows_max_rel_diff": float(rel.max()),
            "z_max_abs_diff": float(np.abs(np.asarray(pz) - ref_r["z_pred"].to_numpy()).max()),
            "stage_mismatch": int((np.asarray(pst).astype(int) != ref_r["stage_pred"].to_numpy()).sum()),
            "reference": "aggregator via ORT + native predict_proba / predict (float32 features)"}


def contract(T: int, NL: int, heads, res: dict, prov: dict) -> dict:
    c, r, z = heads
    ident = {col: {"center": 0.0, "scale": 1.0, "kind": "identity"} for col in up2.COLUMNS}
    return {
        "contract_version": 3,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "AI-engine/aiengine/up2_export.py",
        "provenance": prov,
        "model_unit": "one biaxial bearing: 16 raw features = X's 8 then Y's 8 (COE C7 order)",
        "wire_payload": {"size_bytes": PAYLOAD_SIZE, "byte_order": "little-endian",
                         "fields": [{"offset": 0, "type": "float32", "name": "rpm"},
                                    {"offset": 4, "type": "float32[32]", "name": "features"},
                                    {"offset": 132, "type": "float32[4]", "name": DISPLACEMENT_NAMES},
                                    {"offset": 148, "type": "uint32", "name": "t20_ms"}],
                         "bearing_slices": {"1": [0, 16], "2": [16, 32]},
                         "note": "COE doc: 152 bytes; FDR deck: 168 (unreconciled)."},
        "raw_bearing_feature_order": BEARING_FEATURES,
        "feature_engineering": {
            "engineered_order": feat.ENGINEERED,
            "baseline": {"windows": feat.BASELINE_WINDOWS, "statistic": "median", "rel_floor": feat.BASELINE_REL_FLOOR,
                         "rig_policy": "commissioning capture (spec M5); else median of first 24 windows"},
            "eps": feat.EPS,
            "health_index": {"r_ref": feat.HI_R_REF, "curve_k": feat.HI_CURVE_K, "weights": feat.HI_WEIGHTS,
                             "clip_max": feat.HI_CLIP_MAX},
            "rpm_norm": {"center": feat.RPM_CENTER, "scale": feat.RPM_SCALE},
            "reference_impl": "aiengine/features.py::engineer_arrays (unchanged from contract v2)",
        },
        "onset": {"sigmas": labels.ONSET_SIGMAS, "consecutive": labels.ONSET_CONSECUTIVE, "sigma_floor": labels.SIGMA_FLOOR,
                  "baseline_windows": feat.BASELINE_WINDOWS, "run_in_skip_windows": labels.ONSET_RUN_IN_SKIP,
                  "use": "v3: reported as onset_detected only; class, RUL and stage come from the heads"},
        "classes": {"labels": list(up2.CLASS_ORDER), "index_is_output_channel": True,
                    "note": "alphabetical; logits = log probabilities, apply softmax (identity on probabilities). "
                            "No decision gate (plain argmax)."},
        "models": {
            "tft": {"onnx": CLS_FILE, "family": f"gradient-boosted trees ({c['cfg']['kind']}) in the TFT slot",
                    "encoder_length": T - 1, "decoder_length": 1, "columns": up2.COLUMNS, "scalers": ident,
                    "warmup_windows": T, **res["cls"]},
            "nhits": {"onnx": RUL_FILE, "family": f"gradient-boosted trees (rul {r['cfg']['kind']}, z {z['cfg']['kind']}) in the N-HiTS slot",
                      "encoder_length": NL, "prediction_length": 1, "columns": up2.COLUMNS, "scalers": ident,
                      "left_pad": "repeat first row", "output": "rul_head [B,3] = [rul_windows, z, stage]",
                      "in_graph_features": {"aggregates": uw.AGGS, "hi_levels": uw.HI_LEVELS if r["cfg"].get("levels") else [],
                                            "reference_impl": "aiengine/up2_window.py"}, **res["rul"]},
        },
        "rul": {"method": "direct_v3",
                "procedure": "rul_hours = rul_head[0] * cadence_hours (median spacing of the last <=48 window times); "
                             "stage = rul_head[2]; z = rul_head[1]. RUL is meaningful when stage >= 2.",
                # fields required by the v2 C# loader; unused by method direct_v3
                "history_windows": 24, "blend_w": 0.5, "u_min": 0.05, "class_thresholds": {}, "pooled_threshold": 1.0,
                "max_rul_factor": 20.0, "loglin_coef": [], "interval_rel": res.get("interval_rel", 0.25),
                "class_smoothing_windows": 5},
        "stage": f"rul_head[2] (from the z head): 1 if z < {z['cfg'].get('z_cut', 0.0)} (z_cut, chosen on validation) else min(6, 2 + floor(5 clip(z, 0, 1)))",
    }


def run(df_test, out_dir: Path = MODELS_DIR, prov: dict | None = None, interval_rel: float = 0.25) -> dict:
    from .up2_train import load_heads
    heads = load_heads()
    c, r, z = heads
    T, NL, R = c["cfg"]["T"], r["cfg"]["NL"], len(up2.COLUMNS)
    m_cls = build_cls(c["model"], T, R, importances_per_column(c["model"], R))
    m_rul = build_rul(r["model"], z["model"], NL, R, levels=r["cfg"].get("levels", False), z_cut=z["cfg"].get("z_cut", 0.0))
    onnx.save(m_cls, str(out_dir / CLS_FILE))
    onnx.save(m_rul, str(out_dir / RUL_FILE))
    par = parity(out_dir / CLS_FILE, out_dir / RUL_FILE, heads, df_test)
    res = {"cls": {"size_kb": round((out_dir / CLS_FILE).stat().st_size / 1024, 1), "onnx_inputs": ["encoder_cont", "decoder_cont"],
                   "outputs": ["logits", "variable_weights"], "batch_dim": "dynamic", "parity": par},
           "rul": {"size_kb": round((out_dir / RUL_FILE).stat().st_size / 1024, 1), "onnx_inputs": ["encoder_cont"],
                   "outputs": ["rul_head"], "batch_dim": "dynamic"},
           "interval_rel": interval_rel}
    con = contract(T, NL, heads, res, prov or {})
    (out_dir / CONTRACT_FILE).write_text(json.dumps(con, indent=2, default=float))
    from .engine import HybridEngine
    eng = HybridEngine(out_dir, contract_file=CONTRACT_FILE)
    rtf = df_test[df_test["run_to_failure"]]
    unit_id = rtf.groupby("unit_id").size().idxmin()
    unit = df_test[df_test["unit_id"] == unit_id].sort_values("window_index")
    golden = eng.golden_vectors(unit, n=12)
    # second segment of cases from the degraded part, so parity also covers stage >= 2 / non-healthy classes
    golden["cases_late"] = eng.golden_cases_at(unit, [int(x) for x in np.linspace(len(unit) * 0.6, len(unit) - 1, 8)])
    golden["provenance"] = "[SYNTHETIC] synth_v3 test run"
    (out_dir / GOLDEN_FILE).write_text(json.dumps(golden, indent=1, default=float))
    return {"contract": str(out_dir / CONTRACT_FILE), "parity": par, "golden_unit": unit_id, "T": T, "NL": NL}
