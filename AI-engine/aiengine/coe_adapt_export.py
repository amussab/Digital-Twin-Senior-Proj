"""Export the COE-adapted classifier: models/v3b_coe_cls.onnx + model_contract_v3b_coe.json + golden_vectors_v3b_coe.json.

The contract is a copy of model_contract_v3b.json with ONLY the classifier (TFT slot) swapped; the RUL/z/stage head
(v3b_rul.onnx), feature engineering, post-process and tensor names are unchanged, so the C# host loads it as is.
A calibration map (if selected) is embedded IN the classifier graph ahead of the window aggregator, so the host still
feeds the same identity-scaled `encoder_cont` / `decoder_cont` tensors. A cross-bearing gate (if selected) is the one
step that needs the two bearings of a window together; it is specified in the contract block `cross_bearing_gate` and
implemented in the Python reference engine (aiengine/engine.py, `HybridEngine.process`).
[SIMULATION]/[SYNTHETIC] provenance throughout.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import onnx
from onnx import helper, numpy_helper

from . import coe_adapt as A
from . import up2
from . import up2_export as X
from .config import CHECKPOINT_DIR, MODELS_DIR
from .coe_adapt_eval import (BASE_CLS, BASE_CONTRACT, NEW_CLS, NEW_CONTRACT, NEW_GOLDEN, RUN_DIR, load_coe,
                             selected_probs)


def _map_nodes(kind: str, maps: dict, src: str, dst: str, p: str):
    """Element-wise per-column map src [B, T, R] -> dst. affine: (x - c) / s. quantile: y0 + sum_k d_k relu(clip(x) - k_k)."""
    f = lambda v, n: numpy_helper.from_array(np.asarray(v, np.float32), n)   # noqa: E731
    if kind == "affine":
        sc = maps["affine"]
        c = [sc[k]["center"] for k in A.COLS]
        s = [sc[k]["scale"] for k in A.COLS]
        return ([helper.make_node("Sub", [src, p + "c"], [p + "xc"]), helper.make_node("Div", [p + "xc", p + "s"], [dst])],
                [f(c, p + "c"), f(s, p + "s")])
    lo, hi, y0, kn, de = A.quantile_tensors(maps["quantile"])
    nodes = [helper.make_node("Max", [src, p + "lo"], [p + "a"]),
             helper.make_node("Min", [p + "a", p + "hi"], [p + "xc"]),               # [B,T,R]
             helper.make_node("Unsqueeze", [p + "xc", p + "ax"], [p + "xu"]),         # [B,T,1,R]
             helper.make_node("Sub", [p + "xu", p + "kn"], [p + "dx"]),               # [B,T,K,R]
             helper.make_node("Relu", [p + "dx"], [p + "r"]),
             helper.make_node("Mul", [p + "r", p + "de"], [p + "rd"]),
             helper.make_node("ReduceSum", [p + "rd", p + "ax"], [p + "sum"], keepdims=0),
             helper.make_node("Add", [p + "sum", p + "y0"], [dst])]
    inits = [f(lo, p + "lo"), f(hi, p + "hi"), f(y0, p + "y0"), f(kn, p + "kn"), f(de, p + "de"),
             numpy_helper.from_array(np.array([2], np.int64), p + "ax")]
    return nodes, inits


def with_input_map(model: onnx.ModelProto, kind: str, maps: dict) -> onnx.ModelProto:
    g = model.graph
    ren = {"encoder_cont": "map_enc_out", "decoder_cont": "map_dec_out"}
    for n in g.node:
        for i, nm in enumerate(n.input):
            if nm in ren:
                n.input[i] = ren[nm]
    n1, i1 = _map_nodes(kind, maps, "encoder_cont", "map_enc_out", "mE_")
    n2, i2 = _map_nodes(kind, maps, "decoder_cont", "map_dec_out", "mD_")
    nodes = n1 + n2 + list(g.node)
    graph = helper.make_graph(nodes, g.name + "_coe_map", list(g.input), list(g.output),
                              initializer=list(g.initializer) + i1 + i2)
    m = helper.make_model(graph, opset_imports=list(model.opset_import), producer_name="aiengine.coe_adapt_export")
    m.ir_version = model.ir_version
    onnx.checker.check_model(m)
    return m


def build(sel: dict) -> tuple[onnx.ModelProto, dict]:
    maps = json.loads((RUN_DIR / "maps.json").read_text())
    R = len(A.COLS)
    name = sel["candidate"]
    if name.startswith("mix"):
        with open(RUN_DIR / f"{name}.pkl", "rb") as f:
            mdl = pickle.load(f)["model"]
        info = {"family": "gradient-boosted trees (xgb) in the TFT slot, retrained on synth_v3 train(+train2) + COE-train",
                "preprocessing": "none (identity scalers, as v3b)"}
    else:
        with open(CHECKPOINT_DIR / "up2_v3" / "cls.pkl", "rb") as f:
            mdl = pickle.load(f)["model"]
        info = {"family": "gradient-boosted trees (xgb) in the TFT slot: the v3/v3b trees unchanged",
                "preprocessing": "none" if name == "base" else f"in-graph {name} map (see coe_adaptation.input_map)"}
    m = X.build_cls(mdl, A.T_CLS, R, X.importances_per_column(mdl, R))
    if name in ("affine", "quantile"):
        m = with_input_map(m, name, maps)
    return m, {**info, "maps": maps}


def gate_block(sel: dict) -> dict | None:
    if not sel.get("gate_tau"):
        return None
    sig = sel["gate_signal"]
    return {
        "type": "partner_transmission_gate", "version": 1, "required": True,
        "tau": float(sel["gate_tau"]), "signal": sig,
        "family_of_class": {c: f"log_{f}_ratio" for c, f in A.FAMILY_OF.items()} if sig == "family"
        else {c: "log_env_ratio" for c in A.FAMILY_OF},
        "applies_to": "the classifier output (fault_class) of BOTH bearings of the SAME payload window, after each "
                      "bearing's TFT-slot argmax; RUL/stage/HI unaffected",
        "procedure": ("for k in (1, 2), using the PRE-gate classes of both bearings: c_k = argmax class of bearing k; if "
                      "c_k != 'healthy' and the partner p = 3-k has status 'ok': col = family_of_class[c_k]; "
                      "x_k = exp(e_k[col]) - 1; x_p = exp(e_p[col]) - 1, where e = the CURRENT window's engineered row "
                      "(features.py::engineer_arrays, the same row fed as decoder_cont); if x_p > 0 and x_k < tau * x_p: "
                      "fault_class_k = 'healthy' (report transmitted_from_partner = true; class_probs unchanged, "
                      "diagnostic). If the partner is warming up, no gate."),
        "reference_impl": "aiengine/coe_adapt.py::gate (offline), aiengine/engine.py::HybridEngine._gate (streaming)",
        "selected_on": "synth_coe COE-val runs only",
    }


def parity(onnx_path, sel, B) -> dict:
    P_ref = selected_probs(sel, B)
    P = A.onnx_probs(onnx_path, B)
    out = {"windows": int(len(B)), "prob_max_abs_diff": float(np.abs(P - P_ref).max()),
           "argmax_mismatch": int((P.argmax(1) != P_ref.argmax(1)).sum()),
           "reference": "numpy map + base ONNX / native predict_proba"}
    if sel["candidate"] == "quantile":
        maps = json.loads((RUN_DIR / "maps.json").read_text())
        out["relu_form_vs_interp_max_abs"] = float(np.abs(A.apply_quantile_relu(B[:2000], maps["quantile"])
                                                          - A.apply_quantile(B[:2000], maps["quantile"])).max())
    return out


def run() -> dict:
    sel = json.loads((RUN_DIR / "selection.json").read_text())
    model, info = build(sel)
    onnx.save(model, str(MODELS_DIR / NEW_CLS))
    coe = load_coe()
    va = coe[coe["split"] == "val"]                       # parity only (no scoring): val runs
    B, _ = A.blocks_for(va)
    par = parity(MODELS_DIR / NEW_CLS, sel, B)
    con = json.loads((MODELS_DIR / BASE_CONTRACT).read_text())
    from .train import git_commit
    con["contract_version"] = "3b-coe"
    con["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con["generated_by"] = "AI-engine/aiengine/coe_adapt_export.py (copy of model_contract_v3b.json, classifier swapped)"
    con["provenance"]["datasets"] = list(con["provenance"]["datasets"]) + ["synth_coe (COE simulator -> CoePipeline; train/val)"]
    con["provenance"]["coe_adapt"] = {"git_commit": git_commit(), "selection": sel,
                                      "note": "[SIMULATION]+[SYNTHETIC] classifier adapted to COE-pipeline features; "
                                              "selected on COE-val only; RUL/z/stage heads = v3b unchanged"}
    t = con["models"]["tft"]
    t["onnx"] = NEW_CLS
    t["family"] = info["family"]
    t["size_kb"] = round((MODELS_DIR / NEW_CLS).stat().st_size / 1024, 1)
    t["parity"] = par
    t["onnx_inputs"], t["outputs"] = ["encoder_cont", "decoder_cont"], ["logits", "variable_weights"]
    ad = {"candidate": sel["candidate"], "tensor_names_unchanged": True,
          "host_side_change": "none" if not sel.get("gate_tau") else "cross_bearing_gate (see block)"}
    if sel["candidate"] in ("affine", "quantile"):
        ad["input_map"] = {"kind": sel["candidate"], "where": "inside v3b_coe_cls.onnx, applied element-wise to "
                           "encoder_cont and decoder_cont BEFORE the window aggregator; the host keeps identity scalers",
                           "columns": A.COLS, "params": info["maps"][sel["candidate"]],
                           "reference_impl": f"aiengine/coe_adapt.py::apply_{sel['candidate']}",
                           "applies_to_rul_head": False}
    con["coe_adaptation"] = ad
    g = gate_block(sel)
    if g:
        con["cross_bearing_gate"] = g
    con["disclosures"]["classifier"] = (f"contract 3b-coe: classifier = {sel['candidate']}"
                                        + (f" + cross-bearing gate (tau {sel['gate_tau']}, {sel['gate_signal']})" if g else "")
                                        + "; selected on synth_coe COE-val; COE-test scored once (reports/coe_adapt_*.md)")
    con["disclosures"]["provenance"] += (" Contract 3b-coe adds synth_coe [SIMULATION] windows (COE rig simulator through "
                                         "the real CoePipeline), never rig/hardware data.")
    (MODELS_DIR / NEW_CONTRACT).write_text(json.dumps(con, indent=2, default=float))

    # golden vectors: single-unit parity cases (as v3b) + rig-level cases that exercise the gate
    from .engine import HybridEngine
    eng = HybridEngine(MODELS_DIR, contract_file=NEW_CONTRACT)
    te = coe[coe["split"] == "test"]
    from .config import BEARING_FEATURES
    runs = te[(te["scenario"] == "b1")].groupby("bearing_key").size().index.tolist()
    rk = runs[0]
    run_df = te[te["bearing_key"] == rk]
    u1 = run_df[run_df["bearing"] == 1].sort_values("window_index")
    golden = eng.golden_vectors(u1, n=12)
    golden["cases_late"] = eng.golden_cases_at(u1, [int(x) for x in np.linspace(40, len(u1) - 1, 8)])
    eng.reset()
    u2 = run_df[run_df["bearing"] == 2].sort_values("window_index")
    rig_in, rig_out = [], []
    for (r1, r2) in zip(u1.itertuples(index=False), u2.itertuples(index=False)):
        f32 = np.concatenate([[getattr(r1, c) for c in BEARING_FEATURES], [getattr(r2, c) for c in BEARING_FEATURES]])
        t_ms = int(r1.t20_ms)
        o = eng.process(float(r1.rpm), f32, t_ms)
        rig_in.append({"window_index": int(r1.window_index), "rpm": float(r1.rpm), "t20_ms": t_ms, "features32": f32.tolist()})
        rig_out.append([{k: v for k, v in b.items() if k in ("bearing", "status", "fault_class", "transmitted_from_partner",
                                                             "class_probs", "health_stage")} for b in o["bearings"]])
    golden["rig_stream"] = {"bearing_key": rk, "scenario": "bearing 1 faulted, bearing 2 healthy (COE-test run)",
                            "note": "feed features32 in order through ONE engine; compare per-bearing fault_class "
                                    "(post-gate) and class_probs (1e-4 abs)", "inputs": rig_in, "outputs": rig_out}
    golden["provenance"] = "[SIMULATION] synth_coe test run (COE simulator -> CoePipeline)"
    (MODELS_DIR / NEW_GOLDEN).write_text(json.dumps(golden, indent=1, default=float))
    return {"contract": str(MODELS_DIR / NEW_CONTRACT), "onnx": NEW_CLS, "parity": par, "golden_run": rk,
            "gate": bool(g)}
