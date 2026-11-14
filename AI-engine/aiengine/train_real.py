"""ONE command: train + evaluate + export a contract from the recorder's labelled rig data.

    python -m aiengine.train_real --data <recorder training_data dir | CSV> [--include-sim] [--out v4_real]

Steps (all printed, all written to reports/train_real_<ts>.{md,json}):
 1. load runs (aiengine.real_data), drop the settle windows of every run (policy: 120 s -> ceil(120*rpm/1200) windows),
    build the per-rpm COMMISSIONING BASELINE from is_baseline runs (settle dropped, next 60 s, median; fallback = median of
    the healthy runs, said in the report).
 2. engineered features exactly as the engine (features.engineer_arrays(extended=True), the 16 V3E columns) -> 25-window
    blocks -> the same ONNX window aggregator the deployed graphs use. Labels PER BEARING using the run's `position`
    (the bearing that does not hold the faulty part is labelled healthy).
 3. split BY RUN/MOUNT: held-out set chosen by hash BEFORE fitting and registered in splits.json (section rig_real;
    frozen on later invocations). Model selection = grouped CV on the training part only (group = bearing_id, else run).
 4. classifier (xgb, TFT slot) + health-level/stage head (xgb regression on z, N-HiTS slot; severity 0/1/2/3 -> stage
    1/3/4/6), optional synth_coe_node rows (--include-sim, weight --sim-weight) chosen vs real-only by grouped CV only.
 5. score on held-out real runs against IS3 / S8 targets; export models/model_contract_<out>.json (+ onnx, golden vectors,
    baseline_<out>.json) loadable by NodeEngine(contract_file=...) and node_live (NODE_CONTRACT=<out> NODE_BASELINE=...).
The RUL output of the exported head is a constant 0 (no run-to-failure data on the rig): it is NOT a RUL model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from . import real_data as R
from . import up2
from . import up2_window as uw
from .config import MODELS_DIR, REPORTS_DIR, ROOT, SEED

CLASS_ORDER = list(up2.CLASS_ORDER)
V3E = list(feat.V3E_COLUMNS)
EXT_IDX = [feat.ENGINEERED_EXT.index(c) for c in V3E]
T = R.ENCODER_LEN + 1
NL = R.ENCODER_LEN
STAGE_OF_SEV = {0: 1, 1: 3, 2: 4, 3: 6}
Z_OF_SEV = {0: -0.1, 1: 0.3, 2: 0.5, 3: 0.9}      # stage_from_z(z, 0) -> 1, 3, 4, 6
GRID = [dict(max_depth=3, n_estimators=150, learning_rate=0.1), dict(max_depth=5, n_estimators=250, learning_rate=0.06)]
SPLIT_SECTION = "rig_real"


class Insufficient(SystemExit):
    pass


# ------------------------------------------------------------------------------------------------ baselines / samples
def build_baselines(df: pd.DataFrame, rt: pd.DataFrame, settle_s: float, baseline_s: float):
    """-> ({rpm_key: {rpm, windows, settle_windows, b1, b2, source}}, notes)."""
    out, notes = {}, []
    for key in sorted(rt["rpm_key"].unique()):
        sw, bw = R.settle_windows(key, settle_s), R.baseline_windows(key, baseline_s)
        slices, src = [], None
        for rid in rt[(rt["rpm_key"] == key) & rt["is_baseline"]]["run_id"]:
            g = df[df["run_id"] == rid].sort_values("order")[R.FEATURE_COLUMNS].to_numpy()
            seg = g[sw:sw + bw]
            if len(seg) < max(10, bw // 3):
                notes.append(f"baseline run {rid} @ {key} rpm too short ({len(g)} windows; needs settle {sw} + >= {max(10, bw // 3)}): ignored")
                continue
            if len(seg) < bw:
                notes.append(f"baseline run {rid} @ {key} rpm: only {len(seg)}/{bw} baseline windows after settle")
            slices.append(seg)
            src = "is_baseline run(s)"
        if not slices:
            for rid in rt[(rt["rpm_key"] == key) & (rt["label"] == "healthy") & ~rt["is_baseline"]]["run_id"]:
                g = df[df["run_id"] == rid].sort_values("order")[R.FEATURE_COLUMNS].to_numpy()
                if len(g) > sw:
                    slices.append(g[sw:])
            if slices:
                src = "FALLBACK: median of healthy runs (no is_baseline run at this speed)"
                notes.append(f"{key} rpm: no commissioning-baseline run; using the median of the healthy runs (record a baseline run!)")
        if not slices:
            notes.append(f"{key} rpm: no baseline possible (no baseline or healthy run): runs at this speed are dropped")
            continue
        A = np.concatenate(slices)
        out[int(key)] = {"rpm": int(key), "windows": int(len(A)), "settle_windows": int(sw), "source": src,
                         "b1": feat._floor(np.median(A[:, :16], axis=0)).tolist(),
                         "b2": feat._floor(np.median(A[:, 16:], axis=0)).tolist()}
    return out, notes


def bearing_samples(raw16: np.ndarray, rpm: np.ndarray, base16) -> tuple[np.ndarray, np.ndarray]:
    """engine-identical rows -> (Xc [m,128] classifier features, Xs [m,128] stage-head features) for windows with 25 rows of history."""
    eng = feat.engineer_arrays(raw16, rpm, np.asarray(base16, float), extended=True)[:, EXT_IDX]
    B, full = uw.blocks(eng, T)
    Bf = B[full]
    if not len(Bf):
        return np.zeros((0, 8 * len(V3E)), np.float32), np.zeros((0, 8 * len(V3E)), np.float32)
    return uw.compute(Bf, True), uw.compute(Bf[:, 1:], False)


def build_samples(df: pd.DataFrame, rt: pd.DataFrame, baselines: dict, settle_s: float, baseline_s: float = R.BASELINE_S):
    """Baseline runs contribute only the windows AFTER settle + the baseline period (healthy, both bearings), so truly healthy
    bearings (features near the baseline) are in the training data and the baseline windows themselves are never trained on."""
    parts, notes = [], []
    for r in rt.itertuples():
        base = baselines.get(int(r.rpm_key))
        if base is None:
            notes.append(f"run {r.run_id}: no baseline at {r.rpm_key} rpm: dropped")
            continue
        g = df[df["run_id"] == r.run_id].sort_values("order")
        skip = R.settle_windows(r.rpm_key, settle_s) + (R.baseline_windows(r.rpm_key, baseline_s) if r.is_baseline else 0)
        g = g.iloc[skip:]
        if len(g) < T + 5:
            notes.append(f"run {r.run_id}: only {len(g)} windows after the settle{' + baseline' if r.is_baseline else ''} drop (< {T + 5}): dropped")
            continue
        raw, rpm = g[R.FEATURE_COLUMNS].to_numpy(float), g["rpm"].to_numpy(float)
        sev_run = np.nan if pd.isna(r.severity) else float(r.severity)
        for k in (1, 2):
            faulty = r.label != "healthy" and r.position in ("both", f"b{k}")
            Xc, Xs = bearing_samples(raw[:, (k - 1) * 16:k * 16], rpm, base[f"b{k}"])
            m = len(Xc)
            parts.append({"Xc": Xc, "Xs": Xs, "y": np.full(m, CLASS_ORDER.index(r.label if faulty else "healthy")),
                          "sev": np.full(m, sev_run if faulty else 0.0), "run_id": [r.run_id] * m, "bearing": np.full(m, k),
                          "rpm_key": np.full(m, r.rpm_key), "group": [r.bearing_id or r.run_id] * m,
                          "fault_run": np.full(m, faulty), "sim": np.zeros(m, bool)})
    return parts, notes


def cat(parts: list[dict]) -> dict:
    return {k: (np.concatenate([p[k] for p in parts]) if k not in ("run_id", "group") else sum((list(p[k]) for p in parts), []))
            for k in parts[0]}


def sim_samples(max_per_class: int = 4000) -> dict:
    """[SIMULATION] synth_coe_node train-split windows through the same engineering/aggregation (subsampled for speed)."""
    from . import coe_adapt as A
    from . import coe_dataset as D
    df = D.generate()
    df = df[df["split"] == "train"]
    eng = feat.engineer_table(df, extended=True)
    eng["observable_class"] = eng["observable_class"].astype(str)
    Bk, idx = A.blocks_for(eng, columns=V3E)
    t = up2.merge_truth(idx, eng, ["observable_class", "transition", "severity", "rpm"])
    keep = ~t["transition"].astype(bool).to_numpy()
    Bk, t = Bk[keep], t[keep].reset_index(drop=True)
    y = t["observable_class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    rng = np.random.default_rng(SEED)
    sel = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:max_per_class] for c in np.unique(y)])
    Bk, t, y = Bk[sel], t.iloc[sel].reset_index(drop=True), y[sel]
    s = t["severity"].to_numpy(float)
    sev = np.where(y == CLASS_ORDER.index("healthy"), 0, np.where(s <= 1 / 3, 1, np.where(s <= 2 / 3, 2, 3))).astype(float)
    n = len(y)
    return {"Xc": uw.compute(Bk, True), "Xs": uw.compute(Bk[:, 1:], False), "y": y, "sev": sev,
            "run_id": list(t["unit_id"]), "bearing": np.zeros(n, int), "rpm_key": np.array([R.rpm_key(r) for r in t["rpm"]]),
            "group": list(t["unit_id"]), "fault_run": y != CLASS_ORDER.index("healthy"), "sim": np.ones(n, bool)}


# ------------------------------------------------------------------------------------------------ split registry
def _h(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def choose_holdout(rt: pd.DataFrame, frozen: dict[str, str]) -> dict[str, str]:
    """run_id -> 'heldout'|'train'. Unit of splitting = bearing_id (else run_id) so one physical bearing never sits on both sides.
    Rule: for every (label, level) group with >= 3 units hold out max(1, floor(25 %)); then every label with >= 3 units and
    no held-out unit yet gets one. Frozen assignments (from splits.json) are kept."""
    rt = rt[~rt["is_baseline"]].copy()
    rt["unit"] = [b or r for b, r in zip(rt["bearing_id"], rt["run_id"])]
    unit_of = dict(zip(rt["run_id"], rt["unit"]))
    fz_unit = {}
    for rid, sp in frozen.items():
        if rid in unit_of:
            fz_unit[unit_of[rid]] = sp
    units = rt.groupby("unit").agg(label=("label", "first"), sev=("severity", "first")).reset_index()
    asg = dict(fz_unit)

    def pick(group):
        free = sorted([u for u in group["unit"] if u not in asg], key=lambda u: _h(SEED, u))
        have = sum(1 for u in group["unit"] if asg.get(u) == "heldout")
        need = max(1, len(group) // 4) - have
        for u in free[:max(need, 0)]:
            asg[u] = "heldout"
    for _, g in units.assign(sev=units["sev"].fillna(-1)).groupby(["label", "sev"]):
        if len(g) >= 3:
            pick(g)
    for _, g in units.groupby("label"):
        if len(g) >= 3 and not any(asg.get(u) == "heldout" for u in g["unit"]):
            free = sorted(g["unit"], key=lambda u: _h(SEED, u))
            asg[free[0]] = "heldout"
    return {rid: asg.get(u, "train") for rid, u in unit_of.items()}


def register_split(path: Path, assignment: dict[str, str], note: str) -> dict:
    d = json.loads(path.read_text()) if path.exists() else {"sections": {}}
    sec = d.setdefault("sections", {}).setdefault(SPLIT_SECTION, {"units": {}})
    sec["units"].update(assignment)
    sec["counts"] = {s: int(sum(1 for v in sec["units"].values() if v == s)) for s in ("train", "heldout")}
    sec["policy"] = "train_real.choose_holdout (hash of run/bearing, frozen once registered)"
    sec["created_utc"] = sec.get("created_utc") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    sec["note"] = note
    path.write_text(json.dumps(d, indent=1))
    return sec


# ------------------------------------------------------------------------------------------------ models
class Clf:
    """xgb multi-class over the classes present in training; predict_proba is always [n, 5] in CLASS_ORDER."""

    def __init__(self, hp: dict, threads: int = 8):
        self.hp, self.threads = hp, threads

    def fit(self, X, y, w=None):
        import xgboost as xgb
        self.present = sorted(int(c) for c in np.unique(y))
        if len(self.present) < 2:
            raise Insufficient("need at least 2 classes with data to train a classifier")
        ym = np.searchsorted(self.present, y)
        cnt = np.bincount(ym, minlength=len(self.present)).astype(float)
        cw = ((len(ym) / (len(cnt) * cnt)) ** 0.5)[ym]
        sw = cw if w is None else cw * w
        self.m = xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.6, min_child_weight=3, reg_lambda=1.0, tree_method="hist",
                                   objective="multi:softprob", random_state=SEED, n_jobs=self.threads, **self.hp)
        self.m.fit(X, ym, sample_weight=sw)
        return self

    def proba(self, X):
        P = np.zeros((len(X), len(CLASS_ORDER)))
        P[:, self.present] = self.m.predict_proba(X)
        return P


def fit_sev(Xs, sev, w=None, threads: int = 8):
    import xgboost as xgb
    ok = ~np.isnan(sev)
    z = np.array([Z_OF_SEV[int(s)] for s in sev[ok]])
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.6,
                         min_child_weight=3, tree_method="hist", random_state=SEED, n_jobs=threads)
    m.fit(Xs[ok], z, sample_weight=None if w is None else w[ok])
    return m


def macro_f1(y, p) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0))


def grouped_cv(real: dict, sim: dict | None, hp: dict, sim_w: float, threads: int) -> dict:
    from sklearn.model_selection import GroupKFold
    groups = np.array(real["group"])
    ng = len(set(groups))
    if ng < 2:
        return {"macro_f1": None, "folds": 0}
    k = min(5, ng)
    oof = np.full(len(real["y"]), -1)
    for tr, va in GroupKFold(k).split(real["Xc"], real["y"], groups):
        X, y, w = real["Xc"][tr], real["y"][tr], None
        if sim is not None and sim_w > 0:
            X, y = np.concatenate([X, sim["Xc"]]), np.concatenate([y, sim["y"]])
            w = np.concatenate([np.ones(len(tr)), np.full(len(sim["y"]), sim_w)])
        if len(set(y)) < 2:
            continue
        oof[va] = Clf(hp, threads).fit(X, y, w).proba(real["Xc"][va]).argmax(1)
    ok = oof >= 0
    return {"macro_f1": macro_f1(real["y"][ok], oof[ok]) if ok.any() else None, "folds": k, "scored_windows": int(ok.sum())}


# ------------------------------------------------------------------------------------------------ metrics
def evaluate(d: dict, P: np.ndarray, z: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    y, pred = d["y"], P.argmax(1)
    labels = sorted(set(y))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per = {CLASS_ORDER[c]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "windows": int(sup[i])}
           for i, c in enumerate(labels)}
    allc = sorted(set(labels) | set(pred))
    cm = confusion_matrix(y, pred, labels=allc)
    per_rpm = {}
    for k in sorted(set(d["rpm_key"])):
        m = d["rpm_key"] == k
        per_rpm[str(int(k))] = {"macro_f1": macro_f1(y[m], pred[m]), "windows": int(m.sum())}
    healthy = CLASS_ORDER.index("healthy")
    res = {"macro_f1": macro_f1(y, pred), "per_class": per, "classes_in_truth": [CLASS_ORDER[c] for c in labels],
           "confusion": {"labels": [CLASS_ORDER[c] for c in allc], "rows_true_cols_pred": cm.tolist()},
           "per_rpm": per_rpm, "windows": int(len(y)),
           "healthy_false_alarm_rate": float((pred[y == healthy] != healthy).mean()) if (y == healthy).any() else None}
    known = ~np.isnan(d["sev"])
    if known.any():
        st = up2.stage_from_z(z[known], 0.0)
        tgt = np.array([STAGE_OF_SEV[int(s)] for s in d["sev"][known]])
        err = np.abs(st - tgt)
        res["stage"] = {"windows": int(known.sum()), "frac_error_le_1": float((err <= 1).mean()), "max_error": int(err.max()),
                        "mean_abs_error": float(err.mean()), "mapping_severity_to_stage": STAGE_OF_SEV}
    det = known & (d["sev"] >= 1) & d["fault_run"]
    if det.any():
        caught = pred[det] != healthy
        runs = {}
        for rid in sorted(str(x) for x in set(np.array(d["run_id"])[det])):
            m = det & (np.array(d["run_id"]) == rid)
            runs[rid] = float((pred[m] != healthy).mean())
        res["detection"] = {"fault_window_recall": float(caught.mean()), "per_run_recall": runs,
                            "runs_caught_ge_50pct": int(sum(v >= 0.5 for v in runs.values())), "runs": len(runs)}
    return res


# ------------------------------------------------------------------------------------------------ export
def _build_cls_padded(model: Clf, R_: int):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from . import up2_export as X
    agg = uw.aggregator_graph(T, R_, with_decoder=True, out_name="feat")
    tn, ti, touts, tops = X._inline(X._convert(model.m, len(uw.AGGS) * R_), "c_", "feat")
    k = len(model.present)
    M = np.zeros((k, len(CLASS_ORDER)), np.float32)
    for j, c in enumerate(model.present):
        M[j, c] = 1.0
    imp = X.importances_per_column(model.m, R_)
    post = [helper.make_node("MatMul", [touts["probabilities"], "c_pad"], ["c_prob5"]),
            helper.make_node("Max", ["c_prob5", "c_eps"], ["c_pclip"]),
            helper.make_node("Log", ["c_pclip"], ["c_logp"]),
            helper.make_node("Unsqueeze", ["c_logp", "c_ax1"], ["logits"]),
            helper.make_node("Identity", ["c_vw"], ["variable_weights"])]
    inits = list(agg.graph.initializer) + ti + [numpy_helper.from_array(M, "c_pad"),
                                                numpy_helper.from_array(np.array(1e-12, np.float32), "c_eps"),
                                                numpy_helper.from_array(np.array([1], np.int64), "c_ax1"),
                                                numpy_helper.from_array(np.asarray(imp, np.float32).reshape(1, R_), "c_vw")]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1, len(CLASS_ORDER)]),
               helper.make_tensor_value_info("variable_weights", TensorProto.FLOAT, [1, R_])]
    return X._model(list(agg.graph.node) + tn + post, inits, list(agg.graph.input), outputs, list(agg.opset_import) + tops, "real_cls")


def _build_sev(z_model, R_: int):
    from onnx import TensorProto, helper, numpy_helper
    from . import up2_export as X
    agg = uw.aggregator_graph(NL, R_, with_decoder=False, out_name="feat")
    zn, zi, zouts, zops = X._inline(X._convert(z_model, len(uw.AGGS) * R_), "z_", "feat")
    zv = list(zouts.values())[0]
    post = [helper.make_node("Mul", [zv, "p_zero"], ["p_rul"]),                       # RUL: constant 0 (not trained)
            helper.make_node("Clip", [zv, "p_zero", "p_one"], ["p_zc"]),
            helper.make_node("Mul", ["p_zc", "p_five"], ["p_z5"]),
            helper.make_node("Floor", ["p_z5"], ["p_fl"]),
            helper.make_node("Add", ["p_fl", "p_two"], ["p_st0"]),
            helper.make_node("Min", ["p_st0", "p_six"], ["p_st1"]),
            helper.make_node("Less", [zv, "p_zero"], ["p_pre"]),
            helper.make_node("Where", ["p_pre", "p_one", "p_st1"], ["p_stage"]),
            helper.make_node("Concat", ["p_rul", zv, "p_stage"], ["rul_head"], axis=1)]
    c = lambda v, n: numpy_helper.from_array(np.array(v, np.float32), n)   # noqa: E731
    inits = list(agg.graph.initializer) + zi + [c(1.0, "p_one"), c(0.0, "p_zero"), c(5.0, "p_five"), c(2.0, "p_two"), c(6.0, "p_six")]
    outputs = [helper.make_tensor_value_info("rul_head", TensorProto.FLOAT, ["batch", 3])]
    return X._model(list(agg.graph.node) + zn + post, inits, list(agg.graph.input), outputs, list(agg.opset_import) + zops, "real_sev")


def export(out: str, clf: Clf, zm, baselines: dict, summary: dict, models_dir: Path, settle_s: float, baseline_s: float,
           golden_unit: pd.DataFrame | None, prov: dict) -> dict:
    import onnx
    from . import coe_adapt as A
    models_dir.mkdir(parents=True, exist_ok=True)
    cls_f, sev_f = f"{out}_cls.onnx", f"{out}_sev.onnx"
    onnx.save(_build_cls_padded(clf, len(V3E)), str(models_dir / cls_f))
    onnx.save(_build_sev(zm, len(V3E)), str(models_dir / sev_f))
    con = json.loads((MODELS_DIR / "model_contract_v3e_node.json").read_text())
    con.pop("cross_bearing_gate", None)
    con.pop("coe_adaptation", None)
    con["contract_version"] = f"4-real.1:{out}"
    con["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con["generated_by"] = "AI-engine/aiengine/train_real.py (copy of model_contract_v3e_node.json; heads + inputs replaced)"
    con["provenance"] = prov
    pol = con["feature_engineering"]["baseline"]["policy"]
    pol["settle_seconds"], pol["baseline_seconds"] = settle_s, baseline_s
    ident = {c: {"center": 0.0, "scale": 1.0, "kind": "identity"} for c in V3E}
    t, n = con["models"]["tft"], con["models"]["nhits"]
    t.update({"onnx": cls_f, "family": "gradient-boosted trees (xgb) in the TFT slot, trained on rig recordings [MEASURED on rig]",
              "columns": V3E, "scalers": ident, "size_kb": round((models_dir / cls_f).stat().st_size / 1024, 1),
              "parity": summary.get("onnx_parity", {}), "present_classes": [CLASS_ORDER[c] for c in clf.present]})
    n.update({"onnx": sev_f, "family": "gradient-boosted regression (xgb) of z on the health level, in the N-HiTS slot: rul_head = [0, z, stage]",
              "encoder_length": NL, "prediction_length": 1, "columns": V3E, "scalers": ident,
              "size_kb": round((models_dir / sev_f).stat().st_size / 1024, 1),
              "output": "rul_head [B,3] = [0 (RUL NOT TRAINED), z, stage]; stage = 1 if z < 0 else min(6, 2 + floor(5 clip(z,0,1))); "
                        "severity 0/1/2/3 -> z -.1/.3/.5/.9 -> stage 1/3/4/6"})
    con["rul"]["procedure"] = "RUL is NOT estimated by this contract (rul_head[0] = 0); only health_stage / z are meaningful."
    con["disclosures"]["provenance"] = prov["note"]
    con["real_data"] = summary.get("data_summary", {})
    (models_dir / f"model_contract_{out}.json").write_text(json.dumps(con, indent=2, default=float))
    from .node_adapter import NodeEngine
    ne = NodeEngine(contract_file=f"model_contract_{out}.json", model_dir=models_dir)
    ne.saved = {int(k): v for k, v in baselines.items()}
    ne.save_baseline(models_dir / f"baseline_{out}.json")
    if golden_unit is not None:
        from .engine import HybridEngine
        eng = HybridEngine(models_dir, contract_file=f"model_contract_{out}.json")
        gv = eng.golden_vectors(golden_unit, n=12)
        gv["provenance"] = prov["note"]
        (models_dir / f"golden_vectors_{out}.json").write_text(json.dumps(gv, indent=1, default=float))
    return {"contract": str(models_dir / f"model_contract_{out}.json"), "baseline": str(models_dir / f"baseline_{out}.json")}


def onnx_parity(models_dir: Path, out: str, clf: Clf, X_blocks_feat_src: np.ndarray) -> dict:
    """ONNX classifier (aggregator inside the graph) vs native xgb on the same engineered blocks [n,25,16]."""
    from . import coe_adapt as A
    P = A.onnx_probs(models_dir / f"{out}_cls.onnx", X_blocks_feat_src)
    Pr = clf.proba(uw.compute(X_blocks_feat_src, True))
    return {"windows": int(len(P)), "prob_max_abs_diff": float(np.abs(P - Pr).max()), "argmax_mismatch": int((P.argmax(1) != Pr.argmax(1)).sum()),
            "reference": "native xgboost predict_proba"}


def stream_check(models_dir: Path, out: str, baselines: dict, df: pd.DataFrame, rt: pd.DataFrame, settle_s: float) -> dict:
    """Feed one (held-out if any) run through NodeEngine with the exported contract; latency + class counts."""
    from .node_adapter import NodeEngine
    ne = NodeEngine(contract_file=f"model_contract_{out}.json", model_dir=models_dir)
    ne.saved = {int(k): v for k, v in baselines.items()}
    cands = rt[~rt["is_baseline"] & rt["rpm_key"].isin(baselines.keys())]
    if cands.empty:
        return {}
    r = cands.sort_values("label", key=lambda c: c == "healthy").iloc[0]      # prefer a faulty run
    g = df[df["run_id"] == r["run_id"]].sort_values("order").iloc[R.settle_windows(r["rpm_key"], settle_s):]
    lat, counts = [], {1: {}, 2: {}}
    for w in g.itertuples():
        msg = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": w.rpm, "window_id": int(w.window_id),
               "features": [getattr(w, c) for c in R.FEATURE_COLUMNS]}
        o = ne.predict(msg)
        if o["bearings"][0].get("status") == "ok":
            lat.append(o["latency_ms"])
        for b in o["bearings"]:
            if b.get("status") == "ok":
                counts[b["bearing"]][b["fault_class"]] = counts[b["bearing"]].get(b["fault_class"], 0) + 1
    return {"run_id": r["run_id"], "label": r["label"], "windows_ok": len(lat), "class_counts": {str(k): v for k, v in counts.items()},
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "max": float(max(lat))} if lat else None}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.train_real", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="v4_real")
    ap.add_argument("--include-sim", action="store_true")
    ap.add_argument("--sim-weight", type=float, default=0.3)
    ap.add_argument("--settle-s", type=float, default=R.SETTLE_S)
    ap.add_argument("--baseline-s", type=float, default=R.BASELINE_S)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--reports-dir", default=str(REPORTS_DIR))
    ap.add_argument("--splits-file", default=str(ROOT / "splits.json"))
    ap.add_argument("--tag", default="[MEASURED on rig]", help="provenance tag written on every number (rehearsal: '[SIMULATION rehearsal]')")
    a = ap.parse_args(argv)
    if a.out in ("v3e_node", "v3d_node", "v3b", "v3", "v2", "v3b_coe"):
        raise SystemExit(f"refusing to overwrite the existing contract '{a.out}'")
    t0 = time.perf_counter()
    models_dir, reports_dir = Path(a.models_dir), Path(a.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = R.load_runs(a.data)
    if df.empty:
        raise SystemExit(f"no windows found under {a.data}")
    rt = R.run_table(df)
    notes: list[str] = []
    print(f"{a.tag} {len(rt)} runs, {len(df)} windows ({df.attrs.get('dropped_invalid_windows', 0)} invalid windows dropped)")
    runs_per_label = rt[~rt["is_baseline"]].groupby("label")["run_id"].nunique().to_dict()
    mounts_per_label = rt[~rt["is_baseline"]].groupby("label")["bearing_id"].apply(lambda s: len({x for x in s if x})).to_dict()
    for c in R.CONDITIONS:
        if runs_per_label.get(c, 0) < 2:
            notes.append(f"INSUFFICIENT DATA: class '{c}' has {runs_per_label.get(c, 0)} run(s) (< 2): no train/held-out split by run is "
                         "possible for it; its numbers are not trustworthy" + (" (class absent: never predicted)" if not runs_per_label.get(c) else ""))
    nonbase = rt[~rt["is_baseline"]]
    if nonbase["label"].nunique() < 2:
        msg = (f"cannot train: only class(es) {sorted(nonbase['label'].unique())} recorded (need healthy + at least one fault, "
               "each from >= 2 runs). Run `python -m aiengine.data_coverage <dir>` to see what is missing.")
        print("ERROR", msg)
        raise SystemExit(2)
    baselines, bn = build_baselines(df, rt, a.settle_s, a.baseline_s)
    notes += bn
    parts, sn = build_samples(df, rt, baselines, a.settle_s, a.baseline_s)
    notes += sn
    if not parts:
        print("ERROR no usable windows after settle drop / baseline:\n  " + "\n  ".join(notes))
        raise SystemExit(2)
    real = cat(parts)
    # ---- split registry (BEFORE any fit) ----
    sp_path = Path(a.splits_file)
    sec0 = json.loads(sp_path.read_text()).get("sections", {}).get(SPLIT_SECTION, {}).get("units", {}) if sp_path.exists() else {}
    asg = choose_holdout(rt, sec0)
    sec = register_split(sp_path, asg, f"{a.tag} registered before fitting by train_real; data {a.data}")
    held_runs = sorted(r for r, s in asg.items() if s == "heldout")
    print("held-out runs (registered in splits.json before fitting):", held_runs or "NONE (too few runs per condition: CV only)")
    is_held = np.array([r in set(held_runs) for r in real["run_id"]])
    tr = {k: (v[~is_held] if isinstance(v, np.ndarray) else [x for x, h in zip(v, is_held) if not h]) for k, v in real.items()}
    ho = {k: (v[is_held] if isinstance(v, np.ndarray) else [x for x, h in zip(v, is_held) if h]) for k, v in real.items()} if is_held.any() else None
    if len(set(tr["y"])) < 2:
        print("ERROR training part has < 2 classes after the held-out split")
        raise SystemExit(2)
    # ---- model selection: grouped CV on the training part only ----
    sim = sim_samples() if a.include_sim else None
    cands = [(hp, 0.0) for hp in GRID] + ([(hp, a.sim_weight) for hp in GRID] if sim is not None else [])
    cv_rows, best = [], None
    for hp, sw in cands:
        cv = grouped_cv(tr, sim, hp, sw, a.threads)
        cv_rows.append({"hp": hp, "sim_weight": sw, **cv})
        print(f"  CV hp={hp} sim_w={sw}: macro-F1 {cv['macro_f1']}")
        sc = -1 if cv["macro_f1"] is None else cv["macro_f1"]
        if best is None or sc > best[0] + 1e-9:
            best = (sc, hp, sw)
    _, hp, sw = best
    # ---- final fit (train part [+ sim]) ----
    X, y, w, Xs, sev = tr["Xc"], tr["y"], None, tr["Xs"], tr["sev"]
    if sim is not None and sw > 0:
        X, y = np.concatenate([X, sim["Xc"]]), np.concatenate([y, sim["y"]])
        Xs, sev = np.concatenate([Xs, sim["Xs"]]), np.concatenate([sev, sim["sev"]])
        w = np.concatenate([np.ones(len(tr["y"])), np.full(len(sim["y"]), sw)])
    clf = Clf(hp, a.threads).fit(X, y, w)
    zm = fit_sev(Xs, sev, w, a.threads)
    # ---- score ----
    res = {"heldout": None}
    if ho is not None:
        res["heldout"] = evaluate(ho, clf.proba(ho["Xc"]), zm.predict(ho["Xs"]))
    cvsel = next(r for r in cv_rows if r["hp"] == hp and r["sim_weight"] == sw)
    # real-only vs real+sim comparison on the held-out set (reported, NOT used for the choice)
    cmp = None
    if sim is not None and ho is not None:
        alt_sw = 0.0 if sw > 0 else a.sim_weight
        Xa, ya = (tr["Xc"], tr["y"]) if alt_sw == 0 else (np.concatenate([tr["Xc"], sim["Xc"]]), np.concatenate([tr["y"], sim["y"]]))
        wa = None if alt_sw == 0 else np.concatenate([np.ones(len(tr["y"])), np.full(len(sim["y"]), alt_sw)])
        ca = Clf(hp, a.threads).fit(Xa, ya, wa)
        cmp = {"selected_by_cv": "real+sim" if sw > 0 else "real-only",
               "heldout_macro_f1_real_only": macro_f1(ho["y"], (clf if sw == 0 else ca).proba(ho["Xc"]).argmax(1)),
               "heldout_macro_f1_real_plus_sim": macro_f1(ho["y"], (clf if sw > 0 else ca).proba(ho["Xc"]).argmax(1)),
               "note": "informational; the choice was made by grouped CV only"}
    # ---- export ----
    gu = None
    cand = rt[~rt["is_baseline"] & (rt["label"] != "healthy") & rt["rpm_key"].isin(baselines.keys())]
    if not cand.empty:
        g = df[df["run_id"] == cand.iloc[0]["run_id"]].sort_values("order").iloc[R.settle_windows(cand.iloc[0]["rpm_key"], a.settle_s):]
        gu = pd.DataFrame({"unit_id": cand.iloc[0]["run_id"], "window_index": np.arange(len(g)), "rpm": g["rpm"].to_numpy(),
                           "t_hours": np.arange(len(g)) * R.window_seconds(cand.iloc[0]["rpm_key"]) / 3600.0})
        from .config import BEARING_FEATURES
        for i, c in enumerate(BEARING_FEATURES):
            gu[c] = g[R.FEATURE_COLUMNS[i]].to_numpy()
    prov = {"synthetic": bool(sim is not None and sw > 0), "tag": a.tag, "datasets": [str(a.data)] + (["synth_coe_node (train, weighted)"] if sw > 0 else []),
            "note": f"{a.tag} classifier + health-level head fitted on labelled recorder runs; RUL not trained; selected by grouped CV",
            "seed": SEED, "command": "python -m aiengine.train_real " + " ".join(sys.argv[1:]), "heldout_runs": held_runs,
            "hyper_params": hp, "sim_weight": sw}
    dsum = {"runs": int(len(rt)), "windows": int(len(df)), "runs_per_class": runs_per_label, "mounts_per_class": mounts_per_label,
            "baseline_rpms": {str(k): v["source"] for k, v in baselines.items()}}
    paths = export(a.out, clf, zm, baselines, {"data_summary": dsum}, models_dir, a.settle_s, a.baseline_s, gu, prov)
    pf = None
    ex_eval = ho if ho is not None else tr
    from . import coe_adapt as A
    # parity on engineered blocks of the first run of the eval set (rebuild blocks quickly)
    r0 = rt[~rt["is_baseline"] & rt["rpm_key"].isin(baselines.keys())].iloc[0]
    g0 = df[df["run_id"] == r0["run_id"]].sort_values("order").iloc[R.settle_windows(r0["rpm_key"], a.settle_s):]
    eng0 = feat.engineer_arrays(g0[R.FEATURE_COLUMNS[:16]].to_numpy(float), g0["rpm"].to_numpy(float), np.asarray(baselines[int(r0["rpm_key"])]["b1"]), True)[:, EXT_IDX]
    B0, f0 = uw.blocks(eng0, T)
    pf = onnx_parity(models_dir, a.out, clf, B0[f0][:200])
    sc = stream_check(models_dir, a.out, baselines, df, rt, a.settle_s)
    # ---- spec scorecard ----
    H = res["heldout"]
    score = {}
    if H:
        score["IS3 macro-F1 >= 0.85 (held-out real runs)"] = {"value": H["macro_f1"], "pass": H["macro_f1"] >= 0.85}
        if "stage" in H:
            score["IS3 stage error <= 1 (max over held-out windows; severity->stage mapping assumed)"] = {
                "value": H["stage"]["max_error"], "frac_within_1": H["stage"]["frac_error_le_1"], "pass": H["stage"]["max_error"] <= 1}
        if "detection" in H:
            d = H["detection"]
            score["IS3 fault caught by stage 3 (level >= 1 fault windows classified non-healthy)"] = {
                "value": d["fault_window_recall"], "runs_caught": f"{d['runs_caught_ge_50pct']}/{d['runs']}",
                "pass": d["runs_caught_ge_50pct"] == d["runs"]}
    else:
        score["IS3 (NOT SCORED)"] = {"value": None, "pass": None, "why": "no held-out run: need >= 3 runs/mounts per condition"}
    if sc.get("latency_ms"):
        score["S8 latency < 200 ms/window (NodeEngine host p95)"] = {"value": sc["latency_ms"]["p95"], "pass": sc["latency_ms"]["p95"] < 200}
    report = {"tag": a.tag, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED, "command": prov["command"],
              "data": dsum, "notes": notes, "baselines": {k: {x: v[x] for x in ("windows", "settle_windows", "source")} for k, v in baselines.items()},
              "split": {"registered_in": str(sp_path), "section": SPLIT_SECTION, "counts": sec["counts"], "heldout_runs": held_runs},
              "cv": cv_rows, "selected": {"hp": hp, "sim_weight": sw, "cv_macro_f1": cvsel["macro_f1"]}, "sim_comparison": cmp,
              "heldout": H, "scorecard": score, "onnx_parity": pf, "stream_check": sc, "export": paths,
              "train_windows": int((~is_held).sum()), "heldout_windows": int(is_held.sum()),
              "seconds": round(time.perf_counter() - t0, 1)}
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (reports_dir / f"train_real_{ts}.json").write_text(json.dumps(report, indent=1, default=float))
    (reports_dir / f"train_real_{ts}.md").write_text(render_md(report))
    print(render_md(report))
    print(f"\nLive use:  set NODE_CONTRACT={a.out}  NODE_BASELINE={paths['baseline']}   (then the usual node_live / receiver hook)")
    print(f"Replay:    python -m aiengine.node_eval <csv> --contract {a.out}")
    return 0


def render_md(r: dict) -> str:
    L = [f"# train_real report {r['tag']}", "", f"- utc {r['utc']}, seed {r['seed']}, {r['seconds']} s", f"- command: `{r['command']}`",
         f"- data: {r['data']['runs']} runs, {r['data']['windows']} windows; runs per class {r['data']['runs_per_class']}; mounts per class {r['data']['mounts_per_class']}",
         f"- train windows {r['train_windows']}, held-out windows {r['heldout_windows']}; held-out runs: {r['split']['heldout_runs'] or 'NONE'}",
         f"- baselines: {r['baselines']}", ""]
    if r["notes"]:
        L += ["## Warnings", *[f"- {n}" for n in r["notes"]], ""]
    L += ["## Model selection (grouped CV on training runs only)", *[f"- {c['hp']} sim_weight {c['sim_weight']}: macro-F1 {c['macro_f1']}" for c in r["cv"]],
          f"- selected: {r['selected']}", ""]
    if r["sim_comparison"]:
        L += [f"- real-only vs real+sim on held-out (informational): {r['sim_comparison']}", ""]
    H = r["heldout"]
    if H:
        L += [f"## Held-out real runs {r['tag']}", f"- macro-F1 {H['macro_f1']:.3f} over {H['classes_in_truth']} ({H['windows']} windows); healthy false alarm {H['healthy_false_alarm_rate']}",
              "- per class: " + ", ".join(f"{c} F1 {v['f1']:.2f} (n={v['windows']})" for c, v in H["per_class"].items()),
              f"- confusion (rows true, cols pred) {H['confusion']['labels']}: {H['confusion']['rows_true_cols_pred']}",
              f"- per rpm: {H['per_rpm']}"]
        if "stage" in H:
            L.append(f"- stage: {H['stage']}")
        if "detection" in H:
            L.append(f"- detection: {H['detection']}")
        L.append("")
    else:
        L += ["## Held-out: NONE (too few runs). Numbers above are CV only; do NOT quote as test results.", ""]
    L += ["## Spec scorecard", *[f"- {k}: {v}" for k, v in r["scorecard"].items()], "",
          f"## Export / checks", f"- {r['export']}", f"- ONNX vs native parity: {r['onnx_parity']}", f"- NodeEngine stream check: {r['stream_check']}", "",
          "The exported RUL output is a constant 0: this contract does not estimate RUL."]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
