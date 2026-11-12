"""Contract v3e_node: baseline-relative classifier trained ONLY on the node-matched sim (synth_coe_node) [SIMULATION].

Why: v3d_node used absolute features (kurtosis_max, crest_max, raw_*_share) that are rig/sensor specific, so a healthy
but naturally impulsive channel looked faulty. v3e_node uses only features relative to the unit's own 24-window
commissioning baseline (features.V3E_COLUMNS; chosen on that principle, not from any rig result).

    python -m aiengine.node_v3e httail     # build the heavy-tail VAL variant (sim, val runs only) [robustness proxy]
    python -m aiengine.node_v3e select     # hyper-params + gate tau on val only (+ v3d_node reference on val / HT-val)
    python -m aiengine.node_v3e test       # ONE scoring pass on synth_coe_node test (refuses twice)
    python -m aiengine.node_v3e export     # v3e_node_cls.onnx + model_contract_v3e_node.json + golden vectors

Selection rule (pre-declared): max val macro-F1 (5 classes, per bearing-window, transition windows excluded);
tie (|dF1| < 0.002) -> higher healthy specificity on the heavy-tail val variant. No synth_v3 data (old bearing orders).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import coe_adapt as A
from . import coe_dataset as D
from . import features as feat
from . import splits, up2
from . import up2_metrics as M
from .config import BEARING_FEATURES, CHECKPOINT_DIR, MODELS_DIR, REPORTS_DIR, SEED

COLS = list(feat.V3E_COLUMNS)
T_CLS = A.T_CLS
RUN_DIR = CHECKPOINT_DIR / "node_v3e"
PRED_DIR = REPORTS_DIR / "up2_preds"
HT_PATH = D.CACHE_DIR / "synth_coe_node_httail_val.parquet"
HT_CHANNELS = (1, 3)          # B1Y, B2Y (B2X is a copy of B1Y in the node-matched sim)
HT_DF = 3.5                   # Student-t df of the heavy channels (run's own df, 4.5-8, on the others)
GATE_TAUS = (0.0, 0.3, 0.45, 0.6, 0.75, 1.0, 1.5)
GATE_SIGNALS = ("family", "env")
GRID = [dict(max_depth=d, learning_rate=lr, min_child_weight=5, colsample_bytree=0.6) for d in (4, 6) for lr in (0.05, 0.08)]
CONTRACT, ONNX, GOLDEN = "model_contract_v3e_node.json", "v3e_node_cls.onnx", "golden_vectors_v3e_node.json"
OLD_CONTRACT = "model_contract_v3d_node.json"


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def meta(extra=None) -> dict:
    from .train import git_commit, git_dirty
    return {"provenance": "[SIMULATION] node-matched simulator synth_coe_node (real CoePipeline); no rig data",
            "seed": SEED, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "command": "python -m aiengine.node_v3e " + " ".join(sys.argv[1:]),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}


# --------------------------------------------------------------------------- data

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = feat.engineer_table(df, extended=True)
    out["observable_class"] = out["observable_class"].astype(str)
    return out


def load_node() -> pd.DataFrame:
    df = D.generate()
    sp = splits.load()["sections"].get(D.DATASET)
    if sp is None or (df["unit_id"].map(sp["units"]) != df["split"]).any():
        raise SystemExit("synth_coe_node split missing or disagrees with splits.json")
    return engineer(df)


class HeavyTailSource(D.DualFaultRigSource):
    """Same simulator; channels HT_CHANNELS get Student-t(HT_DF) noise (unit variance, heavier tails)."""

    def noise(self, rng, nb: int) -> np.ndarray:
        ch = getattr(self, "_nc", 0) % 4
        self._nc = getattr(self, "_nc", 0) + 1
        df = HT_DF if ch in HT_CHANNELS else self.NOISE_T_DF
        return self.NOISE_MS2 * rng.standard_t(df, nb) / np.sqrt(df / (df - 2.0))


def _ht_worker(spec_dict: dict) -> dict:
    D.DualFaultRigSource = HeavyTailSource          # simulate_run looks the class up in the module namespace
    return D._worker(spec_dict)


def cmd_httail(a) -> int:
    specs = [s for s in D.plan() if s.split == "val"]
    t0 = time.perf_counter()
    frames = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_ht_worker, [s.to_dict() for s in specs]):
            frames.append(D.to_table(res))
    df = pd.concat(frames, ignore_index=True).sort_values(["unit_id", "window_index"]).reset_index(drop=True)
    df.to_parquet(HT_PATH, index=False)
    h = df[(df["observable_class"] == "healthy") & (df["fault_class"] == "healthy")]
    base = D.generate()
    base = base[base["split"] == "val"]
    bh = base[(base["observable_class"] == "healthy") & (base["fault_class"] == "healthy")]
    st = {"runs": len(specs), "seconds": round(time.perf_counter() - t0, 1), "ht_df": HT_DF, "ht_channels": list(HT_CHANNELS)}
    kc, cc = BEARING_FEATURES[9], BEARING_FEATURES[10]        # y_kurtosis, y_crest (B1Y / B2Y axis)
    for nm, d in (("normal_val", bh), ("httail_val", h)):
        st[nm] = {"y_kurtosis_median_b1": float(d[d["bearing"] == 1][kc].median()), "y_crest_median_b1": float(d[d["bearing"] == 1][cc].median())}
    print(json.dumps(st, indent=1))
    return 0


def load_ht() -> pd.DataFrame:
    if not HT_PATH.exists():
        raise SystemExit("run `python -m aiengine.node_v3e httail` first")
    return engineer(pd.read_parquet(HT_PATH))


# --------------------------------------------------------------------------- fit / predict

def _blocks(df, cols):
    return A.blocks_for(df, columns=cols)


def _labels(idx, df):
    t = up2.merge_truth(idx, df, ["observable_class", "transition"])
    y = t["observable_class"].map({c: i for i, c in enumerate(up2.CLASS_ORDER)}).to_numpy()
    return y, ~t["transition"].astype(bool).to_numpy()


def fit(Btr, ytr, Bva, yva, hp, threads):
    import xgboost as xgb
    Xtr, Xva = up2.uw.compute(Btr, with_decoder=True), up2.uw.compute(Bva, with_decoder=True)
    t0 = time.perf_counter()
    m = xgb.XGBClassifier(n_estimators=600, max_depth=hp["max_depth"], learning_rate=hp["learning_rate"], subsample=0.8,
                          colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"], reg_lambda=1.0,
                          tree_method="hist", objective="multi:softprob", eval_metric="mlogloss",
                          early_stopping_rounds=40, random_state=SEED, n_jobs=threads)
    m.fit(Xtr, ytr, sample_weight=up2.class_weights(ytr, 0.5), eval_set=[(Xva, yva)],
          sample_weight_eval_set=[up2.class_weights(yva, 0.5)], verbose=False)
    return m, {**hp, "best_iteration": int(m.best_iteration) + 1, "fit_seconds": round(time.perf_counter() - t0, 1),
               "rows_train": int(len(Xtr))}


def specificity(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Healthy-bearing windows (observable healthy, transition excluded) predicted healthy."""
    p = pred.merge(truth[["unit_id", "window_index", "observable_class", "partner_faulty", "transition", "bearing", "scenario"]],
                   on=["unit_id", "window_index"], how="left")
    p = p[~p["transition"].astype(bool)]
    h = p[p["observable_class"] == "healthy"]
    ok = (h["pred_class"] == "healthy")
    clean = ~h["partner_faulty"].astype(bool)
    return {"healthy_windows": int(len(h)), "specificity_all": float(ok.mean()),
            "specificity_partner_healthy": float(ok[clean].mean()),
            "specificity_b1": float(ok[h["bearing"] == 1].mean()), "specificity_b2": float(ok[h["bearing"] == 2].mean()),
            "specificity_healthy_run": float(ok[h["scenario"] == "none"].mean())}


def evaluate(P, idx, truth, tau, sig) -> dict:
    g = A.gate(A.frame(idx, P), truth, tau, sig)
    m = A.coe_metrics(g, truth)
    return {**A.short(m), **{"spec": specificity(g, truth)}}


def v3d_probs(truth_eng_blocks):
    return A.onnx_probs(MODELS_DIR / "v3d_node_cls.onnx", truth_eng_blocks)


def v3d_eval(df21_eng: pd.DataFrame, tau=0.75, sig="family") -> dict:
    """v3d_node (as deployed: 21-col classifier + gate tau 0.75 family) on the same rows."""
    B, idx = A.blocks_for(df21_eng, columns=A.COLS)
    return evaluate(v3d_probs(B), idx, df21_eng, tau, sig)


def cmd_select(a) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    coe, ht = load_node(), load_ht()
    tr, va = coe[coe["split"] == "train"], coe[coe["split"] == "val"]
    Btr, itr = _blocks(tr, COLS)
    Bva, iva = _blocks(va, COLS)
    Bht, iht = _blocks(ht, COLS)
    ytr, ktr = _labels(itr, tr)
    yva, kva = _labels(iva, va)
    grid = []
    for k, hp in enumerate(GRID):
        m, info = fit(Btr[ktr], ytr[ktr], Bva[kva], yva[kva], hp, a.threads)
        with open(RUN_DIR / f"hp{k}.pkl", "wb") as f:
            pickle.dump({"model": m, "info": info}, f)
        Pv, Ph = A.tree_probs(m, Bva), A.tree_probs(m, Bht)
        for sig in GATE_SIGNALS:
            for tau in GATE_TAUS:
                if tau == 0 and sig != GATE_SIGNALS[0]:
                    continue
                rv, rh = evaluate(Pv, iva, va, tau, sig), evaluate(Ph, iht, ht, tau, sig)
                grid.append({"hp_id": k, "hp": info, "gate_tau": tau, "gate_signal": sig if tau > 0 else None,
                             "val": rv, "val_httail": rh})
                print(f"hp{k} {sig} tau {tau}: val F1 {rv['macro_f1']} spec {rv['spec']['specificity_all']:.3f} | "
                      f"HT spec {rh['spec']['specificity_all']:.3f}", flush=True)
    best = max(r["val"]["macro_f1"] for r in grid)
    tied = [r for r in grid if best - r["val"]["macro_f1"] < 0.002]
    sel = sorted(tied, key=lambda r: -r["val_httail"]["spec"]["specificity_all"])[0]
    selected = {k: sel[k] for k in ("hp_id", "hp", "gate_tau", "gate_signal", "val", "val_httail")}
    # v3d_node reference on val + HT-val (21-col engineering, as deployed)
    ref = {"val": v3d_eval(A.engineer(pd.read_parquet(D.cache_path()).query("split=='val'"))),
           "val_httail": v3d_eval(A.engineer(pd.read_parquet(HT_PATH)))}
    out = {"meta": meta({"rule": __doc__.split("Selection rule")[1].strip(), "columns": COLS, "ht": {"df": HT_DF, "channels": list(HT_CHANNELS)}}),
           "grid": grid, "selected": selected, "v3d_node_reference": ref}
    (REPORTS_DIR / f"model_selection_node_v3e_{now_tag()}.json").write_text(json.dumps(out, indent=1, default=float))
    (RUN_DIR / "selection.json").write_text(json.dumps(selected, indent=1, default=float))
    print("selected", json.dumps({k: selected[k] for k in ("hp_id", "gate_tau", "gate_signal")}), "val F1", selected["val"]["macro_f1"])
    print("v3d_node reference", json.dumps({k: (v["macro_f1"], v["spec"]) for k, v in ref.items()}))
    return 0


def sel_model():
    sel = json.loads((RUN_DIR / "selection.json").read_text())
    with open(RUN_DIR / f"hp{sel['hp_id']}.pkl", "rb") as f:
        return sel, pickle.load(f)["model"]


def cmd_test(a) -> int:
    marker = PRED_DIR / "node_v3e_test_done.json"
    if marker.exists():
        raise SystemExit(f"test already scored once ({marker}); not re-scoring")
    sel, m = sel_model()
    coe = load_node()
    te = coe[coe["split"] == "test"]
    B, idx = _blocks(te, COLS)
    P = A.tree_probs(m, B)
    g = A.gate(A.frame(idx, P), te, sel["gate_tau"], sel["gate_signal"] or "family")
    res = A.coe_metrics(g, te)
    out = {"meta": meta({"touch": "synth_coe_node test runs, first and only scoring"}), "selected": sel,
           "test": {**A.short(res), "spec": specificity(g, te), "macro_f1_by_rpm": res["macro_f1_by_rpm"],
                    "recall_by_severity": res["recall_by_severity"], "n_windows": res["n_windows"]},
           "ist3_target_macro_f1": 0.85}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(out, indent=1, default=float))
    (REPORTS_DIR / f"node_v3e_{now_tag()}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["test"], indent=1, default=float))
    return 0


# --------------------------------------------------------------------------- export

def cmd_export(a) -> int:
    import onnx
    from . import up2_export as X
    sel, m = sel_model()
    R = len(COLS)
    model = X.build_cls(m, T_CLS, R, X.importances_per_column(m, R))
    onnx.save(model, str(MODELS_DIR / ONNX))
    coe = load_node()
    va = coe[coe["split"] == "val"]
    B, _ = _blocks(va, COLS)
    P = A.onnx_probs(MODELS_DIR / ONNX, B)
    Pr = A.tree_probs(m, B)
    par = {"windows": int(len(B)), "prob_max_abs_diff": float(np.abs(P - Pr).max()),
           "argmax_mismatch": int((P.argmax(1) != Pr.argmax(1)).sum()), "reference": "native xgboost predict_proba"}
    con = json.loads((MODELS_DIR / OLD_CONTRACT).read_text())
    from .train import git_commit
    con["contract_version"] = "3e-node"
    con["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con["generated_by"] = "AI-engine/aiengine/node_v3e.py export (copy of model_contract_v3d_node.json, classifier + inputs swapped)"
    con["provenance"]["datasets"] = ["synth_coe_node (node-matched simulator: new bearing, 1800/3600 rpm, g, B2X=B1Y; train/val)"]
    con["provenance"]["coe_adapt"] = {"git_commit": git_commit(), "selection": sel,
                                      "note": "[SIMULATION] baseline-relative classifier trained on synth_coe_node only; hyper-params + "
                                              "gate tau selected on val only; RUL/z/stage heads = v3b unchanged (NOT valid on node features)"}
    fe = con["feature_engineering"]
    con["contract_version"] = "3e-node.1"
    fe["engineered_order"] = list(feat.ENGINEERED_EXT)
    fe["baseline"]["policy"] = {"settle_seconds": 120, "baseline_seconds": 60, "statistic": "median",
                                "keyed_by": "rpm (+-5 %)", "warmup_output": "fault_class 'warming_up'",
                                "legacy": "policy absent => median of first 24 windows"}   # see aiengine/node_adapter.py
    fe["reference_impl"] = "aiengine/features.py::engineer_arrays(extended=True)"
    fe["crest_rise"] = {"definition": "max(max_over_axes(crest) - max_over_axes(baseline crest), 0); baseline = 24-window median "
                                      "(same form as kurtosis_rise); appended as the 22nd engineered column"}
    fe["classifier_columns_v3e"] = {"columns": COLS, "principle": "relative to the unit's own commissioning baseline only",
                                    "removed_absolute": ["kurtosis_max", "crest_max", "raw_ftf_share", "raw_bsf_share",
                                                         "raw_bpfo_share", "raw_bpfi_share"]}
    t = con["models"]["tft"]
    t["onnx"], t["columns"] = ONNX, COLS
    t["scalers"] = {c: {"center": 0.0, "scale": 1.0, "kind": "identity"} for c in COLS}
    t["family"] = "gradient-boosted trees (xgb) in the TFT slot, trained on synth_coe_node train only, baseline-relative inputs"
    t["size_kb"] = round((MODELS_DIR / ONNX).stat().st_size / 1024, 1)
    t["parity"] = par
    con["coe_adaptation"] = {"candidate": "v3e_relative", "hyper_params": sel["hp"], "tensor_names_unchanged": True,
                             "host_side_change": "engineered row gains column crest_rise; classifier reads 16 relative columns; "
                                                 "cross_bearing_gate unchanged in form"}
    if sel["gate_tau"]:
        sig = sel["gate_signal"]
        g = con["cross_bearing_gate"]
        g["tau"], g["signal"] = float(sel["gate_tau"]), sig
        g["family_of_class"] = ({c: f"log_{f}_ratio" for c, f in A.FAMILY_OF.items()} if sig == "family"
                                else {c: "log_env_ratio" for c in A.FAMILY_OF})
        g["selected_on"] = "synth_coe_node val runs only"
    else:
        con.pop("cross_bearing_gate", None)
    con["disclosures"]["classifier"] = ("contract 3e-node: baseline-relative classifier on synth_coe_node only; "
                                        "selected on val; test scored once (reports/node_v3e_*.json)")
    (MODELS_DIR / CONTRACT).write_text(json.dumps(con, indent=2, default=float))

    from .engine import HybridEngine
    eng = HybridEngine(MODELS_DIR, contract_file=CONTRACT)
    te = coe[coe["split"] == "test"]
    rk = te[te["scenario"] == "b1"].groupby("bearing_key").size().index.tolist()[0]
    run_df = te[te["bearing_key"] == rk]
    u1 = run_df[run_df["bearing"] == 1].sort_values("window_index")
    golden = eng.golden_vectors(u1, n=12)
    golden["cases_late"] = eng.golden_cases_at(u1, [int(x) for x in np.linspace(40, len(u1) - 1, 8)])
    eng.reset()
    u2 = run_df[run_df["bearing"] == 2].sort_values("window_index")
    rig_in, rig_out = [], []
    for r1, r2 in zip(u1.itertuples(index=False), u2.itertuples(index=False)):
        f32 = np.concatenate([[getattr(r1, c) for c in BEARING_FEATURES], [getattr(r2, c) for c in BEARING_FEATURES]])
        o = eng.process(float(r1.rpm), f32, int(r1.t20_ms))
        rig_in.append({"window_index": int(r1.window_index), "rpm": float(r1.rpm), "t20_ms": int(r1.t20_ms), "features32": f32.tolist()})
        rig_out.append([{k: v for k, v in b.items() if k in ("bearing", "status", "fault_class", "transmitted_from_partner",
                                                             "class_probs", "health_stage")} for b in o["bearings"]])
    golden["rig_stream"] = {"bearing_key": rk, "scenario": "bearing 1 faulted, bearing 2 healthy (test run)",
                            "inputs": rig_in, "outputs": rig_out}
    golden["provenance"] = "[SIMULATION] synth_coe_node test run"
    (MODELS_DIR / GOLDEN).write_text(json.dumps(golden, indent=1, default=float))
    print(json.dumps({"parity": par, "onnx": ONNX, "contract": CONTRACT, "run": rk}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.node_v3e")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("httail")
    h.add_argument("--workers", type=int, default=12)
    s = sub.add_parser("select")
    s.add_argument("--threads", type=int, default=12)
    sub.add_parser("test")
    sub.add_parser("export")
    a = ap.parse_args(argv)
    return {"httail": cmd_httail, "select": cmd_select, "test": cmd_test, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
