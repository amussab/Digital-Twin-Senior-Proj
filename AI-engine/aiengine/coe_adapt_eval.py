"""COE domain-gap study for the classifier: baseline (deployed v3b) vs adapted, on the COE-derived dataset.

ALL DATA IS [SIMULATION]/[SYNTHETIC]: raw samples from the COE rig simulator through the real CoePipeline
(aiengine/coe_dataset.py) and synth_v3 payload-level windows. Nothing here is a rig or hardware measurement.

    python -m aiengine.coe_adapt_eval data      # generate synth_coe (parallel) + register the run-level split
    python -m aiengine.coe_adapt_eval select    # candidates fitted on COE-train (+ synth_v3 train), chosen on COE-val
    python -m aiengine.coe_adapt_eval test      # the ONE scoring pass on COE-test (baseline + selected), refuses twice
    python -m aiengine.coe_adapt_eval export    # v3b_coe_cls.onnx + model_contract_v3b_coe.json + golden vectors
    python -m aiengine.coe_adapt_eval           # (= report) re-score the saved predictions -> reports/coe_adapt_<ts>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import coe_adapt as A
from . import coe_dataset as D
from . import splits, up2
from .config import CHECKPOINT_DIR, MODELS_DIR, REPORTS_DIR, SEED

RUN_DIR = CHECKPOINT_DIR / "coe_adapt"
PRED_DIR = REPORTS_DIR / "up2_preds"
BASE_CONTRACT, BASE_CLS = "model_contract_v3b.json", "v3b_cls.onnx"
NEW_CLS, NEW_CONTRACT, NEW_GOLDEN = "v3b_coe_cls.onnx", "model_contract_v3b_coe.json", "golden_vectors_v3b_coe.json"
MIX_WEIGHTS = (0.3, 0.6)
GATE_TAUS = (0.0, 0.3, 0.45, 0.6, 0.75)
GATE_SIGNALS = ("family", "env")


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def meta(extra: dict | None = None) -> dict:
    from .train import git_commit, git_dirty
    return {"provenance": "[SIMULATION] COE simulator -> real CoePipeline (synth_coe) + [SYNTHETIC] synth_v3",
            "seed": SEED, "git_commit": git_commit(), "git_dirty": git_dirty(),
            "command": "python -m aiengine.coe_adapt_eval " + " ".join(sys.argv[1:]),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}


# --------------------------------------------------------------------------- data

def load_coe() -> pd.DataFrame:
    df = D.generate()
    sp = splits.load()["sections"].get(D.DATASET)
    if sp is None:
        raise SystemExit("synth_coe split not registered: run `python -m aiengine.coe_adapt_eval data` first")
    m = df["unit_id"].map(sp["units"])
    if m.isna().any() or (m != df["split"]).any():
        raise SystemExit("synth_coe table disagrees with the frozen splits.json section")
    prov = df.attrs.get("provenance", {})
    eng = A.engineer(df)
    eng.attrs["provenance"] = prov
    return eng


def load_synth(which=("train", "val"), stride: int = 2) -> pd.DataFrame:
    """synth_v3 (+ train2) with the frozen split; training windows thinned by `stride` AFTER block building."""
    from .up2_train import load_main
    df = load_main()
    df = df[df["split"].isin(which)].copy()
    df["bearing"] = df["unit_id"].str[-1].astype(int)
    return df


def cmd_data(a) -> int:
    df = D.generate(workers=a.workers, force=False)
    D.register_split(df, note="UP2 COE domain-gap study (coe_adapt): run-level split registered BEFORE any fit or "
                              "scoring; test = scored once by `coe_adapt_eval test`")
    rep = {**df.attrs.get("provenance", {}),
           "runs_per_split": df.groupby("split")["bearing_key"].nunique().to_dict(),
           "bearing_windows_per_split": df.groupby("split").size().to_dict(),
           "class_windows": df[~df["transition"]].groupby(["split", "observable_class"]).size().unstack(0).to_dict()}
    print(json.dumps(rep, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------- candidates

def _blocks(df, stride: int = 1):
    return A.blocks_for(df, stride=stride)


def _healthy_rows(df):
    return df[(df["observable_class"] == "healthy") & ~df.get("transition", pd.Series(False, index=df.index)).astype(bool)][A.COLS].to_numpy(np.float64)


def fit_mix(Bs_tr, ys_tr, Bc_tr, yc_tr, Bc_va, yc_va, w: float, threads: int):
    import xgboost as xgb
    Xs, Xc, Xv = (up2.uw.compute(b, with_decoder=True) for b in (Bs_tr, Bc_tr, Bc_va))
    X = np.concatenate([Xs, Xc])
    y = np.concatenate([ys_tr, yc_tr])
    cw_s, cw_c = up2.class_weights(ys_tr, 0.5), up2.class_weights(yc_tr, 0.5)
    wt = np.concatenate([cw_s * (1 - w) / cw_s.sum(), cw_c * w / cw_c.sum()])
    wt = wt / wt.mean()
    t0 = time.perf_counter()
    m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.6,
                          min_child_weight=5, reg_lambda=1.0, tree_method="hist", objective="multi:softprob",
                          eval_metric="mlogloss", early_stopping_rounds=40, random_state=SEED, n_jobs=threads)
    m.fit(X, y, sample_weight=wt, eval_set=[(Xv, yc_va)], sample_weight_eval_set=[up2.class_weights(yc_va, 0.5)],
          verbose=False)
    return m, {"best_iteration": int(m.best_iteration) + 1, "fit_seconds": round(time.perf_counter() - t0, 1),
               "rows_synth": int(len(Xs)), "rows_coe": int(len(Xc)), "coe_weight_share": w}


def labels_of(idx, df):
    t = up2.merge_truth(idx, df, ["observable_class", "transition"] if "transition" in df else ["observable_class"])
    y = t["observable_class"].map({c: i for i, c in enumerate(up2.CLASS_ORDER)}).to_numpy()
    keep = ~t["transition"].astype(bool).to_numpy() if "transition" in t else np.ones(len(t), bool)
    return y, keep


def cmd_select(a) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    coe = load_coe()
    ctr, cva = coe[coe["split"] == "train"], coe[coe["split"] == "val"]
    print("loading synth_v3 ...", flush=True)
    syn = load_synth()
    s_tr, s_va = syn[syn["split"] == "train"], syn[syn["split"] == "val"]
    Bc_tr, ic_tr = _blocks(ctr)
    Bc_va, ic_va = _blocks(cva)
    Bs_va, is_va = _blocks(s_va)
    base_onnx = MODELS_DIR / BASE_CLS
    cands = {}

    # calibration maps fitted on COE-train vs synth_v3-train only
    syn_rows_h = _healthy_rows(s_tr)
    syn_rows = s_tr[A.COLS].to_numpy(np.float64)
    coe_rows_tr = ctr[~ctr["transition"]][A.COLS].to_numpy(np.float64)
    aff = A.fit_affine(_healthy_rows(ctr), syn_rows_h)
    qm = A.fit_quantile(coe_rows_tr, syn_rows)
    maps = {"affine": aff, "quantile": qm}
    (RUN_DIR / "maps.json").write_text(json.dumps(maps, indent=1))

    def score_val(name, Pc, Ps):
        pc = A.frame(ic_va, Pc)
        ps = A.frame(is_va, Ps)
        rows = []
        for sig in GATE_SIGNALS:
            for tau in GATE_TAUS:
                if tau == 0 and sig != GATE_SIGNALS[0]:
                    continue
                gc = A.gate(pc, cva, tau, sig)
                gs = A.gate(ps, s_va, tau, sig)
                mc = A.coe_metrics(gc, cva)
                from . import up2_metrics as M
                ms = M.classification(up2.merge_truth(gs, s_va, ["observable_class"]))
                rows.append({"candidate": name, "gate_tau": tau, "gate_signal": sig if tau > 0 else None,
                             "coe_val": A.short(mc), "synth_val_macro_f1": round(ms["macro_f1"], 4),
                             "eligible": ms["macro_f1"] >= A.SYNTH_FLOOR})
                print(json.dumps({k: rows[-1][k] for k in ("candidate", "gate_tau", "gate_signal", "synth_val_macro_f1")}),
                      "coe_val F1 %.4f b2_tff %s" % (mc["classification"]["macro_f1"], mc["b2_transmitted_false_fault_rate"]),
                      flush=True)
        return rows

    grid = []
    grid += score_val("base", A.onnx_probs(base_onnx, Bc_va), A.onnx_probs(base_onnx, Bs_va))
    grid += score_val("affine", A.onnx_probs(base_onnx, A.apply_affine(Bc_va, aff)), A.onnx_probs(base_onnx, Bs_va))
    grid += score_val("quantile", A.onnx_probs(base_onnx, A.apply_quantile(Bc_va, qm)), A.onnx_probs(base_onnx, Bs_va))

    # retrained mixtures
    Bs_tr, is_tr = _blocks(s_tr, a.stride)
    ys_tr, _ = labels_of(is_tr, s_tr)
    yc_tr, kc_tr = labels_of(ic_tr, ctr)
    yc_va, kc_va = labels_of(ic_va, cva)
    print(f"mix rows: synth {len(Bs_tr)}, coe {int(kc_tr.sum())}", flush=True)
    for w in MIX_WEIGHTS:
        m, info = fit_mix(Bs_tr, ys_tr, Bc_tr[kc_tr], yc_tr[kc_tr], Bc_va[kc_va], yc_va[kc_va], w, a.threads)
        name = f"mix_w{w:g}"
        with open(RUN_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump({"model": m, "info": info}, f)
        print(name, json.dumps(info), flush=True)
        rows = score_val(name, A.tree_probs(m, Bc_va), A.tree_probs(m, Bs_va))
        for r in rows:
            r["fit"] = info
        grid += rows
        del m

    elig = [r for r in grid if r["eligible"]]
    best = max(r["coe_val"]["macro_f1"] for r in elig)
    tied = [r for r in elig if best - r["coe_val"]["macro_f1"] < 0.002]
    sel = sorted(tied, key=lambda r: (r["coe_val"]["transmitted_false_fault_rate"], -r["coe_val"]["macro_f1"]))[0]
    sel = {"candidate": sel["candidate"], "gate_tau": sel["gate_tau"], "gate_signal": sel["gate_signal"],
           "coe_val": sel["coe_val"], "synth_val_macro_f1": sel["synth_val_macro_f1"]}
    out = {"meta": meta({"selection": "COE-val only (+ synth_v3-val floor); COE-test untouched",
                         "rule": f"max COE-val macro-F1; eligible iff synth_v3-val macro-F1 >= {A.SYNTH_FLOOR}; "
                                 "tie (<0.002) -> lower transmitted false-fault rate",
                         "coe_provenance": coe.attrs.get("provenance", {})}),
           "grid": grid, "selected": sel}
    path = REPORTS_DIR / f"model_selection_coe_adapt_{now_tag()}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    (RUN_DIR / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selected", json.dumps(sel), "->", path, flush=True)
    return 0


# --------------------------------------------------------------------------- the single test pass

def selected_probs(sel: dict, B: np.ndarray) -> np.ndarray:
    name = sel["candidate"]
    maps = json.loads((RUN_DIR / "maps.json").read_text())
    if name == "base":
        return A.onnx_probs(MODELS_DIR / BASE_CLS, B)
    if name == "affine":
        return A.onnx_probs(MODELS_DIR / BASE_CLS, A.apply_affine(B, maps["affine"]))
    if name == "quantile":
        return A.onnx_probs(MODELS_DIR / BASE_CLS, A.apply_quantile(B, maps["quantile"]))
    with open(RUN_DIR / f"{name}.pkl", "rb") as f:
        return A.tree_probs(pickle.load(f)["model"], B)


def rul_diag(df: pd.DataFrame, rul_onnx: Path) -> pd.DataFrame:
    """v3b N-HiTS-slot head (unchanged) on the COE rows: raw stage + z per window (diagnostic only)."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(rul_onnx), providers=["CPUExecutionProvider"])
    outs = []
    for uid, g in df.groupby("unit_id", sort=True):
        g = g.sort_values("window_index")
        b, _ = up2.uw.blocks(g[A.COLS].to_numpy(np.float64), 48)
        keep = np.arange(len(g)) >= A.T_CLS - 1
        o = s.run(["rul_head"], {"encoder_cont": b[keep].astype(np.float32)})[0]
        f = g.loc[keep, ["unit_id", "window_index"]].copy()
        f["rul_windows_raw"], f["z_raw"], f["stage_raw"] = o[:, 0], o[:, 1], o[:, 2].astype(int)
        outs.append(f)
    return pd.concat(outs, ignore_index=True)


def cmd_test(a) -> int:
    prev = sorted(PRED_DIR.glob("coe_adapt_test_*.index.json"))
    if prev:
        raise SystemExit(f"COE-test was already scored ({prev[-1].name}); re-score with `python -m aiengine.coe_adapt_eval`")
    sel = json.loads((RUN_DIR / "selection.json").read_text())
    coe = load_coe()
    te = coe[coe["split"] == "test"]
    B, idx = A.blocks_for(te)
    tag = now_tag()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    base = A.frame(idx, A.onnx_probs(MODELS_DIR / BASE_CLS, B))
    adapted = A.frame(idx, selected_probs(sel, B))
    for name, p in (("base", base), ("selected_nogate", adapted)):
        f = PRED_DIR / f"coe_adapt_test_{name}_{tag}.cls.csv.gz"
        p.to_csv(f, index=False, float_format="%.6g")
        files[name] = f.name
    diag = rul_diag(te, MODELS_DIR / "v3b_rul.onnx")
    f = PRED_DIR / f"coe_adapt_test_rul_diag_{tag}.csv.gz"
    diag.to_csv(f, index=False, float_format="%.6g")
    files["rul_diag_raw_features"] = f.name
    if sel["candidate"] in ("affine", "quantile"):
        maps = json.loads((RUN_DIR / "maps.json").read_text())
        mte = te.copy()
        rows = mte[A.COLS].to_numpy(np.float32)[None]
        mte[A.COLS] = (A.apply_affine(rows, maps["affine"]) if sel["candidate"] == "affine"
                       else A.apply_quantile(rows, maps["quantile"]))[0]
        d2 = rul_diag(mte, MODELS_DIR / "v3b_rul.onnx")
        f = PRED_DIR / f"coe_adapt_test_rul_diag_mapped_{tag}.csv.gz"
        d2.to_csv(f, index=False, float_format="%.6g")
        files["rul_diag_mapped_features"] = f.name
    out = {"meta": meta({"tag": tag, "touch": "synth_coe test runs: FIRST and only scoring pass (baseline + selected)",
                         "selected": sel}), "files": files}
    (PRED_DIR / f"coe_adapt_test_{tag}.index.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return cmd_report(a)


# --------------------------------------------------------------------------- report (re-score saved predictions)

def _rul_summary(diag: pd.DataFrame, te: pd.DataFrame) -> dict:
    d = up2.merge_truth(diag, te, ["observable_class", "transition", "partner_faulty"])
    d = d[~d["transition"].astype(bool)]
    h = d["observable_class"] == "healthy"
    return {"healthy_windows": int(h.sum()), "healthy_stage_raw_ge2_rate": float((d.loc[h, "stage_raw"] >= 2).mean()),
            "faulted_windows": int((~h).sum()),
            "faulted_stage_raw_hist": {str(k): int(v) for k, v in d.loc[~h, "stage_raw"].value_counts().sort_index().items()},
            "note": "COE runs are constant/ramp severity over 40 contiguous windows, not run-to-failure: there is no RUL or "
                    "stage truth, so only the healthy-bearing false-stage rate is a meaningful check"}


def cmd_report(a=None) -> int:
    idxs = sorted(PRED_DIR.glob("coe_adapt_test_*.index.json"))
    if not idxs:
        raise SystemExit("no COE-test predictions yet: run `python -m aiengine.coe_adapt_eval test` (once)")
    ix = json.loads(idxs[-1].read_text())
    sel = ix["meta"]["selected"]
    coe = load_coe()
    te = coe[coe["split"] == "test"]
    rd = lambda k: pd.read_csv(PRED_DIR / ix["files"][k])  # noqa: E731
    base, adapted = rd("base"), rd("selected_nogate")
    res = {"base": A.coe_metrics(base, te), "selected_nogate": A.coe_metrics(adapted, te)}
    if sel.get("gate_tau", 0) > 0:
        res["selected"] = A.coe_metrics(A.gate(adapted, te, sel["gate_tau"], sel["gate_signal"]), te)
        res["base_plus_gate"] = A.coe_metrics(A.gate(base, te, sel["gate_tau"], sel["gate_signal"]), te)
    else:
        res["selected"] = res["selected_nogate"]
    rul = {"raw_features": _rul_summary(rd("rul_diag_raw_features"), te)}
    if "rul_diag_mapped_features" in ix["files"]:
        rul["mapped_features"] = _rul_summary(rd("rul_diag_mapped_features"), te)
    sels = sorted(REPORTS_DIR.glob("model_selection_coe_adapt_*.json"))
    selection = json.loads(sels[-1].read_text()) if sels else {}
    con_p = MODELS_DIR / NEW_CONTRACT
    tag = now_tag()
    out = {"meta": meta({"tag": tag, "test_predictions": ix["files"], "test_scored_utc": ix["meta"]["utc"],
                         "selection_file": sels[-1].name if sels else None,
                         "coe_provenance": coe.attrs.get("provenance", {})}),
           "selected": sel, "test": res, "rul_stage_diagnostic": rul,
           "val_grid_short": [{k: r[k] for k in ("candidate", "gate_tau", "gate_signal", "synth_val_macro_f1", "eligible")}
                              | {"coe_val_macro_f1": r["coe_val"]["macro_f1"],
                                 "coe_val_b2_tff": r["coe_val"]["b2_transmitted_false_fault_rate"]}
                              for r in selection.get("grid", [])],
           "contract": json.loads(con_p.read_text()).get("coe_adaptation") if con_p.exists() else None}
    jp = REPORTS_DIR / f"coe_adapt_{tag}.json"
    jp.write_text(json.dumps(out, indent=1, default=float))
    (REPORTS_DIR / f"coe_adapt_{tag}.md").write_text(markdown(out))
    print(markdown(out))
    print("->", jp)
    return 0


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f} %"


def markdown(o: dict) -> str:
    t, s = o["test"], o["selected"]
    m = o["meta"]
    cp = m.get("coe_provenance", {})
    L = [f"# COE domain gap: deployed v3b classifier vs COE-adapted classifier",
         "",
         f"Generated {m['utc']} | commit {m['git_commit']}{' (dirty)' if m['git_dirty'] else ''} | seed {m['seed']} | "
         f"command `python -m aiengine.coe_adapt_eval`",
         "",
         "**ALL DATA [SIMULATION]/[SYNTHETIC].** Test windows = raw samples from the COE rig simulator "
         "(aiengine/coe_dataset.py DualFaultRigSource; invented amplitudes, not rig-calibrated) through the real causal "
         "CoePipeline (C1-C9). Training data = synth_v3 [SYNTHETIC] + COE-train [SIMULATION]. Nothing here validates "
         "anything on real hardware; it only shows the models now accept the features the COE pipeline produces.",
         "",
         f"Dataset synth_coe: {cp.get('n_runs')} runs, {cp.get('rig_windows')} rig windows "
         f"({cp.get('rows')} bearing-windows), split BY RUN in splits.json section `synth_coe` (registered before any fit), "
         f"test scored once ({m['test_scored_utc']}). Each run: {cp.get('healthy_windows')} healthy (commissioning) windows "
         f"then {cp.get('fault_windows')} with defect(s) on bearing 1, bearing 2, both or neither; severities "
         f"{cp.get('severities')}; rpm 1750/3600 + off-nominal 1738/3624 (+-1 % band) + 2400/3000 (range policy).",
         "",
         f"Selected on COE-val: **{s['candidate']}**" + (f" + cross-bearing gate (signal {s['gate_signal']}, tau {s['gate_tau']})"
                                                         if s.get("gate_tau", 0) > 0 else " (no gate)") +
         f"; COE-val macro-F1 {s['coe_val']['macro_f1']}, synth_v3-val macro-F1 {s['synth_val_macro_f1']}.",
         "",
         "## COE-test (scored once)",
         "",
         "| Metric | v3b as deployed | " + " | ".join(k for k in ("base_plus_gate", "selected_nogate", "selected") if k in t) + " |",
         "|---|---|" + "---|" * len([k for k in ("base_plus_gate", "selected_nogate", "selected") if k in t])]
    keys = ["base"] + [k for k in ("base_plus_gate", "selected_nogate", "selected") if k in t]
    rows = [("macro-F1 (5 classes)", lambda r: f"{r['classification']['macro_f1']:.4f}"),
            ("accuracy", lambda r: f"{r['classification']['accuracy']:.4f}"),
            ("bearing-2 false-fault rate (B2 healthy, B1 faulted)", lambda r: _pct(r["b2_transmitted_false_fault_rate"])),
            ("bearing-1 false-fault rate (B1 healthy, B2 faulted)", lambda r: _pct(r["b1_transmitted_false_fault_rate"])),
            ("false-fault rate, both bearings healthy", lambda r: _pct(r["clean_false_fault_rate"])),
            ("missed-fault rate (faulted -> healthy)", lambda r: _pct(r["missed_fault_rate"])),
            ("wrong fault class (faulted -> other fault)", lambda r: _pct(r["wrong_class_rate"]))]
    for c in ["healthy", "outer_race", "inner_race", "ball", "cage"]:
        rows.append((f"F1 {c}", lambda r, c=c: f"{r['classification']['per_class_f1'][c]:.4f}"))
    for name, f in rows:
        L.append(f"| {name} | " + " | ".join(f(t[k]) for k in keys) + " |")
    L += ["", "Columns: `base_plus_gate` = deployed trees + the selected gate; `selected_nogate` = selected candidate "
          "before the gate (ablation). Rates are per bearing-window; the transition window of each run is excluded.", ""]
    L += ["## Per-bearing role confusion (COE-test)", ""]
    for k in ("base", "selected"):
        L.append(f"**{k}**")
        L.append("")
        L.append("| bearing | role | pred healthy | pred correct fault | pred other fault |")
        L.append("|---|---|---|---|---|")
        for b, roles in t[k]["per_bearing_roles"].items():
            for role, d in roles.items():
                L.append(f"| B{b} | {role} | {d.get('pred_healthy', 0)} | {d.get('pred_correct_fault', 0)} | {d.get('pred_other_fault', 0)} |")
        L.append("")
    L += ["## Slices (COE-test)", "", "| slice | v3b | selected |", "|---|---|---|"]
    for r in t["base"]["macro_f1_by_rpm"]:
        L.append(f"| macro-F1 @ {r} rpm | {t['base']['macro_f1_by_rpm'][r]:.4f} | {t['selected']['macro_f1_by_rpm'].get(r, float('nan')):.4f} |")
    for sv in t["base"]["recall_by_severity"]:
        L.append(f"| fault recall @ severity {sv} | {t['base']['recall_by_severity'][sv]:.3f} | "
                 f"{t['selected']['recall_by_severity'].get(sv, float('nan')):.3f} |")
    L += ["", "## RUL / stage heads (unchanged v3b N-HiTS slot) on COE windows: diagnostic only", ""]
    for k, d in o["rul_stage_diagnostic"].items():
        L.append(f"- {k}: healthy-bearing raw stage >= 2 rate {_pct(d['healthy_stage_raw_ge2_rate'])} "
                 f"({d['healthy_windows']} windows); faulted-bearing raw stage histogram {d['faulted_stage_raw_hist']}.")
    L.append("- " + next(iter(o["rul_stage_diagnostic"].values()))["note"] + ". RUL hours additionally depend on the "
             "window cadence (see aiengine/coe/README.md, 'COE time base').")
    L += ["", "## Validation grid (COE-val; selection only)", "",
          "| candidate | gate | synth_v3-val F1 | eligible | COE-val F1 | COE-val B2 false-fault |", "|---|---|---|---|---|---|"]
    for r in o["val_grid_short"]:
        g = "off" if not r["gate_tau"] else f"{r['gate_signal']} tau {r['gate_tau']}"
        L.append(f"| {r['candidate']} | {g} | {r['synth_val_macro_f1']} | {r['eligible']} | {r['coe_val_macro_f1']} | "
                 f"{_pct(r['coe_val_b2_tff'])} |")
    if o.get("contract"):
        L += ["", "## Deployed artefact", "", "```", json.dumps(o["contract"], indent=1), "```"]
    L += ["", "## Caveats", "",
          "- [SIMULATION] the simulator's transmission (off-plane gain 0.25-0.45 of the partner's impacts), resonance, "
          "noise and amplitudes are invented; the real rig's transmission path and noise floor are unknown until hardware "
          "data exists. Re-fit the calibration / gate threshold on commissioning + seeded-fault data from the rig.",
          "- A per-bearing classifier cannot separate 'small own defect' from 'partner defect transmitted' from its own "
          "16 features; the gate uses the partner bearing of the same window for that.",
          "- RUL/stage heads were not re-fitted: COE simulator runs are not run-to-failure (no truth for RUL)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.coe_adapt_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("data")
    d.add_argument("--workers", type=int, default=10)
    s = sub.add_parser("select")
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--stride", type=int, default=2, help="thin synth_v3 training windows (memory/time)")
    sub.add_parser("test")
    sub.add_parser("export")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        from .coe_adapt_export import run as export_run
        print(json.dumps(export_run(), indent=1, default=str))
        return 0
    return {"data": cmd_data, "select": cmd_select, "test": cmd_test, "report": cmd_report, None: cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
