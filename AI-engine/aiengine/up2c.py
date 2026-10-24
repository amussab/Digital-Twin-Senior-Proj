"""UP2 v3c: PHYSICS-INFORMED RUL head (all [SYNTHETIC] synth_v3 generator + [SIMULATION] FE twin).

    python -m aiengine.up2c data       # generate synth_v3_test3 (FRESH) and register it in splits.json BEFORE any v3c fit
    python -m aiengine.up2c physics    # per-window physics features for train / val2 / test3 (cached)
    python -m aiengine.up2c select     # fit v3c (with physics) + ablation (same, without physics) on TRAIN runs,
                                       # select everything on synth_v3_val2 ONLY
    python -m aiengine.up2c predict    # ONE scoring pass on test3: v3c, ablation, v3b (frozen, for comparison)
    python -m aiengine.up2c export     # models/v3c_rul.onnx + model_contract_v3c.json + golden_vectors_v3c.json
    python -m aiengine.up2c_eval       # re-score the saved predictions: S7, IS3, S8, IS2 (literal + independent), ablation

Design (see aiengine/up2c_physics.py for the exact per-window physics feature function):
  4 payload displacement floats (2 proximity probes x 1x amplitude/phase, COE) -> FE twin (unbalance commissioning,
  K identification = the twin's objective) -> per bearing: K estimate, K-drop fraction, physics RUL (stiffness-trend
  RUL from the twin's fit_rul, or a documented POPULATION-LIFE PRIOR = L_pop(rpm) - age when no significant trend
  exists), confidence (0 = prior). These 4 columns are appended to the 21 engineered accelerometer columns of the
  N-HiTS-slot input (encoder_cont [1, 48, 25]); the GBM RUL head consumes them through the same in-graph aggregator.
  Classifier (v3) and stage/z head (v3b) are reused UNCHANGED (the z head reads the first 21 columns only), so the
  IS3 path is the v3b one; only the RUL head and its post-process are new.

Data sets: train = synth_v3 train + synth_v3_train2 + synth_v3_train3 (the v3b training runs); selection and early
stopping = synth_v3_val2 ONLY; synth_v3_test2 was already scored for v3b, so a NEW fresh set synth_v3_test3 (100 runs,
new seed, run ids 60000+) is registered in splits.json before any v3c model is fitted and is scored exactly once.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import splits
from .config import CHECKPOINT_DIR, SEED

RUN_DIR = CHECKPOINT_DIR / "up2_v3c"
TEST3 = ("synth_v3_test3", 100, 808, 60000)        # name, runs, seed offset, run-id offset


def load_test3() -> pd.DataFrame:
    from .up2b import _gen
    d = _gen(TEST3)
    splits.register_holdout(d, TEST3[0], "test",
                            note="UP2 v3c FRESH test runs (same generator, new seed); registered before any v3c fit; "
                                 "scored once (v3c, its no-physics ablation and the frozen v3b in the same pass)")
    d["split"] = splits.assign(d).to_numpy()
    return d


def cmd_data(a) -> int:
    t3 = load_test3()
    print(json.dumps({"test3": {"runs": int(t3["bearing_key"].nunique()), "windows": int(len(t3)),
                                "rtf_bearings": int(t3[t3["run_to_failure"]]["unit_id"].nunique()),
                                **t3.attrs.get("provenance", {})}}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------- data sets

def load_sets() -> dict[str, pd.DataFrame]:
    """name -> labelled table with `split`. train: synth_v3 train + train2 + train3; val: val2 ONLY; test: test3."""
    from . import up2, up2b
    from .synth_v3 import generator as gen
    from .up2_train import EXTRA_TRAIN
    m = up2.prepare(gen.DATASET)
    name, n, off = EXTRA_TRAIN
    t2 = up2.prepare(name, freeze_splits=False, n_runs=n, seed=SEED + off, run_offset=20000)
    t2["split"] = "train"
    t3 = up2b._gen(up2b.TRAIN3)
    t3["split"] = "train"
    return {"synth_v3_train": m[m["split"] == "train"].copy(), "synth_v3_train2": t2, "synth_v3_train3": t3,
            "synth_v3_val2": up2b.load_val2(), "synth_v3_test3": load_test3()}


def phys_path(name: str):
    from .config import UP2_CACHE_DIR
    return UP2_CACHE_DIR / f"up2c_phys_{name}.parquet"


def compute_physics(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Per bearing-window physics features (PHYS_COLUMNS) + the run's 4 displacement floats + identified K."""
    from concurrent.futures import ProcessPoolExecutor
    from . import up2c_physics as P
    from .payload import DISPLACEMENT_NAMES
    jobs = []
    for key, g in df.groupby("bearing_key", sort=True):
        rtf = g[g["run_to_failure"]]
        first = (rtf if len(rtf) else g)["unit_id"].iloc[0]
        u = g[g["unit_id"] == first].sort_values("window_index")
        jobs.append((key, u[["run_id", "rpm", "t_hours", "window_index", "damage", "run_to_failure", "faulted_bearing"]]
                     .reset_index(drop=True), SEED))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(P.run_features, jobs, chunksize=4):
            base = {"bearing_key": r["key"], "window_index": r["window_index"]}
            for b in (1, 2):
                f = pd.DataFrame(base)
                f["bearing"] = b
                for j, c in enumerate(P.PHYS_COLUMNS):
                    f[c] = r["feats"][:, b - 1, j]
                for j, c in enumerate(DISPLACEMENT_NAMES):
                    f[c] = r["disp"][:, j]
                f["k_ident"] = r[f"k{b}"]
                f["s_ident"] = r[f"s{b}"]
                rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    return out


def attach_physics(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ph = pd.read_parquet(phys_path(name))
    d = df.copy()
    d["bearing"] = d["unit_id"].str[-1].astype(int)
    d = d.merge(ph, on=["bearing_key", "window_index", "bearing"], how="left")
    from .up2c_physics import PHYS_COLUMNS
    assert d[PHYS_COLUMNS].notna().all().all(), f"{name}: rows without physics features"
    d.attrs = df.attrs
    return d


def cmd_physics(a) -> int:
    import time
    from .up2c_physics import population_life
    sets = load_sets()
    print("population life (TRAIN synth_v3_train3, median failure time h):", population_life(sets["synth_v3_train3"]), flush=True)
    for name, d in sets.items():
        p = phys_path(name)
        if p.exists() and not a.force:
            print(name, "cached", p.name, flush=True)
            continue
        t0 = time.perf_counter()
        ph = compute_physics(d, a.workers)
        ph.to_parquet(p, index=False)
        print(name, len(ph), "rows", f"{time.perf_counter() - t0:.0f} s", flush=True)
    return 0


# --------------------------------------------------------------------------- features + selection

NL, WARM = 48, 25
TRUTH_COLS = ["unit_id", "window_index", "run_to_failure", "u_true", "rul_windows", "z_target", "health_stage"]
CANDIDATES = [(phys, rows) for phys in (True, False) for rows in ("deg", "all")]


def columns() -> list[str]:
    from . import up2
    from .up2c_physics import PHYS_COLUMNS
    return list(up2.COLUMNS) + list(PHYS_COLUMNS)


def old_feature_index(R: int, Rp: int) -> np.ndarray:
    """Positions of the v3b (21-column) aggregator features inside the 25-column aggregator output (aggregate-major)."""
    from . import up2_window as uw
    return np.array([a * Rp + c for a in range(len(uw.AGGS)) for c in range(R)], np.int64)


def feat_file(part: str):
    return RUN_DIR / "feat" / f"{part}_NL{NL}_w{WARM}.npz"


def build_feats(df: pd.DataFrame, part: str):
    """Aggregator features (25 columns) of the run-to-failure units of df (RUL head rows), cached."""
    from . import up2
    p = feat_file(part)
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["X"], pd.DataFrame({k: z[k] for k in TRUTH_COLS})
    d = df[df["run_to_failure"]]
    Xs, idxs = [], []
    for ds, g in d.groupby("dataset", sort=True):
        X, idx = up2.unit_features(g, NL, with_decoder=False, warm=WARM, columns=columns())
        Xs.append(X); idxs.append(idx)
    X = np.concatenate(Xs).astype(np.float32)
    t = up2.merge_truth(pd.concat(idxs, ignore_index=True), d, TRUTH_COLS[2:])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, **{k: t[k].to_numpy() for k in TRUTH_COLS})
    return X, t


def cmd_feats(a) -> int:
    sets = load_sets()
    tr = pd.concat([attach_physics(sets[n], n) for n in ("synth_v3_train", "synth_v3_train2", "synth_v3_train3")],
                   ignore_index=True)
    assert tr["split"].eq("train").all()
    X, t = build_feats(tr, "train")
    print("train", X.shape, flush=True)
    va = attach_physics(sets["synth_v3_val2"], "synth_v3_val2")
    assert va["split"].eq("val").all()
    X, t = build_feats(va, "val2")
    print("val2", X.shape, flush=True)
    return 0


def _rows(t: pd.DataFrame, rows: str) -> np.ndarray:
    return (t["u_true"] >= 0).to_numpy() if rows == "deg" else np.ones(len(t), bool)


def cmd_fit(a) -> int:
    """Fit ONE candidate (physics on/off x row set) on TRAIN, early stopping on val2. Same objective/params as v3b."""
    import pickle
    from . import up2, up2b
    phys, rows = CANDIDATES[a.candidate]
    Xtr, ttr = build_feats(None, "train")
    Xva, tva = build_feats(None, "val2")
    R, Rp = len(up2.COLUMNS), len(columns())
    cols = slice(None) if phys else old_feature_index(R, Rp)
    mtr, mva = _rows(ttr, rows), _rows(tva, rows)
    m, info = up2b.fit_rul(Xtr[mtr][:, cols], ttr[mtr].reset_index(drop=True), Xva[mva][:, cols],
                           tva[mva].reset_index(drop=True), "log_l2", SEED, a.threads)
    info.update({"physics": phys, "rows": rows, "n_train": int(mtr.sum()), "n_features": int(Xtr[:, cols].shape[1])})
    print(json.dumps(info), flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / f"cand_{int(phys)}_{rows}.pkl", "wb") as f:
        pickle.dump({"model": m, "info": info}, f)
    return 0


def _val_truth():
    from . import up2b
    va = up2b.load_val2()
    return va


def cmd_select(a) -> int:
    """Select, on synth_v3_val2 ONLY: row set per physics setting + RUL post-process (alpha, stage scales) with the
    v3b stage head unchanged. Same grid and tie rule as v3b (up2b.cmd_select_rulpp)."""
    import pickle
    from . import up2, up2b
    from .up2_train import meta, now_tag
    va = _val_truth()
    Xva, tva = build_feats(None, "val2")
    R, Rp = len(up2.COLUMNS), len(columns())
    oi = old_feature_index(R, Rp)
    hb = up2b.load_heads()
    zsel = hb["sel"]["z"]
    st_raw = up2b.stage_from_z(hb["z"].predict(Xva[:, oi]), zsel["z_cut"], zsel.get("z_shift", 0.0))
    spp = hb["sel"]["postprocess"]
    rows, models = [], {}
    for phys, rset in CANDIDATES:
        with open(RUN_DIR / f"cand_{int(phys)}_{rset}.pkl", "rb") as f:
            c = pickle.load(f)
        models[(phys, rset)] = c
        b = tva[["unit_id", "window_index"]].copy()
        b["rul_pred_windows_raw"] = up2b.rul_windows_from([c["model"]], "log_l2", Xva if phys else Xva[:, oi])
        b["stage_pred_raw"] = st_raw
        for al in up2b.ALPHAS:
            p = up2b.apply_pp(b, {"rul_alpha": al, "stage_k": spp["stage_k"], "stage_max_step": spp["stage_max_step"]})
            for se in up2b.SCALES:
                for sl in up2b.SCALES:
                    s = up2b._rul_score(up2b._scale_rows(p, se, sl), va)
                    rows.append({"physics": phys, "rows": rset, "alpha": al, "scale_early": se, "scale_late": sl,
                                 "val_S7": s["S7_mape_pct"], "val_true_onset": s["alt_mape_true_onset_pct"],
                                 "val_pol": s["alt_percent_of_life_error_pts"]})
        raw = [r for r in rows if r["physics"] == phys and r["rows"] == rset and r["alpha"] == 1.0
               and r["scale_early"] == 1.0 and r["scale_late"] == 1.0][0]
        print("cand phys=%s rows=%s raw val S7 %.2f" % (phys, rset, raw["val_S7"]), flush=True)
    key = lambda r: (round(r["val_S7"] / 0.05), r["rows"] != "deg", -r["alpha"],  # noqa: E731
                     abs(1 - r["scale_early"]) + abs(1 - r["scale_late"]))
    sel = {}
    for phys in (True, False):
        best = min((r for r in rows if r["physics"] == phys), key=key)
        name = "v3c" if phys else "ablation"
        sel[name] = {"physics": phys, "rows": best["rows"], "NL": NL, "warm": WARM, "target": "log_l2",
                     "transform": "log1p", "columns": columns() if phys else list(up2.COLUMNS),
                     "fit": models[(phys, best["rows"])]["info"],
                     "postprocess": {"rul_alpha": best["alpha"], "rul_scale_early": best["scale_early"],
                                     "rul_scale_late": best["scale_late"], "stage_k": spp["stage_k"],
                                     "stage_max_step": spp["stage_max_step"]},
                     "z": {"reused": "v3b z head (checkpoints/up2_v3b/heads.pkl), first 21 columns", **{k: zsel[k] for k in ("z_cut", "z_shift") if k in zsel}},
                     "val": {"S7": best["val_S7"], "true_onset": best["val_true_onset"], "pol": best["val_pol"]}}
        print(name, json.dumps(sel[name]["postprocess"]), json.dumps(sel[name]["val"]), best["rows"], flush=True)
    with open(RUN_DIR / "heads.pkl", "wb") as f:
        pickle.dump({k: {"model": models[(v["physics"], v["rows"])]["model"], "sel": v} for k, v in sel.items()}, f)
    (RUN_DIR / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
    out = {"meta": meta({"selection": "synth_v3_val2 ONLY (early stopping and every choice)"}), "grid": rows, "selected": sel}
    path = REPORTS_DIR_() / f"model_selection_up2_v3c_{now_tag()}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("->", path)
    return 0


def REPORTS_DIR_():
    from .config import REPORTS_DIR
    return REPORTS_DIR


# --------------------------------------------------------------------------- prediction (single scoring pass)

FAMILY = "up2_v3c_gbm"


def load_heads():
    import pickle
    with open(RUN_DIR / "heads.pkl", "rb") as f:
        return pickle.load(f)


def predict_rul(df: pd.DataFrame, head: dict, hb: dict) -> pd.DataFrame:
    """Deployed v3c / ablation RUL+stage logic on every unit of df (df carries the physics columns)."""
    from . import up2, up2b
    sel = head["sel"]
    X, idx = up2.unit_features(df, NL, with_decoder=False, warm=WARM, columns=columns())
    oi = old_feature_index(len(up2.COLUMNS), len(columns()))
    rp = idx.copy()
    rp["rul_pred_windows_raw"] = up2b.rul_windows_from([head["model"]], "log_l2", X if sel["physics"] else X[:, oi])
    rp["z_pred"] = hb["z"].predict(X[:, oi])
    z = hb["sel"]["z"]
    rp["stage_pred_raw"] = up2b.stage_from_z(rp["z_pred"].to_numpy(), z["z_cut"], z.get("z_shift", 0.0))
    rp = up2b.apply_pp(rp, sel["postprocess"])
    from .synth_v3 import generator as gen
    rp["rul_pred_h_raw"] = rp["rul_pred_windows_raw"] * gen.CADENCE_H
    return rp


def cmd_predict(a) -> int:
    from . import up2b
    from .up2_train import meta, now_tag
    from .config import REPORTS_DIR
    pdir = REPORTS_DIR / "up2_preds"
    prev = sorted(pdir.glob(f"{FAMILY}_*.index.json"))
    if prev and not a.force:
        raise SystemExit(f"test3 was already scored ({prev[-1].name}); re-scoring is `python -m aiengine.up2c_eval`")
    tag = now_tag()
    t3 = attach_physics(load_test3(), "synth_v3_test3")
    hb = up2b.load_heads()
    h = load_heads()
    cp, rp_b = up2b.predict(t3, hb)                    # v3b frozen (classifier shared by all three)
    files = {}
    f = pdir / f"{FAMILY}_test3_{tag}.cls.csv.gz"
    cp.to_csv(f, index=False, float_format="%.6g")
    files["cls"] = f.name
    for name, rp in (("v3b", rp_b), ("v3c", predict_rul(t3, h["v3c"], hb)), ("ablation", predict_rul(t3, h["ablation"], hb))):
        f = pdir / f"{FAMILY}_test3_{name}_{tag}.rul.csv.gz"
        rp.to_csv(f, index=False, float_format="%.6g")
        files[name] = f.name
        from .up2_eval import score_set
        s = score_set(cp, rp, t3, WARM - 1)
        print(name, "S7 %.2f  stage max %s  caught %s/%s  F1 %.4f" % (s["rul"]["S7_mape_pct"], s["stage"]["max_abs_error"],
              s["caught"]["caught_by_stage3"], s["caught"]["rtf_bearings"], s["classification"]["macro_f1"]), flush=True)
    out = {"meta": meta({"tag": tag, "touch": "synth_v3_test3: FIRST and only scoring pass (v3c, ablation, v3b frozen)",
                         "selection": json.loads((RUN_DIR / "selection.json").read_text())}), "files": files}
    (pdir / f"{FAMILY}_{tag}.index.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


# --------------------------------------------------------------------------- export

RUL_FILE, CONTRACT_FILE, GOLDEN_FILE = "v3c_rul.onnx", "model_contract_v3c.json", "golden_vectors_v3c.json"


def build_rul_onnx(rul_model, z_model, R: int, Rp: int, z_cut: float, z_shift: float):
    """encoder_cont [B, 48, 25] -> aggregator (25 cols) -> RUL trees; Gather(v3b 21-col features) -> z trees."""
    from onnx import TensorProto, helper, numpy_helper
    from . import up2_export as ux
    from . import up2_window as uw
    agg = uw.aggregator_graph(NL, Rp, with_decoder=False, out_name="feat")
    nodes, inits, opsets = list(agg.graph.node), list(agg.graph.initializer), list(agg.opset_import)
    inits.append(numpy_helper.from_array(old_feature_index(R, Rp), "g_old_idx"))
    nodes.append(helper.make_node("Gather", ["feat", "g_old_idx"], ["feat_old"], axis=1))
    rn, ri, ro, rops = ux._inline(ux._convert(rul_model, uw.n_features(Rp, False)), "r0_", "feat")
    zn, zi, zo, zops = ux._inline(ux._convert(z_model, uw.n_features(R, False)), "z_", "feat_old")
    nodes += rn + zn; inits += ri + zi; opsets += rops + zops
    rv, zv = list(ro.values())[0], list(zo.values())[0]
    c = lambda v, n: numpy_helper.from_array(np.array(v, np.float32), n)   # noqa: E731
    nodes += [
        helper.make_node("Exp", [rv], ["p_e"]), helper.make_node("Sub", ["p_e", "p_one"], ["p_t"]),
        helper.make_node("Max", ["p_t", "p_zero"], ["p_rul"]),
        helper.make_node("Add", [zv, "p_shift"], ["p_zs"]),
        helper.make_node("Clip", ["p_zs", "p_zero", "p_one"], ["p_zc"]),
        helper.make_node("Mul", ["p_zc", "p_five"], ["p_z5"]),
        helper.make_node("Floor", ["p_z5"], ["p_fl"]),
        helper.make_node("Add", ["p_fl", "p_two"], ["p_st0"]),
        helper.make_node("Min", ["p_st0", "p_six"], ["p_st1"]),
        helper.make_node("Less", [zv, "p_cut"], ["p_pre"]),
        helper.make_node("Where", ["p_pre", "p_one", "p_st1"], ["p_stage"]),
        helper.make_node("Concat", ["p_rul", zv, "p_stage"], ["rul_head"], axis=1),
    ]
    inits += [c(1.0, "p_one"), c(0.0, "p_zero"), c(5.0, "p_five"), c(2.0, "p_two"), c(6.0, "p_six"),
              c(float(z_cut), "p_cut"), c(float(z_shift), "p_shift")]
    outputs = [helper.make_tensor_value_info("rul_head", TensorProto.FLOAT, ["batch", 3])]
    return ux._model(nodes, inits, list(agg.graph.input), outputs, opsets, "v3c_rul")


def cmd_export(a) -> int:
    import copy
    import onnx
    import onnxruntime as ort
    from . import up2, up2b
    from . import up2_window as uw
    from . import up2c_physics as P
    from .config import MODELS_DIR
    from .train import git_commit
    hb, h = up2b.load_heads(), load_heads()
    sel = h["v3c"]["sel"]
    R, Rp = len(up2.COLUMNS), len(columns())
    z = hb["sel"]["z"]
    m = build_rul_onnx(h["v3c"]["model"], hb["z"], R, Rp, z["z_cut"], z.get("z_shift", 0.0))
    onnx.save(m, str(MODELS_DIR / RUL_FILE))
    # parity: ONNX vs native trees on test3 units (no scoring)
    t3 = attach_physics(load_test3(), "synth_v3_test3")
    units = t3[t3["run_to_failure"]]["unit_id"].drop_duplicates().iloc[:4].tolist() + \
        t3[~t3["run_to_failure"]]["unit_id"].drop_duplicates().iloc[:2].tolist()
    d = t3[t3["unit_id"].isin(units)]
    rp = predict_rul(d, h["v3c"], hb).sort_values(["unit_id", "window_index"])
    s = ort.InferenceSession(str(MODELS_DIR / RUL_FILE), providers=["CPUExecutionProvider"])
    ro = []
    for uid, g in d.groupby("unit_id", sort=True):
        br, _ = uw.blocks(g.sort_values("window_index")[columns()].to_numpy(np.float64), NL)
        ro.append(s.run(["rul_head"], {"encoder_cont": br[WARM - 1:]})[0])
    ro = np.concatenate(ro)
    ref = rp["rul_pred_windows_raw"].to_numpy()
    par = {"windows": int(len(ro)), "rul_windows_max_rel_diff": float((np.abs(ro[:, 0] - ref) / np.maximum(ref, 1.0)).max()),
           "z_max_abs_diff": float(np.abs(ro[:, 1] - rp["z_pred"].to_numpy()).max()),
           "stage_mismatch": int((ro[:, 2].astype(int) != rp["stage_pred_raw"].to_numpy()).sum())}
    con = copy.deepcopy(json.loads((MODELS_DIR / up2b.CONTRACT_FILE).read_text()))
    con["contract_version"] = "3c"
    con["generated_by"] = "AI-engine/aiengine/up2c.py export"
    con["generated_utc"] = pd.Timestamp.utcnow().isoformat()
    mn = con["models"]["nhits"]
    mn.update({"onnx": RUL_FILE, "columns": columns(),
               "scalers": {c: {"center": 0.0, "scale": 1.0, "kind": "identity"} for c in columns()},
               "family": "gradient-boosted trees in the N-HiTS slot: PHYSICS-INFORMED RUL LightGBM (log1p, 25 input columns = "
                         "21 engineered + 4 physics) + v3b z LightGBM (reads the 21 engineered columns via an in-graph Gather)",
               "size_kb": round((MODELS_DIR / RUL_FILE).stat().st_size / 1024, 1), "parity": par,
               "row_layout": "row = [21 engineered columns in feature_engineering.engineered_order (as v3b), then the 4 "
                             "physics_features.columns of THIS bearing for the same window]"})
    con["physics_features"] = {
        **P.phys_spec(),
        "input": "payload p1_amp_um, p1_phase_rad, p2_amp_um, p2_phase_rad (bytes 132..147) + rpm + t_h (hours since the "
                 "stream started, same clock as the RUL cadence)",
        "state": "per rig (both bearings): DigitalTwin.Twin FeBeamPhysicsTwin commissioning/unbalance/K0 + BearingTracker per "
                 "bearing (gated K history, rul_fit_points) + per-bearing EMA (num, den)",
        "procedure": P.__doc__.split("Algorithm")[1].split("Constants live")[0].strip(),
        "outputs_per_bearing": {"phys_k_drop": "1 - r_ema", "phys_k_est": "K0 * r_ema / 1000 [kN/m]",
                                "phys_rul_h": "trend RUL (clipped) or prior [h]", "phys_conf": "lo/hi of trend interval, 0 = prior"},
        "identification": "Identifier.Identify (twin objective, unchanged). Training data used aiengine/up2c_physics.py::"
                          "batch_identify (same objective, exact condensed FE model); parity vs stock identify: K rel diff <= 5e-6.",
        "reference_impl": ["aiengine/up2c_physics.py::replay_features (offline)", "aiengine/up2c_physics.py::PhysicsFeatures (streaming)",
                           "aiengine/up2c_engine.py::PhysicsHybridEngine"],
        "provenance": "[SIMULATION] FE twin with ASSUMED geometry (aiengine/twin/params.py); training displacements [SYNTHETIC] "
                      "from the twin with K(t) = K0 (1 - 0.25 damage) of the generator (aiengine/up2_residual.py, nominal sensor "
                      "noise +/-5 % amplitude, +/-0.05 rad). Population life L_pop from TRAIN runs [SYNTHETIC].",
    }
    con["postprocess"] = up2b.postprocess_contract(sel["postprocess"])
    con["postprocess"]["type"] = "v3b_stage_median_rul_logema"      # same dispatch key / algorithm as v3b, new constants
    con["postprocess"]["selected_on"] = "synth_v3_val2 ONLY"
    con["postprocess"]["reference_impl"] = ["aiengine/up2b.py::apply_pp (offline)", "aiengine/engine.py::HybridEngine._postprocess (streaming)"]
    con["provenance"] = {"synthetic": True, "run": "up2_v3c_gbm", "git_commit": git_commit(), "seed": SEED,
                         "datasets": ["synth_v3 train", "synth_v3_train2", "synth_v3_train3 (train)", "synth_v3_val2 (val ONLY)"],
                         "selection": sel, "note": "[SYNTHETIC] v3 classifier + v3b z head unchanged; v3c physics-informed RUL head"}
    con["disclosures"]["test_sets"] = ("synth_v3_test3 (100 fresh runs, splits.json) registered before any v3c fit and scored "
                                       "once for v3c, its no-physics ablation and v3b. v3b's test2 numbers are history.")
    con["disclosures"]["circularity"] = ("[SYNTHETIC] the training displacements come from a stiffness trajectory derived from the "
                                         "same generator damage that drives the accelerometer features and the RUL label; the "
                                         "physics features' benefit is therefore an upper bound until rig data exists.")
    (MODELS_DIR / CONTRACT_FILE).write_text(json.dumps(con, indent=2, default=float))
    from .up2c_engine import golden
    g = golden(MODELS_DIR, CONTRACT_FILE, t3, h, hb)
    (MODELS_DIR / GOLDEN_FILE).write_text(json.dumps(g, indent=1, default=float))
    print(json.dumps({"contract": CONTRACT_FILE, "parity": par, "size_kb": mn["size_kb"],
                      "engine_vs_offline": g["engine_vs_offline"]}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2c")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    ph = sub.add_parser("physics")
    ph.add_argument("--workers", type=int, default=20)
    ph.add_argument("--force", action="store_true")
    sub.add_parser("feats")
    fi = sub.add_parser("fit")
    fi.add_argument("--candidate", type=int, required=True, help=f"index into {CANDIDATES}")
    fi.add_argument("--threads", type=int, default=6)
    sub.add_parser("select")
    pr = sub.add_parser("predict")
    pr.add_argument("--force", action="store_true", help="re-run the scoring pass (must be disclosed)")
    sub.add_parser("export")
    a = ap.parse_args(argv)
    return {"data": cmd_data, "physics": cmd_physics, "feats": cmd_feats, "fit": cmd_fit, "select": cmd_select,
            "predict": cmd_predict, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
