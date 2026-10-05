"""AI-engine CLI: N-HiTS (RUL) + TFT (fault class), per biaxial bearing.

    app.py selftest                       checks needing no trained model (~10 s)
    app.py data      [--datasets ...]     build window tables -> features -> labels -> splits.json
    app.py train     [--run R] [...]      pretrain TFT + N-HiTS, select the RUL estimator on val bearings
    app.py finetune  [--run R]            fine-tune the pretrained models on IMS ft_train bearings
    app.py evaluate  [--run R]            score held-out test bearings -> reports/spec_check_*.{json,md}
    app.py export    [--run R]            ONNX + parity + models/model_contract.json + golden vectors
    app.py demo      [--unit U]           stream 152-byte payloads of a test unit through HybridEngine

Datasets: synthetic_rig (SYNTHETIC, always available) and the real caches xjtu_sy, ims, mafaulda
(data/cache/*.parquet, written by the dataset adapters). Default = every real cache present,
else synthetic_rig. Run `app.py <command> --help` for options.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _datasets(args) -> list[str]:
    from aiengine import train
    if getattr(args, "datasets", None):
        return args.datasets
    real = train.available()
    return real if real else ["synthetic_rig"]


def _run_dir(args) -> Path:
    from aiengine.config import CHECKPOINT_DIR
    return CHECKPOINT_DIR / args.run


def _meta(args, names, df) -> dict:
    from aiengine import evaluate, train
    from aiengine.config import SEED
    return {"datasets": names, "real_data": [n for n in names if n != "synthetic_rig"],
            "synthetic": "synthetic_rig" in names, "seed": SEED, "git_commit": train.git_commit(),
            "git_dirty": train.git_dirty(), "command": train.command_line(), "run": args.run,
            "host": evaluate.host_info(), "counts": train.counts(df)}


# --------------------------------------------------------------------------- commands

def cmd_selftest(args) -> int:
    from aiengine import features as feat
    from aiengine import labels, payload, rul, splits
    from aiengine.config import BEARING_FEATURES
    from aiengine.datasets import synthetic_rig
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    w = payload.Window(rpm=1750.0, features=np.arange(32, dtype=np.float32), p1_amp_um=1, p1_phase_rad=0.1,
                       p2_amp_um=2, p2_phase_rad=0.2, t20_ms=123)
    raw = payload.encode(w)
    check("payload 152-byte round trip", len(raw) == 152 and np.array_equal(payload.decode(raw).features, w.features))
    t = synthetic_rig.build(n_runs=6, min_windows=80, max_windows=120)
    check("synthetic_rig schema", set(t["dataset"]) == {"synthetic_rig"} and t["unit_id"].nunique() == 12)
    eng = feat.engineer_table(t)
    check("engineered inputs finite", np.isfinite(eng[feat.ENGINEERED].to_numpy()).all(), f"({len(feat.ENGINEERED)} inputs)")
    u = eng[eng["unit_id"] == eng["unit_id"].iloc[0]]
    b = feat.baseline_first_windows(u)
    one = np.stack([feat.engineer_arrays(u[BEARING_FEATURES].to_numpy()[i:i + 1], u["rpm"].to_numpy()[i:i + 1], b)[0]
                    for i in range(len(u))])
    check("per-window == batch engineering", np.allclose(one, feat.engineer_arrays(u[BEARING_FEATURES].to_numpy(), u["rpm"].to_numpy(), b)))
    sym = feat.engineer_arrays(np.tile(b, (1, 1)), np.array([1750.0]), b)[0]
    check("baseline window -> HI 0, axis_asymmetry 0", abs(sym[0]) < 1e-6 and abs(sym[feat.ENGINEERED.index('axis_asymmetry')]) < 1e-6)
    lab = labels.label_table(eng)
    rtf = lab[lab["run_to_failure"]]
    agree = True
    for uid, g in rtf.groupby("unit_id"):
        det = labels.OnlineOnset()
        for h in g["hi"]:
            det.update(h)
        agree &= (det.onset_index if det.onset_index is not None else -1) == g["onset_index"].iloc[0]
    check("online onset == offline onset (causal)", agree)
    check("stages in 1..6 for RTF units", rtf["health_stage"].between(1, 6).all())
    check("stage rule", list(labels.stage_from_fraction(np.array([-0.1, 0, 0.19, 0.2, 0.99, 1.0]))) == [1, 2, 2, 3, 6, 6])
    c = rul.RULCalibration(method="trend", class_thresholds={"outer_race": 1.0}, pooled_threshold=1.0)
    hi = np.exp(np.linspace(np.log(0.1), np.log(0.5), 24))
    e = rul.estimate(c, hi, hi[-1] * np.exp(np.linspace(0.07, 0.07 * 12, 12)), "outer_race", None, None, 1.0)
    exp_steps = np.log(1.0 / 0.5) / (np.log(5) / 23)
    check("trend RUL on an exact exponential", abs(e["rul_hours"] - exp_steps) / exp_steps < 0.1,
          f"({e['rul_hours']:.1f} vs {exp_steps:.1f} h)")
    m = splits.unit_split_map(splits.load())
    check("splits.json loads", isinstance(m, dict), f"({len(m)} units frozen)")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_data(args) -> int:
    from aiengine import train
    names = _datasets(args)
    df = train.prepare(names)
    print(json.dumps(train.counts(df), indent=1))
    print("observable classes (usable units):")
    print(df[df["cls_usable"]].groupby(["dataset", "observable_class"]).size().to_string())
    rtf = df[df["run_to_failure"]].groupby("unit_id").agg(n=("window_index", "size"), onset=("onset_index", "first"),
                                                          life_h=("life_hours", "first"), split=("split", "first"))
    print(rtf.to_string())
    return 0


def cmd_train(args) -> int:
    from aiengine import evaluate, models, train
    train.set_threads(args.threads)
    train.seed_all()
    names = _datasets(args)
    df = train.prepare(names)
    pre = df[df["dataset"] != "ims"]                  # IMS = fine-tuning domain, never pretrained on
    run = _run_dir(args)
    run.mkdir(parents=True, exist_ok=True)
    tc = models.TFTConfig(max_encoder_length=args.tft_encoder, max_epochs=args.epochs or 25,
                          class_weight_power=args.class_weight_power)
    nc = models.NHiTSConfig(encoder_length=args.nhits_encoder, prediction_length=args.nhits_horizon,
                            max_epochs=args.epochs or 25)
    if args.tft_inputs:
        tc.inputs = args.tft_inputs.split(",")
    if args.nhits_covariates is not None:
        nc.covariates = [c for c in args.nhits_covariates.split(",") if c]
    prev = run / "run_meta.json"
    meta = json.loads(prev.read_text()) if prev.exists() else {}
    meta.setdefault("history", []).append(_meta(args, names, df))
    meta["provenance"] = meta["history"][-1]
    if not args.skip_tft and (args.hierarchical or args.balance):
        # v2 (DESIGN.md change log 2026-10-05 v2): settings chosen by LOBO over train+val bearings
        # (aiengine/hier.py); the final model trains on train+val for the same fixed epochs, no early
        # stopping (test bearings never seen).
        from aiengine import hier
        v = {"hier": args.hierarchical, "power": args.class_weight_power, "balance": args.balance}
        fitdf = pre[pre.split.isin(["train", "val"])]
        print(f"[TFT v2] {v} on {fitdf.unit_id.nunique()} train+val units, {args.epochs} epochs ...", flush=True)
        _, meta["tft"] = hier.fit(fitdf, fitdf.iloc[:0], v, run, encoder=args.tft_encoder, epochs=args.epochs or 20)
        meta["tft"]["v2_variant"] = v
        print(f"  {meta['tft']['train_seconds']} s", flush=True)
    elif not args.skip_tft:
        print(f"[TFT] training on {pre[pre.split == 'train'].unit_id.nunique()} units ...", flush=True)
        _, meta["tft"] = train.train_tft(pre[pre.split == "train"], pre[pre.split == "val"], tc, run)
        print(f"  {meta['tft']['train_seconds']} s, val loss {meta['tft']['best_val_loss']}", flush=True)
    if not args.skip_nhits:
        r = train.rtf_frames(pre)
        print(f"[N-HiTS] training on {r[r.split == 'train'].unit_id.nunique()} run-to-failure units ...", flush=True)
        _, meta["nhits"] = train.train_nhits(r[r.split == "train"], r[r.split == "val"], nc, run)
        print(f"  {meta['nhits']['train_seconds']} s, val MAE {meta['nhits']['best_val_loss']}", flush=True)
    if args.rul_from:
        meta["rul_selection"] = calib_from_selection(run, pre, Path(args.rul_from))
        print(f"[RUL] from LOBO selection {args.rul_from}: {meta['rul_selection']['selected']}")
    else:
        meta["rul_selection"] = select_rul(run, pre)
        print(f"[RUL] selected {meta['rul_selection']['selected']['method']} "
              f"(fit MAPE on val bearings {meta['rul_selection']['fit_mape']})")
    train.write_json(run / "run_meta.json", meta)
    return 0


def select_rul(run: Path, pre: pd.DataFrame, tag: str = "") -> dict:
    """Thresholds from TRAIN bearings; estimator chosen on VAL bearings (out-of-sample forecasts)."""
    from aiengine import train
    tft = train.load_model(run / "tft.ckpt")
    nh = train.load_model(run / f"nhits{tag}.ckpt")
    fit_split = "val" if (pre[(pre.split == "val") & pre.run_to_failure].unit_id.nunique() > 0) else "train"
    fit_df = pre[pre.split == fit_split]
    recs = train.rul_records(nh, fit_df, train.tft_predict(tft, fit_df))
    calib, sel = train.fit_rul(recs, train.end_hi_records(pre[pre.split == "train"]))
    scored = train.apply_rul(calib, recs)
    d = scored[scored["in_deg"] & (scored["rul_true"] > 0)]
    rel = np.abs(d["rul_pred"] - d["rul_true"]) / d["rul_true"]
    calib_d = calib.to_dict()
    calib_d["interval_rel"] = float(np.quantile(rel, 0.8)) if len(rel) else 0.5
    calib_d["fit_split"] = fit_split
    (run / f"rul_calibration{tag}.json").write_text(json.dumps(calib_d, indent=2))
    sel["fit_split"] = fit_split
    sel["per_bearing_fit_mape"] = train.s7_mape(scored)[1]
    return sel


def calib_from_selection(run: Path, pre: pd.DataFrame, sel_file: Path) -> dict:
    """RUL calibration = the LOBO-selected estimator; thresholds re-fit on ALL train+val
    run-to-failure bearings at the selected quantile; interval from the out-of-fold errors."""
    from aiengine import rul, train
    sel = json.loads(sel_file.read_text())
    b = sel["best"]
    fit = pre[pre["split"].isin(["train", "val"])]
    th, pooled = rul.fit_thresholds(train.end_hi_records(fit), quantile=b["q"])
    calib = rul.RULCalibration(method=b["method"], history_windows=b["history"], blend_w=b["blend_w"],
                               u_min=b["u_min"], class_thresholds=th, pooled_threshold=pooled,
                               loglin_coef=b.get("loglin_coef", []))
    d = calib.to_dict()
    d.update({"threshold_quantile": b["q"], "selected_by": str(sel_file.name), "lobo_mape": b["lobo_mape"],
              "interval_rel": float(sel.get("interval_rel", 0.5)), "fit_split": "train+val (LOBO)"})
    (run / "rul_calibration.json").write_text(json.dumps(d, indent=2))
    return {"selected": d, "fit_mape": b["lobo_mape"]}


def cmd_select(args) -> int:
    from aiengine import models, select, train
    train.set_threads(args.threads)
    names = _datasets(args)
    df = train.prepare(names)
    pre = df[(df["dataset"] != "ims") & df["split"].isin(["train", "val"])]   # never test
    run = _run_dir(args)
    meta = _meta(args, names, df)
    if args.what == "rul":
        nc = models.NHiTSConfig(encoder_length=args.nhits_encoder, prediction_length=args.nhits_horizon)
        if args.nhits_covariates is not None:
            nc.covariates = [c for c in args.nhits_covariates.split(",") if c]
        res = select.run_rul(pre, nc, run / "tft.ckpt", run / f"select_rul_E{nc.encoder_length}_H{nc.prediction_length}",
                             epochs=args.epochs)
        # interval half-width = 80th percentile of out-of-fold relative error of the best estimator
        import pickle
        recs = pd.read_pickle(run / f"select_rul_E{nc.encoder_length}_H{nc.prediction_length}" / "lobo_records.pkl")
        b = res["best"]
        from aiengine import rul as rulm
        sc = []
        for fh, g in recs.groupby("fold_end_hi"):
            th, pooled = rulm.fit_thresholds(json.loads(fh), quantile=b["q"])
            c = rulm.RULCalibration(method=b["method"], history_windows=b["history"], blend_w=b["blend_w"],
                                    u_min=b["u_min"], class_thresholds=th, pooled_threshold=pooled,
                                    loglin_coef=b.get("loglin_coef", []))
            sc.append(train.apply_rul(c, g))
        sc = pd.concat(sc)
        d = sc[sc["in_deg"] & (sc["rul_true"] > 0)]
        res["interval_rel"] = float(np.quantile(np.abs(d["rul_pred"] - d["rul_true"]) / d["rul_true"], 0.8))
        res["stage_within1_oof"] = float((np.abs(sc["stage_pred"] - sc["health_stage"]) <= 1).mean())
        p = select.write("rul", res, meta)
        print(f"best: {b}  -> {p}")
    else:
        grid = []
        for L in args.tft_encoders:
            for pw in args.weight_powers:
                grid.append({"max_encoder_length": L, "class_weight_power": pw, "max_epochs": args.epochs})
        res = select.run_tft(df[df["dataset"] != "ims"], grid, run / "select_tft")
        p = select.write("tft", {"grid": res}, meta)
        best = max(res, key=lambda r: r["val_macro_f1"])
        print(f"best: {best['config']} val macro-F1 {best['val_macro_f1']:.3f} -> {p}")
    return 0


def cmd_robustness(args) -> int:
    """SECONDARY: LOBO over all eligible XJTU-SY run-to-failure bearings (see aiengine/robustness.py)."""
    from aiengine import models, robustness, select, train
    train.set_threads(args.threads)
    names = _datasets(args)
    df = train.prepare(names)
    run = _run_dir(args)
    work = run / "robustness_lobo"
    nc = models.NHiTSConfig(encoder_length=args.nhits_encoder, prediction_length=args.nhits_horizon)
    tc = models.TFTConfig(max_encoder_length=args.tft_encoder)
    units = robustness.eligible(df, nc)
    if args.list:
        print(" ".join(str(i) for i in range(len(units))))
        return 0
    if args.fold is not None:
        r = robustness.run_fold(df, units[args.fold], args.fold, nc, tc, work, args.epochs)
        print(json.dumps({k: v for k, v in r.items() if k != "selected_in_fold"}, default=str))
        return 0
    agg = robustness.aggregate(work)
    p = select.write("robustness_lobo", agg, _meta(args, names, df))
    print(f"{agg['n_folds']} folds, mean S7 {agg['mean_S7_mape_pct']:.1f} %, median {agg['median_S7_mape_pct']:.1f} % -> {p}")
    for r in agg["folds"]:
        print(f"  {r['unit_id']:<22} {r['fault_class']:<11} {r['frozen_split']:<6} S7 {r['S7_mape_pct']:6.1f} %  "
              f"stage<=1 {r['stage_within1_frac']:.2f} max {r['stage_max_error']}  [{r['selected_in_fold']['method']}]")
    return 0


def cmd_finetune(args) -> int:
    from aiengine import train
    train.set_threads(args.threads)
    train.seed_all()
    names = _datasets(args)
    if "ims" not in names:
        print("finetune needs the ims cache (fine-tuning demonstration domain).")
        return 1
    df = train.prepare(names)
    run = _run_dir(args)
    ims = df[df["dataset"] == "ims"]
    ftr = ims[ims["split"] == "ft_train"]
    meta = {"provenance": _meta(args, names, df)}
    _, meta["nhits_ft"] = train.finetune(run / "nhits.ckpt", train.rtf_frames(ftr), "nhits", run,
                                         lr_factor=args.lr_factor, freeze_encoder=args.freeze,
                                         max_epochs=args.epochs)
    tft_pre = train.load_model(run / "tft.ckpt")
    if train.is_hierarchical(tft_pre):
        from aiengine import hier
        ft_cls = sorted(set(hier.post_onset(ftr)["observable_class"].astype(str)))
        if len(ft_cls) < 2:
            # A fault-type head fine-tuned on a single fault class collapses onto it; keep the
            # pretrained fault-type TFT (the onset gate is rule-based and needs no fine-tuning).
            (run / "tft_ft.ckpt").write_bytes((run / "tft.ckpt").read_bytes())
            meta["tft_ft"] = {"skipped": True, "reason": f"ft_train post-onset classes {ft_cls}: < 2 fault classes; "
                              "tft_ft.ckpt = pretrained hierarchical TFT"}
        else:
            _, meta["tft_ft"] = train.finetune(run / "tft.ckpt", hier.post_onset(ftr), "tft", run,
                                               lr_factor=args.lr_factor, freeze_encoder=args.freeze,
                                               max_epochs=args.epochs)
    else:
        _, meta["tft_ft"] = train.finetune(run / "tft.ckpt", train.tft_frames(ftr), "tft", run,
                                           lr_factor=args.lr_factor, freeze_encoder=args.freeze,
                                           max_epochs=args.epochs)
    # RUL thresholds re-fitted on the new domain's TRAINING bearings (part of fine-tuning)
    nh = train.load_model(run / "nhits_ft.ckpt")
    tft = train.load_model(run / "tft_ft.ckpt")
    recs = train.rul_records(nh, ftr, train.tft_predict(tft, ftr))
    base = json.loads((run / "rul_calibration.json").read_text())
    th, pooled = train.rul.fit_thresholds(train.end_hi_records(ftr))
    base.update({"class_thresholds": {**base["class_thresholds"], **th}, "pooled_threshold": pooled if th else base["pooled_threshold"],
                 "fit_split": "ims ft_train (thresholds only)"})
    (run / "rul_calibration_ft.json").write_text(json.dumps(base, indent=2))
    meta["ft_train_records"] = int(len(recs))
    train.write_json(run / "finetune_meta.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "provenance"}, indent=1, default=str))
    return 0


def cmd_evaluate(args) -> int:
    from aiengine import evaluate, train
    train.set_threads(args.threads)
    names = _datasets(args)
    df = train.prepare(names)
    run = _run_dir(args)
    meta = _meta(args, names, df)
    pre = df[df["dataset"] != "ims"]
    m, preds, scored = evaluate.score(run, pre, "test", latency_n=args.latency_n)
    m["source_check"] = evaluate.source_check(df[df["split"].isin(["train", "ft_train"])])
    if args.note:
        m["disclosure"] = args.note
    extra = {}
    scored_all, preds_all, truth_all = [scored], [preds], [pre[pre["split"] == "test"]]
    if "ims" in names and (run / "nhits_ft.ckpt").exists():
        ims = df[df["dataset"] == "ims"]
        mf, pf, sf = evaluate.score(run, ims, "ft_test", tft_ckpt=run / "tft_ft.ckpt",
                                   nhits_ckpt=run / "nhits_ft.ckpt", calib_file=run / "rul_calibration_ft.json",
                                   latency_n=0)
        m0, _, _ = evaluate.score(run, ims, "ft_test", latency_n=0)
        def _cls(c):
            return {"macro_f1_testable": c.get("macro_f1"), "testable_classes": c.get("macro_f1_testable_classes"),
                    "macro_f1_all5": c.get("macro_f1_all5"), "untestable": c.get("untestable_note"),
                    "per_dataset_recall": {k: v.get("recall") for k, v in c.get("per_dataset", {}).items()},
                    "note": "IMS B4 = ball only after onset; ball has no output channel, so the testable set is "
                            "{healthy} alone and macro_f1_testable is NOT informative here; read macro_f1_all5/recall."}
        extra["ims_finetuned"] = {"rul": mf.get("rul"), "classification": _cls(mf["classification"]),
                                  "detection": {k: v for k, v in mf["detection"].items() if k != "per_unit"}}
        extra["ims_zero_shot_pretrained"] = {"rul": m0.get("rul"), "classification": _cls(m0["classification"])}
        scored_all.append(sf)
        preds_all.append(pf)
        truth_all.append(ims[ims["split"] == "ft_test"])
    allsc = pd.concat([s for s in scored_all if len(s)], ignore_index=True) if any(len(s) for s in scored_all) else pd.DataFrame()
    if len(allsc):
        primary = m.get("rul", {})
        m["rul"] = evaluate.rul_metrics(allsc)
        m["rul"]["note"] = ("Headline S7 = all held-out run-to-failure test bearings, each scored by the model "
                            "deployed for its domain (pretrained for xjtu_sy/synthetic, fine-tuned for ims).")
        extra["pretrain_test_only_rul"] = {k: primary.get(k) for k in ("S7_mape_pct", "per_bearing_mape_pct")}
    if len(preds_all) > 1:
        P = pd.concat([p for p in preds_all if len(p)], ignore_index=True)
        T = pd.concat(truth_all, ignore_index=True)
        extra["pretrain_test_only_classification"] = {k: m["classification"].get(k) for k in ("macro_f1", "n")}
        m["classification"] = evaluate.classification_metrics(P, T)
        m["detection"] = evaluate.caught_by_stage3(P, T)
        preds = P
    rob = sorted(Path(evaluate.REPORTS_DIR).glob("model_selection_robustness_lobo_*.json"))
    if args.robustness and rob:
        r = json.loads(rob[-1].read_text())
        extra["SECONDARY_xjtu_lobo_robustness"] = {
            "file": rob[-1].name, "note": r["note"], "mean_S7_mape_pct": r["mean_S7_mape_pct"],
            "median_S7_mape_pct": r["median_S7_mape_pct"],
            "frac_bearings_meeting_15pct": r["frac_bearings_meeting_15pct"],
            "per_bearing": {f["unit_id"]: {"S7_mape_pct": f["S7_mape_pct"], "split": f["frozen_split"],
                                           "class": f["fault_class"], "stage_max_error": f["stage_max_error"],
                                           "stage_within1_frac": f["stage_within1_frac"]} for f in r["folds"]}}
    m["extra_tracks"] = extra
    jp, mp = evaluate.write_report(m, meta, tag=args.tag)
    T_all = pd.concat(truth_all, ignore_index=True)
    tcols = ["unit_id", "window_index", "dataset", "t_hours", "observable_class", "fault_class",
             "health_stage", "rul_hours", "in_degradation_window", "split"]
    preds = preds.merge(T_all[tcols], on=["unit_id", "window_index"], how="left")
    preds["model"] = np.where(preds["dataset"] == "ims", "finetuned", "pretrained")
    preds.to_csv(Path(jp).with_suffix(".tft_preds.csv.gz"), index=False)
    if len(allsc):
        allsc.drop(columns=["hi_hist", "forecast"]).to_csv(Path(jp).with_suffix(".rul_preds.csv.gz"), index=False)
    print(Path(mp).read_text(encoding="utf-8"))
    return 0


def cmd_export(args) -> int:
    from aiengine import export, train
    names = _datasets(args)
    df = train.prepare(names)
    run = _run_dir(args)
    res = export.run(run, df[df["dataset"] != "ims"], provenance=_meta(args, names, df))
    print(json.dumps(res, indent=1, default=str))
    ok = res["tft"]["parity_ok"] and res["nhits"]["parity_ok"] and \
        res["tft"]["builder_vs_dataset_max_abs_diff"] < 1e-4 and res["nhits"]["builder_vs_dataset_max_abs_diff"] < 1e-4
    print("EXPORT", "OK" if ok else "PARITY PROBLEM")
    return 0 if ok else 1


def cmd_demo_payloads(args) -> int:
    from aiengine import demo_stream, train
    df = train.prepare(_datasets(args))
    print(json.dumps(demo_stream.build(df, args.bearing1, args.bearing2), indent=1))
    return 0


def cmd_demo(args) -> int:
    from aiengine import payload, train
    from aiengine.config import BEARING_FEATURES, MODELS_DIR
    from aiengine.engine import HybridEngine
    names = _datasets(args)
    df = train.prepare(names)
    test = df[df["split"].isin(["test", "ft_test"])]
    if args.unit:
        uid = args.unit
    else:
        r = test[test["run_to_failure"]]
        uid = (r if len(r) else test).groupby("unit_id").size().idxmax()
    unit = df[df["unit_id"] == uid].sort_values("window_index")
    eng = HybridEngine(MODELS_DIR)
    prov = "SYNTHETIC" if unit["dataset"].iloc[0] == "synthetic_rig" else f"REAL ({unit['dataset'].iloc[0]})"
    print(f"Replaying {uid} [{prov}] split={unit['split'].iloc[0]} true class={unit['fault_class'].iloc[0]} "
          f"windows={len(unit)}  (both payload bearings carry this unit's 16 features)")
    print(f"{'win':>5} {'t_h':>8} {'stg':>3} {'HI':>6} {'class':>11} {'conf':>5} {'RUL h':>8} {'true h':>8} {'ms':>6}")
    step = max(1, len(unit) // args.lines)
    for i, r in enumerate(unit.itertuples(index=False)):
        f16 = np.array([getattr(r, c) for c in BEARING_FEATURES], np.float32)
        w = payload.Window(rpm=float(r.rpm), features=np.concatenate([f16, f16]), p1_amp_um=0.0,
                           p1_phase_rad=0.0, p2_amp_um=0.0, p2_phase_rad=0.0,
                           t20_ms=int(round(r.t_hours * 3.6e6)) % 2 ** 32)
        out = eng.process_payload(payload.encode(w))["bearings"][0]
        if out.get("status") != "ok" or (i % step and i != len(unit) - 1):
            continue
        print(f"{r.window_index:>5} {r.t_hours:>8.2f} {r.health_stage:>3} {out['health_index']:>6.3f} "
              f"{out['fault_class']:>11} {out['confidence']:>5.2f} {out['rul_hours']:>8.2f} "
              f"{(r.rul_hours if np.isfinite(r.rul_hours) else float('nan')):>8.2f} {out['latency_ms']['total']:>6.2f}")
        if args.pause:
            time.sleep(args.pause)
    print("last output JSON:\n" + json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="app.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(s, run=True):
        s.add_argument("--datasets", nargs="+", help="dataset caches to use (default: all real present, else synthetic_rig)")
        if run:
            s.add_argument("--run", default="main", help="checkpoint run name (checkpoints/<run>/)")
        s.add_argument("--threads", type=int, default=12, help="torch threads for training")

    common(sub.add_parser("selftest", help="checks needing no trained model"), run=False)
    common(sub.add_parser("data", help="prepare tables and frozen splits"), run=False)
    t = sub.add_parser("train", help="pretrain TFT + N-HiTS and select the RUL estimator")
    common(t)
    t.add_argument("--epochs", type=int)
    t.add_argument("--tft-encoder", type=int, default=6)
    t.add_argument("--tft-inputs", help="comma-separated engineered inputs for TFT")
    t.add_argument("--nhits-encoder", type=int, default=24)
    t.add_argument("--nhits-horizon", type=int, default=12)
    t.add_argument("--skip-tft", action="store_true")
    t.add_argument("--skip-nhits", action="store_true")
    t.add_argument("--rul-from", help="model_selection_rul_*.json to take the RUL estimator from")
    t.add_argument("--class-weight-power", type=float, default=1.0)
    t.add_argument("--hierarchical", action="store_true", help="v2: onset gate + fault-type TFT (post-onset windows)")
    t.add_argument("--balance", action="store_true", help="v2: bearing-balanced sampling (replicate short bearings)")
    t.add_argument("--nhits-covariates", help="comma-separated N-HiTS covariates ('' = none)")
    sl = sub.add_parser("select", help="model selection on train+val only (LOBO for RUL, val grid for TFT)")
    common(sl)
    sl.add_argument("what", choices=["rul", "tft"])
    sl.add_argument("--epochs", type=int, default=12)
    sl.add_argument("--nhits-encoder", type=int, default=24)
    sl.add_argument("--nhits-horizon", type=int, default=12)
    sl.add_argument("--nhits-covariates")
    sl.add_argument("--tft-encoders", type=int, nargs="+", default=[4, 6, 8])
    sl.add_argument("--weight-powers", type=float, nargs="+", default=[0.5, 1.0])
    rb = sub.add_parser("robustness", help="SECONDARY LOBO over all eligible XJTU-SY bearings")
    common(rb)
    rb.add_argument("--fold", type=int)
    rb.add_argument("--list", action="store_true")
    rb.add_argument("--epochs", type=int, default=12)
    rb.add_argument("--tft-encoder", type=int, default=6)
    rb.add_argument("--nhits-encoder", type=int, default=24)
    rb.add_argument("--nhits-horizon", type=int, default=12)
    f = sub.add_parser("finetune", help="fine-tune on IMS ft_train bearings")
    common(f)
    f.add_argument("--epochs", type=int, default=15)
    f.add_argument("--lr-factor", type=float, default=0.2)
    f.add_argument("--freeze", action="store_true", help="freeze early layers")
    e = sub.add_parser("evaluate", help="score test units, write reports/")
    common(e)
    e.add_argument("--latency-n", type=int, default=300)
    e.add_argument("--tag", default="")
    e.add_argument("--robustness", action="store_true", help="attach the latest LOBO robustness file")
    e.add_argument("--note", default="", help="disclosure line printed on the report face")
    x = sub.add_parser("export", help="ONNX + contract + golden vectors to models/")
    common(x)
    d = sub.add_parser("demo", help="stream a test unit through HybridEngine")
    common(d, run=False)
    d.add_argument("--unit")
    d.add_argument("--lines", type=int, default=40)
    d.add_argument("--pause", type=float, default=0.0)
    dp = sub.add_parser("demo-payloads", help="write models/demo_payloads.bin (real held-out replay stream)")
    common(dp, run=False)
    dp.add_argument("--bearing1", default="xjtu_sy:Bearing2_5")
    dp.add_argument("--bearing2", default="xjtu_sy:Bearing3_4")
    a = p.parse_args(argv)
    if not hasattr(a, "run"):
        a.run = "main"
    return {"selftest": cmd_selftest, "data": cmd_data, "train": cmd_train, "finetune": cmd_finetune,
            "evaluate": cmd_evaluate, "export": cmd_export, "demo": cmd_demo, "select": cmd_select,
            "robustness": cmd_robustness, "demo-payloads": cmd_demo_payloads}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
