"""UP2: the spec-named ICS models -- TFT (fault class) + N-HiTS (HI forecast -> RUL) from pytorch-forecasting --
trained on the SAME SYNTHETIC synth_v3 runs and frozen splits as the gradient-boosted heads. [SYNTHETIC]

    python -m aiengine.up2_nn fit --epochs 12 --threads 8     # train on train runs, early stop on val runs
    python -m aiengine.up2_nn predict                         # val (calibration) + ONE pass over test (+ robustness sets)
    python -m aiengine.up2_nn export                          # models/v3_nn_{tft,nhits}.onnx + model_contract_v3_nn.json

RUL from N-HiTS follows the v2 procedure unchanged (rul.py: HI forecast -> threshold crossing, estimator chosen
by train.fit_rul on VALIDATION bearings, thresholds from TRAINING bearings' end-of-life HI). Stage = rul.stage_estimate
(tau from the causal onset detector). TFT = flat 5-class softmax, no gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import up2
from .config import MODELS_DIR, REPORTS_DIR, SEED

NN_DIR = up2.RUN_DIR / "nn"


def nn_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["in_degradation_window"] = d["in_deg"]
    d["in_degradation_full"] = d["in_deg_full"]
    d["cls_usable"] = True
    return d


def cmd_fit(a) -> int:
    from . import models, train
    from .up2_train import load_main, meta
    train.set_threads(a.threads)
    train.seed_all(SEED)
    df = nn_frame(load_main())
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    if a.max_train_runs:
        # CPU budget: a seeded subset of TRAINING runs (by run, never by window), stratified by fault class
        runs = tr.groupby("bearing_key")["fault_class"].agg(lambda s: "|".join(sorted(set(s) - {"healthy"})) or "healthy")
        rng = np.random.default_rng(SEED)
        keep = []
        for _, g in runs.groupby(runs):
            ks = sorted(g.index)
            rng.shuffle(ks)
            keep += ks[:max(1, round(len(ks) * a.max_train_runs / len(runs)))]
        tr = tr[tr["bearing_key"].isin(keep)]
    NN_DIR.mkdir(parents=True, exist_ok=True)
    out = {"meta": meta({"train_runs_used": int(tr["bearing_key"].nunique()), "val_runs": int(va["bearing_key"].nunique())})}
    if not a.skip_tft:
        cfg = models.TFTConfig(max_encoder_length=a.tft_encoder, max_epochs=a.epochs, class_weight_power=0.5,
                               batch_size=512)
        p, m = train.train_tft(tr, va, cfg, NN_DIR)
        out["tft"] = m
        print("tft", json.dumps({k: v for k, v in m.items() if k != "config"}, default=str), flush=True)
    if not a.skip_nhits:
        nc = models.NHiTSConfig(encoder_length=24, prediction_length=12, max_epochs=a.epochs, batch_size=512)
        p, m = train.train_nhits(train.rtf_frames(tr), train.rtf_frames(va), nc, NN_DIR)
        out["nhits"] = m
        print("nhits", json.dumps({k: v for k, v in m.items() if k != "config"}, default=str), flush=True)
    (NN_DIR / "fit_meta.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def _predict_set(tft, nh, calib, d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from . import train
    tp = train.tft_predict(tft, d)
    recs = train.rul_records(nh, d[d["run_to_failure"]], tp)
    sc = train.apply_rul(calib, recs)
    rp = sc[["unit_id", "window_index", "rul_pred", "stage_pred"]].rename(columns={"rul_pred": "rul_pred_h"})
    return tp[["unit_id", "window_index", "pred_class"] + [c for c in tp.columns if c.startswith("p_")]], rp


def cmd_predict(a) -> int:
    from . import rul as rulmod
    from . import train
    from .up2_train import ROBUST_SETS, load_main, load_robust, meta, now_tag
    train.set_threads(a.threads)
    df = nn_frame(load_main())
    tft = train.load_model(NN_DIR / "tft.ckpt")
    nh = train.load_model(NN_DIR / "nhits.ckpt")
    va = df[df["split"] == "val"]
    # RUL estimator calibration on VALIDATION bearings (thresholds from TRAINING bearings' end-of-life HI)
    tpv = train.tft_predict(tft, va)
    recs_v = train.rul_records(nh, va[va["run_to_failure"]], tpv)
    calib, sel = train.fit_rul(recs_v, train.end_hi_records(df[df["split"] == "train"]))
    (NN_DIR / "rul_calibration.json").write_text(json.dumps(calib.to_dict(), indent=1))
    tag = now_tag()
    pred_dir = REPORTS_DIR / "up2_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    sets = {"test": df[df["split"] == "test"], **{n: nn_frame(load_robust(n)) for n in ROBUST_SETS}}
    for name, d in sets.items():
        cp, rp = _predict_set(tft, nh, calib, d)
        f1, f2 = pred_dir / f"up2_v3_nn_{name}_{tag}.cls.csv.gz", pred_dir / f"up2_v3_nn_{name}_{tag}.rul.csv.gz"
        cp.to_csv(f1, index=False, float_format="%.6g")
        rp.to_csv(f2, index=False, float_format="%.6g")
        files[name] = {"cls": f1.name, "rul": f2.name}
        print(name, "done", flush=True)
    idx = {"meta": meta({"tag": tag, "rul_selection_on": "validation bearings", "rul_selected": sel["selected"],
                         "rul_val_mape": sel["fit_mape"]}), "files": files}
    (pred_dir / f"up2_v3_nn_{tag}.index.json").write_text(json.dumps(idx, indent=1, default=float))
    (NN_DIR / "latest_predict.json").write_text(json.dumps({"tag": tag, "files": files}, indent=1))
    return 0


def cmd_export(a) -> int:
    from . import export
    from .up2_train import load_main, meta
    df = nn_frame(load_main())
    tmp = NN_DIR / "export"
    tmp.mkdir(parents=True, exist_ok=True)
    prov = {"synthetic": True, "datasets": ["synth_v3"], "run": "up2_v3_nn", "git_commit": meta()["git_commit"],
            "note": "[SYNTHETIC] TFT + N-HiTS trained on synth_v3; NOT the deployed v3 heads (see model_contract_v3.json)"}
    res = export.run(NN_DIR, df, out_dir=tmp, provenance=prov, calib_file=NN_DIR / "rul_calibration.json")
    c = json.loads((tmp / "model_contract.json").read_text())
    c["contract_version"] = 3
    c["models"]["tft"]["onnx"] = "v3_nn_tft.onnx"
    c["models"]["nhits"]["onnx"] = "v3_nn_nhits.onnx"
    shutil.copy(tmp / "tft.onnx", MODELS_DIR / "v3_nn_tft.onnx")
    shutil.copy(tmp / "nhits.onnx", MODELS_DIR / "v3_nn_nhits.onnx")
    (MODELS_DIR / "model_contract_v3_nn.json").write_text(json.dumps(c, indent=2, default=float))
    g = json.loads((tmp / "golden_vectors.json").read_text())
    g["provenance"] = "[SYNTHETIC] synth_v3 test run"
    (MODELS_DIR / "golden_vectors_v3_nn.json").write_text(json.dumps(g, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "contract"}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2_nn")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--epochs", type=int, default=12)
    f.add_argument("--threads", type=int, default=8)
    f.add_argument("--tft-encoder", type=int, default=12)
    f.add_argument("--skip-tft", action="store_true")
    f.add_argument("--max-train-runs", type=int, default=0, help="seeded, class-stratified subset of training runs (CPU budget)")
    f.add_argument("--skip-nhits", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--threads", type=int, default=8)
    sub.add_parser("export")
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "predict": cmd_predict, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
