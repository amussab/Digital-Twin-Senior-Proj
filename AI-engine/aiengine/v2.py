"""IS3 classification v2: joint TFT on XJTU-SY + IMS ft_train + MaFaulDa (DESIGN.md change log 2026-10-05 v2-MaFaulDa).

Frames (test units are never loaded here):
  fit  = XJTU-SY train + IMS ft_train + MaFaulDa train
  val  = XJTU-SY val + MaFaulDa val          (early stopping AND model selection)
Labels = observable_class. MaFaulDa seeded faults are observable from window 0 (the fault is physically
present for the whole record), so their label is the fault class on every window.
Dataset identity is NOT an input (the TFT sees only the engineered, baseline-normalised inputs + rpm_norm).

Pre-declared selection rule (written before any grid result): the configuration with the highest
POOLED validation macro-F1 over the classes present in validation wins; ties -> lower class-weight
power, then shorter encoder. The onset-gate post-processing (hier.py gate, run-to-failure units only)
is adopted only if it raises that same pooled validation macro-F1.

    python -m aiengine.v2 fit --idx K       (one grid point; run several in parallel)
    python -m aiengine.v2 aggregate         (selection table -> reports/model_selection_tft_v2_<ts>.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import models, train  # noqa: E402
from .config import CHECKPOINT_DIR, SEED  # noqa: E402

DATASETS = ["xjtu_sy", "ims", "mafaulda"]
GRID = [{"max_encoder_length": L, "class_weight_power": p} for L in (4, 6) for p in (0.0, 0.25, 0.5, 1.0)]
WORK = CHECKPOINT_DIR / "real_v2" / "select"
PRE = CHECKPOINT_DIR / "real_v2" / "pre.pkl"


def load_pre() -> pd.DataFrame:
    if PRE.exists():
        return pd.read_pickle(PRE)
    df = train.prepare(DATASETS)
    PRE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(PRE)
    return df


def frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = df[((df["dataset"] == "ims") & (df["split"] == "ft_train"))
             | ((df["dataset"] != "ims") & (df["split"] == "train"))]
    val = df[(df["dataset"] != "ims") & (df["split"] == "val")]
    assert not fit["split"].isin(["test", "ft_test"]).any() and not val["split"].isin(["test", "ft_test"]).any()
    return fit, val


def val_metrics(pred: pd.DataFrame, val: pd.DataFrame) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    j = pred.merge(val[["unit_id", "window_index", "observable_class", "dataset"]], on=["unit_id", "window_index"])
    y, yh = j["observable_class"].astype(str).to_numpy(), j["pred_class"].astype(str).to_numpy()
    pres = sorted(set(y))
    out = {"n": int(len(j)), "classes": pres,
           "val_macro_f1": float(f1_score(y, yh, labels=pres, average="macro", zero_division=0)),
           "per_class_f1": {c: float(f1_score(y, yh, labels=[c], average="macro", zero_division=0)) for c in pres},
           "per_dataset": {}}
    for ds, g in j.groupby("dataset"):
        gy, gp = g["observable_class"].astype(str).to_numpy(), g["pred_class"].astype(str).to_numpy()
        p = sorted(set(gy))
        out["per_dataset"][ds] = {"n": int(len(g)), "macro_f1": float(f1_score(gy, gp, labels=p, average="macro", zero_division=0)),
                                  "labels": sorted(set(gy) | set(gp)),
                                  "confusion": confusion_matrix(gy, gp, labels=sorted(set(gy) | set(gp))).tolist()}
    return out


def gated(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Onset-gate decision on run-to-failure (continuous-monitoring) units only, same rule as the
    runtime engine (contract classes.decision.type = "rtf_onset_gate"): before the causal onset is
    CONFIRMED -> p(healthy) = 1; after it -> the TFT's fault-channel probabilities renormalised to sum
    to 1, p(healthy) = 0. Seeded-fault records (MaFaulDa) keep the flat TFT softmax."""
    from . import hier
    rtf_units = set(df.loc[df["run_to_failure"], "unit_id"])
    p = pred.copy()
    if not rtf_units:
        return p
    g = hier.gate_table(df[df["unit_id"].isin(rtf_units)])
    p = p.merge(g, on=["unit_id", "window_index"], how="left")
    m = p["unit_id"].isin(rtf_units).to_numpy()
    pcols = [c for c in p.columns if c.startswith("p_") and c != "p_fault_gate"]
    faults = [c for c in pcols if c != "p_healthy"]
    p[pcols] = p[pcols].astype(np.float64)
    gate = p["p_fault_gate"].fillna(0.0).to_numpy() > 0.5
    F = p[faults].to_numpy(np.float64)
    F = F / np.maximum(F.sum(1, keepdims=True), 1e-12)
    for i, c in enumerate(faults):
        p.loc[m, c] = np.where(gate[m], F[m, i], 0.0)
    p.loc[m, "p_healthy"] = np.where(gate[m], 0.0, 1.0)
    P = p[pcols].to_numpy(np.float64)
    p["pred_class"] = [pcols[i][2:] for i in P.argmax(1)]
    p["confidence"] = P.max(1)
    return p.drop(columns=["p_fault_gate"])


DECISION_SUFFIX = ".decision.json"


def write_decision(ckpt: Path, decision: str, source: str) -> None:
    Path(str(ckpt) + DECISION_SUFFIX).write_text(json.dumps({"decision": decision, "selected_by": source}))


def read_decision(ckpt) -> str | None:
    f = Path(str(ckpt) + DECISION_SUFFIX)
    return json.loads(f.read_text())["decision"] if f.exists() else None


def fit_one(idx: int, epochs: int) -> dict:
    df = load_pre()
    fit, val = frames(df)
    g = GRID[idx]
    cfg = models.TFTConfig(min_encoder_length=2, max_epochs=epochs, patience=5, **g)
    out = WORK / f"tft_{idx:02d}"
    train.seed_all(SEED)
    path, meta = train.train_tft(fit, val, cfg, out)
    model = train.load_model(path)
    pred = train.tft_predict(model, val)
    pred.to_pickle(out / "val_preds.pkl")
    res = {"idx": idx, "config": g, "ckpt": str(path), "epochs_run": meta["epochs_run"],
           "best_val_loss": meta["best_val_loss"], "train_seconds": meta["train_seconds"],
           "class_order": meta["class_order"], "class_weights": meta["class_weights"],
           "train_samples": meta["train_samples"], "val_samples": meta["val_samples"],
           "flat": val_metrics(pred, val), "flat_plus_rtf_gate": val_metrics(gated(pred, val), val)}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def aggregate() -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(WORK.glob("tft_*/result.json"))]
    cands = []
    for r in rows:
        for variant in ("flat", "flat_plus_rtf_gate"):
            cands.append({"idx": r["idx"], "variant": variant, **r["config"],
                          "val_macro_f1": r[variant]["val_macro_f1"],
                          "per_dataset_f1": {k: v["macro_f1"] for k, v in r[variant]["per_dataset"].items()},
                          "per_class_f1": r[variant]["per_class_f1"]})
    flat = [c for c in cands if c["variant"] == "flat"]
    best_flat = sorted(flat, key=lambda c: (-round(c["val_macro_f1"], 4), c["class_weight_power"], c["max_encoder_length"]))[0]
    gate_same = next(c for c in cands if c["idx"] == best_flat["idx"] and c["variant"] == "flat_plus_rtf_gate")
    winner = gate_same if gate_same["val_macro_f1"] > best_flat["val_macro_f1"] else best_flat
    return {"rule": __doc__.split("Pre-declared selection rule")[1].split("python -m")[0].strip(),
            "candidates": cands, "results": rows, "winner": winner}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "fit", "rescore", "aggregate"])
    ap.add_argument("--idx", type=int)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    train.set_threads(a.threads)
    if a.cmd == "prepare":
        df = load_pre()
        fit, val = frames(df)
        print(fit.groupby(["dataset", "observable_class"]).size().to_string())
        print(val.groupby(["dataset", "observable_class"]).size().to_string())
    elif a.cmd == "rescore":
        # recompute the val metrics from the saved val predictions (no retraining)
        _, val = frames(load_pre())
        for f in sorted(WORK.glob("tft_*/result.json")):
            r = json.loads(f.read_text())
            pred = pd.read_pickle(f.parent / "val_preds.pkl")
            r["flat"], r["flat_plus_rtf_gate"] = val_metrics(pred, val), val_metrics(gated(pred, val), val)
            f.write_text(json.dumps(r, indent=1, default=str))
    elif a.cmd == "fit":
        r = fit_one(a.idx, a.epochs)
        print(json.dumps({k: r[k] for k in ("idx", "config", "epochs_run")}), r["flat"]["val_macro_f1"],
              r["flat_plus_rtf_gate"]["val_macro_f1"])
    else:
        from . import select
        res = aggregate()
        meta = {"seed": SEED, "git_commit": train.git_commit(), "git_dirty": train.git_dirty(),
                "command": train.command_line(), "datasets": DATASETS,
                "data": "[MEASURED on XJTU-SY / IMS / MaFaulDa public datasets]; validation units only, test never loaded"}
        p = select.write("tft_v2", res, meta)
        for c in sorted(res["candidates"], key=lambda c: -c["val_macro_f1"]):
            print(f"{c['idx']:>2} {c['variant']:<19} L{c['max_encoder_length']} p{c['class_weight_power']:<5} "
                  f"val F1 {c['val_macro_f1']:.3f} {json.dumps({k: round(v, 3) for k, v in c['per_dataset_f1'].items()})}")
        print("WINNER", res["winner"], "->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
