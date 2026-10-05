"""IS3 v2: hierarchical decision + bearing-balanced TFT training, and the LOBO selection harness.

Hierarchical decision (DESIGN.md section 6/8, change log 2026-10-05 v2):
  1. The causal onset detector (labels.OnlineOnset, the same rule that defines the labels) gates
     healthy vs fault: before the onset is CONFIRMED the window is `healthy`.
  2. After confirmation the TFT (spec S8 names TFT) picks the fault type among the fault classes
     it has training bearings for. It is trained on post-onset windows only.
  Output contract kept: class probabilities over [fault channels + healthy], with
  p(healthy) = 1 - P(fault), P(fault) = 1 once the onset is confirmed, else 0; fault probs =
  TFT softmax * P(fault).

Bearing-balanced sampling: each training bearing's frame is replicated (as distinct group ids,
`<unit>#r<k>`) so that bearings contribute comparable numbers of samples (cap `max_rep`).
A single long cage bearing (XJTU-SY Bearing2_3, 406 post-onset windows) otherwise dominates.

Selection harness: leave-one-bearing-out over the XJTU-SY TRAIN+VAL classifiable bearings only
(test bearings and IMS never loaded). One process per (variant, fold) so folds run in parallel.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import labels, models, train  # noqa: E402
from .config import CHECKPOINT_DIR, SEED  # noqa: E402

FAULTS = ("outer_race", "inner_race", "ball", "cage")

VARIANTS = {
    # flat = v1 architecture (TFT classifies healthy + faults on every window)
    "flat_p025": {"hier": False, "power": 0.25, "balance": False},            # v1 config
    "flat_p0": {"hier": False, "power": 0.0, "balance": False},
    "flat_p025_bal": {"hier": False, "power": 0.25, "balance": True},
    # hierarchical = onset gate + fault-type TFT on post-onset windows
    "hier_p0": {"hier": True, "power": 0.0, "balance": False},
    "hier_p05": {"hier": True, "power": 0.5, "balance": False},
    "hier_p0_bal": {"hier": True, "power": 0.0, "balance": True},
    "hier_p05_bal": {"hier": True, "power": 0.5, "balance": True},
}


# --------------------------------------------------------------------------- gate

def confirmed_onset(hi: np.ndarray) -> int | None:
    """Index of the window at which the causal detector CONFIRMS the onset (runtime-equivalent)."""
    det = labels.OnlineOnset()
    for i, h in enumerate(np.asarray(hi, float)):
        if det.update(float(h)) is not None:
            return i
    return None


def gate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per window: p_fault_gate in {0,1} from the causal onset detector."""
    out = []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("window_index")
        c = confirmed_onset(g["hi"].to_numpy())
        idx = g["window_index"].to_numpy()
        pos = np.arange(len(g))
        out.append(pd.DataFrame({"unit_id": uid, "window_index": idx,
                                 "p_fault_gate": (pos >= c).astype(float) if c is not None else 0.0}))
    return pd.concat(out, ignore_index=True)


def compose(tft_preds: pd.DataFrame, gate: pd.DataFrame, fault_order: list[str]) -> pd.DataFrame:
    """TFT fault-type probs + gate -> probs over fault channels + healthy, pred_class, confidence."""
    j = tft_preds.merge(gate, on=["unit_id", "window_index"], how="left")
    pf = j["p_fault_gate"].fillna(0.0).to_numpy()
    for c in fault_order:
        j[f"p_{c}"] = j[f"p_{c}"].to_numpy() * pf
    j["p_healthy"] = 1.0 - pf
    pcols = [f"p_{c}" for c in sorted(fault_order + ["healthy"])]
    P = j[pcols].to_numpy()
    j["pred_class"] = [pcols[i][2:] for i in P.argmax(1)]
    j["confidence"] = P.max(1)
    return j.drop(columns=["p_fault_gate"])


def predict(model, df: pd.DataFrame, hierarchical: bool) -> pd.DataFrame:
    """train.tft_predict composes the gate itself for a hierarchical (no-healthy-channel) model."""
    return train.tft_predict(model, df)


# --------------------------------------------------------------------------- training frames

def post_onset(df: pd.DataFrame) -> pd.DataFrame:
    """Fault-labelled windows only (observable_class != healthy), classifiable units."""
    d = df[df["cls_usable"] & (df["observable_class"].astype(str) != "healthy")]
    return d.copy()


def balance(df: pd.DataFrame, target: int | None = None, max_rep: int = 8) -> pd.DataFrame:
    """Replicate each unit's frame round(target / n) times (1..max_rep) as distinct group ids."""
    sizes = df.groupby("unit_id").size()
    target = int(target or np.median(sizes))
    parts = []
    for uid, g in df.groupby("unit_id"):
        rep = int(np.clip(round(target / len(g)), 1, max_rep))
        for r in range(rep):
            parts.append(g.assign(unit_id=f"{uid}#r{r}") if r else g)
    return pd.concat(parts, ignore_index=True)


def fit(df_train: pd.DataFrame, df_val: pd.DataFrame, variant: dict, run_dir: Path,
        encoder: int = 12, epochs: int = 20, patience: int = 5) -> tuple[Path, dict]:
    tr, va = df_train, df_val
    if variant["hier"]:
        tr, va = post_onset(tr), post_onset(va)
    if variant["balance"]:
        tr = balance(tr)
    cfg = models.TFTConfig(max_encoder_length=encoder, max_epochs=epochs, patience=patience,
                           class_weight_power=variant["power"])
    return train.train_tft(tr, va, cfg, run_dir)


# --------------------------------------------------------------------------- LOBO harness

def lobo_units(pre: pd.DataFrame) -> list[str]:
    fit = pre[(pre["dataset"] == "xjtu_sy") & pre["split"].isin(["train", "val"]) & pre["cls_usable"]]
    return sorted(fit["unit_id"].unique())


def run_fold(pre: pd.DataFrame, variant_name: str, k: int, work: Path, epochs: int, seed: int = SEED) -> Path:
    v = VARIANTS[variant_name]
    units = lobo_units(pre)
    held = units[k]
    fitdf = pre[(pre["dataset"] == "xjtu_sy") & pre["split"].isin(["train", "val"])]
    trn = fitdf[fitdf["unit_id"] != held]
    out = work / variant_name / f"fold_{k:02d}"
    train.seed_all(seed + k)
    # fixed epochs, no early stopping (no held-out data may steer the fold)
    path, meta = fit(trn, trn.iloc[:0], v, out, epochs=epochs)
    model = train.load_model(path)
    p = predict(model, fitdf[fitdf["unit_id"] == held], v["hier"])
    p["fold"] = k
    p.to_pickle(out / "oof.pkl")
    (out / "meta.json").write_text(json.dumps({"held_out": held, **{kk: meta[kk] for kk in ("train_seconds", "epochs_run", "class_order", "class_weights", "train_samples")}}, default=str))
    return out


def aggregate(pre: pd.DataFrame, work: Path) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    truth = pre[["unit_id", "window_index", "observable_class"]]
    res = {}
    for vdir in sorted(p for p in work.iterdir() if p.is_dir() and p.name in VARIANTS):
        files = sorted(vdir.glob("fold_*/oof.pkl"))
        if not files:
            continue
        P = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
        j = P.merge(truth, on=["unit_id", "window_index"])
        y, yh = j["observable_class"].astype(str).to_numpy(), j["pred_class"].astype(str).to_numpy()
        labs = ["healthy", "outer_race", "inner_race", "cage"]
        per_bearing = {}
        for uid, g in j.groupby("unit_id"):
            gy, gp = g["observable_class"].astype(str), g["pred_class"].astype(str)
            f = gy != "healthy"
            per_bearing[uid] = {"fault_recall": float((gp[f] == gy[f]).mean()) if f.any() else None,
                                "healthy_recall": float((gp[~f] == "healthy").mean()) if (~f).any() else None,
                                "fault_pred_counts": gp[f].value_counts().to_dict()}
        res[vdir.name] = {
            "folds": len(files),
            "lobo_macro_f1_4": float(f1_score(y, yh, labels=labs, average="macro", zero_division=0)),
            "lobo_macro_f1_h_o_i": float(f1_score(y, yh, labels=labs[:3], average="macro", zero_division=0)),
            "per_class_f1": {c: float(f1_score(y, yh, labels=[c], average="macro", zero_division=0)) for c in labs},
            "cage_false_calls": int(((yh == "cage") & (y != "cage")).sum()),
            "confusion_labels": labs,
            "confusion": confusion_matrix(y, yh, labels=labs).tolist(),
            "per_bearing": per_bearing,
        }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fold", "aggregate", "list"])
    ap.add_argument("--variant")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--work", default=str(CHECKPOINT_DIR / "real" / "select_is3"))
    ap.add_argument("--pre", default=str(CHECKPOINT_DIR / "v2" / "pre.pkl"))
    a = ap.parse_args(argv)
    train.set_threads(a.threads)
    pre = pd.read_pickle(a.pre) if Path(a.pre).exists() else train.prepare(["xjtu_sy", "ims"])
    pre = pre[(pre["dataset"] != "ims") & pre["split"].isin(["train", "val"])]   # never test, never IMS
    work = Path(a.work)
    if a.cmd == "list":
        print("\n".join(f"{i} {u}" for i, u in enumerate(lobo_units(pre))))
    elif a.cmd == "fold":
        print(run_fold(pre, a.variant, a.fold, work, a.epochs))
    else:
        r = aggregate(pre, work)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_bearing"} for k, v in r.items()}, indent=1))
        (work / "aggregate.json").write_text(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
