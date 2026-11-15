"""[SIMULATION] score the exported v4_real classifier on the HELD-OUT simulated test runs (synth_coe_node, split == "test").

Those runs (their own simulated bearings / units, registered in splits.json by run BEFORE the v3d/v3e fits) were never in
any training set, so this is a proper held-out-by-unit test, but of SIMULATED data: provenance [SIMULATION calibrated to the rig
healthy data]. It is NOT a rig measurement. B2X := B2Y policy applied exactly as in training. Scored once, no tuning.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import coe_adapt as A
from . import coe_dataset as D
from . import features as feat
from . import train_real as T_
from . import up2_window as uw
from .config import MODELS_DIR, REPORTS_DIR


def main(contract="v4_real", split="test"):
    from sklearn.metrics import confusion_matrix, f1_score
    df = D.generate()
    df = df[df["split"] == split].copy()
    b2 = df["bearing"].to_numpy() == 2
    for c in [c for c in df.columns if c.startswith("x_")]:
        df.loc[b2, c] = df.loc[b2, "y_" + c[2:]].to_numpy()
    eng = feat.engineer_table(df, extended=True)
    eng["observable_class"] = eng["observable_class"].astype(str)
    Bk, idx = A.blocks_for(eng, columns=T_.V3E)
    from . import up2
    t = up2.merge_truth(idx, eng, ["observable_class", "transition", "severity", "rpm"])
    keep = ~t["transition"].astype(bool).to_numpy()
    Bk, t = Bk[keep], t[keep].reset_index(drop=True)
    y = t["observable_class"].map({c: i for i, c in enumerate(T_.CLASS_ORDER)}).to_numpy()
    P = A.onnx_probs(MODELS_DIR / f"{contract}_cls.onnx", Bk).argmax(1)
    labs = list(range(len(T_.CLASS_ORDER)))
    f1 = f1_score(y, P, labels=labs, average=None, zero_division=0)
    out = {"tag": f"[SIMULATION] held-out simulated {split} runs, NOT rig data", "contract": contract, "split": split, "windows": int(len(y)),
           "units": int(t["unit_id"].nunique()), "macro_f1": float(f1.mean()), "accuracy": float((y == P).mean()),
           "f1": {c: float(f1[i]) for i, c in enumerate(T_.CLASS_ORDER)},
           "confusion_labels": T_.CLASS_ORDER, "confusion": confusion_matrix(y, P, labels=labs).tolist()}
    for r in (1800, 3600):
        m = np.isclose(t["rpm"].to_numpy(float), r, rtol=0.05)
        if m.any():
            out[f"macro_f1_{r}rpm"] = float(f1_score(y[m], P[m], labels=sorted(set(y[m])), average="macro", zero_division=0))
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=1))
    (REPORTS_DIR / f"v4_sim_{contract}_{split}_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:3]))
