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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.up2c")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    a = ap.parse_args(argv)
    return {"data": cmd_data}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
