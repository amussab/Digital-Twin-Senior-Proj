"""XJTU-SY run-to-failure adapter (15 bearings, 3 conditions).

Raw: data/raw/xjtu_sy/<condition>/Bearing<c>_<n>/<k>.csv ; 32768 rows x 2 columns
(col 0 = horizontal = X, col 1 = vertical = Y), fs 25.6 kHz, one 1.28 s snapshot per minute.
Keeps the FIRST 20-rev window of each snapshot (DESIGN.md section 3).

[CITED] rpm per condition 2100/2250/2400 (35/37.5/40 Hz); fault classes from Wang, Lei, Li, Li,
IEEE Trans. Reliability 69(1), 2020, failure-position table; geometry in geometry.py.
Bearing3_2 (outer+inner+ball+cage) and Bearing1_5 (inner+outer) -> 'mixed'.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .. import dsp
from ..config import RAW_DIR
from ..geometry import BEARING_ORDERS
from . import common

DATASET = "xjtu_sy"
FS = 25_600.0
SNAPSHOT_MIN = 1.0  # one snapshot per minute
CONDITIONS = {1: ("35Hz12kN", 2100.0), 2: ("37.5Hz11kN", 2250.0), 3: ("40Hz10kN", 2400.0)}
FAULT = {
    "1_1": "outer_race", "1_2": "outer_race", "1_3": "outer_race", "1_4": "cage", "1_5": "mixed",
    "2_1": "inner_race", "2_2": "outer_race", "2_3": "cage", "2_4": "outer_race", "2_5": "outer_race",
    "3_1": "outer_race", "3_2": "mixed", "3_3": "inner_race", "3_4": "inner_race", "3_5": "outer_race",
}


def _task(args):
    path, rpm = args
    cfg = common.pipeline(FS, BEARING_ORDERS[DATASET])
    span = dsp.cut_windows(10**9, rpm, FS, max_windows=1)[0][1]
    a = pd.read_csv(path, nrows=span).to_numpy(dtype=np.float64)
    if a.shape != (span, 2):
        raise ValueError(f"{path}: unexpected shape {a.shape}")
    return dsp.bearing_features(a[:, 0], a[:, 1], rpm, cfg)


def build(workers: int = 16) -> pd.DataFrame:
    t0 = time.time()
    root = RAW_DIR / "xjtu_sy"
    if (root / "XJTU-SY_Bearing_Datasets").is_dir():  # HuggingFace mirror nests one level
        root = root / "XJTU-SY_Bearing_Datasets"
    cfg = common.pipeline(FS, BEARING_ORDERS[DATASET])
    jobs, metas = [], []
    for cond, (cdir, rpm) in CONDITIONS.items():
        for n in range(1, 6):
            d = root / cdir / f"Bearing{cond}_{n}"
            if not d.is_dir():
                raise FileNotFoundError(d)
            files = sorted(d.glob("*.csv"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
            nums = [int(p.stem) for p in files]
            if nums != list(range(1, len(nums) + 1)):
                raise ValueError(f"{d}: file numbering not consecutive 1..N")
            key = f"Bearing{cond}_{n}"
            meta = common.UnitMeta(DATASET, f"{DATASET}:{key}", f"{DATASET}:{key}",
                                   FAULT[f"{cond}_{n}"], True,
                                   life_hours=len(files) * SNAPSHOT_MIN / 60.0)
            for i, p in enumerate(files):
                jobs.append((str(p), rpm))
                metas.append((meta, i, i * SNAPSHOT_MIN / 60.0, rpm))
    print(f"[xjtu_sy] {len(jobs)} snapshots, {workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, feats in enumerate(ex.map(_task, jobs, chunksize=16)):
            m, i, t, rpm = metas[k]
            rows.append(common.row(m, i, t, rpm, feats, cfg))
            if (k + 1) % 1000 == 0:
                print(f"[xjtu_sy] {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    frame = common.validate(pd.DataFrame(rows))
    common.save(frame, DATASET)
    print(f"[xjtu_sy] done {len(frame)} rows in {time.time() - t0:.0f}s", flush=True)
    return frame
