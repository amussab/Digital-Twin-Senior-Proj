"""IMS / NASA bearing 1st_test adapter (Qiu et al. 2006).

Raw: data/raw/ims/**/1st_test/<YYYY.MM.DD.hh.mm.ss> ; tab-separated 20480 x 8, fs 20 kHz.
Bearing k (1..4) uses columns (2k-2, 2k-1) = (X, Y) [UNVERIFIED which physical axis each is;
the readme only states two accelerometers per bearing]. Shaft 2000 rpm [CITED Qiu 2006].
t_hours from filename timestamps (relative to first file; interval is not uniform).
[CITED readme] 1st test ended with inner-race failure on bearing 3 and roller-element failure on
bearing 4; B1/B2 did not fail -> healthy, run_to_failure=False.
2nd_test/3rd_test have ONE accelerometer per bearing (4 channels), so they cannot form the
biaxial bearing unit and are skipped.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from .. import dsp
from ..config import RAW_DIR
from ..geometry import BEARING_ORDERS
from . import common

DATASET = "ims"
FS = 20_000.0
RPM = 2000.0
UNITS = {1: ("healthy", False), 2: ("healthy", False), 3: ("inner_race", True), 4: ("ball", True)}


def _find_test1():
    hits = [p for p in (RAW_DIR / "ims").rglob("1st_test") if p.is_dir()]
    if not hits:
        raise FileNotFoundError("raw/ims/**/1st_test not found")
    return hits[0]


def _task(path):
    cfg = common.pipeline(FS, BEARING_ORDERS[DATASET])
    span = dsp.cut_windows(10**9, RPM, FS, max_windows=1)[0][1]
    a = pd.read_csv(path, sep=r"\s+", header=None, nrows=span).to_numpy(dtype=np.float64)
    if a.shape != (span, 8):
        raise ValueError(f"{path}: unexpected shape {a.shape}")
    return [dsp.bearing_features(a[:, 2 * k - 2], a[:, 2 * k - 1], RPM, cfg) for k in range(1, 5)]


def build(workers: int = 16) -> pd.DataFrame:
    t0 = time.time()
    d = _find_test1()
    files = sorted(p for p in d.iterdir() if p.is_file())
    ts = [datetime.strptime(p.name, "%Y.%m.%d.%H.%M.%S") for p in files]
    t_h = np.array([(t - ts[0]).total_seconds() / 3600.0 for t in ts])
    life = float(t_h[-1])
    cfg = common.pipeline(FS, BEARING_ORDERS[DATASET])
    print(f"[ims] {len(files)} files, life {life:.1f} h", flush=True)
    metas = {k: common.UnitMeta(DATASET, f"ims:test1:B{k}", f"ims:test1:B{k}", c, r,
                                life_hours=life if r else float("nan"))
             for k, (c, r) in UNITS.items()}
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, feats in enumerate(ex.map(_task, [str(p) for p in files], chunksize=8)):
            for k in range(1, 5):
                rows.append(common.row(metas[k], i, t_h[i], RPM, feats[k - 1], cfg))
            if (i + 1) % 200 == 0:
                print(f"[ims] {i + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    frame = common.validate(pd.DataFrame(rows))
    common.save(frame, DATASET)
    print(f"[ims] done {len(frame)} rows in {time.time() - t0:.0f}s", flush=True)
    return frame
