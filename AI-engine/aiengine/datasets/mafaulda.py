"""MaFaulDa adapter (UFRJ, SpectraQuest MFS): normal + underhang + overhang packages.

Raw: data/raw/mafaulda/{normal/<f>.csv, <underhang|overhang>/<fault>/[<severity>/]<f>.csv}
250000 x 8 columns, fs 50 kHz [CITED dataset page; shape asserted per record]:
col 0 tachometer, 1-3 underhang accelerometer (axial, radial, tangential), 4-6 overhang
accelerometer (axial, radial, tangential), 7 microphone. X = radial, Y = tangential.
Shaft rpm is estimated per record from tachometer rising edges (1 pulse/rev); the filename (Hz)
is only a sanity check. A fault record emits the unit of the position named by its folder;
`normal` records emit both positions as healthy. All 20-rev windows per record,
run_to_failure=False.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path

import numpy as np
import pandas as pd

from .. import dsp
from ..config import RAW_DIR
from ..geometry import BEARING_ORDERS
from . import common

DATASET = "mafaulda"
FS = 50_000.0
COLS = {"underhang": (2, 3), "overhang": (5, 6)}  # (radial, tangential)
FAULT_MAP = {"ball_fault": "ball", "cage_fault": "cage", "outer_race": "outer_race",
             "inner_race": "inner_race"}


def tach_rpm(tach: np.ndarray, fs: float) -> float:
    lo, hi = np.percentile(tach, 2), np.percentile(tach, 98)
    mid = 0.5 * (lo + hi)
    up = np.flatnonzero((tach[:-1] < mid) & (tach[1:] >= mid))
    if len(up) < 3:
        return float("nan")
    return 60.0 * (len(up) - 1) * fs / (up[-1] - up[0])


def _nominal_rpm(path) -> float:
    """MaFaulDa filenames are the record's rotation frequency in Hz [CITED dataset page]."""
    try:
        return float(Path(path).stem) * 60.0
    except ValueError:
        return float("nan")


def _task(args):
    path, positions = args
    cfg = common.pipeline(FS, BEARING_ORDERS[DATASET])
    a = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if a.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns, got {a.shape}")
    rpm = tach_rpm(a[:, 0], FS)
    nominal = _nominal_rpm(path)
    # The simple mid-level crossing count over-counts on some noisy faulted records
    # (up to ~19k rpm on a 737-3686 rpm rig). When it disagrees with the dataset's own
    # filename frequency by >5%, trust the filename. Decided from metadata, not results.
    if np.isfinite(nominal) and not (np.isfinite(rpm) and abs(rpm - nominal) / nominal <= 0.05):
        rpm = nominal
    out = {}
    for pos in positions:
        cx, cy = COLS[pos]
        out[pos] = [dsp.bearing_features(a[s:e, cx], a[s:e, cy], rpm, cfg)
                    for s, e in dsp.cut_windows(len(a), rpm, FS)]
    return rpm, out


def _jobs():
    root = RAW_DIR / "mafaulda"
    jobs = []  # (path, pos_or_normal, sub, fault, severity, positions)
    for f in sorted((root / "normal").rglob("*.csv")):
        jobs.append((f, "normal", "", "healthy", "", ("underhang", "overhang")))
    for pos in ("underhang", "overhang"):
        for f in sorted((root / pos).rglob("*.csv")):
            rel = f.relative_to(root / pos).parts
            sev = rel[1] if len(rel) == 3 else ""
            jobs.append((f, pos, rel[0], FAULT_MAP[rel[0]], sev, (pos,)))
    return jobs


def build(workers: int = 16) -> pd.DataFrame:
    t0 = time.time()
    jobs = _jobs()
    cfg = common.pipeline(FS, BEARING_ORDERS[DATASET])
    print(f"[mafaulda] {len(jobs)} records", flush=True)
    rows, mism = [], 0
    with ProcessPoolExecutor(workers) as ex:
        for k, (rpm, out) in enumerate(ex.map(_task, [(str(j[0]), j[5]) for j in jobs], chunksize=4)):
            f, top, sub, fault, sev, _ = jobs[k]
            nominal = _nominal_rpm(f)
            if np.isfinite(nominal) and rpm == nominal:
                mism += 1   # tach disagreed; filename frequency used (see _task)
            dur_h = 20.0 * 60.0 / rpm / 3600.0
            for pos, feats in out.items():
                if top == "normal":
                    tag = f"normal/{f.stem}"
                else:
                    tag = f"{pos}/{sub}/{sev}/{f.stem}" if sev else f"{pos}/{sub}/{f.stem}"
                meta = common.UnitMeta(DATASET, f"{DATASET}:{tag}:{pos}",
                                       f"{DATASET}:{pos}:{fault}:{sev}", fault, False, severity=sev)
                for i, v in enumerate(feats):
                    rows.append(common.row(meta, i, i * dur_h, rpm, v, cfg))
            if (k + 1) % 100 == 0:
                print(f"[mafaulda] {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"[mafaulda] tach disagreed with filename by >5%, filename rpm used: {mism} records", flush=True)
    frame = common.validate(pd.DataFrame(rows))
    common.save(frame, DATASET)
    print(f"[mafaulda] done {len(frame)} rows in {time.time() - t0:.0f}s", flush=True)
    return frame
