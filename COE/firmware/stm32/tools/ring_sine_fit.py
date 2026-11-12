"""Fit the main sine wave in each ring snapshot (from tools/ring_snapshots.ps1).

Prints frequency and peak-to-peak amplitude per snapshot, and, with --steps,
a gain table relative to the first step. Slot 1 = AIN2 (B1X); the ring holds
int16 frames [4096][8], +/-5 V range.

    uv run --no-sync python ../coe_firmware/tools/ring_sine_fit.py ../coe_firmware/build/logs/ring --steps 100 1000 5000 8000 10000 12000 14000
(run from coe_pipeline, which has numpy)
"""
import argparse
import glob
import os
import re

import numpy as np

FS = 29998.5
VOLTS_PER_COUNT = 10.0 / 65536


def fit(samples):
    """Return (frequency, peak-to-peak, residual) for the cleanest 1024-sample segment."""
    best = None
    for k in range(len(samples) // 1024):
        s = samples[k * 1024:(k + 1) * 1024]
        t = np.arange(len(s)) / FS
        spec = np.abs(np.fft.rfft((s - s.mean()) * np.hanning(len(s)), 1 << 16))
        f0 = np.fft.rfftfreq(1 << 16, 1 / FS)[1 + np.argmax(spec[1:])]
        basis = np.column_stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t), np.ones(len(s))])
        c, *_ = np.linalg.lstsq(basis, s, rcond=None)
        residual = (s - basis @ c).std()
        if best is None or residual < best[2]:  # avoids the segment holding the ring wrap point
            best = (f0, 2 * np.hypot(c[0], c[1]), residual)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--steps", type=float, nargs="*", help="generator frequencies, first = reference")
    args = parser.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.folder, "snap_*.bin")), key=os.path.getmtime):
        data = np.fromfile(path, dtype="<i2").reshape(-1, 8)[:, args.slot] * VOLTS_PER_COUNT
        f0, pp, residual = fit(data)
        rows.append((f0, pp))
        print("%-28s %9.1f Hz  %.3f V pp  fit error %.4f V" % (os.path.basename(path), f0, pp, residual))
    if args.steps:
        rows = np.array(rows)
        table = []
        for step in args.steps:
            near = np.abs(rows[:, 0] - step) < 0.02 * step + 20
            table.append((step, np.median(rows[near, 1]) if near.any() else np.nan, near.sum()))
        ref = table[0][1]
        print("\nInput Hz   Output V pp   Gain    dB   (snapshots)")
        for step, pp, count in table:
            print("%8.0f   %10.3f   %5.3f  %+6.2f   (%d)" % (step, pp, pp / ref, 20 * np.log10(pp / ref), count))


if __name__ == "__main__":
    main()
