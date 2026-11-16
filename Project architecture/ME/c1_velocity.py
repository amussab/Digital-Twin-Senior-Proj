"""C1 live check (ME): bearing-housing vibration velocity in mm/s RMS from the accelerometers.

Reads the raw-sample snapshots that COE's tools/ring_snapshots.ps1 saves while the
STM32 runs in LIVE_RAW mode (136 ms of all 8 ADC slots every 2 s), and prints the
velocity RMS of each accelerometer, 10-1000 Hz (the ISO 10816 band), next to the
C1 limit of 10 mm/s.

Method, per snapshot and channel:
  1. counts -> volts (+/-5 V, 16 bit) -> g (0.100 V/g, TE 805-0050) -> m/s^2
  2. put the ring in time order (oldest sample first), remove the mean, Hann window
  3. FFT; velocity spectrum V(f) = A(f) / (2 pi f)
  4. velocity RMS = sqrt of the summed power of V between 10 and 1000 Hz

Run (needs numpy), in a second terminal while ring_snapshots.ps1 runs:
    python c1_velocity.py ..\\..\\COE\\firmware\\stm32\\build\\logs\\ring
Standard library + numpy only. Calibration is the nominal 100 mV/g (not yet checked
against a known vibration, see the COE README).
"""
import argparse
import glob
import os
import re
import time

import numpy as np

FS = 29998.5                 # AD7606 samples per second per channel
VOLTS_PER_COUNT = 10.0 / 65536
VOLTS_PER_G = 0.100
G = 9.81
SLOTS = {'B1X': 1, 'B1Y': 0, 'B2X': 2, 'B2Y': 3}   # ADC slot of each accelerometer (COE README)
BAND = (10.0, 1000.0)
LIMIT_MM_S = 10.0


def velocity_rms_mm_s(accel_ms2, fs=FS, band=BAND):
    """Velocity RMS (mm/s) of an acceleration record (m/s^2) inside the band."""
    x = accel_ms2 - accel_ms2.mean()
    w = np.hanning(len(x))
    spec = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    keep = (f >= band[0]) & (f <= band[1])
    v = spec[keep] / (2j * np.pi * f[keep])
    power = 2 * np.sum(np.abs(v) ** 2) / (len(x) * np.sum(w ** 2))
    return 1000 * np.sqrt(power)


def read_snapshot(path):
    """Return {channel: acceleration in m/s^2, oldest sample first}."""
    ring = np.fromfile(path, dtype='<i2').reshape(-1, 8)
    match = re.search(r'_(\d+\.\d)s\.bin$', path)
    if match:   # frames written so far tell where the newest sample sits in the ring
        newest = int(round(float(match.group(1)) * FS)) % len(ring)
        ring = np.roll(ring, -newest, axis=0)
    return {name: ring[:, slot] * VOLTS_PER_COUNT / VOLTS_PER_G * G for name, slot in SLOTS.items()}


def report(path, history):
    acc = read_snapshot(path)
    vel = {name: velocity_rms_mm_s(a) for name, a in acc.items()}
    history.append(max(vel.values()))
    del history[:-5]
    worst = max(history)
    verdict = 'within the 10 mm/s limit' if worst <= LIMIT_MM_S else 'ABOVE the 10 mm/s limit'
    print('%-26s ' % os.path.basename(path)
          + '  '.join('%s %5.2f' % (n, v) for n, v in vel.items())
          + '  mm/s RMS | highest of last %d: %5.2f mm/s, %s' % (len(history), worst, verdict))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--once', action='store_true', help='report the files already there and stop')
    args = parser.parse_args()
    seen, history = set(), []
    print('C1: housing velocity, 10-1000 Hz, limit %.0f mm/s RMS. Press any key (or Ctrl+C) to finish.' % LIMIT_MM_S)
    try:
        import msvcrt          # Windows: stop on any key
    except ImportError:
        msvcrt = None
    while True:
        if msvcrt is not None and msvcrt.kbhit():
            msvcrt.getch()
            break
        files = sorted(glob.glob(os.path.join(args.folder, 'snap_*.bin')), key=os.path.getmtime)
        for path in files:
            if path not in seen:
                seen.add(path)
                if args.once or path == files[-1]:
                    report(path, history)
        if args.once:
            break
        time.sleep(0.5)


if __name__ == '__main__':
    main()
