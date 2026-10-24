"""CLI: python -m aiengine.coe.run --source sim|replay --fault outer_race --severity 0.5 --rpm 1750
                                   --udp 127.0.0.1:5005 --rate 2

Without --udp, windows are printed only. Sources: sim (SimulatedRigSource), replay (--file .npz).
--hours-per-window H re-stamps t20_ms = n * H h (one window per H-hour snapshot) so the backend's cadence-based RUL hours
are on the time base the RUL head was trained on (10 min = 0.1667 h); see README "COE time base". Default: real t20.
"""
from __future__ import annotations

import argparse
import dataclasses

from .. import payload as payload_mod
from .pipeline import CoePipeline
from .sources import FAULT_CLASSES, FileReplaySource, SimulatedRigSource
from .udp_sender import UdpSender, parse_hostport, stream


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["sim", "replay"], default="sim")
    ap.add_argument("--file", help="recorded .npz for --source replay")
    ap.add_argument("--fault", choices=FAULT_CLASSES, default="healthy")
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--rpm", type=float, default=1750.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--udp", help="host:port of the backend ingest (default port 5005)")
    ap.add_argument("--rate", type=float, default=0.0, help="windows/s pacing; 0 = unpaced")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--hours-per-window", type=float, default=None,
                    help="re-stamp t20_ms as one window per H hours (RUL time base); default = real contiguous t20")
    a = ap.parse_args(argv)
    if a.source == "replay":
        if not a.file:
            ap.error("--file required for --source replay")
        src = FileReplaySource(a.file, loop=True)
    else:
        src = SimulatedRigSource(a.fault, a.severity, a.rpm, seed=a.seed)
    pipe = CoePipeline(src)
    windows = pipe.windows() if a.hours_per_window is None else _restamp(pipe.windows(), a.hours_per_window)
    sender = UdpSender(*parse_hostport(a.udp)) if a.udp else None
    n = 0
    try:
        if sender:
            n = stream(windows, sender, a.rate, a.max_windows)
        else:
            for w in windows:
                n += 1
                print(f"t20={w.t20_ms} ms rpm={w.rpm:.1f} env_rms[ch0]={w.features[3]:.4f} "
                      f"bpfo[ch0]={w.features[6]:.4f} bpfi[ch0]={w.features[7]:.4f}")
                if a.max_windows and n >= a.max_windows:
                    break
    except KeyboardInterrupt:
        pass
    print(f"sent/produced {n} windows; stats valid={pipe.stats.valid} discarded={pipe.stats.discarded}")
    return 0


def _restamp(windows, hours: float):
    """t20_ms := n * hours (ms, uint32 wrap) and re-encode the 152-byte payload; features unchanged."""
    for n, w in enumerate(windows):
        if not w.valid:
            yield w
            continue
        t_ms = int(round(n * hours * 3.6e6)) & 0xFFFFFFFF
        p = payload_mod.decode(w.payload)
        yield dataclasses.replace(w, t20_ms=t_ms, payload=payload_mod.encode(dataclasses.replace(p, t20_ms=t_ms)))


if __name__ == "__main__":
    raise SystemExit(main())
