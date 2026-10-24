"""CLI: python -m aiengine.coe.run --source sim|replay --fault outer_race --severity 0.5 --rpm 1750
                                   --udp 127.0.0.1:5005 --rate 2

Without --udp, windows are printed only. Sources: sim (SimulatedRigSource), replay (--file .npz).
"""
from __future__ import annotations

import argparse

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
    a = ap.parse_args(argv)
    if a.source == "replay":
        if not a.file:
            ap.error("--file required for --source replay")
        src = FileReplaySource(a.file, loop=True)
    else:
        src = SimulatedRigSource(a.fault, a.severity, a.rpm, seed=a.seed)
    pipe = CoePipeline(src)
    sender = UdpSender(*parse_hostport(a.udp)) if a.udp else None
    n = 0
    try:
        if sender:
            n = stream(pipe.windows(), sender, a.rate, a.max_windows)
        else:
            for w in pipe.windows():
                n += 1
                print(f"t20={w.t20_ms} ms rpm={w.rpm:.1f} env_rms[ch0]={w.features[3]:.4f} "
                      f"bpfo[ch0]={w.features[6]:.4f} bpfi[ch0]={w.features[7]:.4f}")
                if a.max_windows and n >= a.max_windows:
                    break
    except KeyboardInterrupt:
        pass
    print(f"sent/produced {n} windows; stats valid={pipe.stats.valid} discarded={pipe.stats.discarded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
