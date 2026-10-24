"""[SIMULATION] COE simulator scenario: healthy baseline -> ramping outer-race fault, sent over UDP.

Uses the SAME CoePipeline + UdpSender as `python -m aiengine.coe.run`; only the simulated severity is changed over time
(SimulatedRigSource reads self.severity per impact). Swapping in the real STM32/ESP32-C6 node changes only the sender.

  cd AI-engine; .venv/Scripts/python.exe ../scripts/coe_fault_scenario.py --udp 127.0.0.1:5005 --rate 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AI-engine"))
from aiengine.coe.run import _restamp                  # noqa: E402
from aiengine.coe.pipeline import CoePipeline          # noqa: E402
from aiengine.coe.sources import SimulatedRigSource    # noqa: E402
from aiengine.coe.udp_sender import UdpSender, parse_hostport, stream  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--udp", default="127.0.0.1:5005")
    ap.add_argument("--rate", type=float, default=2.0, help="windows/s (a real 1750 rpm node makes 1.46/s)")
    ap.add_argument("--rpm", type=float, default=1750.0)
    ap.add_argument("--fault", default="outer_race")
    ap.add_argument("--healthy-windows", type=int, default=40, help="baseline needs 24 + 25 buffer")
    ap.add_argument("--ramp-windows", type=int, default=160)
    ap.add_argument("--max-severity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours-per-window", type=float, default=None,
                    help="re-stamp t20_ms as one window per H hours (RUL time base, e.g. 0.1667); default real t20")
    a = ap.parse_args(argv)
    src = SimulatedRigSource(a.fault, 0.01, a.rpm, seed=a.seed)
    pipe = CoePipeline(src)
    sender = UdpSender(*parse_hostport(a.udp))
    counter = {"n": 0}

    def gen():
        for w in pipe.windows():
            n = counter["n"] = counter["n"] + 1
            k = max(0, n - a.healthy_windows)
            src.severity = 0.0 if k == 0 else min(a.max_severity, a.max_severity * k / a.ramp_windows)
            yield w

    try:
        wins = gen() if a.hours_per_window is None else _restamp(gen(), a.hours_per_window)
        sent = stream(wins, sender, a.rate, a.healthy_windows + a.ramp_windows + 40)
    except KeyboardInterrupt:
        sent = counter["n"]
    print(f"[SIMULATION] sent {sent} windows to {a.udp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
