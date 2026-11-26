"""FPR demo driver: for each rendered rig_anchor unit, start a FRESH inference receiver (NodeEngine, table time base),
replay the unit through it (replay_node.py), stop it. Receiver -> NodeEngine -> POST /api/live/prediction -> dashboard.

    python scripts/fpr_replay.py [--contract v7_node] [--speed 12] [--loop | --once] [--backend http://127.0.0.1:5080] [files...]

Fresh receiver per unit because the commissioning baseline is keyed by rpm and a new unit must be re-commissioned.
Stdlib only (runs in the python image and natively). Provenance of everything it streams: [SIMULATION; rig-anchored replay].
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_port(port: int, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--contract", default=os.environ.get("NODE_CONTRACT", "v7_node"))
    ap.add_argument("--speed", type=float, default=float(os.environ.get("FPR_SPEED", 12)))
    ap.add_argument("--backend", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:5080"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--loop", action="store_true", default=os.environ.get("FPR_LOOP", "0") == "1")
    ap.add_argument("--once", action="store_true", help="single pass (overrides --loop / FPR_LOOP)")
    a = ap.parse_args()
    files = [str(Path(f).resolve()) for f in a.files] or sorted(str(p) for p in (ROOT / "AI-engine" / "demo").glob("fpr_*.jsonl"))
    if not files:
        print("no demo jsonl found: python -m aiengine.fpr_demo render (from AI-engine)", file=sys.stderr)
        return 2
    env = dict(os.environ, NODE_TIME_BASE="table", NODE_CONTRACT=a.contract, BACKEND_URL=a.backend, PYTHONUNBUFFERED="1",
               RIG_LOG_DIR=os.environ.get("RIG_LOG_DIR", str(ROOT / "logs")))
    inf = ROOT / "COE" / "inference"
    while True:
        for f in files:
            print(f"== replay {Path(f).name} (contract {a.contract}, x{a.speed}) [SIMULATION; rig-anchored replay]", flush=True)
            rx = subprocess.Popen([sys.executable, "inference_receiver.py", "--port", str(a.port)], cwd=inf, env=env)
            try:
                if not wait_port(a.port):
                    print("receiver did not start", file=sys.stderr)
                    return 1
                subprocess.run([sys.executable, "replay_node.py", f, "--host", "127.0.0.1", "--port", str(a.port), "--speed", str(a.speed)],
                               cwd=inf, env=env, check=False)
                time.sleep(2)
            finally:
                rx.terminate()
                try:
                    rx.wait(10)
                except subprocess.TimeoutExpired:
                    rx.kill()
        if a.once or not a.loop:
            return 0


if __name__ == "__main__":
    sys.exit(main())
