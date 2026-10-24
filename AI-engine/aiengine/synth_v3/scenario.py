"""SYNTHETIC demo replay stream: healthy -> fault onset -> failure, as 152-byte COE payloads.

    python -m aiengine.synth_v3.scenario --fault inner_race --out AI-engine/models/up2_demo_payloads.bin
    python -m aiengine.synth_v3.scenario --fault outer_race --rpm 3600 --seed 7 --out up2_or.bin

Same format as aiengine/demo_stream.py (models/demo_payloads.bin + .json): concatenated 152-byte payloads
(rpm f32 + 32 features f32 + 4 displacement f32 + t20_ms u32), plus a JSON sidecar (<out>.json) with the
per-window truth (class, health stage, RUL hours) and provenance.
  bearing 1 = features[0:16]  <- seeded fault, run to failure
  bearing 2 = features[16:32] <- healthy partner bearing (sees attenuated transmitted vibration)
t20_ms = machine time since run start in ms (one window per 10-minute snapshot, uint32).
Displacement fields = 0 (the generator models accelerometer features only).
[SYNTHETIC] -- never a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import payload
from ..config import BEARING_FEATURES, SEED
from . import PROVENANCE
from .generator import CADENCE_H, FAULTS, Knobs, generate_run, provenance


def build(fault: str, out: Path, rpm: float = 1750.0, seed: int = SEED + 999, n_healthy: int = 90,
          n_deg: int | None = None, knobs: Knobs | None = None) -> dict:
    k = knobs or Knobs()
    rng = np.random.default_rng(seed)
    ov = {"faulted_bearing": 1, "rpm_mode": float(rpm), "n_healthy": int(n_healthy)}
    if n_deg is not None:
        ov["n_deg"] = int(n_deg)
    run = generate_run(0, fault, rng, k, overrides=ov)
    b1 = run[run["unit_id"].str.endswith(":B1")].sort_values("window_index")
    b2 = run[run["unit_id"].str.endswith(":B2")].sort_values("window_index")
    f1, f2 = b1[BEARING_FEATURES].to_numpy(np.float32), b2[BEARING_FEATURES].to_numpy(np.float32)
    blob = bytearray()
    for i, r in enumerate(b1.itertuples(index=False)):
        w = payload.Window(rpm=float(r.rpm), features=np.concatenate([f1[i], f2[i]]), p1_amp_um=0.0, p1_phase_rad=0.0,
                           p2_amp_um=0.0, p2_phase_rad=0.0, t20_ms=int(round(float(r.t_hours) * 3.6e6)) % 2 ** 32)
        blob += payload.encode(w)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    healthy = fault == "healthy"
    obs = np.where((b1["u_true"].to_numpy() >= 0) & (not healthy), fault, "healthy")
    meta = {
        "file": out.name, "payload_size_bytes": payload.PAYLOAD_SIZE, "n_payloads": int(len(b1)),
        "provenance": PROVENANCE, "synthetic": True,
        "generator": {**provenance(1, seed, k), "overrides": ov, "command":
                      f"python -m aiengine.synth_v3.scenario --fault {fault} --rpm {rpm:g} --seed {seed} --out {out.as_posix()}"},
        "cadence_hours": CADENCE_H,
        "bearing_1": {"fault_class": fault, "run_to_failure": not healthy,
                      "life_hours": float(b1["t_hours"].iloc[-1]) if not healthy else None,
                      "true_onset_window_index": int(b1["true_onset_index"].iloc[0]),
                      "rpm": float(rpm),
                      "truth_per_window": {"rul_hours": (b1["rul_hours"].round(4).tolist() if not healthy else None),
                                           "health_stage": b1["true_stage"].astype(int).tolist(),
                                           "observable_class": obs.tolist()}},
        "bearing_2": {"fault_class": "healthy", "phase": "healthy partner bearing (transmitted vibration only)"},
        "displacement": "zeros (p1/p2 amp+phase): the synth_v3 generator models accelerometer features only",
        "t20_ms": "machine time since run start, ms (uint32), one window per 10-minute snapshot",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return {k2: v for k2, v in meta.items() if k2 not in ("bearing_1",)} | {
        "bearing_1": {k2: v for k2, v in meta["bearing_1"].items() if k2 != "truth_per_window"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aiengine.synth_v3.scenario", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault", default="inner_race", choices=FAULTS + ["healthy"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rpm", type=float, default=1750.0, help="1750 or 3600 (rig steady modes)")
    ap.add_argument("--seed", type=int, default=SEED + 999)
    ap.add_argument("--healthy-windows", type=int, default=90, help="pre-onset windows (>= 40 so the baseline fills)")
    ap.add_argument("--deg-windows", type=int, default=None, help="degradation windows (default: drawn like training runs)")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    info = build(a.fault, a.out, a.rpm, a.seed, a.healthy_windows, a.deg_windows, Knobs(noise_scale=a.noise_scale))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
