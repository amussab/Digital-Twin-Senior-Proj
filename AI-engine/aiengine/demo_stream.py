"""Replay stream for the live demo: models/demo_payloads.bin (+ demo_payloads.json).

Concatenated 152-byte COE payloads (aiengine/payload.py): rpm f32 + 32 features f32 +
4 displacement f32 (zeros: no displacement in the public datasets) + t20_ms u32.
  bearing 1 = features[0:16]  <- a held-out XJTU-SY TEST bearing, run to failure (real degradation)
  bearing 2 = features[16:32] <- the healthy phase (pre-onset windows) of another held-out test bearing
rpm comes from the bearing-1 record; t20_ms = bearing 1's machine time (ms, uint32).
[MEASURED on XJTU-SY] feature values, replayed; nothing synthetic is added.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import payload
from .config import BEARING_FEATURES, MODELS_DIR


def build(df: pd.DataFrame, b1_unit: str, b2_unit: str, out_dir: Path = MODELS_DIR) -> dict:
    u1 = df[df["unit_id"] == b1_unit].sort_values("window_index")
    u2 = df[df["unit_id"] == b2_unit].sort_values("window_index")
    on2 = int(u2["onset_index"].iloc[0])
    healthy2 = u2[u2["window_index"] < on2] if on2 >= 0 else u2
    if len(healthy2) < len(u1):
        raise ValueError(f"{b2_unit} has only {len(healthy2)} pre-onset windows, need {len(u1)}")
    healthy2 = healthy2.head(len(u1))
    f1 = u1[BEARING_FEATURES].to_numpy(np.float32)
    f2 = healthy2[BEARING_FEATURES].to_numpy(np.float32)
    blob = bytearray()
    for i, r in enumerate(u1.itertuples(index=False)):
        w = payload.Window(rpm=float(r.rpm), features=np.concatenate([f1[i], f2[i]]),
                           p1_amp_um=0.0, p1_phase_rad=0.0, p2_amp_um=0.0, p2_phase_rad=0.0,
                           t20_ms=int(round(float(r.t_hours) * 3.6e6)) % 2 ** 32)
        blob += payload.encode(w)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo_payloads.bin").write_bytes(bytes(blob))
    meta = {
        "file": "demo_payloads.bin", "payload_size_bytes": payload.PAYLOAD_SIZE, "n_payloads": len(u1),
        "provenance": "MEASURED on XJTU-SY public bearing dataset (feature values replayed), not the team rig",
        "bearing_1": {"unit_id": b1_unit, "split": str(u1["split"].iloc[0]),
                      "fault_class": str(u1["fault_class"].iloc[0]), "life_hours": float(u1["life_hours"].iloc[0]),
                      "onset_window_index": int(u1["onset_index"].iloc[0]), "rpm": float(u1["rpm"].median()),
                      "truth_per_window": {"rul_hours": u1["rul_hours"].round(4).tolist(),
                                           "health_stage": u1["health_stage"].astype(int).tolist(),
                                           "observable_class": u1["observable_class"].astype(str).tolist()}},
        "bearing_2": {"unit_id": b2_unit, "split": str(u2["split"].iloc[0]),
                      "windows_used": [int(healthy2["window_index"].min()), int(healthy2["window_index"].max())],
                      "onset_window_index": on2, "phase": "healthy (pre-onset)",
                      "rpm_of_record": float(u2["rpm"].median()),
                      "note": "payload rpm is bearing 1's; bearing 2's own record ran at rpm_of_record"},
        "displacement": "zeros (p1/p2 amp+phase): public datasets carry no shaft displacement",
        "t20_ms": "bearing-1 machine time since run start, ms (uint32)",
    }
    (out_dir / "demo_payloads.json").write_text(json.dumps(meta, indent=1))
    return {k: v for k, v in meta.items() if k not in ("bearing_1",)} | {
        "bearing_1": {k: v for k, v in meta["bearing_1"].items() if k != "truth_per_window"}}
