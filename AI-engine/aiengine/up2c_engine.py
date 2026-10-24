"""v3c streaming engine = HybridEngine (contract v3b logic) + per-window physics features from the 4 displacement floats.

The only differences from the v3b engine (what the C# host must add for contract 3c):
  * per rig window, BEFORE the per-bearing steps: phys = PhysicsFeatures.update(rpm, p1_amp, p1_ph, p2_amp, p2_ph, t_h)
    -> (2, 4) (aiengine/up2c_physics.py; contract `physics_features`)
  * each bearing's N-HiTS-slot row = [21 engineered values, its 4 physics values of the SAME window]
    (during the 24-window feature baseline the rows are buffered; each buffered row keeps its own window's physics)
Everything else (classifier, aggregator-in-ONNX, post-process) is unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import features as feat
from .engine import HybridEngine
from .up2c_physics import PhysicsFeatures


class PhysicsHybridEngine(HybridEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ncols_eng = [c for c in self.ncols if c in self.eidx]

    def reset(self) -> None:
        super().reset()
        self.phys = PhysicsFeatures()
        self._q = {1: deque(), 2: deque()}
        self.last_phys = None

    def process(self, rpm: float, features32: np.ndarray, t20_ms: int, disp4=None) -> dict:
        f = np.asarray(features32, np.float64)
        if f.shape != (32,) or not np.isfinite(f).all() or not np.isfinite(rpm) or disp4 is None:
            return {"error": "invalid window (shape/non-finite/no displacement)"}
        t_h = self._hours(int(t20_ms))
        ph = self.phys.update(float(rpm), *[float(x) for x in disp4], t_h)
        self.last_phys = ph
        for k in (1, 2):
            self._q[k].append(np.asarray(ph[k - 1], np.float64))
        out = [self.step(k, f[(k - 1) * 16:k * 16], float(rpm), t_h) for k in (1, 2)]
        for k, o in zip((1, 2), out):
            o["physics"] = dict(zip(["phys_k_drop", "phys_k_est", "phys_rul_h", "phys_conf"], map(float, ph[k - 1])))
        return {"rpm": float(rpm), "t20_ms": int(t20_ms), "bearings": out}

    def _push(self, b, e: np.ndarray, t_h: float) -> None:
        ph = self._q[b.idx].popleft()
        b.eng.append(e)
        b.nrows.append(np.concatenate([e[[self.eidx[c] for c in self.ncols_eng]], ph]))
        hi = float(e[self.eidx["hi"]])
        b.hi.append(hi)
        b.t.append(t_h)
        on_before = b.onset.onset_index
        on = b.onset.update(hi)
        if on is not None and on_before is None:
            back = b.onset.n - 1 - on
            b.onset_t = b.t[-1 - back] if back < len(b.t) else b.t[0]
            b.onset_hi = b.hi[-1 - back] if back < len(b.hi) else b.hi[0]


def run_rig(eng: PhysicsHybridEngine, run: "pd.DataFrame", n: int | None = None):
    """Feed one RUN (both bearings' rows of a physics-attached table) in window order.
    Yields (window_row_b1, window_row_b2, output dict)."""
    from .config import BEARING_FEATURES
    from .payload import DISPLACEMENT_NAMES
    eng.reset()
    b1 = run[run["unit_id"].str.endswith(":B1")].sort_values("window_index")
    b2 = run[run["unit_id"].str.endswith(":B2")].sort_values("window_index")
    for i, (r1, r2) in enumerate(zip(b1.itertuples(index=False), b2.itertuples(index=False))):
        if n is not None and i >= n:
            break
        f32 = np.array([getattr(r1, c) for c in BEARING_FEATURES] + [getattr(r2, c) for c in BEARING_FEATURES], np.float64)
        t_ms = int(round(float(r1.t_hours) * 3.6e6)) % (2 ** 32)
        disp = [float(getattr(r1, c)) for c in DISPLACEMENT_NAMES]
        yield r1, r2, f32, t_ms, disp, eng.process(float(r1.rpm), f32, t_ms, disp)


def golden(model_dir, contract_file: str, t3, h: dict, hb: dict, n_early: int = 12, n_late: int = 16) -> dict:
    """Golden vectors for the C# port: one test3 run-to-failure RIG fed from window 0 (payload-level inputs incl.
    the 4 displacement floats), outputs of the FAULTED bearing at chosen windows, plus the physics features of every
    window and an engine-vs-offline comparison."""
    from .up2c import predict_rul
    eng = PhysicsHybridEngine(model_dir, contract_file=contract_file)
    rtf = t3[t3["run_to_failure"]]
    sizes = rtf.groupby("unit_id").size()
    sizes = sizes[sizes.index.str.contains("run6")]
    uid = sizes[sizes >= 150].idxmin()                      # shortest run with >= 150 windows (file size)
    fb = int(uid[-1])
    run = t3[t3["bearing_key"] == t3.loc[t3["unit_id"] == uid, "bearing_key"].iloc[0]]
    nw = run["window_index"].nunique()
    late = set(int(x) for x in np.linspace(nw * 0.3, nw - 1, n_late))
    inputs, cases, phys, outs = [], [], [], []
    for r1, r2, f32, t_ms, disp, o in run_rig(eng, run):
        wi = int(r1.window_index)
        inputs.append({"window_index": wi, "rpm": float(r1.rpm), "t20_ms": t_ms, "features32": f32.tolist(), "displacement4": disp})
        phys.append(eng.last_phys.tolist())
        ob = o["bearings"][fb - 1]
        outs.append((wi, ob))
        if ob.get("status") == "ok" and (len([c for c in cases if c["window_index"] < 10 ** 9]) < n_early or wi in late):
            b = eng.b[fb]
            cases.append({"window_index": wi, "nhits_row": list(map(float, b.nrows[-1])),
                          "output": {k: v for k, v in ob.items() if k != "latency_ms"}})
    ok = [(w, o) for w, o in outs if o.get("status") == "ok"]
    rp = predict_rul(run, h["v3c"], hb)
    rp = rp[rp["unit_id"] == uid].set_index("window_index").loc[[w for w, _ in ok]]
    e_r = np.array([o["rul_hours"] for _, o in ok])
    e_s = np.array([o["health_stage"] for _, o in ok])
    eo = {"windows": len(ok), "rul_hours_max_rel_diff": float((np.abs(e_r - rp["rul_pred_h"].to_numpy()) /
                                                               np.maximum(rp["rul_pred_h"].to_numpy(), 0.1)).max()),
          "stage_mismatch": int((e_s != rp["stage_pred"].to_numpy()).sum()),
          "phys_max_abs_diff_vs_offline_batch": float(np.abs(np.asarray(phys)[:, fb - 1, :] -
                                                         run[run["unit_id"] == uid].sort_values("window_index")[
                                                             ["phys_k_drop", "phys_k_est", "phys_rul_h", "phys_conf"]].to_numpy()).max()),
          "note": "engine (streaming, stock twin identifier, ONNX, contract cadence) vs offline (batch identifier, native trees, "
                  "generator cadence)"}
    return {"unit_id": uid, "faulted_bearing": fb, "provenance": "[SYNTHETIC] synth_v3_test3 run + [SIMULATION] displacements",
            "note": "Feed every window of this rig from window 0 in order via process(rpm, features32, t20_ms, displacement4). "
                    "Compare: physics[i][b] (per window, per bearing: phys_k_drop, phys_k_est, phys_rul_h, phys_conf; tol 1e-4 "
                    "relative on k, 1e-3 relative on rul), and bearing `faulted_bearing` outputs at cases[].window_index "
                    "(rul_hours 1e-3 relative, health_stage exact, class probs 1e-4).",
            "inputs": inputs, "physics": phys, "cases": cases, "engine_vs_offline": eo}
