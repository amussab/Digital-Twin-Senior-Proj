"""HybridEngine: the runtime reference implementation (DESIGN.md section 8), mirrored by C#.

Consumes 152-byte COE payloads (or rpm + 32 features + t20_ms), splits each window into the two
biaxial bearings (features[0:16], features[16:32]) and keeps per-bearing state:
  baseline (commissioning; if not supplied, median of the first 24 windows),
  the TFT window buffer (encoder_length + 1 engineered rows),
  the N-HiTS buffer (encoder_length rows of [hi + covariates]) and HI history,
  the causal onset detector, and the class-probability history used to pick the RUL threshold.
Everything runs through ONNX Runtime with the constants in models/model_contract.json -- the
same artefacts the ASP.NET host loads. Nothing is hard-coded here that the contract doesn't carry.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import features as feat
from . import infer, labels, rul
from .config import MODELS_DIR
from .payload import decode


class _Bearing:
    def __init__(self, idx: int, baseline: np.ndarray | None, tl: int, nl: int):
        self.idx = idx
        self.baseline = None if baseline is None else np.asarray(baseline, np.float64)
        self.pending: list[tuple[np.ndarray, float, float]] = []   # raw16, rpm, t_h (pre-baseline)
        self.eng = deque(maxlen=tl + 1)
        self.nrows = deque(maxlen=nl)
        self.hi = deque(maxlen=48)
        self.t = deque(maxlen=48)
        self.probs = deque(maxlen=5)
        self.onset = labels.OnlineOnset()
        self.onset_t: float | None = None
        self.onset_hi: float | None = None
        self.n = 0


class HybridEngine:
    def __init__(self, model_dir: Path = MODELS_DIR, threads: int = 1,
                 baselines: dict[int, np.ndarray] | None = None):
        import onnxruntime as ort
        self.dir = Path(model_dir)
        self.c = json.loads((self.dir / "model_contract.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.tft = ort.InferenceSession(str(self.dir / "tft.onnx"), opts, providers=["CPUExecutionProvider"])
        self.nh = ort.InferenceSession(str(self.dir / "nhits.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tft_in = {i.name for i in self.tft.get_inputs()}
        self.nh_in = {i.name for i in self.nh.get_inputs()}
        mt, mn = self.c["models"]["tft"], self.c["models"]["nhits"]
        self.TL, self.tcols, self.tsc = mt["encoder_length"], mt["columns"], mt["scalers"]
        self.NL, self.NH, self.ncols, self.nsc = mn["encoder_length"], mn["prediction_length"], mn["columns"], mn["scalers"]
        self.classes = self.c["classes"]["labels"]
        # v2 hierarchical decision: before the causal onset is confirmed the bearing is healthy; after
        # it, the TFT's fault-type softmax (its healthy logit is a constant -1e4, so p(healthy) = 0).
        self.hierarchical = self.c["classes"].get("decision", {}).get("type") == "hierarchical_onset_gate"
        # IS3 v2 (MaFaulDa): flat 5-class TFT + the same causal onset gate on the monitoring stream:
        # before confirmation -> healthy; after -> fault-channel probabilities renormalised, p(healthy) = 0.
        self.rtf_gate = self.c["classes"].get("decision", {}).get("type") == "rtf_onset_gate"
        r = dict(self.c["rul"])
        self.calib = rul.RULCalibration.from_dict({k: r[k] for k in rul.RULCalibration.__dataclass_fields__ if k in r})
        self.interval_rel = float(r.get("interval_rel", 0.5))
        self.eidx = {n: i for i, n in enumerate(feat.ENGINEERED)}
        self.baselines = baselines or {}
        self.reset()

    def reset(self) -> None:
        self.b = {k: _Bearing(k, self.baselines.get(k), self.TL, self.NL) for k in (1, 2)}
        self._t0_ms: int | None = None
        self._wrap = 0
        self._last_ms: int | None = None

    # ------------------------------------------------------------------ inputs
    def _hours(self, t20_ms: int) -> float:
        if self._last_ms is not None and t20_ms < self._last_ms:
            self._wrap += 2 ** 32                       # uint32 wrap (~49.7 days)
        self._last_ms = t20_ms
        ms = t20_ms + self._wrap
        if self._t0_ms is None:
            self._t0_ms = ms
        return (ms - self._t0_ms) / 3.6e6

    def process_payload(self, raw: bytes) -> dict:
        w = decode(raw)
        return self.process(w.rpm, w.features, w.t20_ms)

    def process(self, rpm: float, features32: np.ndarray, t20_ms: int) -> dict:
        f = np.asarray(features32, np.float64)
        if f.shape != (32,) or not np.isfinite(f).all() or not np.isfinite(rpm):
            return {"error": "invalid window (shape/non-finite)"}
        t_h = self._hours(int(t20_ms))
        out = [self.step(k, f[(k - 1) * 16:k * 16], float(rpm), t_h) for k in (1, 2)]
        return {"rpm": float(rpm), "t20_ms": int(t20_ms), "bearings": out}

    # ------------------------------------------------------------------ per bearing
    def step(self, k: int, raw16: np.ndarray, rpm: float, t_h: float) -> dict:
        b = self.b[k]
        b.n += 1
        t0 = time.perf_counter()
        if b.baseline is None:
            b.pending.append((raw16, rpm, t_h))
            if len(b.pending) < feat.BASELINE_WINDOWS:
                return {"bearing": k, "status": "warming_up",
                        "reason": f"commissioning baseline {len(b.pending)}/{feat.BASELINE_WINDOWS}"}
            raws = np.stack([p[0] for p in b.pending])
            b.baseline = feat._floor(np.median(raws, axis=0))
            rows = feat.engineer_arrays(raws, np.array([p[1] for p in b.pending]), b.baseline)
            for (r16, rp, th), e in zip(b.pending, rows):
                self._push(b, e, th)
            b.pending = []
            e = rows[-1]
        else:
            e = feat.engineer_arrays(raw16[None], np.array([rpm]), b.baseline)[0]
            self._push(b, e, t_h)
        t_feat = (time.perf_counter() - t0) * 1000
        if len(b.eng) < self.TL + 1:
            return {"bearing": k, "status": "warming_up", "health_index": float(e[self.eidx["hi"]]),
                    "reason": f"TFT buffer {len(b.eng)}/{self.TL + 1}"}

        # TFT
        t1 = time.perf_counter()
        seq = np.stack(b.eng)[:, [self.eidx[c] for c in self.tcols]]
        enc, dec = infer.tft_tensors(seq, self.tcols, self.tsc, self.TL)
        feed = {k2: v for k2, v in {"encoder_cont": enc[-1:], "decoder_cont": dec[-1:]}.items() if k2 in self.tft_in}
        logits, vw = self.tft.run(["logits", "variable_weights"], feed)
        probs = infer.softmax(logits[0, 0].astype(np.float64))
        if self.hierarchical and b.onset.onset_index is None:
            probs = np.zeros_like(probs)
            probs[self.classes.index("healthy")] = 1.0
        elif self.rtf_gate:
            h = self.classes.index("healthy")
            if b.onset.onset_index is None:
                probs = np.zeros_like(probs)
                probs[h] = 1.0
            else:
                probs = probs.copy()
                probs[h] = 0.0
                probs = probs / max(probs.sum(), 1e-12)
        b.probs.append(probs)
        t_tft = (time.perf_counter() - t1) * 1000

        # N-HiTS (left-padded with the first row while the buffer is short, as in evaluation)
        t2 = time.perf_counter()
        rows = np.stack(b.nrows)
        if len(rows) < self.NL:
            rows = np.concatenate([np.repeat(rows[:1], self.NL - len(rows), axis=0), rows])
        nenc = infer.scale(rows, self.ncols, self.nsc)[None]
        ndec = np.repeat(nenc[:, -1:, :], self.NH, axis=1)
        nfeed = {k2: v for k2, v in {"encoder_cont": nenc, "decoder_cont": ndec}.items() if k2 in self.nh_in}
        forecast = self.nh.run(["hi_forecast"], nfeed)[0].reshape(-1).astype(np.float64)
        t_nh = (time.perf_counter() - t2) * 1000

        # RUL + stage
        t3 = time.perf_counter()
        pm = np.mean(np.stack(b.probs), axis=0)
        ci = {c: i for i, c in enumerate(self.classes)}
        fault_idx = [ci[c] for c in self.classes if c != "healthy"]
        thr_cls = "healthy" if pm[ci["healthy"]] >= 0.5 else self.classes[fault_idx[int(np.argmax(pm[fault_idx]))]]
        ts = np.asarray(b.t)
        cad = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
        tau = (t_h - b.onset_t) if b.onset_t is not None else None
        est = rul.estimate(self.calib, np.asarray(b.hi), forecast, thr_cls, tau, b.onset_hi, max(cad, 1e-9))
        stage = rul.stage_estimate(tau, est["rul_hours"])
        t_rul = (time.perf_counter() - t3) * 1000

        top = sorted(zip(self.tcols, vw[0].tolist()), key=lambda kv: -kv[1])[:5]
        cls = self.classes[int(np.argmax(probs))]
        return {
            "bearing": k, "status": "ok", "fault_class": cls,
            "class_probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "confidence": float(probs.max()), "health_index": float(e[self.eidx["hi"]]),
            "health_stage": int(stage), "onset_detected": b.onset_t is not None,
            "rul_hours": est["rul_hours"],
            "rul_interval_hours": rul.interval(est["rul_hours"], self.interval_rel),
            "rul_threshold_class": thr_cls,
            "hi_forecast": forecast.tolist(),
            "top_features": [[n, round(float(v), 4)] for n, v in top],
            "latency_ms": {"features": round(t_feat, 3), "tft": round(t_tft, 3), "nhits": round(t_nh, 3),
                           "rul": round(t_rul, 3),
                           "total": round(t_feat + t_tft + t_nh + t_rul, 3)},
        }

    def _push(self, b: _Bearing, e: np.ndarray, t_h: float) -> None:
        b.eng.append(e)
        b.nrows.append(e[[self.eidx[c] for c in self.ncols]])
        hi = float(e[self.eidx["hi"]])
        b.hi.append(hi)
        b.t.append(t_h)
        on_before = b.onset.onset_index
        on = b.onset.update(hi)
        if on is not None and on_before is None:
            back = b.onset.n - 1 - on                  # windows since the onset window
            b.onset_t = b.t[-1 - back] if back < len(b.t) else b.t[0]
            b.onset_hi = b.hi[-1 - back] if back < len(b.hi) else b.hi[0]

    # ------------------------------------------------------------------ tooling
    def run_unit(self, unit, bearing: int = 1):
        """Feed one window-table unit (16 features) as bearing `bearing`; the other bearing gets
        the same values (only `bearing`'s output is returned)."""
        from .config import BEARING_FEATURES
        self.reset()
        outs = []
        for r in unit.sort_values("window_index").itertuples(index=False):
            f16 = np.array([getattr(r, c) for c in BEARING_FEATURES], np.float64)
            t_ms = int(round(float(r.t_hours) * 3.6e6)) % (2 ** 32)
            o = self.process(float(r.rpm), np.concatenate([f16, f16]), t_ms)
            outs.append(o["bearings"][bearing - 1])
        return outs

    def benchmark(self, unit, n: int = 200) -> dict:
        outs = self.run_unit(unit)[: n + 60]
        tot = [o["latency_ms"]["total"] for o in outs if o.get("status") == "ok"][20:]
        if not tot:
            return {}
        a = np.asarray(tot)
        return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                "max_ms": float(a.max()), "mean_ms": float(a.mean()), "n": int(len(a)),
                "note": "per bearing-window, features+TFT+N-HiTS+RUL via ONNX Runtime"}

    def golden_vectors(self, unit, n: int = 12) -> dict:
        """Input->output vectors for C# parity tests: the first `n` 'ok' windows after warm-up,
        with every intermediate (engineered row, scaled tensors, logits, forecast, RUL)."""
        from .config import BEARING_FEATURES
        self.reset()
        cases, fed = [], []
        for r in unit.sort_values("window_index").itertuples(index=False):
            f16 = np.array([getattr(r, c) for c in BEARING_FEATURES], np.float64)
            t_ms = int(round(float(r.t_hours) * 3.6e6)) % (2 ** 32)
            fed.append({"window_index": int(r.window_index), "rpm": float(r.rpm), "t20_ms": t_ms,
                        "features16": f16.tolist()})
            out = self.process(float(r.rpm), np.concatenate([f16, f16]), t_ms)["bearings"][0]
            if out.get("status") != "ok":
                continue
            b = self.b[1]
            seq = np.stack(b.eng)[:, [self.eidx[c] for c in self.tcols]]
            enc, dec = infer.tft_tensors(seq, self.tcols, self.tsc, self.TL)
            cases.append({"window_index": int(r.window_index), "rpm": float(r.rpm), "t20_ms": t_ms,
                          "features16": f16.tolist(), "engineered": b.eng[-1].tolist(),
                          "tft_encoder_cont": enc[-1].tolist(), "tft_decoder_cont": dec[-1].tolist(),
                          "output": {k: v for k, v in out.items() if k != "latency_ms"}})
            if len(cases) >= n:
                break
        return {"unit_id": str(unit["unit_id"].iloc[0]), "baseline16": self.b[1].baseline.tolist(),
                "note": "feed the windows of this unit from window 0 in order (both bearings get the "
                        "same 16 values); compare bearing-1 outputs at the listed window_index values. "
                        "Tolerance: 1e-4 absolute on probabilities/HI, 1e-3 relative on RUL.",
                "inputs": fed, "cases": cases}
