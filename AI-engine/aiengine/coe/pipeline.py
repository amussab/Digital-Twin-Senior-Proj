"""COE Components 1-9, streaming and block-wise (mirrors the firmware structure of the COE doc).

    process_dma_block()  : Components 1 (window formation), 2 (calibration), 3 (band-pass + running
                           stats), 4 (rectify), 5 (envelope low-pass + decimate x6)
    proximity revolution : Component 8 (1x sine/cosine totals, one revolution at a time)
    finalize_window()    : Components 6 (Hann, zero-pad 4096 FFT, order axis), 7 (32 features)
    build payload        : Component 9 (152 bytes, aiengine.payload)

Raw blocks are never joined into one raw buffer: only the decimated envelope record (~3.4 k samples
per channel at 1750 rpm) and O(1) running totals are kept per window. Filter state and the
decimation phase continue across block boundaries.

Per-window feature maths is deliberately plain (sums, sqrt, one rFFT, band sums) so it can be
reimplemented in C#/C. Floating point is float64 here; the STM32 uses float32 (kurtosis from raw
power sums is the one place where firmware should use shifted/Welford sums).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
from scipy import signal

from .. import payload as payload_mod
from .config import CoeConfig, G
from .sources import SampleBlock, SampleSource

FAMILIES = ("FTF", "BSF", "BPFO", "BPFI")


@dataclass
class WindowResult:
    valid: bool
    reason: str                       # "" when valid
    rpm: float = 0.0
    features: Optional[np.ndarray] = None      # float32[32]
    displacement: Optional[np.ndarray] = None  # float32[4]: p1 amp um, p1 phase rad, p2 amp, p2 phase
    t20_ms: int = 0
    payload: bytes = b""
    t0_s: float = 0.0
    t20_s: float = 0.0
    truncated: bool = False           # envelope record longer than the 4096 FFT (rpm < ~1465)
    finalize_ms: float = 0.0          # Components 6-9 wall time on THIS host
    t20_to_payload_ms: float = 0.0    # last-block processing + finalize, host wall time


@dataclass
class PipelineStats:
    valid: int = 0
    discarded: dict = field(default_factory=dict)
    blocks: int = 0
    dropped_blocks: int = 0

    def discard(self, reason: str) -> None:
        self.discarded[reason] = self.discarded.get(reason, 0) + 1

    @property
    def total_discarded(self) -> int:
        return sum(self.discarded.values())


class _Window:
    """Running totals for one 20-revolution window (what firmware keeps between blocks)."""

    def __init__(self, t0: float):
        self.edges = [t0]
        self.n = 0
        self.s1 = np.zeros(4); self.s2 = np.zeros(4); self.s3 = np.zeros(4); self.s4 = np.zeros(4)
        self.peak = np.zeros(4)
        self.env: list[list[np.ndarray]] = [[] for _ in range(4)]
        # Component 8: current-revolution buffer and totals
        self.rev_idx: list[np.ndarray] = []
        self.rev_val: list[np.ndarray] = []
        self.sin = np.zeros(2); self.cos = np.zeros(2); self.n_prox = 0


class CoePipeline:
    def __init__(self, source: SampleSource, cfg: Optional[CoeConfig] = None, include_discarded: bool = False):
        self.source = source
        self.cfg = cfg or source.cfg
        self.include_discarded = include_discarded
        self.stats = PipelineStats()
        c = self.cfg
        self._sos_bp = signal.butter(c.bandpass_order, [c.bandpass_hz[0], c.bandpass_hz[1]], btype="bandpass",
                                     output="sos", fs=c.fs_hz)
        self._sos_lp = signal.butter(c.envelope_lp_order, c.envelope_lowpass_hz, btype="lowpass",
                                     output="sos", fs=c.fs_hz)
        self._hann_cache: dict[int, np.ndarray] = {}
        self._reset_filters()
        self._win: Optional[_Window] = None
        self._last_seq: Optional[int] = None

    # ----------------------------------------------------------------- public
    def windows(self) -> Iterator[WindowResult]:
        for block in self.source.blocks():
            for r in self.push_block(block):
                if r.valid or self.include_discarded:
                    yield r

    # ----------------------------------------------------------------- state
    def _reset_filters(self) -> None:
        self._zi_bp = np.zeros((self._sos_bp.shape[0], 2, 4))
        self._zi_lp = np.zeros((self._sos_lp.shape[0], 2, 4))
        self._since_reset = 0

    def _abort(self, reason: str, out: list) -> None:
        if self._win is not None:
            self.stats.discard(reason)
            out.append(WindowResult(False, reason))
        self._win = None

    # ----------------------------------------------------------------- Component 2
    def calibrate(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """int16 codes (256,8) -> acceleration m/s2 (256,4) and displacement um (256,2)."""
        c = self.cfg
        volts = codes.astype(np.float64) / c.counts_per_volt
        acc = (volts[:, 0:4] - c.accel_offset_v) / (c.accel_sens_mv_per_g / 1000.0) * G
        disp = (volts[:, 4:6] - c.prox_offset_v) / (c.prox_sens_mv_per_um / 1000.0)
        return acc, disp

    # ----------------------------------------------------------------- Components 1-5, 8 (per block)
    def push_block(self, b: SampleBlock) -> list[WindowResult]:
        t_in = time.perf_counter()
        c, out = self.cfg, []
        self.stats.blocks += 1
        nb = c.block_sets
        if self._last_seq is not None and b.seq != self._last_seq + 1:
            self.stats.dropped_blocks += b.seq - self._last_seq - 1
            self._abort("dropped_block", out)
            self._reset_filters()
        self._last_seq = b.seq

        acc, disp = self.calibrate(b.codes)
        bp, self._zi_bp = signal.sosfilt(self._sos_bp, acc, axis=0, zi=self._zi_bp)        # C3
        env_full, self._zi_lp = signal.sosfilt(self._sos_lp, np.abs(bp), axis=0, zi=self._zi_lp)  # C4 + C5 LP
        base = b.seq * nb
        self._since_reset += nb

        pos = 0
        edges = np.sort(b.edges_s)
        # first sample index at or after each edge (a sample at t >= edge belongs to the next window)
        ks = np.clip(np.ceil(edges * c.fs_hz - base - 1e-9).astype(int), 0, nb)
        for e, k in zip(edges, ks):
            self._add_segment(base, pos, int(k), bp, env_full, disp)
            pos = int(k)
            r = self._on_edge(float(e), t_in)
            if r is not None:
                out.append(r)
        self._add_segment(base, pos, nb, bp, env_full, disp)
        # runaway guard: no 20th edge for far too long (stalled Keyphasor)
        if self._win is not None and self._win.n > c.window_samples(c.rpm_range[0] * 0.5):
            self._abort("timeout", out)
        return out

    def _add_segment(self, base: int, i0: int, i1: int, bp, env_full, disp) -> None:
        w = self._win
        if w is None or i1 <= i0:
            return
        x = bp[i0:i1]
        x2 = x * x
        w.n += i1 - i0
        w.s1 += x.sum(0); w.s2 += x2.sum(0); w.s3 += (x2 * x).sum(0); w.s4 += (x2 * x2).sum(0)
        w.peak = np.maximum(w.peak, np.abs(x).max(0))
        d = self.cfg.decimation
        first = (-(base + i0)) % d                    # decimation phase continues across blocks
        e = env_full[i0 + first:i1:d]
        for ch in range(4):
            w.env[ch].append(e[:, ch].copy())
        w.rev_idx.append(np.arange(base + i0, base + i1))
        w.rev_val.append(disp[i0:i1].copy())

    def _begin(self, t0: float) -> None:
        # filters must have settled before a window may start (startup / after a lost block)
        self._win = _Window(t0) if self._since_reset >= self.cfg.settle_samples else None

    def _on_edge(self, t: float, t_in: float) -> Optional[WindowResult]:
        w = self._win
        if w is None:
            self._begin(t)
            return None
        self._close_revolution(w, w.edges[-1], t)
        w.edges.append(t)
        if len(w.edges) - 1 < self.cfg.revolutions:
            return None
        res = self._finalize(w, t_in)
        self._win = None
        self._begin(t)                                # t20 is also t0 of the next window
        return res

    # ----------------------------------------------------------------- Component 8
    def _close_revolution(self, w: _Window, t_start: float, t_end: float) -> None:
        if not w.rev_idx:
            return
        idx = np.concatenate(w.rev_idx)
        val = np.concatenate(w.rev_val)
        w.rev_idx, w.rev_val = [], []
        t = idx / self.cfg.fs_hz
        m = (t >= t_start) & (t < t_end)
        if m.sum() < 8:
            return
        theta = 2.0 * np.pi * (t[m] - t_start) / (t_end - t_start)
        v = val[m] - val[m].mean(axis=0)              # remove the average probe gap
        w.sin += (v * np.sin(theta)[:, None]).sum(0)
        w.cos += (v * np.cos(theta)[:, None]).sum(0)
        w.n_prox += int(m.sum())

    # ----------------------------------------------------------------- Components 6-9
    def _hann(self, n: int) -> np.ndarray:
        h = self._hann_cache.get(n)
        if h is None:
            h = self._hann_cache[n] = np.hanning(n)
        return h

    def _speed_ok(self, rpm: float) -> bool:
        c = self.cfg
        if c.speed_policy == "range":
            return c.rpm_range[0] <= rpm <= c.rpm_range[1]
        return any(abs(rpm - m) <= c.mode_tolerance * m for m in c.modes_rpm)

    def _finalize(self, w: _Window, t_in: float) -> WindowResult:
        c = self.cfg
        t0, t20 = w.edges[0], w.edges[-1]
        revs = np.diff(w.edges)
        rpm = c.revolutions * 60.0 / (t20 - t0)
        reason = ""
        if np.max(np.abs(revs / revs.mean() - 1.0)) > c.max_rev_deviation:
            reason = "speed_unsteady_or_missed_edge"
        elif not self._speed_ok(rpm):
            reason = "rpm_out_of_range"
        elif abs(w.n - c.window_samples(rpm)) > 3:
            reason = "sample_count_mismatch"
        if reason:
            self.stats.discard(reason)
            return WindowResult(False, reason, rpm=rpm, t0_s=t0, t20_s=t20)

        t_fin = time.perf_counter()
        n = w.n
        feats = np.zeros((4, 8))
        mean = w.s1 / n
        var = w.s2 / n - mean ** 2
        m4 = w.s4 / n - 4 * mean * w.s3 / n + 6 * mean ** 2 * w.s2 / n - 3 * mean ** 4
        rms = np.sqrt(w.s2 / n)
        truncated = False
        shaft_hz = rpm / 60.0
        for ch in range(4):
            env = np.concatenate(w.env[ch])
            env_rms = float(np.sqrt(np.mean(env ** 2)))
            rec = env - env.mean()
            if rec.size > c.fft_size:
                rec, truncated = rec[:c.fft_size], True
            rec = rec * self._hann(rec.size)
            padded = np.zeros(c.fft_size)
            padded[:rec.size] = rec
            mag = np.abs(np.fft.rfft(padded)) * 2.0 / rec.size
            orders = np.fft.rfftfreq(c.fft_size, d=1.0 / c.envelope_rate_hz) / shaft_hz
            fam = []
            for f in FAMILIES:
                sel = (orders >= c.orders[f] - c.order_half_width) & (orders <= c.orders[f] + c.order_half_width)
                fam.append(float(np.sqrt(np.sum(mag[sel] ** 2))))
            kurt = float(m4[ch] / var[ch] ** 2) if var[ch] > 0 else 0.0
            crest = float(w.peak[ch] / rms[ch]) if rms[ch] > 0 else 0.0
            feats[ch] = [rms[ch], kurt, crest, env_rms, *fam]
        # Component 8 finalisation: x(theta) = A cos(theta - phi)
        a_c = 2.0 * w.cos / max(w.n_prox, 1)          # n_prox = samples per probe
        a_s = 2.0 * w.sin / max(w.n_prox, 1)
        amp = np.hypot(a_c, a_s)
        phase = np.arctan2(a_s, a_c)
        disp = np.array([amp[0], phase[0], amp[1], phase[1]], dtype=np.float32)
        f32 = feats.reshape(32).astype(np.float32)
        t20_ms = int(round(t20 * 1000.0)) & 0xFFFFFFFF
        wp = payload_mod.Window(rpm=float(rpm), features=f32, p1_amp_um=float(disp[0]),
                                p1_phase_rad=float(disp[1]), p2_amp_um=float(disp[2]),
                                p2_phase_rad=float(disp[3]), t20_ms=t20_ms)
        if not (np.isfinite(f32).all() and np.isfinite(disp).all()):
            self.stats.discard("non_finite")
            return WindowResult(False, "non_finite", rpm=rpm, t0_s=t0, t20_s=t20)
        raw = payload_mod.encode(wp)
        now = time.perf_counter()
        self.stats.valid += 1
        return WindowResult(True, "", rpm=float(rpm), features=f32, displacement=disp, t20_ms=t20_ms,
                            payload=raw, t0_s=t0, t20_s=t20, truncated=truncated,
                            finalize_ms=(now - t_fin) * 1e3, t20_to_payload_ms=(now - t_in) * 1e3)
