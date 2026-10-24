"""Sample sources: the single boundary between "where the samples come from" and the COE pipeline.

A source yields `SampleBlock`s exactly as the STM32 DMA ISR would hand them to firmware:
256 simultaneous sample sets x 8 AD7606 words of int16 codes at 30 kS/s, plus the Keyphasor
edge timestamps that fell inside that block's time span. Swapping `SimulatedRigSource` for a real
node must not change anything downstream (`CoePipeline`, payload, UDP, backend).

Channel map [COE section 2]: words 0-3 = B1-X, B1-Y, B2-X, B2-Y acceleration; word 4 = plane-1
radial proximity; word 5 = plane-2 radial proximity; words 6-7 reserved (zeros).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from .config import CoeConfig, G

FAULT_FAMILY = {"outer_race": "BPFO", "inner_race": "BPFI", "ball": "BSF", "cage": "FTF"}
FAULT_CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]


@dataclass
class SampleBlock:
    seq: int                    # DMA block counter, +1 per block; a gap means lost data
    codes: np.ndarray           # int16 (256, 8)
    edges_s: np.ndarray         # float64 Keyphasor edge times (s since node start) inside this block

    @property
    def t_start_s(self) -> float:
        return self.seq * self.codes.shape[0] / 30_000.0


class SampleSource(abc.ABC):
    """Abstract source. `blocks()` yields SampleBlock in order (seq may skip if blocks were lost)."""
    cfg: CoeConfig

    @abc.abstractmethod
    def blocks(self) -> Iterator[SampleBlock]: ...

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class SimulatedRigSource(SampleSource):
    """Physics-ish rig simulator [SYNTHETIC].

    Shaft 1x/2x vibration, bearing-defect impact trains at FTF/BSF/BPFO/BPFI orders (1 % slip jitter
    per impact) each exciting a decaying `resonance_hz` ring-down (inside the 2-8 kHz band), white
    noise, and two proximity channels carrying a 1x orbit on a DC gap voltage. The faulted bearing
    sees the full impacts; the other bearing sees `OFF_PLANE` of them. Y axis 0.8x of X.
    Amplitudes are invented (not rig-calibrated); severity 0..1 scales impact amplitude linearly.
    """
    OFF_PLANE = 0.34            # same transmitted fraction as datasets/synthetic_rig.py
    Y_GAIN = 0.8
    IMPACT_A0 = 25.0            # m/s2 peak ring-down amplitude at severity 1
    TAU_S = 0.35e-3
    NOISE_MS2 = 0.6
    SHAFT_1X_MS2 = 3.0
    SHAFT_2X_MS2 = 1.0
    PROX_AMP_UM = (25.0, 20.0)
    PROX_PHASE = (0.6, 1.1)
    PROX_NOISE_UM = 0.4
    GAP_V = -8.0

    def __init__(self, fault: str = "healthy", severity: float = 0.0, rpm: float = 1750.0,
                 seed: int = 0, n_blocks: Optional[int] = None, faulted_bearing: int = 1,
                 resonance_hz: float = 5000.0, rpm_jitter: float = 0.001,
                 rpm_schedule: Optional[Callable[[float], float]] = None,
                 cfg: Optional[CoeConfig] = None, start_seq: int = 0):
        if fault not in FAULT_CLASSES:
            raise ValueError(f"fault must be one of {FAULT_CLASSES}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        self.cfg = cfg or CoeConfig()
        self.fault, self.severity, self.rpm = fault, (0.0 if fault == "healthy" else severity), rpm
        self.seed, self.n_blocks, self.faulted_bearing = seed, n_blocks, faulted_bearing
        self.resonance_hz, self.rpm_jitter, self.rpm_schedule = resonance_hz, rpm_jitter, rpm_schedule
        self.start_seq = start_seq

    def blocks(self) -> Iterator[SampleBlock]:
        cfg, rng = self.cfg, np.random.default_rng(self.seed)
        fs, nb = cfg.fs_hz, cfg.block_sets
        dt_blk = nb / fs
        phase = float(rng.uniform(0.0, 1.0))          # shaft revolutions
        drift = 0.0
        fam = FAULT_FAMILY.get(self.fault)
        order = cfg.orders[fam] if fam else None
        next_ev = (1.0 + float(rng.uniform(0, 1))) / order if fam else None
        events: list[tuple[float, float]] = []        # (t0_s, amplitude m/s2 before gains)
        seq, emitted = self.start_seq, 0
        idx = np.arange(nb)
        while self.n_blocks is None or emitted < self.n_blocks:
            tb = seq * dt_blk
            drift = 0.98 * drift + rng.normal(0.0, self.rpm_jitter * 0.2)
            nominal = self.rpm_schedule(tb) if self.rpm_schedule else self.rpm
            rate = nominal * (1.0 + drift) / 60.0     # rev/s, constant inside a block
            p0, p1 = phase, phase + rate * dt_blk
            edges = np.array([tb + (k - p0) / rate
                              for k in range(int(np.floor(p0)) + 1, int(np.floor(p1)) + 1)], dtype=np.float64)
            # defect impacts whose shaft phase falls in this block
            if fam:
                while next_ev < p1:
                    if next_ev >= p0:
                        t_ev = tb + (next_ev - p0) / rate
                        amp = self.IMPACT_A0 * self.severity * float(rng.lognormal(0.0, 0.1))
                        if fam == "BPFI":
                            amp *= 0.6 + 0.4 * np.cos(2 * np.pi * next_ev)
                        elif fam == "BSF":
                            amp *= 0.6 + 0.4 * np.cos(2 * np.pi * cfg.orders["FTF"] * next_ev)
                        elif fam == "FTF":
                            amp *= 0.5
                        events.append((t_ev, amp))
                    next_ev += (1.0 + float(rng.normal(0.0, 0.01))) / order
            t = tb + idx / fs
            ph = p0 + rate * idx / fs
            ring = np.zeros(nb)
            keep = []
            for (t0, amp) in events:
                if t0 + 6 * self.TAU_S < tb:
                    continue
                keep.append((t0, amp))
                d = t - t0
                m = (d >= 0) & (d < 6 * self.TAU_S)
                if m.any():
                    ring[m] += amp * np.exp(-d[m] / self.TAU_S) * np.sin(2 * np.pi * self.resonance_hz * d[m])
            events = keep
            acc = np.empty((nb, 4))
            for ch in range(4):
                bearing, axis = ch // 2 + 1, ch % 2
                g_b = 1.0 if bearing == self.faulted_bearing else self.OFF_PLANE
                g_a = 1.0 if axis == 0 else self.Y_GAIN
                shaft = (self.SHAFT_1X_MS2 * np.cos(2 * np.pi * ph - 0.4 * ch)
                         + self.SHAFT_2X_MS2 * np.cos(4 * np.pi * ph - 0.9 * ch))
                acc[:, ch] = g_a * (g_b * ring + shaft) + rng.normal(0.0, self.NOISE_MS2, nb)
            v_acc = acc / G * (cfg.accel_sens_mv_per_g / 1000.0) + cfg.accel_offset_v
            prox = np.empty((nb, 2))
            for p in range(2):
                amp = self.PROX_AMP_UM[p] * (1.0 + (0.5 * self.severity if p + 1 == self.faulted_bearing else 0.0))
                x_um = amp * np.cos(2 * np.pi * ph - self.PROX_PHASE[p]) + rng.normal(0.0, self.PROX_NOISE_UM, nb)
                prox[:, p] = self.GAP_V + cfg.prox_offset_v + x_um * cfg.prox_sens_mv_per_um / 1000.0
            volts = np.zeros((nb, 8))
            volts[:, 0:4], volts[:, 4:6] = v_acc, prox
            codes = np.clip(np.rint(volts * cfg.counts_per_volt), -32768, 32767).astype(np.int16)
            yield SampleBlock(seq=seq, codes=codes, edges_s=edges)
            phase = p1
            seq += 1
            emitted += 1


class DropInjector(SampleSource):
    """Wraps a source and injects acquisition faults: random lost DMA blocks (seq gap) and missed
    Keyphasor edges. Used by the C6 window-loss simulation [SIMULATION]."""

    def __init__(self, inner: SampleSource, p_block_drop: float = 0.0, p_edge_miss: float = 0.0, seed: int = 1):
        self.inner, self.cfg = inner, inner.cfg
        self.p_block, self.p_edge = p_block_drop, p_edge_miss
        self.rng = np.random.default_rng(seed)
        self.dropped_blocks = 0
        self.missed_edges = 0

    def blocks(self) -> Iterator[SampleBlock]:
        for b in self.inner.blocks():
            if self.p_block and self.rng.random() < self.p_block:
                self.dropped_blocks += 1
                continue
            if self.p_edge and b.edges_s.size:
                keep = self.rng.random(b.edges_s.size) >= self.p_edge
                self.missed_edges += int((~keep).sum())
                b = SampleBlock(b.seq, b.codes, b.edges_s[keep])
            yield b


def record_blocks(source: SampleSource, path: str | Path, n_blocks: int) -> Path:
    """Record n blocks to an .npz (keys: codes (N,256,8) int16, seq (N,), edges_s (M,)) for replay."""
    codes, seqs, edges = [], [], []
    for i, b in enumerate(source.blocks()):
        if i >= n_blocks:
            break
        codes.append(b.codes)
        seqs.append(b.seq)
        edges.append(b.edges_s)
    path = Path(path)
    np.savez(path, codes=np.stack(codes), seq=np.asarray(seqs, dtype=np.int64),
             edges_s=np.concatenate(edges) if edges else np.zeros(0))
    return path


class FileReplaySource(SampleSource):
    """Replays blocks recorded by `record_blocks` (.npz), or a raw int16 `.npy`/`.bin` of shape
    (N,256,8) with a sibling `<stem>.edges.npy` holding absolute edge times (seq = 0..N-1)."""

    def __init__(self, path: str | Path, cfg: Optional[CoeConfig] = None, loop: bool = False):
        self.cfg, self.path, self.loop = cfg or CoeConfig(), Path(path), loop

    def _load(self):
        p = self.path
        if p.suffix == ".npz":
            z = np.load(p)
            return z["codes"], z["seq"], z["edges_s"]
        if p.suffix == ".npy":
            codes = np.load(p)
        else:
            codes = np.fromfile(p, dtype="<i2").reshape(-1, self.cfg.block_sets, 8)
        edges = np.load(p.with_suffix(".edges.npy"))
        return codes, np.arange(len(codes), dtype=np.int64), edges

    def blocks(self) -> Iterator[SampleBlock]:
        codes, seq, edges = self._load()
        dt = self.cfg.block_sets / self.cfg.fs_hz
        offset = 0
        while True:
            for c, s in zip(codes, seq):
                t0 = int(s) * dt
                m = (edges >= t0) & (edges < t0 + dt)
                yield SampleBlock(int(s) + offset, np.asarray(c, dtype=np.int16), edges[m] + offset * dt)
            if not self.loop:
                return
            offset += int(seq[-1]) + 1


class HardwareSource(SampleSource):
    """STUB for the real AD7606/STM32 node. See aiengine/coe/README.md, "Hardware day".

    Intended raw-block bring-up stream (a PROPOSAL for the firmware team; the COE document itself
    only defines the 152-byte application payload, not a raw-streaming frame). One frame per DMA
    block, little-endian:
        u32 magic 0x4B424C43, u32 seq (DMA block counter, +1 per block),
        u16 n_edges, u16 reserved, n_edges x u64 Keyphasor timer-capture ticks,
        then 4096 bytes = 256 sample sets x 8 AD7606 words x int16 (V1..V8, sample-set major).
    The timer tick frequency is sent once at connect (f64 Hz); edge_s = tick / tick_hz.
    Transport: UART, SPI-slave or UDP (a frame is ~4.1 kB so UDP needs two datagrams).

    In the delivered system the STM32 runs Components 1-9 itself and sends only the 152-byte
    payload (via ESP32-C6, Wi-Fi, UDP 5005); this raw stream exists to cross-check firmware
    features against this Python pipeline on recorded blocks.
    """

    def __init__(self, transport: str = "udp://0.0.0.0:5006", cfg: Optional[CoeConfig] = None):
        self.cfg, self.transport = cfg or CoeConfig(), transport

    def blocks(self) -> Iterator[SampleBlock]:
        raise NotImplementedError(
            "HardwareSource is a documented stub: implement the frame parser described in its docstring "
            "once the STM32 raw-block stream exists. Nothing downstream needs to change.")
