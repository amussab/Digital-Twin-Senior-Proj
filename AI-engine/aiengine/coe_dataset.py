"""COE-derived dataset [SIMULATION]/[SYNTHETIC]: the REAL CoePipeline run on simulated raw samples.

Why: the deployed v3b heads were fitted on synth_v3 PAYLOAD-level features (aiengine/synth_v3), never on the output
of the causal COE pipeline (aiengine/coe: band-pass, envelope, 20-rev windows, 4096 FFT order bands). The hardware node
produces exactly COE-pipeline features, so this module builds labelled windows the same way the node will:

    DualFaultRigSource (raw int16 blocks, 30 kS/s) -> CoePipeline (C1-C9) -> 32 features / window -> per-bearing table

One RUN = one simulated rig with its own nuisance parameters (resonance, noise, shaft vibration, off-plane transmission,
per-channel sensor gain), one speed, and a scenario: no fault, bearing 1 only, bearing 2 only, or both bearings faulted
(each with its own class and severity). Every run starts with HEALTHY_WINDOWS healthy windows (= the commissioning
capture the engine uses as baseline: median of the first 24 windows), then FAULT_WINDOWS windows with the defect(s)
present (step or ramp profile). Labels per bearing per window are therefore known exactly (generator truth). The window
at the healthy->fault switch is flagged `transition` (part of its samples were generated before the switch) and is
excluded from fitting and scoring.

Splits are BY RUN (registered in splits.json section `synth_coe` before any fit). Nothing here is a rig measurement:
the simulator's amplitudes are invented (aiengine/coe/sources.py docstring), so every number is [SIMULATION].

    python -m aiengine.coe_adapt_eval data      # generate (parallel) + register the split
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .coe.config import CoeConfig, G
from .coe.pipeline import CoePipeline
from .coe.sources import FAULT_FAMILY, SampleBlock, SimulatedRigSource
from .config import BEARING_FEATURES, SEED
from .config import UP2_CACHE_DIR as CACHE_DIR

DATASET = "synth_coe"
PROVENANCE = "[SIMULATION] raw samples from aiengine.coe.sources (DualFaultRigSource) through the real CoePipeline"
FAULTS = ["outer_race", "inner_race", "ball", "cage"]
SEVERITIES = (0.15, 0.3, 0.5, 0.75, 1.0)
# (rpm, weight, speed_policy). 1750/3600 = the COE steady modes; 1738/3624 = off-nominal INSIDE the +-1 % mode band;
# 2400/3000 = off-mode speeds that need speed_policy "range" (COE3 sweep), reported as a separate slice.
RPMS = ((1750.0, 0.34, "modes"), (3600.0, 0.34, "modes"), (1738.0, 0.08, "modes"), (3624.0, 0.08, "modes"),
        (2400.0, 0.08, "range"), (3000.0, 0.08, "range"))
SCENARIOS = (("none", 0.15), ("b1", 0.30), ("b2", 0.30), ("both", 0.25))
HEALTHY_WINDOWS = 30
FAULT_WINDOWS = 40
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.15, "test": 0.25}
N_RUNS = 420
CADENCE_H = 10.0 / 60.0   # t_hours label only (the classifier ignores time); see coe/README "COE time base"


# --------------------------------------------------------------------------- simulator generalisation

class DualFaultRigSource(SimulatedRigSource):
    """SimulatedRigSource with an independent defect on EACH bearing [SYNTHETIC].

    faults = {bearing: (class, severity)}. Bearing b's impact train excites a ring-down seen fully on bearing b's two
    accelerometers and with OFF_PLANE gain on the other bearing's (same transmission model as SimulatedRigSource).
    The first faulted bearing draws its impacts from the main RNG in exactly SimulatedRigSource's order, so a single
    fault reproduces SimulatedRigSource(fault, severity, faulted_bearing=b) bit for bit (tested); the second faulted
    bearing uses its own RNG. `severity` of each bearing may be changed while streaming (set_severity), as the demo
    scenario script does. Optional per-channel sensor gain mismatch `ch_gain` (4 values, 1 = ideal).
    """

    def __init__(self, faults: dict[int, tuple[str, float]], rpm: float = 1750.0, seed: int = 0,
                 n_blocks: Optional[int] = None, resonance_hz: float = 5000.0, rpm_jitter: float = 0.001,
                 cfg: Optional[CoeConfig] = None, ch_gain: Optional[np.ndarray] = None):
        fl = {int(b): (c, float(s)) for b, (c, s) in faults.items() if c != "healthy"}
        first = sorted(fl)[0] if fl else 1
        c0, s0 = fl.get(first, ("healthy", 0.0))
        super().__init__(c0, s0, rpm, seed=seed, n_blocks=n_blocks, faulted_bearing=first, resonance_hz=resonance_hz,
                         rpm_jitter=rpm_jitter, cfg=cfg)
        self.faults = fl
        self.sev = {b: s for b, (c, s) in fl.items()}
        self.ch_gain = None if ch_gain is None else np.asarray(ch_gain, float).reshape(4)

    def set_severity(self, bearing: int, s: float) -> None:
        if bearing in self.sev:
            self.sev[bearing] = float(s)

    def blocks(self) -> Iterator[SampleBlock]:
        cfg, rng = self.cfg, np.random.default_rng(self.seed)
        fs, nb = cfg.fs_hz, cfg.block_sets
        dt_blk = nb / fs
        phase = float(rng.uniform(0.0, 1.0))
        drift = 0.0
        order_b = sorted(self.faults)                   # first = main RNG (SimulatedRigSource order)
        rngs = {b: (rng if i == 0 else np.random.default_rng([self.seed, 7919, b])) for i, b in enumerate(order_b)}
        fam = {b: FAULT_FAMILY[self.faults[b][0]] for b in order_b}
        order = {b: cfg.orders[fam[b]] for b in order_b}
        next_ev = {}
        for b in order_b:
            next_ev[b] = (1.0 + float(rngs[b].uniform(0, 1))) / order[b]
        events = {b: [] for b in order_b}
        seq, emitted = self.start_seq, 0
        idx = np.arange(nb)
        while self.n_blocks is None or emitted < self.n_blocks:
            tb = seq * dt_blk
            drift = 0.98 * drift + rng.normal(0.0, self.rpm_jitter * 0.2)
            nominal = self.rpm_schedule(tb) if self.rpm_schedule else self.rpm
            rate = nominal * (1.0 + drift) / 60.0
            p0, p1 = phase, phase + rate * dt_blk
            edges = np.array([tb + (k - p0) / rate
                              for k in range(int(np.floor(p0)) + 1, int(np.floor(p1)) + 1)], dtype=np.float64)
            for b in order_b:
                r, f = rngs[b], fam[b]
                while next_ev[b] < p1:
                    ne = next_ev[b]
                    if ne >= p0:
                        t_ev = tb + (ne - p0) / rate
                        amp = self.IMPACT_A0 * self.sev[b] * float(r.lognormal(0.0, 0.1))
                        if f == "BPFI":
                            amp *= 0.6 + 0.4 * np.cos(2 * np.pi * ne)
                        elif f == "BSF":
                            amp *= 0.6 + 0.4 * np.cos(2 * np.pi * cfg.orders["FTF"] * ne)
                        elif f == "FTF":
                            amp *= 0.5
                        events[b].append((t_ev, amp))
                    next_ev[b] += (1.0 + float(r.normal(0.0, 0.01))) / order[b]
            t = tb + idx / fs
            ph = p0 + rate * idx / fs
            ring = {}
            for b in order_b:
                rb = np.zeros(nb)
                keep = []
                for (t0, amp) in events[b]:
                    if t0 + 6 * self.TAU_S < tb:
                        continue
                    keep.append((t0, amp))
                    d = t - t0
                    m = (d >= 0) & (d < 6 * self.TAU_S)
                    if m.any():
                        rb[m] += amp * np.exp(-d[m] / self.TAU_S) * np.sin(2 * np.pi * self.resonance_hz * d[m])
                events[b] = keep
                ring[b] = rb
            acc = np.empty((nb, 4))
            for ch in range(4):
                bearing, axis = ch // 2 + 1, ch % 2
                g_a = 1.0 if axis == 0 else self.Y_GAIN
                rr = np.zeros(nb)
                for b in order_b:
                    rr = rr + (1.0 if bearing == b else self.OFF_PLANE) * ring[b]
                shaft = (self.SHAFT_1X_MS2 * np.cos(2 * np.pi * ph - 0.4 * ch)
                         + self.SHAFT_2X_MS2 * np.cos(4 * np.pi * ph - 0.9 * ch))
                acc[:, ch] = g_a * (rr + shaft) + rng.normal(0.0, self.NOISE_MS2, nb)
            if self.ch_gain is not None:
                acc = acc * self.ch_gain[None, :]
            v_acc = acc / G * (cfg.accel_sens_mv_per_g / 1000.0) + cfg.accel_offset_v
            prox = np.empty((nb, 2))
            for p in range(2):
                amp = self.PROX_AMP_UM[p] * (1.0 + 0.5 * self.sev.get(p + 1, 0.0))
                x_um = amp * np.cos(2 * np.pi * ph - self.PROX_PHASE[p]) + rng.normal(0.0, self.PROX_NOISE_UM, nb)
                prox[:, p] = self.GAP_V + cfg.prox_offset_v + x_um * cfg.prox_sens_mv_per_um / 1000.0
            volts = np.zeros((nb, 8))
            volts[:, 0:4], volts[:, 4:6] = v_acc, prox
            codes = np.clip(np.rint(volts * cfg.counts_per_volt), -32768, 32767).astype(np.int16)
            yield SampleBlock(seq=seq, codes=codes, edges_s=edges)
            phase = p1
            seq += 1
            emitted += 1


# --------------------------------------------------------------------------- run plan

@dataclass
class RunSpec:
    run_id: int
    seed: int
    rpm: float
    speed_policy: str
    scenario: str
    b1: tuple            # (class, severity)
    b2: tuple
    profile: str         # step | ramp
    resonance_hz: float
    noise_ms2: float
    shaft_1x: float
    shaft_2x: float
    off_plane: float
    y_gain: float
    ch_gain: tuple
    split: str

    def to_dict(self) -> dict:
        return asdict(self)


def plan(n_runs: int = N_RUNS, seed: int = SEED, run_offset: int = 70000) -> list[RunSpec]:
    """Deterministic run plan. The split is assigned here BY RUN (stratified by scenario) and frozen in splits.json."""
    rng = np.random.default_rng([seed, 2026, 1024])
    rp = np.array([w for _, w, _ in RPMS])
    sc_names, sc_w = [s for s, _ in SCENARIOS], np.array([w for _, w in SCENARIOS])
    specs = []
    for r in range(n_runs):
        sc = str(rng.choice(sc_names, p=sc_w / sc_w.sum()))
        i = int(rng.choice(len(RPMS), p=rp / rp.sum()))
        rpm, _, pol = RPMS[i]
        draw = lambda: (str(rng.choice(FAULTS)), float(rng.choice(SEVERITIES)))   # noqa: E731
        b1 = draw() if sc in ("b1", "both") else ("healthy", 0.0)
        b2 = draw() if sc in ("b2", "both") else ("healthy", 0.0)
        specs.append(RunSpec(
            run_id=run_offset + r, seed=int(rng.integers(0, 2 ** 31 - 1)), rpm=float(rpm), speed_policy=pol, scenario=sc,
            b1=b1, b2=b2, profile=str(rng.choice(["step", "ramp"], p=[0.6, 0.4])),
            resonance_hz=float(rng.uniform(4000, 6500)),
            noise_ms2=float(SimulatedRigSource.NOISE_MS2 * rng.uniform(0.7, 1.4)),
            shaft_1x=float(SimulatedRigSource.SHAFT_1X_MS2 * rng.uniform(0.6, 1.6)),
            shaft_2x=float(SimulatedRigSource.SHAFT_2X_MS2 * rng.uniform(0.6, 1.6)),
            off_plane=float(rng.uniform(0.25, 0.45)), y_gain=float(rng.uniform(0.65, 0.95)),
            ch_gain=tuple(float(x) for x in rng.uniform(0.95, 1.05, 4)), split=""))
    # stratified run-level split: within each scenario, shuffle and cut 60/15/25
    srng = np.random.default_rng([seed, 31337])
    for sc in sc_names:
        ids = [k for k, s in enumerate(specs) if s.scenario == sc]
        srng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(n * SPLIT_FRACTIONS["train"])), int(round(n * SPLIT_FRACTIONS["val"]))
        for j, k in enumerate(ids):
            specs[k].split = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
    return specs


# --------------------------------------------------------------------------- one run

def simulate_run(spec: RunSpec, healthy_windows: int = HEALTHY_WINDOWS, fault_windows: int = FAULT_WINDOWS) -> dict:
    """Stream one run through the real CoePipeline. Returns arrays (window-level, both bearings)."""
    t0 = time.perf_counter()
    cfg = CoeConfig(speed_policy=spec.speed_policy)
    faults = {b: f for b, f in ((1, tuple(spec.b1)), (2, tuple(spec.b2))) if f[0] != "healthy"}
    src = DualFaultRigSource({b: (c, 0.0) for b, (c, s) in faults.items()}, rpm=spec.rpm, seed=spec.seed,
                             resonance_hz=spec.resonance_hz, cfg=cfg, ch_gain=np.asarray(spec.ch_gain))
    src.NOISE_MS2, src.SHAFT_1X_MS2, src.SHAFT_2X_MS2 = spec.noise_ms2, spec.shaft_1x, spec.shaft_2x
    src.OFF_PLANE, src.Y_GAIN = spec.off_plane, spec.y_gain
    pipe = CoePipeline(src, cfg)
    n_total = healthy_windows + fault_windows
    feats, disp, rpm, t20, sev = [], [], [], [], []
    for w in pipe.windows():
        n = len(feats)
        sev.append({b: src.sev[b] for b in faults})       # severity in force while this window was generated
        feats.append(w.features)
        disp.append(w.displacement)
        rpm.append(w.rpm)
        t20.append(w.t20_ms)
        if len(feats) >= n_total:
            break
        k = n + 1 - healthy_windows + 1                    # severity for the NEXT window
        for b, (c, s) in faults.items():
            if k <= 0:
                v = 0.0
            elif spec.profile == "step":
                v = s
            else:
                v = s * (0.3 + 0.7 * min(1.0, k / (0.5 * fault_windows)))
            src.set_severity(b, v)
    sev_arr = {b: np.array([d[b] for d in sev]) for b in faults}
    return {"spec": spec.to_dict(), "features": np.asarray(feats, np.float32), "displacement": np.asarray(disp, np.float32),
            "rpm": np.asarray(rpm), "t20_ms": np.asarray(t20, np.int64),
            "severity": {str(b): v.tolist() for b, v in sev_arr.items()},
            "stats": {"valid": pipe.stats.valid, "discarded": dict(pipe.stats.discarded)},
            "seconds": time.perf_counter() - t0}


def _worker(spec_dict: dict) -> dict:
    spec = RunSpec(**{**spec_dict, "b1": tuple(spec_dict["b1"]), "b2": tuple(spec_dict["b2"]),
                      "ch_gain": tuple(spec_dict["ch_gain"])})
    return simulate_run(spec)


def to_table(res: dict, healthy_windows: int = HEALTHY_WINDOWS) -> pd.DataFrame:
    """One run's result -> per-bearing rows (unit = synth_coe:runNNNNN:B<k>) with labels."""
    sp = res["spec"]
    F = res["features"]
    n = len(F)
    wi = np.arange(n)
    frames = []
    for b in (1, 2):
        cls, s_final = sp[f"b{b}"]
        sev = np.asarray(res["severity"].get(str(b), np.zeros(n)), float)[:n] if cls != "healthy" else np.zeros(n)
        faulty = (cls != "healthy") & (wi >= healthy_windows + 1)
        fr = pd.DataFrame(F[:, (b - 1) * 16:b * 16], columns=BEARING_FEATURES)
        fr.insert(0, "dataset", DATASET)
        fr.insert(1, "unit_id", f"{DATASET}:run{sp['run_id']:05d}:B{b}")
        fr.insert(2, "bearing_key", f"{DATASET}:run{sp['run_id']:05d}")
        fr.insert(3, "window_index", wi)
        fr.insert(4, "t_hours", wi * CADENCE_H)
        fr.insert(5, "rpm", res["rpm"])
        fr["t20_ms"] = res["t20_ms"]
        fr["bearing"] = b
        fr["run_id"] = sp["run_id"]
        fr["scenario"] = sp["scenario"]
        fr["fault_class"] = cls
        fr["observable_class"] = np.where(faulty, cls, "healthy")
        fr["severity"] = np.where(faulty, sev, 0.0)
        fr["partner_class"] = sp[f"b{3 - b}"][0]
        fr["partner_faulty"] = (sp[f"b{3 - b}"][0] != "healthy") & (wi >= healthy_windows + 1)
        fr["transition"] = wi == healthy_windows
        fr["rpm_nominal"] = sp["rpm"]
        fr["speed_policy"] = sp["speed_policy"]
        fr["profile"] = sp["profile"]
        fr["off_plane"] = sp["off_plane"]
        fr["split"] = sp["split"]
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)


def cache_path(name: str = DATASET) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def generate(n_runs: int = N_RUNS, seed: int = SEED, workers: int = 10, force: bool = False, log=print) -> pd.DataFrame:
    p = cache_path()
    meta_p = p.with_suffix(".provenance.json")
    if p.exists() and not force:
        df = pd.read_parquet(p)
        df.attrs["provenance"] = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        return df
    specs = plan(n_runs, seed)
    t0 = time.perf_counter()
    frames, stats, secs = [], {}, []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, s.to_dict()): s.run_id for s in specs}
        for i, f in enumerate(as_completed(futs)):
            res = f.result()
            frames.append(to_table(res))
            secs.append(res["seconds"])
            for k, v in res["stats"]["discarded"].items():
                stats[k] = stats.get(k, 0) + v
            if (i + 1) % 25 == 0:
                log(f"  {i + 1}/{len(specs)} runs, {time.perf_counter() - t0:.0f} s", flush=True)
    df = pd.concat(frames, ignore_index=True).sort_values(["unit_id", "window_index"]).reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prov = {"provenance": PROVENANCE, "synthetic": True, "dataset": DATASET, "n_runs": n_runs, "seed": seed,
            "healthy_windows": HEALTHY_WINDOWS, "fault_windows": FAULT_WINDOWS, "severities": list(SEVERITIES),
            "rpms": [list(r) for r in RPMS], "scenarios": [list(s) for s in SCENARIOS],
            "split_fractions": SPLIT_FRACTIONS, "generator": "aiengine/coe_dataset.py (DualFaultRigSource -> CoePipeline)",
            "pipeline_discarded": stats, "wall_seconds": round(time.perf_counter() - t0, 1),
            "sim_seconds_per_run_median": float(np.median(secs)), "rows": int(len(df)),
            "rig_windows": int(len(df) // 2), "table_hash": table_hash(df)}
    df.to_parquet(p, index=False)
    meta_p.write_text(json.dumps(prov, indent=1))
    (p.with_suffix(".plan.json")).write_text(json.dumps([s.to_dict() for s in specs], indent=0))
    df.attrs["provenance"] = prov
    return df


def table_hash(df: pd.DataFrame) -> str:
    cols = ["unit_id", "window_index", "rpm"] + BEARING_FEATURES
    d = df.sort_values(["unit_id", "window_index"])[cols]
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()[:16]


def register_split(df: pd.DataFrame, note: str = "") -> dict:
    """Freeze the run-level split in splits.json section `synth_coe` (created once; later calls verify it)."""
    from datetime import datetime, timezone

    from . import splits as S
    sp = S.load()
    units = df.groupby("unit_id")["split"].first().to_dict()
    per_run = df.groupby("bearing_key")["split"].nunique()
    if (per_run > 1).any():
        raise RuntimeError("a run maps to more than one split")
    others = S.unit_split_map({"sections": {k: v for k, v in sp["sections"].items() if k != DATASET}})
    if set(units) & set(others):
        raise RuntimeError("synth_coe unit ids clash with another splits.json section")
    sec = sp["sections"].get(DATASET)
    if sec:
        if sec["units"] != dict(sorted(units.items())):
            raise RuntimeError("splits.json section 'synth_coe' is frozen and differs from this table")
        return sp
    counts = pd.Series(units).value_counts().to_dict()
    sp["sections"][DATASET] = {
        "policy": "by_run (stratified by scenario, 60/15/25), aiengine/coe_dataset.py::plan",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {k: int(v) for k, v in counts.items()},
        "note": note,
        "units": dict(sorted(units.items())),
    }
    S.save(sp)
    return sp
