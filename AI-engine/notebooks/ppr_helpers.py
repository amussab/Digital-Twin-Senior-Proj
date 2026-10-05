"""Helpers for PPR_ICS_Evidence.ipynb (Team M001, ICS evidence for the PPR).

Every function here RECOMPUTES something from data/models on disk, or runs a live program
(dotnet test, the ASP.NET backend, the DemoMeasure client). Numbers that come from stored files
are returned with a provenance tag so the notebook can label them.

Nothing in this module changes models, model code or stored reports.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths
NB_DIR = Path(__file__).resolve().parent
AIENGINE = NB_DIR.parent                       # AI-engine/
REPO = AIENGINE.parent                         # worktree root
MODELS = AIENGINE / "models"
REPORTS = AIENGINE / "reports"
if str(AIENGINE) not in sys.path:
    sys.path.insert(0, str(AIENGINE))

SEED = 20261004
V2_TAG = "spec_check_20261005T012112Z_real_v2"
V1_TAG = "spec_check_20261004T214921Z_real"
LOBO_JSON = "model_selection_robustness_lobo_20261004T214321Z.json"
CLASSES = ["healthy", "outer_race", "inner_race", "ball", "cage"]

# Categorical palette (dataviz reference instance, light mode), fixed order.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLOR = dict(zip(CLASSES, PAL))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001 - torch is optional for most cells
        pass


def style_matplotlib() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "font.size": 9.5,
        "axes.prop_cycle": mpl.cycler(color=PAL),
    })


def resolve_data_dir() -> Path:
    """AI-engine/data is git-ignored. Look in: $AIENGINE_DATA_DIR, this worktree, the sibling
    feature/ai-engine checkout (Repo/Digital-Twin-Senior-Proj). Fails loudly if none has XJTU-SY."""
    cands = []
    if os.environ.get("AIENGINE_DATA_DIR"):
        cands.append(Path(os.environ["AIENGINE_DATA_DIR"]))
    cands += [AIENGINE / "data", REPO.parent / "Digital-Twin-Senior-Proj" / "AI-engine" / "data"]
    for c in cands:
        if (c / "raw" / "xjtu_sy").is_dir() and (c / "cache").is_dir():
            return c.resolve()
    raise FileNotFoundError("No AI-engine data dir with raw/xjtu_sy + cache found. Tried: "
                            + ", ".join(str(c) for c in cands) + ". Set AIENGINE_DATA_DIR.")


def resolve_checkpoint_dir() -> Path | None:
    for c in (AIENGINE / "checkpoints" / "real_v2",
              REPO.parent / "Digital-Twin-Senior-Proj" / "AI-engine" / "checkpoints" / "real_v2"):
        if (c / "tft.ckpt").is_file():
            return c.resolve()
    return None


def git_head() -> str:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return head + (" (+uncommitted changes)" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- results registry
@dataclass
class Item:
    id: str
    short: str
    text: str
    owner: str
    ics_role: str
    section: str = ""
    status: str = "NOT ASSESSED HERE"
    rank: str = "-"
    evidence: str = ""


SPEC_ITEMS = [
    Item("C1", "M1", "OH-2 / Goulds 3196 STi+MTi target machine; housing envelope <=10 mm/s RMS", "ME", "none"),
    Item("C2", "M2", "Mains-powered node; SELV IEPE excitation 24 V / 4 mA over coax", "COE", "none"),
    Item("C3", "M3", "Node BOM <=4,000 SAR; sell price <=8,000 SAR/node", "ME (ICS approval: No)", "none"),
    Item("C4", "C1", "All real-time DAQ + signal processing on local hardware, no cloud", "COE", "AI/host part", "2"),
    Item("C5", "I1", "Dashboard local-network only; no external/cloud exposure", "ICS", "owner", "3"),
    Item("C6", "-", "UV-stabilised weather-resistant polymer enclosure", "ME", "none"),
    Item("S1", "M4", "2x biaxial IEPE accelerometers, 4 ch (off-the-shelf)", "ME", "none"),
    Item("S2", "M5", "FE twin identifies bearing stiffness K with <=10% error", "ME", "twin code support", "9"),
    Item("S3", "M6", "Enclosure IP54, continuous operation", "ME", "none"),
    Item("S4", "C2", "Sample rate >=25 kS/s/ch, 4 ch", "COE", "none"),
    Item("S5", "C3", "Window processed, payload available <=333 ms", "COE", "none"),
    Item("S6", "C4", "Fixed-dim, ordered AI input over 1000-3600 rpm", "COE", "consumer (payload)"),
    Item("S7", "I2", "RUL MAPE <=15% on held-out degradation window (N-HiTS)", "ICS", "owner", "6"),
    Item("S8", "I3", "Fault-classification inference <200 ms/window (TFT)", "ICS", "owner", "4"),
    Item("S9", "I4", "Dashboard refresh >=10 Hz; physics+AI RUL+fault ID together", "ICS", "owner", "8"),
    Item("IS1", "N1", "Sensor -> classify+RUL -> dashboard <500 ms", "Integrated", "host leg", "8"),
    Item("IS2", "N2", "Physics-AI RUL residual <=5%, stage 1->3", "Integrated", "AI RUL + twin code", "9"),
    Item("IS3", "N3", "macro-F1 >=85% (5 classes); stage error <=1; caught by stage 3", "Integrated", "owner of models", "7"),
]
ITEMS = {i.id: i for i in SPEC_ITEMS}
ICS_ITEMS = ["C4", "C5", "S7", "S8", "S9", "IS1", "IS2", "IS3"]


def record(item_id: str, status: str, rank: str, evidence: str) -> None:
    it = ITEMS[item_id]
    it.status, it.rank, it.evidence = status, rank, evidence


def scorecard_frame() -> pd.DataFrame:
    rows = []
    for it in SPEC_ITEMS:
        if it.id in ICS_ITEMS:
            st = it.status if it.status != "NOT ASSESSED HERE" else "PENDING (cell not run yet)"
        else:
            st = f"owned by {it.owner.split(' ')[0]}, not assessed here"
        rows.append({"ID": it.id, "Old ID": it.short, "Item": it.text, "Owner dept": it.owner,
                     "ICS role": it.ics_role, "Status (this notebook)": st,
                     "Evidence rank": it.rank if it.id in ICS_ITEMS else "-",
                     "Section": it.section or "-", "Evidence": it.evidence if it.id in ICS_ITEMS else ""})
    return pd.DataFrame(rows)


def styled_scorecard():
    df = scorecard_frame()

    def color(v):
        v = str(v)
        if v.startswith("MET"):
            return "background-color:#d9f2e3;color:#0b0b0b;font-weight:600"
        if v.startswith("NOT MET"):
            return "background-color:#fbe0de;color:#0b0b0b;font-weight:600"
        if v.startswith("PARTIAL"):
            return "background-color:#fdf0cf;color:#0b0b0b;font-weight:600"
        return "color:#52514e"
    return (df.style.map(color, subset=["Status (this notebook)"]).hide(axis="index")
            .set_properties(**{"text-align": "left", "font-size": "11px"}))


C4_DECISION = ("counted for ICS: team decision 2026-10-05 (the system uses no cloud service; AI inference and "
               "dashboard run on the local host). C4 is COE-assigned in the spec sheet; the ICS evidence covers the "
               "AI/host part, the node acquisition/DSP part is COE evidence")


def is1_budget(coe_ms: float, coe_measured: bool, me_ms: float, host_p95: float, host_max: float,
               wifi_ms: float | None = None, budget_ms: float = 500.0) -> dict:
    """IS1 latency budget: COE leg + ME leg + host leg (+ Wi-Fi hop, unmeasured unless given)."""
    tot_p95, tot_max = coe_ms + me_ms + host_p95, coe_ms + me_ms + host_max
    margin = budget_ms - tot_max
    if tot_max + (wifi_ms or 0.0) >= budget_ms:
        status = f"NOT MET (COE+ME+host max = {tot_max:.1f} ms, with Wi-Fi {(wifi_ms or 0):.1f} ms >= {budget_ms:.0f} ms)"
    elif coe_measured and wifi_ms is not None:
        status = f"MET (total {tot_max + wifi_ms:.1f} ms < {budget_ms:.0f} ms)"
    elif coe_measured:
        status = f"PARTIAL: CONDITIONAL MET (COE and host legs measured; Wi-Fi hop unmeasured, <= {margin:.0f} ms allowed)"
    else:
        status = (f"PARTIAL: CONDITIONAL MET (host leg measured; COE leg at spec budget; "
                  f"Wi-Fi hop unmeasured, <= {margin:.0f} ms allowed)")
    return {"tot_p95": tot_p95, "tot_max": tot_max, "margin_max": margin, "margin_p95": budget_ms - tot_p95,
            "wifi_allowed_ms": margin, "status": status}


def ics_verdict() -> dict:
    met = lambda i: ITEMS[i].status.startswith("MET")          # noqa: E731
    cons = [i for i in ("C5", "C4") if met(i)]
    specs = [i for i in ("S7", "S8", "S9") if met(i)]
    integ_met = [i for i in ("IS1", "IS2", "IS3") if met(i)]
    integ_partial = [i for i in ("IS1", "IS2", "IS3") if ITEMS[i].status.startswith("PARTIAL")]
    ics_met_all = [i for i in ICS_ITEMS if met(i)]
    return {
        "constraints_met": cons, "specs_met": specs,
        "exemplary_dept": len(cons) > 1 and len(specs) >= 1,
        "exemplary_dept_strict_two_specs": len(cons) > 1 and len(specs) > 1,
        "integrated_met": integ_met, "integrated_partial": integ_partial,
        "ics_items_met_of_18": ics_met_all,
        "team_needed": 10,
    }


# --------------------------------------------------------------------------- supporting evidence (not MET claims)
def s6_feature_check(data: Path) -> pd.DataFrame:
    """COE S6 support: the Python reference implementation of COE Components 3-7 (aiengine.dsp) gives the same
    fixed-length, fixed-order vector at every shaft speed. Real raw records, nominal rpm from the dataset
    (MaFaulDa filename = rotation Hz; XJTU-SY condition table). NOT the STM32 firmware."""
    from aiengine import dsp, payload
    from aiengine.config import FEATURE_NAMES_PER_ACCEL
    from aiengine.datasets import common, mafaulda as M
    from aiengine.geometry import BEARING_ORDERS
    rows = []
    files = sorted((data / "raw" / "mafaulda" / "normal").glob("*.csv"), key=lambda q: float(q.stem))
    n = len(files)
    cfg = common.pipeline(M.FS, BEARING_ORDERS["mafaulda"])
    for f in [files[0], files[n // 4], files[n // 2], files[(3 * n) // 4], files[-1]]:
        rpm = float(f.stem) * 60.0
        span = dsp.cut_windows(10**9, rpm, M.FS, max_windows=1)[0][1]
        a = pd.read_csv(f, header=None, nrows=span).to_numpy(np.float64)
        b1 = dsp.bearing_features(a[:, 2], a[:, 3], rpm, cfg)      # underhang radial + tangential
        b2 = dsp.bearing_features(a[:, 5], a[:, 6], rpm, cfg)      # overhang radial + tangential
        v = np.concatenate([b1, b2])
        assert b1.shape == (16,) and v.shape == (32,) and np.isfinite(v).all(), (f, v.shape)
        blob = payload.encode(payload.Window(rpm=rpm, features=v.astype(np.float32), p1_amp_um=0.0, p1_phase_rad=0.0,
                                             p2_amp_um=0.0, p2_phase_rad=0.0, t20_ms=0))
        assert len(blob) == 152
        rows.append({"dataset": "MaFaulDa (normal)", "rpm": round(rpm), "samples in 20-rev window": span,
                     "per-bearing vector": b1.shape, "payload vector": v.shape, "payload bytes": len(blob)})
    for brg in ("Bearing1_1", "Bearing2_5", "Bearing3_4"):
        try:
            fl, rpm = xjtu_files(data, brg)
        except (FileNotFoundError, ValueError):
            continue
        f16 = raw_bearing_features(fl, rpm, n=1)
        assert f16.shape == (1, 16) and np.isfinite(f16).all()
        span = dsp.cut_windows(10**9, rpm, XJTU_FS, max_windows=1)[0][1]
        rows.append({"dataset": f"XJTU-SY {brg}", "rpm": round(rpm), "samples in 20-rev window": span,
                     "per-bearing vector": f16[0].shape, "payload vector": (32,), "payload bytes": 152})
    assert list(FEATURE_NAMES_PER_ACCEL) == ["bp_rms", "bp_kurtosis", "bp_crest", "env_rms",
                                             "ftf_mag", "bsf_mag", "bpfo_mag", "bpfi_mag"]
    return pd.DataFrame(rows)


def detect_rates(tft_preds: pd.DataFrame) -> dict:
    """Healthy-vs-fault detection from the per-window TFT predictions (any non-healthy call = alarm)."""
    y, p = tft_preds["observable_class"].astype(str), tft_preds["pred_class"].astype(str)
    h = y == "healthy"
    return {"n_healthy": int(h.sum()), "false_alarm": float((p[h] != "healthy").mean()) if h.any() else float("nan"),
            "n_fault": int((~h).sum()), "missed": float((p[~h] == "healthy").mean()) if (~h).any() else float("nan")}


# --------------------------------------------------------------------------- C4: no network
class NetworkBlocked(RuntimeError):
    pass


@contextlib.contextmanager
def no_network():
    """Make every Python-level outbound network attempt raise NetworkBlocked (DNS, connect,
    connect_ex, create_connection, sendto). The Jupyter kernel's own ZeroMQ channels are C-level
    and unaffected, so the notebook keeps running."""
    def deny(*a, **k):
        raise NetworkBlocked(f"C4 guard: network call blocked ({a[1:] if len(a) > 1 else a})")
    saved = {"connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
             "sendto": socket.socket.sendto, "create_connection": socket.create_connection,
             "getaddrinfo": socket.getaddrinfo, "gethostbyname": socket.gethostbyname}
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.socket.sendto = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny
    socket.gethostbyname = deny
    try:
        yield
    finally:
        socket.socket.connect = saved["connect"]
        socket.socket.connect_ex = saved["connect_ex"]
        socket.socket.sendto = saved["sendto"]
        socket.create_connection = saved["create_connection"]
        socket.getaddrinfo = saved["getaddrinfo"]
        socket.gethostbyname = saved["gethostbyname"]


# --------------------------------------------------------------------------- raw XJTU-SY -> features
XJTU_FS = 25_600.0
XJTU_COND = {"1": ("35Hz12kN", 2100.0), "2": ("37.5Hz11kN", 2250.0), "3": ("40Hz10kN", 2400.0)}


def xjtu_dir(data: Path, bearing: str) -> tuple[Path, float]:
    cond, rpm = XJTU_COND[bearing.replace("Bearing", "")[0]]
    root = data / "raw" / "xjtu_sy"
    if (root / "XJTU-SY_Bearing_Datasets").is_dir():
        root = root / "XJTU-SY_Bearing_Datasets"
    d = root / cond / bearing
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d, rpm


def xjtu_files(data: Path, bearing: str) -> tuple[list[Path], float]:
    d, rpm = xjtu_dir(data, bearing)
    files = sorted(d.glob("*.csv"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(1, len(files) + 1)):
        raise ValueError(f"{d}: snapshot files not numbered 1..N")
    return files, rpm


def raw_bearing_features(files: list[Path], rpm: float, n: int | None = None) -> np.ndarray:
    """COE C3-C7 pipeline (aiengine.dsp) on the FIRST 20-rev window of each 1.28 s snapshot,
    exactly as the dataset adapter does. Returns float64 [n, 16] (X's 8 then Y's 8)."""
    from aiengine import dsp
    from aiengine.datasets import common
    from aiengine.geometry import BEARING_ORDERS
    cfg = common.pipeline(XJTU_FS, BEARING_ORDERS["xjtu_sy"])
    span = dsp.cut_windows(10**9, rpm, XJTU_FS, max_windows=1)[0][1]
    out = []
    for p in files[:n]:
        a = pd.read_csv(p, nrows=span).to_numpy(np.float64)
        if a.shape != (span, 2):
            raise ValueError(f"{p}: unexpected shape {a.shape}, expected ({span}, 2)")
        out.append(dsp.bearing_features(a[:, 0], a[:, 1], rpm, cfg))
    return np.asarray(out)


def build_payloads(f1: np.ndarray, f2: np.ndarray, rpm: float) -> list[bytes]:
    """Encode raw-derived features into 152-byte COE payloads (same rule as demo_stream.build):
    rpm and t20 from bearing 1 (one snapshot per minute), displacement zeros."""
    from aiengine import payload
    out = []
    for i in range(len(f1)):
        w = payload.Window(rpm=float(rpm), features=np.concatenate([f1[i], f2[i]]).astype(np.float32),
                           p1_amp_um=0.0, p1_phase_rad=0.0, p2_amp_um=0.0, p2_phase_rad=0.0,
                           t20_ms=int(round((i / 60.0) * 3.6e6)) % 2 ** 32)
        out.append(payload.encode(w))
    return out


def load_demo_payloads() -> tuple[list[bytes], dict]:
    blob = (MODELS / "demo_payloads.bin").read_bytes()
    meta = json.loads((MODELS / "demo_payloads.json").read_text())
    if len(blob) % 152:
        raise ValueError(f"demo_payloads.bin length {len(blob)} is not a multiple of 152")
    return [blob[i:i + 152] for i in range(0, len(blob), 152)], meta


# --------------------------------------------------------------------------- engine with recording
class RecordingSession:
    """Wraps an onnxruntime session: forwards run() and keeps (feed, outputs) for later timing
    and for full TFT variable weights."""
    def __init__(self, sess):
        self.s = sess
        self.calls: list[tuple[dict, list]] = []

    def run(self, names, feed):
        out = self.s.run(names, feed)
        self.calls.append(({k: v.copy() for k, v in feed.items()}, out))
        return out

    def __getattr__(self, k):
        return getattr(self.s, k)


def run_engine(payloads: list[bytes], record_tft: bool = False):
    from aiengine.engine import HybridEngine
    eng = HybridEngine(MODELS, threads=1)
    rec = None
    if record_tft:
        rec = RecordingSession(eng.tft)
        eng.tft = rec
    outs = []
    for i, p in enumerate(payloads):
        o = eng.process_payload(p)
        if "error" in o:
            raise RuntimeError(f"engine rejected payload {i}: {o['error']}")
        outs.append(o)
    return eng, outs, rec


def engine_frame(outs: list[dict], bearing: int) -> pd.DataFrame:
    rows = []
    for i, o in enumerate(outs):
        b = o["bearings"][bearing - 1]
        r = {"i": i, "t_h": o["t20_ms"] / 3.6e6, "status": b.get("status"),
             "hi": b.get("health_index", np.nan)}
        if b.get("status") == "ok":
            r.update({"pred": b["fault_class"], "conf": b["confidence"], "stage": b["health_stage"],
                      "rul": b["rul_hours"], "onset": b["onset_detected"],
                      "lat_total": b["latency_ms"]["total"], "lat_tft": b["latency_ms"]["tft"],
                      "lat_nhits": b["latency_ms"]["nhits"], "lat_feat": b["latency_ms"]["features"]})
            r.update({f"p_{c}": v for c, v in b["class_probs"].items()})
        rows.append(r)
    return pd.DataFrame(rows)


def time_tft_onnx(feeds: list[dict], n_min: int = 200, warmup: int = 10) -> np.ndarray:
    """Fresh 1-thread ORT session on models/tft.onnx; times one single-window run per recorded
    real feed (cycled if fewer than n_min). Returns ms per call."""
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    s = ort.InferenceSession(str(MODELS / "tft.onnx"), o, providers=["CPUExecutionProvider"])
    if not feeds:
        raise RuntimeError("no recorded TFT feeds to time")
    for f in feeds[:warmup]:
        s.run(["logits", "variable_weights"], f)
    n = max(n_min, len(feeds))
    ms = np.empty(n)
    for k in range(n):
        f = feeds[k % len(feeds)]
        t0 = time.perf_counter()
        s.run(["logits", "variable_weights"], f)
        ms[k] = (time.perf_counter() - t0) * 1000
    return ms


# --------------------------------------------------------------------------- S7 / IS3 re-scoring
def load_report(tag: str) -> dict:
    return json.loads((REPORTS / f"{tag}.json").read_text())


def s7_independent(rul_preds: pd.DataFrame) -> tuple[float, dict, pd.Series]:
    """INDEPENDENT re-implementation of DESIGN.md section 7 (pre-registered):
    window = [t_onset, T_fail - 0.1 (T_fail - t_onset)] built from onset_t_hours and life_hours
    (NOT from the stored in_deg flag); per-bearing mean |pred-true|/true over that window
    (rul_true > 0); MAPE = mean over bearings."""
    on, life, t = rul_preds["onset_t_hours"], rul_preds["life_hours"], rul_preds["t_hours"]
    eps = 1e-9
    mask = (t >= on - eps) & (t <= life - 0.1 * (life - on) + eps) & (rul_preds["rul_true"] > 0)
    mask &= on.notna()
    per = {}
    for u, g in rul_preds[mask].groupby("unit_id"):
        per[u] = float(np.mean(np.abs(g["rul_pred"] - g["rul_true"]) / g["rul_true"]) * 100)
    return float(np.mean(list(per.values()))), per, mask


def is3_independent(tft_preds: pd.DataFrame, rul_preds: pd.DataFrame | None) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    y, p = tft_preds["observable_class"].astype(str), tft_preds["pred_class"].astype(str)
    out = {"n": len(tft_preds),
           "macro_f1_5": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
           "per_class_f1": {c: float(f1_score(y, p, labels=[c], average="macro", zero_division=0)) for c in CLASSES},
           "support": {c: int((y == c).sum()) for c in CLASSES},
           "confusion": confusion_matrix(y, p, labels=CLASSES), "per_dataset": {}}
    for ds, g in tft_preds.groupby("dataset"):
        yy, pp = g["observable_class"].astype(str), g["pred_class"].astype(str)
        pres = [c for c in CLASSES if (yy == c).any()]
        out["per_dataset"][ds] = {"n": len(g), "classes": pres,
                                  "macro_f1": float(f1_score(yy, pp, labels=pres, average="macro", zero_division=0)),
                                  "recall": {c: float((pp[yy == c] == c).mean()) for c in pres}}
    # caught by stage 3: run-to-failure fault units, first run of 3 consecutive correct calls
    rows = []
    for u, g in tft_preds.groupby("unit_id"):
        fc = str(g["fault_class"].iloc[0])
        if fc not in CLASSES or fc == "healthy" or g["dataset"].iloc[0] == "mafaulda":
            continue
        g = g.sort_values("window_index")
        ok = (g["pred_class"].to_numpy() == fc).astype(int)
        run = np.convolve(ok, np.ones(3, int), "valid")
        hit = np.flatnonzero(run == 3)
        at = int(hit[0] + 2) if len(hit) else None
        stage = int(g["health_stage"].iloc[at]) if at is not None else None
        rows.append({"unit_id": u, "fault_class": fc, "caught_at_window": None if at is None else int(g["window_index"].iloc[at]),
                     "caught_at_stage": stage, "caught_by_stage3": stage is not None and stage <= 3})
    out["caught"] = pd.DataFrame(rows)
    out["caught_frac"] = float(out["caught"]["caught_by_stage3"].mean()) if rows else float("nan")
    if rul_preds is not None:
        st = rul_preds[rul_preds["health_stage"] >= 1]
        e = (st["stage_pred"] - st["health_stage"]).abs()
        out["stage"] = {"max": int(e.max()), "mean": float(e.mean()), "within1": float((e <= 1).mean()), "n": int(len(e))}
    return out


# --------------------------------------------------------------------------- dotnet / backend
def dotnet_path() -> str | None:
    return shutil.which("dotnet")


def run_cmd(args: list[str], cwd: Path = REPO, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    t0 = time.time()
    r = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    r.elapsed = time.time() - t0
    if check and r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
        raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(args)}\n{tail}")
    return r


def dotnet_test(filter_expr: str, project: str = "DigitalTwin.slnx", timeout: int = 300) -> tuple[str, dict]:
    r = run_cmd(["dotnet", "test", project, "--filter", filter_expr, "--nologo",
                 "--logger", "console;verbosity=normal"], timeout=timeout)
    text = r.stdout
    lines = [ln for ln in text.splitlines()
             if re.search(r"^\s+(Passed|Failed|Skipped)\s", ln) or re.search(r"(Passed|Failed)!\s+-", ln)]
    tot = {"passed": 0, "failed": 0, "total": 0}
    for m in re.finditer(r"Failed:\s+(\d+), Passed:\s+(\d+), Skipped:\s+\d+, Total:\s+(\d+)", text):
        tot["failed"] += int(m.group(1))
        tot["passed"] += int(m.group(2))
        tot["total"] += int(m.group(3))
    if tot["total"] == 0:                       # verbose console logger prints a block per test project
        for blk in re.split(r"Test Run (?:Successful|Failed)\.", text)[1:]:
            g = lambda k: int(m.group(1)) if (m := re.search(rf"^\s*{k}:\s+(\d+)", blk, re.M)) else 0  # noqa: E731
            tot["total"] += g("Total tests")
            tot["passed"] += g("Passed")
            tot["failed"] += g("Failed")
    lines.append(f"(exit code {r.returncode})")
    if tot["total"] == 0:
        raise RuntimeError(f"no tests matched filter {filter_expr!r}:\n" + text[-3000:])
    tot["elapsed_s"] = round(r.elapsed, 1)
    return "\n".join(lines), tot


def free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


BACKEND_DIR = REPO / "src" / "DigitalTwin.Backend" / "bin" / "Release" / "net10.0"
DEMOMEASURE_DLL = REPO / "tools" / "DemoMeasure" / "bin" / "Release" / "net10.0" / "DemoMeasure.dll"


def build_release() -> dict:
    """Incremental Release build of the backend and the measurement client (a few seconds when
    up to date). Raises with the build log tail on failure."""
    a = run_cmd(["dotnet", "build", "src/DigitalTwin.Backend", "-c", "Release", "--nologo", "-v", "q"], timeout=600)
    b = run_cmd(["dotnet", "build", "tools/DemoMeasure", "-c", "Release", "--nologo", "-v", "q"], timeout=600)
    return {"backend_s": round(a.elapsed, 1), "demomeasure_s": round(b.elapsed, 1)}


class Backend:
    """Starts ONE DigitalTwin.Backend process (Release dll) and kills only that PID on stop()."""

    def __init__(self, replay: bool = True, source: str = "demo", rate_hz: float = 10, host: str = "127.0.0.1"):
        self.port, self.udp = free_port(), free_port(socket.SOCK_DGRAM)
        self.host = host
        self.url = f"http://{host}:{self.port}"
        self.args = ["dotnet", "DigitalTwin.Backend.dll", f"--Urls={self.url}",
                     f"--Backend:Udp:Port={self.udp}", f"--Backend:Replay:Enabled={'true' if replay else 'false'}",
                     f"--Backend:Replay:Source={source}", f"--Backend:Replay:RateHz={rate_hz}"]
        self.log = Path(tempfile.mkstemp(prefix="ppr_backend_", suffix=".log")[1])
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 60) -> "Backend":
        if not (BACKEND_DIR / "DigitalTwin.Backend.dll").is_file():
            raise FileNotFoundError(f"backend not built: {BACKEND_DIR}")
        self._fh = open(self.log, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.args, cwd=str(BACKEND_DIR), stdout=self._fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        import urllib.request
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"backend exited early (code {self.proc.returncode}):\n{self.read_log()[-3000:]}")
            try:
                with urllib.request.urlopen(self.url + "/api/health", timeout=2) as r:
                    if r.status == 200:
                        self.started_s = round(time.time() - t0, 1)
                        return self
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.3)
        self.stop()
        raise TimeoutError(f"backend did not answer /api/health within {timeout}s:\n{self.read_log()[-3000:]}")

    def get(self, path: str) -> tuple[int, str]:
        import urllib.request
        with urllib.request.urlopen(self.url + path, timeout=5) as r:
            return r.status, r.read().decode()

    def read_log(self) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def stop(self) -> str:
        if self.proc is None:
            return "not started"
        pid = self.proc.pid
        if self.proc.poll() is None:
            self.proc.terminate()                       # TerminateProcess on this PID only
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        self._fh.close()
        return (f"backend PID {pid} stopped (exit {self.proc.returncode}); a nonzero exit code is expected here, "
                "the process was terminated on purpose by this notebook")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def demomeasure(args: list[str], timeout: int = 180) -> str:
    if not DEMOMEASURE_DLL.is_file():
        raise FileNotFoundError(f"DemoMeasure not built: {DEMOMEASURE_DLL}")
    r = run_cmd(["dotnet", str(DEMOMEASURE_DLL)] + args, timeout=timeout)
    return r.stdout


def parse_rate(out: str) -> dict:
    m = re.search(r"(\d+) snapshots in ([\d.]+)s = ([\d.]+) Hz; inter-arrival p50=([\d.]+) ms p95=([\d.]+) ms max=([\d.]+) ms", out)
    m2 = re.search(r"per-second counts: min=(\d+) max=(\d+)", out)
    if not (m and m2):
        raise RuntimeError("could not parse DemoMeasure rate output:\n" + out[-2000:])
    return {"snapshots": int(m.group(1)), "seconds": float(m.group(2)), "hz": float(m.group(3)),
            "gap_p50_ms": float(m.group(4)), "gap_p95_ms": float(m.group(5)), "gap_max_ms": float(m.group(6)),
            "per_sec_min": int(m2.group(1)), "per_sec_max": int(m2.group(2))}


def parse_latency(out: str) -> dict:
    res = {}
    for key, pat in (("all", r"ALL n=(\d+): p50=([\d.]+) p95=([\d.]+) max=([\d.]+)"),
                     ("onnx_active", r"ONNX-active windows only \(payload >= 37\) n=(\d+): p50=([\d.]+) p95=([\d.]+) max=([\d.]+)")):
        m = re.search(pat, out)
        if not m:
            raise RuntimeError("could not parse DemoMeasure latency output:\n" + out[-2000:])
        res[key] = {"n": int(m.group(1)), "p50": float(m.group(2)), "p95": float(m.group(3)), "max": float(m.group(4))}
    return res


def parse_twin_series(out: str) -> pd.DataFrame:
    rows = [{"t_s": float(a), "ai_rul_h": float(b), "physics_rul_h": float(c), "residual_pct": float(d)}
            for a, b, c, d in re.findall(r"\[series\] t=\s*([\d.]+)s ai=([\d.]+)h physics=([\d.]+)h residual=([\d.]+)%", out)]
    return pd.DataFrame(rows)


def stored_integration_measurements() -> str:
    """Fallback only: the stored, labelled measurement file."""
    return (REPO / "evidence" / "integration_measurements_20261005.md").read_text(encoding="utf-8")


def source_excerpt(path: Path, pattern: str, before: int = 0, after: int = 12) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if re.search(pattern, ln):
            lo, hi = max(0, i - before), min(len(lines), i + after + 1)
            return "\n".join(f"{k + 1:4d}  {lines[k]}" for k in range(lo, hi))
    raise ValueError(f"pattern {pattern!r} not found in {path}")


def lan_ipv4s() -> list[str]:
    """All private IPv4 addresses of this host (VPN and Hyper-V adapters included: the presenter
    picks the one on the room network, cf. `ipconfig`)."""
    try:
        ips = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []
    return [ip for ip in ips if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", ip)]


# --------------------------------------------------------------------------- parity (Python side)
def golden_python_parity() -> dict:
    """Python HybridEngine (ONNX Runtime) vs models/golden_vectors.json -- the same file the C#
    GoldenParityTests replay. Feeds every input window in order, compares bearing-1 outputs."""
    from aiengine.engine import HybridEngine
    g = json.loads((MODELS / "golden_vectors.json").read_text())
    eng = HybridEngine(MODELS, threads=1)
    cases = {c["window_index"]: c for c in g["cases"]}
    d_prob = d_hi = d_rul = 0.0
    same_cls = same_stage = 0
    for w in g["inputs"]:
        f16 = np.asarray(w["features16"], np.float64)
        o = eng.process(w["rpm"], np.concatenate([f16, f16]), w["t20_ms"])["bearings"][0]
        c = cases.get(w["window_index"])
        if c is None:
            continue
        exp = c["output"]
        if o.get("status") != "ok":
            raise AssertionError(f"window {w['window_index']}: engine not ready, golden expects output")
        d_prob = max(d_prob, max(abs(o["class_probs"][k] - v) for k, v in exp["class_probs"].items()))
        d_hi = max(d_hi, abs(o["health_index"] - exp["health_index"]))
        d_rul = max(d_rul, abs(o["rul_hours"] - exp["rul_hours"]) / max(abs(exp["rul_hours"]), 1e-9))
        same_cls += o["fault_class"] == exp["fault_class"]
        same_stage += o["health_stage"] == exp["health_stage"]
    return {"cases": len(cases), "max_abs_prob_diff": d_prob, "max_abs_hi_diff": d_hi,
            "max_rel_rul_diff": d_rul, "class_identical": f"{same_cls}/{len(cases)}",
            "stage_identical": f"{same_stage}/{len(cases)}", "unit": g["unit_id"]}


def torch_onnx_parity(ckpt_dir: Path) -> dict:
    """PyTorch TFT checkpoint (pytorch-forecasting) vs models/tft.onnx on the golden tensors."""
    import onnxruntime as ort
    import torch
    from aiengine import export, train
    tft = train.load_model(ckpt_dir / "tft.ckpt")
    if train.is_hierarchical(tft):
        raise RuntimeError("hierarchical checkpoint: use HierTFTExport (not the v2 contract)")
    _, _, TL = train.tft_layout(tft)
    w = export.TFTExport(tft, TL, 1, [0.0, 1.0]).eval()
    s = ort.InferenceSession(str(MODELS / "tft.onnx"), providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    g = json.loads((MODELS / "golden_vectors.json").read_text())
    worst = [0.0, 0.0]
    for c in g["cases"]:
        e = np.asarray(c["tft_encoder_cont"], np.float32)[None]
        d = np.asarray(c["tft_decoder_cont"], np.float32)[None]
        with torch.no_grad():
            ref = [r.numpy() for r in w(torch.as_tensor(e), torch.as_tensor(d))]
        got = s.run(["logits", "variable_weights"], {k: v for k, v in {"encoder_cont": e, "decoder_cont": d}.items() if k in names})
        worst = [max(worst[i], float(np.abs(got[i] - ref[i]).max())) for i in range(2)]
    return {"cases": len(g["cases"]), "max_abs_logit_diff": worst[0], "max_abs_varweight_diff": worst[1],
            "checkpoint": str(ckpt_dir / "tft.ckpt")}


def poll_state(b: "Backend", max_payloads: int = 400, timeout_s: float = 40, interval_s: float = 0.03) -> pd.DataFrame:
    """Poll GET /api/state (loopback) and keep one row per new payload until the replay has
    delivered max_payloads windows or loops back. Columns from the DashboardSnapshot."""
    rows, last, t0 = [], -1, time.time()
    while time.time() - t0 < timeout_s:
        _, txt = b.get("/api/state")
        d = json.loads(txt)
        n = int(d.get("payloadCount") or 0)
        if n < last:                                   # replay looped
            break
        if n != last:
            s = d["snapshot"]
            b1 = (d.get("bearings") or [{}])[0] or {}
            rows.append({"payload": n, "ai_rul_h": s.get("aiRulHours"), "physics_rul_h": s.get("physicsRulHours"),
                         "residual_pct": s.get("rulResidualPercent"), "fault": s.get("faultClass"),
                         "stage": (b1.get("details") or {}).get("healthStage"), "machine": s.get("machineId")})
            last = n
            if n >= max_payloads:
                break
        time.sleep(interval_s)
    return pd.DataFrame(rows)
