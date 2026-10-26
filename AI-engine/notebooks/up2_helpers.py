"""Helpers for UP2_Evidence.ipynb (Team M001, UP2 live scorecard, 18 items).

Every function RECOMPUTES something from data/models/reports on disk or runs a live program. Cells never crash the
notebook when an input is missing: they return a dict with ``status: "missing"`` / ``"error"`` and the master table
shows that instead of a number. Nothing here changes models, model code or stored reports.

Rank ladder: .claude/rules/evidence-strength.md (0-9). Provenance tags: [SYNTHETIC] [SIMULATION] [MEASURED host]
[MEASURED on <public dataset>] [CALCULATED] [CITED].
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import ppr_helpers as P          # reuse: paths, Backend, dotnet_test, demomeasure, parse_*, style, palette
from ppr_helpers import AIENGINE, REPO, REPORTS, MODELS, PAL, INK, INK2, GRID  # noqa: F401

EVIDENCE = REPO / "evidence"
PY = sys.executable

# --------------------------------------------------------------------------- file discovery
def newest(directory: Path, pattern: str) -> Path | None:
    """Newest file by name (names carry a UTC stamp) then mtime. None if nothing matches."""
    fs = sorted(Path(directory).glob(pattern), key=lambda p: (p.name, p.stat().st_mtime))
    return fs[-1] if fs else None


def stamp(p: Path | None) -> str:
    if p is None:
        return "missing"
    return f"{p.name} (file mtime {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M})"


def load_json(p: Path | None) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p else None
    except Exception:  # noqa: BLE001
        return None


def miss(what: str) -> dict:
    return {"status": "missing", "what": what}


def err(what: str, e: BaseException) -> dict:
    return {"status": "error", "what": what, "error": f"{type(e).__name__}: {str(e)[:300]}"}


# --------------------------------------------------------------------------- the 18 items (exact wording: Specifications.md)
ITEMS = [  # id, short text, owner dept, PPR status (old, from FACTS.md "Final PPR state")
    ("C1", "OH-2 (API 610) pumps, Goulds 3196 STi+MTi class: drivers <=40/~122 HP, 1750/3500 rpm, housing <=10 mm/s RMS", "ME", "NOT EVIDENCED"),
    ("C2", "Mains-powered node; SELV (<=50 V DC) IEPE excitation 24 V/4 mA over coax, AC-coupled to ADC range", "COE", "NOT EVIDENCED"),
    ("C4", "All real-time DAQ and signal processing run on local hardware, no cloud reliance", "COE (counted for ICS)", "MET"),
    ("C5", "Dashboard accessible only on the local network; no external/cloud exposure of vibration data", "ICS", "MET"),
    ("C6", "Enclosure from UV-stabilized weather-resistant polymer (ABS/PETG)", "ME", "NOT EVIDENCED"),
    ("S1", "2x biaxial IEPE sets (4 ch): 100 mV/g, +-50 g, flat 2-10 kHz (off-the-shelf)", "ME", "NOT EVIDENCED"),
    ("S2", "FE digital twin identifies bearing stiffness K, <=10 % error vs known-stiffness reference", "ME", "NOT EVIDENCED"),
    ("S3", "Enclosure IP54 (IEC 60529), 85 C ambient, UV", "ME", "NOT EVIDENCED"),
    ("S4", "Sample rate >=25 kS/s per channel, 4 channels", "COE", "NOT EVIDENCED"),
    ("S5", "20-rev window processed, payload available within 333 ms", "COE", "NOT EVIDENCED"),
    ("S6", "AI-ready representation, fixed dimensionality and ordering, independent of speed over 1000-3600 rpm", "COE", "PARTIAL"),
    ("S7", "Pump-bearing RUL MAPE <=15 % on held-out degradation window (N-HiTS)", "ICS", "NOT MET"),
    ("S8", "Fault-classification inference <200 ms/window (TFT)", "ICS", "MET"),
    ("S9", "Dashboard refresh >=10 Hz; physics RUL + AI RUL + fault ID updated together within IS1 budget", "ICS", "MET"),
    ("IS1", "Sensor -> classify+RUL -> dashboard < 500 ms", "Integrated", "PARTIAL"),
    ("IS2", "Physics-AI residual <=5 % (max relative error FE-identified physics RUL vs AI RUL), stage 1->3", "Integrated", "NOT MET"),
    ("IS3", "macro-F1 >=85 % over {Healthy,OD,ID,BD,CF}; stage error <=1; any injected fault caught by stage 3", "Integrated", "NOT MET"),
]
EXCLUDED = {"C3": "Node BOM <= 4,000 SAR: EXCLUDED by the reviewer ('constraint not related to CS'); not one of the counted items."}
N_ITEMS = len(ITEMS)                       # 17
BAR = int(0.75 * N_ITEMS) + 1              # MORE THAN 75 % -> 13 of 17

# EASY UPDATE POINT: when an owner supplies evidence, add an entry here, e.g.
#   OVERRIDES["S1"] = {"status": "MET [CITED]", "value": "datasheet 100 mV/g", "target": "100 mV/g", "rank": 3, "firm": False,
#                      "prov": "[CITED] datasheet", "source": "ME/S1_datasheet.pdf"}
# Any field of the master table row can be overridden (status, value, target, note, rank, achievable, step, prov, source, firm).
# Counts, the >75 % verdict and the firm / design-level tally are all recomputed from the master table.
OVERRIDES: dict[str, dict] = {}

ICS_OWN = {"constraint": ["C5"], "specs": ["S7", "S8"]}   # the department's 1 constraint + 2 specs (S9 approval still 'No')

REG: dict[str, dict] = {}


def record(item_id: str, **kw) -> None:
    """kw: status, value, target, verdict_note, rank, achievable, step, prov, source, live (bool)."""
    REG.setdefault(item_id, {}).update(kw)


# --------------------------------------------------------------------------- ML: live re-score of saved predictions
def rescore_v3b() -> dict:
    """Re-score the SAVED v3b predictions on the FRESH synth_v3_test2 runs, with the code of `python -m aiengine.up2_eval`
    (up2b.load_test2 + up2_eval.score_set). No retraining; the data table is regenerated deterministically from its seed."""
    try:
        from aiengine import up2b
        from aiengine import up2_eval as E
        t0 = time.perf_counter()
        idx = E.latest_index("gbm", prefix="up2_v3b")
        if idx is None:
            return miss("saved v3b predictions (reports/up2_preds/up2_v3b_gbm_*.index.json)")
        t2 = up2b.load_test2()
        cp, rp = E.load_preds(idx, "test2")
        s = E.score_set(cp, rp, t2, 24)
        h = rp[rp["window_index"] >= 24].merge(t2[["unit_id", "window_index", "run_to_failure", "health_stage"]], on=["unit_id", "window_index"])
        h = h[~h["run_to_failure"]]
        healthy_stage2 = float((h["stage_pred"] == 2).mean()) if len(h) else None
        healthy_true_stage1 = float((h["health_stage"] == 1).mean()) if len(h) else None
        # per-window RUL/stage frame of the run-to-failure bearings (for the charts; same windows score_set uses)
        from aiengine import up2
        from aiengine.up2_train import TRUTH
        rr = up2.merge_truth(rp[rp["window_index"] >= 24], t2, TRUTH)
        rr = rr[rr["run_to_failure"]].copy()
        return {"rr": rr, "per_unit_caught": s["caught"].get("per_unit", []),
                "healthy_stage2_share": healthy_stage2, "healthy_true_stage1_share": healthy_true_stage1, "healthy_windows": int(len(h)),
                "status": "ok", "index": idx["index_file"], "seconds": round(time.perf_counter() - t0, 1),
                "S7": s["rul"]["S7_mape_pct"], "S7_strict": s["rul"]["alt_mape_true_onset_pct"],
                "S7_pol": s["rul"]["alt_percent_of_life_error_pts"], "n_rtf": s["rul"]["n_bearings"],
                "F1": s["classification"]["macro_f1"], "acc": s["classification"]["accuracy"],
                "per_class_f1": s["classification"]["per_class_f1"],
                "stage_max": s["stage"]["max_abs_error"], "stage_within1": s["stage"]["within1_frac"],
                "caught": s["caught"]["caught_by_stage3"], "rtf": s["caught"]["rtf_bearings"],
                "confusion": s["classification"]["confusion"], "table_hash": t2.attrs.get("provenance", {}).get("table_hash"),
                "windows": int(len(t2)), "score": s}
    except Exception as e:  # noqa: BLE001
        return err("v3b live re-score", e)


def stored_spec_check() -> tuple[Path | None, dict | None]:
    p = newest(REPORTS, "up2_spec_check_*.json")
    return p, load_json(p)


def s8_latency_live(n: int = 300) -> dict:
    """S8: classifier ONNX (v3b contract), 1 intra-op thread, batch 1, real encoder windows of a [SYNTHETIC] run."""
    try:
        from aiengine import up2, up2_eval as E, up2b
        from aiengine import up2_window as uw
        t2 = up2b.load_test2()
        unit = t2[t2["run_to_failure"]].groupby("unit_id").size().idxmax()
        g = t2[t2["unit_id"] == unit].sort_values("window_index")
        rows = g[up2.COLUMNS].to_numpy(np.float64)
        c = json.loads((MODELS / "model_contract_v3b.json").read_text())
        T = c["models"]["tft"]["encoder_length"] + 1
        bc, _ = uw.blocks(rows, T)
        k = lambda i: T + i % (len(bc) - T)  # noqa: E731
        cls, cls_ms = _onnx_times(MODELS / c["models"]["tft"]["onnx"], lambda i: {
            "encoder_cont": bc[k(i): k(i) + 1, :-1], "decoder_cont": bc[k(i): k(i) + 1, -1:]}, n=n)
        from aiengine.engine import HybridEngine
        # same as HybridEngine.benchmark(g, n=200) (first 260 windows, drop 20 warm-up), keeping the samples for the chart
        outs = HybridEngine(MODELS, contract_file="model_contract_v3b.json").run_unit(g)[: 200 + 60]
        step_ms = np.asarray([o["latency_ms"]["total"] for o in outs if o.get("status") == "ok"][20:], float)
        step = {"p50_ms": float(np.percentile(step_ms, 50)), "p95_ms": float(np.percentile(step_ms, 95)), "max_ms": float(step_ms.max()),
                "mean_ms": float(step_ms.mean()), "n": int(len(step_ms)), "note": "per bearing-window, features+classifier+RUL+post via ONNX Runtime"}
        return {"status": "ok", "cls": cls, "cls_ms": cls_ms, "engine_step": step, "step_ms": step_ms, "unit": unit, "contract": "model_contract_v3b.json"}
    except Exception as e:  # noqa: BLE001
        return err("S8 live latency", e)


def _onnx_times(path: Path, feeds_fn, n: int = 300) -> tuple[dict, np.ndarray]:
    """Same procedure as aiengine.up2_eval.onnx_latency (fresh session, 1 intra-op thread, batch 1, 20 warm-up calls
    dropped) but also returns the per-call samples so the notebook can plot the distribution."""
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
    names = {x.name for x in s.get_inputs()}
    ts = []
    for i in range(n + 20):
        f = {k: v for k, v in feeds_fn(i).items() if k in names}
        t0 = time.perf_counter()
        s.run(None, f)
        if i >= 20:
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return ({"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
             "n": int(len(a)), "threads": 1, "batch": 1}, a)


# --------------------------------------------------------------------------- COE live
def coe_live(quick: bool = True) -> dict:
    try:
        from aiengine.coe import verify as V
        t0 = time.perf_counter()
        c1 = V.coe1(seconds=4.0 if quick else 10.0)
        c2 = V.coe2(40 if quick else 240)
        c3 = V.coe3()
        return {"status": "ok", "seconds": round(time.perf_counter() - t0, 1), "COE1": c1, "COE2": c2, "COE3": c3}
    except Exception as e:  # noqa: BLE001
        return err("COE verify (quick)", e)


def coe_stored() -> tuple[Path | None, dict | None]:
    p = REPORTS / "coe_verification.json"
    return (p if p.exists() else None), load_json(p if p.exists() else None)


def coe_stage_figure(out_png: Path | None = None, rpm: float = 1750.0, fault: str = "outer_race", severity: float = 0.6):
    """Pipeline stages drawn from the REAL coe code (SimulatedRigSource -> CoePipeline filters): raw waveform, band-passed,
    envelope, envelope spectrum with FTF/BSF/BPFO/BPFI marks. [SIMULATION] [SYNTHETIC] source, not a rig measurement."""
    import matplotlib.pyplot as plt
    from scipy import signal
    from aiengine.coe.pipeline import CoePipeline
    from aiengine.coe.sources import SimulatedRigSource
    n_blocks = int(1.6 * 30000 / 256)
    src = SimulatedRigSource(fault, severity, rpm, seed=20261024, n_blocks=n_blocks)
    pipe = CoePipeline(src)
    acc = np.concatenate([pipe.calibrate(b.codes)[0] for b in src.blocks()], axis=0)    # Component 2, m/s2, (N,4)
    bp = signal.sosfilt(pipe._sos_bp, acc, axis=0)                                       # Component 3
    env = signal.sosfilt(pipe._sos_lp, np.abs(bp), axis=0)[:: pipe.cfg.decimation]       # Components 4+5
    fs, fse = pipe.cfg.fs_hz, pipe.cfg.fs_hz / pipe.cfg.decimation
    ch = 0
    t = np.arange(acc.shape[0]) / fs
    seg = slice(int(0.30 * fs), int(0.30 * fs) + int(0.06 * fs))
    e = env[int(0.2 * fse):, ch]
    e = (e - e.mean()) * np.hanning(len(e))
    nfft = 1 << int(np.ceil(np.log2(len(e) * 4)))
    sp = np.abs(np.fft.rfft(e, nfft)); fr = np.fft.rfftfreq(nfft, 1 / fse)
    fig, ax = plt.subplots(4, 1, figsize=(11.5, 10))
    ax[0].plot(t[seg] * 1e3, acc[seg, ch], color=PAL[0], lw=0.8); ax[0].set_title(f"1. Raw accelerometer, B1-X (calibrated, 30 kS/s)  [{fault}, {rpm:.0f} rpm]")
    ax[0].set_ylabel("m/s2")
    ax[1].plot(t[seg] * 1e3, bp[seg, ch], color=PAL[1], lw=0.8); ax[1].set_title("2. Band-pass 2-8 kHz (Component 3, same filter as the pipeline)"); ax[1].set_ylabel("m/s2")
    te = np.arange(len(env)) / fse
    s2 = slice(int(0.30 * fse), int(0.30 * fse) + int(0.06 * fse))
    ax[2].plot(te[s2] * 1e3, env[s2, ch], color=PAL[2], lw=1.1); ax[2].set_title("3. Envelope: rectify + low-pass + decimate x6 (Components 4-5, 5 kS/s)")
    ax[2].set_ylabel("m/s2"); ax[2].set_xlabel("time in record, ms")
    m = fr < 6 * pipe.cfg.orders["BPFI"] * rpm / 60 / 1.0
    ax[3].plot(fr[m], sp[m], color=PAL[6], lw=1.0)
    fr_rot = rpm / 60.0
    for j, (name, col) in enumerate(zip(("FTF", "BSF", "BPFO", "BPFI"), (PAL[3], PAL[4], PAL[7], PAL[5]))):
        f = pipe.cfg.orders[name] * fr_rot
        ax[3].axvline(f, color=col, ls="--", lw=1.2)
        ax[3].text(f, ax[3].get_ylim()[1] * (0.95 if j % 2 == 0 else 0.62), f" {name} {f:.0f} Hz", color=col, fontsize=9.5, va="top", fontweight="bold",
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    ax[3].set_title("4. Envelope spectrum (Component 6): outer-race fault, look for energy at BPFO (defect order x shaft speed)"); ax[3].set_xlabel("Hz"); ax[3].set_ylabel("amplitude")
    fig.suptitle("Raw vibration -> features: COE pipeline stages from the real aiengine.coe code  [SIMULATION]+[SYNTHETIC] source, not a rig measurement",
                 fontsize=12, y=0.995, x=0.01, ha="left")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=130)
    return fig


# --------------------------------------------------------------------------- tests (live)
def pytest_live(paths: list[str], timeout: int = 900) -> dict:
    try:
        t0 = time.time()
        r = subprocess.run([PY, "-m", "pytest", *paths, "-q", "-x", "--no-header", "-p", "no:cacheprovider"], cwd=str(AIENGINE),
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        tail = re.sub(r"\[[0-9;]*m", "", (r.stdout.strip().splitlines() or [""])[-1])
        m_p, m_f = re.search(r"(\d+) passed", tail), re.search(r"(\d+) failed", tail)
        return {"status": "ok", "exit": r.returncode, "passed": int(m_p.group(1)) if m_p else 0,
                "failed": int(m_f.group(1)) if m_f else 0, "tail": tail, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return err("pytest " + " ".join(paths), e)


def c5_live() -> dict:
    """C5: backend LAN-guard unit tests (HTTP 403 for public addresses, UDP public senders dropped), run live."""
    if P.dotnet_path() is None:
        return miss("dotnet SDK")
    try:
        summary, tot = P.dotnet_test("FullyQualifiedName~LanOnlyTests|FullyQualifiedName~UdpLanGuardTests",
                                     project="tests/DigitalTwin.Backend.Tests")
        return {"status": "ok", **tot}
    except Exception as e:  # noqa: BLE001
        return err("C5 dotnet test", e)


def c4_no_network_live() -> dict:
    """C4: run the v3b engine over the demo stream with sockets/DNS disabled (P.no_network guard)."""
    try:
        from aiengine.engine import HybridEngine
        import socket
        with P.no_network():
            blocked = False
            try:
                socket.getaddrinfo("pypi.org", 443)
            except P.NetworkBlocked:
                blocked = True
            eng = HybridEngine(MODELS, contract_file="model_contract_v3b.json")
            raw = (MODELS / "up2_demo_payloads.bin").read_bytes()
            n = min(120, len(raw) // 152)
            out = None
            for k in range(n):
                out = eng.process_payload(raw[k * 152:(k + 1) * 152]) if hasattr(eng, "process_payload") else None
        return {"status": "ok", "guard_blocks_dns": blocked, "windows": n, "ran": out is not None}
    except Exception as e:  # noqa: BLE001
        return err("C4 offline run", e)


# --------------------------------------------------------------------------- live backend (S9 / IS1)
CONTRACT_V3B = MODELS / "model_contract_v3b.json"


class UP2Backend(P.Backend):
    """P.Backend + the v3b contract + physics twin off (same flags as scripts/run-up2-demo.ps1)."""

    def __init__(self, replay: bool, source: str = "up2", rate_hz: float = 10):
        super().__init__(replay=replay, source=source, rate_hz=rate_hz)
        self.args += [f"--Backend:ModelContractPath={CONTRACT_V3B}", "--Backend:PhysicsTwin=None"]


def dotnet_ready() -> dict:
    if P.dotnet_path() is None:
        return miss("dotnet SDK on PATH")
    try:
        b = P.build_release()
        return {"status": "ok", **b}
    except Exception as e:  # noqa: BLE001
        # build can fail only because another process holds the dll (a running backend); use the existing Release build
        if (P.BACKEND_DIR / "DigitalTwin.Backend.dll").is_file() and P.DEMOMEASURE_DLL.is_file():
            return {"status": "ok", "note": "rebuild failed (" + str(e)[:160].replace(chr(10), " ") + "); using the EXISTING Release build, mtime "
                    + datetime.fromtimestamp((P.BACKEND_DIR / "DigitalTwin.Backend.dll").stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
        return err("dotnet build (backend + DemoMeasure)", e)


def s9_live(seconds: int = 20) -> dict:
    try:
        be = UP2Backend(replay=True, source="up2", rate_hz=10).start()
        try:
            out = P.demomeasure(["rate", be.url, str(seconds)])
            _, metrics = be.get("/api/metrics")
        finally:
            be.stop()
        return {"status": "ok", **P.parse_rate(out), "raw": out.strip()[:600], "metrics": json.loads(metrics)}
    except Exception as e:  # noqa: BLE001
        return err("S9 live rate", e)


def is1_live(n: int = 120, wifi=(10, 60), hz: float = 20) -> dict:
    """IS1 live: UDP -> backend (v3b) -> client, [SIMULATION] 10-60 ms Wi-Fi hop, COE leg from coe_verification.json."""
    try:
        coe = REPORTS / "coe_verification.json"
        if not coe.exists():
            return miss("coe_verification.json")
        be = UP2Backend(replay=False).start()
        out_md = Path(tempfile.gettempdir()) / f"is1_live_{int(time.time())}.md"
        try:
            out = P.demomeasure(["is1", be.url, str(be.udp), str(MODELS / "up2_demo_payloads.bin"), str(n),
                                 str(wifi[0]), str(wifi[1]), str(coe), str(out_md), str(hz), "1"], timeout=300)
        finally:
            be.stop()
        txt = out_md.read_text(encoding="utf-8") if out_md.exists() else out
        rows = {}
        for key, pat in (("wifi", r"\| Wi-Fi hop[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"),
                         ("backend", r"\| UDP \+ backend[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"),
                         ("sum_stm32", r"measured legs \+ COE STM32 pessimistic[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"),
                         ("sum_host", r"measured legs \+ COE host[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")):
            m = re.search(pat, txt)
            if m:
                rows[key] = {"p50": float(m.group(1)), "p95": float(m.group(2)), "max": float(m.group(3))}
        if "sum_stm32" not in rows:
            return {"status": "error", "what": "IS1 live", "error": "could not parse DemoMeasure output", "raw": txt[-800:]}
        return {"status": "ok", "legs": rows, "n": n, "report_md": txt}
    except Exception as e:  # noqa: BLE001
        return err("IS1 live", e)


def stored_integration() -> tuple[Path | None, dict]:
    """Newest evidence/integration_measurements_up2_*.md parsed for the S9 rate and IS1 totals."""
    p = newest(EVIDENCE, "integration_measurements_up2_*.md")
    if p is None:
        return None, {}
    t = p.read_text(encoding="utf-8")
    out = {}
    m = re.search(r"measured legs \+ COE STM32 pessimistic[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", t)
    if m:
        out["is1"] = {"p50": float(m.group(1)), "p95": float(m.group(2)), "max": float(m.group(3))}
    m = re.search(r"Client snapshot rate[^|]*\|\s*([\d.]+) Hz", t)
    if m:
        out["s9_hz"] = float(m.group(1))
    out["text"] = t
    return p, out


# --------------------------------------------------------------------------- IS2
def residual_latest() -> tuple[Path | None, dict | None]:
    cands = list(REPORTS.glob("up2_residual_*.json")) + list(REPORTS.glob("up2_is2_*.json"))
    p = max(cands, key=lambda q: q.stat().st_mtime) if cands else None
    return p, load_json(p)


def residual_headline(d: dict, md_path: Path | None = None) -> dict:
    """Headline IS2 numbers in PERCENT. Primary: sweep entry 'nominal' -> score.residual (fractions in the JSON).
    Fallback: the 'IS2 literal' row of the sibling .md report."""
    try:
        sc = next(s for s in d["sweep"] if s.get("name") == "nominal")["score"]
        r = sc["residual"]
        return {"max": 100 * r["max"], "p95": 100 * r["p95"], "median": 100 * r["median"], "share5": 100 * r["frac_le_5pct"],
                "coverage": 100 * sc["coverage"], "n": r["n"], "src": "json sweep.nominal.score.residual"}
    except Exception:  # noqa: BLE001
        pass
    try:
        t = md_path.read_text(encoding="utf-8")
        m = re.search(r"IS2 literal[^|]*\|\s*([\d.]+) %", t)
        if m:
            return {"max": float(m.group(1)), "src": "md 'IS2 literal' row"}
    except Exception:  # noqa: BLE001
        pass
    return {"max": None, "src": "unparsed"}


# --------------------------------------------------------------------------- S2 (stored twin verification)
def s2_stored() -> dict:
    p = REPORTS / "twin_s2_verification.json"
    d = load_json(p if p.exists() else None)
    if not d:
        return miss("twin_s2_verification.json")
    c = pd.DataFrame(d["cases"])
    n = c[c.noise != "none"]
    r1750, r3600 = n[n.rpm == 1750.0], n[n.rpm == 3600.0]
    return {"status": "ok", "file": stamp(p), "label": d.get("label"), "mean_1750": float(r1750.err_mean_pct.max()),
            "p95_1750": float(r1750.err_p95_pct.max()), "max_1750": float(r1750.err_max_pct.max()),
            "mean_3600": float(r3600.err_mean_pct.max()), "noise": d.get("noise_model")}


# --------------------------------------------------------------------------- master table
def verdict_counts(df: pd.DataFrame) -> dict:
    s = df["UP2 status"]
    met = s.str.startswith("MET").sum()
    part = s.str.startswith("PARTIAL").sum()
    notmet = s.str.startswith("NOT MET").sum()
    noev = s.str.startswith("NOT EVIDENCED").sum()
    firm = int(((df["UP2 status"].str.startswith("MET")) & (df["Firm?"] == "yes")).sum())
    return {"firm": firm, "design_level": int(met) - firm, "MET": int(met), "PARTIAL": int(part), "NOT MET": int(notmet), "NOT EVIDENCED": int(noev), "total": len(df),
            "pct_met": 100.0 * met / len(df), "bar": BAR, "gt75": bool(met >= BAR)}


def master_frame() -> pd.DataFrame:
    rows = []
    for iid, text, owner, ppr in ITEMS:
        r = {**REG.get(iid, {}), **OVERRIDES.get(iid, {})}
        rows.append({"ID": iid, "Item": text, "Owner dept": owner, "PPR status (old)": ppr,
                     "UP2 status": r.get("status", "NOT EVIDENCED"), "Measured": r.get("value", "-"),
                     "Target": r.get("target", "-"), "Verdict / note": r.get("note", ""),
                     "Model actually deployed": r.get("deployed", "-"), "Spec-named model (N-HiTS/TFT), same data": r.get("specnamed", "-"),
                     "Firm?": ("yes" if r.get("firm") else "no"), "Rank achieved": r.get("rank", 0), "Rank achievable": r.get("achievable", "-"),
                     "Step that raises it": r.get("step", ""), "Provenance": r.get("prov", "-"), "Source": r.get("source", "-")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- demo replay timeline (picture + data)
def demo_timeline(out_png: Path | None = None):
    """Run the v3b engine over the [SYNTHETIC] up2 demo stream (what the dashboard replays) and plot class / stage / HI / RUL
    per window for bearing 1 and 2. Returns (DataFrame, figure). This is the same stream and contract the backend uses."""
    import matplotlib.pyplot as plt
    from aiengine.engine import HybridEngine
    meta = load_json(MODELS / "up2_demo_payloads.json") or {}
    eng = HybridEngine(MODELS, contract_file="model_contract_v3b.json")
    raw = (MODELS / "up2_demo_payloads.bin").read_bytes()
    rows = []
    for k in range(len(raw) // 152):
        o = eng.process_payload(raw[k * 152:(k + 1) * 152])
        for b in o["bearings"]:
            if b.get("status") == "ok":
                rows.append({"i": k, "t_h": o["t20_ms"] / 3.6e6, "bearing": b["bearing"], "cls": b["fault_class"],
                             "stage": b["health_stage"], "hi": b["health_index"], "rul_h": b["rul_hours"],
                             "lat_ms": b["latency_ms"]["total"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(3, 1, figsize=(11.5, 8), sharex=True)
    for bn, col in ((1, PAL[0]), (2, PAL[1])):
        d = df[df.bearing == bn]
        ax[0].plot(d.t_h, d.rul_h, color=col, label=f"bearing {bn}")
        ax[1].plot(d.t_h, d.stage, color=col, drawstyle="steps-post")
        ax[2].plot(d.t_h, d.hi, color=col)
    cm = {c: i for i, c in enumerate(["healthy", "outer_race", "inner_race", "ball", "cage"])}
    d1 = df[df.bearing == 1]
    ax[0].set_ylabel("AI RUL, h"); ax[0].legend(); ax[1].set_ylabel("health stage"); ax[2].set_ylabel("health index"); ax[2].set_xlabel("stream time, h")
    chg = d1[d1.cls != d1.cls.shift()]
    last_lab = -1e9
    span = float(d1.t_h.max() - d1.t_h.min()) if len(d1) else 1.0
    for _, r in chg.iterrows():
        ax[0].axvline(r.t_h, color=INK2, lw=0.6, ls=":")
        if r.t_h - last_lab > 0.06 * span:          # label only well-separated class changes (avoid overprinting)
            ax[0].text(r.t_h, ax[0].get_ylim()[1] * 0.97, " " + r.cls, fontsize=9, rotation=90, va="top"); last_lab = r.t_h
    ax[0].set_title("What the dashboard replays: bearing 1 degrades (RUL falls, stage rises); bearing 2 shown for contrast\n"
                    "[SYNTHETIC] stream through the deployed v3b engine [MEASURED host]; dotted = class change of bearing 1", loc="left", fontsize=11.5)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=130)
    return df, fig, meta


# --------------------------------------------------------------------------- live ONNX run of the DEPLOYED heads
def live_engine_run(step: int = 5, n_healthy: int = 8) -> dict:
    """Runs the DEPLOYED v3b engine (ONNX heads + features + post-process, `aiengine.engine.HybridEngine`) live on
    freshly regenerated [SYNTHETIC] synth_v3_test2 windows: every `step`-th run-to-failure bearing plus `n_healthy`
    healthy-companion bearings. The full metric set (S7 MAPE, macro-F1, stage error, caught-by-stage-3, latency) is
    recomputed here FROM THE MODEL OUTPUTS (not from stored predictions). It is a SUBSET of test2 (labelled with n)."""
    try:
        from aiengine import up2, up2b, up2_metrics as M
        from aiengine.config import MODELS_DIR
        from aiengine.engine import HybridEngine
        from aiengine.up2_train import TRUTH
        t0 = time.perf_counter()
        t2 = up2b.load_test2()
        rtf = sorted(t2[t2["run_to_failure"]].unit_id.unique())[::step]
        hl = sorted(t2[~t2["run_to_failure"]].unit_id.unique())[: n_healthy]
        eng = HybridEngine(MODELS_DIR, contract_file="model_contract_v3b.json")
        cls_rows, rul_rows, lat = [], [], []
        for u in rtf + hl:
            g = t2[t2.unit_id == u].sort_values("window_index")
            for wi, o in zip(g.window_index.to_numpy(), eng.run_unit(g, 1)):
                if o.get("status") != "ok" or wi < 24:
                    continue
                cls_rows.append({"unit_id": u, "window_index": int(wi), "pred_class": o["fault_class"]})
                rul_rows.append({"unit_id": u, "window_index": int(wi), "rul_pred_h": o["rul_hours"], "stage_pred": o["health_stage"]})
                lat.append(o["latency_ms"]["total"])
        cp, rp = pd.DataFrame(cls_rows), pd.DataFrame(rul_rows)
        c = up2.merge_truth(cp, t2, TRUTH)
        r = up2.merge_truth(rp, t2, TRUTH)
        rr = r[r["run_to_failure"]]
        j = c.merge(rp[["unit_id", "window_index", "stage_pred"]], on=["unit_id", "window_index"], how="left")
        out = {"status": "ok", "n_rtf": len(rtf), "n_healthy": len(hl), "windows": int(len(cp)), "seconds": round(time.perf_counter() - t0, 1),
               "rul": M.rul(rr), "cls": M.classification(c), "stage": M.stage(rr), "caught": M.caught_by_stage3(j),
               "lat_p50": float(np.percentile(lat, 50)), "lat_p95": float(np.percentile(lat, 95)), "lat_max": float(np.max(lat)),
               "lat_ms": np.asarray(lat, float)}
        # agreement with the stored predictions of the same windows
        try:
            from aiengine import up2_eval as E
            idx = E.latest_index("gbm", prefix="up2_v3b")
            scp, srp = E.load_preds(idx, "test2")
            m = rp.merge(srp[["unit_id", "window_index", "rul_pred_h"]], on=["unit_id", "window_index"], suffixes=("", "_stored"))
            mc = cp.merge(scp[["unit_id", "window_index", "pred_class"]], on=["unit_id", "window_index"], suffixes=("", "_stored"))
            out["agree_rul_median_abs_rel_diff_pct"] = float(np.median(np.abs(m.rul_pred_h - m.rul_pred_h_stored) / np.maximum(m.rul_pred_h_stored, 1e-9)) * 100)
            out["agree_class_frac"] = float((mc.pred_class == mc.pred_class_stored).mean())
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception as e:  # noqa: BLE001
        return err("live ONNX engine run", e)


def spec_named_models(spec: dict | None) -> dict:
    """N-HiTS / TFT numbers on the same synth_v3 test data from the stored spec check (the spec-named models)."""
    try:
        nn = spec["test"]["nn"]
        lat = spec["latency"]["v3_nn_tft_onnx"]
        return {"S7": nn["rul"]["S7_mape_pct"], "F1": nn["classification"]["macro_f1"], "stage_max": nn["stage"]["max_abs_error"],
                "caught": f"{nn['caught']['caught_by_stage3']}/{nn['caught']['rtf_bearings']}", "tft_p95": lat["p95_ms"]}
    except Exception:  # noqa: BLE001
        return {}


def is1_sensitivity(backend_p95: float, backend_max: float, coe_ms: float, hops=(("sim 10-60 ms (max)", 60.0), ("100 ms", 100.0), ("300 ms", 300.0))) -> pd.DataFrame:
    """[CALCULATED] IS1 budget = Wi-Fi hop + measured backend leg + COE leg. No new measurement."""
    rows = []
    for name, hop in hops:
        rows.append({"Wi-Fi hop assumed": name, "COE leg ms": coe_ms, "backend p95 ms": backend_p95, "backend max ms": backend_max,
                     "total (p95 backend) ms": round(hop + backend_p95 + coe_ms, 1), "total (max backend) ms": round(hop + backend_max + coe_ms, 1),
                     "< 500 ms": "yes" if hop + backend_max + coe_ms < 500 else "NO"})
    brk = 500 - backend_max - coe_ms
    return pd.DataFrame(rows), brk


# --------------------------------------------------------------------------- COE-integration readiness (domain gap + adaptation)
def coe_adapt_rescore() -> dict:
    """Re-score the SAVED COE-test predictions with the logic of `python -m aiengine.coe_adapt_eval` (cmd_report) WITHOUT writing a
    new report. Returns metrics for: v3b as deployed, base+gate, adapted without gate, adapted + gate (selected). [SIMULATION]/[SYNTHETIC]."""
    try:
        from aiengine import coe_adapt as A
        from aiengine import coe_adapt_eval as CE
        idxs = sorted(CE.PRED_DIR.glob("coe_adapt_test_*.index.json"))
        if not idxs:
            return miss("saved COE-test predictions (reports/up2_preds/coe_adapt_test_*.index.json)")
        ix = json.loads(idxs[-1].read_text())
        sel = ix["meta"]["selected"]
        coe = CE.load_coe()
        te = coe[coe["split"] == "test"]
        rd = lambda k: pd.read_csv(CE.PRED_DIR / ix["files"][k])  # noqa: E731
        base, adapted = rd("base"), rd("selected_nogate")
        res = {"v3b as deployed": A.coe_metrics(base, te), "adapted, no gate (ablation: retrain alone)": A.coe_metrics(adapted, te)}
        if sel.get("gate_tau", 0) > 0:
            res["deployed + gate (ablation: gate alone)"] = A.coe_metrics(A.gate(base, te, sel["gate_tau"], sel["gate_signal"]), te)
            res["SELECTED: adapted + gate"] = A.coe_metrics(A.gate(adapted, te, sel["gate_tau"], sel["gate_signal"]), te)
        rul = CE._rul_summary(rd("rul_diag_raw_features"), te)
        key = "run_id" if "run_id" in te.columns else None
        n_runs = int(te[key].nunique()) if key else None
        rows = []
        for name, r in res.items():
            rows.append({"classifier": name, "macro-F1": r["classification"]["macro_f1"], "B2 false-fault %": 100 * r["b2_transmitted_false_fault_rate"],
                         "B1 false-fault %": 100 * r["b1_transmitted_false_fault_rate"], "both-healthy false-fault %": 100 * r["clean_false_fault_rate"],
                         "missed-fault %": 100 * r["missed_fault_rate"], "wrong-class %": 100 * r["wrong_class_rate"],
                         "F1 cage": r["classification"]["per_class_f1"].get("cage")})
        return {"status": "ok", "index": idxs[-1].name, "scored_utc": ix["meta"]["utc"], "selected": {k: sel[k] for k in ("candidate", "gate_tau", "gate_signal")},
                "table": pd.DataFrame(rows), "n_runs_test": n_runs, "rul_diag": rul, "windows": int(len(te)),
                "sel_metrics": res.get("SELECTED: adapted + gate")}
    except Exception as e:  # noqa: BLE001
        return err("COE adapt re-score", e)


# --------------------------------------------------------------------------- IS2 wording v2
def is2_v2_rescore(name: str = "test4") -> dict:
    """Re-score the SAVED synth_v3_test4 AI predictions with the logic of `python -m aiengine.up2d_eval` (evaluate(); no report written).
    The physics RUL comes from the FE-twin attach_physics step recomputed here; the AI RUL is the stored v3b/v3c prediction."""
    try:
        from aiengine import up2d_eval as E
        from aiengine import up2d_is2 as D
        t0 = time.perf_counter()
        orig = D.load_set
        if not _LOADER:
            _LOADER.append(orig)
        D.load_set = _cached_set                      # same function, memoised so the chart data reuse one load
        try:
            r = E.evaluate(name)
        finally:
            D.load_set = orig
        r["seconds"] = round(time.perf_counter() - t0, 1)
        r["status"] = "ok"
        # per-bearing values behind the two v2 clauses (for the scatter chart; same windows as evaluate())
        truth = _cached_set(name)
        _, pr = D.latest_preds(name)
        m = D.scored_windows(truth, pr["v3b"])
        r["pb_v3b"] = D.per_bearing(m)
        r["pb_floor"] = D.per_bearing(D.perfect_ai(m))
        return r
    except Exception as e:  # noqa: BLE001
        return err("IS2 v2 re-score", e)


_SET_CACHE: dict[str, pd.DataFrame] = {}
_LOADER: list = []


def _cached_set(name: str) -> pd.DataFrame:
    if not _LOADER:
        from aiengine import up2d_is2 as D
        _LOADER.append(D.load_set)                     # the ORIGINAL loader (captured before any patching)
    if name not in _SET_CACHE:
        _SET_CACHE[name] = _LOADER[0](name)
    return _SET_CACHE[name]


def is2_live_subset(step: int = 4) -> dict:
    """Runs the DEPLOYED v3b engine (ONNX) live on every `step`-th faulted run-to-failure bearing of test4 and scores IS2 v2 with the
    frozen thresholds against the twin physics RUL. SUBSET (n bearings printed); [SYNTHETIC] + [SIMULATION]."""
    try:
        from aiengine import up2d_is2 as D
        from aiengine.config import MODELS_DIR
        from aiengine.engine import HybridEngine
        t0 = time.perf_counter()
        truth = _cached_set("test4")
        _, pr = D.latest_preds("test4")
        units = sorted(D.scored_windows(truth, pr["v3b"]).unit_id.unique())[::step]
        eng = HybridEngine(MODELS_DIR, contract_file="model_contract_v3b.json")
        rows = []
        for u in units:
            g = truth[truth.unit_id == u].sort_values("window_index")
            for wi, o in zip(g.window_index.to_numpy(), eng.run_unit(g, 1)):
                if o.get("status") == "ok":
                    rows.append({"unit_id": u, "window_index": int(wi), "rul_pred_h": o["rul_hours"]})
        m = D.scored_windows(truth, pd.DataFrame(rows))
        res = D.is2_v2(m, D.FROZEN["T_MED"], D.FROZEN["T_P95"])
        return {"status": "ok", "n_bearings": len(units), "seconds": round(time.perf_counter() - t0, 1), "v2": res}
    except Exception as e:  # noqa: BLE001
        return err("IS2 live subset", e)


# ======================================================================================================================
# PRESENTATION LAYER (notebook restyle 2026-10-26, PPR style). Nothing below computes a spec number: it only formats the
# values the cells above recompute (badges, tables, charts). Every chart draws its target line.
# ======================================================================================================================
FIGDIR = EVIDENCE / "up2_figures"
# status colours (dataviz reference status palette; always paired with an icon + a text label, never colour alone)
STATUS = {
    "MET":           {"fg": "#0b6b0b", "bar": "#0ca30c", "bg": "#e3f5e3", "icon": "&#10004;"},
    "PARTIAL":       {"fg": "#7a5300", "bar": "#fab219", "bg": "#fdf0cf", "icon": "&#9680;"},
    "NOT MET":       {"fg": "#a52a2a", "bar": "#d03b3b", "bg": "#fbe0de", "icon": "&#10008;"},
    "NOT EVIDENCED": {"fg": "#52514e", "bar": "#a9a7a1", "bg": "#efeeea", "icon": "&#8211;"},
    "PENDING":       {"fg": "#52514e", "bar": "#c9c7c1", "bg": "#f4f3ef", "icon": "&#8230;"},
    "MISSING":       {"fg": "#52514e", "bar": "#a9a7a1", "bg": "#efeeea", "icon": "?"},
}
GOOD, BAD, WARN, GREY = STATUS["MET"]["bar"], STATUS["NOT MET"]["bar"], STATUS["PARTIAL"]["bar"], "#a9a7a1"
LIGHT_GOOD = "#7fd07f"


def kind(status) -> str:
    s = str(status).upper().strip()
    if s.startswith("PASS"):
        return "MET"
    if s.startswith("FAIL"):
        return "NOT MET"
    for k in ("NOT EVIDENCED", "NOT MET", "PARTIAL", "MET", "PENDING", "MISSING"):
        if s.startswith(k):
            return k
    return "NOT EVIDENCED"


def style() -> None:
    """PPR matplotlib style, slightly larger fonts for readability."""
    import matplotlib as mpl
    P.style_matplotlib()
    mpl.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.dpi": 100, "savefig.bbox": "tight",
                         "figure.titlesize": 13, "figure.titleweight": "bold"})
    FIGDIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> str:
    try:
        fig.savefig(FIGDIR / name, dpi=130, bbox_inches="tight", facecolor="white")
        return str((FIGDIR / name).relative_to(REPO))
    except Exception as e:  # noqa: BLE001
        return f"(not saved: {e})"


def _esc(s) -> str:
    import html
    return html.escape(str(s))


def tag(prov: str) -> str:
    """Small grey provenance tag(s). '[SYNTHETIC] + [MEASURED host]' -> one pill per bracket."""
    parts = re.findall(r"\[[^\]]+\]", str(prov)) or [str(prov)]
    return " ".join(f'<span style="display:inline-block;background:#efeeea;color:#52514e;border-radius:4px;padding:0 6px;'
                    f'margin:1px 2px;font:11px/18px ui-monospace,Consolas,monospace">{_esc(p)}</span>' for p in parts)


def pill(status: str, text: str | None = None, small: bool = False) -> str:
    k = kind(status)
    c = STATUS[k]
    t = _esc(text if text is not None else k)
    fs = "11px" if small else "12.5px"
    return (f'<span style="display:inline-block;background:{c["bg"]};color:{c["fg"]};border:1px solid {c["bar"]};'
            f'border-radius:12px;padding:1px 9px;margin:1px 2px;font-weight:700;font-size:{fs};white-space:nowrap">'
            f'{c["icon"]} {t}</span>')


def verdict(item: str, ok, headline: str, prov: str = "", note: str = ""):
    """One-line coloured verdict badge: PASS / FAIL / MISSING + the number vs the target + provenance tags."""
    from IPython.display import HTML
    k = "MISSING" if ok is None else ("MET" if ok else "NOT MET")
    word = "MISSING" if ok is None else ("PASS" if ok else "FAIL")
    c = STATUS[k]
    return HTML(f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;border:1px solid {c["bar"]};border-left:8px solid {c["bar"]};'
                f'background:{c["bg"]};border-radius:6px;padding:8px 12px;margin:6px 0 10px 0;font-size:14px">'
                f'<span style="font-weight:800;color:{c["fg"]};font-size:16px;min-width:92px">{c["icon"]} {word}</span>'
                f'<span style="font-weight:700;color:#0b0b0b">{_esc(item)}</span>'
                f'<span style="color:#0b0b0b">{_esc(headline)}</span>'
                f'<span style="margin-left:auto">{tag(prov) if prov else ""}</span></div>'
                + (f'<div style="color:#52514e;font-size:12px;margin:-6px 0 8px 4px">{_esc(note)}</div>' if note else ""))


def missing(what):
    from IPython.display import HTML
    return HTML(f'<div style="margin:4px 0">{pill("MISSING", "input missing")} <span style="color:#52514e;font-size:12px">'
                f'{_esc(str(what)[:400])}</span></div>')


def md_html(md: str) -> str:
    try:
        import mistune
        return mistune.html(md) if hasattr(mistune, "html") else mistune.markdown(md)
    except Exception:  # noqa: BLE001
        return "<pre style='white-space:pre-wrap'>" + _esc(md) + "</pre>"


def details(summary: str, body_md: str = "", body_html: str = "", open_: bool = False):
    """Collapsible disclosure block (every caveat is kept, just folded)."""
    from IPython.display import HTML
    body = (md_html(body_md) if body_md else "") + body_html
    return HTML(f'<details {"open" if open_ else ""} style="border:1px solid #e4e3df;border-radius:6px;padding:4px 10px;margin:6px 0;'
                f'background:#fbfbf9;font-size:12.5px"><summary style="cursor:pointer;font-weight:600;color:#52514e">'
                f'{_esc(summary)}</summary><div style="padding:4px 2px">{body}</div></details>')


def reading(text: str):
    from IPython.display import HTML
    return HTML(f'<div style="border-left:4px solid #2a78d6;background:#f2f7fd;padding:6px 10px;margin:8px 0;font-size:13.5px">'
                f'<b>Reading.</b> {md_html(text).replace("<p>", "").replace("</p>", "")}</div>')


def table(df: pd.DataFrame, status_cols=(), fmt: dict | None = None, precision: int = 2, caption: str | None = None):
    """PPR-style table: hidden index, left aligned, small font, colour-coded status cells."""
    def col(v):
        c = STATUS[kind(v)]
        return f"background-color:{c['bg']};color:{c['fg']};font-weight:700;border-left:4px solid {c['bar']}"
    st = df.style.hide(axis="index").format(precision=precision, na_rep="-")
    if fmt:
        st = st.format(fmt, na_rep="-")
    for c in status_cols:
        if c in df.columns:
            st = st.map(col, subset=[c])
    st = st.set_properties(**{"text-align": "left", "font-size": "12px", "padding": "3px 8px"}) \
           .set_table_styles([{"selector": "th", "props": "text-align:left;font-size:12px;background:#f4f3ef;color:#0b0b0b;padding:4px 8px"},
                              {"selector": "", "props": "border-collapse:collapse;margin:4px 0 10px 0"},
                              {"selector": "td", "props": "border-bottom:1px solid #eeede9"},
                              {"selector": "caption", "props": "caption-side:top;text-align:left;font-weight:700;color:#0b0b0b;font-size:13px;padding:2px 0"}])
    if caption:
        st = st.set_caption(caption)
    return st


# --------------------------------------------------------------------------- scorecard, PPR -> UP2, KPI banner
SHORT = {"C1": "Target machine envelope (OH-2 / Goulds 3196)", "C2": "Mains power, SELV IEPE excitation", "C4": "Local processing, no cloud",
         "C5": "Dashboard on the LAN only", "C6": "UV-stabilised polymer enclosure", "S1": "2x biaxial IEPE accelerometers",
         "S2": "FE twin identifies stiffness <= 10 %", "S3": "Enclosure IP54 / 85 C / UV", "S4": "Sample rate >= 25 kS/s/ch x 4",
         "S5": "Window -> payload <= 333 ms", "S6": "Fixed-dim AI input, 1000-3600 rpm", "S7": "RUL MAPE <= 15 %",
         "S8": "Classification < 200 ms/window", "S9": "Dashboard >= 10 Hz", "IS1": "Sensor -> dashboard < 500 ms",
         "IS2": "Physics-AI RUL agreement", "IS3": "macro-F1 >= 0.85, stage err <= 1, caught by stage 3"}


def _row(iid):
    return {**REG.get(iid, {}), **OVERRIDES.get(iid, {})}


def scorecard_frame() -> pd.DataFrame:
    rows = []
    for iid, text, owner, _ppr in ITEMS:
        r = _row(iid)
        have = iid in REG or iid in OVERRIDES
        st = r.get("status", "NOT EVIDENCED") if have else "PENDING (cell not run yet)"
        rows.append({"ID": iid, "Item": SHORT.get(iid, text), "Owner": owner.split(" ")[0], "Target": r.get("target", "-") if have else "-",
                     "Measured": r.get("value", "-") if have else "-", "Status": st,
                     "Level": ("firm" if r.get("firm") else ("design level" if kind(st) == "MET" else "-")) if have else "-",
                     "Rank": r.get("rank", 0) if have else "-", "Provenance": r.get("prov", "-") if have else "-"})
    return pd.DataFrame(rows)


def styled_scorecard():
    return table(scorecard_frame(), status_cols=["Status"]) \
        .set_properties(subset=["Provenance"], **{"color": "#52514e", "font-family": "Consolas,monospace", "font-size": "11px"}) \
        .set_properties(subset=["ID"], **{"font-weight": "700"})


def ppr_vs_up2_frame() -> pd.DataFrame:
    rows = []
    for iid, text, owner, ppr in ITEMS:
        have = iid in REG or iid in OVERRIDES
        new = _row(iid).get("status", "NOT EVIDENCED") if have else "PENDING"
        ko, kn = kind(ppr), kind(new)
        if kn == "PENDING":
            ch = "pending"
        elif ko != "MET" and kn == "MET":
            ch = "▲ newly met"
        elif ko == "MET" and kn == "MET":
            ch = "= still met"
        elif ko == "MET":
            ch = "▼ lost"
        else:
            ch = "= not yet"
        short_new = new if kn == "PENDING" else kn + (" " + " ".join(re.findall(r"\[[^\]]+\]", new)) if kn == "MET" else "")
        rows.append({"ID": iid, "Item": SHORT.get(iid, text), "Owner": owner.split(" ")[0], "PPR (5 Oct)": ppr, " ": "→",
                     "UP2 (now)": short_new, "Change": ch})
    return pd.DataFrame(rows)


def styled_ppr_vs_up2():
    def chg(v):
        return ("color:#0b6b0b;font-weight:700" if "newly" in v else "color:#a52a2a;font-weight:700" if "lost" in v else "color:#52514e")
    return table(ppr_vs_up2_frame(), status_cols=["PPR (5 Oct)", "UP2 (now)"]).map(chg, subset=["Change"]) \
        .set_properties(subset=["ID"], **{"font-weight": "700"})


def kpi_banner(M: pd.DataFrame | None = None, cons_met: int | None = None):
    from IPython.display import HTML
    tile = "flex:1 1 210px;border:1px solid #e4e3df;border-radius:8px;padding:10px 14px;background:#ffffff;min-width:200px"
    lab = "font-size:11.5px;color:#52514e;text-transform:uppercase;letter-spacing:.04em;font-weight:700"
    data_badge = (f'<div style="{tile}"><div style="{lab}">Data basis</div><div style="margin-top:6px">{pill("PARTIAL", "synthetic / simulated")}</div>'
                  f'<div style="margin-top:4px">{pill("MET", "coach-approved", small=True)}</div>'
                  f'<div style="font-size:12px;color:#52514e;margin-top:6px">substitute for rig data (sensor parts may not arrive); '
                  f'every number keeps its provenance tag. Public-data PPR results are kept as history and still fail.</div></div>')
    if M is None:
        body = (f'<div style="{tile};flex:2 1 320px"><div style="{lab}">Items met</div><div style="font-size:30px;font-weight:800;color:#a9a7a1">PENDING</div>'
                f'<div style="font-size:12px;color:#52514e">filled in by the Summary cell at the end, from the live registry</div></div>' + data_badge)
    else:
        cnt = verdict_counts(M)
        s = M.set_index("ID")
        n, met, bar = cnt["total"], cnt["MET"], cnt["bar"]
        need = max(0, bar - met)
        fw, dw, barpct = 100 * cnt["firm"] / n, 100 * cnt["design_level"] / n, 100 * bar / n
        prog = (f'<div style="position:relative;height:14px;background:#efeeea;border-radius:7px;margin:10px 0 2px 0">'
                f'<div style="position:absolute;left:0;top:0;height:14px;width:{fw:.1f}%;background:{GOOD};border-radius:7px 0 0 7px"></div>'
                f'<div style="position:absolute;left:{fw:.1f}%;top:0;height:14px;width:{dw:.1f}%;background:{LIGHT_GOOD}"></div>'
                f'<div style="position:absolute;left:{barpct:.1f}%;top:-5px;height:24px;border-left:3px solid #0b0b0b"></div></div>'
                f'<div style="position:relative;height:16px;font-size:11.5px;color:#0b0b0b">'
                f'<span style="position:absolute;left:{barpct:.1f}%;transform:translateX(-50%);font-weight:700">bar {bar}</span></div>')
        t1 = (f'<div style="{tile};flex:2 1 320px"><div style="{lab}">Items met (of {n} counted; C3 excluded by the reviewer)</div>'
              f'<div style="font-size:34px;font-weight:800;color:#0b0b0b;line-height:1.15">{met} / {n} '
              f'<span style="font-size:16px;color:#52514e;font-weight:600">= {100 * met / n:.1f} % &nbsp;(rubric: more than 75 % = {bar})</span></div>{prog}'
              f'<div style="font-size:12.5px;margin-top:2px"><span style="color:#0b6b0b;font-weight:700">&#9632; {cnt["firm"]} firm</span> &nbsp; '
              f'<span style="color:#3a8f3a;font-weight:700">&#9632; {cnt["design_level"]} design level</span> &nbsp; '
              f'<span style="color:#52514e">&#9632; {cnt["NOT EVIDENCED"]} not evidenced here (owners supply)</span></div>'
              f'<div style="font-size:12.5px;margin-top:6px">{pill("MET" if met >= bar else "PARTIAL", "bar reached" if met >= bar else f"bar NOT reached: {need} more needed")}'
              + (f' <span style="color:#52514e">conservative reading: {cons_met} / {n}</span>' if cons_met is not None else "") + '</div></div>')
        ics = ["C5", "S7", "S8"]
        ok_ics = all(kind(s.loc[i, "UP2 status"]) == "MET" for i in ics)
        t2 = (f'<div style="{tile}"><div style="{lab}">ICS block (1 constraint + 2 specs)</div><div style="margin:6px 0">'
              + "".join(pill(s.loc[i, "UP2 status"], i) for i in ics) + '</div>'
              f'<div style="font-size:22px;font-weight:800;color:{STATUS["MET" if ok_ics else "NOT MET"]["fg"]}">{"&#10004; ALL MET" if ok_ics else "&#10008; NOT ALL MET"}</div>'
              f'<div style="font-size:12px;color:#52514e">rubric part 1 (50 pts)</div></div>')

        def note(i):
            if kind(s.loc[i, "UP2 status"]) != "MET":
                return ""
            x = "firm" if s.loc[i, "Firm?"] == "yes" else "design level"
            return x + {"IS1": " &middot; Wi-Fi hop simulated", "IS2": " &middot; wording v2 unconfirmed"}.get(i, "")
        t3 = (f'<div style="{tile}"><div style="{lab}">Integrated specs</div><div style="margin:6px 0;line-height:2.1">'
              + "<br>".join(f'{pill(s.loc[i, "UP2 status"], i)} <span style="font-size:11.5px;color:#52514e">{note(i)}</span>' for i in ("IS1", "IS2", "IS3"))
              + '</div></div>')
        body = t1 + t2 + t3 + data_badge
    return HTML(f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 6px 0;font-family:system-ui,Segoe UI,Arial">{body}</div>')


# --------------------------------------------------------------------------- generic chart helpers
def _plt():
    import matplotlib.pyplot as plt
    return plt


def target_line(ax, v, label, axis="x", color=None, ls="--", text_pos=0.97):
    color = color or BAD
    if axis == "x":
        ax.axvline(v, color=color, ls=ls, lw=1.6, zorder=5)
        ax.text(v, text_pos, f" {label}", transform=ax.get_xaxis_transform(), color=color, fontsize=10, va="bottom" if text_pos > 1 else "top",
                ha="left", fontweight="bold")
    else:
        ax.axhline(v, color=color, ls=ls, lw=1.6, zorder=5)
        ax.text(text_pos, v, f"{label} ", transform=ax.get_yaxis_transform(), color=color, fontsize=10, va="bottom", ha="right", fontweight="bold")


def fig_checks(title: str, checks: list):
    """Row of check tiles (tick / cross + label + detail)."""
    plt = _plt()
    n = len(checks)
    fig, ax = plt.subplots(figsize=(11, 0.72 * n + 0.6))
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, n - 0.5); ax.axis("off")
    for i, (lab, ok, det) in enumerate(checks):
        y = n - 1 - i
        k = "MET" if ok else "NOT MET"
        ax.add_patch(plt.Rectangle((0, y - 0.4), 1, 0.8, color=STATUS[k]["bg"], lw=0))
        ax.add_patch(plt.Rectangle((0, y - 0.4), 0.008, 0.8, color=STATUS[k]["bar"], lw=0))
        ax.text(0.025, y, "✔ PASS" if ok else "✘ FAIL", color=STATUS[k]["fg"], fontsize=12, fontweight="bold", va="center")
        ax.text(0.14, y + 0.14, lab, fontsize=11.5, fontweight="bold", va="center")
        ax.text(0.14, y - 0.18, det, fontsize=10, color=INK2, va="center")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- section charts
def fig_status_summary(M: pd.DataFrame, cons: pd.DataFrame | None = None):
    plt = _plt()
    rows = [("Registry (as labelled)", M)] + ([("Conservative reading", cons)] if cons is not None else [])
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.8 * len(rows)))
    segs = [("MET, firm", GOOD), ("MET, design level", LIGHT_GOOD), ("PARTIAL", WARN), ("NOT MET", BAD), ("NOT EVIDENCED here", GREY)]
    used = set()
    for yi, (name, df) in enumerate(rows[::-1]):
        st, firm = df["UP2 status"].map(kind), df["Firm?"] == "yes"
        vals = [int(((st == "MET") & firm).sum()), int(((st == "MET") & ~firm).sum()), int((st == "PARTIAL").sum()),
                int((st == "NOT MET").sum()), int((st == "NOT EVIDENCED").sum())]
        left = 0
        for (lab, colr), v in zip(segs, vals):
            if v:
                ax.barh(yi, v, left=left, color=colr, edgecolor="white", linewidth=2, height=0.62, label=None if lab in used else lab)
                used.add(lab)
                ax.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=11.5, fontweight="bold", color="white" if colr in (GOOD, BAD) else INK)
            left += v
        ax.text(len(df) + 0.25, yi, f"MET {vals[0] + vals[1]}/{len(df)}", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(0, len(M) + 2.8); ax.set_xticks(range(0, len(M) + 1))
    ax.axvline(BAR, color=INK, ls="--", lw=1.6, zorder=6)
    ax.text(BAR, len(rows) - 0.45, f"bar {BAR} (>75 %)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=5, frameon=False)
    met = int(M["UP2 status"].map(kind).eq("MET").sum())
    ax.set_title(f"{met} of {len(M)} items MET; the >75 % bar needs {BAR}: {max(0, BAR - met)} more, from the owner-supplied items", loc="left", pad=20)
    fig.tight_layout()
    return fig


def fig_data_basis(sp: pd.DataFrame):
    plt = _plt()
    from matplotlib.patches import Patch
    d = sp.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 0.4 * len(d) + 1.5))
    cols = {"train": PAL[0], "val": PAL[3], "test": PAL[1]}
    tot = d[["train", "val", "test"]].fillna(0).sum(1)
    for i in range(len(d)):
        r = d.iloc[len(d) - 1 - i]
        left = 0
        for k in ("train", "val", "test"):
            v = r[k] or 0
            if v:
                ax.barh(i, v, left=left, color=cols[k], edgecolor="white", linewidth=1.5, height=0.66)
                if v >= 40:
                    ax.text(left + v / 2, i, str(int(v)), ha="center", va="center", fontsize=9.5, color="white" if k != "val" else INK)
                left += v
        ax.text(left + 8, i, r["role"], va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(d)), list(d["set"].iloc[::-1]))
    ax.set_xlabel("units in splits.json (synthetic sets: 2 bearing series per run; splits are by run, never by window)")
    ax.legend(handles=[Patch(color=cols[k], label=k) for k in ("train", "val", "test")], loc="lower right")
    ax.set_xlim(0, tot.max() * 1.75)
    ax.grid(axis="y", visible=False)
    ax.set_title("Data basis: fresh test sets were registered before any fit and scored once  [splits.json, live]", loc="left")
    fig.tight_layout()
    return fig


def fig_s8(cls_ms, eng_ms=None, tft_p95=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    rows = [("deployed classifier ONNX\n(1 thread, batch 1)", np.asarray(cls_ms), PAL[0])]
    if eng_ms is not None and len(eng_ms):
        rows.append(("full engine step per window\n(features+classifier+RUL+post)", np.asarray(eng_ms), PAL[6]))
    for i, (lab, a, colr) in enumerate(rows):
        y = len(rows) - i
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(a))
        ax.scatter(a, y + jit, s=6, alpha=0.3, color=colr, lw=0)
        ax.boxplot(a, positions=[y], vert=False, widths=0.5, showfliers=False, medianprops={"color": INK, "lw": 2},
                   boxprops={"color": INK}, whiskerprops={"color": INK}, capprops={"color": INK}, manage_ticks=False)
        ax.text(a.max() * 1.5, y, f"p95 {np.percentile(a, 95):.2f} ms  (n={len(a)})", va="center", fontsize=10.5, fontweight="bold")
    ticks, pos = [r[0] for r in rows], list(range(len(rows), 0, -1))
    if tft_p95 is not None:
        ax.scatter([tft_p95], [0], marker="D", s=70, color=PAL[1], zorder=6)
        ax.text(tft_p95 * 1.5, 0, f"p95 {tft_p95:.2f} ms (stored report)", va="center", fontsize=10.5)
        ticks.append("original TFT ONNX\n(spec-named, transparency)"); pos.append(0)
    ax.set_yticks(pos, ticks); ax.set_ylim(min(pos) - 0.6, max(pos) + 0.6)
    ax.set_xscale("log"); ax.set_xlim(0.01, 1000)
    target_line(ax, 200, "S8 limit 200 ms", axis="x")
    ax.set_xlabel("ms per window (log scale)  [MEASURED host], [SYNTHETIC] inputs")
    ax.grid(axis="y", visible=False)
    p95 = np.percentile(cls_ms, 95)
    ax.set_title(f"S8: classification p95 {p95:.2f} ms, about {200 / p95:,.0f}x under the 200 ms limit", loc="left")
    fig.tight_layout()
    return fig


def fig_coe(c1: dict, c2_worst: float, stm_ms: float, c3: dict):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [0.8, 1.2, 1.9]})
    r = c1["calculated"]["per_channel_rate_ksps"]
    ax[0].bar(["AD7606, per ch"], [r], color=GOOD, width=0.4); ax[0].set_xlim(-0.5, 1.2)
    ax[0].text(0, r, f"{r:.0f} kS/s", ha="center", va="bottom", fontweight="bold")
    target_line(ax[0], 25, "target 25", axis="y", text_pos=0.98); ax[0].set_ylim(0, 40); ax[0].set_ylabel("kS/s per channel (x 4 ch)")
    ax[0].set_title("S4: 30 >= 25 kS/s", loc="left", fontsize=11.5); ax[0].grid(axis="x", visible=False)
    b = ax[1].bar(["host PC worst\n[MEASURED host]", "STM32 pessimistic\n[CALCULATED]"], [c2_worst, stm_ms], color=[PAL[0], PAL[6]], width=0.55)
    for rect, v in zip(b, [c2_worst, stm_ms]):
        ax[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f} ms", ha="center", va="bottom", fontweight="bold")
    ax[1].set_yscale("log"); ax[1].set_ylim(1, 1500); target_line(ax[1], 333, "limit 333 ms", axis="y", text_pos=0.98)
    ax[1].set_ylabel("t20 -> payload, ms (log)"); ax[1].set_title("S5: >10x margin on both legs", loc="left", fontsize=11.5)
    ax[1].grid(axis="x", visible=False)
    rows = pd.DataFrame(c3["rows"])
    tr = rows["envelope_record_truncated_to_4096"].astype(bool)
    nf = rows["features_shape"].map(lambda s: s[0])
    ax[2].scatter(rows.loc[~tr, "rpm_nominal"], nf[~tr], color=GOOD, s=45, label="(32,) features, 152 B payload", zorder=4)
    ax[2].scatter(rows.loc[tr, "rpm_nominal"], nf[tr], color=WARN, s=45, marker="s", label="(32,) but record truncated to 4096 pts", zorder=4)
    for v in (1750, 3600):
        ax[2].axvline(v, color=INK2, ls=":", lw=1); ax[2].text(v, 12, f" {v} rpm\n mode", fontsize=9, color=INK2)
    ax[2].set_ylim(0, 40); ax[2].set_xlabel("shaft speed, rpm"); ax[2].set_ylabel("features per window")
    ax[2].axhline(32, color=GOOD, lw=1, zorder=1)
    ax[2].legend(loc="lower right", fontsize=9)
    ax[2].set_title(f"S6: same (32,) vector at all {c3['n_speeds']} speeds  [SIMULATION]", loc="left", fontsize=11.5)
    fig.tight_layout()
    return fig


def fig_s7_curves(rr: pd.DataFrame, per: dict):
    """One bearing per fault class: the bearing whose MAPE is closest to its class median (typical, not best)."""
    plt = _plt()
    d = rr.copy()
    d["elapsed_h"] = d["life_hours"] - d["rul_hours"]
    pick = []
    for fc in ["outer_race", "inner_race", "ball", "cage"]:
        units = [u for u in d.loc[d.fault_class == fc, "unit_id"].unique() if u in per]
        if units:
            med = np.median([per[u] for u in units])
            pick.append((fc, min(units, key=lambda u: abs(per[u] - med))))
    fig, axs = plt.subplots(1, len(pick), figsize=(12.5, 3.7))
    axs = np.atleast_1d(axs)
    for a, (fc, u) in zip(axs, pick):
        g = d[d.unit_id == u].sort_values("window_index")
        w = g[g["in_deg_det"] & (g["rul_hours"] > 0)]
        if len(w):
            a.axvspan(w.elapsed_h.min(), w.elapsed_h.max(), color="#e8f0fb", lw=0, label="S7 scoring window")
        a.plot(g.elapsed_h, g.rul_hours, color=INK2, ls="--", lw=1.5, label="true RUL")
        a.plot(g.elapsed_h, g.rul_pred_h, color=PAL[0], lw=1.8, label="predicted (deployed v3b)")
        a.set_ylim(0, max(g.rul_hours.max(), 1) * 1.35)
        a.set_title(f"{fc.replace('_', ' ')}: MAPE {per[u]:.1f} %", fontsize=11, loc="left", color=STATUS["MET" if per[u] <= 15 else "NOT MET"]["fg"])
        a.set_xlabel("elapsed hours")
    axs[0].set_ylabel("remaining life, h")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("S7: predicted RUL tracks the true RUL (one typical test2 bearing per fault class)  [SYNTHETIC]", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    return fig


def fig_s7_bars(per: dict, s7: float):
    plt = _plt()
    v = pd.Series(per).sort_values()
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.bar(range(len(v)), v.values, color=[GOOD if x <= 15 else BAD for x in v.values], width=0.8)
    target_line(ax, 15, "target 15 % (applies to the mean)", axis="y", text_pos=0.5)
    ax.axhline(s7, color=PAL[0], lw=2)
    ax.text(1, s7 * 0.92, f"S7 = mean over bearings = {s7:.2f} %", color=PAL[0], fontsize=10.5, fontweight="bold", va="top")
    ax.set_xticks([]); ax.set_xlabel(f"{len(v)} held-out run-to-failure bearings of synth_v3_test2, sorted by error"); ax.set_ylabel("per-bearing MAPE, %")
    ax.grid(axis="x", visible=False)
    k = int((v <= 15).sum())
    ax.set_title(f"S7: mean {s7:.2f} % PASSES; {k} of {len(v)} bearings are individually under 15 % (median {v.median():.1f} %)", loc="left")
    fig.tight_layout()
    return fig


def fig_compare(rows: list, target: float, xlabel: str, title: str, log: bool = False, better: str = "lower", fmt="{:.2f}"):
    """Horizontal bars: (label, value, passes, provenance)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.3))
    mx = max(r[1] for r in rows)
    for i, (lab, v, ok, prov) in enumerate(rows[::-1]):
        ax.barh(i, v, color=GOOD if ok else BAD, height=0.6)
        xt = max(v * 1.08, target * 1.2) if log else max(v, target) + 0.02 * max(mx, target)
        ax.text(xt, i, fmt.format(v) + f"   {prov}", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    if log:
        ax.set_xscale("log"); ax.set_xlim(min(r[1] for r in rows) / 2, mx * 60)
    else:
        ax.set_xlim(0, max(mx, target) * 1.7)
    target_line(ax, target, f"target {'<=' if better == 'lower' else '>='} {target:g}", axis="x", text_pos=1.01)
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", pad=16)
    fig.tight_layout()
    return fig


def class_table(confusion: dict) -> pd.DataFrame:
    labs = confusion["labels"]
    m = np.asarray(confusion["matrix"], float)
    rows = []
    for i, c in enumerate(labs):
        tp = m[i, i]; fp = m[:, i].sum() - tp; fn = m[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        f = 2 * p * r / (p + r) if p + r else np.nan
        rows.append({"class": c, "windows": int(m[i].sum()), "precision": p, "recall": r, "F1": f, "vs 0.85": "PASS" if f >= 0.85 else "FAIL"})
    df = pd.DataFrame(rows)
    mf = df.F1.mean()
    df.loc[len(df)] = {"class": "macro average (IS3a)", "windows": int(m.sum()), "precision": df.precision.mean(), "recall": df.recall.mean(),
                       "F1": mf, "vs 0.85": "PASS" if mf >= 0.85 else "FAIL"}
    return df


def fig_is3(confusion: dict, per_class_f1: dict, macro: float, compare: list):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.5), gridspec_kw={"width_ratios": [1.3, 1, 1.05]})
    labs = confusion["labels"]
    m = np.asarray(confusion["matrix"], float)
    rn = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax[0].imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labs)):
        for k in range(len(labs)):
            if m[i, k]:
                ax[0].text(k, i, f"{rn[i, k]:.0%}\n{int(m[i, k])}", ha="center", va="center", fontsize=8.5, color="white" if rn[i, k] > 0.55 else INK)
    short = [l.replace("_race", " race") for l in labs]
    ax[0].set_xticks(range(len(labs)), short, rotation=30, ha="right"); ax[0].set_yticks(range(len(labs)), short)
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].grid(False); ax[0].set_title("Confusion (colour = recall)", loc="left")
    f = [per_class_f1[c] for c in labs]
    yy = list(range(len(labs)))[::-1]
    ax[1].barh(yy, f, color=[GOOD if x >= .85 else BAD for x in f], height=0.6)
    for i, x in zip(yy, f):
        ax[1].text(x + 0.005, i, f"{x:.3f}", ha="left", va="center", color=INK, fontweight="bold", fontsize=10)
    ax[1].set_yticks(yy, short); ax[1].set_xlim(0.5, 1.09)
    target_line(ax[1], 0.85, "0.85", axis="x", text_pos=1.01)
    ax[1].set_title(f"Per-class F1 (macro {macro:.3f})", loc="left", pad=16); ax[1].grid(axis="y", visible=False)
    yy = list(range(len(compare)))[::-1]
    for i, (lab, v, ok) in zip(yy, compare):
        ax[2].barh(i, v, color=GOOD if ok else BAD, height=0.6)
        ax[2].text(max(v, 0.86) + 0.01, i, f"{v:.3f}", ha="left", va="center", color=INK, fontweight="bold", fontsize=10)
    ax[2].set_yticks(yy, [c[0] for c in compare], fontsize=9.5); ax[2].set_xlim(0, 1.15)
    target_line(ax[2], 0.85, "0.85", axis="x", text_pos=1.01)
    ax[2].set_title("macro-F1: PPR -> UP2", loc="left", pad=16); ax[2].grid(axis="y", visible=False)
    fig.suptitle(f"IS3a: macro-F1 {macro:.3f} >= 0.85, and every class is above 0.85  [SYNTHETIC test2]", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_is3_stage(rr: pd.DataFrame, per_unit: list):
    plt = _plt()
    pu = pd.DataFrame(per_unit)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    d = rr.copy(); d["elapsed_h"] = d["life_hours"] - d["rul_hours"]
    cand = sorted(d[d.fault_class == "outer_race"].unit_id.unique()) or sorted(d.unit_id.unique())
    lens = {u: int((d.unit_id == u).sum()) for u in cand}
    u = sorted(cand, key=lambda x: lens[x])[len(cand) // 2]          # median-length bearing: typical, not chosen by score
    g = d[d.unit_id == u].sort_values("window_index")
    ax[0].fill_between(g.elapsed_h, g.health_stage - 1, g.health_stage + 1, step="post", color="#e8f0fb", lw=0, label="allowed: true stage +-1")
    ax[0].step(g.elapsed_h, g.health_stage, where="post", color=INK2, ls="--", lw=1.6, label="true stage")
    ax[0].step(g.elapsed_h, g.stage_pred, where="post", color=PAL[0], lw=2, label="predicted stage")
    r = pu[pu.unit_id == u] if len(pu) else pu
    if len(r) and bool(r.iloc[0]["caught"]):
        row = g[g.window_index == int(r.iloc[0]["caught_at_window"])]
        if len(row):
            ax[0].scatter(row.elapsed_h, row.health_stage, s=160, marker="*", color=PAL[1], zorder=6,
                          label=f"fault class confirmed (true stage {int(r.iloc[0]['caught_at_stage'])})")
    e = int((g.stage_pred - g.health_stage).abs().max())
    ax[0].set_ylim(0.3, 7.6); ax[0].set_yticks(range(1, 7)); ax[0].set_xlabel("elapsed hours"); ax[0].set_ylabel("health stage")
    ax[0].legend(loc="upper left", fontsize=9, ncol=2)
    ax[0].set_title(f"IS3b: typical outer-race bearing, max stage error {e} (limit 1)", loc="left")
    pu = pu.copy(); pu["stage"] = pu["caught_at_stage"].fillna(9).astype(int)
    stages = list(range(1, 7))
    bottom = np.zeros(len(stages))
    for i, fc in enumerate(["outer_race", "inner_race", "ball", "cage"]):
        cnt = np.array([int(((pu.fault_class == fc) & (pu.stage == s)).sum()) for s in stages])
        ax[1].bar(stages, cnt, bottom=bottom, color=PAL[i], edgecolor="white", linewidth=1.5, label=fc.replace("_", " "), width=0.7)
        bottom += cnt
    ax[1].axvspan(3.5, 6.6, color=STATUS["NOT MET"]["bg"], lw=0, zorder=0)
    ax[1].text(5.05, max(bottom.max(), 1) * 0.55, "too late\n(stage 4+)", ha="center", color=STATUS["NOT MET"]["fg"], fontsize=10, fontweight="bold")
    ax[1].set_xticks(stages); ax[1].set_xlabel("true stage when first caught\n(3 correct windows in a row)"); ax[1].set_ylabel("bearings")
    k = int((pu.stage <= 3).sum())
    ax[1].legend(fontsize=9, loc="upper right"); ax[1].grid(axis="x", visible=False)
    ax[1].set_title(f"IS3c: {k}/{len(pu)} caught by stage 3", loc="left")
    fig.tight_layout()
    return fig, u


def fig_s9(rate: dict, stages: dict | None = None, replay_hz: float = 10.0):
    plt = _plt()
    if stages:
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
        a0 = ax[0]
    else:
        fig, a0 = plt.subplots(figsize=(7, 3.9))
    labs = ["client: mean", "client: worst 1-s", "client: best 1-s", "NEW data\n(replay rate)"]
    vals = [rate["hz"], rate["per_sec_min"], rate["per_sec_max"], replay_hz]
    a0.bar(labs, vals, color=[GOOD if v >= 10 else BAD for v in vals[:3]] + [LIGHT_GOOD], width=0.6)
    for i, v in enumerate(vals):
        if i == 3:
            a0.text(i, v / 2, f"{v:.1f} Hz", ha="center", va="center", fontweight="bold")
        else:
            a0.text(i, v, f"{v:.1f} Hz", ha="center", va="bottom", fontweight="bold")
    target_line(a0, 10, "S9 >= 10 Hz", axis="y", text_pos=0.99)
    a0.set_ylim(0, max(vals) * 1.3); a0.set_ylabel("updates per second"); a0.grid(axis="x", visible=False)
    a0.set_title("S9: every 1-s window >= 10 Hz at the SignalR client", loc="left")
    if stages:
        order = [k for k in ("decode", "features", "tft_onnx", "nhits_onnx", "inference", "broadcast", "total_ingest_to_broadcast") if k in stages]
        p95 = [stages[k]["p95Ms"] for k in order]
        yy = list(range(len(order)))[::-1]
        ax[1].barh(yy, p95, color=[PAL[0]] * (len(order) - 1) + [PAL[6]], height=0.6)
        for i, v in zip(yy, p95):
            ax[1].text(v * 1.2, i, f"{v:.2f} ms", va="center", fontsize=10)
        ax[1].set_yticks(yy, [o.replace("_", " ").replace("tft onnx", "classifier slot (tft_onnx)").replace("nhits onnx", "RUL slot (nhits_onnx)") for o in order])
        ax[1].set_xscale("log"); ax[1].set_xlim(min(p95) / 3, 1000)
        target_line(ax[1], 100, "100 ms = one 10 Hz frame", axis="x", text_pos=1.01)
        ax[1].set_xlabel("backend p95, ms (log)  [MEASURED host]"); ax[1].grid(axis="y", visible=False)
        ax[1].set_title("Backend work per snapshot, p95", loc="left", pad=16)
    fig.tight_layout()
    return fig


def fig_is1(legs: dict, coe_stm32: float, backend_max: float):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    parts = [("COE node t20 -> payload, STM32 pessimistic [CALCULATED]", coe_stm32, coe_stm32, PAL[6]),
             ("Wi-Fi hop [SIMULATION, 10-60 ms jitter]", legs["wifi"]["p95"], legs["wifi"]["max"], PAL[3]),
             ("UDP + backend + SignalR client [MEASURED host]", legs["backend"]["p95"], legs["backend"]["max"], PAL[0])]
    for yi, which in enumerate(("p95", "max")):
        left = 0
        for lab, p95, mx, colr in parts:
            v = p95 if which == "p95" else mx
            ax[0].barh(1 - yi, v, left=left, color=colr, edgecolor="white", linewidth=2, height=0.55, label=lab if yi == 0 else None)
            if v > 14:
                ax[0].text(left + v / 2, 1 - yi, f"{v:.0f}", ha="center", va="center", color=INK if colr == PAL[3] else "white", fontsize=10, fontweight="bold")
            left += v
        ax[0].text(left + 6, 1 - yi, f"{left:.0f} ms", va="center", fontweight="bold")
    s = legs["sum_stm32"]
    ax[0].set_yticks([1, 0], ["sum of leg p95s", "sum of leg maxima"])
    ax[0].set_xlim(0, 530); ax[0].set_ylim(-0.6, 1.6)
    target_line(ax[0], 500, "IS1 < 500 ms", axis="x", text_pos=1.01)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=9)
    ax[0].set_xlabel("ms per window (browser render NOT included)"); ax[0].grid(axis="y", visible=False)
    ax[0].set_title(f"IS1: measured per-window total p95 {s['p95']:.0f} ms, max {s['max']:.0f} ms", loc="left", pad=16)
    hop = np.linspace(0, 530, 200)
    ax[1].plot(hop, hop + backend_max + coe_stm32, color=PAL[0], lw=2, label="hop + backend max + COE (STM32)")
    ax[1].axvspan(10, 60, color=STATUS["PARTIAL"]["bg"], lw=0, label="simulated hop range 10-60 ms")
    brk = 500 - backend_max - coe_stm32
    ax[1].scatter([brk], [500], color=BAD, s=60, zorder=6)
    ax[1].annotate(f"breaks only if the real\nhop > ~{brk:.0f} ms", (brk, 500), xytext=(150, 560), fontsize=10, color=STATUS["NOT MET"]["fg"],
                   arrowprops={"arrowstyle": "->", "color": BAD})
    ax[1].axhline(500, color=BAD, ls="--", lw=1.6)
    ax[1].set_ylim(0, 650); ax[1].set_xlim(0, 530)
    ax[1].set_xlabel("assumed Wi-Fi hop, ms  [CALCULATED]"); ax[1].set_ylabel("end-to-end worst case, ms")
    ax[1].legend(loc="lower right", fontsize=9); ax[1].set_title("IS1 sensitivity to the unmeasured Wi-Fi hop", loc="left", pad=16)
    fig.tight_layout()
    return fig


def fig_is2_scatter(pb: pd.DataFrame, MED: float, P95: float, T_MED: float, T_P95: float, controls: list, floor: dict | None = None,
                    v3c: dict | None = None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.add_patch(plt.Rectangle((0, 0), 100 * T_MED, 100 * T_P95, color=STATUS["MET"]["bg"], lw=0, zorder=0))
    ax.text(0.5, 3, "PASS box for the aggregates:\n(a) median <= 18 % of L_ref\n(b) p95 <= 63 % of L_ref", fontsize=9.5,
            color=STATUS["MET"]["fg"], va="bottom", fontweight="bold")
    ax.scatter(100 * pb["med"], 100 * pb["p95"], s=22, color=PAL[0], alpha=0.4, lw=0, label=f"v3b, one dot per bearing (n={len(pb)})")
    ax.scatter([100 * MED], [100 * P95], s=300, marker="*", color=PAL[0], edgecolor="white", lw=1.5, zorder=8,
               label=f"v3b aggregate {100 * MED:.1f} % / {100 * P95:.1f} %: PASS")
    if v3c:
        ax.scatter([100 * v3c["MED"]], [100 * v3c["P95"]], s=120, marker="P", color=PAL[6], zorder=7,
                   label=f"v3c research {100 * v3c['MED']:.1f} % / {100 * v3c['P95']:.1f} %: FAIL")
    if floor:
        ax.scatter([100 * floor["MED"]], [100 * floor["P95"]], s=110, marker="D", color=PAL[3], edgecolor=INK, zorder=7,
                   label=f"exact AI (RUL_AI = truth) {100 * floor['MED']:.1f} % / {100 * floor['P95']:.1f} %: fails (a)")
    for c in controls:
        x, y = 100 * c["MED"], min(100 * c["P95"], 195)
        ax.scatter([x], [y], s=90, marker="X", color=BAD, zorder=7)
        ax.annotate(c["control"].split(" (")[0] + (f" (p95 {100 * c['P95']:.0f} %)" if 100 * c["P95"] > 195 else ""), (x, y), xytext=(7, -3),
                    textcoords="offset points", fontsize=8.5, color=STATUS["NOT MET"]["fg"])
    ax.scatter([], [], s=90, marker="X", color=BAD, label="six deliberately wrong AIs: all FAIL")
    ax.axvline(100 * T_MED, color=BAD, ls="--", lw=1.4); ax.axhline(100 * T_P95, color=BAD, ls="--", lw=1.4)
    ax.text(100 * T_MED, 0.97, " (a) 18 %", transform=ax.get_xaxis_transform(), color=BAD, fontweight="bold", va="top")
    ax.text(1.0, 100 * T_P95, " (b) 63 %", transform=ax.get_yaxis_transform(), color=BAD, fontweight="bold", ha="left", va="center")
    ax.set_xlim(0, max(36, 100 * max(c["MED"] for c in controls) * 1.3)); ax.set_ylim(0, 205)
    ax.set_xlabel("median disagreement |RUL_phys - RUL_AI| / L_ref, %"); ax.set_ylabel("p95 disagreement, % of L_ref")
    ax.legend(loc="upper left", bbox_to_anchor=(1.07, 1.0), fontsize=9.5)
    ax.set_title("IS2 (wording v2): the deployed AI lands inside the box, every wrong AI outside  [SYNTHETIC]+[SIMULATION] test4", loc="left")
    fig.tight_layout()
    return fig


def fig_is2_v1v2(v1: list, v2: list, T_MED: float, T_P95: float):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.1), gridspec_kw={"width_ratios": [1, 1.3]})
    yy = list(range(len(v1)))[::-1]
    ax[0].barh(yy, [v for _, v in v1], color=BAD, height=0.55)
    for i, (_, v) in zip(yy, v1):
        ax[0].text(v * 1.12, i, f"{v:.0f} %", va="center", fontweight="bold")
    ax[0].set_yticks(yy, [l for l, _ in v1], fontsize=9.5); ax[0].set_xscale("log"); ax[0].set_xlim(2, 5000)
    target_line(ax[0], 5, "v1 target 5 %", axis="x", text_pos=1.01)
    ax[0].set_xlabel("max |phys - AI| / RUL_AI, % (log)"); ax[0].grid(axis="y", visible=False)
    ax[0].set_title("v1 literal max (history): FAIL\n(RUL_AI -> 0 near failure, so the max explodes)", loc="left", fontsize=11.5, pad=16)
    x = np.arange(len(v2)); w = 0.36
    med = [m for _, m, _ in v2]; p95 = [p for _, _, p in v2]
    b1 = ax[1].bar(x - w / 2, med, w, color=[GOOD if m <= 100 * T_MED else BAD for m in med], label="(a) median of bearing medians")
    b2 = ax[1].bar(x + w / 2, p95, w, color=[GOOD if p <= 100 * T_P95 else BAD for p in p95], alpha=0.6, hatch="//", edgecolor="white",
                   label="(b) p95 of bearing p95s")
    for rect, v in list(zip(b1, med)) + list(zip(b2, p95)):
        ax[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax[1].hlines(100 * T_MED, -0.5, len(v2) - 0.5, color=BAD, ls="--", lw=1.4)
    ax[1].text(len(v2) - 0.5, 100 * T_MED, " (a) 18 %", color=BAD, va="center", fontweight="bold")
    ax[1].hlines(100 * T_P95, -0.5, len(v2) - 0.5, color=BAD, ls=":", lw=1.6)
    ax[1].text(len(v2) - 0.5, 100 * T_P95, " (b) 63 %", color=BAD, va="center", fontweight="bold")
    ax[1].set_xticks(x, [l for l, _, _ in v2], fontsize=9.5); ax[1].set_ylabel("% of L_ref (rated life)"); ax[1].set_ylim(0, max(p95 + [100 * T_P95]) * 1.3)
    ax[1].set_xlim(-0.6, len(v2) + 0.2)
    ax[1].legend(loc="upper left", fontsize=9); ax[1].grid(axis="x", visible=False)
    ax[1].set_title("v2 reworded (current): divide by the rated life L_ref;\nthresholds frozen on validation before test4", loc="left", fontsize=11.5)
    fig.suptitle("IS2: the same physics-AI disagreement under two yardsticks", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_coe_adapt(tab: pd.DataFrame):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    labs = [c.replace(" (ablation: ", "\n(").replace("SELECTED: ", "SELECTED: ") for c in tab["classifier"]]
    yy = list(range(len(tab)))[::-1]
    f1 = tab["macro-F1"].to_numpy(); ff = tab["B2 false-fault %"].to_numpy()
    ax[0].barh(yy, f1, color=[GOOD if v >= .85 else BAD for v in f1], height=0.6)
    for i, v in zip(yy, f1):
        ax[0].text(max(v, 0.86) + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    ax[0].set_yticks(yy, labs, fontsize=9.5); ax[0].set_xlim(0, 1.12); target_line(ax[0], 0.85, "0.85", axis="x", text_pos=1.01)
    ax[0].set_xlabel("macro-F1 on COE-pipeline test windows"); ax[0].grid(axis="y", visible=False)
    ax[1].barh(yy, ff, color=[GOOD if v <= 5 else BAD for v in ff], height=0.6)
    for i, v in zip(yy, ff):
        ax[1].text(v + 1, i, f"{v:.1f} %", va="center", fontweight="bold")
    ax[1].set_yticks(yy, [""] * len(tab)); ax[1].set_xlim(0, 105)
    ax[1].set_xlabel("healthy partner bearing (B2) flagged faulty, % of windows"); ax[1].grid(axis="y", visible=False)
    fig.suptitle(f"COE domain gap in simulation: macro-F1 {f1[0]:.3f} -> {f1[-1]:.3f}; bearing-2 false faults {ff[0]:.1f} % -> {ff[-1]:.1f} %  [SIMULATION]+[SYNTHETIC]",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig


def fig_tests(suites: dict):
    plt = _plt()
    names = list(suites)
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(names) + 1.2))
    mx = max([(v.get("passed", 0) or 0) + (v.get("failed", 0) or 0) for v in suites.values()] + [5])
    for i, k in enumerate(names[::-1]):
        v = suites[k]
        if v.get("status") != "ok":
            ax.barh(i, mx * 0.02, color=GREY, height=0.55)
            ax.text(mx * 0.04, i, "not run here (missing): " + str(v.get("what", ""))[:70], va="center", fontsize=10, color=INK2)
            continue
        p, f = v.get("passed", 0) or 0, v.get("failed", 0) or 0
        ax.barh(i, p, color=GOOD, height=0.55)
        if f:
            ax.barh(i, f, left=p, color=BAD, height=0.55)
        ax.text(p + f + mx * 0.01, i, f"{p} passed, {f} failed", va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(names)), names[::-1]); ax.set_xlabel("tests"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, mx * 1.5)
    tot_f = sum((v.get("failed", 0) or 0) for v in suites.values())
    ax.set_title("Live test suites: " + ("all green" if tot_f == 0 else f"{tot_f} FAILING"), loc="left")
    fig.tight_layout()
    return fig


def splits_frame() -> pd.DataFrame:
    """Live read of splits.json: every registered data set, its run counts per split and when it was registered."""
    sp = json.loads((AIENGINE / "splits.json").read_text())["sections"]
    role = {"synth_v3": "train/val/test (first look)", "synth_v3_val2": "extra validation (v3b selection)",
            "synth_v3_test2": "FRESH test: S7/IS3 headline", "synth_v3_test3": "FRESH test: 2nd confirmation",
            "synth_v3_val4": "IS2-v2 threshold selection only", "synth_v3_test4": "FRESH test: IS2 wording v2",
            "synth_coe": "COE-pipeline domain-gap study", "xjtu_sy": "public, PPR history", "ims": "public, PPR history",
            "mafaulda": "public, PPR history", "synthetic_rig": "early synthetic testbench (history)"}
    prov = {"xjtu_sy": "[MEASURED on XJTU-SY]", "ims": "[MEASURED on IMS]", "mafaulda": "[MEASURED on MaFaulDa]", "synth_coe": "[SIMULATION]+[SYNTHETIC]"}
    order = list(role)
    rows = []
    for k, v in sp.items():
        c = v.get("counts", {})
        rows.append({"set": k, "role": role.get(k, "-"), "train": c.get("train", 0) + c.get("ft_train", 0), "val": c.get("val", 0),
                     "test": c.get("test", 0) + c.get("ft_test", 0), "registered (UTC)": str(v.get("created_utc", "-"))[:16].replace("T", " "),
                     "provenance": prov.get(k, "[SYNTHETIC]"), "_o": order.index(k) if k in order else 99})
    return pd.DataFrame(rows).sort_values("_o").drop(columns="_o").reset_index(drop=True)
