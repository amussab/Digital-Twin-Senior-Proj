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
        return {"healthy_stage2_share": healthy_stage2, "healthy_true_stage1_share": healthy_true_stage1, "healthy_windows": int(len(h)),
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
        cls = E.onnx_latency(MODELS / c["models"]["tft"]["onnx"], lambda i: {
            "encoder_cont": bc[k(i): k(i) + 1, :-1], "decoder_cont": bc[k(i): k(i) + 1, -1:]}, n=n)
        from aiengine.engine import HybridEngine
        step = HybridEngine(MODELS, contract_file="model_contract_v3b.json").benchmark(g, n=200)
        return {"status": "ok", "cls": cls, "engine_step": step, "unit": unit, "contract": "model_contract_v3b.json"}
    except Exception as e:  # noqa: BLE001
        return err("S8 live latency", e)


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
    fig, ax = plt.subplots(4, 1, figsize=(10, 9))
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
    for name, col in zip(("FTF", "BSF", "BPFO", "BPFI"), (PAL[3], PAL[4], PAL[7], PAL[5])):
        f = pipe.cfg.orders[name] * fr_rot
        ax[3].axvline(f, color=col, ls="--", lw=1.2); ax[3].text(f, ax[3].get_ylim()[1] * 0.92, f" {name}\n {f:.0f} Hz", color=col, fontsize=8, va="top")
    ax[3].set_title("4. Envelope spectrum (Component 6) with bearing-defect orders x shaft speed"); ax[3].set_xlabel("Hz"); ax[3].set_ylabel("amplitude")
    fig.suptitle("COE pipeline stages from the real aiengine.coe code  [SIMULATION] + [SYNTHETIC] source; not a rig measurement", fontsize=10, y=0.995)
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
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for bn, col in ((1, PAL[0]), (2, PAL[1])):
        d = df[df.bearing == bn]
        ax[0].plot(d.t_h, d.rul_h, color=col, label=f"bearing {bn}")
        ax[1].plot(d.t_h, d.stage, color=col, drawstyle="steps-post")
        ax[2].plot(d.t_h, d.hi, color=col)
    cm = {c: i for i, c in enumerate(["healthy", "outer_race", "inner_race", "ball", "cage"])}
    d1 = df[df.bearing == 1]
    ax[0].set_ylabel("AI RUL, h"); ax[0].legend(); ax[1].set_ylabel("health stage"); ax[2].set_ylabel("health index"); ax[2].set_xlabel("stream time, h")
    chg = d1[d1.cls != d1.cls.shift()]
    for _, r in chg.iterrows():
        ax[0].axvline(r.t_h, color=INK2, lw=0.6, ls=":"); ax[0].text(r.t_h, ax[0].get_ylim()[1] * 0.95, " " + r.cls, fontsize=8, rotation=90, va="top")
    ax[0].set_title("UP2 demo stream through the deployed v3b engine  [SYNTHETIC] stream, [MEASURED host] engine; dotted = class change of bearing 1")
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
               "lat_p50": float(np.percentile(lat, 50)), "lat_p95": float(np.percentile(lat, 95)), "lat_max": float(np.max(lat))}
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
        t0 = time.perf_counter()
        r = E.evaluate(name)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        r["status"] = "ok"
        return r
    except Exception as e:  # noqa: BLE001
        return err("IS2 v2 re-score", e)


def is2_live_subset(step: int = 4) -> dict:
    """Runs the DEPLOYED v3b engine (ONNX) live on every `step`-th faulted run-to-failure bearing of test4 and scores IS2 v2 with the
    frozen thresholds against the twin physics RUL. SUBSET (n bearings printed); [SYNTHETIC] + [SIMULATION]."""
    try:
        from aiengine import up2d_is2 as D
        from aiengine.config import MODELS_DIR
        from aiengine.engine import HybridEngine
        t0 = time.perf_counter()
        truth = D.load_set("test4")
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
