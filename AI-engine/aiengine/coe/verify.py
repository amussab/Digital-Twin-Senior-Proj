"""Verification harness for the COE specs -> AI-engine/reports/coe_verification.{json,md}.

    python -m aiengine.coe.verify            # full run (includes the 1-hour C6 simulation, a few minutes)
    python -m aiengine.coe.verify --quick    # C6 over 5 simulated minutes (smoke test only)

Provenance tags on every number: [SIMULATION] = produced by running this code on simulated
samples; [MEASURED host] = wall time on this PC (NOT STM32 time); [CALCULATED] = arithmetic from
the COE document / stated assumptions. Nothing here is a measurement of the team's rig.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from ..config import REPORTS_DIR
from .config import BLOCK_MS, CoeConfig
from .pipeline import CoePipeline
from .sources import DropInjector, SimulatedRigSource

SEED = 20261024
FAULTS = ["healthy", "outer_race", "inner_race", "ball", "cage"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPORTS_DIR.parent,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _pct(a, q):
    return float(np.percentile(a, q))


def stm32_cycle_estimate(rpm: float) -> dict:
    """[CALCULATED] operation-count estimate of work after t20 on the STM32H755 Cortex-M7.

    Assumptions (all stated, none measured): CM7 at 480 MHz (ST datasheet maximum, not stated in the
    COE document); single-precision FPU; 2 cycles per flop to absorb loads/stores and non-DSP-library
    code (a CMSIS-DSP FFT is faster, so this is conservative); pessimistic 10 cycles/flop case also given.
    """
    n_env = 20 * 60.0 / rpm * 5000.0
    n_fft = 2048                                     # 4096-pt real FFT = 2048-pt complex + split
    fft_flops = 5 * n_fft * math.log2(n_fft) + 10 * n_fft
    per_ch_spectrum = fft_flops + 3 * n_env + 4 * 12 * 4        # mean/Hann/pad + 4 bands x ~12 bins
    spectra = 4 * per_ch_spectrum
    feats = 4 * 40                                   # moments -> rms/kurtosis/crest, sqrt
    last_block = 256 * 4 * 70                        # calibrate + 4 BP biquads + abs + 2 LP biquads + stats
    rev_samples = rpm and (60.0 / rpm * 30000.0)     # last revolution of proximity samples
    prox_last_rev = rev_samples * 2 * 150            # incremental sin/cos + 2 MAC per sample per probe
    prox_final = 2 * 100                             # hypot + atan2
    payload = 200
    flops = spectra + feats + last_block + prox_last_rev + prox_final + payload
    out = {"rpm": rpm, "flops_after_t20": round(flops), "components": {
        "C6_spectra_4ch": round(spectra), "C7_features": feats, "last_dma_block_C2_C5": last_block,
        "C8_last_revolution": round(prox_last_rev), "C9_payload": payload}}
    for name, cpf in (("nominal_2cyc_per_flop", 2.0), ("pessimistic_10cyc_per_flop", 10.0)):
        ms = flops * cpf / 480e6 * 1e3
        out[name] = {"processing_ms": round(ms, 2),
                     "plus_wait_for_block_end_ms": round(ms + BLOCK_MS, 2),
                     "margin_vs_333ms": round(333.0 / (ms + BLOCK_MS), 1)}
    stream_flops = 256 * 4 * 70 + 256 * 2 * 150 * 0.05   # per-block streaming load, prox sin/cos cost amortised
    out["streaming_cpu_load_pct_of_8.533ms_block"] = round(stream_flops * 2.0 / 480e6 * 1e3 / BLOCK_MS * 100, 2)
    return out


# --------------------------------------------------------------------------- COE1
def coe1(seconds: float = 10.0) -> dict:
    cfg = CoeConfig()
    n_blocks = int(seconds * cfg.fs_hz / cfg.block_sets)
    src = SimulatedRigSource("outer_race", 0.5, 1750.0, seed=SEED, n_blocks=n_blocks)
    pipe = CoePipeline(src)
    t0 = time.perf_counter()
    n_sets, stds, last_seq = 0, np.zeros(4), None
    seq_ok = True
    for b in src.blocks():
        n_sets += b.codes.shape[0]
        stds = np.maximum(stds, b.codes[:, :4].astype(float).std(0))
        seq_ok &= (last_seq is None or b.seq == last_seq + 1)
        last_seq = b.seq
        pipe.push_block(b)
    wall = time.perf_counter() - t0
    sim_s = n_sets / cfg.fs_hz
    return {
        "spec": "COE1 / S4: >=4 channels simultaneously at >=25 kS/s per channel",
        "verdict_design": "MET (design) [CALCULATED + SIMULATION]; hardware test pending",
        "calculated": {
            "tag": "[CALCULATED]",
            "per_channel_rate_ksps": cfg.fs_hz / 1e3, "target_ksps": 25.0,
            "margin_pct": round((cfg.fs_hz / 25e3 - 1) * 100, 1),
            "convst_period_us": round(1e6 / cfg.fs_hz, 3),
            "ad7606_conversion_max_us": 4.2, "spi_read_128bit_at_10MHz_us": 12.8,
            "conversion_plus_read_us": 17.0, "slack_before_next_convst_us": round(1e6 / cfg.fs_hz - 17.0, 1),
            "dma_block_ms": round(BLOCK_MS, 3), "sample_sets_per_s": cfg.fs_hz,
            "note": "AD7606 samples V1-V8 simultaneously (track-and-hold per channel); timing from the COE doc and AD7606 datasheet figures quoted there.",
        },
        "simulation": {
            "tag": "[SIMULATION]", "simulated_seconds": round(sim_s, 3), "sample_sets": n_sets,
            "samples_per_channel": n_sets, "effective_rate_sps_per_channel": n_sets / sim_s,
            "channels_with_signal": int((stds > 0).sum()), "dma_sequence_gap_free": bool(seq_ok),
            "pipeline_windows_valid": pipe.stats.valid, "pipeline_dropped_blocks": pipe.stats.dropped_blocks,
            "host_processing_wall_s": round(wall, 2),
            "host_realtime_factor": round(sim_s / wall, 1),
            "note": "Simultaneity is a property of the ADC; the simulation shows the pipeline ingests 4 accel + 2 prox words per sample set at 30 kS/s with no sequence gaps.",
        },
    }


# --------------------------------------------------------------------------- COE2
def coe2(n_windows: int = 240) -> dict:
    res = {"spec": "COE2 / S5: <=333 ms from t20 to payload ready (4 ch, 20 revolutions)",
           "host": {"tag": "[MEASURED host]", "platform": platform.platform(), "processor": platform.processor(),
                    "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                    "caveat": "wall time of the Python/NumPy finalisation on this PC, NOT STM32 time. "
                              "finalize_ms = Components 6-9; t20_to_payload_ms = processing of the block holding t20 "
                              "(filters+stats for that 256-set block) + finalize."},
           "by_rpm": {}}
    rng = np.random.default_rng(SEED)
    for rpm in (1750.0, 3600.0):
        dur = n_windows * 1200.0 / rpm
        # mix of classes so FFT/feature paths see realistic data; one long stream per class chunk
        fin, tot, valid = [], [], 0
        per = n_windows // len(FAULTS) + 1
        for fi, fault in enumerate(FAULTS):
            src = SimulatedRigSource(fault, float(rng.uniform(0.3, 0.9)) if fault != "healthy" else 0.0,
                                     rpm, seed=SEED + fi, n_blocks=int((per + 2) * 1200 / rpm * 30000 / 256) + 400)
            pipe = CoePipeline(src)
            for w in pipe.windows():
                fin.append(w.finalize_ms); tot.append(w.t20_to_payload_ms); valid += 1
                if len([1 for _ in fin]) >= (fi + 1) * per:
                    break
        fin, tot = np.array(fin), np.array(tot)
        res["by_rpm"][str(int(rpm))] = {
            "windows": int(len(fin)),
            "finalize_ms": {"p50": _pct(fin, 50), "p95": _pct(fin, 95), "max": float(fin.max())},
            "t20_to_payload_ms": {"p50": _pct(tot, 50), "p95": _pct(tot, 95), "max": float(tot.max())},
            "host_max_vs_333ms_budget_pct": round(float(tot.max()) / 333.0 * 100, 2),
        }
    worst = max(v["t20_to_payload_ms"]["max"] for v in res["by_rpm"].values())
    res["host_verdict"] = f"host-PC worst case {worst:.2f} ms <= 333 ms [MEASURED host]; not an STM32 measurement"
    res["stm32_estimate"] = {"tag": "[CALCULATED]", "by_rpm": {str(r): stm32_cycle_estimate(r) for r in (1750.0, 3600.0)},
                             "assumptions": ["Cortex-M7 at 480 MHz (ST datasheet maximum; not stated in the COE doc, confirm)",
                                             "5*N*log2(N) flops for the 2048-pt complex FFT behind each 4096-pt real FFT",
                                             "~70 flops per sample-channel for calibrate + 4 BP biquads + rectify + 2 LP biquads + moment sums",
                                             "~150 flops per proximity sample per probe for the last revolution (sin/cos totals)",
                                             "2 cycles/flop nominal, 10 cycles/flop pessimistic; ADC wait for the DMA block holding t20 added (8.533 ms)",
                                             "UART transfer to the ESP32-C6 is outside the 'payload available in STM32 memory' definition"]}
    res["verdict_design"] = "MET (design) [CALCULATED]; host-PC measurement supports feasibility; STM32 measurement pending"
    return res


# --------------------------------------------------------------------------- COE3
def coe3() -> dict:
    cfg = CoeConfig(speed_policy="range", rpm_range=(950.0, 3650.0))   # +-5 % margin around 1000-3600 for speed jitter
    rows, ok = [], True
    shapes, sizes, dom_ok = set(), set(), 0
    rpms = list(range(1000, 3601, 100)) + [1750]
    for rpm in rpms:
        src = SimulatedRigSource("outer_race", 0.7, float(rpm), seed=SEED, n_blocks=int(3.2 * 30000 / 256), cfg=cfg)
        w = next(iter(CoePipeline(src, cfg).windows()))
        shapes.add(w.features.shape)
        sizes.add(len(w.payload))
        fam = w.features.reshape(4, 8)[0, 4:]
        dom = int(np.argmax(fam)) == 2          # BPFO
        dom_ok += dom
        ok &= bool(np.isfinite(w.features).all())
        rows.append({"rpm_nominal": rpm, "rpm_measured": round(w.rpm, 1), "payload_bytes": len(w.payload),
                     "features_shape": list(w.features.shape), "envelope_record_truncated_to_4096": w.truncated,
                     "bpfo_band_dominant_ch0": bool(dom)})
    same = len(shapes) == 1 and len(sizes) == 1 and shapes == {(32,)} and sizes == {152}
    return {"spec": "COE3 / S6: fixed dimension & ordering for every rpm in 1000-3600",
            "tag": "[SIMULATION]", "n_speeds": len(rows), "identical_shapes": bool(same),
            "payload_bytes_set": sorted(sizes), "feature_shapes": [list(s) for s in shapes], "all_finite": ok,
            "bpfo_dominant_at_n_speeds": dom_ok, "rows": rows,
            "verdict": "MET (design) [SIMULATION]" if same and ok else "NOT MET",
            "caveat": "Below ~1465 rpm a 20-rev envelope record exceeds 4096 samples (1200/rpm*5000); the COE doc fixes the "
                      "FFT at 4096 so the first 4096 samples are used (flag 'envelope_record_truncated_to_4096'). The doc's "
                      "own supported modes are 1750 and 3600 rpm, where nothing is truncated. Open question for COE.",
            "speed_policy_used": "range 950-3650 so jitter at the 1000/3600 edges is accepted (default pipeline accepts only 1750/3600 +-1 %)"}


# --------------------------------------------------------------------------- C6
def gilbert_elliott(n: int, rng, p_gb=0.001, p_bg=0.3, loss_bad=0.8, loss_good=0.0):
    lost = np.zeros(n, dtype=bool)
    bad = False
    u = rng.random((n, 2))
    for i in range(n):
        bad = (u[i, 0] < p_gb) if not bad else not (u[i, 0] < p_bg)
        lost[i] = u[i, 1] < (loss_bad if bad else loss_good)
    return lost


def c6(hours: float = 1.0, rpm: float = 1750.0, p_block: float = 1e-5, p_edge: float = 1e-5) -> dict:
    cfg = CoeConfig()
    sim_s = hours * 3600.0
    n_blocks = int(sim_s * cfg.fs_hz / cfg.block_sets)
    inner = SimulatedRigSource("outer_race", 0.5, rpm, seed=SEED, n_blocks=n_blocks)
    inj = DropInjector(inner, p_block_drop=p_block, p_edge_miss=p_edge, seed=SEED + 1)
    pipe = CoePipeline(inj)
    t0 = time.perf_counter()
    delivered_at_node = sum(1 for _ in pipe.windows())
    wall = time.perf_counter() - t0
    ideal = int(sim_s // (1200.0 / rpm))
    rng = np.random.default_rng(SEED + 2)
    lost = gilbert_elliott(delivered_at_node, rng)
    udp_lost = int(lost.sum())
    delivered = delivered_at_node - udp_lost
    node_discarded = pipe.stats.total_discarded
    loss_pct = (ideal - delivered) / ideal * 100.0
    node_loss_pct = (ideal - delivered_at_node) / ideal * 100.0
    sens = {f"iid_udp_loss_{p}%": round(node_loss_pct + p * (1 - node_loss_pct / 100), 3) for p in (0.1, 0.5, 1.0, 2.0)}
    return {
        "spec": "C6 (proposed wireless-link reliability constraint): window loss <= 1 % over 1 h",
        "tag": "[SIMULATION]", "simulated_hours": hours, "rpm": rpm, "host_wall_s": round(wall, 1),
        "loss_model_assumptions": {
            "dma_block_drop_probability": p_block, "keyphasor_edge_miss_probability": p_edge,
            "wifi_udp_loss": "Gilbert-Elliott burst model: P(good->bad)=0.001 per payload, P(bad->good)=0.3, loss prob in bad state 0.8, 0 in good; "
                             "no retransmission (COE doc: no acks/retries). Stationary mean loss ~0.27 %.",
            "status": "ASSUMED rates, not measured on the rig or on the team's Wi-Fi; replace with measured values when hardware exists",
        },
        "injected": {"dropped_blocks": inj.dropped_blocks, "missed_keyphasor_edges": inj.missed_edges},
        "windows": {"ideal_windows_in_period": ideal, "valid_at_node": delivered_at_node,
                    "discarded_at_node": node_discarded, "discard_reasons": pipe.stats.discarded,
                    "lost_on_wifi_udp": udp_lost, "delivered_to_host": delivered},
        "window_loss_pct": {"node_only": round(node_loss_pct, 3), "end_to_end": round(loss_pct, 3), "limit": 1.0},
        "sensitivity_end_to_end_loss_pct_vs_iid_udp_loss": sens,
        "verdict": ("MET under stated loss assumptions [SIMULATION]" if loss_pct <= 1.0
                    else "NOT MET under stated loss assumptions [SIMULATION]"),
        "note": "Window loss = (ideal back-to-back 20-rev windows in the period - windows delivered) / ideal. "
                "A lost block also costs the filter re-settle (100 ms) before the next window may start. "
                "Spec-sheet C6 is currently the ME enclosure constraint; the wireless-reliability constraint is only a proposal.",
    }


def to_md(r: dict) -> str:
    c1, c2, c3, c6r = r["COE1"], r["COE2"], r["COE3"], r["C6"]
    L = ["# COE pipeline verification", "",
         f"Generated {r['meta']['generated']} | commit {r['meta']['commit']} | seed {r['meta']['seed']} | "
         f"command `{r['meta']['command']}`", "",
         "ALL DATA SIMULATED. Tags: [SIMULATION] code run on simulated samples; [MEASURED host] wall time on this PC "
         "(not STM32); [CALCULATED] arithmetic from the COE doc / stated assumptions. Nothing is a rig measurement.", "",
         "| Metric | Value | Verdict | Provenance |", "|---|---|---|---|",
         f"| COE1 sample rate x channels | {c1['calculated']['per_channel_rate_ksps']} kS/s x 4 ch (target >=25); "
         f"{c1['simulation']['simulated_seconds']} s simulated, {c1['simulation']['samples_per_channel']} sets/ch, gap-free="
         f"{c1['simulation']['dma_sequence_gap_free']}; ADC conv+read {c1['calculated']['conversion_plus_read_us']} us of "
         f"{c1['calculated']['convst_period_us']} us | {c1['verdict_design']} | [CALCULATED]+[SIMULATION] |"]
    for rpm, v in c2["by_rpm"].items():
        L.append(f"| COE2 host t20->payload @{rpm} rpm ({v['windows']} windows) | p50 {v['t20_to_payload_ms']['p50']:.2f} / "
                 f"p95 {v['t20_to_payload_ms']['p95']:.2f} / max {v['t20_to_payload_ms']['max']:.2f} ms "
                 f"(finalize only p50 {v['finalize_ms']['p50']:.2f}, max {v['finalize_ms']['max']:.2f}) | <=333 ms on host PC | [MEASURED host] |")
    for rpm, e in c2["stm32_estimate"]["by_rpm"].items():
        L.append(f"| COE2 STM32H755 estimate @{int(float(rpm))} rpm | {e['flops_after_t20']} flops after t20; "
                 f"{e['nominal_2cyc_per_flop']['plus_wait_for_block_end_ms']} ms nominal / "
                 f"{e['pessimistic_10cyc_per_flop']['plus_wait_for_block_end_ms']} ms pessimistic incl. 8.533 ms block wait | "
                 f"<=333 ms (margin {e['pessimistic_10cyc_per_flop']['margin_vs_333ms']}x pessimistic) | [CALCULATED] |")
    L += [f"| COE3 fixed dimension | {c3['n_speeds']} speeds 1000-3600: payload bytes {c3['payload_bytes_set']}, "
          f"feature shapes {c3['feature_shapes']}, BPFO dominant at {c3['bpfo_dominant_at_n_speeds']}/{c3['n_speeds']} | {c3['verdict']} | [SIMULATION] |",
          f"| C6 window loss, {c6r['simulated_hours']} h @ {c6r['rpm']:.0f} rpm | node {c6r['window_loss_pct']['node_only']} % / "
          f"end-to-end {c6r['window_loss_pct']['end_to_end']} % (limit 1 %); valid {c6r['windows']['valid_at_node']}, discarded "
          f"{c6r['windows']['discarded_at_node']}, Wi-Fi lost {c6r['windows']['lost_on_wifi_udp']} of {c6r['windows']['ideal_windows_in_period']} ideal | "
          f"{c6r['verdict']} | [SIMULATION] |", "",
          "## Caveats", "",
          f"- COE2: {c2['host']['caveat']} Host: {c2['host']['processor']} / {c2['host']['platform']}.",
          f"- COE3: {c3['caveat']}",
          f"- C6: {c6r['loss_model_assumptions']['status']}. Loss model: block drop {c6r['loss_model_assumptions']['dma_block_drop_probability']}, "
          f"edge miss {c6r['loss_model_assumptions']['keyphasor_edge_miss_probability']}; {c6r['loss_model_assumptions']['wifi_udp_loss']}",
          f"- C6: {c6r['note']}", "",
          "## STM32 estimate assumptions", ""] + [f"- {a}" for a in c2["stm32_estimate"]["assumptions"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="C6 over 5 simulated minutes, fewer COE2 windows")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    a = ap.parse_args(argv)
    cmd = "python -m aiengine.coe.verify" + (" --quick" if a.quick else "")
    report = {"meta": {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": _git_commit(), "seed": SEED,
                       "command": cmd},
              "COE1": coe1(), "COE2": coe2(60 if a.quick else 240), "COE3": coe3(),
              "C6": c6(hours=5 / 60 if a.quick else 1.0)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "coe_verification_quick" if a.quick else "coe_verification"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float))
    (out / f"{stem}.md").write_text(to_md(report), encoding="utf8")
    print(to_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
