"""Markdown rendering of the v5_node test report."""
from __future__ import annotations

import json


def _f(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def render(o: dict) -> str:
    m, t, g, c, rep = o["meta"], o["test"], o["gates"], o["controls"], o["reported_not_gated"]
    s7, b7, cl, st, ca = t["S7"], t["S7_baseline"], t["IS3_classification"], t["IS3_stage"], t["IS3_caught"]

    def ok(b):
        return "PASS" if b else "FAIL"
    L = ["# v5_node test report", "", f"Provenance: {m['provenance']}", f"Limits: {m['limits']}", "",
         f"seed {m['seed']}, commit {m['git_commit']} (dirty={m['git_dirty']}), command `{m['command']}`, utc {m['utc']}",
         f"frozen choices file `{m['frozen_file']}` sha256 {m['frozen_sha256'][:16]}... written BEFORE the test units were scored; test scored once.",
         f"data counts: {json.dumps(m['data_counts'])}", "",
         "## Frozen choices (selected on VAL only)", "```", json.dumps(o["frozen"], indent=1, default=float), "```", "",
         "## Gates on TEST (held-out simulated trajectories, not held-out bearings)", "",
         "| gate | value | result |", "|---|---|---|",
         f"| S7 MAPE <= 15 % (detected-onset degradation window, bearing-averaged) | {_f(s7['S7_mape_pct'], 2)} % over {s7['n_bearings']} units | {ok(g['S7_mape_le_15'])} |",
         f"| IS3 macro-F1 >= 0.85 (5 classes, every window, both bearings) | {_f(cl['macro_f1'])} | {ok(g['IS3_macro_f1_ge_0.85'])} |",
         f"| IS3 stage error <= 1 (max abs, all windows of faulted units) | {st['max_abs_error']} (within-1 {_f(st['within1_frac'], 4)}, windows err>=2: {st.get('windows_error_ge2')}) | {ok(g['IS3_stage_error_le_1'])} |",
         f"| IS3 caught by stage 3 (3 consecutive correct windows) | {ca['caught_by_stage3']}/{ca['rtf_bearings']} (never caught {ca['never_caught']}; by class {ca.get('per_class_by_stage3')}) | {ok(g['IS3_caught_by_stage3_all'])} |",
         "", f"VAL (selection) numbers: {json.dumps(o['val_selected'], default=float)}", "",
         "## Controls (S7)", "",
         f"- Trivial baseline (log-linear on one health index HI, same scoring, own alpha chosen on val): **{_f(b7['S7_mape_pct'], 2)} %** vs GBM {_f(s7['S7_mape_pct'], 2)} %. "
         "If the baseline is close to the GBM, the number measures the ASSUMED damage curve, not diagnostic skill.",
         f"- Per-unit S7 summary GBM {json.dumps(s7['per_bearing_summary'])}; baseline {json.dumps(b7['per_bearing_summary'])}",
         f"- Wrong-shape control (frozen heads, same test units with near-linear curves): {json.dumps(c['wrong_shape'], default=float)}", "",
         "## Reported, not gated", "",
         f"- Cross-speed (1800-only train -> 3600 test): {json.dumps(rep['cross_speed_1800_to_3600'], default=float)}",
         f"- Per-speed macro-F1 on test: {json.dumps(rep['per_rpm_macro_f1'])}",
         f"- S7 per rpm {json.dumps(s7.get('per_rpm_mode'))}; per fault class {json.dumps(s7.get('per_fault_class'))}",
         f"- Healthy false alarms (test): {json.dumps(t['healthy_alarms'], default=float)}",
         f"- Real windows via NodeEngine (anchor source, not independent): {json.dumps(rep['real_checks']['real_windows'], default=float)}",
         f"- Real BA-1 failure trace: {json.dumps(rep['real_checks']['failure_trace'], default=float)}",
         f"- Latency: {json.dumps(rep['real_checks']['latency'])}", "",
         "## Limits",
         "- No real run-to-failure data exists for 4 of 5 fault classes; units resample 1 bearing/fault; the curve family is assumed and test curves are OOD within it.",
         "- Labels are a monotone function of the assumed curve: S7/IS3 here are Rank 6 at most (simulation + data), never 'measured on rig'.",
         "- RUL is in table windows (10 min snapshots); a host must feed one window per snapshot.", ""]
    return "\n".join(L)
