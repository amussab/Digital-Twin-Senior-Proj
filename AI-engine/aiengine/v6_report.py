"""Markdown rendering of the v6_node fresh-test report (v5 and v6 side by side)."""
from __future__ import annotations

import json


def _f(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def _tbl(rows: list, cols: list, n: int | None = None) -> list:
    rows = rows[:n] if n else rows
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(json.dumps(r.get(c)) if isinstance(r.get(c), (dict, list)) else _f(r.get(c)) for c in cols) + " |")
    return out


def render(o: dict, fz: dict) -> str:
    m, t, g, c, rep, vv = o["meta"], o["test"], o["gates"], o["controls"], o["reported_not_gated"], o["v5_vs_v6"]
    s7, b7, cl, st, ca = t["S7"], t["S7_baseline"], t["IS3_classification"], t["IS3_stage"], t["IS3_caught"]
    grid = fz["val_grid"]

    def ok(b):
        return "PASS" if b else "FAIL"
    v5o, v5f, v6 = vv["v5_registered_test_scored_once"], vv["v5_frozen_heads_on_fresh_units"], vv["v6_fresh_test"]
    fl_val, fl_te = fz["oracle_floor_val"]["S7_mape_pct"], c["oracle_floor_fresh_test"]["S7_mape_pct"]
    L = ["# v6_node fresh-test report", "", f"Provenance: {m['provenance']}", f"Limits: {m['limits']}", "",
         f"seed {m['seed']}, commit {m['git_commit']} (dirty={m['git_dirty']}), command `{m['command']}`, utc {m['utc']}",
         f"frozen choices file `{m['frozen_file']}` sha256 {m['frozen_sha256']} written BEFORE the fresh test units were scored; scored once. "
         f"ONNX/contract sha256[:16] {json.dumps(m['onnx_sha256_16'])}",
         f"data counts: {json.dumps(m['data_counts'])}", "",
         "## Disclosures (read first)",
         "- v5_node FAILED all four gates on its registered test split (reports/v5_node_*.json, marker v5_node_test_done.json). That result stays in the evidence unchanged and is not re-scored.",
         "- v6's design (more training units, long-context RUL/stage features, wider post-process grid, classifier context / class-weight grid) was MOTIVATED BY the diagnostics of that v5 test pass. "
         "The old test units are therefore not used for anything in v6; the arbiter is a FRESH test set (splits.json section `rig_anchor_test2`, committed before any v6 fit, ids u1000-u1079, new per-unit seeds).",
         "- The fresh test units are NOT new bearings: they re-use the SAME real anchor windows (the last 20 % time blocks of the same 5 real rig runs, test pool) as the v5 test units, "
         "with new damage-curve parameters, lives, onsets, nuisance and block draws. All selection used the rig_anchor VAL units only.",
         "- Simulation tag applies to every number: curves are an assumed family; labels are a monotone function of the assumed curve.", "",
         "## Gates on the FRESH test", "",
         "| gate | value | result |", "|---|---|---|",
         f"| S7 MAPE <= 15 % (detected-onset degradation window, bearing-averaged) | {_f(s7['S7_mape_pct'], 2)} % over {s7['n_bearings']} units | {ok(g['S7_mape_le_15'])} |",
         f"| IS3 macro-F1 >= 0.85 (5 classes, every window, both bearings) | {_f(cl['macro_f1'])} | {ok(g['IS3_macro_f1_ge_0.85'])} |",
         f"| IS3 stage error <= 1 (max abs, all windows of faulted units) | {st['max_abs_error']} (within-1 {_f(st['within1_frac'], 4)}, windows err>=2: {st.get('windows_error_ge2')}) | {ok(g['IS3_stage_error_le_1'])} |",
         f"| IS3 caught by stage 3 (3 consecutive correct windows) | {ca['caught_by_stage3']}/{ca['rtf_bearings']} (never caught {ca['never_caught']}; by class {ca.get('per_class_by_stage3')}) | {ok(g['IS3_caught_by_stage3_all'])} |",
         "", "## v5 vs v6 side by side", "",
         "| | S7 MAPE % | macro-F1 | stage max err | caught by stage 3 |", "|---|---|---|---|---|",
         f"| v5 on its registered test split (60 units, scored once, FAIL) | {_f(v5o['S7'], 2)} | {_f(v5o['F1'])} | {v5o['stage_max']} | {v5o['caught']}/48 |",
         f"| v5 frozen heads on the FRESH units (new set; not a re-score of the old one) | {_f(v5f['S7'], 2)} | {_f(v5f['F1'])} | {v5f['stage_max']} | {v5f['caught']} |",
         f"| **v6 on the FRESH units (scored once)** | **{_f(v6['S7'], 2)}** | **{_f(v6['F1'])}** | **{v6['stage_max']}** | **{v6['caught']}/{ca['rtf_bearings']}** |",
         "", f"VAL (selection) numbers of v6: {json.dumps(o['val_selected'], default=float)}", "",
         "## Frozen choices (selected on VAL only)", "```", json.dumps(o["frozen"], indent=1, default=float), "```", "",
         "## Controls (S7)", "",
         f"- Trivial baseline (log-linear on one health index, same scoring, own alpha chosen on val): **{_f(b7['S7_mape_pct'], 2)} %** vs v6 {_f(s7['S7_mape_pct'], 2)} %.",
         f"- Per-unit S7 summary v6 {json.dumps(s7['per_bearing_summary'])}; baseline {json.dumps(b7['per_bearing_summary'])}",
         f"- Wrong-shape control (80 NEW units, near-linear curves, frozen heads): {json.dumps(c['wrong_shape'], default=float)}",
         f"- Irreducible-error estimate: a privileged oracle that KNOWS each unit's life, curve shape, late scale and anchor level but sees the same noisy features "
         f"scores S7 {_f(fl_val, 2)} % on VAL and {_f(fl_te, 2)} % on the fresh test (descriptive, not used for any choice). Detail: {json.dumps(c['oracle_floor_fresh_test'], default=float)}",
         "", "## Reported, not gated", "",
         f"- Cross-speed (1800-only train -> 3600 fresh test): {json.dumps(rep['cross_speed_1800_to_3600'], default=float)}",
         f"- Per-speed macro-F1: {json.dumps(rep['per_rpm_macro_f1'])}",
         f"- S7 per rpm {json.dumps(s7.get('per_rpm_mode'))}; per fault class {json.dumps(s7.get('per_fault_class'))}; error vs life fraction {json.dumps(s7.get('error_vs_life_fraction'))}",
         f"- Classifier per-class F1 / recall: {json.dumps(cl['per_class_f1'])} / {json.dumps(cl['per_class_recall'])}",
         f"- Healthy false alarms: {json.dumps(t['healthy_alarms'], default=float)}",
         f"- Real windows via NodeEngine (anchor source, not independent): {json.dumps(rep['real_checks']['real_windows'], default=float)}",
         f"- Real BA-1 failure trace (n=1): {json.dumps(rep['real_checks']['failure_trace'], default=float)}",
         f"- Latency (S8, NodeEngine end-to-end): {json.dumps(rep['real_checks']['latency'])}", "",
         "## VAL grids (everything tried; selection rules in the frozen json)", "",
         "### Classifier (Tc x hp x gate)"] + _tbl(grid["classifier"], ["Tc", "hp", "gate_tau", "best_iteration", "val_macro_f1", "val_false_alarm", "val_caught_by_stage3"]) + [
         "", "### RUL head"] + _tbl(grid["rul"], ["NL", "max_depth", "mono", "tail", "best_iteration", "val_S7_mape_pct_alpha1", "val_S7_mape_pct_alpha0.5"]) + [
         "", "### Stage (best 12 by selection key)"] + _tbl(sorted(grid["stage"], key=lambda r: (r["val_stage_max"], r["val_stage_n_ge2"], -r["val_stage_within1"], r["val_stage_mae"])),
                                                            ["z_head", "z_cut", "stage_k", "stage_max_step", "val_stage_max", "val_stage_n_ge2", "val_stage_within1"], 12) + [
         "", "### RUL post-process (best 12)"] + _tbl(sorted(grid["rul_postprocess"], key=lambda r: r["val_S7_mape_pct"]),
                                                     ["rul_alpha", "rul_scale_early", "rul_scale_late", "val_S7_mape_pct"], 12) + [
         "", "## Limits",
         "- No real run-to-failure data exists for 4 of 5 fault classes; units resample 1 bearing/fault; the curve family is assumed and test curves are OOD within it.",
         "- Labels are a monotone function of the assumed curve: S7/IS3 here are Rank 6 at most (simulation + data), never 'measured on rig'.",
         "- Fresh test units share the real anchor windows with the old test units (same test pool blocks): sampling noise is re-drawn, the 5 physical bearings are not.",
         "- RUL is in table windows (10 min snapshots); a host must feed one window per snapshot. On a real node the window cadence is ~0.33 s (3600 rpm), so RUL in real time is not calibrated.", ""]
    return "\n".join(L)
