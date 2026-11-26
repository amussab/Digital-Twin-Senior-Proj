"""Markdown rendering of the v7_node fresh-test report."""
from __future__ import annotations

import glob
import json

from .config import REPORTS_DIR

TAG = "[SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape + life variance assumed]"


def _f(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def _sens() -> list[str]:
    fs = sorted(glob.glob(str(REPORTS_DIR / "s7_research_*.json")))
    if not fs:
        return []
    d = json.loads(open(fs[-1], encoding="utf-8").read())
    L = ["## Sensitivity of S7 to the assumed life variance (research, VAL units only)", "",
         f"{TAG}. Same model recipe (v5 recipe: xgb, NL=48 + HI levels, depth 6), retrained per profile; val S7 (detected onset).", "",
         "| profile | life_sigma | other nuisances | val S7 % |", "|---|---|---|---|"]
    for r in d.get("sensitivity_table", []):
        L.append(f"| {r['profile']} | {r['life_sigma']} | {r['nuisances']} | {r['val_S7']:.2f} |")
    L += ["", f"Research report: `{fs[-1].replace(chr(92), '/').split('/')[-1][:-5]}.md`", ""]
    return L


def render(o: dict) -> str:
    m, t, g, c, rep = o["meta"], o["test"], o["gates"], o["controls"], o["reported_not_gated"]
    s7, b7, cl, st, ca = t["S7"], t["S7_baseline"], t["IS3_classification"], t["IS3_stage"], t["IS3_caught"]

    def ok(b):
        return "PASS" if b else "FAIL"
    L = ["# v7_node fresh-test report", "", f"Provenance: {TAG}", "",
         "**Everything below is simulated (coach-approved basis).** The pass/fail depends on the ASSUMED life variance and curve shape of the "
         "learnable (L) profile; see the sensitivity table. Real-rig checks are in their own section and are not flattering.", "",
         f"Limits: {m['limits']}", "",
         f"seed {m['seed']}, commit {m['git_commit']} (dirty={m['git_dirty']}), command `{m['command']}`, utc {m['utc']}",
         f"frozen choices file `{m['frozen_file']}` sha256 {m['frozen_sha256'][:16]}... written BEFORE the fresh test units were scored; scored once.",
         f"data counts: {json.dumps(m['data_counts'])}", "",
         "## Frozen choices (selected on VAL only)", "```", json.dumps(o["frozen"], indent=1, default=float), "```", "",
         "## Gates on the fresh TEST (PRIMARY set test_L: held-out simulated trajectories, OOD shape; not held-out bearings)", "",
         "| gate | value | result |", "|---|---|---|",
         f"| S7 MAPE <= 15 % (detected-onset degradation window, bearing-averaged) | {_f(s7['S7_mape_pct'], 2)} % over {s7['n_bearings']} units | {ok(g['S7_mape_le_15'])} |",
         f"| IS3 macro-F1 >= 0.85 | {_f(cl['macro_f1'])} | {ok(g['IS3_macro_f1_ge_0.85'])} |",
         f"| IS3 stage error <= 1 (max abs) | {st['max_abs_error']} (within-1 {_f(st['within1_frac'], 4)}, windows err>=2: {st.get('windows_error_ge2')}) | {ok(g['IS3_stage_error_le_1'])} |",
         f"| IS3 caught by stage 3 | {ca['caught_by_stage3']}/{ca['rtf_bearings']} (never caught {ca['never_caught']}; by class {ca.get('per_class_by_stage3')}) | {ok(g['IS3_caught_by_stage3_all'])} |",
         "", f"VAL (selection) numbers: {json.dumps(o['val_selected'], default=float)}", "",
         "## Same frozen heads on the other pre-declared variants of the same test unit ids (reported, not gated)", "",
         "| set | S7 % | S7 true-onset % | baseline S7 % | macro-F1 | stage max | caught by stage 3 |", "|---|---|---|---|---|---|---|"]
    rows = [("test_L (primary)", {"S7_mape_pct": s7["S7_mape_pct"], "S7_true_onset_pct": s7["alt_mape_true_onset_pct"], "S7_baseline_mape_pct": b7["S7_mape_pct"],
                                  "IS3_macro_f1": cl["macro_f1"], "IS3_stage_max": st["max_abs_error"], "IS3_caught": ca})] + list(o["side_by_side_same_units"].items())
    for nm, r in rows:
        cc = r["IS3_caught"]
        L.append(f"| {nm} | {_f(r['S7_mape_pct'], 2)} | {_f(r['S7_true_onset_pct'], 2)} | {_f(r['S7_baseline_mape_pct'], 2)} | {_f(r['IS3_macro_f1'])} | {r['IS3_stage_max']} | {cc['caught_by_stage3']}/{cc['rtf_bearings']} |")
    L += ["", *[f"- {k}: {v}" for k, v in o["test_set_definitions"].items()], "",
          "## Controls (S7)", "",
          f"- Trivial baseline (log-linear on HI): **{_f(b7['S7_mape_pct'], 2)} %** vs GBM {_f(s7['S7_mape_pct'], 2)} %.",
          f"- Per-unit S7 summary GBM {json.dumps(s7['per_bearing_summary'])}; baseline {json.dumps(b7['per_bearing_summary'])}",
          f"- Wrong-shape control: S7 {_f(c['wrong_shape']['S7_mape_pct'], 2)} % (baseline {_f(c['wrong_shape']['S7_baseline_mape_pct'], 2)} %)", "",
          "## Reported, not gated", "",
          f"- Cross-speed (1800-only train -> 3600 test): {json.dumps(rep['cross_speed_1800_to_3600'], default=float)}",
          f"- Per-speed macro-F1 on test: {json.dumps(rep['per_rpm_macro_f1'])}",
          f"- S7 per rpm {json.dumps(s7.get('per_rpm_mode'))}; per fault class {json.dumps(s7.get('per_fault_class'))}",
          f"- Healthy false alarms (test): {json.dumps(t['healthy_alarms'], default=float)}", "",
          "## Real-rig checks (NOT simulated; not independent of the anchors; unflattering numbers stay in)", "",
          f"- Real windows via NodeEngine: {json.dumps(rep['real_checks']['real_windows'], default=float)}",
          f"- Real BA-1 failure trace (n=1): {json.dumps(rep['real_checks']['failure_trace'], default=float)}",
          f"- Latency: {json.dumps(rep['real_checks']['latency'])}", "", *_sens(),
          "## Limits",
          "- No real run-to-failure data exists for 4 of 5 fault classes; units resample 1 bearing/fault; the curve family AND the run-to-run life variance are assumed.",
          "- The L profile narrows life variance / shape / amplitude nuisances to make the run-to-run rate observable (UP2 precedent); a pass here is conditional on that.",
          "- Labels are a monotone function of the assumed curve: S7/IS3 here are Rank 6 at most (simulation + data), never 'measured on rig'.",
          "- RUL is in table windows (10 min snapshots); a host must feed one window per snapshot.", ""]
    return "\n".join(L)
