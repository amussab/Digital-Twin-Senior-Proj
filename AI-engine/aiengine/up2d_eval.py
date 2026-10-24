"""IS2 wording v2 check: ONE command that re-scores the SAVED synth_v3_test4 predictions (no retraining, no new choices).
[SYNTHETIC] synth_v3 generator + [SIMULATION] FE twin.

    AI-engine/.venv/Scripts/python.exe -m aiengine.up2d_eval [--set test4]
    -> reports/up2d_is2_<ts>.json + .md

Reports, on the same windows: IS2 v2 (reworded, verdict; thresholds FROZEN in aiengine/up2d_is2.py from validation),
IS2 v1 literal (max per-window |RUL_phys - RUL_AI| / RUL_AI <= 5 %, still reported), a normaliser/aggregation sweep,
the perfect-AI floor, negative controls and the residual-as-flag AUROC; deployed v3b is the verdict model, v3c research.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import up2d_is2 as D
from .config import REPORTS_DIR

V2_TEXT = (
    "IS2 Physics-AI RUL agreement (wording v2). For the faulted bearing of every run-to-failure run, over every window "
    "in true health stages 1-3 after the AI warm-up, the disagreement e = |RUL_physics - RUL_AI| / L_ref between the "
    "FE-twin physics RUL and the deployed AI RUL, expressed as a fraction of the bearing's reference (rated) life L_ref "
    "at its duty speed, shall satisfy: (a) median over bearings of each bearing's median e <= 0.18 (18 % of L_ref), and "
    "(b) 95th percentile over bearings of each bearing's 95th-percentile e <= 0.63 (63 % of L_ref).")


def evaluate(name: str = "test4") -> dict:
    from .up2_eval import score_set
    if D.FROZEN is None:
        raise SystemExit("thresholds not frozen")
    T1, T2 = D.FROZEN["T_MED"], D.FROZEN["T_P95"]
    truth = D.load_set(name)
    idx, pr = D.latest_preds(name)
    rep = {"meta": {**D_meta(), "index": idx["index_file"], "scored_utc": idx["meta"].get("utc"),
                    "table_hash": truth.attrs.get("provenance", {}).get("table_hash"), "set": name,
                    "provenance": "[SYNTHETIC] synth_v3 generator + [SIMULATION] FE twin (assumed geometry)"},
           "v2_text": V2_TEXT, "v1_text": D.V1_TEXT, "frozen": D.FROZEN, "models": {}}
    for k in ("v3b", "v3c"):
        m = D.scored_windows(truth, pr[k])
        s = score_set(pr["cls"], pr[k], truth, D.WARM_MIN_WINDOW)
        rep["models"][k] = {
            "role": "deployed (verdict)" if k == "v3b" else "research (physics-informed: consumes RUL_phys -> consistency, not independent)",
            "v2": D.is2_v2(m, T1, T2), "v1": D.is2_v1(m), "sweep": D.sweep(m), "flag_auroc": D.flag_auroc(m),
            "context": {"S7_mape_pct": s["rul"]["S7_mape_pct"], "macro_f1": s["classification"]["macro_f1"],
                        "stage_max": s["stage"]["max_abs_error"],
                        "caught": f"{s['caught']['caught_by_stage3']}/{s['caught']['rtf_bearings']}"}}
    m = D.scored_windows(truth, pr["v3b"])
    rep["floor_perfect_ai"] = D.is2_v2(D.perfect_ai(m), T1, T2)
    rep["controls"] = D.controls(m, T1, T2)
    sel = REPORTS_DIR / Path(D.FROZEN["source"]).name
    if sel.exists():
        j = json.loads(sel.read_text())
        rep["validation"] = {"source": sel.name, "chosen": j["chosen"],
                             "values": {n: {k: {"MED": j["results"][n][k]["v2"]["MED"], "P95": j["results"][n][k]["v2"]["P95"],
                                                "v1_max": j["results"][n][k]["v1"]["max"]} for k in ("v3b", "v3c")}
                                        for n in ("val4", "val2")},
                             "controls": {n: j["results"][n]["controls"] for n in ("val4", "val2")}}
    old = sorted(REPORTS_DIR.glob("up2c_spec_check_*.json"))
    if old:
        o = json.loads(old[-1].read_text())["is2"]
        rep["history_v1"] = {"source": old[-1].name, "set": "synth_v3_test3",
                             **{k: o[k]["residual"]["max"] for k in ("v3b", "v3c", "ablation")}}
    return rep


def D_meta() -> dict:
    from .up2_train import meta
    return meta({"tool": "aiengine.up2d_eval", "command": "python -m aiengine.up2d_eval"})


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f} %"


def write_md(rep: dict, path: Path) -> None:
    b = rep["models"]["v3b"]
    v2 = b["v2"]
    L = ["# IS2 wording v2 check -- [SYNTHETIC] synth_v3_test4 + [SIMULATION] FE twin", "",
         "**Provenance: synthetic generator runs (aiengine/synth_v3) and a simulated FE twin with ASSUMED geometry; not a "
         "measurement of the team rig.**", "",
         f"- generated {rep['meta']['utc']}, commit `{rep['meta']['git_commit']}` (dirty={rep['meta']['git_dirty']}), seed {rep['meta']['seed']}",
         f"- command: `python -m aiengine.up2d_eval`; predictions `{rep['meta']['index']}` (made once, {rep['meta']['scored_utc']})",
         f"- set: {rep['meta']['set']} (100 FRESH runs, hash `{rep['meta']['table_hash']}`), registered in splits.json before any "
         f"IS2-v2 choice; thresholds frozen from validation in `{rep['frozen']['source']}` and committed before this set was scored", "",
         "## Spec text", "", f"**v2 (verdict):** {rep['v2_text']}", "", f"**v1 (kept, superseded):** \"{rep['v1_text']}\"", "",
         "## Verdict", "", "| reading | AI | value | target | verdict |", "|---|---|---|---|---|"]
    for k in ("v3b", "v3c"):
        s = rep["models"][k]["v2"]
        L.append(f"| IS2 v2 (a) median of bearing medians | {k} | {pct(s['MED'])} of L_ref | <= {pct(s['T_MED'])} | "
                 f"{'PASS' if s['MED'] <= s['T_MED'] else 'FAIL'} |")
        L.append(f"| IS2 v2 (b) p95 of bearing p95s | {k} | {pct(s['P95'])} of L_ref | <= {pct(s['T_P95'])} | "
                 f"{'PASS' if s['P95'] <= s['T_P95'] else 'FAIL'} |")
        L.append(f"| **IS2 v2 overall** | {k} ({rep['models'][k]['role'].split(' ')[0]}) | | | **{s['verdict']}** |")
        v1 = rep["models"][k]["v1"]
        L.append(f"| IS2 v1 literal: max per-window rel. to RUL_AI | {k} | {pct(v1['max'])} | <= 5 % | **{v1['verdict']}** |")
    L += ["", f"n = {v2['n_windows']} windows, {v2['n_bearings']} bearings; physics value = rated-life prior on "
          f"{pct(v2['share_physics_is_prior'])} of windows. Window-level e (v3b): median {pct(v2['window_median'])}, p95 "
          f"{pct(v2['window_p95'])}, max {pct(v2['window_max'])}; worst bearing p95 {pct(v2['worst_bearing_p95'])}.", "",
          f"Context, same pass (v3b): S7 {b['context']['S7_mape_pct']:.2f} %, macro-F1 {b['context']['macro_f1']:.4f}, stage max "
          f"{b['context']['stage_max']}, caught {b['context']['caught']}.", "",
          "By stage (v3b): " + ", ".join(f"stage {s}: median {pct(x['median'])}, p95 {pct(x['p95'])}" for s, x in v2["by_stage"].items()),
          "", "By rpm (v3b): " + ", ".join(f"{r} rpm: {x['bearings']} bearings, MED {pct(x['MED'])}, P95 {pct(x['P95'])}" for r, x in v2["by_rpm"].items()),
          "", "## Why the thresholds (validation only)", "",
          "| set | v3b MED | v3b P95 | perfect-AI floor MED | floor P95 | v3c MED | v3c P95 |", "|---|---|---|---|---|---|---|"]
    va = rep.get("validation", {})
    j = json.loads((REPORTS_DIR / va["source"]).read_text()) if va else None
    for n in ("val4", "val2"):
        if not j:
            break
        f = j["results"][n]["floor_perfect_ai"]
        L.append(f"| {n} | {pct(va['values'][n]['v3b']['MED'])} | {pct(va['values'][n]['v3b']['P95'])} | {pct(f['MED'])} | {pct(f['P95'])} | "
                 f"{pct(va['values'][n]['v3c']['MED'])} | {pct(va['values'][n]['v3c']['P95'])} |")
    if va:
        L += ["", f"Rule: {va['chosen']['rule']} -> T_MED = {va['chosen']['T_MED']}, T_P95 = {va['chosen']['T_P95']}. "
              "The perfect-AI floor is the disagreement the TRUE RUL itself has with the physics RUL on the same windows: "
              "on validation a tighter threshold would fail an exact AI, i.e. it would test the physics prior, not the AI.", ""]
    L += ["## Negative controls (deliberately wrong AI RULs; all must FAIL)", "", "| control | MED | P95 | verdict test4 |", "|---|---|---|---|"]
    for c in rep["controls"]:
        L.append(f"| {c['control']} | {pct(c['MED'])} | {pct(c['P95'])} | {c['verdict']} |")
    fl = rep["floor_perfect_ai"]
    L += ["", f"Perfect-AI floor on this set (RUL_AI := RUL_true): MED {pct(fl['MED'])}, P95 {pct(fl['P95'])} -> {fl['verdict']}.",
          "" if fl["verdict"] == "PASS" else
          ("**Disclosure: an exact AI would FAIL clause (a) on this set.** The median clause therefore does not reward accuracy: "
           "it rewards agreement with the twin's (mostly prior) RUL, and its threshold sits inside the set-to-set variation of the "
           "physics floor (validation 15.3-15.5 %, this set " + pct(fl["MED"]) + "). The deployed AI passes because its RUL is closer "
           "to the physics prior than the truth is, not because it is more accurate than truth. Clause (b) and the controls are the "
           "parts with discriminating power."), "",
          f"Residual as a flag of a bad AI estimate (|AI - truth| > 0.20 L_ref), v3b: AUROC {b['flag_auroc'].get('auroc', float('nan')):.3f} "
          f"(share bad {pct(b['flag_auroc'].get('share_bad'))}). Weak: in stages 1-3 the physics RUL carries little information "
          "beyond the rated life, so the residual catches gross bias/inversion (controls) better than ordinary AI error.", "",
          "## Sweep (same windows, v3b; NOT the spec metric)", "",
          "| normaliser | window max | window p95 | window median | bearing-median median | bearing-p95 p95 | worst bearing p95 |",
          "|---|---|---|---|---|---|---|"]
    for r in b["sweep"]:
        L.append(f"| {r['normaliser']} | {pct(r['window_max'])} | {pct(r['window_p95'])} | {pct(r['window_median'])} | "
                 f"{pct(r['bearing_med_median'])} | {pct(r['bearing_p95_p95'])} | {pct(r['bearing_p95_max'])} |")
    if "history_v1" in rep:
        h = rep["history_v1"]
        L += ["", f"v1 literal history ({h['source']}, synth_v3_test3): max v3b {pct(h['v3b'])}, v3c {pct(h['v3c'])}, ablation {pct(h['ablation'])}."]
    L += ["", "## Limitations (state alongside the number)", "",
          "1. **Circularity.** The twin's stiffness trajectory K(t) = K0 (1 - 0.25 damage) is derived from the same synthetic "
          "damage that drives the accelerometer features and the RUL label; agreement here is agreement of two estimators of one "
          "latent variable on generated data, not evidence about real bearings.",
          "2. **The physics value is mostly the prior.** In stages 1-3 K drops by <= ~5 % against ~6 % (1750 rpm) / ~25 % (3600 rpm) "
          "per-window identification noise, so the twin's RUL is the rated-life prior max(L_ref - age, 0.1 L_ref) on "
          f"{pct(v2['share_physics_is_prior'])} of windows. IS2 v2 therefore checks that the AI stays within the physically "
          "plausible envelope of the twin (no gross bias, no inverted trend), not that the twin independently confirms the AI's "
          "accuracy; accuracy is S7 / IS3.",
          "3. **3600 rpm identification gate.** K is weakly observable above the bearing mode; the gate rejects most 3600 rpm "
          "windows, so at 3600 rpm the physics RUL is the prior almost everywhere.",
          "4. **L_ref is synthetic.** L_ref = median TRAIN-run life per speed (72.9 h at 1750, 46.5 h at 3600 rpm); on the rig it "
          "becomes the ISO 281 L10 (or L50) at the duty point, and the thresholds must be re-derived on rig validation data.",
          "5. **v3c is research.** It consumes RUL_physics as an input, so its residual measures consistency, not independent validation.",
          "6. Thresholds were set from the deployed v3b on validation; the median clause separates v3b from an uninformative constant "
          "AI by a small margin (see controls)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    from .up2_train import now_tag
    ap = argparse.ArgumentParser(prog="aiengine.up2d_eval")
    ap.add_argument("--set", default="test4", choices=["test4", "val4", "val2"])
    a = ap.parse_args(argv)
    rep = evaluate(a.set)
    ts = now_tag()
    suffix = "" if a.set == "test4" else f"_{a.set}"
    jp, mp = REPORTS_DIR / f"up2d_is2{suffix}_{ts}.json", REPORTS_DIR / f"up2d_is2{suffix}_{ts}.md"
    jp.write_text(json.dumps(rep, indent=1, default=float))
    write_md(rep, mp)
    print(f"[SYNTHETIC]+[SIMULATION] IS2 v2 on {a.set} (saved predictions re-scored)")
    for k in ("v3b", "v3c"):
        s, v1 = rep["models"][k]["v2"], rep["models"][k]["v1"]
        print(f"{k}: v2 MED {s['MED']:.4f} (<= {s['T_MED']}) P95 {s['P95']:.4f} (<= {s['T_P95']}) -> {s['verdict']} | "
              f"v1 literal max {v1['max']:.3f} (<= 0.05) -> {v1['verdict']}")
    for c in rep["controls"]:
        print("control", c["control"], c["verdict"])
    print("->", jp, "\n->", mp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
