"""ISO 281 basic rating life of the rig bearing S6700/61700 and the simulated-life scale choice."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from ..config import REPORTS_DIR, RIG_BEARING_ORDERS
from . import constants as K

Z, DW, DPW = 11, 1.2, 12.45       # balls, ball diameter, pitch diameter (mm) [measured, COE 2026-11-12]
# fc values recalled from the ISO 281:1990 table (radial ball bearings): ASSUMED, not re-checked against the standard text.
FC_ISO281 = {0.05: 46.7, 0.06: 49.1, 0.07: 51.1, 0.08: 52.8, 0.09: 54.3, 0.10: 55.5, 0.12: 57.5}


def c_iso281_estimate() -> tuple[float, float]:
    """C = fc (i cos a)^0.7 Z^(2/3) Dw^1.8 (Dw <= 25.4 mm). ASSUMED fc; modern catalogues rate higher."""
    x = DW / DPW
    xs = sorted(FC_ISO281)
    fc = float(np.interp(x, xs, [FC_ISO281[k] for k in xs]))
    return fc * Z ** (2 / 3) * DW ** 1.8, fc


def l10_hours(rpm: float, pc: float, a1: float = 1.0) -> float:
    """L10h = a1 * 1e6/(60 n) * (C/P)^3  (ball bearing, p = 3)."""
    return a1 * 1e6 / (60.0 * rpm) * (1.0 / pc) ** 3


def l_ref_hours(rpm: float) -> float:
    return l10_hours(rpm, K.RIG_PC_SIM)


def median_life_hours(rpm: float) -> float:
    return K.LIFE_RHO * l_ref_hours(rpm)


def table() -> list[dict]:
    rows = []
    for pc in (0.05, 0.1, 0.2, 0.3, 0.5):
        for rpm in (1800, 3600):
            l10 = l10_hours(rpm, pc)
            sim = median_life_hours(rpm)
            rows.append({"P/C": pc, "P_N": round(pc * K.RIG_C_N, 1), "rpm": rpm, "L10_h": round(l10, 1),
                         "sim_median_life_h": round(sim, 1), "ratio_sim_life_over_L10": round(sim / l10, 4)})
    return rows


def write_report(path=None) -> str:
    c_est, fc = c_iso281_estimate()
    rows = table()
    sim = {r: median_life_hours(r) for r in (1800, 3600)}
    lines = [
        "# rig_anchor scale check (ISO 281 rating life vs simulated unit life)",
        "",
        f"Provenance: {K.PROVENANCE}. Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `python -m aiengine.rig_anchor.scale`.",
        "",
        "## Inputs",
        f"- Bearing S6700 / 61700, open, 10x15x3 mm, Z={Z} balls, Dw={DW} mm, Dpw={DPW} mm, contact angle 0 (COE, measured 2026-11-12; orders {RIG_BEARING_ORDERS}).",
        f"- Basic dynamic load rating C: **{K.RIG_C_N:.0f} N [CITED: supplier listing W 61700, 110 lbf]**; other listings of the 61700 family give 853-859 N "
        "(FAG F61700-2RSR 857 N; a 4 mm-wide sealed version 859 N). The exact brand/suffix of the rig bearing is unknown -> **ASSUMED within 489-857 N**. "
        f"My own ISO 281 formula estimate C = fc (i cos a)^0.7 Z^(2/3) Dw^1.8 with an older-table fc={fc:.1f} (recalled, ASSUMED) gives {c_est:.0f} N [CALCULATED]; modern catalogue ratings are higher.",
        "- L10 depends on C only through P/C: L10h = 1e6/(60 n) (C/P)^3 (ISO 281 basic rating life, a1=1, 90 % reliability). The rig's real equivalent "
        "radial load is NOT known (not measured) -> P/C is a stated assumption.",
        "",
        "## L10 table (hours)",
        "| P/C | P (N, C=489) | rpm | L10 (h) | simulated median unit life (h) | sim life / L10 |", "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['P/C']} | {r['P_N']} | {r['rpm']} | {r['L10_h']} | {r['sim_median_life_h']} | {r['ratio_sim_life_over_L10']} |")
    lines += [
        "",
        "## Choice",
        f"- Simulation-equivalent duty **P/C = {K.RIG_PC_SIM}** (P = {K.RIG_PC_SIM * K.RIG_C_N:.0f} N): L_ref = L10 = {l_ref_hours(1800):.1f} h at 1800 rpm and {l_ref_hours(3600):.1f} h at 3600 rpm.",
        f"- Simulated onset-to-failure life = LIFE_RHO x L_ref x lognormal(0, {K.LIFE_SIGMA}) with LIFE_RHO = {K.LIFE_RHO}: median {sim[1800]:.1f} h (1800) and {sim[3600]:.1f} h (3600) "
        f"= {sim[1800] / K.CADENCE_H:.0f} / {sim[3600] / K.CADENCE_H:.0f} windows at the {K.CADENCE_H * 60:.0f}-min snapshot cadence. **Ratio life/L_ref = {K.LIFE_RHO}.**",
        "- Why not the light real duty: at P/C = 0.1 L10 is "
        f"{l10_hours(1800, 0.1):.0f} h (1800 rpm); a simulated unit of tens of hours then has life/L_ref ~ {median_life_hours(1800) / l10_hours(1800, 0.1):.3f}, "
        "so an IS2 error normalised by L_ref (e = |RUL_phys - RUL_AI| / L_ref) would be < 1 % of L_ref for ANY prediction between 0 and the unit life = vacuous pass. "
        f"With life/L_ref = {K.LIFE_RHO}, an 18 % L_ref band is about {0.18 / K.LIFE_RHO * 100:.0f} % of the unit's median life, so a constant or x1.5/x0.6 wrong RUL can fail it (to be verified by the IS2 controls on val).",
        "- Honesty: P/C = 0.5 is the heavy end of the ISO 281 range, a stated simulation-equivalent (accelerated) duty, not a rig measurement. "
        "LIFE_RHO = 0.7 is an assumption: a seeded-defect bearing's onset-to-failure time is taken as at most its rated life. The absolute hours therefore carry no physical claim; "
        "only the life/L_ref ratio and the speed law (life in revolutions fixed, hours ~ 1/rpm) are used.",
    ]
    text = "\n".join(lines) + "\n"
    out = path or REPORTS_DIR / "rig_anchor_scale_check.md"
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "C_N": K.RIG_C_N, "PC": K.RIG_PC_SIM, "rho": K.LIFE_RHO, "L_ref_h": {str(r): l_ref_hours(r) for r in (1800, 3600)},
        "sim_median_life_h": {str(r): sim[r] for r in sim}, "ratio": K.LIFE_RHO, "table": rows,
        "C_iso281_estimate_N": c_est, "fc_assumed": fc}, indent=1))
    return str(out)


if __name__ == "__main__":
    print(write_report())
