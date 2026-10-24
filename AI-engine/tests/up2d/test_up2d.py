"""IS2 wording v2 tests: metric definition (pure functions), frozen thresholds, split registration, single-pass guard,
and that the saved test4 report re-scores identically.

Run: AI-engine/.venv/Scripts/python.exe -m pytest AI-engine/tests/up2d -q
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aiengine import splits
from aiengine import up2d_is2 as D
from aiengine.config import REPORTS_DIR


def _frame(n_bearings=10, n=40, rpm=1750.0, phys_off=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    L = D.l_ref([rpm])[0]
    for b in range(n_bearings):
        t = np.arange(n) + 24
        rul = np.linspace(0.8 * L, 0.4 * L, n)
        rows.append(pd.DataFrame({"unit_id": f"u{b}", "window_index": t, "run_to_failure": True, "true_stage": 2,
                                  "rul_hours": rul, "rpm_mode": rpm, "t_hours": t / 6.0,
                                  "phys_rul_h": rul + phys_off * L, "phys_conf": 0.0}))
    truth = pd.concat(rows, ignore_index=True)
    rp = truth[["unit_id", "window_index"]].copy()
    rp["rul_pred_h"] = truth["rul_hours"] * (1 + rng.normal(0, 0.01, len(truth)))
    return truth, rp


def test_lref_normalisation_and_aggregation():
    truth, rp = _frame(phys_off=0.1)
    rp["rul_pred_h"] = truth["rul_hours"]
    m = D.scored_windows(truth, rp)
    assert np.allclose(m["e_lref"], 0.1)                    # |phys - AI| = 0.1 L_ref everywhere
    r = D.is2_v2(m, 0.18, 0.63)
    assert r["MED"] == pytest.approx(0.1) and r["P95"] == pytest.approx(0.1) and r["verdict"] == "PASS"
    assert D.is2_v1(m)["max"] == pytest.approx((0.1 * D.l_ref([1750])[0] / m["rul_pred_h"]).max())
    assert D.is2_v2(m, 0.05, 0.63)["verdict"] == "FAIL"     # both clauses must hold


def test_v1_blows_up_late_but_v2_does_not():
    truth, rp = _frame(phys_off=0.05)
    truth.loc[truth.index[-1], "rul_hours"] = 0.2           # tiny late RUL
    truth.loc[truth.index[-1], "phys_rul_h"] = 0.2 + 0.05 * D.l_ref([1750])[0]
    rp["rul_pred_h"] = truth["rul_hours"]
    m = D.scored_windows(truth, rp)
    assert D.is2_v1(m)["max"] > 10                          # ratio to a tiny RUL_AI explodes
    assert D.is2_v2(m, 0.18, 0.63)["window_max"] == pytest.approx(0.05)


def test_window_filter_stage_warmup_rul_positive():
    truth, rp = _frame()
    truth.loc[truth.index[:5], "true_stage"] = 4
    truth.loc[truth.index[5:7], "window_index"] = 3         # before warm-up (dropped)
    rp.loc[rp.index[5:7], "window_index"] = 3
    truth.loc[truth.index[7], "rul_hours"] = 0.0
    m = D.scored_windows(truth, rp)
    assert len(m) == len(truth) - 8


def test_perfect_ai_floor_and_controls():
    truth, rp = _frame(phys_off=0.0)
    m = D.scored_windows(truth, rp)
    assert D.is2_v2(D.perfect_ai(m), 0.18, 0.63)["MED"] == pytest.approx(0.0)
    c = {x["control"]: x["verdict"] for x in D.controls(m, 0.18, 0.63)}
    assert c["AI + 0.3 L_ref"] == "FAIL"


def test_choose_rule():
    ch = D.choose([{"MED": 0.1564, "P95": 0.536}, {"MED": 0.153, "P95": 0.477}], [{"MED": 0.155, "P95": 0.571}])
    assert ch["T_MED"] == pytest.approx(0.18) and ch["T_P95"] == pytest.approx(0.63)


def test_frozen_thresholds_match_selection_file():
    assert D.FROZEN is not None
    j = json.loads((REPORTS_DIR / D.FROZEN["source"].split("/")[-1]).read_text())
    assert j["chosen"]["T_MED"] == D.FROZEN["T_MED"] and j["chosen"]["T_P95"] == D.FROZEN["T_P95"]
    assert j["chosen"]["controls_all_fail_on_validation"] is True
    assert "VALIDATION ONLY" in j["meta"]["data"]


def test_test4_val4_registered_and_disjoint():
    s = splits.load()["sections"]
    assert set(s["synth_v3_test4"]["units"].values()) == {"test"}
    assert set(s["synth_v3_val4"]["units"].values()) == {"val"}
    others = set().union(*(set(v["units"]) for k, v in s.items() if k not in ("synth_v3_test4",)))
    assert not (set(s["synth_v3_test4"]["units"]) & others)


def test_test4_single_pass_guard():
    if not sorted(D.PRED_DIR.glob(f"{D.FAMILY}_test4_*.index.json")):
        pytest.skip("test4 not scored yet")
    with pytest.raises(SystemExit, match="already scored"):
        D.cmd_predict(SimpleNamespace(set="test4", force=False))


def test_saved_report_rescores_identically():
    reps = sorted(REPORTS_DIR.glob("up2d_is2_2*.json"))
    if not reps:
        pytest.skip("no up2d_is2 report yet")
    old = json.loads(reps[-1].read_text())
    from aiengine.up2d_eval import evaluate
    new = evaluate("test4")
    for k in ("v3b", "v3c"):
        assert new["models"][k]["v2"]["MED"] == pytest.approx(old["models"][k]["v2"]["MED"], abs=1e-9)
        assert new["models"][k]["v2"]["P95"] == pytest.approx(old["models"][k]["v2"]["P95"], abs=1e-9)
        assert new["models"][k]["v2"]["verdict"] == old["models"][k]["v2"]["verdict"]
        assert new["models"][k]["v1"]["max"] == pytest.approx(old["models"][k]["v1"]["max"], abs=1e-9)
