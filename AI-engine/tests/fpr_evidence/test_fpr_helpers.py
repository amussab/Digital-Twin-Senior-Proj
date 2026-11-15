"""Tests for notebooks/fpr_helpers.py: statuses are computed from numbers, never hard-coded."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import fpr_helpers as F  # noqa: E402


def mk(s7=10.0, f1=0.9, stage=1, caught=48, rtf=48):
    return {"contract": "x", "s7": s7, "s7_base": 60.0, "n_units": 48, "f1": f1, "stage_max": stage, "caught": caught, "rtf": rtf,
            "pass_s7": s7 <= 15, "pass_f1": f1 >= 0.85, "pass_stage": stage <= 1, "pass_caught": caught == rtf}


def test_s7_met_and_not_met():
    assert F.status_s7(mk(10))[0].startswith("MET")
    st, d = F.status_s7(mk(23.4))
    assert st == "NOT MET" and "conflicts with the team statement" in d["note"]
    assert F.status_s7(None)[0].startswith("PENDING")


def test_is3_all_gates_needed():
    assert F.status_is3(mk())[0].startswith("MET")
    assert not F.status_is3(mk(stage=2))[0].startswith("MET")
    assert not F.status_is3(mk(caught=46))[0].startswith("MET")
    assert not F.status_is3(mk(f1=0.838))[0].startswith("MET")


def test_is2_pending_and_validity():
    assert F.status_is2("v5_node", {"test": None, "frozen": None})[0].startswith("PENDING")
    fz = {"valid_for_test": False, "T_MED": .5, "T_P95": 1, "val": {"v2": {"MED": .3, "P95": .7}}}
    assert F.status_is2("v5_node", {"test": None, "frozen": fz})[0].startswith("PENDING")
    t = {"T_MED": .5, "T_P95": 1.0, "val": {"v2": {"MED": .3, "P95": .7, "n_bearings": 5}, "controls_all_fail": False}}
    assert F.status_is2("v6", {"test": t, "frozen": {"valid_for_test": True}})[0] == "PARTIAL"
    t["val"]["controls_all_fail"] = True
    assert F.status_is2("v6", {"test": t, "frozen": {"valid_for_test": True}})[0].startswith("MET")
    t["val"]["v2"]["MED"] = .9
    assert F.status_is2("v6", {"test": t, "frozen": {"valid_for_test": True}})[0] == "NOT MET"


def test_master_and_counts_from_registry():
    F.REG.clear()
    assert F.master_frame()["Status"].str.startswith("PENDING").all()
    F.record("C1", status=f"MET [{F.OWNER_STATEMENT}]", prov=f"[{F.OWNER_STATEMENT}]")
    F.record("S8", status="MET [MEASURED host]", prov="[MEASURED host]")
    F.record("S7", status="NOT MET", prov=F.SIM)
    c = F.counts(F.master_frame())
    assert (c["MET"], c["owner_met"], c["computed_met"], c["NOT MET"], c["total"], c["bar"]) == (2, 1, 1, 1, 17, 13)
    assert not c["gt75"]
    F.REG.clear()


def test_norm_recomputes_gates():
    d = {"test": {"S7": {"S7_mape_pct": 23.4, "n_bearings": 48}, "IS3_classification": {"macro_f1": .838}, "IS3_stage": {"max_abs_error": 2},
                  "IS3_caught": {"caught_by_stage3": 46, "rtf_bearings": 48}}, "meta": {}}
    n = F.norm(d)
    assert not (n["pass_s7"] or n["pass_f1"] or n["pass_stage"] or n["pass_caught"])


def test_final_reports_ignore_frozen():
    for c, (p, d) in F.final_reports().items():
        assert "frozen" not in p.name and "test" in d
