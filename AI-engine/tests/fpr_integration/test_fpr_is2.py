"""fpr_is2: metric reuse, freeze/test discipline. Pure frames; the data-dependent test skips when the git-ignored parquet is absent."""
import numpy as np
import pandas as pd
import pytest

from aiengine import fpr_is2 as F
from aiengine.config import ROOT


def frame(n_units=6, n=60, bias=1.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        L = 74.07
        for w in range(n):
            truth = max(L * 0.7 * (1 - w / n), 0.1)
            phys = truth * (1 + noise * rng.normal())
            rows.append({"unit_id": f"u{u}:B1", "window_index": w + 24, "rpm_mode": 1800.0, "true_stage": 1 + w * 3 // n, "rul_hours": truth,
                         "phys_rul_h": phys, "rul_pred_h": phys * bias, "L_ref": L, "phys_conf": 1.0, "phys_source": "trend"})
    m = pd.DataFrame(rows)
    m["diff_h"] = (m["phys_rul_h"] - m["rul_pred_h"]).abs()
    m["e_lref"] = m["diff_h"] / m["L_ref"]
    m["e_v1"] = m["diff_h"] / m["rul_pred_h"]
    return m


def test_summarize_agreeing_pair_passes_and_controls_fail():
    m = frame()
    s = F.summarize(m, 0.18, 0.63)
    assert s["v2"]["verdict"] == "PASS"
    assert s["twin"]["share_trend"] == 1.0 and "v1_literal" in s and len(s["controls"]) == 6


def test_biased_ai_fails():
    s = F.summarize(frame(bias=2.5), 0.18, 0.63)
    assert s["v2"]["verdict"] == "FAIL"


def test_resolve_split_refuses_unregistered():
    with pytest.raises(SystemExit):
        F.resolve_split("no_such_section:test")


def test_test_mode_refuses_without_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "REPORTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no frozen thresholds"):
        F.latest_frozen("v9_nothing")


def test_test_mode_refuses_second_run(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "REPORTS_DIR", tmp_path)
    (tmp_path / "fpr_is2_test_done_v5_node_rig_anchor_test.json").write_text("{}")
    with pytest.raises(SystemExit, match="already scored once"):
        F.main(["test", "--split", "test", "--contract", "v5_node"])


@pytest.mark.skipif(not (ROOT / "data" / "rig_anchor" / "rig_anchor.parquet").exists(), reason="rig_anchor parquet is git-ignored")
def test_engine_predictions_one_unit():
    r = F.resolve_split("val")
    df = F.load_units(r)
    p = F.predict_units(df, "v5_node", units=1)
    m = F.scored(df, p)
    assert len(m) > 50 and m["e_lref"].notna().all() and set(m["true_stage"]) <= {1, 2, 3}
