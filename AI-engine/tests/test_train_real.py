"""train_real / data_coverage / recorder-CSV reader. Data here is [SIMULATION rehearsal] (synth_coe_node resampled into recorder CSVs)."""
import csv
import json

import numpy as np
import pytest

from aiengine import data_coverage as DC
from aiengine import node_eval
from aiengine import real_data as R
from aiengine import rehearsal_fake as RF
from aiengine import train_real as TR

OLD_COLS = ["run_id", "label", "note", "received_at", "rpm", "window_id", "sequence", "device_id", "valid_channels"] + R.FEATURE_COLUMNS


def _write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)


def _row(rid, label, i, extra=()):
    return [rid, label, "", "t", 1800, i, i, "N", 15] + [0.01 * (j + 1) for j in range(32)] + list(extra)


def test_reader_old_and_new_columns(tmp_path):
    _write(tmp_path / "old.csv", OLD_COLS, [_row("run_a", "healthy", i) for i in range(3)] + [_row("run_b", "ball", i) for i in range(2)])
    _write(tmp_path / "new.csv", OLD_COLS + ["severity", "bearing_id", "position", "is_baseline"],
           [_row("run_c", "inner_race", i, (2, "B-03", "b2", 0)) for i in range(4)] + [_row("run_d", "healthy", i, (0, "B-00", "both", 1)) for i in range(2)])
    df = R.load_runs(tmp_path)
    rt = R.run_table(df).set_index("run_id")
    assert rt.loc["run_a", "severity"] == 0 and rt.loc["run_a", "position"] == "both" and not rt.loc["run_a", "is_baseline"]
    assert np.isnan(rt.loc["run_b", "severity"])                       # old faulty run: level unknown
    assert (rt.loc["run_c", "severity"], rt.loc["run_c", "bearing_id"], rt.loc["run_c", "position"]) == (2, "B-03", "b2")
    assert bool(rt.loc["run_d", "is_baseline"]) and rt.loc["run_c", "n_windows"] == 4
    runs = node_eval.read_runs(tmp_path / "new.csv")                    # node_eval reads by header name too
    assert runs["run_c"]["position"] == "b2" and runs["run_d"]["is_baseline"] and len(runs["run_c"]["messages"]) == 4
    assert len(node_eval.read_runs(tmp_path / "old.csv")["run_a"]["messages"]) == 3


def test_coverage_missing(tmp_path):
    n = R.settle_windows(1800) + int(2 * 60 / R.window_seconds(1800))
    _write(tmp_path / "x.csv", OLD_COLS + ["severity", "bearing_id", "position", "is_baseline"],
           [_row("run_h1", "healthy", i, (0, "B1", "both", 0)) for i in range(n)])
    df = R.load_runs(tmp_path)
    t = DC.usable_table(df)
    assert abs(t["usable_min"].iloc[0] - 2.0) < 0.02
    miss = {(m["cond"], m["severity"], m["rpm"]): m["missing_runs"] for m in DC.missing(t, "minimum")}
    assert miss[("healthy", 0, 1800)] == 2 and miss[("inner_race", 2, 3600)] == 2 and miss[("baseline", 0, 1800)] == 1
    assert "Still missing" in DC.report(df)


def test_single_class_fails_gracefully(tmp_path):
    n = R.settle_windows(1800) + 60
    _write(tmp_path / "h.csv", OLD_COLS, [_row("run_h1", "healthy", i) for i in range(n)])
    with pytest.raises(SystemExit) as e:
        TR.main(["--data", str(tmp_path), "--out", "t_single", "--models-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r"),
                 "--splits-file", str(tmp_path / "s.json")])
    assert e.value.code == 2


def test_refuses_to_overwrite_existing_contract(tmp_path):
    with pytest.raises(SystemExit):
        TR.main(["--data", str(tmp_path), "--out", "v3e_node"])


def test_choose_holdout_frozen_and_by_mount():
    import pandas as pd
    rows = [{"run_id": f"r{i}", "label": "ball", "severity": 2, "bearing_id": f"B{i // 2}", "is_baseline": False} for i in range(8)]
    rt = pd.DataFrame(rows)
    a = TR.choose_holdout(rt, {})
    assert "heldout" in a.values()
    for i in range(0, 8, 2):                                            # runs of one mount are never split across sets
        assert a[f"r{i}"] == a[f"r{i + 1}"]
    assert TR.choose_holdout(rt, a) == a                                # frozen
    assert "heldout" not in TR.choose_holdout(rt.iloc[:4], {}).values()   # < 3 mounts: nothing held out


def test_end_to_end_small(tmp_path):
    data = tmp_path / "d"
    RF.make(data, "minimum", "train", 3, "T", usable_min=0.6)
    out = tmp_path / "m"
    rc = TR.main(["--data", str(data), "--out", "t_e2e", "--tag", "[SIMULATION rehearsal]", "--models-dir", str(out),
                  "--reports-dir", str(tmp_path / "r"), "--splits-file", str(tmp_path / "s.json"), "--threads", "4"])
    assert rc == 0
    assert (out / "model_contract_t_e2e.json").exists() and (out / "t_e2e_cls.onnx").exists() and (out / "baseline_t_e2e.json").exists()
    rep = json.loads(next((tmp_path / "r").glob("train_real_*.json")).read_text())
    assert rep["onnx_parity"]["argmax_mismatch"] == 0 and rep["onnx_parity"]["prob_max_abs_diff"] < 1e-4
    sp = json.loads((tmp_path / "s.json").read_text())["sections"][TR.SPLIT_SECTION]
    assert set(sp["units"]).issuperset(rep["split"]["heldout_runs"])
    from aiengine.node_adapter import NodeEngine
    eng = NodeEngine(contract_file="model_contract_t_e2e.json", model_dir=out)
    eng.load_baseline(out / "baseline_t_e2e.json")
    row = RF.pool_from_sim("test")
    r = row[(row["cls1"] == "healthy") & np.isclose(row["rpm"], 1800, atol=90)].iloc[0]
    msg = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": 1800, "window_id": 1,
           "features": [float(r[c]) for c in R.FEATURE_COLUMNS]}
    o = eng.predict(msg)
    assert len(o["bearings"]) == 2
