"""v3e_node: baseline-relative feature definitions + golden vectors + node_eval [SIMULATION]."""
import json

import numpy as np

from aiengine import features as feat
from aiengine.config import MODELS_DIR
from aiengine.engine import HybridEngine

ABSOLUTE = {"kurtosis_max", "crest_max", "raw_ftf_share", "raw_bsf_share", "raw_bpfo_share", "raw_bpfi_share"}


def _raw(rng, n=30):
    one = np.array([0.04, 4.0, 6.0, 0.03, 0.001, 0.002, 0.003, 0.004])
    return np.tile(np.concatenate([one, one * 0.8]), (n, 1)) * rng.normal(1, 0.02, (n, 16))


def test_v3e_columns_are_relative_and_present():
    assert not ABSOLUTE & set(feat.V3E_COLUMNS)
    assert set(feat.V3E_COLUMNS) <= set(feat.ENGINEERED_EXT)
    assert len(feat.V3E_COLUMNS) == 16 and "rpm_norm" in feat.V3E_COLUMNS


def test_default_engineering_unchanged_and_extension_appends_one_column():
    rng = np.random.default_rng(1)
    raw = _raw(rng)
    base = _raw(rng).mean(0)
    a = feat.engineer_arrays(raw, np.full(30, 1800.0), base)
    b = feat.engineer_arrays(raw, np.full(30, 1800.0), base, extended=True)
    assert a.shape[1] == 21 and b.shape[1] == 22 and np.array_equal(a, b[:, :21])


def test_crest_rise_definition_and_stationary_impulsive_not_flagged():
    base = np.array([0.04, 4.0, 6.0, 0.03, 0.001, 0.001, 0.001, 0.001] * 2)
    raw = base.copy()[None]
    raw[0, 2], raw[0, 10] = 9.0, 7.0                      # crest X 9, Y 7 -> max 9; baseline max 6
    e = feat.engineer_arrays(raw, np.array([1800.0]), base, extended=True)[0]
    assert abs(e[feat.ENGINEERED_EXT.index("crest_rise")] - 3.0) < 1e-9
    raw[0, 2], raw[0, 10] = 5.0, 5.0                                       # below baseline -> clipped at 0
    assert feat.engineer_arrays(raw, np.array([1800.0]), base, extended=True)[0, -1] == 0.0
    imp = base.copy()
    imp[[1, 9]], imp[[2, 10]] = 7.0, 8.0                  # impulsive in BOTH baseline and window
    e = feat.engineer_arrays(imp[None], np.array([1800.0]), imp, extended=True)[0]
    cols = {c: e[i] for i, c in enumerate(feat.ENGINEERED_EXT)}
    assert cols["kurtosis_rise"] == 0 and cols["crest_rise"] == 0 and abs(cols["log_env_ratio"]) < 1e-6


def test_contract_v3e_golden_vectors_reproduce():
    con = json.loads((MODELS_DIR / "model_contract_v3e_node.json").read_text())
    assert con["models"]["tft"]["columns"] == feat.V3E_COLUMNS
    assert con["feature_engineering"]["engineered_order"] == feat.ENGINEERED_EXT
    g = json.loads((MODELS_DIR / "golden_vectors_v3e_node.json").read_text())
    eng = HybridEngine(MODELS_DIR, contract_file="model_contract_v3e_node.json")
    outs = []
    for i in g["rig_stream"]["inputs"]:
        o = eng.process(i["rpm"], np.array(i["features32"]), i["t20_ms"])
        outs.append(o["bearings"])
    exp = g["rig_stream"]["outputs"]
    assert len(outs) == len(exp)
    n_ok = 0
    for ob, eb in zip(outs, exp):
        for x, y in zip(ob, eb):
            assert x["status"] == y["status"] and x.get("fault_class") == y.get("fault_class")
            if x["status"] == "ok":
                n_ok += 1
                for c, p in y["class_probs"].items():
                    assert abs(x["class_probs"][c] - p) < 1e-4
    assert n_ok > 20


def test_node_eval_reads_csv_and_jsonl(tmp_path):
    from aiengine import node_eval as ne
    msg = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": 1800, "window_id": 0,
           "features": [0.04, 4.0, 6.0, 0.03, 0.001, 0.001, 0.001, 0.001] * 4}
    j = tmp_path / "a.jsonl"
    j.write_text("\n".join(json.dumps({**msg, "window_id": i}) for i in range(3)) + "\n")
    assert len(ne.read_runs(j)["a"]["messages"]) == 3
    c = tmp_path / "b.csv"
    head = ["run_id", "label", "note", "received_at", "rpm", "window_id", "sequence", "device_id", "valid_channels"] + ne.FEATURE_COLUMNS
    rows = [",".join(head)] + [",".join(["r1" if i < 2 else "r2", "healthy", "", "t", "1800", str(i), str(i), "d", "15"] + ["0.04"] * 32)
                              for i in range(3)]
    c.write_text("\n".join(rows) + "\n")
    runs = ne.read_runs(c)
    assert set(runs) == {"r1", "r2"} and len(runs["r1"]["messages"]) == 2 and runs["r1"]["label"] == "healthy"
