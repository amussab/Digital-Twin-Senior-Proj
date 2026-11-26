"""node_adapter: schema validation + the engine path on a simulated node stream [SIMULATION]."""
import numpy as np
import pytest

from aiengine.node_adapter import NodeEngine, node_json_to_raw


def msg(**kw):
    m = {"schema_version": 1, "acceleration_unit": "g", "valid_channels": 15, "rpm": 1800, "window_id": 1,
         "features": [0.04, 4.0, 6.0, 0.03, 0.001, 0.001, 0.001, 0.001] * 4}
    m.update(kw)
    return m


def test_valid_message_roundtrip():
    x = node_json_to_raw(msg())
    assert x.dtype == np.float32 and x.shape == (32,)


@pytest.mark.parametrize("bad", [dict(schema_version=2), dict(acceleration_unit="m/s2"), dict(valid_channels=7),
                                 dict(features=[0.0] * 31), dict(features=[float("nan")] * 32)])
def test_rejects_bad_messages(bad):
    with pytest.raises(ValueError):
        node_json_to_raw(msg(**bad))


def test_node_engine_streams_and_warms_up():
    e = NodeEngine(policy=None)      # legacy 24-window baseline
    rng = np.random.default_rng(0)
    outs = [e.predict(msg(window_id=i, features=list(np.array(msg()["features"]) * rng.normal(1, 0.03, 32))))
            for i in range(40)]
    assert outs[0]["bearings"][0]["status"] == "warming_up"
    last = outs[-1]["bearings"]
    assert all(b["status"] == "ok" and set(b["class_probs"]) == {"ball", "cage", "healthy", "inner_race", "outer_race"}
               for b in last)
    assert "onset_detected" in last[0] and outs[-1]["latency_ms"] > 0


# ---- commissioning baseline policy (settle / baseline / per-rpm / save-load) ---------------------------------
def _stream(n, rpm=1800, scale=1.0, start=0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([0.04, 4.0, 6.0, 0.03, 0.001, 0.001, 0.001, 0.001] * 4) * scale
    return [msg(window_id=start + i, rpm=rpm, features=list(base * rng.normal(1, 0.03, 32))) for i in range(n)]


POL = {"settle_seconds": 120, "baseline_seconds": 60}


def test_policy_windows_from_seconds():
    from aiengine.node_adapter import policy_windows
    assert policy_windows(1800, POL) == (180, 90)
    assert policy_windows(3600, POL) == (360, 180)


def test_settle_skipped_and_warming_up_class():
    e = NodeEngine(policy=POL, settle_windows=5, baseline_windows=30)
    outs = [e.predict(m) for m in _stream(5 + 30 + 40)]
    assert all(b["fault_class"] == "warming_up" and b["status"] == "warming_up" for o in outs[:34] for b in o["bearings"])
    assert outs[-1]["bearings"][0]["status"] == "ok"
    # baseline = median of windows 5..34 only (settle windows excluded)
    stream = _stream(5 + 30)
    exp = np.median(np.array([m["features"] for m in stream[5:35]]), axis=0)
    got = np.array(e.saved[1800]["b1"] + e.saved[1800]["b2"])
    assert np.allclose(got, exp, rtol=1e-3, atol=1e-4)


def test_legacy_contract_without_policy_unchanged():
    e = NodeEngine(policy=None)
    assert e.phase == "legacy"
    outs = [e.predict(m) for m in _stream(40)]
    assert outs[-1]["bearings"][0]["status"] == "ok"


def test_per_rpm_baseline_switch():
    e = NodeEngine(policy=POL, settle_windows=2, baseline_windows=30)
    for m in _stream(2 + 30 + 10, rpm=1800):
        e.predict(m)
    o = e.predict(_stream(1, rpm=3600, scale=2.0)[0])      # rpm change -> warm-up at the new speed, buffers reset
    assert o["bearings"][0]["fault_class"] == "warming_up"
    for m in _stream(2 + 30 + 10, rpm=3600, scale=2.0, start=100):
        o = e.predict(m)
    assert set(e.saved) == {1800, 3600} and o["bearings"][0]["status"] == "ok"
    assert e.saved[3600]["b1"][0] > 1.5 * e.saved[1800]["b1"][0]
    o = e.predict(_stream(1, rpm=1800)[0])                 # back to 1800: stored baseline reused, buffer refills
    assert e.key == 1800 and o["bearings"][0]["status"] == "warming_up" and e.engine.b[1].baseline is not None


def test_save_load_roundtrip(tmp_path):
    e = NodeEngine(policy=POL, settle_windows=2, baseline_windows=30)
    for m in _stream(40):
        e.predict(m)
    p = tmp_path / "bl.json"
    e.save_baseline(p)
    f = NodeEngine()
    f.load_baseline(p)
    assert f.saved == e.saved
    o = f.predict(_stream(1)[0])
    assert o["bearings"][0]["status"] == "warming_up" and f.key == 1800        # no settle/baseline needed, only buffer
    for m in _stream(30, start=1):
        o = f.predict(m)
    assert o["bearings"][0]["status"] == "ok"
    with pytest.raises(ValueError):
        NodeEngine(contract_file="model_contract_v3d_node.json").load_baseline(p)
