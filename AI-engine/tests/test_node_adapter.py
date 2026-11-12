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
    e = NodeEngine()
    rng = np.random.default_rng(0)
    outs = [e.predict(msg(window_id=i, features=list(np.array(msg()["features"]) * rng.normal(1, 0.03, 32))))
            for i in range(40)]
    assert outs[0]["bearings"][0]["status"] == "warming_up"
    last = outs[-1]["bearings"]
    assert all(b["status"] == "ok" and set(b["class_probs"]) == {"ball", "cage", "healthy", "inner_race", "outer_race"}
               for b in last)
    assert "onset_detected" in last[0] and outs[-1]["latency_ms"] > 0
