"""rig_anchor generator tests. [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed]"""
import json

import numpy as np
import pandas as pd
import pytest

from aiengine import real_data as R
from aiengine import splits as S
from aiengine.datasets.common import SCHEMA
from aiengine.rig_anchor import constants as K
from aiengine.rig_anchor import generator as G
from aiengine.rig_anchor import pools as P
from aiengine.rig_anchor import registry, scale
from aiengine.synth_v3.generator import stage_from_u


@pytest.fixture(scope="module")
def pools():
    return P.Pools()


@pytest.fixture(scope="module")
def sub(pools):
    plan = G.plan_units()
    pick = [p for p in plan if p["split"] == "train"][:10] + [p for p in plan if p["split"] == "val"][:5] + [p for p in plan if p["split"] == "test"][:15]
    return pick, G.build(pools, pick)


def test_determinism(pools, sub):
    pick, df = sub
    again = G.generate_unit(pick[3], pools)
    ref = df[df["rig_unit"] == pick[3]["unit_id"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(again.reset_index(drop=True), ref)


def test_schema(sub):
    _, df = sub
    assert list(df.columns) == G.OUT_COLUMNS
    assert set(SCHEMA) <= set(df.columns)
    assert set(R.FEATURE_COLUMNS) <= set(df.columns)
    assert (df["provenance"] == K.PROVENANCE).all()
    assert (df["dataset"] == "rig_anchor").all()
    assert set(df["fault_class"]) <= set(K.CLASSES)


def test_no_nan_and_positive(sub):
    _, df = sub
    X = df[R.FEATURE_COLUMNS + ["B1X_rms"]].to_numpy(float)
    assert np.isfinite(X).all() and (X > 0).all()
    assert df["t_hours"].notna().all()


def test_b2x_policy_applied(sub):
    _, df = sub
    assert np.allclose(df[[f"B2X_{f}" for f in R.FEATS]].to_numpy(), df[[f"B2Y_{f}" for f in R.FEATS]].to_numpy())
    b2 = df[df["unit_id"].str.endswith(":B2")]
    assert np.allclose(b2["x_bp_rms"].to_numpy(), b2["B2X_rms"].to_numpy(), rtol=1e-5)


def test_labels_monotone_and_stage_rule(sub):
    _, df = sub
    f = df[df["run_to_failure"]]
    assert len(f)
    for uid, g in f.groupby("unit_id"):
        g = g.sort_values("window_index")
        assert (np.diff(g["rul_hours"]) < 0).all() and abs(g["rul_hours"].iloc[-1]) < 1e-9
        assert (np.diff(g["true_stage"]) >= 0).all() and (np.diff(g["damage"]) >= -1e-12).all() and (np.diff(g["u_true"]) >= 0).all()
        assert (g["true_stage"].to_numpy() == stage_from_u(g["u_true"].to_numpy())).all()
        assert g["true_stage"].iloc[0] == 1 and g["true_stage"].iloc[-1] == 6
        # anchor: g reaches 1 inside the stage region of the severity (sev3 -> stage 6, sev2 -> stage 4)
        ga = g[g["g"] >= 0.999].iloc[0]
        sev = int(g["severity"].iloc[0][3])
        assert ga["true_stage"] == {3: 6, 2: 4}[sev]


def test_healthy_units_have_no_failure(sub):
    _, df = sub
    h = df[df["fault_class"].eq("healthy") & df["rig_unit"].isin(df[df["faulted_bearing"] == 0]["rig_unit"])]
    assert not h["run_to_failure"].any() and h["rul_hours"].isna().all() and (h["true_stage"] == 1).all()


def test_ball_only_1800_and_speeds(sub):
    _, df = sub
    assert set(df[df["fault_class"] == "ball"]["rpm"]) <= {1800.0}
    assert set(df["rpm"]) <= {1800.0, 3600.0}


def test_faulted_bearing_is_housing1(sub):
    _, df = sub
    f = df[df["run_to_failure"]]
    assert f["unit_id"].str.endswith(":B1").all() and (f["faulted_bearing"] == 1).all()


def test_baseline_runin_enough(sub):
    _, df = sub
    for uid, g in df[df["run_to_failure"]].groupby("unit_id"):
        assert (g["u_true"] < 0).sum() >= K.BASELINE_WINDOWS


def test_split_registered_and_matches_plan():
    assert registry.verify()
    sec = S.load()["sections"][K.SPLIT_SECTION]
    assert sec["counts"] == K.UNITS_PER_SPLIT
    units = sec["units"]
    assert len(units) == len(set(units)) == sum(K.UNITS_PER_SPLIT.values())
    for s in K.SPLITS:
        comp = sec["composition"][s]
        assert {k.split("@")[0] for k in comp} == set(K.CLASSES)


def test_unit_split_matches_registry(sub):
    _, df = sub
    units = S.load()["sections"][K.SPLIT_SECTION]["units"]
    for uid, sp in df.groupby("rig_unit")["split"].first().items():
        assert units[uid] == sp


def test_pool_windows_never_shared_across_splits(pools):
    k = {s: pools.window_keys(s) for s in K.SPLITS}
    assert not (k["train"] & k["val"]) and not (k["train"] & k["test"]) and not (k["val"] & k["test"])
    # time blocks: within each run the test block comes after val after train
    for r in pools.runs_used:
        assert r["blocks"]["train"] + r["blocks"]["val"] + r["blocks"]["test"] == r["post_settle_windows"]


def test_unit_windows_only_from_own_pool(pools, sub):
    _, df = sub
    keys = {s: pools.window_keys(s) for s in K.SPLITS}
    b1 = df[df["unit_id"].str.endswith(":B1")]
    for s in K.SPLITS:
        d = b1[b1["split"] == s]
        used = set(zip(d["h_run"], d["h_order"])) | set(zip(d["f_run"], d["f_order"]))
        assert used and used <= keys[s]
        for o in K.SPLITS:
            if o != s:
                assert not (used & keys[o])


def test_test_shape_wider_than_train(pools):
    plan = G.plan_units()
    sh = {}
    for s in ("train", "test"):
        specs = [p for p in plan if p["split"] == s and p["fault_class"] != "healthy"][:40]
        d = pd.concat([G.generate_unit(p, pools).head(1) for p in specs])
        sh[s] = (d["shape_a"].to_numpy(), d["shape_p"].to_numpy())
    assert sh["train"][0].min() >= K.SHAPE_TRAIN["a"][0] and sh["train"][0].max() <= K.SHAPE_TRAIN["a"][1]
    assert sh["train"][1].min() >= K.SHAPE_TRAIN["p"][0] and sh["train"][1].max() <= K.SHAPE_TRAIN["p"][1]
    assert (sh["test"][1] > K.SHAPE_TRAIN["p"][1]).any() or (sh["test"][1] < K.SHAPE_TRAIN["p"][0]).any()


def test_scale_check_ratio():
    assert scale.median_life_hours(1800) / scale.l_ref_hours(1800) == pytest.approx(K.LIFE_RHO)
    assert scale.l10_hours(1800, 0.5) == pytest.approx(1e6 / (60 * 1800) * 8)
    assert scale.l10_hours(3600, 0.5) == pytest.approx(scale.l10_hours(1800, 0.5) / 2)
    assert 0.3 <= K.LIFE_RHO <= 1.5                                 # not vacuous (life ~ L_ref)


def test_late_profile_matches_trace_if_present():
    prof = P.load_late_profile()
    assert prof["rms_ratio_end_over_seg0"] > 3
    if K.FAILURE_TRACE.exists():
        m = P.measure_late_profile()
        assert np.allclose(m["ln_end"], prof["ln_end"]) and np.allclose(m["ln_mid"], prof["ln_mid"])


def test_render_unit_roundtrip(tmp_path, pools):
    from aiengine.rig_anchor import render_unit as RU
    uid = [p for p in G.plan_units() if p["fault_class"] == "outer_race"][0]["unit_id"]
    out = tmp_path / "u.jsonl"
    info = RU.render(uid, str(out))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == info["windows"] and len(rows[0]["message"]["features"]) == 32 and "received_at" in rows[0]
    df = R.load_runs  # recorder reader accepts the {"message":..} lines
    X = np.array([r["message"]["features"] for r in rows])
    ref = G.window_matrix(G.generate_unit({p["unit_id"]: p for p in G.plan_units()}[uid], pools))
    assert np.allclose(X, ref)
