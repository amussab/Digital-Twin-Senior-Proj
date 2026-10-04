"""ONNX export of both models + parity checks + models/model_contract.json + golden vectors.

Graph inputs (both models): encoder_cont float32[B, L, R], decoder_cont float32[B, D, R]
(N-HiTS ignores decoder_cont; ONNX may prune it -- the contract lists the real inputs).
Outputs: N-HiTS `hi_forecast` [B, H, 1]; TFT `logits` [B, 1, C] (+ `variable_weights` [B, R],
the TFT decoder variable-selection weights of the current window, for top_features).

Parity is checked twice: (1) ONNX Runtime vs the PyTorch wrapper; (2) our own tensor builder
(infer.py, what C# mirrors) vs pytorch-forecasting's TimeSeriesDataSet tensors for the same
windows -- an exported graph fed differently-built tensors would compute the wrong thing silently.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_forecasting import TimeSeriesDataSet

from . import features as feat
from . import infer, labels, models, rul, train
from .config import BEARING_FEATURES, MODELS_DIR, SEED
from .payload import DISPLACEMENT_NAMES, PAYLOAD_SIZE

OPSET = 17
PARITY_TOL = 1e-4


class TFTExport(infer.ExportWrapper):
    def forward(self, encoder_cont, decoder_cont):
        b = encoder_cont.shape[0]
        x = {
            "encoder_cont": encoder_cont, "decoder_cont": decoder_cont,
            "encoder_cat": torch.zeros((b, self.L, 0), dtype=torch.long),
            "decoder_cat": torch.zeros((b, self.D, 0), dtype=torch.long),
            "encoder_target": encoder_cont[..., 0] * 0.0, "decoder_target": decoder_cont[..., 0] * 0.0,
            "encoder_lengths": torch.full((b,), self.L, dtype=torch.long),
            "decoder_lengths": torch.full((b,), self.D, dtype=torch.long),
            "decoder_time_idx": torch.arange(self.D, dtype=torch.long).expand(b, self.D),
            "groups": torch.zeros((b, 1), dtype=torch.long),
            "target_scale": self.ts.expand(b, -1),
        }
        out = self.model(x)
        w = out["decoder_variables"].reshape(b, self.D, -1).mean(dim=1)
        return out["prediction"], w


def _export(wrapper: nn.Module, enc: np.ndarray, dec: np.ndarray, path: Path, outputs: list[str]) -> dict:
    """Trace with batch 1 (the deployed call), then check parity sample by sample, and record
    whether the graph also accepts batch > 1 (N-HiTS bakes the traced batch into a Reshape)."""
    e1, d1 = torch.as_tensor(enc[:1]), torch.as_tensor(dec[:1])
    torch.onnx.export(wrapper, (e1, d1), str(path), input_names=["encoder_cont", "decoder_cont"],
                      output_names=outputs, opset_version=OPSET, do_constant_folding=True, dynamo=False,
                      dynamic_axes={"encoder_cont": {0: "batch"}, "decoder_cont": {0: "batch"},
                                    **{o: {0: "batch"} for o in outputs}})
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    diffs = np.zeros(len(outputs))
    with torch.no_grad():
        for i in range(len(enc)):
            ref = wrapper(torch.as_tensor(enc[i:i + 1]), torch.as_tensor(dec[i:i + 1]))
            ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
            feed = {k: v for k, v in {"encoder_cont": enc[i:i + 1], "decoder_cont": dec[i:i + 1]}.items() if k in names}
            got = s.run(outputs, feed)
            diffs = np.maximum(diffs, [float(np.max(np.abs(g - r))) for g, r in zip(got, ref)])
    try:
        feed = {k: v for k, v in {"encoder_cont": enc[:4], "decoder_cont": dec[:4]}.items() if k in names}
        s.run(outputs, feed)
        batch_ok = True
    except Exception:  # noqa: BLE001
        batch_ok = False
    return {"path": path.name, "size_kb": round(path.stat().st_size / 1024, 1), "onnx_inputs": names,
            "outputs": outputs, "parity_max_abs_diff": diffs.tolist(),
            "parity_ok": bool(all(x <= PARITY_TOL for x in diffs)), "parity_samples": int(len(enc)),
            "batch_dim": "dynamic" if batch_ok else "fixed at 1 (call once per bearing-window)"}


def builder_parity_tft(model, unit: pd.DataFrame) -> float:
    """Our tft_tensors vs TimeSeriesDataSet tensors for full-length windows of one unit."""
    params = model.dataset_parameters
    cols, sc, L = train.tft_layout(model)
    frame = unit[["unit_id", "window_index", "observable_class"] + cols].copy()
    frame["unit_id"] = frame["unit_id"].astype(str)
    frame["observable_class"] = frame["observable_class"].astype(str)
    ds = TimeSeriesDataSet.from_parameters(params, frame, predict=False, stop_randomization=True)
    enc, dec = infer.tft_tensors(unit[cols].to_numpy(np.float64), cols, sc, L)
    worst = 0.0
    for x, _ in ds.to_dataloader(train=False, batch_size=512, num_workers=0):
        idx = ds.x_to_index(x)
        full = (x["encoder_lengths"] == L).numpy()
        for j in np.flatnonzero(full):
            k = int(idx["window_index"].iloc[j]) - L          # row in enc/dec
            worst = max(worst, float(np.abs(x["encoder_cont"][j].numpy() - enc[k]).max()),
                        float(np.abs(x["decoder_cont"][j].numpy() - dec[k]).max()))
    return worst


def builder_parity_nhits(model, unit: pd.DataFrame) -> float:
    params = model.dataset_parameters
    cols, sc, L, H = train.nhits_layout(model)
    frame = unit[["unit_id", "window_index"] + cols].copy()
    frame["unit_id"] = frame["unit_id"].astype(str)
    ds = TimeSeriesDataSet.from_parameters(params, frame, predict=False, stop_randomization=True)
    enc, _ = infer.nhits_tensors(unit[cols].to_numpy(np.float64), cols, sc, L, H)
    worst = 0.0
    for x, _ in ds.to_dataloader(train=False, batch_size=512, num_workers=0):
        idx = ds.x_to_index(x)
        for j in range(len(idx)):
            k = int(idx["window_index"].iloc[j]) - 1          # encoder ends one step before decoder
            worst = max(worst, float(np.abs(x["encoder_cont"][j].numpy() - enc[k]).max()))
    return worst


def run(run_dir: Path, df: pd.DataFrame, tft_ckpt=None, nhits_ckpt=None, calib_file=None,
        out_dir: Path = MODELS_DIR, provenance: dict | None = None, golden_unit: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tft = train.load_model(Path(tft_ckpt or run_dir / "tft.ckpt"))
    nh = train.load_model(Path(nhits_ckpt or run_dir / "nhits.ckpt"))
    calib_path = Path(calib_file or run_dir / "rul_calibration.json")
    calib = json.loads(calib_path.read_text())

    tcols, tsc, TL = train.tft_layout(tft)
    ncols, nsc, NL, NH = train.nhits_layout(nh)
    test = df[df["split"].isin(["test", "ft_test"])]
    rtf = test[test["run_to_failure"]]
    unit_id = golden_unit or (rtf if len(rtf) else test).groupby("unit_id").size().idxmax()
    unit = df[df["unit_id"] == unit_id].sort_values("window_index")

    enc, dec = infer.tft_tensors(unit[tcols].to_numpy(np.float64), tcols, tsc, TL)
    tft_w = TFTExport(tft, TL, 1, [0.0, 1.0]).eval()
    res_t = _export(tft_w, enc[:64], dec[:64], out_dir / "tft.onnx", ["logits", "variable_weights"])
    ne, nd = infer.nhits_tensors(unit[ncols].to_numpy(np.float64), ncols, nsc, NL, NH)
    res_n = _export(train.nhits_wrapper(nh), ne[-64:], nd[-64:], out_dir / "nhits.onnx", ["hi_forecast"])
    res_t["builder_vs_dataset_max_abs_diff"] = builder_parity_tft(tft, unit)
    res_n["builder_vs_dataset_max_abs_diff"] = builder_parity_nhits(nh, unit)

    order = models.class_order(TimeSeriesDataSet.from_parameters(
        tft.dataset_parameters, unit[["unit_id", "window_index", "observable_class"] + tcols]
        .assign(unit_id=lambda d: d["unit_id"].astype(str),
                observable_class=lambda d: d["observable_class"].astype(str)), predict=False))
    contract = {
        "contract_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "AI-engine/aiengine/export.py",
        "provenance": provenance or {},
        "model_unit": "one biaxial bearing: 16 raw features = X's 8 then Y's 8 (COE C7 order)",
        "wire_payload": {"size_bytes": PAYLOAD_SIZE, "byte_order": "little-endian",
                         "fields": [{"offset": 0, "type": "float32", "name": "rpm"},
                                    {"offset": 4, "type": "float32[32]", "name": "features"},
                                    {"offset": 132, "type": "float32[4]", "name": DISPLACEMENT_NAMES},
                                    {"offset": 148, "type": "uint32", "name": "t20_ms"}],
                         "bearing_slices": {"1": [0, 16], "2": [16, 32]},
                         "note": "COE doc: 152 bytes; FDR deck: 168 (unreconciled)."},
        "raw_bearing_feature_order": BEARING_FEATURES,
        "feature_engineering": {
            "engineered_order": feat.ENGINEERED,
            "baseline": {"windows": feat.BASELINE_WINDOWS, "statistic": "median",
                         "rel_floor": feat.BASELINE_REL_FLOOR,
                         "rig_policy": "commissioning capture (spec M5); else median of first 24 windows"},
            "eps": feat.EPS,
            "health_index": {"r_ref": feat.HI_R_REF, "curve_k": feat.HI_CURVE_K, "weights": feat.HI_WEIGHTS,
                             "clip_max": feat.HI_CLIP_MAX,
                             "formula": "fused = max(env_ratio_max,1)^w_env * max(bp_ratio_max,1)^w_bp * "
                                        "max(family_ratio_max,1)^w_fam; hi = clip(log1p(k(fused-1))/log1p(k(r_ref-1)), 0, clip_max)"},
            "rpm_norm": {"center": feat.RPM_CENTER, "scale": feat.RPM_SCALE},
            "reference_impl": "aiengine/features.py::engineer_arrays",
        },
        "onset": {"sigmas": labels.ONSET_SIGMAS, "consecutive": labels.ONSET_CONSECUTIVE,
                  "sigma_floor": labels.SIGMA_FLOOR, "baseline_windows": feat.BASELINE_WINDOWS,
                  "run_in_skip_windows": labels.ONSET_RUN_IN_SKIP,
                  "rule": "threshold = mean + sigmas*max(std, sigma_floor) of HI windows [skip, skip+baseline); "
                          "onset = first window of the first run of `consecutive` windows above it"},
        "classes": {"labels": order, "index_is_output_channel": True,
                    "note": "alphabetical (label encoder) order; apply softmax to logits"},
        "models": {
            "tft": {"onnx": "tft.onnx", "encoder_length": TL, "decoder_length": 1, "columns": tcols,
                    "scalers": tsc, "warmup_windows": TL + 1, **res_t},
            "nhits": {"onnx": "nhits.onnx", "encoder_length": NL, "prediction_length": NH,
                      "columns": ncols, "scalers": nsc, "target": "hi", "left_pad": "repeat first row",
                      **res_n},
        },
        "rul": {**calib, "procedure": "aiengine/rul.py::estimate; class for threshold = argmax of mean "
                "fault probability over the last 5 windows if mean p(healthy) < 0.5, else pooled",
                "class_smoothing_windows": 5},
        "stage": "1 before confirmed onset; else 2 + floor(5 * tau/(tau+RUL)), max 6",
    }
    (out_dir / "model_contract.json").write_text(json.dumps(contract, indent=2, default=float))
    # golden vectors via the reference engine (reads the contract we just wrote)
    from .engine import HybridEngine
    eng = HybridEngine(out_dir)
    golden = eng.golden_vectors(unit, n=12)
    # Engine (ONNX Runtime, streaming) vs evaluation path (PyTorch, batch) on the same unit.
    consistency = {}
    if bool(unit["run_to_failure"].iloc[0]):
        cal = rul.RULCalibration.from_dict(calib)
        sc = train.apply_rul(cal, train.rul_records(nh, unit, train.tft_predict(tft, unit)))
        outs = eng.run_unit(unit)
        ok = [(w, o) for w, o in zip(unit["window_index"], outs) if o.get("status") == "ok"]
        er = pd.DataFrame({"window_index": [w for w, _ in ok], "rul_engine": [o["rul_hours"] for _, o in ok],
                           "stage_engine": [o["health_stage"] for _, o in ok]})
        j = sc.merge(er, on="window_index")
        j = j[j["window_index"] >= TL + 6]
        rel = np.abs(j["rul_engine"] - j["rul_pred"]) / np.maximum(np.abs(j["rul_pred"]), 1e-6)
        consistency = {"windows": int(len(j)), "rul_max_rel_diff": float(rel.max()),
                       "rul_p99_rel_diff": float(np.quantile(rel, 0.99)),
                       "stage_mismatch_frac": float((j["stage_engine"] != j["stage_pred"]).mean())}
    golden["engine_vs_evaluation"] = consistency
    (out_dir / "golden_vectors.json").write_text(json.dumps(golden, indent=1, default=float))
    shutil.copy(calib_path, out_dir / "rul_calibration.json")
    return {"tft": res_t, "nhits": res_n, "golden_unit": unit_id, "engine_vs_evaluation": consistency, "contract": str(out_dir / "model_contract.json")}
