"""Export both models to ONNX and emit the contract the .NET host needs.

Why this module is not an afterthought
--------------------------------------
The production backend is ASP.NET Core, and the ICS layer doc's B9 design runs
both models in-process via ONNX Runtime's C# binding rather than calling out to
a Python service -- that in-process choice is what removes a network hop from
I1's 500 ms end-to-end budget. So ONNX export is not a convenience: it is the
boundary between this repository and the backend, and if a model will not
export, the B9 design needs to know now rather than during integration.

This module therefore does three things and reports honestly on all three:

1.  Exports each model to ONNX.
2.  Re-runs the exported graph under ONNX Runtime and checks it against
    PyTorch's own output. An export that loads but computes something else is
    worse than one that fails.
3.  Writes `model_contract.json` -- the single artifact the C# side reads. It
    carries the tensor layout, the exact column ordering of the input tensor,
    the categorical encodings, every feature-engineering constant from
    features.py, the class labels in index order, and the RUL calibration.
    Nothing in the C# code should hardcode any of it.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import features as feat_mod
from config import (
    ARTIFACT_DIR,
    BEARING_ORDERS,
    DISPLACEMENT_NAMES,
    FAULT_CLASSES,
    FEATURE_NAMES,
    N_HEALTH_STAGES,
    OPERATING_RPM,
    STAGE_DAMAGE_EDGES,
    RunConfig,
)
from payload import PAYLOAD_SIZE

CONTRACT_FILE = ARTIFACT_DIR / "model_contract.json"
NHITS_ONNX = ARTIFACT_DIR / "nhits.onnx"
TFT_ONNX = ARTIFACT_DIR / "tft.onnx"

OPSET = 17
PARITY_TOLERANCE = 1e-3


class ExportWrapper(nn.Module):
    """Turns a pytorch-forecasting model's dict input into positional tensors.

    ONNX graphs take tensors, not dictionaries. The lengths and time-index
    tensors the library expects are fully determined by the fixed encoder and
    decoder lengths, so they are reconstructed inside the wrapper rather than
    crossing the boundary -- which keeps the C# call site down to four inputs.
    """

    def __init__(self, model: nn.Module, encoder_length: int, decoder_length: int,
                 output_key: str = "prediction"):
        super().__init__()
        self.model = model
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.output_key = output_key

    def forward(
        self,
        encoder_cont: torch.Tensor,
        encoder_cat: torch.Tensor,
        decoder_cont: torch.Tensor,
        decoder_cat: torch.Tensor,
        encoder_target: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch = encoder_cont.shape[0]
        device = encoder_cont.device
        payload = {
            "encoder_cont": encoder_cont,
            "encoder_cat": encoder_cat,
            "encoder_target": encoder_target,
            "encoder_lengths": torch.full(
                (batch,), self.encoder_length, dtype=torch.long, device=device
            ),
            "decoder_cont": decoder_cont,
            "decoder_cat": decoder_cat,
            "decoder_target": torch.zeros(
                (batch, self.decoder_length), dtype=encoder_target.dtype, device=device
            ),
            "decoder_lengths": torch.full(
                (batch,), self.decoder_length, dtype=torch.long, device=device
            ),
            "decoder_time_idx": torch.arange(
                self.decoder_length, dtype=torch.long, device=device
            ).expand(batch, self.decoder_length),
            "groups": torch.zeros((batch, 1), dtype=torch.long, device=device),
            "target_scale": target_scale,
        }
        return self.model(payload)[self.output_key]


def _sample_inputs(dataset, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    x = next(iter(loader))[0]
    return (
        x["encoder_cont"],
        x["encoder_cat"],
        x["decoder_cont"],
        x["decoder_cat"],
        x["encoder_target"],
        x["target_scale"],
    )


def export_model(model, dataset, encoder_length: int, decoder_length: int,
                 path: Path, name: str) -> dict:
    """Export one model, then verify the exported graph agrees with PyTorch."""
    print(f"\n[{name}] exporting to ONNX")
    model.eval()
    wrapper = ExportWrapper(model, encoder_length, decoder_length).eval()
    inputs = _sample_inputs(dataset)

    input_names = [
        "encoder_cont", "encoder_cat", "decoder_cont",
        "decoder_cat", "encoder_target", "target_scale",
    ]
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["output"] = {0: "batch"}

    result: dict = {"name": name, "path": str(path)}
    try:
        with torch.no_grad():
            reference = wrapper(*inputs).detach().cpu().numpy()
        torch.onnx.export(
            wrapper,
            inputs,
            str(path),
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        result["exported"] = True
        result["output_shape"] = list(reference.shape)
        print(f"  exported -> {path.name}  output shape {tuple(reference.shape)}")
    except Exception as exc:                                  # noqa: BLE001
        result["exported"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=6)
        print(f"  EXPORT FAILED: {type(exc).__name__}: {exc}")
        return result

    # Parity check: an export that loads but computes something different is a
    # worse outcome than one that fails loudly.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        # Constant folding prunes inputs the graph never reads, and ONNX
        # Runtime rejects feeds for names that no longer exist. Feed only what
        # the graph declares, and record it: those names are exactly the
        # tensors the C# call site has to build, and it is a shorter list than
        # the PyTorch signature suggests.
        required = [entry.name for entry in session.get_inputs()]
        available = dict(zip(input_names, inputs))
        result["onnx_required_inputs"] = required
        result["onnx_pruned_inputs"] = [n for n in input_names if n not in required]
        feeds = {
            name: available[name].detach().cpu().numpy()
            for name in required
            if name in available
        }
        onnx_out = session.run(["output"], feeds)[0]
        max_diff = float(np.max(np.abs(onnx_out - reference)))
        result["parity_max_abs_diff"] = max_diff
        result["parity_ok"] = bool(max_diff <= PARITY_TOLERANCE)
        verdict = "OK" if result["parity_ok"] else "MISMATCH"
        print(f"  parity vs PyTorch: {verdict} (max abs diff {max_diff:.2e})")
    except Exception as exc:                                  # noqa: BLE001
        result["parity_ok"] = False
        result["parity_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  parity check failed: {type(exc).__name__}: {exc}")

    return result


def build_contract(nhits_dataset, tft_dataset, cfg: RunConfig,
                   exports: list[dict], calibration: dict,
                   class_labels: list[str]) -> dict:
    """The one artifact the .NET host reads. No magic numbers on the C# side."""

    def encoder_classes(dataset) -> dict:
        """Value -> integer index for each categorical the model embeds.

        The C# side fills `encoder_cat` with these integers, so the mapping has
        to travel with the model. In pytorch-forecasting 1.8 the public
        `categorical_encoders` property can come back None while the private
        attribute holds the dict, so both are tried. The internal group-id
        encoder is skipped: it encodes run identity, which the deployed host
        does not have and the graph does not use.
        """
        encoders = (
            getattr(dataset, "categorical_encoders", None)
            or getattr(dataset, "_categorical_encoders", None)
            or {}
        )
        used = set(getattr(dataset, "flat_categoricals", []) or [])
        out = {}
        for column, encoder in encoders.items():
            if column not in used:
                continue
            classes = getattr(encoder, "classes_", None)
            if classes is not None:
                out[column] = {str(k): int(v) for k, v in dict(classes).items()}
        return out

    return {
        "contract_version": 1,
        "generated_by": "AI-test/export_onnx.py",
        "data_provenance": "SYNTHETIC -- models trained on generated "
                           "run-to-failure data, not on rig measurements.",
        "wire_payload": {
            "size_bytes": PAYLOAD_SIZE,
            "byte_order": "little-endian",
            "fields": [
                {"offset": 0, "type": "float32", "name": "rpm"},
                {"offset": 4, "type": "float32[32]", "name": "features"},
                {"offset": 132, "type": "float32[4]", "name": DISPLACEMENT_NAMES},
                {"offset": 148, "type": "uint32", "name": "t20_ms"},
            ],
            "note": "COE working docs specify 152 bytes; the FDR deck states "
                    "168. Unreconciled -- confirm with COE before the C# "
                    "parser is finalised.",
        },
        "raw_feature_order": FEATURE_NAMES,
        "feature_engineering": {
            "baseline_windows": feat_mod.BASELINE_WINDOWS,
            "baseline_statistic": "median",
            "health_index": {
                "r_ref": feat_mod.HI_R_REF,
                "curve_k": feat_mod.HI_CURVE_K,
                "weights": feat_mod.HI_WEIGHTS,
                "formula": "hi = log1p(k*(fused-1)) / log1p(k*(r_ref-1)), "
                           "fused = prod(ratio_i ** weight_i), ratios clipped "
                           "at >= 1, result clipped to [0, 1.2]",
            },
            "model_input_order": feat_mod.MODEL_INPUTS,
        },
        "bearing_orders": BEARING_ORDERS,
        "operating_modes_rpm": list(OPERATING_RPM),
        "classes": {
            # Taken from the trained dataset's label encoder, NOT from
            # config.FAULT_CLASSES. The encoder sorts labels alphabetically, so
            # the two differ. The C# side must index output channels with this
            # list; using the config's declaration order permutes every class.
            "labels": class_labels,
            "declared_order_in_config": FAULT_CLASSES,
            "index_is_output_channel": True,
            "note": "TFT output is logits over `labels` in this exact order; "
                    "apply softmax on the C# side for confidence.",
        },
        "health_stages": {
            "count": N_HEALTH_STAGES,
            "damage_edges": STAGE_DAMAGE_EDGES,
        },
        "models": {
            "nhits": {
                "role": "health-index forecast -> RUL (ICS1)",
                "onnx": NHITS_ONNX.name,
                "encoder_length": cfg.nhits.encoder_length,
                "decoder_length": cfg.nhits.prediction_length,
                "encoder_cont_columns": list(getattr(nhits_dataset, "reals", [])),
                "encoder_cat_columns": list(getattr(nhits_dataset, "flat_categoricals", [])),
                "categorical_encodings": encoder_classes(nhits_dataset),
                "output": "float32[batch, decoder_length, 1] -- forecast "
                          "health index, identity-normalised (no inverse "
                          "transform needed)",
            },
            "tft": {
                "role": "per-window fault classification (ICS2)",
                "onnx": TFT_ONNX.name,
                "encoder_length": cfg.tft.encoder_length,
                "decoder_length": cfg.tft.prediction_length,
                "encoder_cont_columns": list(getattr(tft_dataset, "reals", [])),
                "encoder_cat_columns": list(getattr(tft_dataset, "flat_categoricals", [])),
                "categorical_encodings": encoder_classes(tft_dataset),
                "output": "float32[batch, 1, n_classes] -- raw logits",
            },
        },
        "rul_calibration": calibration,
        "rul_procedure": {
            "summary": "Forecast the health index, fit an exponential trend "
                       "through recent history plus the forecast, extrapolate "
                       "to the class-conditional failure threshold, then blend "
                       "with the population prior by trend confidence.",
            "trend_history_windows": 24,
            "trend_floor_hi": 0.08,
            "note": "Implemented in AI-test/rul.py. The C# host must mirror "
                    "it, or call it, but must not invent its own.",
        },
        "exports": exports,
    }


def run(cfg: RunConfig) -> dict:
    import pandas as pd
    from pytorch_forecasting import NHiTS, TemporalFusionTransformer

    import data as data_mod
    from train import CALIB_FILE, ENGINEERED_FILE, NHITS_CKPT, TFT_CKPT, load_splits

    print("=" * 74)
    print("ICS AI testbench -- ONNX export for the .NET host")
    print("=" * 74)

    engineered = pd.read_parquet(ENGINEERED_FILE)
    splits = load_splits()
    train_frame = data_mod.subset(engineered, splits["train"])
    test_frame = data_mod.subset(engineered, splits["test"])

    nhits_train, nhits_sets = data_mod.build_nhits_datasets(
        train_frame, {"test": test_frame}, cfg.nhits
    )
    tft_train, tft_sets = data_mod.build_tft_datasets(
        train_frame, {"test": test_frame}, cfg.tft
    )

    nhits = NHiTS.load_from_checkpoint(NHITS_CKPT, map_location="cpu")
    tft = TemporalFusionTransformer.load_from_checkpoint(TFT_CKPT, map_location="cpu")

    exports = [
        export_model(nhits, nhits_sets["test"], cfg.nhits.encoder_length,
                     cfg.nhits.prediction_length, NHITS_ONNX, "N-HiTS"),
        export_model(tft, tft_sets["test"], cfg.tft.encoder_length,
                     cfg.tft.prediction_length, TFT_ONNX, "TFT"),
    ]

    calibration = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
    class_labels = data_mod.class_order(tft_train)
    contract = build_contract(
        nhits_train, tft_train, cfg, exports, calibration, class_labels
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_FILE.write_text(json.dumps(contract, indent=2, default=float))

    print(f"\nContract written to {CONTRACT_FILE}")
    ok = [e for e in exports if e.get("exported") and e.get("parity_ok")]
    failed = [e for e in exports if not (e.get("exported") and e.get("parity_ok"))]
    print(f"  {len(ok)}/{len(exports)} models export and match PyTorch output.")
    for entry in failed:
        print(f"  ! {entry['name']}: "
              f"{entry.get('error') or entry.get('parity_error') or 'parity mismatch'}")
    if failed:
        print("\n  A model that will not export cannot run in-process under "
              "ONNX Runtime in the ASP.NET host. Options, in the order worth "
              "trying: torch.onnx.export with dynamo=True, a newer opset, or "
              "TorchSharp; failing those, B9 keeps a Python inference sidecar "
              "and I1's latency budget has to absorb a loopback hop.")
    return contract
