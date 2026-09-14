"""Live replay: stream one run's windows through the full host-side path.

This is the closest thing here to what the ASP.NET host will do per window --
decode the wire payload, engineer features against the stored baseline, run
both models, produce a class call and a RUL number, and time the whole thing.
It exists to make the per-window path visible and timeable as one sequence,
rather than only as aggregate metrics.

What it is NOT: a dashboard. The dashboard and digital-twin presentation layer
is ICS3, owned by Abdulrazaq, and is an ASP.NET Core / SignalR application.
This prints to a terminal.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import NHiTS, TemporalFusionTransformer

import data as data_mod
import rul as rul_mod
from config import FAULT_CLASSES, RunConfig
from payload import Window, decode, encode, is_valid
from train import CALIB_FILE, ENGINEERED_FILE, NHITS_CKPT, TFT_CKPT, load_splits


def _row_to_window(row: pd.Series) -> Window:
    from config import FEATURE_NAMES

    return Window(
        rpm=float(row["rpm"]),
        features=np.asarray([row[name] for name in FEATURE_NAMES], dtype=np.float32),
        p1_amp_um=float(row["p1_amp_um"]),
        p1_phase_rad=float(row["p1_phase_rad"]),
        p2_amp_um=float(row["p2_amp_um"]),
        p2_phase_rad=float(row["p2_phase_rad"]),
        t20_ms=int(row["t20_ms"]),
    )


def run(cfg: RunConfig, run_id: int | None = None, limit: int = 60,
        pause: float = 0.0) -> None:
    engineered = pd.read_parquet(ENGINEERED_FILE)
    splits = load_splits()
    calib = rul_mod.RULCalibration.from_dict(json.loads(CALIB_FILE.read_text()))

    if run_id is None:
        # Prefer a faulted run: a healthy one replays a flat health index and
        # shows nothing of what the models are for.
        faulted = [
            int(r) for r in splits["test"]
            if str(engineered.loc[engineered["run_id"] == r, "fault_class"].iloc[0])
            != "healthy"
        ]
        run_id = faulted[0] if faulted else splits["test"][0]
    if run_id not in splits["test"]:
        print(f"  note: run {run_id} is not in the held-out test set "
              f"({splits['test']}). Replaying it anyway.")

    run_frame = engineered[engineered["run_id"] == run_id].sort_values("window_index")
    if run_frame.empty:
        print(f"No run with id {run_id}.")
        return

    true_class = str(run_frame["fault_class"].iloc[0])
    train_frame = data_mod.subset(engineered, splits["train"])
    _, nhits_sets = data_mod.build_nhits_datasets(
        train_frame, {"replay": run_frame}, cfg.nhits
    )
    tft_train, tft_sets = data_mod.build_tft_datasets(
        train_frame, {"replay": run_frame}, cfg.tft
    )
    # Output-channel order, not config.FAULT_CLASSES -- see data.class_order.
    order = data_mod.class_order(tft_train)

    nhits = NHiTS.load_from_checkpoint(NHITS_CKPT, map_location="cpu").eval()
    tft = TemporalFusionTransformer.load_from_checkpoint(
        TFT_CKPT, map_location="cpu"
    ).eval()

    print("=" * 96)
    print(f"Replaying run {run_id}  --  true failure mode: {true_class}  "
          f"({len(run_frame)} windows, {run_frame['rpm'].mean():.0f} rpm)")
    print("=" * 96)
    print(f"{'win':>5} {'stage':>5} {'HI':>6} {'call':>11} {'conf':>6} "
          f"{'RUL pred':>9} {'RUL true':>9} {'err':>7} {'ms':>7}")
    print("-" * 96)

    nhits_loader = nhits_sets["replay"].to_dataloader(
        train=False, batch_size=1, num_workers=0
    )
    tft_loader = tft_sets["replay"].to_dataloader(
        train=False, batch_size=1, num_workers=0
    )
    # The two models have different encoder lengths, so they become valid at
    # different points in the run. Align them on window index, using the
    # dataset's own x_to_index rather than its internal `.index` frame, whose
    # columns are library bookkeeping and not the group/time columns.
    def by_window(dataset, loader) -> dict[int, dict]:
        out: dict[int, dict] = {}
        with torch.no_grad():
            for batch in loader:
                index = dataset.x_to_index(batch[0])
                out[int(index["window_index"].iloc[0])] = batch[0]
        return out

    tft_by_window = by_window(tft_sets["replay"], tft_loader)
    nhits_by_window = by_window(nhits_sets["replay"], nhits_loader)

    hi_series = run_frame.set_index("window_index")["hi"]
    truth = run_frame.set_index("window_index")
    shared = sorted(set(nhits_by_window) & set(tft_by_window))[:limit]

    latencies, errors = [], []
    for window_index in shared:
        row = truth.loc[window_index]

        # Wire round-trip, so the replay exercises the real parser rather than
        # reading the dataframe directly.
        raw = encode(_row_to_window(row))
        window = decode(raw)
        ok, reason = is_valid(window)
        if not ok:
            print(f"{window_index:>5}  rejected: {reason}")
            continue

        started = time.perf_counter()
        with torch.no_grad():
            logits = tft(tft_by_window[window_index])["prediction"]
            forecast = nhits(nhits_by_window[window_index])["prediction"]
        probs = torch.softmax(logits, dim=-1).numpy()[0, 0, :]
        predicted_class = order[int(probs.argmax())]

        history = hi_series.loc[: window_index - 1].to_numpy()[-cfg.nhits.encoder_length:]
        estimate = rul_mod.estimate(
            calib, history, forecast.numpy().reshape(-1), predicted_class
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)

        rul_true = float(row["rul_hours"])
        if np.isfinite(rul_true) and rul_true >= 1.0:
            error = abs(estimate["rul_hours"] - rul_true) / rul_true * 100.0
            errors.append(error)
            error_text = f"{error:6.1f}%"
            true_text = f"{rul_true:8.1f}h"
        else:
            error_text = "      -"
            true_text = "       -"

        print(
            f"{window_index:>5} {int(row['health_stage']):>5} "
            f"{estimate['hi_now']:>6.3f} {predicted_class:>11} "
            f"{probs.max():>6.2f} {estimate['rul_hours']:>8.1f}h {true_text} "
            f"{error_text} {elapsed_ms:>6.1f}"
        )
        if pause:
            time.sleep(pause)

    if latencies:
        arr = np.asarray(latencies)
        print("-" * 96)
        print(f"per-window latency: mean {arr.mean():.1f} ms, "
              f"p95 {np.percentile(arr, 95):.1f} ms, max {arr.max():.1f} ms "
              f"(ICS2 budget: 200 ms)")
        if errors:
            print(f"RUL error over replayed windows: mean "
                  f"{np.mean(errors):.1f}%, median {np.median(errors):.1f}%")
