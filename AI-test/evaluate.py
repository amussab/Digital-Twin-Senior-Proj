"""Spec-check harness: measure both models against ICS1, ICS2, I1 and I3.

Everything reported here is measured on HELD-OUT RUNS the models never saw, on
SYNTHETIC data. The first of those makes the numbers meaningful; the second
caps what they may be used to claim. Both facts are stamped on the report this
module writes.
"""

from __future__ import annotations

import json
import platform
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import NHiTS, TemporalFusionTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import data as data_mod
import rul as rul_mod
from config import (
    ARTIFACT_DIR,
    FAULT_CLASSES,
    REPORT_DIR,
    RunConfig,
    SPEC_TARGETS,
    SpecTarget,
)
from train import (
    CALIB_FILE,
    ENGINEERED_FILE,
    META_FILE,
    NHITS_CKPT,
    TFT_CKPT,
    load_splits,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Consecutive windows a fault call must hold before it counts as "detected".
# One-window agreement is a coin flip on a noisy signal; the dashboard would
# not raise an alarm on it either.
DETECTION_PERSISTENCE = 3


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def _predict_nhits(model: NHiTS, dataset, batch_size: int) -> pd.DataFrame:
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    out = model.predict(loader, mode="prediction", return_index=True, return_x=False)
    forecasts = np.asarray(out.output).reshape(len(out.index), -1)
    frame = out.index.copy()
    frame["forecast"] = list(forecasts)
    return frame


def _predict_tft(model: TemporalFusionTransformer, dataset, batch_size: int,
                 order: list[str]) -> pd.DataFrame:
    """`order` is the model's own output-channel order -- see data.class_order."""
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    out = model.predict(loader, mode="raw", return_index=True, return_x=False)
    logits = out.output["prediction"].detach().cpu()      # (n, 1, n_classes)
    probs = torch.softmax(logits, dim=-1).numpy()[:, 0, :]
    frame = out.index.copy()
    frame["pred_class"] = [order[i] for i in probs.argmax(axis=1)]
    frame["confidence"] = probs.max(axis=1)
    for idx, name in enumerate(order):
        frame[f"p_{name}"] = probs[:, idx]
    return frame


# --------------------------------------------------------------------------
# ICS2 / I1 -- latency
# --------------------------------------------------------------------------

def measure_latency(model, dataset, n_windows: int, label: str) -> dict:
    """Time single-window inference, which is what ICS2's budget covers.

    Batch size 1 on purpose: the node emits one window at a time and the host
    must answer for it before the next arrives. A batched throughput figure
    would look far better and would not describe the deployed path.
    """
    loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    model.eval()

    samples: list[float] = []
    with torch.no_grad():
        iterator = iter(loader)
        # Warm-up: the first calls pay lazy-init and allocator costs that a
        # long-running host pays once at startup, not per window.
        for _ in range(10):
            try:
                model(next(iterator)[0])
            except StopIteration:
                iterator = iter(loader)
        for _ in range(n_windows):
            try:
                batch = next(iterator)[0]
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)[0]
            started = time.perf_counter()
            model(batch)
            samples.append((time.perf_counter() - started) * 1000.0)

    arr = np.asarray(samples)
    return {
        f"{label}_latency_mean_ms": float(arr.mean()),
        f"{label}_latency_p50_ms": float(np.percentile(arr, 50)),
        f"{label}_latency_p95_ms": float(np.percentile(arr, 95)),
        f"{label}_latency_max_ms": float(arr.max()),
        f"{label}_latency_n": int(len(arr)),
    }


# --------------------------------------------------------------------------
# ICS1 -- RUL
# --------------------------------------------------------------------------

def rul_information_floor(test_frame: pd.DataFrame) -> dict:
    """The best MAPE ANY fraction-of-life method could reach on this data.

    Why this belongs in the report rather than in a footnote: the health index
    says how far through its life a bearing is, not how long that life is. If
    RUL = T * (1 - u) and the method recovers u exactly, its relative error is
    exactly the relative error in T -- so the spread of total run-to-failure
    lives sets a hard floor under the achievable MAPE, independent of the model.

    Reporting this alongside the measured MAPE is what separates "the model
    needs more training" from "this formulation cannot reach the target", which
    are very different findings and call for very different next steps.
    """
    faulted = test_frame[test_frame["fault_class"].astype(str) != "healthy"]
    if faulted.empty:
        return {}
    lifetimes = (faulted.groupby("run_id")["window_index"].max() + 1).to_numpy(dtype=float)
    if len(lifetimes) < 2:
        return {}
    candidates = np.linspace(lifetimes.min(), lifetimes.max(), 512)
    errors = [float(np.mean(np.abs(c - lifetimes) / lifetimes) * 100.0) for c in candidates]
    best = int(np.argmin(errors))
    return {
        "rul_information_floor_pct": errors[best],
        "rul_lifetime_spread_ratio": float(lifetimes.max() / lifetimes.min()),
        "rul_lifetime_runs": int(len(lifetimes)),
    }


def evaluate_rul(
    nhits_preds: pd.DataFrame,
    tft_preds: pd.DataFrame,
    test_frame: pd.DataFrame,
    calib: rul_mod.RULCalibration,
    encoder_length: int,
) -> tuple[dict, pd.DataFrame]:
    """Blend forecast-driven and prior-driven RUL, then score against truth."""
    truth = test_frame.set_index(["run_id", "window_index"])
    hi_by_run = {
        int(run_id): run.sort_values("window_index")["hi"].to_numpy()
        for run_id, run in test_frame.groupby("run_id")
    }

    # The classifier's call selects the failure threshold. Where TFT has no
    # prediction for a window (its encoder is shorter, so the two models'
    # valid ranges differ), fall back to the pooled threshold.
    tft_lookup = tft_preds.set_index(["run_id", "window_index"])["pred_class"].to_dict()

    rows = []
    for record in nhits_preds.itertuples(index=False):
        run_id, window_index = int(record.run_id), int(record.window_index)
        key = (run_id, window_index)
        if key not in truth.index:
            continue
        actual = truth.loc[key]
        rul_true = float(actual["rul_hours"])
        if not np.isfinite(rul_true):
            continue                       # healthy run: no failure to predict

        history = hi_by_run[run_id][max(0, window_index - encoder_length) : window_index]
        if len(history) < 4:
            continue

        predicted_class = tft_lookup.get(key, "healthy")
        estimate = rul_mod.estimate(
            calib, history, record.forecast, predicted_class
        )
        rows.append(
            {
                "run_id": run_id,
                "window_index": window_index,
                "fault_class": str(actual["fault_class"]),
                "health_stage": int(actual["health_stage"]),
                "rul_true_hours": rul_true,
                **estimate,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rul_mape_pct": float("nan"), "rul_n": 0}, frame

    def mape(pred: np.ndarray, true: np.ndarray) -> float:
        # Windows at the very end of life have true RUL at or near zero, where
        # a percentage error is undefined or explodes on a rounding-scale
        # difference. Excluded from the percentage metric and reported
        # separately as an absolute error instead.
        usable = true >= 1.0
        if not usable.any():
            return float("nan")
        return float(
            np.mean(np.abs(pred[usable] - true[usable]) / true[usable]) * 100.0
        )

    true = frame["rul_true_hours"].to_numpy()
    blended = frame["rul_hours"].to_numpy()
    prior_only = frame["rul_prior_hours"].to_numpy()

    metrics = {
        "rul_mape_pct": mape(blended, true),
        "rul_mape_prior_only_pct": mape(prior_only, true),
        "rul_mae_hours": float(np.mean(np.abs(blended - true))),
        "rul_n": int(len(frame)),
        "rul_mean_trend_confidence": float(frame["trend_confidence"].mean()),
    }
    metrics.update(rul_information_floor(test_frame))

    # Scale-invariant RUL, as percent of total life remaining rather than
    # absolute hours. The external-datasets doc flags absolute-hours labels as
    # a concrete transfer risk (risk 2): FEMTO-ST and XJTU-SY label in hours of
    # their own accelerated tests, this project labels from a different life
    # law, and pretraining across that mismatch yields a model that is
    # miscalibrated in a way that is easy to miss. Measuring both here shows
    # how much of the absolute-hours error is nothing but life-length spread.
    lifetimes = (
        test_frame.groupby("run_id")["window_index"].max() + 1
    ).to_dict()
    total_hours = frame["run_id"].map(lifetimes).to_numpy() * calib.hours_per_window
    true_pct = true / total_hours * 100.0
    pred_pct = np.clip(blended / total_hours * 100.0, 0.0, 200.0)
    usable = true_pct >= 1.0
    if usable.any():
        metrics["rul_life_fraction_mae_pct_points"] = float(
            np.mean(np.abs(pred_pct[usable] - true_pct[usable]))
        )
        metrics["rul_life_fraction_mape_pct"] = float(
            np.mean(np.abs(pred_pct[usable] - true_pct[usable]) / true_pct[usable]) * 100.0
        )
    for stage, group in frame.groupby("health_stage"):
        metrics[f"rul_mape_stage{int(stage)}_pct"] = mape(
            group["rul_hours"].to_numpy(), group["rul_true_hours"].to_numpy()
        )
    return metrics, frame


# --------------------------------------------------------------------------
# I3 -- classification, staging, early detection
# --------------------------------------------------------------------------

def evaluate_classification(
    tft_preds: pd.DataFrame, test_frame: pd.DataFrame
) -> tuple[dict, pd.DataFrame, str, np.ndarray]:
    truth = test_frame.set_index(["run_id", "window_index"])
    joined = tft_preds.join(
        truth[["observable_class", "fault_class", "health_stage", "fault_detectable"]],
        on=["run_id", "window_index"],
        how="inner",
    )
    joined["observable_class"] = joined["observable_class"].astype(str)
    joined["fault_class"] = joined["fault_class"].astype(str)

    y_true = joined["observable_class"].to_numpy()
    y_pred = joined["pred_class"].to_numpy()

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=FAULT_CLASSES,
                                   average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "classification_n": int(len(joined)),
    }

    # The safety-critical error: a genuinely faulted window called healthy.
    # Ch.2.3's ethical analysis names this as the direct harm pathway, so it is
    # reported on its own rather than buried inside macro-F1.
    faulted = joined[joined["observable_class"] != "healthy"]
    if len(faulted):
        metrics["missed_fault_rate"] = float(
            (faulted["pred_class"] == "healthy").mean()
        )
    healthy = joined[joined["observable_class"] == "healthy"]
    if len(healthy):
        metrics["false_alarm_rate"] = float((healthy["pred_class"] != "healthy").mean())

    report = classification_report(
        y_true, y_pred, labels=FAULT_CLASSES, zero_division=0, digits=3
    )
    matrix = confusion_matrix(y_true, y_pred, labels=FAULT_CLASSES)
    return metrics, joined, report, matrix


def evaluate_early_detection(joined: pd.DataFrame) -> dict:
    """I3: is each fault correctly called no later than health stage 3?"""
    results = []
    for run_id, run in joined.groupby("run_id"):
        run = run.sort_values("window_index")
        true_class = run["fault_class"].iloc[0]
        if true_class == "healthy":
            continue
        correct = (run["pred_class"] == true_class).to_numpy()
        stages = run["health_stage"].to_numpy()

        caught_stage = None
        streak = 0
        for idx, is_correct in enumerate(correct):
            streak = streak + 1 if is_correct else 0
            if streak >= DETECTION_PERSISTENCE:
                caught_stage = int(stages[idx])
                break
        results.append(
            {
                "run_id": int(run_id),
                "fault_class": true_class,
                "caught_at_stage": caught_stage,
                "caught_by_stage3": caught_stage is not None and caught_stage <= 3,
            }
        )

    if not results:
        return {"caught_by_stage3_frac": float("nan"), "detection_runs": 0}
    frame = pd.DataFrame(results)
    caught = frame["caught_at_stage"].dropna()
    return {
        "caught_by_stage3_frac": float(frame["caught_by_stage3"].mean()),
        "detection_runs": int(len(frame)),
        "median_catch_stage": float(caught.median()) if len(caught) else float("nan"),
        "never_caught_runs": int(frame["caught_at_stage"].isna().sum()),
        "_per_run": frame.to_dict(orient="records"),
    }


def evaluate_staging(rul_frame: pd.DataFrame, tft_preds: pd.DataFrame,
                     calib: rul_mod.RULCalibration) -> dict:
    """I3: predicted health stage within +/-1 of the true stage."""
    if rul_frame.empty:
        return {"stage_within_1_frac": float("nan"), "stage_n": 0}
    lookup = tft_preds.set_index(["run_id", "window_index"])["pred_class"].to_dict()
    predicted, actual = [], []
    for record in rul_frame.itertuples(index=False):
        cls = lookup.get((record.run_id, record.window_index), "healthy")
        predicted.append(rul_mod.stage_from_hi(calib, record.hi_now, cls))
        actual.append(record.health_stage)
    predicted_arr = np.asarray(predicted)
    actual_arr = np.asarray(actual)
    error = np.abs(predicted_arr - actual_arr)
    return {
        "stage_within_1_frac": float((error <= 1).mean()),
        "stage_mae": float(error.mean()),
        "stage_exact_frac": float((error == 0).mean()),
        "stage_n": int(len(error)),
    }


# --------------------------------------------------------------------------
# Spec verdicts and reporting
# --------------------------------------------------------------------------

def verdict_for(spec: SpecTarget, metrics: dict) -> dict:
    measured = metrics.get(spec.metric, float("nan"))
    if not np.isfinite(measured):
        status = "NOT MEASURED"
    elif spec.direction == "max":
        status = "PASS" if measured <= spec.target else "FAIL"
    else:
        status = "PASS" if measured >= spec.target else "FAIL"
    return {
        "spec_id": spec.spec_id,
        "description": spec.description,
        "metric": spec.metric,
        "target": spec.target,
        "direction": spec.direction,
        "measured": float(measured),
        "status": status,
        "owner": spec.owner,
        "note": spec.note,
    }


def _fmt(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    if metric.endswith("_ms"):
        return f"{value:.1f} ms"
    if metric.endswith("_pct"):
        return f"{value:.1f} %"
    if metric.endswith("_frac") or metric in ("macro_f1",):
        return f"{value:.3f}"
    return f"{value:.3f}"


def write_report(metrics: dict, verdicts: list[dict], extras: dict) -> tuple:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    json_path = REPORT_DIR / f"spec_check_{stamp}.json"
    json_path.write_text(json.dumps(
        {"generated_utc": stamp, "metrics": metrics, "verdicts": verdicts,
         "environment": extras.get("environment", {}),
         "training": extras.get("training", {})},
        indent=2, default=float,
    ))

    lines: list[str] = []
    lines.append("# ICS AI testbench -- specification check\n")
    lines.append(f"Generated {stamp} (UTC)\n")
    lines.append(
        "> **These results are measured on SYNTHETIC data.** No fault-seeded or\n"
        "> run-to-failure data from the team's own rig exists yet. The numbers\n"
        "> below verify that the pipeline, the models and the latency path\n"
        "> behave as designed; they are NOT measurements of the system's\n"
        "> accuracy on a real bearing and must not be quoted as such. Models\n"
        "> are scored on held-out runs they never saw in training.\n"
    )

    lines.append("\n## Specification verdicts\n")
    lines.append("| Spec | Requirement | Target | Measured | Verdict |")
    lines.append("|---|---|---|---|---|")
    for verdict in verdicts:
        arrow = "<=" if verdict["direction"] == "max" else ">="
        lines.append(
            f"| {verdict['spec_id']} | {verdict['description']} | "
            f"{arrow} {_fmt(verdict['target'], verdict['metric'])} | "
            f"{_fmt(verdict['measured'], verdict['metric'])} | "
            f"**{verdict['status']}** |"
        )

    lines.append("\n### Notes on the verdicts above\n")
    for verdict in verdicts:
        if verdict["note"]:
            lines.append(f"- **{verdict['spec_id']}** -- {verdict['note']}")

    lines.append("\n## ICS1 -- RUL detail\n")
    lines.append(f"- Blended estimator MAPE: **{_fmt(metrics.get('rul_mape_pct', np.nan), 'x_pct')}**")
    lines.append(
        f"- Population-prior-only baseline MAPE: "
        f"{_fmt(metrics.get('rul_mape_prior_only_pct', np.nan), 'x_pct')} "
        f"(what the system would score with no N-HiTS forecast at all)"
    )
    lines.append(f"- Mean absolute error: {metrics.get('rul_mae_hours', float('nan')):.1f} h")
    lines.append(f"- Evaluated windows: {metrics.get('rul_n', 0):,}")

    floor = metrics.get("rul_information_floor_pct", float("nan"))
    if np.isfinite(floor):
        lines.append(
            f"\n**Information floor: {floor:.1f}% MAPE.** Across the faulted "
            f"test runs, total life varies by "
            f"{metrics.get('rul_lifetime_spread_ratio', float('nan')):.1f}x. "
            f"The health index reveals how far through its life a bearing is, "
            f"not how long that life is; since RUL = total_life x "
            f"(1 - fraction_consumed), the relative error in RUL can be no "
            f"better than the relative error in total life. No forecaster of "
            f"any quality beats this bound while RUL is expressed in absolute "
            f"hours. Read the measured MAPE against this floor, not against "
            f"zero."
        )
        lines.append(
            f"\n- Scale-invariant alternative -- percent-of-life-remaining "
            f"error: "
            f"{metrics.get('rul_life_fraction_mae_pct_points', float('nan')):.1f} "
            f"percentage points. This is the metric the external-datasets doc "
            f"recommends for cross-dataset work (risk 2), and it is not "
            f"subject to the floor above."
        )
    stage_keys = sorted(k for k in metrics if k.startswith("rul_mape_stage"))
    if stage_keys:
        lines.append("\nMAPE by true health stage:\n")
        lines.append("| Stage | MAPE |")
        lines.append("|---|---|")
        for key in stage_keys:
            stage = key.replace("rul_mape_stage", "").replace("_pct", "")
            lines.append(f"| {stage} | {_fmt(metrics[key], 'x_pct')} |")

    lines.append("\n## ICS2 / I3 -- classification detail\n")
    lines.append(f"- macro-F1: **{metrics.get('macro_f1', float('nan')):.3f}**")
    lines.append(f"- Accuracy: {metrics.get('accuracy', float('nan')):.3f}")
    lines.append(
        f"- Missed-fault rate (faulted window called healthy): "
        f"{metrics.get('missed_fault_rate', float('nan')):.3f}"
    )
    lines.append(
        f"- False-alarm rate (healthy window called faulted): "
        f"{metrics.get('false_alarm_rate', float('nan')):.3f}"
    )
    lines.append(
        f"- Faults caught by stage 3: "
        f"{metrics.get('caught_by_stage3_frac', float('nan')):.3f} "
        f"across {metrics.get('detection_runs', 0)} faulted test runs "
        f"(median catch at stage {metrics.get('median_catch_stage', float('nan')):.0f})"
    )

    if "classification_report" in extras:
        lines.append("\n```\n" + extras["classification_report"].rstrip() + "\n```\n")
    if "confusion_matrix" in extras:
        lines.append("Confusion matrix (rows = true, columns = predicted):\n")
        header = "| | " + " | ".join(FAULT_CLASSES) + " |"
        lines.append(header)
        lines.append("|---" * (len(FAULT_CLASSES) + 1) + "|")
        for name, row in zip(FAULT_CLASSES, extras["confusion_matrix"]):
            lines.append(f"| **{name}** | " + " | ".join(str(int(v)) for v in row) + " |")

    lines.append("\n## Latency detail\n")
    lines.append("Single-window inference, batch size 1, the path the deployed host runs.\n")
    lines.append("| Path | mean | p50 | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for label, title in (("tft", "TFT classification"), ("nhits", "N-HiTS forecast"),
                         ("ics_inference", "Both models (ICS leg of I1)")):
        if f"{label}_latency_p50_ms" in metrics:
            lines.append(
                f"| {title} | {metrics[f'{label}_latency_mean_ms']:.1f} ms "
                f"| {metrics[f'{label}_latency_p50_ms']:.1f} ms "
                f"| {metrics[f'{label}_latency_p95_ms']:.1f} ms "
                f"| {metrics[f'{label}_latency_max_ms']:.1f} ms |"
            )
    env = extras.get("environment", {})
    lines.append(
        f"\nMeasured on: {env.get('processor', 'unknown')} / "
        f"{env.get('platform', 'unknown')}, PyTorch {env.get('torch', '?')}, CPU.\n"
        "\n> The deployment target is still open between a laptop and a "
        "Raspberry Pi 5 (ICS layer doc, section 6). These timings are from the "
        "development machine and do not stand in for either candidate. Re-run "
        "this on the chosen hardware before quoting a latency figure.\n"
    )

    markdown_path = REPORT_DIR / f"spec_check_{stamp}.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    latest = REPORT_DIR / "latest.md"
    latest.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, json_path


def run(cfg: RunConfig) -> dict:
    print("=" * 74)
    print("ICS AI testbench -- specification check")
    print("=" * 74)

    engineered = pd.read_parquet(ENGINEERED_FILE)
    splits = load_splits()
    calib = rul_mod.RULCalibration.from_dict(json.loads(CALIB_FILE.read_text()))

    train_frame = data_mod.subset(engineered, splits["train"])
    val_frame = data_mod.subset(engineered, splits["val"])
    test_frame = data_mod.subset(engineered, splits["test"])
    print(f"  test set: {len(splits['test'])} runs, {len(test_frame):,} windows")

    nhits = NHiTS.load_from_checkpoint(NHITS_CKPT, map_location="cpu")
    tft = TemporalFusionTransformer.load_from_checkpoint(TFT_CKPT, map_location="cpu")

    _, nhits_sets = data_mod.build_nhits_datasets(
        train_frame, {"test": test_frame}, cfg.nhits
    )
    tft_train, tft_sets = data_mod.build_tft_datasets(
        train_frame, {"test": test_frame}, cfg.tft
    )
    order = data_mod.class_order(tft_train)

    print("\n[predict] running both models over the test runs...")
    nhits_preds = _predict_nhits(nhits, nhits_sets["test"], cfg.nhits.batch_size)
    tft_preds = _predict_tft(tft, tft_sets["test"], cfg.tft.batch_size, order)
    print(f"  N-HiTS forecasts: {len(nhits_preds):,}   "
          f"TFT classifications: {len(tft_preds):,}")

    metrics: dict = {}

    print("\n[ICS1] RUL accuracy")
    rul_metrics, rul_frame = evaluate_rul(
        nhits_preds, tft_preds, test_frame, calib, cfg.nhits.encoder_length
    )
    metrics.update(rul_metrics)
    print(f"  MAPE {rul_metrics.get('rul_mape_pct', float('nan')):.1f}% "
          f"(prior-only baseline {rul_metrics.get('rul_mape_prior_only_pct', float('nan')):.1f}%)")

    print("\n[I3] classification, staging, early detection")
    class_metrics, joined, report, matrix = evaluate_classification(tft_preds, test_frame)
    metrics.update(class_metrics)
    detection = evaluate_early_detection(joined)
    per_run = detection.pop("_per_run", [])
    metrics.update(detection)
    metrics.update(evaluate_staging(rul_frame, tft_preds, calib))
    print(f"  macro-F1 {class_metrics['macro_f1']:.3f}, "
          f"caught by stage 3 {detection.get('caught_by_stage3_frac', float('nan')):.2f}, "
          f"stage within 1 {metrics.get('stage_within_1_frac', float('nan')):.3f}")

    print("\n[ICS2 / I1] latency")
    metrics.update(measure_latency(tft, tft_sets["test"], cfg.latency_windows, "tft"))
    metrics.update(measure_latency(nhits, nhits_sets["test"], cfg.latency_windows, "nhits"))
    metrics["ics_inference_mean_ms"] = (
        metrics["tft_latency_mean_ms"] + metrics["nhits_latency_mean_ms"]
    )
    metrics["ics_inference_p50_ms"] = (
        metrics["tft_latency_p50_ms"] + metrics["nhits_latency_p50_ms"]
    )
    # Worst case, not a convolution of the two distributions: the host runs
    # both models on the same window, so the pessimistic sum is the honest
    # bound for a budget check.
    metrics["ics_inference_p95_ms"] = (
        metrics["tft_latency_p95_ms"] + metrics["nhits_latency_p95_ms"]
    )
    metrics["ics_inference_max_ms"] = (
        metrics["tft_latency_max_ms"] + metrics["nhits_latency_max_ms"]
    )
    print(f"  TFT p95 {metrics['tft_latency_p95_ms']:.1f} ms, "
          f"both models p95 {metrics['ics_inference_p95_ms']:.1f} ms")

    verdicts = [verdict_for(spec, metrics) for spec in SPEC_TARGETS]

    extras = {
        "classification_report": report,
        "confusion_matrix": matrix,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "training": json.loads(META_FILE.read_text()) if META_FILE.exists() else {},
    }
    markdown_path, json_path = write_report(metrics, verdicts, extras)

    print("\n" + "=" * 74)
    print("SPECIFICATION VERDICTS")
    print("=" * 74)
    for verdict in verdicts:
        arrow = "<=" if verdict["direction"] == "max" else ">="
        print(f"  {verdict['status']:<12} {verdict['spec_id']:<6} "
              f"{verdict['description']:<38} "
              f"{_fmt(verdict['measured'], verdict['metric'])} "
              f"(need {arrow} {_fmt(verdict['target'], verdict['metric'])})")
    print("\nReport: " + str(markdown_path))
    print("JSON:   " + str(json_path))

    if rul_frame is not None and not rul_frame.empty:
        rul_frame.to_csv(REPORT_DIR / "rul_predictions.csv", index=False)
    if per_run:
        pd.DataFrame(per_run).to_csv(REPORT_DIR / "detection_per_run.csv", index=False)

    return {"metrics": metrics, "verdicts": verdicts}
