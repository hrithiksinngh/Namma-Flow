"""Scoring windows with a trained model and turning scores into per-split reports.

:func:`evaluate` runs the model over a loader and returns flat, aligned arrays over the
scored (non-warm-up) node-steps. :func:`split_report` turns them into the metrics stored in
``metrics.json`` for one split at a FIXED calibration and threshold (both chosen on the
validation split by the trainer). :func:`strongest_baseline` picks the baseline the app's
headline delta is measured against.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.calibration import apply_calibration
from src.training.metrics import binary_metrics, evaluate_predictions, ranking_metrics, reliability_curve

__all__ = ["HEADLINE_KEYS", "evaluate", "per_step_metrics", "sigmoid", "split_report", "strongest_baseline"]

# Keys every split report carries (the headline block of metrics.json repeats them).
HEADLINE_KEYS: tuple[str, ...] = ("pr_auc", "roc_auc", "f2", "precision", "recall", "csi", "brier", "ece", "log_loss",
                                  "threshold", "reliability", "per_step", "n", "n_pos", "pos_rate")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.from_numpy(np.asarray(logits, dtype=np.float64))).numpy()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str | torch.device,
             temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every window of ``loader`` on its non-warm-up steps.

    Returns flat, aligned arrays over (window, step, node): ``y_true`` (uint8), logits
    divided by ``temperature`` (float32) and the step index within the window (int32,
    ``>= warmup_steps``). The model's train/eval mode is restored afterwards.
    """
    if isinstance(temperature, bool) or not (isinstance(temperature, (int, float)) and 0 < temperature < math.inf):
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    labels, logits, steps = [], [], []
    try:
        for batch in loader:
            out, _ = model.forward_sequence(batch["x"].to(device), batch["edge_index"].to(device),
                                            batch["edge_attr"].to(device))
            mask = torch.as_tensor(batch["mask"], dtype=torch.bool).cpu()
            scored = out.detach().float().cpu()[mask]
            labels.append((batch["y"][mask] > 0.5).numpy().astype(np.uint8).reshape(-1))
            logits.append((scored / float(temperature)).numpy().astype(np.float32).reshape(-1))
            steps.append(np.repeat(np.flatnonzero(mask.numpy()), scored.shape[1]).astype(np.int32))
    finally:
        model.train(was_training)
    if not labels:
        return np.zeros(0, np.uint8), np.zeros(0, np.float32), np.zeros(0, np.int32)
    return np.concatenate(labels), np.concatenate(logits), np.concatenate(steps)


def per_step_metrics(y: np.ndarray, probs: np.ndarray, steps: np.ndarray, threshold: float) -> list[dict]:
    """Metrics per hour-of-window (how skill evolves as the GRU state builds up)."""
    rows = []
    for step in np.unique(steps):
        sel = steps == step
        ranking = ranking_metrics(y[sel], probs[sel])
        binary = binary_metrics(y[sel], probs[sel], threshold)
        rows.append({"step": int(step), "n": ranking["n"], "n_pos": ranking["n_pos"], "pr_auc": ranking["pr_auc"],
                     "roc_auc": ranking["roc_auc"], "brier": ranking["brier"],
                     **{k: binary[k] for k in ("precision", "recall", "f2", "csi")}})
    return rows


def split_report(arrays: tuple[np.ndarray, np.ndarray, np.ndarray], calibration: Mapping[str, Any],
                 threshold: float, *, n_bins: int, threshold_metric: str) -> dict[str, Any]:
    """Every metric of one split at the given (validation-chosen) calibration and threshold.

    Includes :data:`HEADLINE_KEYS`, the full :func:`evaluate_predictions` output, and the
    uncalibrated Brier / log loss / ECE for comparison.
    """
    y, logits, steps = arrays
    probs = apply_calibration(np.asarray(logits, dtype=np.float64), calibration)
    raw = sigmoid(logits)
    raw_ranking = ranking_metrics(y, raw)
    report = evaluate_predictions(y, probs, threshold, n_bins=n_bins, threshold_metric=threshold_metric)
    report["per_step"] = per_step_metrics(y, probs, steps, threshold)
    report["mean_probability"] = float(probs.mean()) if probs.size else None
    report["uncalibrated"] = {"brier": raw_ranking["brier"], "log_loss": raw_ranking["log_loss"],
                              "ece": reliability_curve(y, raw, n_bins=n_bins)["ece"],
                              "mean_probability": float(raw.mean()) if raw.size else None}
    return report


def strongest_baseline(baselines: Mapping[str, Any], split: str, gnn_pr_auc: float | None) -> dict[str, Any]:
    """The baseline with the highest PR-AUC on ``split`` ("test" or "validation").

    Always a dict: ``{"name", "pr_auc", "split", "gnn_pr_auc", "delta_pr_auc"}`` with
    ``None`` values when no baseline produced a PR-AUC on that split.
    """
    best_name, best_value = None, None
    for name, result in baselines.items():
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            continue
        value = (result.get(split) or {}).get("pr_auc")
        if value is not None and (best_value is None or value > best_value):
            best_name, best_value = name, float(value)
    delta = None if best_value is None or gnn_pr_auc is None else float(gnn_pr_auc) - best_value
    return {"name": best_name, "pr_auc": best_value, "split": split, "gnn_pr_auc": gnn_pr_auc, "delta_pr_auc": delta}
