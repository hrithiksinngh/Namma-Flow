"""Evaluation metrics for rare-event (flood / no-flood) junction predictions.

All functions take flat or n-D ``y_true`` (0/1 labels) and ``y_prob`` (probabilities in
``[0, 1]``) as numpy arrays, torch tensors or sequences, validate them once, and return
plain Python ``int`` / ``float`` / ``None`` values (JSON- and ``torch.load(weights_only=True)``
safe) so results can be stored in checkpoints and ``metrics.json`` directly.

* :func:`binary_metrics` — confusion matrix and threshold metrics (precision, recall, F1,
  F2, CSI, accuracy, ...). A prediction is positive when ``prob >= threshold``.
* :func:`ranking_metrics` — threshold-free PR-AUC (average precision), ROC-AUC (both
  ``None`` when only one class is present), Brier score and log loss.
* :func:`best_threshold` — threshold maximising F-beta / CSI / MCC / Youden's J (candidates adapt
  to the probability scale, so calibrated rare-event probabilities of ~1e-5 are handled).
* :func:`reliability_curve` — calibration curve with expected / maximum calibration error.
* :func:`evaluate_predictions` — all of the above merged into one dict.

Zero-division conventions follow scikit-learn's ``zero_division=0``: an undefined
precision, recall or F-score is ``0.0``, never NaN.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

THRESHOLD_METRICS: tuple[str, ...] = ("f1", "f2", "f0.5", "csi", "mcc", "youden")
_FBETA = {"f1": 1.0, "f2": 2.0, "f0.5": 0.5}
LOG_LOSS_EPS = 1e-15
SINGLE_CLASS_THRESHOLD = 0.5
# Geometric steps below 1 % (calibrated rare-event probabilities are small), then 1 % steps.
DEFAULT_THRESHOLD_GRID: np.ndarray = np.unique(
    np.round(np.concatenate([np.geomspace(1e-4, 1e-2, 21)[:-1], np.linspace(0.01, 0.99, 99)]), 8)
)
DEFAULT_THRESHOLD_GRID.setflags(write=False)


# --------------------------------------------------------------------------- input validation


def _flat(values: Any, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    try:
        return np.asarray(values).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like, got {type(values).__name__}: {exc}") from exc


def _labels(y: np.ndarray) -> np.ndarray:
    """0/1 labels (bool, integer or float dtype) → bool array; anything else raises."""
    if y.dtype == bool:
        return y
    if y.dtype.kind in "ui":
        if y.size and (y.min() < 0 or y.max() > 1):
            raise ValueError(f"y_true must contain only 0 or 1 labels, got values in [{y.min()}, {y.max()}]")
        return y.astype(bool)
    if y.dtype.kind == "f":
        valid = (y == 0) | (y == 1)
        if not valid.all():
            bad = y[~valid]
            raise ValueError(f"y_true must contain only 0 or 1 labels; {bad.size} other values (e.g. {bad[0]!r})")
        return y == 1
    raise ValueError(f"y_true must contain only 0 or 1 labels, got dtype {y.dtype}")


def _probabilities(p: np.ndarray) -> np.ndarray:
    try:
        probs = p.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"y_prob must be numeric probabilities, got dtype {p.dtype}") from exc
    if probs.size == 0:
        return probs
    if not np.isfinite(probs).all():
        raise ValueError(f"y_prob must be finite; found {int((~np.isfinite(probs)).sum())} NaN/inf values")
    low, high = float(probs.min()), float(probs.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"y_prob must lie in [0, 1] (probabilities, not logits); got range [{low:.4g}, {high:.4g}]")
    return probs


def check_inputs(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten ``(y_true, y_prob)`` → ``(bool labels, float64 probabilities)``."""
    y, p = _flat(y_true, "y_true"), _flat(y_prob, "y_prob")
    if y.size != p.size:
        raise ValueError(f"y_true and y_prob must have the same number of elements, got {y.size} and {p.size}")
    return _labels(y), _probabilities(p)


def _check_threshold(threshold: Any) -> float:
    ok = isinstance(threshold, (int, float, np.integer, np.floating)) and not isinstance(threshold, (bool, np.bool_))
    if not ok or not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be a number in [0, 1], got {threshold!r}")
    return float(threshold)


def _check_metric(metric: Any) -> str:
    name = str(metric).strip().lower()
    if name not in THRESHOLD_METRICS:
        raise ValueError(f"Unknown threshold metric {metric!r}; expected one of {THRESHOLD_METRICS}")
    return name


# --------------------------------------------------------------------------- scores from counts


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def fbeta_from_counts(tp: float, fp: float, fn: float, beta: float) -> float:
    """``F_beta = (1 + b^2) tp / ((1 + b^2) tp + b^2 fn + fp)``; 0 when undefined."""
    if not beta > 0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    b2 = beta * beta
    return _ratio((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)


def _score_curve(metric: str, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Vectorised threshold metric over arrays of confusion-matrix counts (float64)."""
    tp, fp, fn, tn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn, tn))
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric in _FBETA:
            b2 = _FBETA[metric] ** 2
            score = (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)
        elif metric == "csi":
            score = tp / (tp + fp + fn)
        elif metric == "mcc":
            score = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        else:  # youden
            score = tp / (tp + fn) - fp / (fp + tn)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- public metrics


def _binary(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probs >= threshold
    n = int(labels.size)
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted)) - tp
    n_pos = int(np.count_nonzero(labels))
    fn = n_pos - tp
    tn = n - tp - fp - fn
    return {
        "threshold": threshold, "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "f1": fbeta_from_counts(tp, fp, fn, 1.0), "f2": fbeta_from_counts(tp, fp, fn, 2.0),
        "csi": _ratio(tp, tp + fp + fn), "accuracy": _ratio(tp + tn, n),
        "specificity": _ratio(tn, tn + fp), "pos_rate": _ratio(n_pos, n), "predicted_pos_rate": _ratio(tp + fp, n),
    }


def binary_metrics(y_true: Any, y_prob: Any, threshold: float) -> dict[str, Any]:
    """Confusion matrix and threshold metrics at ``threshold`` (positive when ``prob >= threshold``).

    Keys: ``threshold, n, tp, fp, fn, tn, precision, recall, f1, f2, csi`` (critical success
    index), ``accuracy, specificity, pos_rate, predicted_pos_rate``.
    """
    threshold = _check_threshold(threshold)
    labels, probs = check_inputs(y_true, y_prob)
    return _binary(labels, probs, threshold)


def _ranking(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    n, n_pos = int(labels.size), int(np.count_nonzero(labels))
    out: dict[str, Any] = {"n": n, "n_pos": n_pos, "pr_auc": None, "roc_auc": None, "brier": None, "log_loss": None}
    if n == 0:
        return out
    target = labels.astype(np.float64)
    out["brier"] = float(np.mean((probs - target) ** 2))
    clipped = np.clip(probs, LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
    out["log_loss"] = float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)))
    if 0 < n_pos < n:
        y_int = labels.astype(np.uint8)
        out["pr_auc"] = float(average_precision_score(y_int, probs))
        out["roc_auc"] = float(roc_auc_score(y_int, probs))
    return out


def ranking_metrics(y_true: Any, y_prob: Any) -> dict[str, Any]:
    """PR-AUC (average precision), ROC-AUC, Brier score and log loss.

    PR-AUC and ROC-AUC are ``None`` when ``y_true`` holds a single class (undefined); every
    value is ``None`` for empty input.
    """
    return _ranking(*check_inputs(y_true, y_prob))


def _candidates(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Default thresholds: the fixed grid plus every distinct positive-class probability.

    F-beta and CSI are maximised at one of the positives' scores (lowering the threshold
    between two of them only adds false positives), so including them makes the search
    exact and adaptive: calibrated rare-event probabilities can sit far below any fixed grid.
    """
    return np.union1d(DEFAULT_THRESHOLD_GRID, probs[labels])


def _check_grid(grid: Any) -> np.ndarray | None:
    if grid is None:
        return None
    values = _flat(grid, "grid")
    try:
        values = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold grid must be numeric: {exc}") from exc
    if values.size == 0 or not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("threshold grid must be a non-empty list of finite thresholds in [0, 1]")
    return np.unique(values)


def _best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str,
                    grid: np.ndarray | None) -> tuple[float, float]:
    n_pos = int(np.count_nonzero(labels))
    if n_pos == 0 or n_pos == labels.size:
        return SINGLE_CLASS_THRESHOLD, 0.0
    grid = _candidates(labels, probs) if grid is None else grid
    pos, neg = np.sort(probs[labels]), np.sort(probs[~labels])
    tp = pos.size - np.searchsorted(pos, grid, side="left")  # positives with prob >= threshold
    fp = neg.size - np.searchsorted(neg, grid, side="left")
    scores = _score_curve(metric, tp, fp, pos.size - tp, neg.size - fp)
    best = float(scores.max())
    tied = np.flatnonzero(scores >= best - 1e-12)
    return float(grid[tied[tied.size // 2]]), best  # middle of the plateau: most robust choice


def best_threshold(y_true: Any, y_prob: Any, metric: str = "f2", grid: Any = None) -> tuple[float, float]:
    """``(threshold, score)`` maximising ``metric`` over candidate thresholds.

    The default candidates are :data:`DEFAULT_THRESHOLD_GRID` plus every distinct
    positive-class probability, which makes F-beta / CSI optimisation exact at any
    probability scale; pass ``grid`` to restrict the search. ``metric`` is one of
    :data:`THRESHOLD_METRICS`. Ties resolve to the middle of the tied thresholds. A
    single-class ``y_true`` (the metric is degenerate) returns ``(0.5, 0.0)``. Runs in
    ``O((n + g) log n)``, so millions of node-hours are cheap.
    """
    name = _check_metric(metric)
    thresholds = _check_grid(grid)
    labels, probs = check_inputs(y_true, y_prob)
    return _best_threshold(labels, probs, name, thresholds)


def _reliability(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = [float(c) for c in (edges[:-1] + edges[1:]) / 2.0]
    index = np.minimum((probs * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    sum_prob = np.bincount(index, weights=probs, minlength=n_bins)
    sum_pos = np.bincount(index, weights=labels.astype(np.float64), minlength=n_bins)
    filled = counts > 0
    mean_pred = np.divide(sum_prob, counts, out=np.zeros(n_bins), where=filled)
    observed = np.divide(sum_pos, counts, out=np.zeros(n_bins), where=filled)
    gaps = np.abs(observed - mean_pred)
    total = int(counts.sum())
    return {
        "n_bins": n_bins, "bin_edges": [float(e) for e in edges], "bin_centers": centers,
        "mean_predicted": [float(v) if f else None for v, f in zip(mean_pred, filled)],
        "observed_freq": [float(v) if f else None for v, f in zip(observed, filled)],
        "counts": [int(c) for c in counts],
        "ece": float((counts * gaps).sum() / total) if total else None,
        "mce": float(gaps[filled].max()) if total else None,
    }


def reliability_curve(y_true: Any, y_prob: Any, n_bins: int = 10) -> dict[str, Any]:
    """Calibration curve over ``n_bins`` equal-width probability bins.

    Returns ``bin_edges``, ``bin_centers``, per-bin ``mean_predicted``, ``observed_freq``
    (``None`` for empty bins) and ``counts``, plus ``ece`` (count-weighted mean
    ``|observed - predicted|``) and ``mce`` (worst non-empty bin); ``None`` for empty input.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    labels, probs = check_inputs(y_true, y_prob)
    return _reliability(labels, probs, int(n_bins))


def evaluate_predictions(
    y_true: Any, y_prob: Any, threshold: float, *, n_bins: int = 10, threshold_metric: str = "f2"
) -> dict[str, Any]:
    """Every metric at once: binary metrics at ``threshold``, ranking metrics, ``ece``,
    ``reliability`` (curve dict) and ``best_threshold`` (``{"metric", "threshold", "score"}``)."""
    threshold = _check_threshold(threshold)
    metric = _check_metric(threshold_metric)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    labels, probs = check_inputs(y_true, y_prob)
    reliability = _reliability(labels, probs, int(n_bins))
    best_thr, best_score = _best_threshold(labels, probs, metric, None)
    return {
        **_binary(labels, probs, threshold),
        **_ranking(labels, probs),
        "ece": reliability["ece"],
        "reliability": reliability,
        "best_threshold": {"metric": metric, "threshold": best_thr, "score": best_score},
    }
