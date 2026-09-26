"""Graph-free reference models on the same-hour node features.

Each baseline sees exactly the GNN's normalised inputs for one junction-hour (static
terrain + current rain + trailing rain sums) but no graph (no upstream junctions) and no
memory (no GRU state):

* ``logreg`` - :class:`~sklearn.linear_model.LogisticRegression` (linear in the features);
* ``hist_gbdt`` - :class:`~sklearn.ensemble.HistGradientBoostingClassifier` (non-linear:
  it can represent threshold rules such as "floods when the 6 h rain exceeds x at a low
  junction", which a linear model cannot).

The gap between the GNN and the STRONGEST baseline - not the linear one - is what message
passing and temporal state add on top of a flexible per-junction model. Both are trained
on one seeded, class-balanced subsample of the TRAIN scored node-steps (all flooded ones up
to half the budget, the rest random dry ones; ``class_weight="balanced"``). Exactly like the
GNN, their probabilities are Platt-calibrated and the alert threshold chosen on the
VALIDATION split, then every metric is reported on validation and, when given, on the
held-out TEST split.
"""

from __future__ import annotations

import math
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.data_pipeline.dataset import FloodSequenceDataset
from src.training.calibration import apply_calibration, fit_platt
from src.training.metrics import best_threshold, evaluate_predictions
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

__all__ = ["BASELINE_DEFAULTS", "BASELINE_NAMES", "BaselineConfig", "logistic_baseline", "run_baselines"]

BASELINE_DEFAULTS: dict[str, Any] = {
    "enabled": True, "max_samples": 200_000, "max_train_windows": 400,
    "hist_gbdt": {"enabled": True, "max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
                  "min_samples_leaf": 20},
}
BASELINE_NAMES: tuple[str, ...] = ("logreg", "hist_gbdt")
_MAX_ITER = 1000
_PROB_EPS = 1e-7
_NOTE = ("graph-free, memory-free model on the same per-junction-hour input features; Platt-calibrated and "
         "threshold-tuned on validation like the GNN")


@dataclass(frozen=True)
class BaselineConfig:
    """Sampling and model settings of :func:`run_baselines`."""

    max_samples: int = 200_000
    max_train_windows: int = 400
    seed: int = 42
    threshold_metric: str = "f2"
    n_bins: int = 10
    models: tuple[str, ...] = BASELINE_NAMES
    gbdt_params: Mapping[str, Any] = field(default_factory=lambda: dict(
        {k: v for k, v in BASELINE_DEFAULTS["hist_gbdt"].items() if k != "enabled"}))
    num_threads: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_samples", "max_train_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"training.baseline.{name} must be an integer >= 2, got {value!r}")
        unknown = [m for m in self.models if m not in BASELINE_NAMES]
        if unknown:
            raise ValueError(f"Unknown baseline model(s) {unknown}; expected a subset of {list(BASELINE_NAMES)}")


# --------------------------------------------------------------------------- data


def _scored_rows(dataset: FloodSequenceDataset, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Scored (non-warm-up) node-steps of window ``index`` → ``(x [M, F] float32, y [M] uint8)``."""
    item = dataset[index]
    mask = item["mask"].numpy()
    x = item["x"].numpy()[mask]
    y = item["y"].numpy()[mask]
    return x.reshape(-1, x.shape[-1]), (y.reshape(-1) > 0.5).astype(np.uint8)


def _training_windows(dataset: FloodSequenceDataset, max_windows: int, rng: np.random.Generator) -> np.ndarray:
    """All flood windows (up to half the budget) plus random dry windows, sorted."""
    flood = np.flatnonzero(dataset.window_has_flood)
    dry = np.flatnonzero(~dataset.window_has_flood)
    n_flood = min(flood.size, max(1, max_windows // 2))
    chosen_flood = rng.choice(flood, n_flood, replace=False) if n_flood < flood.size else flood
    n_dry = min(dry.size, max_windows - chosen_flood.size)
    chosen_dry = rng.choice(dry, n_dry, replace=False) if n_dry > 0 else dry[:0]
    return np.sort(np.concatenate([chosen_flood, chosen_dry]))


def _training_sample(dataset: FloodSequenceDataset, windows: np.ndarray, max_samples: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """At most ``max_samples // 2`` positives (all, if fewer) and an equal per-window share of negatives."""
    neg_per_window = max(1, max_samples // (2 * max(windows.size, 1)))
    xs, ys = [], []
    for index in windows:
        x, y = _scored_rows(dataset, int(index))
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if neg.size > neg_per_window:
            neg = rng.choice(neg, neg_per_window, replace=False)
        keep = np.concatenate([pos, neg])
        xs.append(x[keep])
        ys.append(y[keep])
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    positives = np.flatnonzero(y_all == 1)
    if positives.size > max_samples // 2:
        dropped = rng.choice(positives, positives.size - max_samples // 2, replace=False)
        keep = np.setdiff1d(np.arange(y_all.size), dropped, assume_unique=True)
        x_all, y_all = x_all[keep], y_all[keep]
    return x_all, y_all


def _threads(num_threads: int | None) -> Any:
    if num_threads is None:
        return nullcontext()
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=int(num_threads))


# --------------------------------------------------------------------------- models


def _fit_logreg(x: np.ndarray, y: np.ndarray, config: BaselineConfig) -> tuple[Any, dict[str, Any]]:
    model = LogisticRegression(class_weight="balanced", max_iter=_MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        LOGGER.warning("Logistic baseline did not fully converge in %d iterations", _MAX_ITER)
    return model, {"intercept": float(model.intercept_[0])}


def _fit_gbdt(x: np.ndarray, y: np.ndarray, config: BaselineConfig) -> tuple[Any, dict[str, Any]]:
    params = dict(config.gbdt_params)
    model = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=10, random_state=int(config.seed), **params)
    model.fit(x, y)
    return model, {"n_iter": int(model.n_iter_), "params": params}


_FITTERS = {"logreg": _fit_logreg, "hist_gbdt": _fit_gbdt}


def _score(models: Mapping[str, Any], dataset: FloodSequenceDataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Labels and every model's probabilities over all scored node-steps (features built once per window)."""
    labels, probs = [], {name: [] for name in models}
    for index in range(len(dataset)):
        x, y = _scored_rows(dataset, index)
        labels.append(y)
        for name, model in models.items():
            probs[name].append(model.predict_proba(x)[:, 1])
    return np.concatenate(labels), {name: np.concatenate(p) for name, p in probs.items()}


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64), _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _evaluate_model(name: str, val: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray] | None,
                    config: BaselineConfig) -> dict[str, Any]:
    """Platt + threshold on validation; metrics on validation and test."""
    (y_val, p_val) = val
    calibration = fit_platt(_logit(p_val), y_val)
    cal_val = apply_calibration(_logit(p_val), calibration)
    threshold, score = best_threshold(y_val, cal_val, metric=config.threshold_metric)
    out = {"calibration": calibration, "threshold": threshold, "threshold_metric": config.threshold_metric,
           "threshold_score": score,
           "validation": evaluate_predictions(y_val, cal_val, threshold, n_bins=config.n_bins,
                                              threshold_metric=config.threshold_metric), "test": None}
    if test is not None:
        cal_test = apply_calibration(_logit(test[1]), calibration)
        out["test"] = evaluate_predictions(test[0], cal_test, threshold, n_bins=config.n_bins,
                                           threshold_metric=config.threshold_metric)
    LOGGER.info("Baseline %s: val PR-AUC %s%s, %s %.4f @ %.4g", name, _fmt(out["validation"]["pr_auc"]),
                "" if out["test"] is None else f", test PR-AUC {_fmt(out['test']['pr_auc'])}", config.threshold_metric,
                score, threshold)
    return out


def _unavailable(reason: str, models: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {name: {"status": "unavailable", "reason": reason} for name in models}


def run_baselines(train_ds: FloodSequenceDataset, val_ds: FloodSequenceDataset,
                  test_ds: FloodSequenceDataset | None = None,
                  config: BaselineConfig | None = None) -> dict[str, dict[str, Any]]:
    """Fit every baseline in ``config.models`` on TRAIN; calibrate / threshold on VAL; score VAL (+ TEST).

    Returns ``{name: {"status": "ok", "validation": {...}, "test": {...} | None, "threshold",
    "calibration", "n_train_samples", ...}}``. A model that cannot be trained gets
    ``{"status": "unavailable" | "failed", "reason": ...}``; the others are unaffected.
    """
    config = config or BaselineConfig()
    if len(train_ds) == 0 or len(val_ds) == 0:
        return _unavailable("a dataset split has no windows", config.models)
    if not train_ds.window_has_flood.any():
        LOGGER.warning("Baselines skipped: the training split has no flooded node-steps")
        return _unavailable("the training split has no flooded node-steps", config.models)
    clock = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    windows = _training_windows(train_ds, int(config.max_train_windows), rng)
    x_train, y_train = _training_sample(train_ds, windows, int(config.max_samples), rng)
    sample = {"n_train_samples": int(y_train.size), "n_train_pos": int(y_train.sum()),
              "n_train_windows": int(windows.size), "note": _NOTE}
    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    with _threads(config.num_threads):
        for name in config.models:
            start = time.perf_counter()
            try:
                fitted[name], extra = _FITTERS[name](x_train, y_train, config)
            except (ValueError, RuntimeError, MemoryError) as exc:  # one failing model never hides the others
                LOGGER.warning("Baseline %s failed to train (%s: %s)", name, type(exc).__name__, exc)
                results[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                continue
            results[name] = {"status": "ok", **sample, **extra, "fit_time_s": round(time.perf_counter() - start, 3)}
        if not fitted:
            return results
        y_val, p_val = _score(fitted, val_ds)
        scored_test = _score(fitted, test_ds) if test_ds is not None and len(test_ds) else None
    for name in fitted:
        test = None if scored_test is None else (scored_test[0], scored_test[1][name])
        results[name].update(_evaluate_model(name, (y_val, p_val[name]), test, config))
    if "logreg" in fitted:
        names = list(train_ds.feature_names)
        results["logreg"]["coefficients"] = {n: float(c) for n, c in zip(names, fitted["logreg"].coef_[0])}
    LOGGER.info("Baselines %s done in %.1f s (%d training samples)", list(fitted), time.perf_counter() - clock,
                y_train.size)
    return results


def logistic_baseline(
    train_ds: FloodSequenceDataset,
    val_ds: FloodSequenceDataset,
    *,
    max_samples: int = 200_000,
    max_train_windows: int = 400,
    seed: int = 42,
    threshold_metric: str = "f2",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Backward-compatible logistic baseline on VAL: ``{"status": "ok", <val metrics>, ...}``.

    ``status`` is ``"unavailable"`` (with a ``reason``) when the training split has no
    flooded node-steps or either split is empty.
    """
    config = BaselineConfig(max_samples=max_samples, max_train_windows=max_train_windows, seed=seed,
                            threshold_metric=threshold_metric, n_bins=n_bins, models=("logreg",))
    result = run_baselines(train_ds, val_ds, None, config)["logreg"]
    if result.get("status") != "ok":
        return result
    flat = {k: v for k, v in result.items() if k not in ("validation", "test")}
    return {**flat, **result["validation"], "status": "ok", "threshold": result["threshold"]}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"
