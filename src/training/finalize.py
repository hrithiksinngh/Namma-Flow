"""Finalisation of a training run: calibrate, evaluate, publish.

1. reload this run's per-epoch best weights from ``best_candidate.pt`` (the last weights
   when no epoch improved);
2. fit the probability calibration (Platt by default) on the VALIDATION logits and choose
   the alert threshold on the calibrated VALIDATION probabilities (``threshold_metric``);
3. report every metric on VALIDATION and on the held-out TEST split (when present) at that
   fixed calibration and threshold - the headline block of ``metrics.json`` is the TEST
   split (``evaluation_split: "test"``), because validation also chose the epoch, the
   calibration and the threshold and is therefore optimistic;
4. fit the graph-free baselines (logistic, HistGradientBoosting) with the same protocol;
5. measure the areal-only skill (:mod:`src.training.areal_skill`, X6: the same split scored
   when only the corridor-average rain is known) when ``training.areal_skill.enabled``;
6. publish ``best.pt`` (format v2 with ``calibration``, ``threshold``, ``metrics``, the full
   run ``history``, ``finalized_utc`` and a fresh ``created_utc`` / ``published_id``: a
   re-finalised model is a new identity, F2-04), ``metrics.json``, ``training_history.csv``
   (staged during training, F2-03) and ``areal_skill.json`` together
   (:func:`~src.training.reports.publish_final`). Nothing before this step touches the
   published files, so a crashed or interrupted run never replaces the previous model.

Steps 2-4 always use EVERY validation window, even when ``training.max_val_windows``
subsampled the per-epoch validation (F2-02).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data_pipeline.dataset import make_loader
from src.models.stgcn import count_parameters
from src.training import areal_skill as areal_module
from src.training.baseline import BaselineConfig, run_baselines
from src.training.calibration import apply_calibration, fit_calibration, temperature_of
from src.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    load_checkpoint,
    published_id,
    to_builtin,
)
from src.training.evaluation import evaluate, sigmoid, split_report, strongest_baseline
from src.training.metrics import best_threshold, ranking_metrics
from src.training.reports import HISTORY_NAME, discard_partial_history, history_csv, publish_final
from src.training.run_state import LoopState, RunContext, TrainingError, TrainResult, fmt, make_checkpoint, now_utc
from src.training.splits import Splits
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

__all__ = ["finalize", "load_candidate"]

HEADLINE_NOTE = ("Headline metrics are on the held-out TEST split at the calibration and alert threshold chosen on "
                 "the VALIDATION split (which also selected the epoch). The model is fed the teacher's own stochastic "
                 "junction rain field (the one that produced the labels), so they measure emulation of the teacher "
                 "given that field - not forecast skill: with only corridor-average rain (any forecast or design "
                 "storm) see areal_skill / areal_skill.json.")
VALIDATION_NOTE = ("No test split: headline metrics are on the VALIDATION split, which also selected the epoch, the "
                   "calibration and the threshold (optimistic).")


def load_candidate(run: RunContext, model: nn.Module) -> dict[str, Any] | None:
    """Load this run's ``best_candidate.pt`` weights into ``model``; None (current weights kept) otherwise."""
    path = run.paths.candidate
    if path.exists():
        try:
            candidate = load_checkpoint(path)
        except CheckpointError as exc:
            LOGGER.warning("%s is unreadable (%s); finalising the current weights instead", path, exc)
            return None
        if candidate.get("run_id") == run.meta["run_id"]:
            model.load_state_dict(candidate["model_state"])
            return candidate
        LOGGER.warning("%s belongs to an earlier run; it is ignored (and replaced by this run's best)", path)
    LOGGER.warning("No epoch improved %s; finalising the last weights as best.pt", run.monitor.name)
    return None


def _score(run: RunContext, model: nn.Module, loader: DataLoader, split: str) -> tuple[np.ndarray, ...]:
    arrays = evaluate(model, loader, run.device)
    finite = np.isfinite(arrays[1])
    if not finite.all():
        raise TrainingError(f"The best model produces {int((~finite).sum())} non-finite {split} logits; "
                            "lower model.learning_rate and retrain")
    return arrays


def _baseline_config(run: RunContext) -> BaselineConfig:
    s = run.settings
    models = ("logreg", "hist_gbdt") if s.gbdt_enabled else ("logreg",)
    return BaselineConfig(max_samples=s.baseline_max_samples, max_train_windows=s.baseline_max_train_windows,
                          seed=s.seed, threshold_metric=s.threshold_metric, n_bins=s.reliability_bins, models=models,
                          gbdt_params={"max_iter": s.gbdt_max_iter, "learning_rate": s.gbdt_learning_rate,
                                       "max_leaf_nodes": s.gbdt_max_leaf_nodes,
                                       "min_samples_leaf": s.gbdt_min_samples_leaf},
                          num_threads=s.num_threads)


def _baselines(run: RunContext, splits: Splits) -> dict[str, dict[str, Any]]:
    names = ("logreg", "hist_gbdt")
    if not run.settings.baseline_enabled:
        return {name: {"status": "disabled"} for name in names}
    try:
        results = run_baselines(splits.train, splits.val, splits.test, _baseline_config(run))
    except (ValueError, RuntimeError, MemoryError) as exc:  # auxiliary: never fail the run
        LOGGER.warning("Baselines failed (%s: %s); metrics.json will not include them", type(exc).__name__, exc)
        return {name: {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"} for name in names}
    return {name: results.get(name, {"status": "disabled"}) for name in names}


def _legacy_logreg(baselines: Mapping[str, Any], split: str) -> dict[str, Any]:
    """``baseline_logreg`` (pre-v2 consumers): the logistic baseline's metrics on the headline split."""
    result = baselines.get("logreg") or {"status": "disabled"}
    if result.get("status") != "ok":
        return dict(result)
    flat = {k: v for k, v in result.items() if k not in ("validation", "test")}
    return {**flat, **(result.get(split) or {}), "status": "ok", "split": split, "threshold": result["threshold"]}


def _calibration_summary(calibration: Mapping[str, Any], arrays: tuple[np.ndarray, ...]) -> dict[str, Any]:
    y, logits, _ = arrays
    before = ranking_metrics(y, sigmoid(logits))
    after = ranking_metrics(y, apply_calibration(np.asarray(logits, dtype=np.float64), calibration))
    return {**calibration, "fitted_on": "validation", "n": int(y.size),
            "base_rate": float(y.mean()) if y.size else None,
            "log_loss_uncalibrated": before["log_loss"], "log_loss_calibrated": after["log_loss"]}


def _run_summary(run: RunContext, model: nn.Module, state: LoopState, started: float) -> dict[str, Any]:
    s = run.settings
    best_record = next((r for r in state.history if r.get("epoch") == state.best_epoch), {})
    epoch_times = [r["epoch_time_s"] for r in state.history if r.get("epoch_time_s") is not None]
    return {
        "status": "early_stopped" if state.early_stopped else "completed",
        "val_loss": best_record.get("val_loss"),
        "best_epoch": state.best_epoch, "epochs_run": state.epoch, "epochs_target": s.epochs,
        "monitor": run.monitor.name, "best_score": state.best_score, "dataset": run.dataset_info,
        "model": {"architecture": model.architecture, "n_parameters": count_parameters(model),
                  "config": model.get_config()},
        "training": {"device": str(run.device), "batch_size": s.batch_size, "micro_batch_size": s.micro_batch_size,
                     "learning_rate": s.learning_rate, "windows_per_epoch": s.windows_per_epoch,
                     "positive_oversample": s.positive_oversample, "seed": s.seed,
                     "optimizer_steps": int(sum(r.get("n_steps") or 0 for r in state.history)),
                     "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else None,
                     "total_time_s": round(time.perf_counter() - started, 2)},
        "created_utc": now_utc(),
    }


def _final_metrics(run: RunContext, reports: Mapping[str, Any], calibration: Mapping[str, Any],
                   threshold: tuple[float, float], baselines: Mapping[str, Any], summary: Mapping[str, Any],
                   val_arrays: tuple[np.ndarray, ...]) -> dict[str, Any]:
    split = "test" if reports["test"] is not None else "validation"
    headline = reports[split]
    return to_builtin({
        **headline, "evaluation_split": split, "note": HEADLINE_NOTE if split == "test" else VALIDATION_NOTE,
        "validation": reports["validation"], "test": reports["test"],
        "calibration": _calibration_summary(calibration, val_arrays), "temperature": temperature_of(calibration),
        "threshold": threshold[0], "threshold_metric": run.settings.threshold_metric,
        "threshold_score": threshold[1], "threshold_selected_on": "validation",
        "baselines": dict(baselines), "baseline_logreg": _legacy_logreg(baselines, split),
        "strongest_baseline": strongest_baseline(baselines, split, headline.get("pr_auc")),
        **summary,
    })


def _log_final(metrics: Mapping[str, Any]) -> None:
    strongest = metrics["strongest_baseline"]
    LOGGER.info("Final (%s, calibrated on validation): PR-AUC %s | ROC-AUC %s | %s %s @ threshold %s | calibration "
                "slope %.3f intercept %.3f | strongest baseline %s PR-AUC %s", metrics["evaluation_split"],
                fmt(metrics["pr_auc"]), fmt(metrics["roc_auc"]), metrics["threshold_metric"],
                fmt(metrics.get(metrics["threshold_metric"])), fmt(metrics["threshold"]),
                metrics["calibration"]["slope"], metrics["calibration"]["intercept"], strongest["name"] or "n/a",
                fmt(strongest["pr_auc"]))


def _areal_skill(run: RunContext, final: Mapping[str, Any], splits: Splits) -> dict[str, Any]:
    """The X6 report for the model about to be published (never fails the run)."""
    s = run.settings
    split, dataset = ("test", splits.test) if splits.test is not None else ("validation", splits.val)
    if not s.areal_skill_enabled:
        return areal_module.unavailable_report("training.areal_skill.enabled is false", split=split,
                                               members=s.areal_skill_members, checkpoint=final)
    try:
        return areal_module.compute_areal_skill(run.cfg, final, members=s.areal_skill_members, split=split,
                                                payload=dataset.payload)
    except Exception as exc:  # noqa: BLE001 - auxiliary report: it must never fail the training run
        LOGGER.warning("The areal-only skill evaluation failed (%s: %s); areal_skill.json reports it as unavailable",
                       type(exc).__name__, exc)
        return areal_module.unavailable_report(f"{type(exc).__name__}: {exc}", split=split,
                                               members=s.areal_skill_members, checkpoint=final)


def _with_areal_threshold(run: RunContext, final: dict[str, Any], splits: Splits) -> dict[str, Any]:
    """best.pt + the alert threshold for ensemble-averaged (forecast / design-storm) probabilities.

    Chosen on VALIDATION like the exact-field threshold; the test split stays untouched. Any
    failure keeps best.pt without it (the predictor then falls back to ``threshold``).
    """
    s = run.settings
    if not s.areal_skill_enabled:
        return final
    try:
        info = areal_module.areal_serving_threshold(run.cfg, final, splits.val.payload, members=s.areal_skill_members,
                                                    metric=s.threshold_metric)
    except Exception as exc:  # noqa: BLE001 - optional serving refinement: never fail the training run
        LOGGER.warning("Could not choose the areal serving threshold (%s: %s); forecasts will use the exact-field "
                       "threshold", type(exc).__name__, exc)
        return final
    final = {**final, "areal_threshold": float(info["threshold"]), "areal_threshold_info": info}
    return {**final, "published_id": published_id(final)}


def _final_checkpoint(run: RunContext, model: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau,
                      state: LoopState, candidate: Mapping[str, Any] | None, calibration: Mapping[str, Any],
                      threshold: float, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """best.pt: the candidate weights + calibration, threshold, metrics and a NEW published identity."""
    base = candidate if candidate is not None else make_checkpoint(
        run, model, optimizer, scheduler, replace(state, best_epoch=state.epoch), kind="best")
    finalized = now_utc()
    final = {**base, "format_version": CHECKPOINT_FORMAT_VERSION, "kind": "best",
             "calibration": {k: calibration[k] for k in ("method", "slope", "intercept")},
             "temperature": temperature_of(calibration), "threshold": float(threshold), "metrics": dict(metrics),
             "history": to_builtin(list(state.history)),  # the whole run, not only up to the best epoch
             "graph_attributes_sha256": run.meta.get("graph_attributes_sha256"),
             "candidate_created_utc": base.get("created_utc"), "created_utc": finalized, "finalized_utc": finalized}
    return {**final, "published_id": published_id(final)}


def _full_validation(run: RunContext, splits: Splits, val_loader: DataLoader) -> tuple[Splits, DataLoader]:
    """Splits / loader over EVERY validation window (the epoch loop may have scored a subsample)."""
    full = splits.validation_full
    if full is splits.val:
        return splits, val_loader
    LOGGER.info("Calibration, threshold and reports use all %d validation windows (%d were scored per epoch)",
                len(full), len(splits.val))
    return replace(splits, val=full, val_full=None), make_loader(full, run.settings.micro_batch_size, shuffle=False)


def finalize(run: RunContext, model: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau, splits: Splits,
             val_loader: DataLoader, state: LoopState, started: float, guard: Any) -> TrainResult:
    """Calibrate the best model on validation, evaluate validation + test, publish best.pt + reports.

    ``guard(run, num_nodes)`` is the out-of-memory context manager of the trainer.
    """
    s = run.settings
    candidate = load_candidate(run, model)
    splits, val_loader = _full_validation(run, splits, val_loader)
    with guard(run, splits.train.num_nodes):
        val_arrays = _score(run, model, val_loader, "validation")
        test_arrays = None if splits.test is None else _score(
            run, model, make_loader(splits.test, s.micro_batch_size, shuffle=False), "test")
    y_val, logits_val, _ = val_arrays
    calibration = fit_calibration(logits_val, y_val, s.calibration)
    threshold = best_threshold(y_val, apply_calibration(np.asarray(logits_val, dtype=np.float64), calibration),
                               metric=s.threshold_metric)
    if 0 == int(y_val.sum()) or int(y_val.sum()) == y_val.size:
        LOGGER.warning("Validation labels are single-class; the alert threshold defaults to %.2f", threshold[0])
    options = {"n_bins": s.reliability_bins, "threshold_metric": s.threshold_metric}
    reports = {"validation": split_report(val_arrays, calibration, threshold[0], **options),
               "test": None if test_arrays is None else split_report(test_arrays, calibration, threshold[0], **options)}
    metrics = _final_metrics(run, reports, calibration, threshold, _baselines(run, splits),
                             _run_summary(run, model, state, started), val_arrays)
    final = _final_checkpoint(run, model, optimizer, scheduler, state, candidate, calibration, threshold[0], metrics)
    final = _with_areal_threshold(run, final, splits)
    areal = _areal_skill(run, final, splits)
    metrics = to_builtin({**metrics, "areal_skill": areal_module.summary_of(areal)})
    final = {**final, "metrics": metrics}
    extras = {run.reports_dir / HISTORY_NAME: history_csv(state.history),
              run.reports_dir / areal_module.REPORT_NAME: areal_module.report_text(areal)}
    publish_final(final, run.paths.best, metrics, run.reports_dir / "metrics.json", extras)
    discard_partial_history(run.reports_dir)
    _log_final(metrics)
    return TrainResult(run.paths.best, run.paths.last, int(final.get("epoch", state.best_epoch)), metrics,
                       [dict(r) for r in state.history])
