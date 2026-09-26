"""Training loop for the Namma-Flow spatio-temporal flood GNN.

:func:`train` runs the whole stage:

1. seed everything, pick the device, load the TRAIN / VAL (and optional held-out TEST)
   windows (building the datasets with stage 04 when missing) and check that all splits
   share the graph, features, scaler and window geometry (:mod:`src.training.splits`);
2. build the model (``model`` section; its input dims must match the dataset), the loss
   (``loss`` section, class balance from the TRAIN split), AdamW and ReduceLROnPlateau;
3. per epoch, draw ``training.windows_per_epoch`` windows with a seeded
   :class:`~torch.utils.data.WeightedRandomSampler` (flood windows weighted
   ``positive_oversample``). Each optimizer step uses ``model.batch_size`` windows,
   processed as micro-batches of ``training.micro_batch_size`` windows whose gradients
   accumulate (weighted by their share of the batch, so the step is the one of the full
   batch - smaller stacked graphs are simply faster on CPU). Loss on non-warm-up steps,
   gradient clipping; non-finite steps are skipped and ``max_nonfinite_steps`` in a row
   abort. Then validate: val loss, PR-AUC (the early-stopping metric), ROC-AUC, F2;
4. save ``best_candidate.pt`` on improvement (by more than ``min_delta``; relative to the
   best score by default, see :class:`~src.training.settings.Monitor`) and ``last.pt``
   every epoch (atomic), stop early after ``early_stopping_patience`` epochs without
   improvement;
5. finalise (:mod:`src.training.finalize`): reload the best candidate, fit the calibration
   and the alert threshold on VALIDATION (all windows), evaluate VALIDATION and TEST, fit the
   graph-free baselines, measure the areal-only skill (:mod:`src.training.areal_skill`), and
   only then publish ``best.pt`` + ``metrics.json`` + ``training_history.csv`` +
   ``areal_skill.json`` together (the per-epoch history is staged until then). An
   interrupted or crashed run therefore never replaces the previous published model.

``resume=True`` continues from ``last.pt`` (weights, optimizer, scheduler, history, best
score and RNG states). Ctrl-C keeps ``last.pt`` at the last COMPLETED epoch (only its
``interrupted`` flag is set), so a resumed run repeats the interrupted epoch from exactly
the state an uninterrupted run had. A resume is refused when the model, the features / graph /
scaler or the dataset BUILD differ (``build_id``, dataset config hash, weather and report
fingerprints: the stored best score would come from other labels). When only the validation
windows changed (``training.max_val_windows``), the run's best candidate is re-scored on the
current validation set and early stopping / the LR plateau counter restart from it (F2-02).
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, Sampler, WeightedRandomSampler

from src.data_pipeline.dataset import FloodSequenceDataset, collate_windows, make_loader
from src.models.loss import build_loss
from src.models.stgcn import build_model, count_parameters
from src.training import finalize as finalize_module
from src.training.checkpoint import (
    CheckpointError,
    checkpoint_paths,
    load_checkpoint,
    model_from_checkpoint,
    restore_rng_state,
    save_checkpoint,
    to_builtin,
)
from src.training.evaluation import evaluate, sigmoid
from src.training.metrics import best_threshold, ranking_metrics
from src.training.reports import HISTORY_FIELDS, write_history
from src.training.run_state import (
    NEUTRAL_THRESHOLD,
    LoopState,
    RunContext,
    TrainingError,
    TrainResult,
    fmt,
    make_checkpoint,
)
from src.training.settings import MONITOR_MODES, Monitor, TrainingSettings
from src.training.splits import (
    BUILD_KEYS,
    REBUILD_HINT,
    Splits,
    attributes_digest,
    dataset_info,
    load_datasets,
    load_splits,
)
from src.utils.config import config_hash, get_section, resolve_path
from src.utils.logger import get_logger
from src.utils.runtime import resolve_device, set_seed

__all__ = [
    "TrainResult", "TrainingError", "LoopState", "train", "evaluate", "load_datasets", "load_splits",
    "load_checkpoint", "model_from_checkpoint", "window_sampling_weights", "epoch_sampler", "CheckpointError",
    "HISTORY_FIELDS",
]

LOGGER = get_logger(__name__)

TRAIN_HASH_SECTIONS = ("model", "loss", "training")
_CORE_MODEL_KEYS = ("architecture", "node_in_dim", "edge_dim", "hidden_dim", "heads", "dropout")
_RESUME_KEYS = ("feature_names", "graph_signature", "seq_len", "warmup_steps", "scaler")


# --------------------------------------------------------------------------- setup


def _checkpoint_meta(cfg: Mapping[str, Any], train_ds: FloodSequenceDataset) -> dict:
    payload = train_ds.payload
    return to_builtin({
        "feature_names": train_ds.feature_names, "static_feature_names": train_ds.static_feature_names,
        "rolling_windows_h": list(train_ds.rolling_windows_h), "lookback_hours": train_ds.lookback_hours,
        "seq_len": train_ds.seq_len, "warmup_steps": train_ds.warmup_steps, "scaler": train_ds.scaler.to_dict(),
        "graph_signature": train_ds.graph_signature, "graph_attributes_sha256": attributes_digest(train_ds),
        "node_ids": train_ds.node_ids,
        "config_hash": config_hash(cfg, TRAIN_HASH_SECTIONS), "dataset_config_hash": payload.get("config_hash"),
        "label_source": payload.get("label_source"), "flood_threshold_m": payload.get("flood_threshold_m"),
        "timezone": payload.get("timezone"), "graph_provenance": payload.get("graph_provenance", {}),
        "edge_features": ["length", "grade"], "weather_fingerprint": payload.get("weather_fingerprint"),
        "run_id": uuid.uuid4().hex,
    })


def _dataset_identity(train_ds: FloodSequenceDataset) -> dict[str, Any]:
    """The dataset build a run trains on (all splits share it, see :data:`BUILD_KEYS`)."""
    return {key: train_ds.payload.get(key) for key in BUILD_KEYS}


def _val_identity(val_ds: FloodSequenceDataset, max_val_windows: int | None) -> dict[str, Any]:
    """Which validation windows and labels the per-epoch scores come from."""
    digest = hashlib.sha256(np.ascontiguousarray(val_ds.window_start_times, dtype=np.int64).tobytes())
    labels = val_ds.payload["labels"]
    digest.update(np.ascontiguousarray(labels.numpy() if hasattr(labels, "numpy") else labels).tobytes())
    return {"max_val_windows": max_val_windows, "windows": len(val_ds), "digest": digest.hexdigest()[:16]}


def _build_model(cfg: Mapping[str, Any], train_ds: FloodSequenceDataset) -> nn.Module:
    try:
        model = build_model(get_section(cfg, "model", {}))
    except ValueError as exc:
        raise TrainingError(f"Invalid model configuration: {exc}") from exc
    edge_dim = int(train_ds.edge_attr.shape[1])
    if model.node_in_dim != train_ds.num_features:
        raise TrainingError(f"model.node_in_dim={model.node_in_dim} but the dataset has {train_ds.num_features} node "
                            f"features {train_ds.feature_names}; set model.node_in_dim: {train_ds.num_features} or "
                            f"{REBUILD_HINT}")
    if model.edge_dim != edge_dim:
        raise TrainingError(f"model.edge_dim={model.edge_dim} but the dataset has {edge_dim} edge features "
                            f"(length, grade); set model.edge_dim: {edge_dim}")
    return model


def _choose_monitor(settings: TrainingSettings, val_ds: FloodSequenceDataset) -> Monitor:
    name = settings.early_stopping_metric
    if name != "val_loss" and not val_ds.window_has_flood.any():
        LOGGER.warning("The validation windows contain no flooded node-steps, so %s is undefined; early stopping, "
                       "the LR schedule and the best checkpoint use val_loss instead", name)
        name = "val_loss"
    return Monitor(name, MONITOR_MODES[name], settings.min_delta, settings.min_delta_mode == "rel")


def _setup(cfg: Mapping[str, Any], settings: TrainingSettings) -> tuple[RunContext, nn.Module, nn.Module, Splits]:
    if settings.num_threads is not None:
        torch.set_num_threads(settings.num_threads)
    device = resolve_device(settings.device)
    splits = load_splits(cfg, settings.max_val_windows)
    set_seed(settings.seed)  # after a possible dataset build, so initial weights never depend on it
    model = _build_model(cfg, splits.train).to(device)
    loss_cfg = {**get_section(cfg, "loss", {}), "reduction": "mean"}
    criterion = build_loss(loss_cfg, pos_rate=splits.train.pos_rate).to(device)
    meta = {**_checkpoint_meta(cfg, splits.train), "dataset_identity": _dataset_identity(splits.train),
            "val_set": _val_identity(splits.val, settings.max_val_windows)}
    run = RunContext(settings=settings, device=device, monitor=_choose_monitor(settings, splits.val),
                     paths=checkpoint_paths(cfg), reports_dir=resolve_path(cfg, "reports_dir", "artifacts/reports"),
                     meta=to_builtin(meta), dataset_info=dataset_info(cfg, splits), cfg=cfg)
    return run, model, criterion, splits


# --------------------------------------------------------------------------- sampling


def window_sampling_weights(has_flood: Sequence[bool] | np.ndarray, positive_oversample: float) -> torch.Tensor:
    """Per-window sampling weights: ``positive_oversample`` for flood windows, 1 otherwise (float64)."""
    if isinstance(positive_oversample, bool) or not float(positive_oversample) > 0:
        raise ValueError(f"positive_oversample must be > 0, got {positive_oversample!r}")
    flags = np.asarray(has_flood, dtype=bool).reshape(-1)
    return torch.from_numpy(np.where(flags, float(positive_oversample), 1.0))


def epoch_sampler(has_flood: Sequence[bool] | np.ndarray, settings: TrainingSettings, epoch: int) -> Sampler:
    """Seeded window sampler for ``epoch`` (identical across resumed runs).

    ``windows_per_epoch`` null with ``positive_oversample == 1`` visits every window once;
    otherwise windows are drawn with replacement, flood windows weighted ``positive_oversample``.
    """
    flags = np.asarray(has_flood, dtype=bool).reshape(-1)
    generator = torch.Generator().manual_seed((settings.seed * 1_000_003 + int(epoch)) % (2 ** 63))
    if settings.windows_per_epoch is None and settings.positive_oversample == 1.0:
        return RandomSampler(range(flags.size), generator=generator)
    num_samples = settings.windows_per_epoch or int(flags.size)
    weights = window_sampling_weights(flags, settings.positive_oversample)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=generator)


def _item_list(items: list[dict]) -> list[dict]:
    return list(items)


def _epoch_loader(train_ds: FloodSequenceDataset, settings: TrainingSettings, epoch: int) -> DataLoader:
    """Batches of ``batch_size`` window items (lists; micro-batches are collated in the step)."""
    return DataLoader(train_ds, batch_size=settings.batch_size, shuffle=False,
                      sampler=epoch_sampler(train_ds.window_has_flood, settings, epoch), collate_fn=_item_list)


# --------------------------------------------------------------------------- one epoch


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


@contextmanager
def _oom_guard(run: RunContext, num_nodes: int) -> Iterator[None]:
    """Turn device out-of-memory errors into an actionable :class:`TrainingError`."""
    try:
        yield
    except RuntimeError as exc:
        if isinstance(exc, TrainingError) or not _is_oom(exc):
            raise
        raise TrainingError(
            f"Out of memory on {run.device} (micro-batch of {run.settings.micro_batch_size} windows x {num_nodes} "
            "junctions): lower training.micro_batch_size (gradients still accumulate to model.batch_size) or "
            "model.max_chunk_edges, set model.gradient_checkpointing: true, or train on the CPU with --device cpu"
        ) from exc


def _criterion_input(criterion: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    return logits if getattr(criterion, "from_logits", True) else torch.sigmoid(logits)


def _batch_loss(model: nn.Module, batch: Mapping[str, Any], criterion: nn.Module, device: torch.device) -> torch.Tensor:
    logits, _ = model.forward_sequence(batch["x"].to(device), batch["edge_index"].to(device),
                                       batch["edge_attr"].to(device))
    return criterion(_criterion_input(criterion, logits), batch["y"].to(device), mask=batch["mask"].to(device))


def _accumulate(model: nn.Module, items: list[dict], criterion: nn.Module, settings: TrainingSettings,
                device: torch.device, collate: Callable[[list[dict]], dict]) -> tuple[float, bool]:
    """Forward/backward every micro-batch of ``items``; returns (batch loss, all finite).

    Each micro-batch loss (a mean over its scored node-steps) is weighted by its share of
    the batch's windows; every window has the same number of scored node-steps, so the
    accumulated gradient equals the gradient of the full-batch mean loss.
    """
    total, value = len(items), 0.0
    for start in range(0, total, settings.micro_batch_size):
        chunk = items[start: start + settings.micro_batch_size]
        loss = _batch_loss(model, collate(chunk), criterion, device)
        if not bool(torch.isfinite(loss)):
            return float(loss.detach()), False
        weight = len(chunk) / total
        (loss * weight).backward()
        value += float(loss.detach()) * weight
    return value, True


def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optimizer,
                 settings: TrainingSettings, device: torch.device) -> dict[str, Any]:
    """One pass over the sampled windows; skips (and counts) steps with non-finite loss or gradients."""
    model.train()
    dataset = loader.dataset
    collate = partial(collate_windows, edge_index=dataset.edge_index, edge_attr=dataset.edge_attr,
                      num_nodes=dataset.num_nodes)
    losses, norms, skipped, consecutive = [], [], 0, 0
    clip = settings.grad_clip_norm if settings.grad_clip_norm is not None else math.inf
    for step, items in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss, finite = _accumulate(model, items, criterion, settings, device, collate)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip) if finite else None
        if norm is None or not bool(torch.isfinite(norm)):
            optimizer.zero_grad(set_to_none=True)
            skipped, consecutive = skipped + 1, consecutive + 1
            LOGGER.warning("Training step %d: non-finite loss or gradient (loss=%s); batch skipped (%d in a row)",
                           step, loss, consecutive)
            if consecutive >= settings.max_nonfinite_steps:
                raise TrainingError(f"Aborting: {consecutive} consecutive training steps had a non-finite loss or "
                                    "gradient; lower model.learning_rate, keep training.grad_clip_norm set, or "
                                    "check the datasets for corrupt values")
            continue
        optimizer.step()
        consecutive = 0
        losses.append(loss)
        norms.append(float(norm))
        LOGGER.debug("step %d: loss %.6f, grad norm %.4f", step, losses[-1], norms[-1])
    if skipped and not losses:
        raise TrainingError("Every training step of the epoch had a non-finite loss or gradient")
    return {"train_loss": float(np.mean(losses)) if losses else None, "n_steps": len(losses), "n_skipped": skipped,
            "grad_norm": float(np.mean(norms)) if norms else None}


def _validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
              threshold_metric: str = "f2") -> dict[str, Any]:
    """Validation loss and ranking metrics of one epoch (None where undefined or non-finite)."""
    y, logits, _ = evaluate(model, loader, device)
    stats: dict[str, Any] = {"val_loss": None, "pr_auc": None, "roc_auc": None, "brier": None, "f2": None,
                             "f2_threshold": None, "alert_threshold": None, "n_val": int(y.size),
                             "n_val_pos": int(y.sum())}
    if not np.isfinite(logits).all():
        LOGGER.warning("Validation produced %d non-finite logits; epoch metrics are undefined",
                       int((~np.isfinite(logits)).sum()))
        return stats
    with torch.no_grad():
        target = torch.from_numpy(y.astype(np.float32))
        stats["val_loss"] = float(criterion(_criterion_input(criterion, torch.from_numpy(logits)), target))
    probs = sigmoid(logits)
    ranking = ranking_metrics(y, probs)
    threshold, f2 = best_threshold(y, probs, metric="f2")
    alert = threshold if threshold_metric == "f2" else best_threshold(y, probs, metric=threshold_metric)[0]
    stats.update(pr_auc=ranking["pr_auc"], roc_auc=ranking["roc_auc"], brier=ranking["brier"],
                 f2=f2 if 0 < ranking["n_pos"] < ranking["n"] else None, f2_threshold=threshold,
                 alert_threshold=alert)
    return stats


# --------------------------------------------------------------------------- resume


def _check_resume_compatible(ckpt: Mapping[str, Any], model: nn.Module, run: RunContext, path: Any) -> None:
    saved = {**dict(ckpt.get("model_config") or {}), "architecture": ckpt.get("architecture")}
    current = {**model.get_config(), "architecture": model.architecture}
    diffs = [f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in _CORE_MODEL_KEYS if saved.get(k) != current.get(k)]
    diffs += [f"{k} differs" for k in _RESUME_KEYS if to_builtin(ckpt.get(k)) != run.meta[k]]
    stored_digest, digest = ckpt.get("graph_attributes_sha256"), run.meta.get("graph_attributes_sha256")
    if stored_digest is not None and digest is not None and stored_digest != digest:
        diffs.append("graph_attributes_sha256 differs")
    diffs += [f"{k} missing" for k in ("optimizer_state", "scheduler_state") if not isinstance(ckpt.get(k), Mapping)]
    if diffs:
        raise TrainingError(f"Cannot resume from {path}: it belongs to a different model or dataset "
                            f"({'; '.join(diffs)}). Train without --resume (the checkpoints will be overwritten) "
                            "or restore the config")
    _check_same_dataset_build(ckpt, run, path)


def _check_same_dataset_build(ckpt: Mapping[str, Any], run: RunContext, path: Any) -> None:
    """Refuse a resume across a dataset rebuild: the stored best score and model come from other labels."""
    current = dict(run.meta.get("dataset_identity") or {})
    stored = ckpt.get("dataset_identity")
    if not isinstance(stored, Mapping):  # checkpoints written before the identity was stored
        stored = {"config_hash": ckpt.get("dataset_config_hash"),
                  "weather_fingerprint": ckpt.get("weather_fingerprint")}
    differ = [f"{key} {stored.get(key)!r} -> {current.get(key)!r}" for key in BUILD_KEYS
              if key in stored and stored.get(key) != current.get(key)]
    if differ:
        raise TrainingError(f"Cannot resume from {path}: it was trained on a different dataset build "
                            f"({'; '.join(differ)}), so its best score, early-stopping state and weights come from "
                            "other labels. Train without --resume (the checkpoints will be overwritten) or restore "
                            "the datasets it was trained on")


def _reset_scheduler_monitor(scheduler: ReduceLROnPlateau, mode: str) -> None:
    """Point a restored ReduceLROnPlateau at a new monitor direction and forget its best / bad epochs."""
    scheduler.mode = mode
    scheduler._init_is_better(mode=mode, threshold=scheduler.threshold, threshold_mode=scheduler.threshold_mode)
    scheduler._reset()


def _apply_current_settings(scheduler: ReduceLROnPlateau, optimizer: Optimizer, run: RunContext,
                            extra: Mapping[str, Any]) -> None:
    """Re-apply the CURRENT LR-schedule / weight-decay config over the restored state (WARNING on changes)."""
    s = run.settings
    saved = {"factor": scheduler.factor, "patience": scheduler.patience, "min_lr": min(scheduler.min_lrs)}
    wanted = {"factor": s.lr_factor, "patience": s.lr_patience, "min_lr": s.min_lr}
    changed = {k: (saved[k], wanted[k]) for k in wanted if not math.isclose(float(saved[k]), float(wanted[k]))}
    if changed:
        LOGGER.warning("training.lr_scheduler changed since the checkpoint (%s); the current values apply",
                       ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items()))
    scheduler.factor, scheduler.patience = s.lr_factor, s.lr_patience
    scheduler.min_lrs = [s.min_lr] * len(optimizer.param_groups)
    if hasattr(scheduler, "default_min_lr"):
        scheduler.default_min_lr = s.min_lr
    stored_wd = extra.get("weight_decay")
    if stored_wd is not None and not math.isclose(float(stored_wd), s.weight_decay):
        LOGGER.warning("model.weight_decay changed since the checkpoint (%s -> %s); the current value applies",
                       stored_wd, s.weight_decay)
    for group in optimizer.param_groups:
        group["weight_decay"] = s.weight_decay
    stored_lr = extra.get("learning_rate")
    if stored_lr is not None and not math.isclose(float(stored_lr), s.learning_rate):
        LOGGER.warning("model.learning_rate changed since the checkpoint (%s -> %s); the resumed run continues the "
                       "checkpoint's LR schedule (current LR %.3g) - train without --resume to use the new rate",
                       stored_lr, s.learning_rate, optimizer.param_groups[0]["lr"])


def _resume_state(ckpt: Mapping[str, Any], run: RunContext, scheduler: ReduceLROnPlateau) -> LoopState:
    extra = dict(ckpt.get("train_state") or {})
    bad_epochs = int(extra.get("bad_epochs", 0))
    # Re-derived with the CURRENT patience: raising early_stopping_patience lets a stopped run continue.
    state = LoopState(epoch=int(ckpt.get("epoch", 0)), best_score=ckpt.get("best_score"),
                      best_epoch=int(extra.get("best_epoch", 0)), bad_epochs=bad_epochs,
                      history=tuple(ckpt.get("history") or ()),
                      early_stopped=0 < bad_epochs and bad_epochs >= run.settings.early_stopping_patience)
    old_name, old_mode = extra.get("monitor", run.monitor.name), extra.get("monitor_mode", run.monitor.mode)
    if (old_name, old_mode) != (run.monitor.name, run.monitor.mode):
        LOGGER.warning("The early-stopping metric changed (%s/%s -> %s/%s); the best score, early stopping and the "
                       "LR plateau counter are reset", old_name, old_mode, run.monitor.name, run.monitor.mode)
        state = replace(state, best_score=None, bad_epochs=0, early_stopped=False)
        _reset_scheduler_monitor(scheduler, run.monitor.mode)
    return state


def _rescore_best(run: RunContext, model: nn.Module, criterion: nn.Module, val_loader: DataLoader) -> float | None:
    """This run's best candidate scored on the CURRENT validation set (None if no candidate of this run)."""
    path = run.paths.candidate
    try:
        candidate = load_checkpoint(path) if path.exists() else None
    except CheckpointError as exc:
        LOGGER.warning("%s is unreadable (%s); the best score restarts from the next epoch", path, exc)
        return None
    if candidate is None or candidate.get("run_id") != run.meta.get("run_id"):
        return None
    scratch = copy.deepcopy(model)
    scratch.load_state_dict(candidate["model_state"])
    value = _validate(scratch, val_loader, criterion, run.device, run.settings.threshold_metric).get(run.monitor.name)
    return value if value is not None and math.isfinite(value) else None


def _revalidated_state(ckpt: Mapping[str, Any], run: RunContext, state: LoopState, scheduler: ReduceLROnPlateau,
                       rescore: Callable[[], float | None]) -> LoopState:
    """Make the stored best score comparable when the validation windows changed since ``ckpt`` (F2-02)."""
    stored, current = ckpt.get("val_set"), run.meta.get("val_set")
    if state.best_score is None or (isinstance(stored, Mapping) and dict(stored) == dict(current or {})):
        return state
    score = rescore()
    LOGGER.warning("The validation windows differ from those the checkpoint's scores were computed on (%s -> %s); "
                   "the best candidate was re-scored on the current validation set (%s = %s, was %s) and early "
                   "stopping / the LR plateau counter restart from it", _describe_val(stored), _describe_val(current),
                   run.monitor.name, fmt(score), fmt(state.best_score))
    _reset_scheduler_monitor(scheduler, run.monitor.mode)
    return replace(state, best_score=score, bad_epochs=0, early_stopped=False)


def _describe_val(identity: Any) -> str:
    if not isinstance(identity, Mapping):
        return "unknown (older checkpoint)"
    limit = identity.get("max_val_windows")
    return f"{identity.get('windows')} windows ({'all' if limit is None else f'max_val_windows={limit}'})"


def _resume(run: RunContext, model: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau,
            criterion: nn.Module, val_loader: DataLoader) -> tuple[RunContext, LoopState]:
    path = run.paths.last
    if not path.exists():
        LOGGER.warning("--resume: %s does not exist; starting a fresh run", path)
        return run, LoopState()
    ckpt = load_checkpoint(path)
    _check_resume_compatible(ckpt, model, run, path)
    try:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise TrainingError(f"Cannot resume from {path}: {type(exc).__name__}: {exc}") from exc
    restore_rng_state(ckpt.get("rng_state"))
    run = replace(run, meta={**run.meta, "run_id": ckpt.get("run_id", run.meta["run_id"])})
    state = _resume_state(ckpt, run, scheduler)
    state = _revalidated_state(ckpt, run, state, scheduler, partial(_rescore_best, run, model, criterion, val_loader))
    _apply_current_settings(scheduler, optimizer, run, dict(ckpt.get("train_state") or {}))
    LOGGER.info("Resumed from %s: %d epoch(s) done, best %s = %s at epoch %d%s", path, state.epoch, run.monitor.name,
                fmt(state.best_score), state.best_epoch, " (early-stopped)" if state.early_stopped else "")
    return run, state


# --------------------------------------------------------------------------- epoch loop


def _epoch_record(epoch: int, lr: float, train_stats: Mapping[str, Any], val_stats: Mapping[str, Any],
                  times: tuple[float, float], monitor_value: float | None, improved: bool) -> dict[str, Any]:
    return to_builtin({
        "epoch": epoch, **{k: train_stats[k] for k in ("train_loss", "n_steps", "n_skipped", "grad_norm")},
        **{k: val_stats[k] for k in ("val_loss", "pr_auc", "roc_auc", "f2", "f2_threshold", "brier")},
        "lr": lr, "train_time_s": round(times[0], 3), "val_time_s": round(times[1], 3),
        "epoch_time_s": round(times[0] + times[1], 3), "monitor_value": monitor_value, "improved": improved,
    })


def _next_state(state: LoopState, epoch: int, record: dict, improved: bool, value: float | None,
                settings: TrainingSettings) -> LoopState:
    bad = 0 if improved else state.bad_epochs + 1
    return replace(state, epoch=epoch, history=(*state.history, record), bad_epochs=bad,
                   best_score=value if improved else state.best_score,
                   best_epoch=epoch if improved else state.best_epoch,
                   early_stopped=not improved and bad >= settings.early_stopping_patience and epoch < settings.epochs)


def _run_epoch(run: RunContext, epoch: int, model: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau,
               criterion: nn.Module, train_ds: FloodSequenceDataset, val_loader: DataLoader,
               state: LoopState) -> LoopState:
    settings, lr = run.settings, float(optimizer.param_groups[0]["lr"])
    loader = _epoch_loader(train_ds, settings, epoch)
    with _oom_guard(run, train_ds.num_nodes):
        clock = time.perf_counter()
        train_stats = _train_epoch(model, loader, criterion, optimizer, settings, run.device)
        train_time, clock = time.perf_counter() - clock, time.perf_counter()
        val_stats = _validate(model, val_loader, criterion, run.device, settings.threshold_metric)
        val_time = time.perf_counter() - clock
    value = val_stats.get(run.monitor.name)
    improved = run.monitor.improved(value, state.best_score)
    if value is not None and math.isfinite(value):
        scheduler.step(value)
    record = _epoch_record(epoch, lr, train_stats, val_stats, (train_time, val_time), value, improved)
    state = _next_state(state, epoch, record, improved, value, settings)
    if improved:  # per-epoch best: a candidate only; best.pt is published by the finalisation
        threshold = val_stats.get("alert_threshold") or val_stats.get("f2_threshold") or NEUTRAL_THRESHOLD
        save_checkpoint(make_checkpoint(run, model, optimizer, scheduler, state, kind="best_candidate",
                                        metrics=record, threshold=threshold), run.paths.candidate)
    save_checkpoint(make_checkpoint(run, model, optimizer, scheduler, state, kind="last", metrics=record),
                    run.paths.last)
    write_history(run.reports_dir, state.history)  # staged; published with best.pt by the finalisation (F2-03)
    LOGGER.info("Epoch %d/%d | train loss %s (%d steps, %d skipped) | val loss %s | PR-AUC %s | ROC-AUC %s | "
                "F2 %s @ %s | lr %.2e | %.1f s%s", epoch, settings.epochs, fmt(train_stats["train_loss"], 5),
                train_stats["n_steps"], train_stats["n_skipped"], fmt(val_stats["val_loss"], 5),
                fmt(val_stats["pr_auc"]), fmt(val_stats["roc_auc"]), fmt(val_stats["f2"]),
                fmt(val_stats["f2_threshold"], 3), lr, train_time + val_time, "  * best" if improved else "")
    if state.early_stopped:
        LOGGER.info("Early stopping: no %s improvement > %g (%s) for %d epoch(s)", run.monitor.name,
                    settings.min_delta, settings.min_delta_mode, state.bad_epochs)
    return state


def _read_last(run: RunContext) -> dict[str, Any] | None:
    try:
        return load_checkpoint(run.paths.last) if run.paths.last.exists() else None
    except CheckpointError as exc:
        LOGGER.warning("%s is unreadable (%s)", run.paths.last, exc)
        return None


def _save_interrupted(run: RunContext, state: LoopState, snapshot: dict[str, Any] | None) -> None:
    """Leave a CONSISTENT resume point: last.pt of the last completed epoch (flagged), or the start snapshot.

    The live weights / optimizer / RNG are mid-epoch and are never saved: resuming from
    them would train part of an epoch twice under a wrong epoch label.
    """
    on_disk = _read_last(run)
    if on_disk is not None and on_disk.get("run_id") == run.meta["run_id"] and int(on_disk["epoch"]) >= state.epoch:
        extra = {**dict(on_disk.get("train_state") or {}), "interrupted": True}
        save_checkpoint({**on_disk, "train_state": extra}, run.paths.last)
        LOGGER.warning("Interrupted; %s keeps the state after epoch %d (continue with --resume)", run.paths.last,
                       int(on_disk["epoch"]))
    elif snapshot is not None:
        save_checkpoint(snapshot, run.paths.last)
        LOGGER.warning("Interrupted before the first epoch completed; %s holds the start state (continue with "
                       "--resume)", run.paths.last)
    else:  # pragma: no cover - the loop always has a snapshot or its own last.pt
        LOGGER.warning("Interrupted; no consistent state to save - %s was left unchanged", run.paths.last)


def _fit(run: RunContext, model: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau, criterion: nn.Module,
         train_ds: FloodSequenceDataset, val_loader: DataLoader, state: LoopState) -> LoopState:
    """Epoch loop; on Ctrl-C a consistent resume point is left in ``last.pt`` and the interrupt re-raised."""
    snapshot = None
    if state.epoch == 0:  # nothing of this run on disk yet: keep the exact start state (cheap)
        snapshot = make_checkpoint(run, model, optimizer, scheduler, state, kind="last", interrupted=True,
                                   snapshot=True)
    try:
        for epoch in range(state.epoch + 1, run.settings.epochs + 1):
            if state.early_stopped:
                break
            state = _run_epoch(run, epoch, model, optimizer, scheduler, criterion, train_ds, val_loader, state)
    except KeyboardInterrupt:
        _save_interrupted(run, state, snapshot)
        raise
    return state


# --------------------------------------------------------------------------- entry point


def train(cfg: Mapping[str, Any], *, epochs: int | None = None, device: str | None = None,
          resume: bool | None = None, windows_per_epoch: int | None = None,
          max_val_windows: int | None = None) -> TrainResult:
    """Train, calibrate and evaluate the flood GNN; see the module docstring.

    Arguments override the config (``epochs`` → ``model.epochs``; the others →
    ``training.*``; ``windows_per_epoch`` / ``max_val_windows`` of 0 mean "all windows").
    With ``resume`` the run continues from ``last.pt`` up to ``epochs`` in total.
    Raises :class:`TrainingError`, ``ConfigError`` or ``DatasetError`` with actionable
    messages; ``KeyboardInterrupt`` propagates after a consistent ``last.pt`` is left.
    """
    started = time.perf_counter()
    settings = TrainingSettings.from_config(cfg, epochs=epochs, device=device, resume=resume,
                                            windows_per_epoch=windows_per_epoch, max_val_windows=max_val_windows)
    run, model, criterion, splits = _setup(cfg, settings)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode=run.monitor.mode, factor=settings.lr_factor,
                                  patience=settings.lr_patience, min_lr=settings.min_lr)
    train_ds, val_ds = splits.train, splits.val
    # Scores do not depend on the batching (eval mode); micro-batch-sized graphs are faster and smaller.
    val_loader = make_loader(val_ds, settings.micro_batch_size, shuffle=False)
    state = LoopState()
    if settings.resume:
        run, state = _resume(run, model, optimizer, scheduler, criterion, val_loader)
    LOGGER.info("Training %s (%d parameters) on %s: %d train windows (%d with floods), %d val windows, %s test "
                "windows, %d junctions; %s windows/epoch, batch %d (micro-batches of %d), epochs %d -> %d, monitor %s",
                model.architecture, count_parameters(model), run.device, len(train_ds),
                int(train_ds.window_has_flood.sum()), len(val_ds), len(splits.test) if splits.test else "no",
                train_ds.num_nodes, settings.windows_per_epoch or "all", settings.batch_size,
                settings.micro_batch_size, state.epoch, settings.epochs, run.monitor.name)
    state = _fit(run, model, optimizer, scheduler, criterion, train_ds, val_loader, state)
    return finalize_module.finalize(run, model, optimizer, scheduler, splits, val_loader, state, started, _oom_guard)
