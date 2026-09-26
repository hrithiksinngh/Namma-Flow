"""Types shared by the epoch loop (:mod:`src.training.trainer`) and the finalisation
(:mod:`src.training.finalize`): the run constants, the immutable loop state, the result,
the handled-error type and the checkpoint-dict builder (format v2, see
:mod:`src.training.checkpoint`)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.calibration import identity_calibration, validate_calibration
from src.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointPaths,
    capture_rng_state,
    cpu_state_dict,
    to_builtin,
)
from src.training.settings import Monitor, TrainingSettings

__all__ = ["LoopState", "NEUTRAL_THRESHOLD", "RunContext", "TrainResult", "TrainingError", "fmt", "make_checkpoint",
           "now_utc"]

NEUTRAL_THRESHOLD = 0.5


class TrainingError(RuntimeError):
    """A handled training failure with an actionable message (the CLI prints it and exits 1)."""


@dataclass(frozen=True)
class TrainResult:
    """Outcome of :func:`src.training.trainer.train`."""

    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    metrics: dict
    history: list[dict]


@dataclass(frozen=True)
class LoopState:
    """Progress of the epoch loop (replaced, never mutated, after every epoch)."""

    epoch: int = 0
    best_score: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0
    history: tuple[dict, ...] = ()
    early_stopped: bool = False


@dataclass(frozen=True)
class RunContext:
    """Constants of one training run."""

    settings: TrainingSettings
    device: torch.device
    monitor: Monitor
    paths: CheckpointPaths
    reports_dir: Path
    meta: dict            # static checkpoint fields (features, scaler, graph, hashes, run id)
    dataset_info: dict
    cfg: Mapping[str, Any] = field(default_factory=dict)  # the run's configuration (areal-skill evaluation)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt(value: float | None, digits: int = 4) -> str:
    """Fixed decimals, switching to scientific notation for tiny rare-event values (PR-AUC ~1e-4)."""
    if value is None:
        return "n/a"
    return f"{value:.2e}" if 0 < abs(value) < 10 ** -(digits - 1) else f"{value:.{digits}f}"


def make_checkpoint(run: RunContext, model: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau,
                    state: LoopState, *, kind: str, metrics: Mapping[str, Any] | None = None,
                    threshold: float = NEUTRAL_THRESHOLD, calibration: Mapping[str, Any] | None = None,
                    interrupted: bool = False, snapshot: bool = False) -> dict[str, Any]:
    """Checkpoint dict in the documented format (never carries ``finalized_utc``).

    ``snapshot`` deep-copies the optimizer / scheduler states so the dict stays valid while
    training continues (the live optimizer state tensors are updated in place).
    """
    cal = validate_calibration(calibration or identity_calibration("none"))
    optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
    if snapshot:
        optimizer_state, scheduler_state = copy.deepcopy(optimizer_state), copy.deepcopy(scheduler_state)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION, "kind": kind,
        "model_state": cpu_state_dict(model), "architecture": model.architecture,
        "model_config": to_builtin(model.get_config()), **run.meta,
        "calibration": cal, "temperature": 1.0 / cal["slope"], "threshold": float(threshold),
        "metrics": to_builtin(metrics or {}),
        "epoch": int(state.epoch), "best_score": state.best_score, "history": to_builtin(list(state.history)),
        "optimizer_state": optimizer_state, "scheduler_state": scheduler_state,
        "rng_state": capture_rng_state(run.device),
        "train_state": {"best_epoch": state.best_epoch, "bad_epochs": state.bad_epochs,
                        "early_stopped": state.early_stopped, "interrupted": interrupted,
                        "monitor": run.monitor.name, "monitor_mode": run.monitor.mode,
                        "epochs_target": run.settings.epochs, "learning_rate": run.settings.learning_rate,
                        "weight_decay": run.settings.weight_decay},
        "created_utc": now_utc(),
    }
