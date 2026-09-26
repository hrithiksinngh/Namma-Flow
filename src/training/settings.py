"""Validated training settings: the ``training`` section plus the loop-related ``model`` keys.

Every key has a default (:data:`TRAINING_DEFAULTS`, :data:`MODEL_TRAINING_DEFAULTS`), so a
partial or missing section never crashes; invalid values raise :class:`ConfigError` naming
the offending key. Explicit arguments of :func:`src.training.trainer.train` (CLI flags)
override the configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from src.training.baseline import BASELINE_DEFAULTS
from src.training.calibration import CALIBRATION_METHODS
from src.training.metrics import THRESHOLD_METRICS
from src.utils.config import ConfigError, deep_merge, get_section
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

TRAINING_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "num_threads": None,
    "windows_per_epoch": 512,
    "micro_batch_size": 4,     # windows per forward/backward; gradients accumulate to model.batch_size
    "positive_oversample": 3.0,
    "max_val_windows": None,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 10,
    "early_stopping_metric": "pr_auc",
    "min_delta": 0.001,
    "min_delta_mode": "rel",
    "lr_scheduler": {"factor": 0.5, "patience": 4, "min_lr": 1.0e-5},
    "threshold_metric": "f2",
    "calibration": "platt",    # platt | temperature | none (fitted on the validation logits)
    "calibrate_temperature": True,  # legacy switch: false disables calibration (= "none")
    "resume": False,
    "seed": None,
    "max_nonfinite_steps": 3,
    "reliability_bins": 10,
    "baseline": deep_merge(BASELINE_DEFAULTS, {}),
    # Skill when only corridor-average rain is known (src/training/areal_skill.py, X6): computed after the test
    # evaluation and published as reports_dir/areal_skill.json with best.pt.
    "areal_skill": {"enabled": True, "members": 8},
}
MODEL_TRAINING_DEFAULTS: dict[str, Any] = {"learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16,
                                           "epochs": 60}
# Early-stopping metric -> optimisation direction.
MONITOR_MODES: dict[str, str] = {"pr_auc": "max", "roc_auc": "max", "f2": "max", "val_loss": "min"}
DEVICES = ("auto", "cpu", "cuda", "mps")
DELTA_MODES = ("rel", "abs")


def _int(key: str, value: Any, minimum: int, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or not math.isfinite(float(value)) or float(value) != int(value) or int(value) < minimum:
        suffix = " (or null)" if allow_none else ""
        raise ConfigError(f"{key} must be a whole number >= {minimum}{suffix}, got {value!r}")
    return int(value)


def _float(key: str, value: Any, low: float, high: float = math.inf, *, allow_none: bool = False,
           open_low: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    number = float(value) if ok else math.nan
    if not ok or not math.isfinite(number) or number > high or number < low or (open_low and number == low):
        bound = f"({low}, {high}]" if open_low else f"[{low}, {high}]"
        raise ConfigError(f"{key} must be a number in {bound}, got {value!r}")
    return number


def _bool(key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false, got {value!r}")
    return value


def _choice(key: str, value: Any, choices: tuple[str, ...] | Mapping[str, Any]) -> str:
    name = str(value).strip().lower()
    if name not in choices:
        raise ConfigError(f"{key} must be one of {list(choices)}, got {value!r}")
    return name


def _micro_batch(value: Any, batch_size: int) -> int:
    """``training.micro_batch_size`` (null => one forward pass per batch), capped at the batch size."""
    micro = _int("training.micro_batch_size", value, 1, allow_none=True)
    if micro is None:
        return batch_size
    if micro > batch_size:
        LOGGER.warning("training.micro_batch_size=%d exceeds model.batch_size=%d; using %d", micro, batch_size,
                       batch_size)
    return min(micro, batch_size)


def _all_or_count(value: Any) -> Any:
    """CLI convention: 0 means "all windows" (null in the config)."""
    return None if value == 0 and not isinstance(value, bool) else value


@dataclass(frozen=True)
class TrainingSettings:
    """Everything the training loop needs, validated."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    num_threads: int | None
    windows_per_epoch: int | None
    micro_batch_size: int
    positive_oversample: float
    max_val_windows: int | None
    grad_clip_norm: float | None
    early_stopping_patience: int
    early_stopping_metric: str
    min_delta: float
    min_delta_mode: str
    lr_factor: float
    lr_patience: int
    min_lr: float
    threshold_metric: str
    calibration: str
    resume: bool
    seed: int
    max_nonfinite_steps: int
    reliability_bins: int
    baseline_enabled: bool
    baseline_max_samples: int
    baseline_max_train_windows: int
    gbdt_enabled: bool
    gbdt_max_iter: int
    gbdt_learning_rate: float
    gbdt_max_leaf_nodes: int
    gbdt_min_samples_leaf: int
    areal_skill_enabled: bool = True
    areal_skill_members: int = 8

    @property
    def calibrate_temperature(self) -> bool:
        """Legacy view: True when a calibration is fitted at all."""
        return self.calibration != "none"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, epochs: int | None = None, device: str | None = None,
                    resume: bool | None = None, windows_per_epoch: int | None = None,
                    max_val_windows: int | None = None) -> "TrainingSettings":
        t = get_section(cfg, "training", TRAINING_DEFAULTS)
        m = get_section(cfg, "model", MODEL_TRAINING_DEFAULTS)
        sched = t["lr_scheduler"] if isinstance(t["lr_scheduler"], Mapping) else {}
        base = t["baseline"] if isinstance(t["baseline"], Mapping) else {}
        sched = {**TRAINING_DEFAULTS["lr_scheduler"], **sched}
        base = deep_merge(BASELINE_DEFAULTS, base)
        gbdt = base["hist_gbdt"] if isinstance(base["hist_gbdt"], Mapping) else {}
        gbdt = {**BASELINE_DEFAULTS["hist_gbdt"], **gbdt}
        areal = t["areal_skill"] if isinstance(t["areal_skill"], Mapping) else {}
        areal = {**TRAINING_DEFAULTS["areal_skill"], **areal}
        seed = t["seed"] if t["seed"] is not None else (cfg.get("project") or {}).get("seed", 42)
        batch_size = _int("model.batch_size", m["batch_size"], 1)
        calibration = _choice("training.calibration", t["calibration"], CALIBRATION_METHODS)
        if not _bool("training.calibrate_temperature", t["calibrate_temperature"]):
            calibration = "none"
        return cls(
            epochs=_int("epochs (model.epochs / --epochs)", m["epochs"] if epochs is None else epochs, 1),
            batch_size=batch_size,
            learning_rate=_float("model.learning_rate", m["learning_rate"], 0.0, 10.0, open_low=True),
            weight_decay=_float("model.weight_decay", m["weight_decay"], 0.0, 1.0),
            device=_choice("training.device", t["device"] if device is None else device, DEVICES),
            num_threads=_int("training.num_threads", t["num_threads"], 1, allow_none=True),
            windows_per_epoch=_int("training.windows_per_epoch", _all_or_count(
                t["windows_per_epoch"] if windows_per_epoch is None else windows_per_epoch), 1, allow_none=True),
            micro_batch_size=_micro_batch(t["micro_batch_size"], batch_size),
            positive_oversample=_float("training.positive_oversample", t["positive_oversample"], 0.0, 1e6,
                                       open_low=True),
            max_val_windows=_int("training.max_val_windows", _all_or_count(
                t["max_val_windows"] if max_val_windows is None else max_val_windows), 1, allow_none=True),
            grad_clip_norm=_float("training.grad_clip_norm", t["grad_clip_norm"], 0.0, 1e9, allow_none=True,
                                  open_low=True),
            early_stopping_patience=_int("training.early_stopping_patience", t["early_stopping_patience"], 0),
            early_stopping_metric=_choice("training.early_stopping_metric", t["early_stopping_metric"],
                                          MONITOR_MODES),
            min_delta=_float("training.min_delta", t["min_delta"], 0.0),
            min_delta_mode=_choice("training.min_delta_mode", t["min_delta_mode"], DELTA_MODES),
            lr_factor=_float("training.lr_scheduler.factor", sched["factor"], 0.0, 0.999, open_low=True),
            lr_patience=_int("training.lr_scheduler.patience", sched["patience"], 0),
            min_lr=_float("training.lr_scheduler.min_lr", sched["min_lr"], 0.0),
            threshold_metric=_choice("training.threshold_metric", t["threshold_metric"], THRESHOLD_METRICS),
            calibration=calibration,
            resume=_bool("training.resume / --resume", t["resume"] if resume is None else resume),
            seed=_int("training.seed / project.seed", seed, 0),
            max_nonfinite_steps=_int("training.max_nonfinite_steps", t["max_nonfinite_steps"], 1),
            reliability_bins=_int("training.reliability_bins", t["reliability_bins"], 1),
            baseline_enabled=_bool("training.baseline.enabled", base["enabled"]),
            baseline_max_samples=_int("training.baseline.max_samples", base["max_samples"], 2),
            baseline_max_train_windows=_int("training.baseline.max_train_windows", base["max_train_windows"], 2),
            gbdt_enabled=_bool("training.baseline.hist_gbdt.enabled", gbdt["enabled"]),
            gbdt_max_iter=_int("training.baseline.hist_gbdt.max_iter", gbdt["max_iter"], 1),
            gbdt_learning_rate=_float("training.baseline.hist_gbdt.learning_rate", gbdt["learning_rate"], 0.0, 1.0,
                                      open_low=True),
            gbdt_max_leaf_nodes=_int("training.baseline.hist_gbdt.max_leaf_nodes", gbdt["max_leaf_nodes"], 2),
            gbdt_min_samples_leaf=_int("training.baseline.hist_gbdt.min_samples_leaf", gbdt["min_samples_leaf"], 1),
            areal_skill_enabled=_bool("training.areal_skill.enabled", areal["enabled"]),
            areal_skill_members=_int("training.areal_skill.members", areal["members"], 1),
        )

    @property
    def monitor_mode(self) -> str:
        return MONITOR_MODES[self.early_stopping_metric]


@dataclass(frozen=True)
class Monitor:
    """The validation metric that drives early stopping, LR scheduling and ``best.pt``.

    An epoch improves on ``best`` when it beats it by more than ``min_delta`` - an absolute
    margin, or with ``relative`` a fraction of ``|best|``. Relative is the default because
    rare-event PR-AUC lives on the scale of the base rate (a val set with 0.002 % floods
    scores ~1e-3), where any fixed absolute margin is either meaningless or unreachable.
    """

    name: str
    mode: str
    min_delta: float = 0.0
    relative: bool = True

    def improved(self, value: float | None, best: float | None) -> bool:
        if value is None or not math.isfinite(value):
            return False
        if best is None or not math.isfinite(best):
            return True
        margin = self.min_delta * abs(best) if self.relative else self.min_delta
        return value > best + margin if self.mode == "max" else value < best - margin
