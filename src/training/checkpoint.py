"""Checkpoint format (version 2), persistence and model reconstruction.

A checkpoint is a plain ``dict`` saved with :func:`torch.save` whose values are only
tensors and Python builtins (no numpy objects), so it loads with the safe
``torch.load(..., weights_only=True)`` - the ONLY loader used: a file that needs the full
unpickler is rejected as "not a Namma-Flow checkpoint" (it could run arbitrary code).
Keys (:data:`CHECKPOINT_KEYS`)::

    format_version, model_state, architecture, model_config, feature_names,
    static_feature_names, rolling_windows_h, lookback_hours, seq_len, warmup_steps,
    scaler, graph_signature, graph_attributes_sha256, node_ids, calibration,
    temperature, threshold, metrics, epoch, best_score, history, optimizer_state,
    scheduler_state, rng_state, config_hash, dataset_config_hash, created_utc

plus ``finalized_utc`` (set only on the published ``best.pt`` by the trainer's finalisation;
per-epoch bests go to ``best_candidate.pt`` and never carry it) and informational extras
(``kind``, ``train_state``, ``label_source``, ``flood_threshold_m``, ``timezone``,
``graph_provenance``, ``edge_features``, ``run_id``, ``dataset_identity``, ``val_set``). The
published ``best.pt`` also carries ``published_id`` (:func:`published_id`) and
``candidate_created_utc``; its ``created_utc`` is the publication time, so re-finalising the
same weights with another calibration or threshold is a new identity. Paths are never stored
absolute (:func:`to_builtin`). Version 2 added ``calibration``
(``{"method", "slope", "intercept"}``: ``p = sigmoid(slope * logit + intercept)``, see
:mod:`src.training.calibration`), ``graph_attributes_sha256`` and ``finalized_utc``;
``temperature`` is kept as the informational ``1 / slope``. Version 1 files (no
``calibration``: ``p = sigmoid(logit / temperature)``) still load. Inference needs only
:data:`INFERENCE_KEYS` (+ ``calibration`` in v2); the rest supports resuming and reporting.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from src.models.stgcn import build_model
from src.training.calibration import validate_calibration
from src.training.reports import portable_path
from src.utils.config import resolve_path
from src.utils.logger import get_logger
from src.utils.runtime import atomic_torch_save

LOGGER = get_logger(__name__)

CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS: tuple[int, ...] = (1, 2)
BEST_NAME, LAST_NAME, CANDIDATE_NAME = "best.pt", "last.pt", "best_candidate.pt"
CHECKPOINT_KEYS: tuple[str, ...] = (
    "format_version", "model_state", "architecture", "model_config", "feature_names", "static_feature_names",
    "rolling_windows_h", "lookback_hours", "seq_len", "warmup_steps", "scaler", "graph_signature",
    "graph_attributes_sha256", "node_ids", "calibration", "temperature", "threshold", "metrics", "epoch",
    "best_score", "history", "optimizer_state", "scheduler_state", "rng_state", "config_hash",
    "dataset_config_hash", "created_utc",
)
FINAL_KEYS: tuple[str, ...] = ("finalized_utc",)
INFERENCE_KEYS: tuple[str, ...] = (
    "model_state", "model_config", "feature_names", "rolling_windows_h", "lookback_hours", "seq_len",
    "warmup_steps", "scaler", "graph_signature", "node_ids", "temperature", "threshold",
)
TRAIN_HINT = "train a model first: python src/training/train.py"


class CheckpointError(ValueError):
    """Raised when a checkpoint file is unreadable, incomplete or inconsistent."""


@dataclass(frozen=True)
class CheckpointPaths:
    """Checkpoint locations under ``paths.checkpoint_dir``: the published (finalised) best
    model, the resumable last state, and this run's per-epoch best candidate."""

    directory: Path
    best: Path
    last: Path
    candidate: Path


def checkpoint_paths(cfg: Mapping[str, Any]) -> CheckpointPaths:
    directory = resolve_path(cfg, "checkpoint_dir", "artifacts/checkpoints")
    return CheckpointPaths(directory, directory / BEST_NAME, directory / LAST_NAME, directory / CANDIDATE_NAME)


# --------------------------------------------------------------------------- serialisation helpers


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy / torch scalars and arrays, paths and tuples to JSON-safe builtins.

    Non-finite floats become ``None`` (strict JSON has no NaN); dict keys become strings;
    paths become :func:`~src.training.reports.portable_path` (project-relative or file name).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, torch.Tensor):
        return to_builtin(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, Path):
        return portable_path(value)  # never an absolute path in a checkpoint or report (F3-07)
    if isinstance(value, Mapping):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    return str(value)


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def capture_rng_state(device: str | torch.device | None = None) -> dict[str, Any]:
    """Python, numpy and torch (CPU + CUDA + MPS) RNG states in a ``weights_only``-safe form.

    The MPS generator (dropout masks of a model on ``device="mps"``) is captured when MPS
    is available and ``device`` is ``None`` or an MPS device; a CPU run does not touch it.
    """
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {"name": str(name), "keys": [int(k) for k in keys], "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the development machine
        state["cuda"] = torch.cuda.get_rng_state_all()
    wants_mps = device is None or torch.device(device).type == "mps"
    if wants_mps and _mps_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except RuntimeError as exc:  # pragma: no cover - MPS driver failure
            LOGGER.warning("Could not capture the MPS RNG state (%s)", exc)
    return state


def _as_tuple(value: Any) -> Any:
    return tuple(_as_tuple(v) for v in value) if isinstance(value, (list, tuple)) else value


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore RNG states captured by :func:`capture_rng_state` (missing parts are skipped with a WARNING)."""
    if not isinstance(state, Mapping):
        LOGGER.warning("Checkpoint has no RNG state; the resumed run will not be bit-identical")
        return
    try:
        if "python" in state:
            random.setstate(_as_tuple(state["python"]))
        if "numpy" in state:
            np_state = state["numpy"]
            np.random.set_state((np_state["name"], np.asarray(np_state["keys"], dtype=np.uint32), int(np_state["pos"]),
                                 int(np_state["has_gauss"]), float(np_state["cached_gaussian"])))
        if "torch" in state:
            torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
        if "cuda" in state and torch.cuda.is_available():  # pragma: no cover - no CUDA here
            torch.cuda.set_rng_state_all(state["cuda"])
        if "mps" in state:
            if _mps_available():
                torch.mps.set_rng_state(torch.as_tensor(state["mps"], dtype=torch.uint8).cpu())
            else:
                LOGGER.warning("The checkpoint holds an MPS RNG state but MPS is not available; it is not restored")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        LOGGER.warning("Could not restore the RNG state (%s); the resumed run will not be bit-identical", exc)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of ``model.state_dict()`` (safe to keep while training continues)."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


# --------------------------------------------------------------------------- save / load / validate


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write ``checkpoint`` (a crash never leaves a truncated file behind)."""
    return atomic_torch_save(dict(checkpoint), Path(path))


def _check_number(ckpt: Mapping[str, Any], key: str, low: float, high: float, source: str) -> None:
    value = ckpt[key]
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if not ok or not low <= float(value) <= high:
        raise CheckpointError(f"{source}: '{key}' must be a number in [{low}, {high}], got {value!r}")


def _check_version(ckpt: Mapping[str, Any], source: str) -> int:
    version = ckpt.get("format_version")
    if version is None:
        LOGGER.warning("%s has no format_version; assuming 1 (temperature calibration)", source)
        return 1
    if isinstance(version, bool) or version not in SUPPORTED_FORMAT_VERSIONS:
        raise CheckpointError(f"{source} has format_version {version!r}; this code reads versions "
                              f"{list(SUPPORTED_FORMAT_VERSIONS)} - retrain with: python src/training/train.py")
    return int(version)


def validate_checkpoint(ckpt: Any, source: str = "checkpoint") -> dict[str, Any]:
    """Check structure and inference-critical values; returns ``ckpt`` as a dict.

    Missing training-only keys are tolerated (DEBUG log) so hand-made inference checkpoints
    work; a missing ``format_version`` is assumed to be 1 with a WARNING. Version 2 must
    carry a valid ``calibration`` dict.
    """
    if not isinstance(ckpt, Mapping):
        raise CheckpointError(f"{source} holds a {type(ckpt).__name__}, not a checkpoint dict - {TRAIN_HINT}")
    version = _check_version(ckpt, source)
    required = INFERENCE_KEYS + (("calibration",) if version >= 2 else ())
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise CheckpointError(f"{source} is missing keys {missing} - {TRAIN_HINT}")
    optional = [key for key in CHECKPOINT_KEYS if key not in ckpt and key != "format_version"]
    if optional:
        LOGGER.debug("%s lacks training-only keys %s", source, optional)
    for key in ("model_state", "model_config", "scaler", "graph_signature"):
        if not isinstance(ckpt[key], Mapping):
            raise CheckpointError(f"{source}: '{key}' must be a dict, got {type(ckpt[key]).__name__}")
    names = ckpt["feature_names"]
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise CheckpointError(f"{source}: 'feature_names' must be a non-empty list of strings, got {names!r}")
    _check_number(ckpt, "temperature", 1e-6, 1e6, source)
    _check_number(ckpt, "threshold", 0.0, 1.0, source)
    if version >= 2:
        try:
            validate_calibration(ckpt["calibration"])
        except ValueError as exc:
            raise CheckpointError(f"{source}: invalid 'calibration' ({exc}) - {TRAIN_HINT}") from exc
    for key in ("seq_len", "warmup_steps", "lookback_hours"):
        _check_number(ckpt, key, 0, 1e6, source)
    if int(ckpt["warmup_steps"]) >= int(ckpt["seq_len"]):
        raise CheckpointError(f"{source}: warmup_steps={ckpt['warmup_steps']} must be < seq_len={ckpt['seq_len']}")
    return dict(ckpt)


def published_id(ckpt: Mapping[str, Any]) -> str:
    """sha256[:16] over the weights, calibration, threshold, run id and epoch of a checkpoint.

    Two published models share it only when they would predict identically: re-finalising
    the same weights with another calibration or threshold gives a new id (F2-04).
    """
    digest = hashlib.sha256()
    for key in sorted((ckpt.get("model_state") or {}).keys()):
        value = ckpt["model_state"][key]
        digest.update(str(key).encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    identity = {k: to_builtin(ckpt.get(k)) for k in ("calibration", "threshold", "run_id", "epoch")}
    digest.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def is_finalized(ckpt: Mapping[str, Any]) -> bool:
    """True for a checkpoint published by the trainer's finalisation (calibrated, evaluated)."""
    return bool(ckpt.get("finalized_utc"))


def _unpickling_reason(exc: BaseException) -> str:
    """The safe loader's own diagnosis, without torch's advice to retry with the full unpickler."""
    for line in str(exc).splitlines():
        if "WeightsUnpickler error" in line or "Unsupported global" in line:
            return line.strip()
    return type(exc).__name__


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a checkpoint written by the trainer.

    Only the safe ``weights_only`` loader is used: a file that references other Python
    globals (which is what a malicious pickle looks like) is rejected, never unpickled.
    Raises ``FileNotFoundError`` when absent and :class:`CheckpointError` when unreadable,
    not a Namma-Flow checkpoint, or incomplete.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}; {TRAIN_HINT}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise CheckpointError(f"{path} is not a Namma-Flow checkpoint: the safe (weights-only) loader refused it "
                              f"({_unpickling_reason(exc)}). The file is corrupt or truncated, or it holds Python "
                              "objects the trainer never writes, so it is not loaded with the full unpickler - "
                              f"{TRAIN_HINT}") from exc
    except Exception as exc:  # noqa: BLE001 - torch raises many types for truncated/corrupt files
        raise CheckpointError(f"Could not read checkpoint {path} ({type(exc).__name__}: {exc}); the file is corrupt "
                              f"or truncated - {TRAIN_HINT}") from exc
    return validate_checkpoint(ckpt, source=f"checkpoint {path}")


def model_from_checkpoint(ckpt: Mapping[str, Any] | str | Path) -> nn.Module:
    """Rebuild the trained model (CPU, eval mode) from a checkpoint dict or path."""
    if isinstance(ckpt, (str, Path)):
        ckpt = load_checkpoint(ckpt)
    ckpt = validate_checkpoint(ckpt)
    config = dict(ckpt["model_config"])
    config.setdefault("architecture", ckpt.get("architecture", "gatv2_gru"))
    try:
        model = build_model(config)
        model.load_state_dict(ckpt["model_state"], strict=True)
    except (ValueError, RuntimeError, TypeError) as exc:
        raise CheckpointError(f"Checkpoint weights do not fit the model described by its model_config "
                              f"({type(exc).__name__}: {exc})") from exc
    model.eval()
    return model
