"""Loading and validating training checkpoints for serving (safe loader, fit checks, calibration).

Checkpoints load only with the safe ``weights_only`` unpickler — a file that needs the full
unpickler is refused, never executed. Only finalized checkpoints are served (``finalized_utc``);
the graph signature, junction order, static / edge features and the graph-attribute digest must
match the current road graph (:class:`~src.inference.errors.CheckpointMismatch` otherwise).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.data_pipeline.features import FeatureScaler, dynamic_feature_names, validate_windows
from src.data_pipeline.graph_io import GraphArrays
from src.inference.errors import (
    FINALIZE_HINT,
    RETRAIN_HINT,
    TRAIN_HINT,
    CheckpointMismatch,
    ModelNotFinalized,
    ModelNotReady,
)
from src.models.stgcn import build_model
from src.training.calibration import checkpoint_calibration, identity_calibration, temperature_of
from src.utils.logger import get_logger

LOGGER = get_logger("src.inference.predictor")

SUPPORTED_FORMAT_VERSIONS = (1, 2)
REQUIRED_CHECKPOINT_KEYS = (  # "architecture" / "static_feature_names" are optional (derived when absent)
    "model_state", "model_config", "feature_names", "rolling_windows_h", "lookback_hours", "seq_len",
    "warmup_steps", "scaler", "graph_signature", "node_ids", "temperature", "threshold",
)
EDGE_FEATURES = ("length", "grade")


def load_checkpoint_file(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Load and structurally validate a training checkpoint (format 1 or 2).

    Only the safe ``weights_only`` loader is used. A file that references Python globals
    outside its allow-list — which is exactly what a malicious pickle looks like — raises
    :class:`ModelNotReady` without ever being unpickled.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelNotReady(f"No trained model at {path}; {TRAIN_HINT}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:  # corrupt bytes, or Python objects beyond tensors and plain values
        raise ModelNotReady(f"Checkpoint {path} is unreadable or not a Namma-Flow checkpoint: the safe (weights-only) "
                            "loader refused it, and the full unpickler is never used because it could run arbitrary "
                            f"code; {TRAIN_HINT}") from exc
    except Exception as exc:  # noqa: BLE001 - torch raises many types for corrupt / truncated files
        raise ModelNotReady(f"Checkpoint {path} is unreadable ({type(exc).__name__}: {exc}); {TRAIN_HINT}") from exc
    if not isinstance(checkpoint, dict):
        raise ModelNotReady(f"Checkpoint {path} holds a {type(checkpoint).__name__}, not a checkpoint dict")
    version = checkpoint.get("format_version", 1)
    if isinstance(version, bool) or version not in SUPPORTED_FORMAT_VERSIONS:
        raise ModelNotReady(f"Checkpoint {path} has format_version {version!r} (this code reads "
                            f"{list(SUPPORTED_FORMAT_VERSIONS)}); {TRAIN_HINT}")
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ModelNotReady(f"Checkpoint {path} is missing keys {missing}; {TRAIN_HINT}")
    return checkpoint


def check_graph_fit(checkpoint: Mapping[str, Any], graph: GraphArrays, static_names: list[str]) -> None:
    expected = checkpoint["graph_signature"]
    actual = graph.signature()
    if not isinstance(expected, Mapping) or any(expected.get(k) != actual[k] for k in ("num_nodes", "num_edges",
                                                                                         "sha256")):
        hint = f"restore the original graph file, or {RETRAIN_HINT}"
        raise CheckpointMismatch(f"The checkpoint was trained on a different road graph ({expected}) than "
                                 f"{actual}; {hint}", hint=hint)
    if [str(n) for n in checkpoint["node_ids"]] != [str(n) for n in graph.node_ids]:
        raise CheckpointMismatch(f"The checkpoint's junction order differs from the road graph; {RETRAIN_HINT}")
    missing = [n for n in static_names if n not in graph.node_attrs or not np.isfinite(graph.node_attrs[n]).all()]
    if graph.num_edges:
        missing += [n for n in EDGE_FEATURES if n not in graph.edge_attrs or not np.isfinite(graph.edge_attrs[n]).all()]
    if missing:
        hint = "re-run python src/data_pipeline/02_elevation_engine.py"
        raise CheckpointMismatch(f"The road graph lacks finite {missing} needed by the model; {hint}", hint=hint)


def check_finalized(checkpoint: Mapping[str, Any], label: str) -> None:
    """Only a checkpoint the trainer published after calibration (``finalized_utc``) is served."""
    if not checkpoint.get("finalized_utc"):
        raise ModelNotFinalized(f"The {label} checkpoint is not finalized (a per-epoch file of an in-progress or "
                                f"interrupted training run, not calibrated); {FINALIZE_HINT}")


def check_attributes(checkpoint: Mapping[str, Any], graph: GraphArrays, static_names: list[str]) -> None:
    """Compare the graph-attribute digest the model was trained on with the current graph's."""
    names = [*static_names, *EDGE_FEATURES]
    stored = checkpoint.get("graph_attributes_sha256")
    if not stored:
        LOGGER.warning("The checkpoint (format v%s) records no graph attribute digest, so it cannot be verified that "
                       "the road graph's %s still match the training data", checkpoint.get("format_version", 1),
                       ", ".join(names))
        return
    try:
        current = graph.attributes_signature(static_names, list(EDGE_FEATURES))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointMismatch(f"Cannot verify the road graph's attributes against the checkpoint ({exc}); "
                                 f"{RETRAIN_HINT}") from exc
    if current != str(stored):
        raise CheckpointMismatch(f"The road graph's attributes ({', '.join(names)}) differ from the ones the model was "
                                 f"trained on (digest {current} vs {stored}: the graph was re-enriched after "
                                 f"training); {RETRAIN_HINT}")


def checkpoint_features(checkpoint: Mapping[str, Any]) -> tuple[FeatureScaler, tuple[int, ...], list[str]]:
    """``(scaler, rolling windows, static feature names)`` after checking they agree with ``feature_names``."""
    try:
        scaler = FeatureScaler.from_dict(checkpoint["scaler"])
        windows = validate_windows(checkpoint["rolling_windows_h"])
    except ValueError as exc:
        raise ModelNotReady(f"Checkpoint feature scaler / rolling windows are invalid: {exc}") from exc
    names = [str(n) for n in checkpoint["feature_names"]]
    dynamic = dynamic_feature_names(windows)
    static = checkpoint.get("static_feature_names")
    static = [str(n) for n in static] if static is not None else names[: max(0, len(names) - len(dynamic))]
    expected = [*static, *dynamic]
    if names != expected or scaler.feature_names != names:
        raise CheckpointMismatch(f"Checkpoint features are inconsistent: feature_names={names}, expected {expected}, "
                                 f"scaler={scaler.feature_names}")
    return scaler, windows, static


def checkpoint_window(checkpoint: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        seq_len, warmup, lookback = (int(checkpoint[k]) for k in ("seq_len", "warmup_steps", "lookback_hours"))
    except (TypeError, ValueError) as exc:
        raise ModelNotReady(f"Checkpoint window settings are not integers: {exc}") from exc
    if not (seq_len > warmup >= 0 and lookback >= 0):
        raise ModelNotReady(f"Checkpoint has invalid seq_len={seq_len} / warmup_steps={warmup} / lookback={lookback}")
    return seq_len, warmup, lookback


def _stored_calibration(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """The checkpoint's calibration dict (v2) or its temperature (v1); invalid values fall back
    (WARNING) to the temperature, then to the uncalibrated ``sigmoid(logit)``."""
    try:
        return checkpoint_calibration(checkpoint)
    except ValueError as exc:
        LOGGER.warning("Checkpoint calibration is invalid (%s)", exc)
    if checkpoint.get("calibration") is not None:
        try:
            return checkpoint_calibration({"temperature": checkpoint.get("temperature")})
        except ValueError:
            pass
    LOGGER.warning("Checkpoint temperature %r is invalid; using uncalibrated probabilities sigmoid(logit)",
                   checkpoint.get("temperature"))
    return identity_calibration("none")


def checkpoint_serving_calibration(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], float, float]:
    """``(calibration, informational temperature 1 / slope, threshold)`` with safe fallbacks (WARNING)."""
    calibration = _stored_calibration(checkpoint)
    try:
        threshold = float(checkpoint["threshold"])
    except (TypeError, ValueError):
        threshold = float("nan")
    if not (math.isfinite(threshold) and 0.0 < threshold < 1.0):
        LOGGER.warning("Checkpoint threshold %r is invalid; using 0.5", checkpoint["threshold"])
        threshold = 0.5
    return calibration, temperature_of(calibration), threshold


def build_checkpoint_model(checkpoint: Mapping[str, Any], n_features: int, device: torch.device) -> torch.nn.Module:
    config = checkpoint["model_config"]
    if not isinstance(config, Mapping):
        raise ModelNotReady(f"Checkpoint model_config must be a mapping, got {type(config).__name__}")
    config = {**config, "architecture": config.get("architecture") or checkpoint.get("architecture") or "gatv2_gru"}
    try:
        model = build_model(config)
    except ValueError as exc:
        raise ModelNotReady(f"Checkpoint model_config is invalid: {exc}") from exc
    if model.node_in_dim != n_features:
        raise CheckpointMismatch(f"Model expects {model.node_in_dim} node features but the checkpoint lists "
                                 f"{n_features}")
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (RuntimeError, TypeError, KeyError) as exc:
        raise CheckpointMismatch(f"Checkpoint weights do not fit the {config['architecture']} model: {exc}") from exc
    return model.to(device).eval()


def warn_config_drift(checkpoint: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    stored = checkpoint.get("dataset_config_hash")
    if not stored:
        return
    try:
        from src.data_pipeline.dataset import dataset_config_hash

        current = dataset_config_hash(cfg)
    except Exception as exc:  # noqa: BLE001 - diagnostics only; never block inference
        LOGGER.debug("Could not compute the dataset config hash: %s", exc)
        return
    if current != stored:
        LOGGER.warning("The data configuration (rain field / hydrology / features) changed since the model was trained "
                       "(hash %s != %s); predictions may not match the training distribution", current, stored)
