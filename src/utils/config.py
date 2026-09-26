"""Configuration loading, validation and path resolution.

Every module reads its own section through :func:`get_section`, which merges the
module's defaults with whatever ``config/config.yaml`` provides, so a partial or
missing section never crashes a stage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

ENV_CONFIG = "NAMMA_FLOW_CONFIG"
ENV_OFFLINE = "NAMMA_FLOW_OFFLINE"

REQUIRED_SECTIONS = ("project", "paths")
TRUTHY = {"1", "true", "yes", "on"}
# Mapping-valued settings that an override replaces wholesale instead of deep-merging into
# (an OSM tag filter: merging would silently keep default keys the user dropped on purpose).
REPLACED_PATHS: tuple[tuple[str, ...], ...] = (("drains", "osm_tags"),)


class ConfigError(ValueError):
    """Raised when the configuration is missing or invalid."""


def project_root() -> Path:
    return PROJECT_ROOT


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` recursively merged over ``base``."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, merge and validate the YAML configuration.

    Resolution order for the file: explicit ``path`` > ``$NAMMA_FLOW_CONFIG`` >
    ``config/config.yaml``. ``overrides`` are deep-merged last, except the
    :data:`REPLACED_PATHS` settings, which an override replaces as a whole. Setting
    ``NAMMA_FLOW_OFFLINE=1`` forces ``project.offline = true``.
    """
    config_path = Path(path or os.environ.get(ENV_CONFIG) or DEFAULT_CONFIG_PATH)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {config_path} is not valid YAML: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    cfg = _replace_paths(deep_merge(raw, overrides or {}), overrides or {})
    if os.environ.get(ENV_OFFLINE, "").strip().lower() in TRUTHY:
        cfg = deep_merge(cfg, {"project": {"offline": True}})
    cfg["_config_path"] = str(config_path)
    validate_config(cfg)
    return cfg


def _replace_paths(cfg: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the :data:`REPLACED_PATHS` values of ``overrides`` verbatim (not merged)."""
    for path in REPLACED_PATHS:
        source: Any = overrides
        for key in path:
            source = source.get(key) if isinstance(source, Mapping) else None
        if source is None:
            continue
        target = cfg
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = copy.deepcopy(source)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Fail fast on structural problems that would otherwise surface deep in a stage."""
    for section in REQUIRED_SECTIONS:
        if not isinstance(cfg.get(section), Mapping):
            raise ConfigError(f"Config section '{section}' is missing or not a mapping")

    bbox = cfg.get("region", {}).get("bbox")
    if bbox is not None:
        from src.utils.geo import validate_bbox

        try:
            validate_bbox(bbox)
        except ValueError as exc:
            raise ConfigError(f"region.bbox is invalid: {exc}") from exc

    _validate_dataset_shape(cfg)

    loss = cfg.get("loss", {})
    if loss:
        alpha = loss.get("alpha", 0.5)
        gamma = loss.get("gamma", 0.0)
        if alpha is not None and not 0.0 <= float(alpha) <= 1.0:
            raise ConfigError("loss.alpha must lie in [0, 1]")
        if float(gamma) < 0.0:
            raise ConfigError("loss.gamma must be >= 0")


def _whole(name: str, value: Any) -> int:
    """``value`` as an int, or :class:`ConfigError` when it is not a whole number (booleans rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{name} must be a whole number, got {value!r}")
    if not isinstance(value, numbers.Integral) and not (math.isfinite(value) and float(value).is_integer()):
        raise ConfigError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def effective_dataset_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``dataset`` section merged over the dataset module's defaults: the values stage 04 really uses.

    Imported lazily: :mod:`src.data_pipeline.dataset_config` itself imports this module.
    """
    from src.data_pipeline.dataset_config import DEFAULTS as DATASET_DEFAULTS

    return get_section(cfg, "dataset", DATASET_DEFAULTS)


def _validate_dataset_shape(cfg: Mapping[str, Any]) -> None:
    """Check ``model.node_in_dim`` and the window lengths against the EFFECTIVE dataset settings (F4-02).

    Keys omitted from the YAML take the dataset module's defaults, exactly as stage 04 and training do, so a
    partial ``dataset`` section with a matching ``node_in_dim`` loads and a mismatching one fails here.
    """
    model = cfg.get("model") or {}
    if not isinstance(model, Mapping):
        raise ConfigError("Config section 'model' must be a mapping")
    dataset = effective_dataset_section(cfg)
    if model.get("node_in_dim") is not None:
        from src.data_pipeline.features import feature_names

        names = feature_names({**cfg, "dataset": dataset})
        node_in_dim = _whole("model.node_in_dim", model["node_in_dim"])
        if node_in_dim != len(names):
            raise ConfigError(
                f"model.node_in_dim={model['node_in_dim']} but the dataset features imply {len(names)} "
                f"(static_features + precipitation + rolling_windows_h, module defaults included): {names}"
            )
    seq_len = _whole("dataset.seq_len", dataset.get("seq_len", 1))
    warmup = _whole("dataset.warmup_steps", dataset.get("warmup_steps", 0))
    if seq_len <= 0 or warmup < 0 or warmup >= seq_len:
        raise ConfigError(f"dataset.seq_len ({seq_len}) must be > warmup_steps ({warmup}) >= 0")
    windows = dataset.get("rolling_windows_h") or []
    if isinstance(windows, (str, bytes)) or not isinstance(windows, (list, tuple)):
        raise ConfigError(f"dataset.rolling_windows_h must be a list of whole hours, got {windows!r}")
    hours = [_whole("dataset.rolling_windows_h", w) for w in windows]
    lookback = _whole("dataset.lookback_hours", dataset.get("lookback_hours", 0))
    if hours and lookback < max(hours):
        raise ConfigError(
            f"dataset.lookback_hours ({lookback}) must cover the longest rolling window ({max(hours)} h)"
        )


def get_section(
    cfg: Mapping[str, Any],
    name: str,
    defaults: Mapping[str, Any] | None = None,
    replace: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return ``cfg[name]`` merged over ``defaults`` (a fresh dict, safe to modify).

    Nested mappings are deep-merged, except the keys listed in ``replace``: a user value for
    one of those replaces the default wholesale (e.g. an OSM tag filter, where merging would
    silently re-add default keys the user left out on purpose).
    """
    section = cfg.get(name) or {}
    if not isinstance(section, Mapping):
        raise ConfigError(f"Config section '{name}' must be a mapping")
    merged = deep_merge(defaults or {}, section)
    for key in replace:
        if key in section:
            merged[key] = copy.deepcopy(section[key])
    return merged


def resolve_path(cfg: Mapping[str, Any], key: str, default: str | None = None) -> Path:
    """Resolve ``cfg['paths'][key]`` against the project root."""
    value = cfg.get("paths", {}).get(key, default)
    if value is None:
        raise ConfigError(f"paths.{key} is not configured")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_offline(cfg: Mapping[str, Any]) -> bool:
    if os.environ.get(ENV_OFFLINE, "").strip().lower() in TRUTHY:
        return True
    return bool(cfg.get("project", {}).get("offline", False))


def config_hash(cfg: Mapping[str, Any], sections: tuple[str, ...]) -> str:
    """Stable short hash of the given sections, used to detect stale artifacts."""
    subset = {name: cfg.get(name) for name in sections}
    payload = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
