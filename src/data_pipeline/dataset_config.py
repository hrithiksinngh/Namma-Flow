"""Validated ``dataset`` settings, payload constants and split paths for stage 04.

Shared by the dataset builder (:mod:`src.data_pipeline.dataset`), window selection
(:mod:`src.data_pipeline.windows`), input fingerprints (:mod:`src.data_pipeline.provenance`)
and the payload loader (:mod:`src.data_pipeline.sequence_dataset`).

Splits are whole calendar years of a window's LAST hour: ``test`` if in
``dataset.test_years``, ``val`` if in ``dataset.val_years``, else ``train``. ``test_years``
may be empty (no test split); the two held-out year sets must not overlap.

**Dataset config hash** (:func:`dataset_config_hash`, stored in every payload as ``config_hash``
and in checkpoints as ``dataset_config_hash``) covers exactly the settings that stage 04 applies
itself (F1-03):

* the whole ``rainfall_field``, ``hydrology``, ``labels`` and ``dataset`` sections, as written in
  the config (:data:`HASH_SECTIONS`);
* ``project.timezone`` (local calendar: seasons, split years), ``region.bbox`` (anchors the
  rain-field placement domain), ``elevation.max_abs_grade`` (edge-grade scaling of the feature
  scaler) and the weather-record keys ``weather.{provider, start_date, end_date, latitude,
  longitude, models, bias_correction_factor, max_precip_mm_h, max_fill_gap_hours,
  synthetic_scale}`` - effective values, defaults applied (:data:`HASH_KEYS`).

Everything upstream is covered by content fingerprints instead: the road graph by its topology
signature and attribute digest, the weather record by ``weather_fingerprint``, observed reports
by ``reports_fingerprint``; enrichment settings (elevation / drains / region) by the graph's own
``enrichment_config_hash``. Fetch-only keys (URLs, timeouts, retries, forecast settings) are not
hashed, so changing them never marks datasets or models stale. The value is a sha256[:16] of
sorted-key JSON, stable across runs, processes and machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data_pipeline.features import EDGE_FEATURES, feature_names, validate_windows
from src.data_pipeline.weather_schema import DEFAULTS as WEATHER_DEFAULTS
from src.utils.config import ConfigError, config_hash, get_section, project_root, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, 2)  # version 1 payloads load but always count as stale (no input fingerprints)
DEFAULTS: dict[str, Any] = {
    "season_months": [5, 6, 7, 8, 9, 10, 11],
    # scored steps per window = seq_len - warmup_steps = 12 = val_stride_h: val/test scored hours tile exactly
    "seq_len": 16, "warmup_steps": 4, "lookback_hours": 24, "rolling_windows_h": [3, 6, 12, 24],
    "train_stride_h": 6, "val_stride_h": 12, "val_years": [2022], "test_years": [2024],
    "wet_window_min_mm": 5.0, "dry_window_keep_frac": 0.05,
    "static_features": ["elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink"],
    "edge_features": ["length", "grade"],
    "scaler_max_samples": 2_000_000,  # max node-hour samples for the dynamic feature statistics
    "seed": None,                     # null => project.seed (dry-window sampling, scaler subsample)
}
DEFAULT_MAX_ABS_GRADE = 0.3  # elevation.max_abs_grade default (edge-grade scaling)
#: Sections hashed whole, as written in the config.
HASH_SECTIONS = ("rainfall_field", "hydrology", "labels", "dataset")
#: Individual keys hashed with their effective (default-applied) values: {section: {key: default}}.
HASH_KEYS: dict[str, dict[str, Any]] = {
    "project": {"timezone": "Asia/Kolkata"},
    "region": {"bbox": None},
    "elevation": {"max_abs_grade": DEFAULT_MAX_ABS_GRADE},
    "weather": {key: WEATHER_DEFAULTS[key] for key in (
        "provider", "start_date", "end_date", "latitude", "longitude", "models", "bias_correction_factor",
        "max_precip_mm_h", "max_fill_gap_hours", "synthetic_scale")},
}
SPLITS = ("train", "val", "test")
SPLIT_PATH_KEYS = {"train": "train_dataset", "val": "val_dataset", "test": "test_dataset"}
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
DEFAULT_TEST_DATASET = "data/processed/test_dataset.pt"
DEFAULT_VAL_DATASET = "data/processed/val_dataset.pt"
TEST_DATASET_NAME = "test_dataset.pt"
BUILD_HINT = "rebuild it with: python src/data_pipeline/04_dataset_builder.py --force"


class DatasetError(ValueError):
    """Raised when datasets cannot be built or a dataset payload is invalid."""


# --------------------------------------------------------------------------- validators


def _int(name: str, value: Any, minimum: int) -> int:
    ok = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    if not ok or not np.isfinite(float(value)) or float(value) != int(value) or int(value) < minimum:
        raise ConfigError(f"dataset.{name} must be a whole number >= {minimum}, got {value!r}")
    return int(value)


def _float(name: str, value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"dataset.{name} must be a number, got {value!r}") from exc
    if isinstance(value, bool) or not (np.isfinite(number) and lo <= number <= hi):
        raise ConfigError(f"dataset.{name} must lie in [{lo}, {hi}], got {value!r}")
    return number


def _int_list(name: str, value: Any, lo: int, hi: int, allow_empty: bool = False) -> tuple[int, ...]:
    if value is None and allow_empty:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or (len(value) == 0 and not allow_empty):
        kind = "a list" if allow_empty else "a non-empty list"
        raise ConfigError(f"dataset.{name} must be {kind} of integers, got {value!r}")
    items = tuple(_int(name, v, lo) for v in value)
    if any(v > hi for v in items):
        raise ConfigError(f"dataset.{name} values must lie in [{lo}, {hi}], got {list(value)}")
    return tuple(sorted(set(items)))


def _held_out_years(section: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    val = _int_list("val_years", section["val_years"], 1, 9999)
    test = _int_list("test_years", section["test_years"], 1, 9999, allow_empty=True)
    overlap = sorted(set(val) & set(test))
    if overlap:
        raise ConfigError(f"dataset.val_years {list(val)} and dataset.test_years {list(test)} overlap in {overlap}; "
                          "a year can be either the validation (model selection / calibration) or the held-out test "
                          "year, not both")
    return val, test


def _warn_on_val_tiling(seq_len: int, warmup: int, val_stride: int) -> None:
    scored = seq_len - warmup
    if val_stride < scored:
        LOGGER.warning("dataset.val_stride_h=%d is shorter than the %d scored steps per window (seq_len %d - "
                       "warmup_steps %d): consecutive val/test windows score the same hours twice, which double-counts "
                       "them in every metric; set val_stride_h = seq_len - warmup_steps", val_stride, scored, seq_len,
                       warmup)
    elif val_stride > scored:
        LOGGER.info("dataset.val_stride_h=%d exceeds the %d scored steps per window: %d hour(s) between consecutive "
                    "val/test windows are never scored", val_stride, scored, val_stride - scored)


@dataclass(frozen=True)
class DatasetSettings:
    """Validated ``dataset`` section (plus the project seed and timezone)."""

    season_months: tuple[int, ...]
    seq_len: int
    warmup_steps: int
    lookback_hours: int
    rolling_windows_h: tuple[int, ...]
    train_stride_h: int
    val_stride_h: int
    val_years: tuple[int, ...]
    test_years: tuple[int, ...]
    wet_window_min_mm: float
    dry_window_keep_frac: float
    static_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    scaler_max_samples: int
    seed: int
    timezone: str
    feature_names: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DatasetSettings":
        s = get_section(cfg, "dataset", DEFAULTS)
        project = cfg.get("project") or {}
        names = tuple(feature_names(cfg))
        windows = validate_windows(s["rolling_windows_h"], ConfigError)
        static = names[: len(names) - 1 - len(windows)]
        if not static:
            raise ConfigError("dataset.static_features must name at least one node attribute")
        seq_len, warmup = _int("seq_len", s["seq_len"], 1), _int("warmup_steps", s["warmup_steps"], 0)
        if warmup >= seq_len:
            raise ConfigError(f"dataset.warmup_steps={warmup} must be smaller than dataset.seq_len={seq_len}")
        lookback = _int("lookback_hours", s["lookback_hours"], 0)
        if windows and lookback < max(windows):
            raise ConfigError(f"dataset.lookback_hours={lookback} must cover the longest rolling window {max(windows)}")
        edges = tuple(s["edge_features"]) if isinstance(s["edge_features"], Sequence) else ()
        if edges != EDGE_FEATURES:
            raise ConfigError(f"dataset.edge_features must be {list(EDGE_FEATURES)} (model.edge_dim=2), got "
                              f"{s['edge_features']!r}")
        val_stride = _int("val_stride_h", s["val_stride_h"], 1)
        val_years, test_years = _held_out_years(s)
        _warn_on_val_tiling(seq_len, warmup, val_stride)
        seed = s["seed"] if s["seed"] is not None else project.get("seed", 42)
        return cls(
            season_months=_int_list("season_months", s["season_months"], 1, 12),
            seq_len=seq_len, warmup_steps=warmup, lookback_hours=lookback, rolling_windows_h=windows,
            train_stride_h=_int("train_stride_h", s["train_stride_h"], 1), val_stride_h=val_stride,
            val_years=val_years, test_years=test_years,
            wet_window_min_mm=_float("wet_window_min_mm", s["wet_window_min_mm"], 0.0, 1e6),
            dry_window_keep_frac=_float("dry_window_keep_frac", s["dry_window_keep_frac"], 0.0, 1.0),
            static_features=static, edge_features=edges,
            scaler_max_samples=_int("scaler_max_samples", s["scaler_max_samples"], 1),
            seed=_int("seed", seed, 0), timezone=str(project.get("timezone") or "Asia/Kolkata"),
            feature_names=names,
        )

    @property
    def scored_steps(self) -> int:
        """Steps per window that are scored (after the GRU warm-up)."""
        return self.seq_len - self.warmup_steps

    @property
    def has_test_split(self) -> bool:
        return bool(self.test_years)

    @property
    def splits(self) -> tuple[str, ...]:
        """Splits this configuration builds (``test`` only when ``test_years`` is non-empty)."""
        return SPLITS if self.has_test_split else SPLITS[:2]

    def split_codes(self, years: np.ndarray) -> np.ndarray:
        """Split code (int8, :data:`SPLIT_CODES`: 0 train, 1 val, 2 test) of each calendar year in ``years``."""
        years = np.asarray(years)
        codes = np.zeros(years.shape, dtype=np.int8)
        codes[np.isin(years, self.val_years)] = SPLIT_CODES["val"]
        codes[np.isin(years, self.test_years)] = SPLIT_CODES["test"]
        return codes


def dataset_hash_inputs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Exactly what :func:`dataset_config_hash` hashes (see the module docstring)."""
    subset: dict[str, Any] = {name: cfg.get(name) for name in HASH_SECTIONS}
    for name, keys in HASH_KEYS.items():
        section = get_section(cfg, name, keys)
        subset[f"{name}.keys"] = {key: section.get(key) for key in keys}
    return subset


def dataset_config_hash(cfg: Mapping[str, Any]) -> str:
    """sha256[:16] of :func:`dataset_hash_inputs`: the settings stage 04 applies itself (stored in payloads)."""
    subset = dataset_hash_inputs(cfg)
    return config_hash(subset, tuple(subset))


def _test_path(cfg: Mapping[str, Any], val_path: Path) -> Path:
    """``paths.test_dataset``; falls back to ``test_dataset.pt`` next to the val dataset.

    The fallback also applies when ``paths.test_dataset`` still holds the shipped default while
    ``paths.val_dataset`` was relocated (a config overriding only some paths, e.g. a test or
    sandbox config): the three splits of one build always stay together instead of one of them
    landing in (or being deleted from) the project's real ``data/processed`` directory.
    """
    configured = (cfg.get("paths") or {}).get("test_dataset")
    beside_val = val_path.with_name(TEST_DATASET_NAME)
    if configured is None:
        return beside_val
    default_val = project_root() / DEFAULT_VAL_DATASET
    if Path(str(configured)) == Path(DEFAULT_TEST_DATASET) and val_path.resolve() != default_val.resolve():
        LOGGER.debug("paths.test_dataset is the default but paths.val_dataset was relocated; using %s", beside_val)
        return beside_val
    return resolve_path(cfg, "test_dataset")


def dataset_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Payload paths ``{"train", "val", "test"}`` (the single resolver; consumers should use it)."""
    train, val = resolve_path(cfg, "train_dataset"), resolve_path(cfg, "val_dataset")
    return {"train": train, "val": val, "test": _test_path(cfg, val)}
