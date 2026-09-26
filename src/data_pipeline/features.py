"""The single definition of Namma-Flow node and edge features (contract schema 2.5).

Training (:mod:`src.data_pipeline.dataset`) and inference both build model inputs with
:func:`build_node_features` and a fitted :class:`FeatureScaler`, so there is no
train/serve skew. Feature order (:func:`feature_names`)::

    [*dataset.static_features, "precip_mm_h", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"]

* **Static** junction attributes (elevation, distance to drain, relative elevation) are
  z-scored.
* **Dynamic** rainfall features - the hour's rain and trailing sums over
  ``dataset.rolling_windows_h`` (hours ``(t - w, t]``, current hour included) - are
  ``log1p``-transformed, then z-scored.
* **Edges** ``[length, grade]``: ``log1p(length)`` z-scored and ``grade / max_abs_grade``
  clipped to ``[-1, 1]``.

Standard deviations are floored at ``1e-6``; non-finite values become 0 after scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.utils.config import ConfigError, get_section
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "static_features": ["elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink"],
    "rolling_windows_h": [3, 6, 12, 24],
}
PRECIP_FEATURE = "precip_mm_h"
EDGE_FEATURES = ("length", "grade")
STD_FLOOR = 1e-6
SCALER_VERSION = 1

_ARRAY_FIELDS = ("static_mean", "static_std", "dynamic_mean", "dynamic_std")
_SCALAR_FIELDS = ("edge_len_mean", "edge_len_std", "max_abs_grade")
_NAME_FIELDS = ("static_names", "dynamic_names")
# Heavy-tailed static attributes that are log1p-transformed before z-scoring (upstream junction
# counts span 1..O(100)); every other static attribute is z-scored as is.
LOG_STATIC_FEATURES = frozenset({"flow_accumulation"})


# --------------------------------------------------------------------------- names


def rolling_feature_name(window_h: int) -> str:
    """Name of the trailing ``window_h``-hour rain sum feature, e.g. ``rain_6h_mm``."""
    return f"rain_{int(window_h)}h_mm"


def validate_windows(windows: Any, error: type[ValueError] = ValueError) -> tuple[int, ...]:
    """Return rolling windows as a tuple of distinct positive ints, raising ``error`` otherwise."""
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Iterable):
        raise error(f"rolling windows must be a list of positive whole hours, got {windows!r}")
    out: list[int] = []
    for value in windows:
        is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        if not is_number or not math.isfinite(float(value)) or float(value) != int(value) or int(value) < 1:
            raise error(f"rolling window {value!r} must be a positive whole number of hours")
        out.append(int(value))
    if len(set(out)) != len(out):
        raise error(f"rolling windows must be distinct, got {out}")
    return tuple(out)


def dynamic_feature_names(windows: Sequence[int]) -> list[str]:
    """``["precip_mm_h", "rain_<w>h_mm", ...]`` for the given rolling windows."""
    return [PRECIP_FEATURE, *(rolling_feature_name(w) for w in validate_windows(windows))]


def _static_names(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ConfigError(f"dataset.static_features must be a list of node attribute names, got {value!r}")
    names = list(value)
    if any(not isinstance(n, str) or not n.strip() for n in names):
        raise ConfigError(f"dataset.static_features entries must be non-empty strings, got {names!r}")
    if len(set(names)) != len(names):
        raise ConfigError(f"dataset.static_features contains duplicates: {names}")
    return names


def feature_names(cfg: Mapping[str, Any]) -> list[str]:
    """Ordered node feature names: static features, then the hour's rain and rolling sums."""
    section = get_section(cfg, "dataset", DEFAULTS)
    static = _static_names(section["static_features"])
    dynamic = [PRECIP_FEATURE, *(rolling_feature_name(w)
                                 for w in validate_windows(section["rolling_windows_h"], ConfigError))]
    clash = sorted(set(static) & set(dynamic))
    if clash:
        raise ConfigError(f"dataset.static_features {clash} clash with the dynamic rainfall feature names")
    return [*static, *dynamic]


# --------------------------------------------------------------------------- dynamic rainfall features


def _rain_matrix(rain: Any) -> np.ndarray:
    """Float64 copy of a ``[T, N]`` rain array; non-finite / negative values become 0 (WARNING)."""
    values = np.array(rain, dtype=np.float64, copy=True)
    if values.ndim != 2:
        raise ValueError(f"rain must be a 2-D [T, N] array (hours x nodes), got shape {values.shape}")
    bad = ~np.isfinite(values)
    bad[~bad] = values[~bad] < 0
    if bad.any():
        LOGGER.warning("rain: %d non-finite/negative values treated as 0 mm", int(bad.sum()))
        values[bad] = 0.0
    return values


def rolling_sums(rain: np.ndarray, windows: Sequence[int]) -> np.ndarray:
    """Trailing sums ``float32 [T, N, len(windows)]`` over hours ``(t - w, t]`` (current hour included).

    The first ``w - 1`` hours are partial sums over the available history. Accumulation is in
    float64 and a window without rain is exactly 0.
    """
    wins = validate_windows(windows)
    values = _rain_matrix(rain)
    n_hours, n_nodes = values.shape
    out = np.zeros((n_hours, n_nodes, len(wins)), dtype=np.float32)
    if n_hours == 0 or n_nodes == 0 or not wins:
        return out
    cumulative = np.zeros((n_hours + 1, n_nodes), dtype=np.float64)
    np.cumsum(values, axis=0, out=cumulative[1:])
    upper = np.arange(1, n_hours + 1)
    for k, window in enumerate(wins):
        lower = np.maximum(upper - window, 0)
        out[..., k] = np.maximum(cumulative[upper] - cumulative[lower], 0.0)
    return out


def _check_lookback(lookback: Any, n_rows: int) -> int:
    is_int = isinstance(lookback, (int, np.integer)) and not isinstance(lookback, bool)
    if not is_int or not 0 <= int(lookback) <= n_rows:
        raise ValueError(f"lookback must be a whole number of hours in [0, {n_rows}] (rows of rain), got {lookback!r}")
    return int(lookback)


def dynamic_features(rain_with_lookback: np.ndarray, lookback: int, windows: Sequence[int]) -> np.ndarray:
    """``[L + T, N]`` rain → ``float32 [T, N, 1 + len(windows)]`` for the last ``T`` hours.

    Column 0 is the hour's rain, then one trailing sum per window. The ``L`` lookback hours
    only feed the rolling sums (sums are partial if ``L`` is shorter than a window).
    """
    values = _rain_matrix(rain_with_lookback)
    start = _check_lookback(lookback, values.shape[0])
    sums = rolling_sums(values, windows)[start:]
    return np.concatenate([values[start:, :, None].astype(np.float32), sums], axis=2)


def dynamic_features_at(rain: np.ndarray, hours: np.ndarray, windows: Sequence[int],
                        node_chunk: int = 128) -> np.ndarray:
    """Raw dynamic features ``float32 [len(hours), N, 1 + len(windows)]`` at selected hours of a long record.

    Identical to :func:`dynamic_features` at those hours (same trailing sums over the full
    ``rain [T, N]`` history) but computed in node chunks so a multi-year record never needs a
    ``[T, N, W]`` array at once.
    """
    values = np.asarray(rain)
    if values.ndim != 2:
        raise ValueError(f"rain must be a 2-D [T, N] array (hours x nodes), got shape {values.shape}")
    idx = np.asarray(hours, dtype=np.int64).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= values.shape[0]):
        raise ValueError(f"hours must index the {values.shape[0]} rows of rain")
    wins = validate_windows(windows)
    out = np.empty((idx.size, values.shape[1], 1 + len(wins)), dtype=np.float32)
    for lo in range(0, values.shape[1], max(1, int(node_chunk))):
        cols = slice(lo, lo + max(1, int(node_chunk)))
        block = _rain_matrix(values[:, cols])
        out[:, cols, 0] = block[idx]
        out[:, cols, 1:] = rolling_sums(block, wins)[idx]
    return out


# --------------------------------------------------------------------------- scaler


def _column_stats(values: np.ndarray, label: str, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean / std over finite values (std floored); all-NaN columns → (0, 1) with WARNING."""
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    mean = np.divide(filled.sum(axis=0), count, out=np.zeros(values.shape[1]), where=count > 0)
    squared = np.where(finite, (filled - mean) ** 2, 0.0)
    var = np.divide(squared.sum(axis=0), count, out=np.ones(values.shape[1]), where=count > 0)
    std = np.maximum(np.sqrt(var), STD_FLOOR)
    empty = count == 0
    if empty.any():
        LOGGER.warning("FeatureScaler: no finite values for %s feature(s) %s; using mean 0 / std 1",
                       label, [names[i] for i in np.flatnonzero(empty)])
        mean[empty], std[empty] = 0.0, 1.0
    return mean, std


def _names(names: Sequence[str] | None, count: int, prefix: str, field: str) -> tuple[str, ...]:
    if names is None:
        return tuple(f"{prefix}_{i}" for i in range(count))
    out = tuple(str(n) for n in names)
    if len(out) != count:
        raise ValueError(f"{field} has {len(out)} names but the data has {count} columns")
    return out


def _log_columns(values: np.ndarray, flags: Sequence[bool]) -> np.ndarray:
    """Copy of ``values`` [N, S] with ``log1p(max(x, 0))`` applied to the flagged columns."""
    out = np.array(values, dtype=np.float64, copy=True)
    for column, flag in enumerate(flags):
        if flag:
            out[:, column] = np.log1p(np.maximum(out[:, column], 0.0))
    return out


def _finite_to_zero(values: np.ndarray) -> np.ndarray:
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32)


def _readonly_vector(value: Any, field: str) -> np.ndarray:
    try:
        arr = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FeatureScaler.{field} must be numeric: {exc}") from exc
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError(f"FeatureScaler.{field} must be a finite 1-D vector, got {value!r}")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class FeatureScaler:
    """Normalisation statistics fitted on the TRAIN split (immutable; arrays are read-only)."""

    static_mean: np.ndarray
    static_std: np.ndarray
    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    edge_len_mean: float
    edge_len_std: float
    max_abs_grade: float
    static_names: tuple[str, ...]
    dynamic_names: tuple[str, ...]
    static_log: tuple[bool, ...] = ()  # per static column: log1p before z-scoring (empty = none)

    def __post_init__(self) -> None:
        for field in _ARRAY_FIELDS:
            object.__setattr__(self, field, _readonly_vector(getattr(self, field), field))
        for field in _SCALAR_FIELDS:
            try:
                value = float(getattr(self, field))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"FeatureScaler.{field} must be numeric: {exc}") from exc
            if not math.isfinite(value) or (field != "edge_len_mean" and value <= 0):
                rule = "finite" if field == "edge_len_mean" else "finite and > 0"
                raise ValueError(f"FeatureScaler.{field} must be {rule}, got {getattr(self, field)!r}")
            object.__setattr__(self, field, value)
        for field in _NAME_FIELDS:
            object.__setattr__(self, field, tuple(str(n) for n in getattr(self, field)))
        flags = tuple(bool(f) for f in self.static_log) or (False,) * len(self.static_names)
        if len(flags) != len(self.static_names):
            raise ValueError(f"FeatureScaler.static_log has {len(flags)} flags for {len(self.static_names)} "
                             "static features")
        object.__setattr__(self, "static_log", flags)
        for kind in ("static", "dynamic"):
            mean, std = getattr(self, f"{kind}_mean"), getattr(self, f"{kind}_std")
            names = getattr(self, f"{kind}_names")
            if not len(mean) == len(std) == len(names):
                raise ValueError(f"FeatureScaler {kind} statistics disagree: {len(mean)} means, "
                                 f"{len(std)} stds, {len(names)} names")
            if (std <= 0).any():
                raise ValueError(f"FeatureScaler.{kind}_std must be positive, got {std.tolist()}")

    # ------------------------------------------------------------------ construction
    @classmethod
    def fit(
        cls,
        static: np.ndarray,
        dynamic_samples: np.ndarray,
        edge_length: np.ndarray,
        max_abs_grade: float,
        *,
        static_names: Sequence[str] | None = None,
        dynamic_names: Sequence[str] | None = None,
    ) -> "FeatureScaler":
        """Fit on static node attributes ``[N, S]``, raw dynamic samples ``[..., D]`` and edge lengths ``[E]``."""
        static_arr = np.asarray(static, dtype=np.float64)
        if static_arr.ndim != 2:
            raise ValueError(f"static must be a 2-D [N, S] array, got shape {static_arr.shape}")
        dyn = np.asarray(dynamic_samples, dtype=np.float64)
        if dyn.ndim < 2:
            raise ValueError(f"dynamic_samples must be [..., D] with D features, got shape {dyn.shape}")
        dyn = np.log1p(np.maximum(dyn.reshape(-1, dyn.shape[-1]), 0.0))
        lengths = np.asarray(edge_length, dtype=np.float64)
        if lengths.ndim != 1:
            raise ValueError(f"edge_length must be a 1-D [E] array, got shape {lengths.shape}")
        grade = float(max_abs_grade)
        if not math.isfinite(grade) or grade <= 0:
            raise ValueError(f"max_abs_grade must be a finite positive number, got {max_abs_grade!r}")
        s_names = _names(static_names, static_arr.shape[1], "static", "static_names")
        d_names = _names(dynamic_names, dyn.shape[1], "dynamic", "dynamic_names")
        s_log = tuple(name in LOG_STATIC_FEATURES for name in s_names)
        s_mean, s_std = _column_stats(_log_columns(static_arr, s_log), "static", s_names)
        if dyn.shape[0] == 0:
            LOGGER.warning("FeatureScaler: no dynamic samples; using mean 0 / std 1 for %s", list(d_names))
            d_mean, d_std = np.zeros(dyn.shape[1]), np.ones(dyn.shape[1])
        else:
            d_mean, d_std = _column_stats(dyn, "dynamic", d_names)
        log_len = np.log1p(lengths[np.isfinite(lengths) & (lengths >= 0)])
        if log_len.size == 0:
            LOGGER.debug("FeatureScaler: no edge lengths (graph without edges); identity edge scaling")
            len_mean, len_std = 0.0, 1.0
        else:
            len_mean, len_std = float(log_len.mean()), max(float(log_len.std()), STD_FLOOR)
        return cls(s_mean, s_std, d_mean, d_std, len_mean, len_std, grade, s_names, d_names, s_log)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureScaler":
        """Rebuild a scaler from :meth:`to_dict` output (validated)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"FeatureScaler.from_dict needs a mapping, got {type(data).__name__}")
        version = data.get("version", SCALER_VERSION)
        if version != SCALER_VERSION:
            raise ValueError(f"FeatureScaler dict has version {version!r}; this code reads version {SCALER_VERSION}")
        required = (*_ARRAY_FIELDS, *_SCALAR_FIELDS, *_NAME_FIELDS)
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"FeatureScaler dict is missing keys {missing}")
        # static_log is optional: scalers written before it existed never log-transformed.
        return cls(**{key: data[key] for key in required}, static_log=tuple(data.get("static_log") or ()))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (stored in datasets and checkpoints)."""
        out: dict[str, Any] = {"version": SCALER_VERSION}
        out.update({field: getattr(self, field).tolist() for field in _ARRAY_FIELDS})
        out.update({field: float(getattr(self, field)) for field in _SCALAR_FIELDS})
        out.update({field: list(getattr(self, field)) for field in _NAME_FIELDS})
        out["static_log"] = list(self.static_log)
        return out

    # ------------------------------------------------------------------ properties / equality
    @property
    def num_static(self) -> int:
        return len(self.static_names)

    @property
    def num_dynamic(self) -> int:
        return len(self.dynamic_names)

    @property
    def feature_names(self) -> list[str]:
        return [*self.static_names, *self.dynamic_names]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureScaler):
            return NotImplemented
        return (
            all(np.array_equal(getattr(self, f), getattr(other, f)) for f in _ARRAY_FIELDS)
            and all(getattr(self, f) == getattr(other, f) for f in (*_SCALAR_FIELDS, *_NAME_FIELDS, "static_log"))
        )

    __hash__ = None  # type: ignore[assignment]  # mutable-looking value object: not hashable

    # ------------------------------------------------------------------ transforms
    def transform_static(self, static: np.ndarray) -> np.ndarray:
        """``[N, S]`` raw static attributes → z-scores (float32; non-finite → 0)."""
        values = np.array(static, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[1] != self.num_static:
            raise ValueError(f"static features must be [N, {self.num_static}] ({list(self.static_names)}), "
                             f"got shape {values.shape}")
        return _finite_to_zero((_log_columns(values, self.static_log) - self.static_mean) / self.static_std)

    def transform_dynamic(self, dynamic: np.ndarray) -> np.ndarray:
        """``[..., D]`` raw rain features → ``(log1p(max(x, 0)) - mean) / std`` (float32; non-finite → 0)."""
        values = np.asarray(dynamic, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.num_dynamic:
            raise ValueError(f"dynamic features must have {self.num_dynamic} columns ({list(self.dynamic_names)}), "
                             f"got shape {values.shape}")
        return _finite_to_zero((np.log1p(np.maximum(values, 0.0)) - self.dynamic_mean) / self.dynamic_std)

    def transform_edges(self, length: np.ndarray, grade: np.ndarray) -> np.ndarray:
        """Edge ``length`` (m) and ``grade`` ``[E]`` → normalised ``float32 [E, 2]``."""
        lengths = np.asarray(length, dtype=np.float64).reshape(-1)
        grades = np.asarray(grade, dtype=np.float64).reshape(-1)
        if lengths.shape != grades.shape:
            raise ValueError(f"length and grade must have the same length, got {lengths.size} and {grades.size}")
        scaled_len = (np.log1p(np.maximum(lengths, 0.0)) - self.edge_len_mean) / self.edge_len_std
        scaled_grade = np.clip(grades / self.max_abs_grade, -1.0, 1.0)
        return _finite_to_zero(np.stack([scaled_len, scaled_grade], axis=1))


# --------------------------------------------------------------------------- model inputs


def _check_scaler_windows(scaler: FeatureScaler, windows: Sequence[int]) -> None:
    expected = dynamic_feature_names(windows)
    if len(expected) != scaler.num_dynamic:
        raise ValueError(f"{len(expected) - 1} rolling windows {list(windows)} give {len(expected)} dynamic "
                         f"features but the scaler was fitted on {scaler.num_dynamic} {list(scaler.dynamic_names)}")
    if scaler.dynamic_names[0] == PRECIP_FEATURE and tuple(expected) != scaler.dynamic_names:
        raise ValueError(f"rolling windows {list(windows)} do not match the scaler's dynamic features "
                         f"{list(scaler.dynamic_names)}")


def build_node_features(
    static_raw: np.ndarray,
    rain_with_lookback: np.ndarray,
    lookback: int,
    windows: Sequence[int],
    scaler: FeatureScaler,
) -> np.ndarray:
    """Normalised model inputs ``float32 [T, N, F]`` in :func:`feature_names` order.

    ``static_raw`` is ``[N, S]`` raw node attributes, ``rain_with_lookback`` the node rain
    ``[L + T, N]`` whose first ``lookback`` rows are history for the rolling sums.
    """
    if not isinstance(scaler, FeatureScaler):
        raise TypeError(f"scaler must be a FeatureScaler (use FeatureScaler.from_dict), got {type(scaler).__name__}")
    static = np.asarray(static_raw, dtype=np.float64)
    if static.ndim != 2:
        raise ValueError(f"static_raw must be a 2-D [N, S] array, got shape {static.shape}")
    rain = np.asarray(rain_with_lookback)
    if rain.ndim == 2 and rain.shape[1] != static.shape[0]:
        raise ValueError(f"static_raw has {static.shape[0]} nodes but rain has {rain.shape[1]} nodes")
    _check_scaler_windows(scaler, windows)
    static_z = scaler.transform_static(static)
    dynamic_z = scaler.transform_dynamic(dynamic_features(rain, lookback, windows))
    n_hours, n_nodes, _ = dynamic_z.shape
    out = np.empty((n_hours, n_nodes, scaler.num_static + scaler.num_dynamic), dtype=np.float32)
    out[..., : scaler.num_static] = static_z[None]
    out[..., scaler.num_static:] = dynamic_z
    return out
