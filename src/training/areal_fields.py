"""Junction rain fields for the areal-only skill evaluation (:mod:`src.training.areal_skill`).

The training labels come from the physics teacher driven by ONE stochastic junction rain field
per event, seeded by ``rainfall_field.seed`` and the event's start hour
(:mod:`src.data_pipeline.rain_field`). This module regenerates that field and independent
realisations of it EXACTLY the way stage 04 does - :func:`downscale_rainfall` over the FULL
weather record (an event's cells depend on where the event starts, which a sub-range can cut)
- and gathers the rows of a dataset split's stored hours.

Member ``k`` of a rain-field ensemble uses ``rainfall_field.seed = base + (offset + k) * stride``
with ``stride = inference.field_seed_stride`` (X5; offset 0, member 0 is the label field). The
evaluation uses offset 1, so no member ever reuses the label seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_pipeline import rain_field
from src.data_pipeline.dataset_config import DatasetSettings
from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays, load_graph
from src.utils.config import deep_merge, get_section, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "DEFAULT_FIELD_SEED_STRIDE", "DEFAULT_PHYSICS_SOFTNESS_M", "FIELD_SEED_OFFSET", "ArealSkillError", "WeatherInputs",
    "field_member_seed", "junction_field", "label_seed", "load_weather_inputs", "member_config", "physics_graph",
    "PHYSICS_SPINUP_H", "as_numpy", "physics_softness", "physics_span", "scored_values", "teacher_run",
]

FIELD_SEED_OFFSET = 1              # evaluation members start at offset 1: never the label seed
DEFAULT_FIELD_SEED_STRIDE = 7919   # inference.field_seed_stride (X5)
DEFAULT_PHYSICS_SOFTNESS_M = 0.03  # inference.physics_softness_m (the app's physics baseline)
DEFAULT_LABEL_SEED = 42            # rainfall_field.seed default
PHYSICS_SPINUP_H = 720             # teacher re-run from 30 days before a split (see physics_span)


class ArealSkillError(RuntimeError):
    """The areal-skill evaluation cannot run; reported as ``status: "unavailable"`` with the reason."""


def label_seed(cfg: Mapping[str, Any]) -> int:
    """``rainfall_field.seed``: the seed of the field that produced the training labels."""
    return int(get_section(cfg, "rainfall_field", {"seed": DEFAULT_LABEL_SEED})["seed"])


def _stride(cfg: Mapping[str, Any]) -> int:
    value = get_section(cfg, "inference", {"field_seed_stride": DEFAULT_FIELD_SEED_STRIDE})["field_seed_stride"]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ArealSkillError(f"inference.field_seed_stride must be an integer >= 1, got {value!r}")
    return int(value)


def field_member_seed(cfg: Mapping[str, Any], member: int, offset: int = FIELD_SEED_OFFSET) -> int:
    """``rainfall_field.seed`` of ensemble member ``member`` (X5 scheme: base + (offset + k) * stride)."""
    if isinstance(member, bool) or not isinstance(member, (int, np.integer)) or member < 0:
        raise ValueError(f"member must be an integer >= 0, got {member!r}")
    return label_seed(cfg) + (int(offset) + int(member)) * _stride(cfg)


def member_config(cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """A copy of ``cfg`` whose rain field uses ``seed`` (the caller's config is not modified)."""
    return deep_merge(cfg, {"rainfall_field": {"seed": int(seed)}})


@dataclass(frozen=True)
class WeatherInputs:
    """The full hourly areal record and where each stored hour of a split lies in it."""

    timestamps: pd.DatetimeIndex
    areal: np.ndarray
    rows: np.ndarray            # record row of every stored payload hour [H]
    lon: np.ndarray
    lat: np.ndarray
    fingerprint: str


def as_numpy(value: Any, dtype: Any) -> np.ndarray:
    """A numpy view/copy of a tensor or array-like with ``dtype``."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _record_rows(record: pd.DatetimeIndex, payload_epoch: np.ndarray) -> np.ndarray:
    epoch = record.as_unit("s").asi8.astype(np.int64)
    rows = np.searchsorted(epoch, payload_epoch)
    inside = rows < epoch.size
    found = np.zeros(payload_epoch.size, dtype=bool)
    found[inside] = epoch[rows[inside]] == payload_epoch[inside]
    if not found.all():
        raise ArealSkillError(f"The weather record lacks {int((~found).sum())} of the dataset's {payload_epoch.size} "
                              "hours (weather.start_date / end_date changed since the build?); rebuild the datasets "
                              "with: python src/data_pipeline/04_dataset_builder.py")
    return rows.astype(np.int64)


def load_weather_inputs(cfg: Mapping[str, Any], payload: Mapping[str, Any]) -> WeatherInputs:
    """The areal record stage 04 used (read from the cache - offline, never refetched) aligned to ``payload``."""
    from src.data_pipeline.provenance import load_weather_record

    offline = deep_merge(cfg, {"project": {"offline": True}})
    try:
        record = load_weather_record(offline, DatasetSettings.from_config(offline))
    except (OSError, ValueError, RuntimeError) as exc:  # ConfigError / WeatherFormatError / DatasetError
        raise ArealSkillError(f"The weather record is unavailable ({type(exc).__name__}: {exc})") from exc
    rows = _record_rows(record.timestamps, as_numpy(payload["timestamps"], np.int64))
    return WeatherInputs(record.timestamps, np.asarray(record.areal, dtype=np.float64), rows,
                         as_numpy(payload["lon"], np.float64), as_numpy(payload["lat"], np.float64), record.fingerprint)


def junction_field(inputs: WeatherInputs, cfg: Mapping[str, Any], seed: int) -> np.ndarray:
    """The FULL-record junction rain field ``float32 [T, N]`` of ``seed`` (stage-04 procedure)."""
    try:
        field = rain_field.downscale_rainfall(inputs.areal, inputs.timestamps, inputs.lon, inputs.lat,
                                              member_config(cfg, seed))
    except ValueError as exc:  # includes ConfigError
        raise ArealSkillError(f"Could not downscale the rain record ({exc})") from exc
    return np.asarray(field, dtype=np.float32)


def scored_values(values: np.ndarray, starts: np.ndarray, seq_len: int, warmup_steps: int) -> np.ndarray:
    """``values[start + step]`` for every window and scored (non-warm-up) step, flattened [W * S * N]."""
    steps = np.arange(int(warmup_steps), int(seq_len), dtype=np.int64)
    index = np.asarray(starts, dtype=np.int64)[:, None] + steps[None, :]
    return np.asarray(values)[index.reshape(-1)].reshape(-1)


def physics_graph(cfg: Mapping[str, Any], payload: Mapping[str, Any]) -> GraphArrays:
    """``paths.graph_file`` as arrays; must be the graph the payload was built on (topology and order)."""
    path = resolve_path(cfg, "graph_file")
    try:
        arrays = graph_to_arrays(load_graph(path))
    except (OSError, ValueError, KeyError) as exc:  # FileNotFoundError / GraphFormatError
        raise ArealSkillError(f"The road graph {path.name} is unreadable ({type(exc).__name__}: {exc})") from exc
    expected = dict(payload.get("graph_signature") or {})
    if {k: arrays.signature().get(k) for k in expected} != expected:
        raise ArealSkillError(f"The road graph {path.name} is not the one the dataset was built on "
                              f"({arrays.signature()} vs {expected})")
    if [str(n) for n in arrays.node_ids] != [str(n) for n in payload["node_ids"]]:
        raise ArealSkillError(f"The junction order of {path.name} differs from the dataset's")
    return arrays


def physics_span(inputs: WeatherInputs, spinup_h: int = PHYSICS_SPINUP_H) -> slice:
    """Record rows the teacher is re-run over: ``spinup_h`` hours before the split's first stored hour
    (dry start) to its last one. The teacher's storage memory is a few hours (ponds recede within
    ~6 h, antecedent window 24 h), so this reproduces the full-record simulation - 0 of 2.5 M
    test / val node-hours differ on the real record even with a 72 h spin-up - at a fraction of
    the memory; :mod:`src.training.areal_skill` re-checks it on every run."""
    return slice(max(0, int(inputs.rows.min()) - int(spinup_h)), int(inputs.rows.max()) + 1)


def teacher_run(arrays: GraphArrays, cfg: Mapping[str, Any], rain_window: np.ndarray, local_rows: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, depth)`` of the physics teacher at ``local_rows`` of ``rain_window`` (simulated from dry)."""
    from src.hydrology.simulator import simulate_labels

    labels, depth = simulate_labels(arrays, rain_window, cfg)
    return np.asarray(labels)[local_rows], np.asarray(depth)[local_rows]


def physics_softness(cfg: Mapping[str, Any]) -> float:
    """``inference.physics_softness_m``: width of the physics baseline's depth -> probability map."""
    value = get_section(cfg, "inference", {"physics_softness_m": DEFAULT_PHYSICS_SOFTNESS_M})["physics_softness_m"]
    try:
        softness = float(value)
    except (TypeError, ValueError) as exc:
        raise ArealSkillError(f"inference.physics_softness_m must be a number, got {value!r}") from exc
    if not np.isfinite(softness) or softness <= 0:
        raise ArealSkillError(f"inference.physics_softness_m must be > 0, got {value!r}")
    return softness
