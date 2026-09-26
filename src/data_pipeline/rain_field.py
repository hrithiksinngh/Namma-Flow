"""Stochastic convective downscaling of areal rainfall to individual road junctions.

Reanalysis / forecast rainfall is one number per hour for the whole corridor (schema 2.2).
Bengaluru's damaging storms, however, are convective: a few kilometre-scale cells that
drift across the city, so one junction receives a cloudburst while another 3 km away stays
much drier. This module turns the areal series ``[T]`` into a node-level field ``[T, N]``
(schema 2.3) that keeps the areal value as its node mean in every hour.

Model (a simplified moving-cell / cell-lifecycle model in the spirit of the spatial-temporal
Neyman-Scott rectangular-pulse family used for stochastic rainfall, e.g. Cowpertwait 1995,
Northrop 1998, reduced to what an hourly corridor-mean record can constrain)
-----------------------------------------------------------------------------------------
* **Events** are maximal runs of wet hours (areal > 0), allowing up to ``max_dry_gap_h`` dry
  hours inside an event (missing timestamps count as dry). An event is one storm system with
  one steering velocity (speed in ``advection_speed_mps``, random heading): cells embedded in
  the same mid-level flow move together.
* **Cells.** The storm carries ``n_cells`` cell *slots*. Each slot hosts a sequence of cells
  (generations): a cell lives ``cell_lifetime_h`` hours, its intensity rising linearly from
  zero to its peak amplitude (``amplitude_range``) at mid-life and decaying back to zero
  (growth - maturity - dissipation). A new cell is born in the slot every half-lifetime, so
  the envelopes of overlapping generations sum to exactly one at every instant: each slot
  always represents one cell-equivalent of convection and the field never "switches off"
  inside an event. Generations are staggered by a random phase per slot.
* **Where cells are.** Each cell is Gaussian with radius (standard deviation) in
  ``cell_radius_m``. At mid-life (its mature stage) it sits at a uniformly random point of
  the corridor bounding box grown by ``domain_margin_factor x max radius`` and it moves with
  the steering velocity, so it crosses the corridor near maturity and is weaker and farther
  away when young or dissipating. The domain is *not* periodic: a cell that leaves the
  corridor is gone, and the next generation replaces it (no artificial wrap-around that
  would smear a streak over the whole corridor within an hour).
* **Hourly accumulation.** Hourly rain is the mean over ``substeps_per_hour`` sub-steps
  (an hourly total of a moving cell is a streak, not a snapshot). Overlapping cells combine as
  ``1 - prod(1 - a_c e_c g_c)`` so the intensity stays in [0, 1].
* **Conditioning on the observed areal rain** (``normalise_intensity``). A wet hour in the
  record means rain *did* fall on the corridor, so its convective share fell somewhere on it.
  The hour's intensity pattern is therefore rescaled so that its most-affected junction
  reaches the amplitude of the strongest cell active in that hour; otherwise hour-averaging
  and far-away cells would leave only flat, low-intensity tails and the field would be almost
  uniform (the fix-round review measured a within-hour coefficient of variation of 0.08).
* ``field = b(t) + (1 - b(t)) * intensity`` (stratiform share + convective share), normalised
  to node mean 1, clipped to ``multiplier_clip`` and renormalised, then multiplied by the areal
  value. Dry hours are exactly zero. The stratiform share is ``b = background_fraction``, or,
  with ``stratiform_rate_mm_h = r0`` set, ``b(t) = background_fraction * r0 / (r0 + areal(t))``:
  light, widespread rain is mostly stratiform while intense hours are dominated by
  convective cores (the convective fraction of tropical rain rises steeply with rain rate,
  e.g. Houze 1997; Steiner et al. 1995), so an ERA5 hour of 10 mm/h over the corridor is a
  few heavy cores rather than 10 mm/h everywhere.

Consistency guarantee
---------------------
The field of an hour depends only on the node coordinates, the configuration and the start
timestamp of the event containing it — never on the event's length, its later hours or on
other events (every random number is drawn from a generator seeded with
``(rainfall_field.seed, event start hour[, slot])``, and generation ``k`` always uses the same
words of that generator's state however many generations a call needs). Any sub-range whose
first hour is not inside an event that is already in progress (i.e. it starts at an event
start or in a dry spell longer than ``max_dry_gap_h``) therefore reproduces the full-series
field exactly; use :func:`find_rain_events` to pad a sub-range back to its event start when
exact reproduction matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.config import ConfigError, get_section
from src.utils.geo import bbox_center, lonlat_to_local_xy, validate_bbox
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Calibrated defaults (mirrored in the ``rainfall_field`` section of config/config.yaml); see
# src.hydrology.calibrate for the resulting junction-rain statistics on the 2018-2024 record.
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "n_cells": 3,
    "cell_radius_m": [400.0, 1200.0],
    "advection_speed_mps": [1.0, 6.0],
    "background_fraction": 0.2,
    "stratiform_rate_mm_h": 2.0,
    "multiplier_clip": [0.05, 4.0],
    "seed": 42,
    "max_dry_gap_h": 2,
    "substeps_per_hour": 4,
    "domain_margin_factor": 0.5,
    "amplitude_range": [0.5, 1.0],
    "cell_lifetime_h": 2.0,
    "normalise_intensity": True,
}

SECONDS_PER_HOUR = 3600
_CHUNK_ELEMENTS = 2_000_000     # hours x nodes processed per vectorised block (~16 MB float64)
_CLIP_ITERATIONS = 12
_MIN_FIELD_MEAN = 1e-12
_MIN_PEAK_INTENSITY = 1e-9      # below this an hour has no usable convective pattern -> uniform
_MAX_NORMALISATION_GAIN = 20.0  # cells far from the corridor are not stretched into steep edges
_SEED_MODULUS = 2**63
_DRAWS_PER_CELL = 4             # peak x, peak y, radius, amplitude
_UINT53 = float(2**53)


# --------------------------------------------------------------------------- parameters


def _as_range(name: str, value: Any, *, allow_zero: bool) -> tuple[float, float]:
    """Parse ``[lo, hi]`` (or a scalar meaning ``[v, v]``) and validate ``0 (<)= lo <= hi``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo = hi = float(value)
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        try:
            lo, hi = float(value[0]), float(value[1])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"rainfall_field.{name} must be numeric, got {value!r}") from exc
    else:
        raise ConfigError(f"rainfall_field.{name} must be a number or a [min, max] pair, got {value!r}")
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ConfigError(f"rainfall_field.{name} must satisfy min <= max (finite), got {value!r}")
    if lo < 0 or (lo == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"rainfall_field.{name} values must be {bound}, got {value!r}")
    return lo, hi


def _as_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigError(f"rainfall_field.{name} must be an integer, got {value!r}")
    if not math.isfinite(float(value)) or float(value) != int(value) or int(value) < minimum:
        raise ConfigError(f"rainfall_field.{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _as_float(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"rainfall_field.{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"rainfall_field.{name} must be a number, got {value!r}") from exc
    if not (np.isfinite(number) and lo <= number <= hi):
        raise ConfigError(f"rainfall_field.{name} must lie in [{lo}, {hi}], got {value!r}")
    return number


def _optional_positive(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _as_float(name, value, 1e-6, 1e6)


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ConfigError(f"rainfall_field.{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class RainFieldParams:
    """Validated ``rainfall_field`` configuration (see the module docstring for the meaning)."""

    enabled: bool
    n_cells: int
    cell_radius_m: tuple[float, float]
    advection_speed_mps: tuple[float, float]
    background_fraction: float
    multiplier_clip: tuple[float, float]
    seed: int
    max_dry_gap_h: int
    substeps_per_hour: int
    domain_margin_factor: float
    amplitude_range: tuple[float, float]
    cell_lifetime_h: float
    normalise_intensity: bool
    stratiform_rate_mm_h: float | None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RainFieldParams":
        """Build (and validate) the parameters from ``cfg['rainfall_field']`` merged over defaults."""
        section = get_section(cfg, "rainfall_field", DEFAULTS)
        clip = _as_range("multiplier_clip", section["multiplier_clip"], allow_zero=True)
        if not clip[0] <= 1.0 <= clip[1]:
            raise ConfigError(f"rainfall_field.multiplier_clip must contain 1.0, got {list(clip)}")
        amplitude = _as_range("amplitude_range", section["amplitude_range"], allow_zero=True)
        if amplitude[1] > 1.0:
            raise ConfigError("rainfall_field.amplitude_range must lie within [0, 1]")
        return cls(
            enabled=bool(section["enabled"]),
            n_cells=_as_int("n_cells", section["n_cells"], 0),
            cell_radius_m=_as_range("cell_radius_m", section["cell_radius_m"], allow_zero=False),
            advection_speed_mps=_as_range("advection_speed_mps", section["advection_speed_mps"], allow_zero=True),
            background_fraction=_as_float("background_fraction", section["background_fraction"], 0.0, 1.0),
            multiplier_clip=clip,
            seed=_as_int("seed", section["seed"], 0),
            max_dry_gap_h=_as_int("max_dry_gap_h", section["max_dry_gap_h"], 0),
            substeps_per_hour=_as_int("substeps_per_hour", section["substeps_per_hour"], 1),
            domain_margin_factor=_as_float("domain_margin_factor", section["domain_margin_factor"], 0.0, 100.0),
            amplitude_range=amplitude,
            cell_lifetime_h=_as_float("cell_lifetime_h", section["cell_lifetime_h"], 0.25, 48.0),
            normalise_intensity=_as_bool("normalise_intensity", section["normalise_intensity"]),
            stratiform_rate_mm_h=_optional_positive("stratiform_rate_mm_h", section["stratiform_rate_mm_h"]),
        )

    def background_for(self, areal_mm_h: np.ndarray) -> np.ndarray:
        """Stratiform share of each hour ``[H, 1]``: constant, or decreasing with the areal rain rate."""
        rate = np.asarray(areal_mm_h, dtype=np.float64).reshape(-1, 1)
        if self.stratiform_rate_mm_h is None:
            return np.full_like(rate, self.background_fraction)
        return self.background_fraction * self.stratiform_rate_mm_h / (self.stratiform_rate_mm_h + rate)

    @property
    def half_life_s(self) -> float:
        """Spacing of successive cell generations in a slot (half the lifetime), in seconds."""
        return 0.5 * self.cell_lifetime_h * SECONDS_PER_HOUR


# --------------------------------------------------------------------------- events


@dataclass(frozen=True)
class RainEvents:
    """Segmentation of an hourly series into rain events.

    ``start_idx`` / ``end_idx`` are inclusive positions of each event's first and last wet
    hour, ``start_hour`` the UTC epoch hour of the first wet hour (the event's seed) and
    ``event_of_hour[t]`` the event id of hour ``t`` (``-1`` outside every event).
    """

    start_idx: np.ndarray
    end_idx: np.ndarray
    start_hour: np.ndarray
    event_of_hour: np.ndarray

    @property
    def num_events(self) -> int:
        return int(self.start_idx.size)


def _epoch_seconds(timestamps: Any, tz: str) -> np.ndarray:
    """UTC epoch seconds of ``timestamps`` (tz-naive values are local time in ``tz``)."""
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError(f"timestamps must be a pandas DatetimeIndex, got {type(timestamps).__name__}")
    stamps = timestamps.tz_localize(tz) if timestamps.tz is None else timestamps
    if stamps.hasnans:
        raise ValueError("timestamps must not contain NaT")
    seconds = stamps.tz_convert("UTC").as_unit("s").asi8.astype(np.int64)
    if seconds.size > 1 and not np.all(np.diff(seconds) > 0):
        raise ValueError("timestamps must be strictly increasing (sorted, no duplicates)")
    return seconds


def _events_from_seconds(wet: np.ndarray, seconds: np.ndarray, max_dry_gap_h: int) -> RainEvents:
    wet_idx = np.flatnonzero(wet)
    event_of_hour = np.full(wet.size, -1, dtype=np.int64)
    if wet_idx.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return RainEvents(empty, empty.copy(), empty.copy(), event_of_hour)
    wet_seconds = seconds[wet_idx]
    dry_gap_h = np.rint(np.diff(wet_seconds) / SECONDS_PER_HOUR).astype(np.int64) - 1
    is_start = np.concatenate([[True], dry_gap_h > max_dry_gap_h])
    starts = wet_idx[is_start]
    ends = wet_idx[np.concatenate([is_start[1:], [True]])]
    for event_id, (lo, hi) in enumerate(zip(starts, ends)):
        event_of_hour[lo : hi + 1] = event_id
    start_hour = np.floor_divide(seconds[starts], SECONDS_PER_HOUR)
    return RainEvents(starts.astype(np.int64), ends.astype(np.int64), start_hour.astype(np.int64), event_of_hour)


def find_rain_events(
    areal_mm: Any,
    timestamps: pd.DatetimeIndex,
    max_dry_gap_h: int = 2,
    tz: str = "Asia/Kolkata",
) -> RainEvents:
    """Segment an areal series into events (runs of wet hours with <= ``max_dry_gap_h`` dry hours)."""
    areal = _clean_areal(areal_mm, warn=False)
    seconds = _epoch_seconds(timestamps, tz)
    if areal.size != seconds.size:
        raise ValueError(f"areal_mm has {areal.size} values but timestamps has {seconds.size}")
    return _events_from_seconds(areal > 0, seconds, int(max_dry_gap_h))


# --------------------------------------------------------------------------- input validation


def _clean_areal(areal_mm: Any, *, warn: bool = True) -> np.ndarray:
    """Return a float64 1-D copy with non-finite and negative values replaced by 0."""
    try:
        areal = np.array(areal_mm, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"areal_mm must be numeric: {exc}") from exc
    if areal.ndim != 1:
        raise ValueError(f"areal_mm must be 1-D [T], got shape {areal.shape}")
    bad = ~np.isfinite(areal)
    if bad.any():
        if warn:
            LOGGER.warning("Areal rainfall has %d non-finite values; treating them as 0 mm", int(bad.sum()))
        areal[bad] = 0.0
    negative = areal < 0
    if negative.any():
        if warn:
            LOGGER.warning("Areal rainfall has %d negative values; treating them as 0 mm", int(negative.sum()))
        areal[negative] = 0.0
    return areal


def _clean_coords(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        lon_arr = np.array(lon, dtype=np.float64, copy=True).reshape(-1)
        lat_arr = np.array(lat, dtype=np.float64, copy=True).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lon/lat must be numeric arrays: {exc}") from exc
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(f"lon and lat must have the same length, got {lon_arr.size} and {lat_arr.size}")
    if not (np.isfinite(lon_arr).all() and np.isfinite(lat_arr).all()):
        raise ValueError("lon/lat must be finite")
    if lon_arr.size and (np.abs(lat_arr).max() > 90 or np.abs(lon_arr).max() > 180):
        raise ValueError("lon/lat out of range (expected degrees, EPSG:4326)")
    return lon_arr, lat_arr


# --------------------------------------------------------------------------- geometry & cells


@dataclass(frozen=True)
class _Domain:
    """Rectangle (local metres) in which mature cells are placed; it contains every node."""

    x_min: float
    y_min: float
    width: float
    height: float


def _node_domain(
    lon: np.ndarray, lat: np.ndarray, cfg: Mapping[str, Any], params: RainFieldParams
) -> tuple[np.ndarray, np.ndarray, _Domain]:
    """Project nodes to local metres around the corridor centre and build the placement box.

    The box is anchored on ``region.bbox`` (when configured) so a node's field does not
    depend on which other nodes happen to be in the graph; nodes outside the bbox extend it.
    """
    bbox = (cfg.get("region") or {}).get("bbox")
    corner_lon, corner_lat = lon, lat
    origin = (float(lon.mean()), float(lat.mean()))
    if bbox is not None:
        try:
            west, south, east, north = validate_bbox(bbox)
        except ValueError as exc:
            raise ConfigError(f"region.bbox is invalid: {exc}") from exc
        origin = bbox_center(bbox)
        corner_lon = np.concatenate([lon, [west, east]])
        corner_lat = np.concatenate([lat, [south, north]])
    x_all, y_all = lonlat_to_local_xy(corner_lon, corner_lat, origin=origin)
    margin = params.domain_margin_factor * params.cell_radius_m[1] + 1.0
    x_min, x_max = float(x_all.min()) - margin, float(x_all.max()) + margin
    y_min, y_max = float(y_all.min()) - margin, float(y_all.max()) + margin
    x_nodes, y_nodes = x_all[: lon.size], y_all[: lat.size]
    return x_nodes, y_nodes, _Domain(x_min, y_min, x_max - x_min, y_max - y_min)


def _uniform_words(entropy: Sequence[int], n: int) -> np.ndarray:
    """``n`` uniforms in [0, 1) from a SeedSequence's state words (word ``i`` never depends on ``n``)."""
    words = np.random.SeedSequence(list(entropy)).generate_state(n, dtype=np.uint64)
    return (words >> np.uint64(11)).astype(np.float64) / _UINT53


def _first_generation(params: RainFieldParams) -> int:
    """Lowest generation index any sub-step of an event can touch (sub-steps start 1 h before t=0)."""
    return -int(math.ceil(SECONDS_PER_HOUR / params.half_life_s)) - 1


@dataclass(frozen=True)
class _EventCells:
    """Cell parameters of every event: velocity / phase per event and a flat generation table.

    Generation ``k`` of slot ``c`` of event ``e`` is row ``base[e, c] + k - first_generation`` of
    the ``peak_x`` / ``peak_y`` / ``radius`` / ``amplitude`` arrays.
    """

    vx: np.ndarray
    vy: np.ndarray
    phase: np.ndarray
    base: np.ndarray
    peak_x: np.ndarray
    peak_y: np.ndarray
    radius: np.ndarray
    amplitude: np.ndarray
    first_generation: int


def _draw_event_cells(
    start_hours: np.ndarray, last_dt_s: np.ndarray, params: RainFieldParams, domain: _Domain
) -> _EventCells:
    """Draw each event's steering velocity, slot phases and every cell generation it needs.

    ``last_dt_s[e]`` is the latest hour-end offset (s) of event ``e`` in this call; it only
    decides how many generations are drawn, never their values (consistency guarantee).
    """
    n_events, n_cells = start_hours.size, params.n_cells
    head = np.empty((n_events, 2 + n_cells), dtype=np.float64)
    first = _first_generation(params)
    n_gen = np.floor_divide(np.maximum(last_dt_s, 0.0), params.half_life_s).astype(np.int64) + 2 - first
    base = np.zeros((n_events, n_cells), dtype=np.int64)
    base.flat[1:] = np.cumsum(np.repeat(n_gen, n_cells))[:-1]
    table = np.empty((int(n_gen.sum()) * n_cells, _DRAWS_PER_CELL), dtype=np.float64)
    for e, hour in enumerate(start_hours):
        key = int(hour) % _SEED_MODULUS
        head[e] = _uniform_words((params.seed, key), 2 + n_cells)
        for c in range(n_cells):
            rows = slice(base[e, c], base[e, c] + n_gen[e])
            draws = _uniform_words((params.seed, key, c + 1), _DRAWS_PER_CELL * int(n_gen[e]))
            table[rows] = draws.reshape(-1, _DRAWS_PER_CELL)
    s_lo, s_hi = params.advection_speed_mps
    speed = s_lo + (s_hi - s_lo) * head[:, 0]
    heading = 2.0 * np.pi * head[:, 1]
    (r_lo, r_hi), (a_lo, a_hi) = params.cell_radius_m, params.amplitude_range
    return _EventCells(
        vx=speed * np.cos(heading),
        vy=speed * np.sin(heading),
        phase=head[:, 2:],
        base=base,
        peak_x=domain.x_min + domain.width * table[:, 0],
        peak_y=domain.y_min + domain.height * table[:, 1],
        radius=r_lo + (r_hi - r_lo) * table[:, 2],
        amplitude=a_lo + (a_hi - a_lo) * table[:, 3],
        first_generation=first,
    )


def _alive_generations(
    t: np.ndarray, phase: np.ndarray, half_life_s: float
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """The two cell generations alive at times ``t`` in a slot: ``(index, peak time, envelope)`` each.

    Generation ``k`` peaks at ``(k + phase) * half_life`` and has a triangular envelope of
    half-width ``half_life``; at any instant exactly two generations overlap and their
    envelopes sum to one (a partition of unity), so a slot always carries one cell-equivalent.
    """
    k_now = np.floor(t / half_life_s - phase).astype(np.int64)
    out = []
    for k in (k_now, k_now + 1):
        peak_t = (k + phase) * half_life_s
        out.append((k, peak_t, np.clip(1.0 - np.abs(t - peak_t) / half_life_s, 0.0, 1.0)))
    return tuple(out)


def _cell_intensity(
    x: np.ndarray,
    y: np.ndarray,
    dt_s: np.ndarray,
    event_ids: np.ndarray,
    cells: _EventCells,
    params: RainFieldParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Hour-averaged convective intensity ``[H, N]`` in [0, 1] and the strongest active amplitude ``[H]``."""
    total = np.zeros((dt_s.size, x.size), dtype=np.float64)
    strongest = np.zeros(dt_s.size, dtype=np.float64)
    half = params.half_life_s
    vx, vy = cells.vx[event_ids], cells.vy[event_ids]
    substeps = params.substeps_per_hour
    for j in range(substeps):
        # Sub-step mid-points inside the hour that ends at the timestamp.
        t = dt_s - SECONDS_PER_HOUR + (j + 0.5) * SECONDS_PER_HOUR / substeps
        dry_fraction = np.ones_like(total)
        for c in range(params.n_cells):
            phase = cells.phase[event_ids, c]
            for k, peak_t, envelope in _alive_generations(t, phase, half):
                row = cells.base[event_ids, c] + k - cells.first_generation
                cx = cells.peak_x[row] + vx * (t - peak_t)
                cy = cells.peak_y[row] + vy * (t - peak_t)
                dist2 = (x[None, :] - cx[:, None]) ** 2 + (y[None, :] - cy[:, None]) ** 2
                gauss = np.exp(-dist2 / (2.0 * cells.radius[row] ** 2)[:, None])
                weight = cells.amplitude[row] * envelope
                dry_fraction *= 1.0 - weight[:, None] * gauss
                strongest = np.maximum(strongest, np.where(envelope > 0, cells.amplitude[row], 0.0))
        total += 1.0 - dry_fraction
    return total / substeps, strongest


def _normalise_intensity(intensity: np.ndarray, strongest: np.ndarray) -> np.ndarray:
    """Rescale each hour so its most-affected node reaches the strongest active cell's amplitude.

    The gain compensates hour-averaging of moving cells and their life-cycle envelope; it is
    capped at ``_MAX_NORMALISATION_GAIN`` so an hour whose cells are all far from the corridor
    (it sees < 1/20 of a mature cell) stays mostly stratiform instead of having a distant cell's
    Gaussian tail stretched into an unphysically steep edge.
    """
    peak = intensity.max(axis=1)
    usable = peak > _MIN_PEAK_INTENSITY
    gain = np.where(usable, strongest / np.where(usable, peak, 1.0), 0.0)
    gain = np.minimum(gain, _MAX_NORMALISATION_GAIN)
    return np.minimum(intensity * gain[:, None], 1.0)


def _multipliers(intensity: np.ndarray, params: RainFieldParams, background: np.ndarray | None = None) -> np.ndarray:
    """Turn intensity into per-hour multipliers with node mean 1, clipped to ``multiplier_clip``.

    ``background`` is the stratiform share per hour ``[H, 1]`` (default: ``background_fraction``).
    """
    bg = params.background_fraction if background is None else background
    field = bg + (1.0 - bg) * intensity
    mean = field.mean(axis=1, keepdims=True)
    degenerate = mean[:, 0] <= _MIN_FIELD_MEAN
    mult = field / np.where(mean > _MIN_FIELD_MEAN, mean, 1.0)
    mult[degenerate] = 1.0
    lo, hi = params.multiplier_clip
    for _ in range(_CLIP_ITERATIONS):
        if mult.min() >= lo - 1e-9 and mult.max() <= hi + 1e-9:
            break
        mult = np.clip(mult, lo, hi)
        mult /= mult.mean(axis=1, keepdims=True)
    return mult


# --------------------------------------------------------------------------- public API


def downscale_rainfall(
    areal_mm: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lon: np.ndarray,
    lat: np.ndarray,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Downscale areal rainfall ``[T]`` (mm/h) to junction rainfall ``float32 [T, N]``.

    Every wet hour's node mean equals the areal value (to float32 precision), dry hours are
    exactly zero, the output has no NaN / negative values and is deterministic for a given
    configuration. Non-finite or negative areal values are treated as 0 with a WARNING.
    ``timestamps`` must be strictly increasing; tz-naive values are read as local time in
    ``project.timezone``. See the module docstring for the model and consistency guarantee.
    """
    params = RainFieldParams.from_config(cfg)
    tz = str((cfg.get("project") or {}).get("timezone", "Asia/Kolkata"))
    areal = _clean_areal(areal_mm)
    seconds = _epoch_seconds(timestamps, tz)
    if areal.size != seconds.size:
        raise ValueError(f"areal_mm has {areal.size} values but timestamps has {seconds.size}")
    lon_arr, lat_arr = _clean_coords(lon, lat)
    n_hours, n_nodes = areal.size, lon_arr.size
    if n_nodes == 0:
        LOGGER.warning("downscale_rainfall called with zero nodes; returning an empty [%d, 0] field", n_hours)
        return np.zeros((n_hours, 0), dtype=np.float32)
    uniform = not params.enabled or params.n_cells == 0 or params.background_fraction >= 1.0
    if uniform or n_hours == 0 or not (areal > 0).any():
        return np.repeat(areal[:, None], n_nodes, axis=1).astype(np.float32)
    return _stochastic_field(areal, seconds, lon_arr, lat_arr, cfg, params)


def _stochastic_field(
    areal: np.ndarray,
    seconds: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    cfg: Mapping[str, Any],
    params: RainFieldParams,
) -> np.ndarray:
    events = _events_from_seconds(areal > 0, seconds, params.max_dry_gap_h)
    x, y, domain = _node_domain(lon, lat, cfg, params)
    wet_idx = np.flatnonzero(areal > 0)
    event_ids = events.event_of_hour[wet_idx]
    dt_s = (seconds[wet_idx] - seconds[events.start_idx[event_ids]]).astype(np.float64)
    last_dt = np.zeros(events.num_events, dtype=np.float64)
    np.maximum.at(last_dt, event_ids, dt_s)
    cells = _draw_event_cells(events.start_hour, last_dt, params, domain)
    out = np.zeros((areal.size, lon.size), dtype=np.float32)
    block = max(1, _CHUNK_ELEMENTS // lon.size)
    for lo in range(0, wet_idx.size, block):
        sl = slice(lo, lo + block)
        intensity, strongest = _cell_intensity(x, y, dt_s[sl], event_ids[sl], cells, params)
        if params.normalise_intensity:
            intensity = _normalise_intensity(intensity, strongest)
        rows = wet_idx[sl]
        mult = _multipliers(intensity, params, params.background_for(areal[rows]))
        out[rows] = (areal[rows, None] * mult).astype(np.float32)
    LOGGER.debug(
        "Downscaled %d wet hours in %d events onto %d nodes", wet_idx.size, events.num_events, lon.size
    )
    return out
