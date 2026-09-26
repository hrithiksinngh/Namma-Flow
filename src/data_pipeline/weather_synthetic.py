"""Synthetic rainfall: Bengaluru climatology generator (offline fallback) and design storms.

Both produce schema-2.2 frames (see :mod:`src.data_pipeline.weather_schema`) and are
re-exported by :mod:`src.data_pipeline.weather`, which is the module callers should import.

The climatology generator has two scales (``weather.synthetic_scale``):

* ``areal`` (default) - reanalysis-like areal rain fitted to the 2018-2024 Open-Meteo/ERA5
  record at the pilot corridor (~1000 mm/yr on ~135 wet days, 2-16 h wet spells with an early
  peak and a long light tail, hourly peaks rarely above 15-20 mm/h). The hydrology labels and
  the model are calibrated on that record, so offline builds and gap fills keep the input
  distribution the pipeline was tuned for.
* ``point`` - rain-gauge-like convective rain (IMD-style normals, ~58 wet days/yr, 1-6 h
  bursts, 60-130 mm/day extremes), kept for what-if studies at point scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_pipeline.weather_schema import (
    DEFAULT_TZ,
    STORM_SHAPES,
    _bound,
    _check_tz,
    _parse_range,
    _schema_frame,
    _settings,
    _tz,
)

_CHICAGO_B_H, _CHICAGO_C = 0.25, 0.8  # IDF i = a / (t + b)^c shape parameters (t in hours)
# Hours after the end of a month that a burst starting on its last day may still reach
# (latest nocturnal start 29 h + longest burst 16 h = 45 h after the last day's 00:00).
SPILL_PAD_H = 48


@dataclass(frozen=True)
class Climatology:
    """Monthly wet-day statistics and burst shapes of one synthetic rainfall scale."""

    monthly_mm: np.ndarray          # mean rain per calendar month
    wet_days: np.ndarray            # mean wet days per calendar month
    extreme_prob: np.ndarray        # probability that a wet day is an extreme event
    extreme_range_mm: tuple[float, float]
    max_ordinary_day_mm: float
    persistence: float              # lag-1 autocorrelation of the wet/dry day Markov chain
    gamma_shape: float              # daily amounts ~ Gamma(shape, mean / shape)
    burst_hours: np.ndarray         # burst duration choices (h) ...
    burst_probs: np.ndarray         # ... and their probabilities
    extreme_hours: tuple[int, int]  # extreme-day burst duration range [lo, hi)
    afternoon_prob: float           # share of bursts starting in the afternoon (else nocturnal)
    afternoon_start: tuple[float, float, float, float]  # (mean, sd, min, max) start hour
    nocturnal_start: tuple[float, float, float, float]
    two_burst_prob: float           # probability that a wet day >= 5 mm has two bursts
    profile_width: float            # burst profile width as a fraction of its duration
    hours_per_mm: float = 0.0       # extra burst hours per mm of the burst (areal: heavy days last longer)
    max_burst_h: int = 16
    exponential_tail: bool = False  # peak early then decay exponentially (areal) instead of a Gaussian


POINT = Climatology(
    monthly_mm=np.array([3, 6, 15, 45, 115, 90, 115, 140, 200, 170, 55, 15], dtype=float),
    wet_days=np.array([0.3, 0.5, 1.2, 3.5, 8.0, 7.0, 9.0, 11.0, 11.0, 10.0, 5.0, 1.5]),
    extreme_prob=np.array([0, 0, 0, 0.003, 0.012, 0.004, 0.004, 0.01, 0.02, 0.018, 0.006, 0]),
    extreme_range_mm=(60.0, 130.0),
    max_ordinary_day_mm=110.0,
    persistence=0.35,
    gamma_shape=0.8,
    burst_hours=np.arange(1, 7),
    burst_probs=np.array([0.15, 0.30, 0.25, 0.15, 0.10, 0.05]),
    extreme_hours=(3, 7),
    afternoon_prob=0.75,
    afternoon_start=(16.5, 2.5, 8.0, 23.0),
    nocturnal_start=(23.0, 3.0, 18.0, 29.0),
    two_burst_prob=0.3,
    profile_width=1.0 / 3.0,
)
# Fitted to the 2018-2024 Open-Meteo (ERA5) record at the corridor centre (12.9325 N, 77.6775 E).
AREAL = Climatology(
    monthly_mm=np.array([12, 3, 15, 24, 102, 116, 157, 141, 133, 164, 99, 41], dtype=float),
    wet_days=np.array([2.0, 1.0, 2.7, 4.3, 14.4, 19.9, 24.6, 19.7, 20.7, 18.6, 12.9, 5.9]),
    extreme_prob=np.array([0, 0, 0, 0, 0.004, 0.002, 0.002, 0.004, 0.006, 0.008, 0.008, 0.004]),
    extreme_range_mm=(45.0, 80.0),
    max_ordinary_day_mm=45.0,
    persistence=0.4,
    gamma_shape=1.0,
    burst_hours=np.arange(2, 13),
    burst_probs=np.array([0.10, 0.14, 0.15, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.03]),
    extreme_hours=(8, 13),
    afternoon_prob=0.7,
    afternoon_start=(13.5, 2.5, 8.0, 21.0),
    nocturnal_start=(21.0, 3.0, 17.0, 29.0),
    two_burst_prob=0.35,
    profile_width=0.2,
    hours_per_mm=0.25,
    exponential_tail=True,
)
CLIMATOLOGIES: dict[str, Climatology] = {"areal": AREAL, "point": POINT}


# --------------------------------------------------------------------------- synthetic climatology


def _burst_profile(duration: int, width: float, exponential_tail: bool = False) -> np.ndarray:
    """Front-loaded burst profile (sums to 1).

    Gaussian (point scale): sharp onset, peak ~30 % into the burst, slower decay. Exponential
    tail (areal scale): peak ~20 % into the burst, then an e-folding decay of ``width x duration``
    hours - the long, light tail reanalysis rain shows after a convective peak.
    """
    k = np.arange(duration, dtype=float)
    if exponential_tail:
        peak, tau = 0.2 * (duration - 1), max(0.5, width * duration)
        weights = np.where(k <= peak, np.exp(-(peak - k) / max(0.5, 0.5 * tau)), np.exp(-(k - peak) / tau))
    else:
        weights = np.exp(-(((k - 0.3 * (duration - 1)) / max(0.7, duration * width)) ** 2))
    return weights / weights.sum()


def _start_hour(rng: np.random.Generator, clim: Climatology) -> float:
    mean, sd, lo, hi = clim.afternoon_start if rng.random() < clim.afternoon_prob else clim.nocturnal_start
    return float(np.clip(rng.normal(mean, sd), lo, hi))


def _place_bursts(
    hours: np.ndarray, day: int, amount: float, extreme: bool, rng: np.random.Generator, clim: Climatology
) -> None:
    """Add one wet day's ``amount`` as 1-2 bursts; ``hours`` is padded so no burst is truncated."""
    n_bursts = 2 if amount >= 5.0 and rng.random() < clim.two_burst_prob else 1
    shares = rng.dirichlet(np.full(n_bursts, 2.0)) if n_bursts > 1 else np.ones(1)
    for share in shares:
        start = day * 24 + int(_start_hour(rng, clim))
        if extreme:
            duration = int(rng.integers(*clim.extreme_hours))
        else:
            duration = int(rng.choice(clim.burst_hours, p=clim.burst_probs))
        duration = min(duration + int(clim.hours_per_mm * amount * share), clim.max_burst_h)
        if start + duration > hours.size:  # pragma: no cover - guarded by SPILL_PAD_H
            raise AssertionError("synthetic burst exceeds the month padding")
        hours[start : start + duration] += amount * share * _burst_profile(duration, clim.profile_width, clim.exponential_tail)


def _daily_amount(rng: np.random.Generator, clim: Climatology, month_index: int, extreme: bool) -> float:
    if extreme:
        return float(rng.uniform(*clim.extreme_range_mm))
    p_extreme = clim.extreme_prob[month_index]
    mean_day = clim.monthly_mm[month_index] / clim.wet_days[month_index]
    mean_ordinary = max(0.5, (mean_day - p_extreme * np.mean(clim.extreme_range_mm)) / (1.0 - p_extreme))
    return min(float(rng.gamma(clim.gamma_shape, mean_ordinary / clim.gamma_shape)), clim.max_ordinary_day_mm)


def _padded_month(year: int, month: int, seed: int, tz: str, clim: Climatology) -> np.ndarray:
    """Rain generated for one month from ``(seed, year, month)``: the month's hours + ``SPILL_PAD_H``.

    The tail holds the part of late-evening / nocturnal bursts of the last day that falls
    into the next month; :func:`_month_series` adds it to that month's first hours.
    """
    n_hours = _month_index(year, month, tz).size
    n_days = pd.Timestamp(year=year, month=month, day=1).days_in_month
    hours = np.zeros(n_hours + SPILL_PAD_H)
    rng = np.random.default_rng(np.random.SeedSequence([seed, year, month]))
    m = month - 1
    p_wet = min(0.95, clim.wet_days[m] / n_days)
    p_after_wet = p_wet + clim.persistence * (1.0 - p_wet)
    p_after_dry = p_wet * (1.0 - clim.persistence)
    wet = False
    for day, u in enumerate(rng.random(n_days)):
        wet = bool(u < (p_wet if day == 0 else (p_after_wet if wet else p_after_dry)))
        if not wet:
            continue
        extreme = bool(rng.random() < clim.extreme_prob[m])
        _place_bursts(hours, day, _daily_amount(rng, clim, m, extreme), extreme, rng, clim)
    return hours


def _month_index(year: int, month: int, tz: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=month, day=1).tz_localize(tz)
    return pd.date_range(start, start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23), freq="h", tz=tz)


def _month_series(period: pd.Period, seed: int, tz: str, clim: Climatology, cache: dict) -> pd.Series:
    """One month of rain: its own bursts plus the previous month's spill-over into its first hours.

    Both parts depend only on ``(seed, year, month)`` of their own month, so any sub-range of a
    longer run is identical to the same hours generated alone.
    """
    def padded(p: pd.Period) -> np.ndarray:
        if p not in cache:
            cache[p] = _padded_month(p.year, p.month, seed, tz, clim)
        return cache[p]

    index = _month_index(period.year, period.month, tz)
    own = padded(period)
    values = own[: index.size].copy()
    previous = period - 1
    prev = padded(previous)
    spill = prev[_month_index(previous.year, previous.month, tz).size :]
    n = min(spill.size, values.size)
    values[:n] += spill[:n]
    return pd.Series(values, index=index)


def generate_synthetic_weather(
    start: Any, end: Any, cfg: Mapping[str, Any], seed: int | None = None
) -> pd.DataFrame:
    """Offline fallback: hourly Bengaluru rainfall for ``[start, end]`` (schema 2.2).

    ``weather.synthetic_scale`` picks the climatology (``areal`` reanalysis-like default, or
    ``point`` gauge-like; see the module docstring): Markov wet spells, gamma daily amounts,
    rare extreme days and front-loaded afternoon/evening (or nocturnal) bursts. Generated
    month by month from ``(seed, year, month)`` - bursts running past a month end spill into
    the next month, never truncated - so every sub-range is identical to the same hours of a
    longer run. ``seed`` defaults to ``project.seed``.
    """
    settings, tz = _settings(cfg), _tz(cfg)
    lo, hi = _parse_range(start, end, tz)
    seed = int((cfg.get("project") or {}).get("seed", 42)) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    clim = CLIMATOLOGIES[settings["synthetic_scale"]]
    months = pd.period_range(lo.tz_localize(None).to_period("M"), hi.tz_localize(None).to_period("M"), freq="M")
    cache: dict = {}
    series = pd.concat([_month_series(p, int(seed), tz, clim, cache) for p in months]).loc[lo:hi]
    series = series.clip(upper=settings["max_precip_mm_h"])
    return _schema_frame(pd.DatetimeIndex(series.index), series.to_numpy(), False, "synthetic")


# --------------------------------------------------------------------------- design storms


def _whole_number(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if not math.isfinite(number) or number != int(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(number)


def _mass_curve(t: np.ndarray, duration: float, shape: str, r: float) -> np.ndarray:
    """Cumulative storm mass at times ``t`` (hours from storm start), unnormalised."""
    peak = r * duration
    before = t <= peak
    out = np.zeros_like(t)
    if shape == "uniform":
        return t / duration
    if shape == "triangular":
        if peak > 0:
            out[before] = t[before] ** 2 / (duration * peak)
        out[~before] = 1.0 - (duration - t[~before]) ** 2 / (duration * (duration - peak))
        return out

    def depth(tau: np.ndarray | float) -> np.ndarray | float:  # Chicago: depth over a window tau
        return tau / (tau + _CHICAGO_B_H) ** _CHICAGO_C

    before_total = r * depth(duration)
    if r > 0:
        out[before] = before_total - r * depth((peak - t[before]) / r)
    if r < 1:
        out[~before] = before_total + (1.0 - r) * depth((t[~before] - peak) / (1.0 - r))
    return out


def _storm_depths(total: float, duration: int, shape: str, r: float) -> np.ndarray:
    mass = _mass_curve(np.arange(duration + 1, dtype=float), float(duration), shape, r)
    weights = np.clip(np.diff(mass), 0.0, None)
    weights = weights / weights.sum()
    depths = total * weights
    depths[int(np.argmax(depths))] += total - math.fsum(depths)  # conserve the total exactly
    return depths


def design_storm(
    total_mm: float,
    duration_h: int,
    *,
    start: Any,
    lead_hours: int = 24,
    horizon_hours: int = 48,
    tz: str = DEFAULT_TZ,
    shape: str = "chicago",
    peak_position: float = 0.4,
) -> pd.DataFrame:
    """Synthetic what-if hyetograph: ``lead_hours`` of dry history then a storm at ``start``.

    The frame spans ``[start - lead_hours, start + horizon_hours)``; the storm occupies the
    ``duration_h`` hourly steps beginning at ``start`` and its depths sum to ``total_mm``.
    Shapes: ``chicago`` (Keifer-Chu IDF-based, peak at ``peak_position`` of the duration),
    ``triangular`` and ``uniform``. ``is_forecast`` is True from ``start`` onwards;
    ``source="design_storm"``. Values are not clipped (the what-if total is authoritative).
    """
    total = float(total_mm) if isinstance(total_mm, (int, float, np.number)) and not isinstance(total_mm, bool) else None
    if total is None or not math.isfinite(total) or total < 0:
        raise ValueError(f"total_mm must be a finite number >= 0, got {total_mm!r}")
    duration = _whole_number(duration_h, "duration_h", 1)
    lead = _whole_number(lead_hours, "lead_hours", 0)
    horizon = _whole_number(horizon_hours, "horizon_hours", 1)
    if duration > horizon:
        raise ValueError(f"duration_h={duration} exceeds horizon_hours={horizon}")
    if shape not in STORM_SHAPES:
        raise ValueError(f"shape must be one of {STORM_SHAPES}, got {shape!r}")
    r = float(peak_position)
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"peak_position must lie in [0, 1], got {peak_position!r}")
    storm_start = _bound(start, _check_tz(tz), end=False)
    index = pd.date_range(storm_start - pd.Timedelta(hours=lead), periods=lead + horizon, freq="h", name="timestamp")
    precip = np.zeros(lead + horizon)
    if total > 0:
        precip[lead : lead + duration] = _storm_depths(total, duration, shape, r)
    return _schema_frame(index, precip, False, "design_storm", index >= storm_start)
