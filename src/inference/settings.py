"""Validated ``inference`` config section shared by the whole inference package and the app.

:class:`InferenceSettings` merges ``cfg["inference"]`` over :data:`DEFAULTS` and validates every
key once (a :class:`~src.utils.config.ConfigError` names the offending key). It is re-exported
by :mod:`src.inference.scenarios`, the historical import path.

Rain-field ensemble (``field_members`` / ``field_seed_stride``)
---------------------------------------------------------------
Training labels come from the physics teacher driven by one stochastic junction rain field per
rain event (``rainfall_field.seed``). When only corridor-average rain is known (forecast, design
storm, custom series) that field is unknown, so predictors average over ``field_members``
independent field realisations; member ``k`` of a run with seed offset ``o`` downscales with
``rainfall_field.seed = base + (o + k) * field_seed_stride`` (see
:mod:`src.inference.field_ensemble`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.data_pipeline import weather
from src.utils.config import ConfigError, get_section

DEFAULTS: dict[str, Any] = {
    "horizons_h": [12, 24, 36, 48],
    "mc_dropout_samples": 20,
    "max_mc_samples": 200,
    "batch_windows": 8,
    "history_hours": 48,
    "checkpoint_name": "best.pt",
    "num_threads": None,
    "physics_softness_m": 0.03,
    "max_scenario_hours": 336,
    "max_event_backfill_h": 168,
    "notable_event_window_h": 24,
    "notable_event_separation_h": 72,
    "backtest_spinup_h": 72,
    "max_backtest_days": 120,
    "design_storm_shape": "chicago",
    "design_storm_peak_position": 0.4,
    "design_storm_offset_h": 6,
    "design_storm_rain_scale": 0.14,
    "design_storm_field_anchor": "2000-01-01T00:00",
    "field_members": 8,
    "field_seed_stride": 7919,
    "risk_tiers": {"low": 0.0, "moderate": 0.25, "high": 0.5, "severe": 0.75},
    "design_storms": [
        {"name": "Moderate shower (30 mm in 3 h)", "total_mm": 30.0, "duration_h": 3},
        {"name": "Heavy downpour (80 mm in 3 h)", "total_mm": 80.0, "duration_h": 3},
        {"name": "Cloudburst (130 mm in 6 h, Sept-2022 analogue)", "total_mm": 130.0, "duration_h": 6},
    ],
}
MAX_FIELD_MEMBERS = 64  # each member is one full model / simulator pass


def whole(value: Any, name: str, minimum: int, error: type[ValueError] = ValueError) -> int:
    """Return ``value`` as an int >= ``minimum`` (whole-valued floats accepted, bools rejected)."""
    if isinstance(value, bool):
        raise error(f"{name} must be an integer >= {minimum}, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error(f"{name} must be an integer >= {minimum}, got {value!r}") from exc
    if not math.isfinite(number) or number != int(number) or number < minimum:
        raise error(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(number)


def real(value: Any, name: str, lo: float, hi: float, error: type[ValueError] = ValueError) -> float:
    """Return ``value`` as a finite float in ``[lo, hi]`` (bools rejected)."""
    if isinstance(value, bool):
        raise error(f"{name} must be a number in [{lo}, {hi}], got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error(f"{name} must be a number in [{lo}, {hi}], got {value!r}") from exc
    if not (math.isfinite(number) and lo <= number <= hi):
        raise error(f"{name} must be a number in [{lo}, {hi}], got {value!r}")
    return number


@dataclass(frozen=True)
class DesignStormPreset:
    """One what-if storm offered by the app / CLI (``inference.design_storms``)."""

    name: str
    total_mm: float
    duration_h: int


def _risk_tiers(value: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError(f"inference.risk_tiers must be a non-empty mapping of tier -> lower bound, got {value!r}")
    tiers = sorted(
        ((str(name).strip().title(), real(bound, f"inference.risk_tiers.{name}", 0.0, 1.0, ConfigError))
         for name, bound in value.items()),
        key=lambda item: item[1],
    )
    bounds = [bound for _, bound in tiers]
    if bounds[0] != 0.0:
        raise ConfigError(f"inference.risk_tiers: the lowest tier must start at 0.0, got {dict(tiers)}")
    if len(set(bounds)) != len(bounds) or bounds[-1] >= 1.0:
        raise ConfigError(f"inference.risk_tiers bounds must be distinct and < 1.0, got {dict(tiers)}")
    return tuple(tiers)


def _presets(value: Any) -> tuple[DesignStormPreset, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise ConfigError(f"inference.design_storms must be a list of {{name, total_mm, duration_h}}, got {value!r}")
    presets = []
    for i, item in enumerate(value):
        if not isinstance(item, Mapping) or not {"total_mm", "duration_h"} <= set(item):
            raise ConfigError(f"inference.design_storms[{i}] needs total_mm and duration_h, got {item!r}")
        total = real(item["total_mm"], f"inference.design_storms[{i}].total_mm", 0.0, 1e4, ConfigError)
        duration = whole(item["duration_h"], f"inference.design_storms[{i}].duration_h", 1, ConfigError)
        name = str(item.get("name") or f"{total:g} mm in {duration} h")
        presets.append(DesignStormPreset(name, total, duration))
    return tuple(presets)


def _horizons(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ConfigError(f"inference.horizons_h must be a list of positive hours, got {value!r}")
    horizons = sorted({whole(h, "inference.horizons_h", 1, ConfigError) for h in value})
    if not horizons:
        raise ConfigError("inference.horizons_h must not be empty")
    return tuple(horizons)


def _anchor(value: Any) -> str:
    """``inference.design_storm_field_anchor``: a whole-hour, tz-naive local date/time (ISO text)."""
    try:
        ts = pd.Timestamp(str(value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"inference.design_storm_field_anchor must be an ISO date/time, got {value!r}") from exc
    if pd.isna(ts) or ts.tzinfo is not None or ts != ts.floor("h"):
        raise ConfigError("inference.design_storm_field_anchor must be a whole local hour without a timezone, "
                          f"e.g. '2000-01-01T00:00', got {value!r}")
    return ts.isoformat()


_INT_KEYS = (
    ("mc_dropout_samples", 0), ("max_mc_samples", 1), ("batch_windows", 1), ("history_hours", 0),
    ("max_scenario_hours", 1), ("max_event_backfill_h", 0), ("notable_event_window_h", 1),
    ("notable_event_separation_h", 1), ("backtest_spinup_h", 0), ("max_backtest_days", 1),
    ("design_storm_offset_h", 0), ("field_members", 1), ("field_seed_stride", 1),
)


@dataclass(frozen=True)
class InferenceSettings:
    """Validated ``inference`` config section (merged over :data:`DEFAULTS`)."""

    horizons_h: tuple[int, ...]
    mc_dropout_samples: int
    max_mc_samples: int
    batch_windows: int
    history_hours: int
    checkpoint_name: str
    num_threads: int | None
    physics_softness_m: float
    max_scenario_hours: int
    max_event_backfill_h: int
    notable_event_window_h: int
    notable_event_separation_h: int
    backtest_spinup_h: int
    max_backtest_days: int
    design_storm_shape: str
    design_storm_peak_position: float
    design_storm_offset_h: int
    design_storm_rain_scale: float
    design_storm_field_anchor: str
    risk_tiers: tuple[tuple[str, float], ...]
    design_storms: tuple[DesignStormPreset, ...]
    field_members: int = 8
    field_seed_stride: int = 7919

    @property
    def max_horizon_h(self) -> int:
        return max(self.horizons_h)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "InferenceSettings":
        s = get_section(cfg, "inference", DEFAULTS)
        ints = {key: whole(s[key], f"inference.{key}", minimum, ConfigError) for key, minimum in _INT_KEYS}
        if ints["field_members"] > MAX_FIELD_MEMBERS:
            raise ConfigError(f"inference.field_members must be <= {MAX_FIELD_MEMBERS}, got {ints['field_members']}")
        shape = str(s["design_storm_shape"])
        if shape not in weather.STORM_SHAPES:
            raise ConfigError(f"inference.design_storm_shape must be one of {weather.STORM_SHAPES}, got {shape!r}")
        threads = s["num_threads"]
        name = str(s["checkpoint_name"] or "").strip()
        if not name:
            raise ConfigError("inference.checkpoint_name must be a file name, e.g. best.pt")
        settings = cls(
            horizons_h=_horizons(s["horizons_h"]),
            checkpoint_name=name,
            num_threads=None if threads is None else whole(threads, "inference.num_threads", 1, ConfigError),
            physics_softness_m=real(s["physics_softness_m"], "inference.physics_softness_m", 1e-6, 10.0, ConfigError),
            design_storm_shape=shape,
            design_storm_peak_position=real(s["design_storm_peak_position"], "inference.design_storm_peak_position",
                                            0.0, 1.0, ConfigError),
            design_storm_rain_scale=real(s["design_storm_rain_scale"], "inference.design_storm_rain_scale",
                                         1e-3, 10.0, ConfigError),
            design_storm_field_anchor=_anchor(s["design_storm_field_anchor"]),
            risk_tiers=_risk_tiers(s["risk_tiers"]),
            design_storms=_presets(s["design_storms"]),
            **ints,
        )
        if settings.max_horizon_h > settings.max_scenario_hours:
            raise ConfigError(f"inference.horizons_h max {settings.max_horizon_h} exceeds max_scenario_hours "
                              f"{settings.max_scenario_hours}")
        return settings
