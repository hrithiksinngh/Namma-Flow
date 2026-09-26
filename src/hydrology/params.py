"""Validated parameters of the hydrology label simulator (``hydrology`` config section).

The physical meaning of every key is documented in :mod:`src.hydrology.simulator`; the
calibration record behind the defaults is in :mod:`src.hydrology.calibrate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

from src.utils.config import ConfigError, get_section
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Calibrated defaults (mirrored in the ``hydrology`` section of config/config.yaml).
# Tuned on the real Bellandur graph (1 035 junctions, real OSM drains) and the 2018-2024
# Open-Meteo (ERA5) record together with the ``rainfall_field`` defaults; they are *effective*
# values for reanalysis-scale rain, which smooths Bengaluru's convective bursts several-fold
# (Sept 4-5 2022 peaks at only 8.5 mm/h areal), so drain capacities and the surcharge
# threshold are lower than design values for gauge rain. Results: python -m src.hydrology.calibrate.
DEFAULTS: dict[str, Any] = {
    "catchment_width_m": 30.0,
    "runoff_coeff_dry": 0.70,
    "runoff_coeff_wet": 0.80,
    "antecedent_window_h": 24,
    "antecedent_saturation_mm": 20.0,
    "drain_capacity_near_mm_h": 17.5,
    "drain_capacity_far_mm_h": 13.4,
    "drain_decay_m": 450.0,
    "surcharge_local_weight": 1.0,
    "surcharge_window_h": 6,
    "surcharge_threshold_mm": 4.1,
    "surcharge_ramp_mm": 3.4,
    "surcharge_min_factor": 0.055,
    "tailwater_threshold_mm": 3.7,
    "tailwater_ramp_mm": 1.3,
    "routing_exponent": 1.0,
    "outflow_rate_per_h": 1.5,
    "min_routing_grade": 0.0022,
    "ponding_fraction": 0.026,
    "infiltration_mm_h": 1.9,
    "spill_depth_m": 0.55,
    "flood_depth_threshold_m": 0.15,
}

_INT_KEYS = frozenset({"antecedent_window_h", "surcharge_window_h"})
_NULLABLE_KEYS = frozenset({"spill_depth_m"})


class SimulationError(RuntimeError):
    """The simulation produced a non-finite routing matrix or water budget.

    Raised instead of returning NaN depths, which would silently read as "dry" labels.
    """


def _coerce(name: str, value: Any) -> float | int | None:
    """Convert a config value to int/float (``None`` only for nullable keys); bools/strings are errors."""
    if value is None and name in _NULLABLE_KEYS:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigError(f"hydrology.{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"hydrology.{name} must be finite, got {value!r}")
    if name in _INT_KEYS:
        if number != int(number):
            raise ConfigError(f"hydrology.{name} must be a whole number of hours, got {value!r}")
        return int(number)
    return number


def _require(name: str, ok: bool, rule: str, value: Any) -> None:
    if not ok:
        raise ConfigError(f"hydrology.{name} must be {rule}, got {value!r}")


@dataclass(frozen=True)
class HydrologyParams:
    """Validated ``hydrology`` configuration (one field per config key; see :mod:`src.hydrology.simulator`)."""

    catchment_width_m: float
    runoff_coeff_dry: float
    runoff_coeff_wet: float
    antecedent_window_h: int
    antecedent_saturation_mm: float
    drain_capacity_near_mm_h: float
    drain_capacity_far_mm_h: float
    drain_decay_m: float
    surcharge_local_weight: float
    surcharge_window_h: int
    surcharge_threshold_mm: float
    surcharge_ramp_mm: float
    surcharge_min_factor: float
    tailwater_threshold_mm: float
    tailwater_ramp_mm: float
    routing_exponent: float
    outflow_rate_per_h: float
    min_routing_grade: float
    ponding_fraction: float
    infiltration_mm_h: float
    spill_depth_m: float | None
    flood_depth_threshold_m: float

    def __post_init__(self) -> None:
        for field in fields(self):  # type / finiteness check; normalises ints / floats in place
            object.__setattr__(self, field.name, _coerce(field.name, getattr(self, field.name)))
        positive = ("catchment_width_m", "antecedent_saturation_mm", "drain_decay_m",
                    "surcharge_threshold_mm", "surcharge_ramp_mm", "tailwater_ramp_mm", "flood_depth_threshold_m")
        for name in positive:
            _require(name, getattr(self, name) > 0, "> 0", getattr(self, name))
        non_negative = ("drain_capacity_near_mm_h", "drain_capacity_far_mm_h", "routing_exponent",
                        "outflow_rate_per_h", "min_routing_grade", "infiltration_mm_h", "antecedent_window_h",
                        "tailwater_threshold_mm")
        for name in non_negative:
            _require(name, getattr(self, name) >= 0, ">= 0", getattr(self, name))
        for name in ("runoff_coeff_dry", "runoff_coeff_wet", "surcharge_min_factor", "surcharge_local_weight"):
            _require(name, 0.0 <= getattr(self, name) <= 1.0, "in [0, 1]", getattr(self, name))
        _require("ponding_fraction", 0.0 < self.ponding_fraction <= 1.0, "in (0, 1]", self.ponding_fraction)
        _require("surcharge_window_h", self.surcharge_window_h >= 1, ">= 1", self.surcharge_window_h)
        _require("runoff_coeff_dry", self.runoff_coeff_dry <= self.runoff_coeff_wet,
                 "<= runoff_coeff_wet (wet catchments shed more water)", self.runoff_coeff_dry)
        if self.spill_depth_m is not None:
            _require("spill_depth_m", self.spill_depth_m > self.flood_depth_threshold_m,
                     "> flood_depth_threshold_m (or null to disable spilling)", self.spill_depth_m)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "HydrologyParams":
        """Build from ``cfg['hydrology']`` merged over :data:`DEFAULTS` (unknown keys are warned about)."""
        section = get_section(cfg, "hydrology", DEFAULTS)
        unknown = sorted(set(section) - set(DEFAULTS))
        if unknown:
            LOGGER.warning("Ignoring unknown hydrology config keys %s (known: %s)", unknown, sorted(DEFAULTS))
        return cls(**{name: _coerce(name, section[name]) for name in DEFAULTS})

    def to_dict(self) -> dict[str, Any]:
        """Plain ``{key: value}`` mapping, suitable for YAML / JSON and :meth:`from_config`."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
