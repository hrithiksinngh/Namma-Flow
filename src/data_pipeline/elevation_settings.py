"""Validated ``elevation`` settings, the enrichment config hash and the synthetic-downgrade check.

Split out of :mod:`src.data_pipeline.elevation` (which re-exports every public name) so the
network stage (01), the elevation stage (02) and the dataset builder (04) can share them
without importing the raster stack.

**Enrichment config hash** (:func:`enrichment_config_hash`, stored by
:func:`~src.data_pipeline.elevation.enrich_graph` as the graph attribute
``enrichment_config_hash``): sha256[:16] over the *effective* values (defaults applied) of
exactly the settings that change the enrichment attributes of a road graph
(``elevation``, ``relative_elevation``, ``dist_to_drain_m``, ``grade``,
``flow_accumulation``, ``is_sink``) - see :data:`ENRICHMENT_HASH_KEYS`. Endpoints, timeouts,
retry and refresh settings are deliberately excluded (they change how data is fetched, not
what it is). :func:`enrichment_drift` compares a graph's stored hash with the current
configuration so stage 01 (cached graph) and stage 04 never keep static features computed
with other settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx

from src.data_pipeline.drains import DEFAULTS as DRAIN_DEFAULTS
from src.data_pipeline.network_settings import NETWORK_DEFAULTS, REGION_DEFAULTS
from src.utils.config import ConfigError, config_hash, get_section

DEFAULTS: dict[str, Any] = {
    "sources": ["local", "srtm", "open_meteo", "synthetic"],
    "srtm_url_template": "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_band}/{tile}.hgt.gz",
    "open_meteo_url": "https://api.open-meteo.com/v1/elevation",
    "open_meteo_batch_size": 100,
    "valid_range_m": [600.0, 1200.0],
    "tpi_radius_m": 300.0,
    "max_abs_grade": 0.3,
    "request_timeout_s": 60,
    "max_retries": 3,
    "backoff_s": 2.0,
    "max_void_fraction": 0.5,
    "void_fill_k": 8,
    "dem_margin_deg": 0.01,
    "max_srtm_tiles": 9,
    "refresh_dem": False,
}
OPEN_METEO_MAX_BATCH = 100

#: Settings covered by :func:`enrichment_config_hash`, per config section.
ENRICHMENT_HASH_KEYS: dict[str, tuple[str, ...]] = {
    "elevation": ("sources", "valid_range_m", "tpi_radius_m", "max_abs_grade", "max_void_fraction",
                  "void_fill_k", "dem_margin_deg"),
    "drains": ("osm_tags", "exclude_water_values", "max_distance_m", "bbox_margin_deg", "min_polygon_area_m2"),
    "region": ("bbox", "drain_fallback_lon"),
    "network": ("min_edge_length_m",),
}
ENRICHMENT_HASH_ATTR = "enrichment_config_hash"
REAL_ELEVATION_SOURCES = frozenset({"local_dem", "srtm", "open_meteo"})
REAL_DRAIN_SOURCES = frozenset({"osm"})
_SECTION_DEFAULTS: dict[str, Mapping[str, Any]] = {
    "elevation": DEFAULTS, "drains": DRAIN_DEFAULTS, "region": REGION_DEFAULTS, "network": NETWORK_DEFAULTS,
}


# --------------------------------------------------------------------------- settings
def _number(section: Mapping[str, Any], key: str, *, minimum: float | None = None, strict: bool = False) -> float:
    try:
        value = float(section.get(key))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"elevation.{key} must be a number, got {section.get(key)!r}") from exc
    if not math.isfinite(value) or (minimum is not None and (value < minimum or (strict and value == minimum))):
        raise ConfigError(f"elevation.{key} must be {'>' if strict else '>='} {minimum}, got {value}")
    return value


def _check_range(valid_range: Any) -> tuple[float, float] | None:
    """Validate an optional ``(low, high)`` elevation band; raises ``ValueError``."""
    if valid_range is None:
        return None
    try:
        low, high = (float(v) for v in valid_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"valid range must be two numbers (low, high), got {valid_range!r}") from exc
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(f"valid range must satisfy low < high, got {valid_range!r}")
    return low, high


def _sources(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"elevation.sources must be a list of source names, got {value!r}")
    return tuple(v.strip().lower() for v in value)


@dataclass(frozen=True)
class ElevationSettings:
    """Validated view of the ``elevation`` config section."""

    sources: tuple[str, ...]
    valid_range: tuple[float, float] | None
    tpi_radius_m: float
    max_abs_grade: float
    max_void_fraction: float
    void_fill_k: int
    batch_size: int
    margin_deg: float
    max_srtm_tiles: int
    refresh_dem: bool
    timeout_s: float
    max_retries: int
    backoff_s: float
    srtm_url_template: str
    open_meteo_url: str

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ElevationSettings":
        s = get_section(cfg, "elevation", DEFAULTS)
        try:
            valid_range = _check_range(s.get("valid_range_m"))
        except ValueError as exc:
            raise ConfigError(f"elevation.valid_range_m: {exc}") from exc
        void_fraction = _number(s, "max_void_fraction", minimum=0.0)
        if void_fraction >= 1.0:
            raise ConfigError("elevation.max_void_fraction must be < 1")
        return cls(
            sources=_sources(s.get("sources")),
            valid_range=valid_range,
            tpi_radius_m=_number(s, "tpi_radius_m", minimum=0.0, strict=True),
            max_abs_grade=_number(s, "max_abs_grade", minimum=0.0, strict=True),
            max_void_fraction=void_fraction,
            void_fill_k=int(_number(s, "void_fill_k", minimum=1.0)),
            batch_size=int(min(_number(s, "open_meteo_batch_size", minimum=1.0), OPEN_METEO_MAX_BATCH)),
            margin_deg=_number(s, "dem_margin_deg", minimum=0.0),
            max_srtm_tiles=int(_number(s, "max_srtm_tiles", minimum=1.0)),
            refresh_dem=bool(s.get("refresh_dem", False)),
            timeout_s=_number(s, "request_timeout_s", minimum=0.0, strict=True),
            max_retries=int(_number(s, "max_retries", minimum=1.0)),
            backoff_s=_number(s, "backoff_s", minimum=0.0),
            srtm_url_template=str(s.get("srtm_url_template")),
            open_meteo_url=str(s.get("open_meteo_url")),
        )


# --------------------------------------------------------------------------- enrichment provenance
def enrichment_settings(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Effective values of every :data:`ENRICHMENT_HASH_KEYS` setting (defaults applied)."""
    subset: dict[str, dict[str, Any]] = {}
    for name, keys in ENRICHMENT_HASH_KEYS.items():
        replace = ("osm_tags",) if name == "drains" else ()
        section = get_section(cfg, name, _SECTION_DEFAULTS[name], replace=replace)
        subset[name] = {key: section.get(key) for key in keys}
    return subset


def enrichment_config_hash(cfg: Mapping[str, Any]) -> str:
    """sha256[:16] of :func:`enrichment_settings` (stable across runs: sorted-key JSON)."""
    subset = enrichment_settings(cfg)
    return config_hash(subset, tuple(subset))


def enrichment_drift(G: nx.DiGraph, cfg: Mapping[str, Any]) -> str | None:
    """Why ``G``'s enrichment does not match ``cfg`` (``None`` when it matches or was never recorded).

    Graphs enriched before the hash existed carry no ``enrichment_config_hash``; they cannot
    be checked and count as matching.
    """
    stored = G.graph.get(ENRICHMENT_HASH_ATTR)
    if not stored:
        return None
    current = enrichment_config_hash(cfg)
    if str(stored) == current:
        return None
    return (f"its elevation / drain attributes were computed with other elevation, drains or region settings "
            f"(enrichment config hash {stored} -> {current})")


def synthetic_downgrades(before: nx.DiGraph, after: nx.DiGraph) -> list[str]:
    """Provenance that ``after`` lost relative to ``before`` (real elevation/drains -> synthetic fallback)."""
    lost = []
    old, new = before.graph.get("elevation_source"), after.graph.get("elevation_source")
    if old in REAL_ELEVATION_SOURCES and new not in REAL_ELEVATION_SOURCES:
        lost.append(f"elevation_source {old} -> {new}")
    old, new = before.graph.get("drain_source"), after.graph.get("drain_source")
    if old in REAL_DRAIN_SOURCES and new not in REAL_DRAIN_SOURCES:
        lost.append(f"drain_source {old} -> {new}")
    return lost
