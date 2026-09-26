"""Validated ``region`` / ``network`` settings of stage 01 (road-network extraction).

Split out of :mod:`src.data_pipeline.network`, which re-exports every public name.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.data_pipeline.overpass_guard import RETRY_DEFAULTS, RetryPolicy
from src.utils.config import ConfigError, config_hash, get_section
from src.utils.geo import validate_bbox

REGION_DEFAULTS: dict[str, Any] = {
    "bbox": [77.655, 12.915, 77.700, 12.950],
    "use_place_query": True,
    "max_place_area_km2": 60.0,
    "min_place_nodes": 50,
    "drain_fallback_lon": 77.6750,
}
NETWORK_DEFAULTS: dict[str, Any] = {
    "network_type": "drive",
    "simplify": True,
    "bidirectional": True,
    "keep_largest_component": True,
    "min_nodes": 20,
    "min_edge_length_m": 1.0,
    "default_edge_length_m": 25.0,
    "request_timeout_s": 180,
    "overpass_urls": ["https://overpass-api.de/api"],
    **RETRY_DEFAULTS,             # overpass_max_attempts / overpass_retry_pause_s (busy 429/504 mirrors)
    "osm_date": None,             # "YYYY-MM-DDTHH:MM:SSZ" pins the Overpass queries to that OSM snapshot
    "synthetic_grid": {"rows": 20, "cols": 20},
}
OSMNX_NETWORK_TYPES = frozenset({"all", "all_public", "bike", "drive", "drive_service", "walk"})
# Settings that change how OSM is fetched, not which streets come back (excluded from network_config_hash).
FETCH_ONLY_KEYS = frozenset({"request_timeout_s", "overpass_urls", *RETRY_DEFAULTS})
_OSM_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
DEFAULT_OVERPASS_SETTINGS = "[out:json][timeout:{timeout}]{maxsize}"


def _to_float(value: Any) -> float:
    """Parse a scalar to float; anything unparseable (incl. bools) becomes NaN."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan



# --------------------------------------------------------------------------- settings
def _config_bool(name: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ConfigError(f"{name} must be true or false, got {value!r}")


def _config_int(name: str, value: Any, minimum: int) -> int:
    numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    if not numeric or not float(value).is_integer() or int(value) < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _config_positive(name: str, value: Any) -> float:
    number = _to_float(value)  # NaN for bools, None and unparseable values
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a positive number, got {value!r}")
    return number


def _config_urls(name: str, value: Any) -> tuple[str, ...]:
    urls = [value] if isinstance(value, str) else value
    if not isinstance(urls, (list, tuple)) or not urls or not all(
        isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in urls
    ):
        raise ConfigError(f"{name} must be a non-empty list of http(s) URLs, got {value!r}")
    return tuple(u.strip().rstrip("/") for u in urls)


def _config_osm_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not _OSM_DATE.match(text):
        raise ConfigError(f"network.osm_date must be null or an ISO 8601 UTC time like \"2026-09-23T12:50:49Z\" "
                          f"(an Overpass [date:...] snapshot), got {value!r}")
    return text


def overpass_query_settings(osm_date: str | None) -> str:
    """osmnx ``settings.overpass_settings`` template; an ``osm_date`` adds an Overpass attic ``[date:...]`` pin."""
    return DEFAULT_OVERPASS_SETTINGS if osm_date is None else f'{DEFAULT_OVERPASS_SETTINGS}[date:"{osm_date}"]'


@dataclass(frozen=True)
class RegionSettings:
    """Validated ``region`` section plus the ``project.region`` place name."""

    place: str | None
    bbox: tuple[float, float, float, float]
    use_place_query: bool
    max_place_area_km2: float
    min_place_nodes: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> RegionSettings:
        section = get_section(cfg, "region", REGION_DEFAULTS)
        try:
            bbox = validate_bbox(section["bbox"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"region.bbox is invalid: {exc}") from exc
        place = (cfg.get("project") or {}).get("region")
        place = place.strip() if isinstance(place, str) and place.strip() else None
        return cls(
            place=place,
            bbox=bbox,
            use_place_query=_config_bool("region.use_place_query", section["use_place_query"]),
            max_place_area_km2=_config_positive("region.max_place_area_km2", section["max_place_area_km2"]),
            min_place_nodes=_config_int("region.min_place_nodes", section["min_place_nodes"], 1),
        )


@dataclass(frozen=True)
class NetworkSettings:
    """Validated ``network`` section."""

    network_type: str
    simplify: bool
    bidirectional: bool
    keep_largest_component: bool
    min_nodes: int
    min_edge_length_m: float
    default_edge_length_m: float
    request_timeout_s: float
    overpass_urls: tuple[str, ...]
    grid_rows: int
    grid_cols: int
    retry: RetryPolicy = RetryPolicy()
    osm_date: str | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> NetworkSettings:
        s = get_section(cfg, "network", NETWORK_DEFAULTS)
        if s["network_type"] not in OSMNX_NETWORK_TYPES:
            raise ConfigError(f"network.network_type must be one of {sorted(OSMNX_NETWORK_TYPES)}, "
                              f"got {s['network_type']!r}")
        grid = s["synthetic_grid"]
        if not isinstance(grid, Mapping):
            raise ConfigError(f"network.synthetic_grid must be a mapping with rows/cols, got {grid!r}")
        min_len = _config_positive("network.min_edge_length_m", s["min_edge_length_m"])
        default_len = _config_positive("network.default_edge_length_m", s["default_edge_length_m"])
        if default_len < min_len:
            raise ConfigError("network.default_edge_length_m must be >= network.min_edge_length_m")
        return cls(
            network_type=str(s["network_type"]),
            simplify=_config_bool("network.simplify", s["simplify"]),
            bidirectional=_config_bool("network.bidirectional", s["bidirectional"]),
            keep_largest_component=_config_bool("network.keep_largest_component", s["keep_largest_component"]),
            min_nodes=_config_int("network.min_nodes", s["min_nodes"], 2),
            min_edge_length_m=min_len,
            default_edge_length_m=default_len,
            request_timeout_s=_config_positive("network.request_timeout_s", s["request_timeout_s"]),
            overpass_urls=_config_urls("network.overpass_urls", s["overpass_urls"]),
            grid_rows=_config_int("network.synthetic_grid.rows", grid.get("rows"), 2),
            grid_cols=_config_int("network.synthetic_grid.cols", grid.get("cols"), 2),
            retry=RetryPolicy.from_config(cfg),
            osm_date=_config_osm_date(s.get("osm_date")),
        )


def network_config_hash(cfg: Mapping[str, Any]) -> str:
    """Hash of every setting that changes the extracted topology (stored on the graph).

    Fetch-only settings (:data:`FETCH_ONLY_KEYS`) are excluded; ``osm_date`` counts only when
    set, so graphs built before it existed keep their hash.
    """
    region = get_section(cfg, "region", REGION_DEFAULTS)
    net = get_section(cfg, "network", NETWORK_DEFAULTS)
    network = {k: v for k, v in net.items() if k not in FETCH_ONLY_KEYS and not (k == "osm_date" and v is None)}
    subset = {
        "place": (cfg.get("project") or {}).get("region"),
        "region": {k: region.get(k) for k in ("bbox", "use_place_query", "max_place_area_km2", "min_place_nodes")},
        "network": network,
    }
    return config_hash(subset, tuple(subset))
