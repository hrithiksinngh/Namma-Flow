"""Distance from every road junction to the nearest drain, stream, canal or lake.

Bengaluru's storm-water network is a chain of open drains (*rajakaluves*) feeding
tank-lakes such as Bellandur and Varthur, so a junction's distance to that network
is a first-order control on how fast its runoff can leave. Resolution chain used by
:func:`drain_distances`:

1. the cached GeoJSON at ``paths.waterways_file`` when its recorded query bbox covers
   the nodes and it was fetched with the same ``drains.osm_tags`` (online, an empty cached
   layer is re-queried: an empty Overpass answer may be a transient failure);
2. OpenStreetMap through ``osmnx.features_from_bbox``, every Overpass mirror in turn
   (cached on success; an empty answer is never cached). A mirror answering HTTP 429 / 504 is
   given up after ``network.overpass_max_attempts`` (:mod:`src.data_pipeline.overpass_guard`)
   instead of osmnx's unbounded retry; the OSM snapshot (``timestamp_osm_base``) is recorded in
   the cache metadata and ``network.osm_date`` pins the query to a past snapshot;
3. a synthetic north-south drain line at ``region.drain_fallback_lon`` (WARNING).

Water bodies that are not part of the storm-water network (``water=wastewater`` treatment
tanks, pools, fountains - ``drains.exclude_water_values``) are ignored.

Distances are metric (UTM zone of the nodes: 43N for Bengaluru), computed with a
shapely ``STRtree`` nearest query, 0 inside water polygons and clipped to
``drains.max_distance_m``. This module also owns the small coordinate/bbox helpers
shared with :mod:`src.data_pipeline.elevation`.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.errors import ShapelyError
from shapely.geometry import LineString

from src.data_pipeline.network_settings import _config_osm_date, overpass_query_settings
from src.data_pipeline.overpass_guard import RetryPolicy, bounded_osmnx_requests
from src.utils.config import ConfigError, get_section, is_offline, resolve_path
from src.utils.geo import bbox_center, haversine_m, point_in_bbox, to_utm, validate_bbox
from src.utils.http import NetworkUnavailable
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "osm_tags": {"waterway": ["drain", "canal", "stream", "river", "ditch"], "natural": ["water"]},
    "max_distance_m": 5000.0,
    "bbox_margin_deg": 0.02,
    "min_polygon_area_m2": 1000.0,
    "request_timeout_s": 180,
    "refresh": False,
    # OSM ``water=*`` values that are not storm-water bodies (treatment tanks, pools, fountains).
    "exclude_water_values": ["wastewater", "reflecting_pool", "swimming_pool", "pool", "fountain"],
}
# Mapping-valued settings a user value replaces wholesale (merging would re-add default tags).
_REPLACED_KEYS = ("osm_tags",)
SOURCE_OSM = "osm"
SOURCE_SYNTHETIC = "synthetic_line"
DEFAULT_DRAIN_LON = 77.6750
DEFAULT_WATERWAYS_FILE = "data/interim/waterways.geojson"
DEFAULT_OSM_CACHE_DIR = "data/interim/osm_cache"
NODE_BBOX_PAD_DEG = 0.001
CACHE_META_KEY = "namma_flow"
WGS84 = "EPSG:4326"
_KEEP_COLUMNS = ("element", "osmid", "waterway", "natural", "water", "name", "tunnel", "intermittent")
_SINGLE_PART_TYPES = ("LineString", "LinearRing", "Polygon")
_MULTI_PART_TYPES = ("MultiLineString", "MultiPolygon", "MultiPoint", "GeometryCollection")
_MAX_EXPLODE_DEPTH = 8


@dataclass(frozen=True)
class DrainResult:
    """Per-node drain distances plus provenance."""

    distance_m: np.ndarray
    source: str
    n_features: int


# --------------------------------------------------------------------------- shared helpers
def validate_coords(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lon, lat)`` as new 1-D float64 arrays after validating shape, finiteness and range."""
    try:
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lon/lat must be numeric: {exc}") from exc
    if lon_arr.ndim > 1 or lat_arr.ndim > 1:
        raise ValueError(f"lon/lat must be scalars or 1-D arrays, got shapes {lon_arr.shape} and {lat_arr.shape}")
    lon_arr = np.atleast_1d(lon_arr).copy()
    lat_arr = np.atleast_1d(lat_arr).copy()
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(f"lon and lat must have the same shape, got {lon_arr.shape} and {lat_arr.shape}")
    if not (np.isfinite(lon_arr).all() and np.isfinite(lat_arr).all()):
        raise ValueError("lon/lat must be finite (found NaN or inf coordinates)")
    if np.any(np.abs(lon_arr) > 180.0):
        raise ValueError("longitude values must lie in [-180, 180]")
    if np.any(np.abs(lat_arr) > 90.0):
        raise ValueError("latitude values must lie in [-90, 90]")
    return lon_arr, lat_arr


def expand_bbox(bbox: Sequence[float], margin_deg: float) -> tuple[float, float, float, float]:
    """Grow ``[west, south, east, north]`` by ``margin_deg`` on every side, clamped to the globe."""
    west, south, east, north = validate_bbox(bbox)
    margin = float(margin_deg)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError(f"bbox margin must be a finite number >= 0, got {margin_deg!r}")
    return (
        max(-180.0, west - margin),
        max(-90.0, south - margin),
        min(180.0, east + margin),
        min(90.0, north + margin),
    )


def nodes_bbox(lon: Any, lat: Any, cfg: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Bbox used for DEM/waterway acquisition: ``region.bbox`` if it holds every node, else the padded node extent.

    Preferring the configured bbox keeps cache file names stable across runs.
    """
    lon, lat = validate_coords(lon, lat)
    if lon.size == 0:
        raise ValueError("Cannot derive a bbox from zero nodes")
    region_bbox = (cfg.get("region") or {}).get("bbox")
    if region_bbox is not None:
        try:
            bbox = validate_bbox(region_bbox)
        except ValueError as exc:
            LOGGER.warning("Ignoring invalid region.bbox %r: %s", region_bbox, exc)
        else:
            if point_in_bbox(lon, lat, bbox).all():
                return bbox
    pad = NODE_BBOX_PAD_DEG
    return validate_bbox(
        (
            max(-180.0, float(lon.min()) - pad),
            max(-90.0, float(lat.min()) - pad),
            min(180.0, float(lon.max()) + pad),
            min(90.0, float(lat.max()) + pad),
        )
    )


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing ``(lon, lat)`` (32643 = 43N for Bengaluru)."""
    zone = int(np.floor((float(lon) + 180.0) / 6.0)) % 60 + 1
    return (32600 if float(lat) >= 0 else 32700) + zone


def _section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``drains`` section over :data:`DEFAULTS` (``osm_tags`` is replaced, not merged)."""
    return get_section(cfg, "drains", DEFAULTS, replace=_REPLACED_KEYS)


def _positive(section: Mapping[str, Any], key: str, *, allow_zero: bool = False) -> float:
    try:
        value = float(section.get(key))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"drains.{key} must be a number, got {section.get(key)!r}") from exc
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ConfigError(f"drains.{key} must be {'>= 0' if allow_zero else '> 0'}, got {value}")
    return value


def _osm_tags(section: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``drains.osm_tags`` (key -> True | value | [values]) and return a clean copy."""
    tags = section.get("osm_tags")
    if not isinstance(tags, Mapping) or not tags:
        raise ConfigError("drains.osm_tags must be a non-empty mapping of OSM key -> value(s)")
    clean: dict[str, Any] = {}
    for key, value in tags.items():
        if isinstance(value, (bool, str)):
            clean[str(key)] = value
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            clean[str(key)] = list(value)
        else:
            raise ConfigError(f"drains.osm_tags.{key} must be true, a string or a list of strings, got {value!r}")
    return clean


def _tags_key(tags: Mapping[str, Any]) -> dict[str, Any]:
    """Order-insensitive representation of an OSM tag filter, for cache comparisons."""
    key: dict[str, Any] = {}
    for name, value in tags.items():
        if isinstance(value, bool):
            key[str(name)] = value
        elif isinstance(value, str):
            key[str(name)] = [value]
        elif isinstance(value, (list, tuple)):
            key[str(name)] = sorted(str(v) for v in value)
        else:
            key[str(name)] = str(value)
    return key


# --------------------------------------------------------------------------- synthetic fallback
def _fallback_drain_lon(cfg: Mapping[str, Any]) -> float:
    region = cfg.get("region") or {}
    value = region.get("drain_fallback_lon")
    if value is None:
        bbox = region.get("bbox")
        return bbox_center(bbox)[0] if bbox is not None else DEFAULT_DRAIN_LON
    try:
        lon = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"region.drain_fallback_lon must be a longitude, got {value!r}") from exc
    if not np.isfinite(lon) or abs(lon) > 180.0:
        raise ConfigError(f"region.drain_fallback_lon must lie in [-180, 180], got {lon}")
    return lon


def synthetic_drain_distance(lon: Any, lat: Any, cfg: Mapping[str, Any]) -> np.ndarray:
    """Spec fallback: east-west metres to a north-south drain at ``region.drain_fallback_lon``, clipped."""
    lon, lat = validate_coords(lon, lat)
    max_distance = _positive(_section(cfg), "max_distance_m")
    drain_lon = _fallback_drain_lon(cfg)
    if lon.size == 0:
        return np.zeros(0, dtype=np.float64)
    dist = np.atleast_1d(haversine_m(lon, lat, np.full_like(lon, drain_lon), lat))
    return np.clip(dist.astype(np.float64), 0.0, max_distance)


# --------------------------------------------------------------------------- geometry cleaning
def _empty_waterways() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"kind": pd.Series([], dtype=object)}, geometry=gpd.GeoSeries([], crs=WGS84), crs=WGS84)


def _stringify(value: Any) -> str | None:
    """GeoJSON-safe tag value: lists joined with ``;``, missing values (None/NaN/NA) -> None."""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # array-like values are not "missing"
        pass
    return str(value)


def _to_wgs84(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.crs is None:
        LOGGER.warning("Waterway geometries have no CRS; assuming EPSG:4326")
        return frame.set_crs(WGS84)
    if frame.crs.to_epsg() != 4326:
        return frame.to_crs(WGS84)
    return frame


def _drop_small_polygons(frame: gpd.GeoDataFrame, min_area_m2: float) -> gpd.GeoDataFrame:
    is_polygon = (frame["kind"] == "polygon").to_numpy()
    if min_area_m2 <= 0 or not is_polygon.any():
        return frame
    west, south, east, north = frame.total_bounds
    areas = frame.to_crs(utm_epsg((west + east) / 2, (south + north) / 2)).area.to_numpy()
    too_small = is_polygon & (areas < min_area_m2)
    if too_small.any():
        LOGGER.info("Ignoring %d water polygon(s) smaller than %.0f m2", int(too_small.sum()), min_area_m2)
    return frame.loc[~too_small]


def _explode_fully(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode until no multi-part geometry or collection is left.

    ``make_valid`` on a self-intersecting lake with a dangling spike returns
    ``GEOMETRYCOLLECTION(MULTIPOLYGON(...), LINESTRING)``; a single ``explode`` leaves the
    MultiPolygon, which the single-part filter would then drop (losing the lake).
    """
    for _ in range(_MAX_EXPLODE_DEPTH):
        frame = frame.explode(index_parts=False, ignore_index=True)
        if not frame.geom_type.isin(_MULTI_PART_TYPES).any():
            break
    return frame


def _exclude_values(section: Mapping[str, Any]) -> frozenset[str]:
    values = section.get("exclude_water_values") or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
        raise ConfigError(f"drains.exclude_water_values must be a list of OSM water=* values, got {values!r}")
    return frozenset(v.strip().lower() for v in values)


def _drop_excluded_water(frame: gpd.GeoDataFrame, excluded: frozenset[str]) -> gpd.GeoDataFrame:
    """Drop features whose ``water`` tag (``;``-joined values allowed) is in ``excluded``."""
    if not excluded or len(frame) == 0 or "water" not in frame.columns:
        return frame
    values = frame["water"].map(lambda v: {p.strip().lower() for p in str(v).split(";")} if isinstance(v, str) else set())
    drop = values.map(lambda tags: bool(tags & excluded)).to_numpy(dtype=bool)
    if not drop.any():
        return frame
    LOGGER.info("Ignoring %d water feature(s) that are not storm-water bodies (water=%s)",
                int(drop.sum()), "|".join(sorted(excluded)))
    kept = frame.loc[~drop]
    return kept if len(kept) else _empty_waterways()


def clean_waterways(gdf: gpd.GeoDataFrame | None, min_polygon_area_m2: float = 0.0) -> gpd.GeoDataFrame:
    """Normalise raw OSM features to single-part lines/polygons in EPSG:4326 (new frame).

    Points are dropped, invalid geometries repaired with ``make_valid``, multi-part and
    collection geometries exploded, tiny polygons (fountains, tanks) removed, list-valued
    tags flattened to ``;``-joined strings. Adds ``kind`` = ``line`` | ``polygon``.
    """
    if gdf is None or len(gdf) == 0 or not isinstance(gdf, gpd.GeoDataFrame):
        return _empty_waterways()
    frame = _to_wgs84(gdf.copy())
    index_names = [name for name in frame.index.names if name is not None]
    # osmnx indexes features by (element, id); keep those as columns unless they already exist.
    frame = frame.reset_index(drop=not index_names or bool(set(index_names) & set(frame.columns)))
    if "id" in frame.columns and "osmid" not in frame.columns:
        frame = frame.rename(columns={"id": "osmid"})
    geometry = frame.geometry
    frame = frame.loc[geometry.notna().to_numpy() & ~geometry.is_empty.to_numpy()]
    invalid = ~frame.geometry.is_valid.to_numpy()
    if invalid.any():
        LOGGER.info("Repairing %d invalid waterway geometries with make_valid", int(invalid.sum()))
        frame = frame.set_geometry(frame.geometry.make_valid())
    frame = _explode_fully(frame)
    frame = frame.loc[frame.geom_type.isin(_SINGLE_PART_TYPES).to_numpy() & ~frame.geometry.is_empty.to_numpy()]
    if len(frame) == 0:
        return _empty_waterways()
    rings = (frame.geom_type == "LinearRing").to_numpy()
    if rings.any():  # GeoJSON has no LinearRing type
        frame.loc[rings, "geometry"] = [LineString(g.coords) for g in frame.geometry[rings]]
    frame["kind"] = np.where(frame.geom_type.to_numpy() == "Polygon", "polygon", "line")
    frame = _drop_small_polygons(frame, float(min_polygon_area_m2))
    if len(frame) == 0:
        return _empty_waterways()
    columns = [c for c in _KEEP_COLUMNS if c in frame.columns]
    data = {c: frame[c].map(_stringify).astype(object).to_numpy() for c in columns}
    data["kind"] = frame["kind"].to_numpy()
    return gpd.GeoDataFrame(data, geometry=frame.geometry.to_numpy(), crs=WGS84)


# --------------------------------------------------------------------------- cache
def save_waterways_cache(
    gdf: gpd.GeoDataFrame, path: str | Path, query_bbox: Sequence[float], tags: Mapping[str, Any],
    osm_base: str | None = None,
) -> Path:
    """Write ``gdf`` as a GeoJSON FeatureCollection (atomically) with the query bbox/tags as metadata.

    ``osm_base`` (the Overpass ``timestamp_osm_base``) records which OSM snapshot the layer is.
    """
    bbox = validate_bbox(query_bbox)
    frame = _to_wgs84(gdf) if gdf is not None and len(gdf) else None
    if frame is None:
        collection: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    else:
        collection = json.loads(frame.to_json(drop_id=True))
    collection[CACHE_META_KEY] = {
        "query_bbox": list(bbox),
        "tags": dict(tags),
        "n_features": 0 if frame is None else int(len(frame)),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "osm_base_utc": osm_base or "unknown",
    }
    return atomic_write_text(Path(path), json.dumps(collection))


def _cache_covers(meta: Any, required_bbox: Sequence[float] | None, tags: Mapping[str, Any] | None) -> bool:
    if not isinstance(meta, Mapping):
        return True  # foreign GeoJSON without our metadata: trust it
    cached_tags = meta.get("tags")
    if tags is not None and cached_tags is not None:
        if not isinstance(cached_tags, Mapping) or _tags_key(cached_tags) != _tags_key(tags):
            return False
    if required_bbox is None:
        return True
    try:
        c_west, c_south, c_east, c_north = validate_bbox(meta.get("query_bbox"))
    except (TypeError, ValueError):
        return False
    west, south, east, north = validate_bbox(required_bbox)
    tol = 1e-9
    return c_west <= west + tol and c_south <= south + tol and c_east >= east - tol and c_north >= north - tol


def load_cached_waterways(
    path: str | Path,
    required_bbox: Sequence[float] | None = None,
    tags: Mapping[str, Any] | None = None,
) -> tuple[gpd.GeoDataFrame | None, bool]:
    """Read the waterways cache. Returns ``(frame | None, covers)``; unreadable files give ``(None, False)``.

    ``covers`` is true when the cached query bbox contains ``required_bbox`` and the
    tag filter matches (files without Namma-Flow metadata are trusted).
    """
    path = Path(path)
    if not path.exists():
        return None, False
    try:
        collection = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
            raise ValueError("not a GeoJSON FeatureCollection")
        features = collection.get("features")
        if not isinstance(features, list):
            raise ValueError("'features' is not a list")
        frame = clean_waterways(gpd.GeoDataFrame.from_features(features, crs=WGS84)) if features else _empty_waterways()
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ShapelyError) as exc:
        LOGGER.warning("Ignoring unreadable waterways cache %s: %s", path, exc)
        return None, False
    return frame, _cache_covers(collection.get(CACHE_META_KEY), required_bbox, tags)


# --------------------------------------------------------------------------- OSM fetch
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api"

@contextmanager
def _osmnx_settings(**values: Any) -> Iterator[Any]:
    """Temporarily override ``osmnx.settings`` (restored afterwards; other stages share the module)."""
    import osmnx as ox

    previous = {name: getattr(ox.settings, name) for name in values}
    for name, value in values.items():
        setattr(ox.settings, name, value)
    try:
        yield ox
    finally:
        for name, value in previous.items():
            setattr(ox.settings, name, value)


def _overpass_urls(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Overpass endpoints to try, shared with the network stage (``network.overpass_urls``)."""
    urls = (cfg.get("network") or {}).get("overpass_urls") or [DEFAULT_OVERPASS_URL]
    if isinstance(urls, str):
        urls = [urls]
    return tuple(str(u).rstrip("/") for u in urls if str(u).startswith(("http://", "https://"))) or (
        DEFAULT_OVERPASS_URL,
    )


def _overpass_overrides(cfg: Mapping[str, Any], section: Mapping[str, Any]) -> dict[str, Any]:
    """osmnx settings of the waterway query (project cache, timeout, optional ``network.osm_date`` pin)."""
    timeout = _positive(section, "request_timeout_s")
    # osmnx embeds the timeout in the query text (its cache key): keep integral values as int
    # so this stage shares cache entries with the network stage.
    timeout = int(timeout) if float(timeout).is_integer() else timeout
    cache_dir = resolve_path(cfg, "osm_cache_dir", DEFAULT_OSM_CACHE_DIR)
    overrides: dict[str, Any] = {"cache_folder": str(cache_dir), "use_cache": True, "requests_timeout": timeout}
    osm_date = _config_osm_date((cfg.get("network") or {}).get("osm_date"))
    if osm_date is not None:
        overrides["overpass_settings"] = overpass_query_settings(osm_date)
    return overrides


def _query_osm_features(
    cfg: Mapping[str, Any], section: Mapping[str, Any], bbox: tuple[float, ...], tags: dict[str, Any]
) -> tuple[gpd.GeoDataFrame | None, str | None]:
    """Query each Overpass mirror in turn: ``(features | None, OSM snapshot timestamp | None)``.

    ``None`` features when every answering mirror had no features. An empty answer (osmnx
    ``InsufficientResponseError``) can be a transient Overpass failure (e.g. a timeout /
    out-of-memory ``remark`` with zero elements), so the next mirror is tried; a busy mirror
    (HTTP 429 / 504) is given up after ``network.overpass_max_attempts``.
    """
    try:
        from osmnx._errors import InsufficientResponseError
    except ImportError:  # pragma: no cover - private module moved in a future osmnx

        class InsufficientResponseError(Exception):  # type: ignore[no-redef]
            """Placeholder that never matches."""

    overrides = _overpass_overrides(cfg, section)
    policy = RetryPolicy.from_config(cfg)
    LOGGER.info("Querying OSM waterways in bbox %s with tags %s", bbox, tags)
    errors, empty = [], []
    urls = _overpass_urls(cfg)
    for index, url in enumerate(urls, start=1):
        LOGGER.info("Overpass mirror %d of %d: %s (waterways)", index, len(urls), url)
        try:
            with _osmnx_settings(**overrides, overpass_url=url) as ox, bounded_osmnx_requests(ox, policy) as log:
                return ox.features_from_bbox(bbox=bbox, tags=tags), log.latest_osm_base
        except InsufficientResponseError as exc:
            LOGGER.warning("Overpass endpoint %s returned no waterway features for bbox %s (%s)", url, bbox, exc)
            empty.append(url)
        except Exception as exc:  # osmnx surfaces transport, HTTP and Overpass failures with many types
            LOGGER.warning("Overpass endpoint %s failed for waterways: %s: %s", url, type(exc).__name__, exc)
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    if empty:
        return None, None
    raise NetworkUnavailable("OSM waterways query failed on every Overpass endpoint: " + " | ".join(errors))


def fetch_waterways(cfg: Mapping[str, Any], bbox: Sequence[float]) -> gpd.GeoDataFrame:
    """Download drains/streams/canals/lakes from OSM for ``bbox`` and cache them to ``paths.waterways_file``.

    Raises :class:`NetworkUnavailable` offline or when Overpass fails. An empty answer is
    returned as an empty frame but never cached (it may be a transient Overpass failure, and
    it must not overwrite a good cache).
    """
    section = _section(cfg)
    query_bbox = validate_bbox(bbox)
    tags = _osm_tags(section)
    min_area = _positive(section, "min_polygon_area_m2", allow_zero=True)
    if is_offline(cfg):
        raise NetworkUnavailable("Offline mode: not querying OSM for waterways")
    raw, osm_base = _query_osm_features(cfg, section, query_bbox, tags)
    frame = clean_waterways(raw, min_area)
    n_polygons = int((frame["kind"] == "polygon").sum()) if len(frame) else 0
    LOGGER.info("OSM waterways: %d features (%d lines, %d polygons)", len(frame), len(frame) - n_polygons, n_polygons)
    path = resolve_path(cfg, "waterways_file", DEFAULT_WATERWAYS_FILE)
    if len(frame) == 0:
        LOGGER.warning("Not caching an empty waterway answer for bbox %s (possibly a transient Overpass "
                       "failure); it will be queried again next time", query_bbox)
        return frame
    try:
        save_waterways_cache(frame, path, query_bbox, tags, osm_base=osm_base)
    except OSError as exc:
        LOGGER.warning("Could not cache waterways to %s: %s", path, exc)
    return frame


# --------------------------------------------------------------------------- distances
def distance_to_drains(
    lon: Any, lat: Any, waterways_gdf: gpd.GeoDataFrame | None, cfg: Mapping[str, Any]
) -> np.ndarray:
    """Metres from each point to the nearest waterway line or water-polygon boundary (0 inside), clipped."""
    lon, lat = validate_coords(lon, lat)
    section = _section(cfg)
    max_distance = _positive(section, "max_distance_m")
    frame = _drop_excluded_water(clean_waterways(waterways_gdf), _exclude_values(section))
    if len(frame) == 0:
        raise ValueError("Waterway layer contains no line or polygon geometries")
    if lon.size == 0:
        return np.zeros(0, dtype=np.float64)
    epsg = utm_epsg(float(lon.mean()), float(lat.mean()))
    geometries = frame.to_crs(epsg).geometry.to_numpy()
    x, y = to_utm(lon, lat, epsg)
    points = shapely.points(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    tree = shapely.STRtree(geometries)
    (input_idx, _), dist = tree.query_nearest(
        points, max_distance=max_distance, return_distance=True, all_matches=False
    )
    out = np.full(lon.shape, max_distance, dtype=np.float64)
    out[input_idx] = dist
    return np.clip(out, 0.0, max_distance)


def _resolve_waterways(
    cfg: Mapping[str, Any], section: Mapping[str, Any], query_bbox: tuple[float, ...]
) -> gpd.GeoDataFrame | None:
    """Cache (if it covers) -> OSM -> stale cache -> ``None``."""
    path = resolve_path(cfg, "waterways_file", DEFAULT_WATERWAYS_FILE)
    tags = _osm_tags(section)
    refresh = bool(section.get("refresh", False))
    cached, covers = load_cached_waterways(path, query_bbox, tags)
    offline = is_offline(cfg)
    if cached is not None and covers and not refresh and (len(cached) or offline):
        LOGGER.info("Using cached waterways %s (%d features)", path, len(cached))
        return cached
    if cached is not None and not covers:
        LOGGER.warning("Cached waterways %s does not cover bbox %s (or its tag filter differs)", path, query_bbox)
    if offline:
        if cached is not None:
            LOGGER.warning("Offline: using cached waterways %s as is", path)
        return cached
    try:
        fresh = fetch_waterways(cfg, query_bbox)
    except NetworkUnavailable as exc:
        LOGGER.warning("OSM waterways unavailable: %s", exc)
        if cached is not None:
            LOGGER.warning("Falling back to the existing waterways cache %s", path)
        return cached
    if len(fresh) == 0 and cached is not None and len(cached):
        LOGGER.warning("OSM answered with no waterways; keeping the existing cache %s (%d features)", path, len(cached))
        return cached
    return fresh


def compute_drain_distances(lon: Any, lat: Any, cfg: Mapping[str, Any]) -> DrainResult:
    """Distance-to-drain for every node via cache -> OSM -> synthetic line; see the module docstring."""
    section = _section(cfg)
    _positive(section, "max_distance_m")
    lon, lat = validate_coords(lon, lat)
    if lon.size == 0:
        return DrainResult(np.zeros(0, dtype=np.float64), SOURCE_SYNTHETIC, 0)
    margin = _positive(section, "bbox_margin_deg", allow_zero=True)
    query_bbox = expand_bbox(nodes_bbox(lon, lat, cfg), margin)
    frame = _resolve_waterways(cfg, section, query_bbox)
    if frame is not None:
        frame = _drop_excluded_water(frame, _exclude_values(section))
    if frame is not None and len(frame):
        try:
            dist = distance_to_drains(lon, lat, frame, cfg)
        except ConfigError:
            raise
        except ValueError as exc:
            LOGGER.warning("Could not use OSM waterways (%s)", exc)
        else:
            LOGGER.info(
                "Drain distances from %d OSM features: median %.0f m, max %.0f m",
                len(frame), float(np.median(dist)), float(dist.max()),
            )
            return DrainResult(dist, SOURCE_OSM, int(len(frame)))
    LOGGER.warning(
        "No OSM waterways available for bbox %s; using the synthetic drain line at lon %.4f",
        query_bbox, _fallback_drain_lon(cfg),
    )
    return DrainResult(synthetic_drain_distance(lon, lat, cfg), SOURCE_SYNTHETIC, 0)


def drain_distances(lon: Any, lat: Any, cfg: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Contract API: ``(distance_m [N], source)`` with source ``osm`` or ``synthetic_line``."""
    result = compute_drain_distances(lon, lat, cfg)
    return result.distance_m, result.source
