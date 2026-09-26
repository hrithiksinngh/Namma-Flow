"""Small geodesy helpers shared by every stage (no heavy GIS dependencies)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
UTM_43N_EPSG = 32643  # UTM zone covering Bengaluru
WGS84_EPSG = 4326


def validate_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate ``[west, south, east, north]`` and return it as a float tuple."""
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must have exactly four values [west, south, east, north]")
    try:
        west, south, east, north = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox values must be numeric: {bbox!r}") from exc
    if not all(np.isfinite([west, south, east, north])):
        raise ValueError("bbox values must be finite")
    if not (-180.0 <= west < east <= 180.0):
        raise ValueError(f"bbox longitudes must satisfy -180 <= west < east <= 180, got {west}, {east}")
    if not (-90.0 <= south < north <= 90.0):
        raise ValueError(f"bbox latitudes must satisfy -90 <= south < north <= 90, got {south}, {north}")
    return west, south, east, north


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    """Return ``(lon, lat)`` of the bbox centre."""
    west, south, east, north = validate_bbox(bbox)
    return (west + east) / 2.0, (south + north) / 2.0


def bbox_area_km2(bbox: Sequence[float]) -> float:
    west, south, east, north = validate_bbox(bbox)
    width = haversine_m(west, (south + north) / 2.0, east, (south + north) / 2.0)
    height = haversine_m(west, south, west, north)
    return float(width * height / 1e6)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray | float:
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(dist) if dist.ndim == 0 else dist


def lonlat_to_local_xy(lon, lat, origin: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to metres around ``origin`` (default: mean point).

    Accurate to well under 0.1 % over a few tens of kilometres, which is all a
    city-scale corridor needs, and avoids a pyproj round-trip in hot loops.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if origin is None:
        origin = (float(np.mean(lon)), float(np.mean(lat))) if lon.size else (0.0, 0.0)
    lon0, lat0 = origin
    k = np.pi / 180.0 * EARTH_RADIUS_M
    x = (lon - lon0) * k * np.cos(np.radians(lat0))
    y = (lat - lat0) * k
    return x, y


def to_utm(lon, lat, epsg: int = UTM_43N_EPSG) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat to a metric UTM CRS with pyproj."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(WGS84_EPSG, epsg, always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def point_in_bbox(lon, lat, bbox: Sequence[float]) -> np.ndarray:
    west, south, east, north = validate_bbox(bbox)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
