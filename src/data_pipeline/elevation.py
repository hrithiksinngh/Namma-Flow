"""Elevation & drainage engine: node elevations, edge grades, topographic position and flow routing.

Node elevations come from the first source in ``elevation.sources`` that works:

* ``local`` - any GeoTIFF in ``paths.dem_dir`` (except our ``srtm_*`` cache files) whose
  bounds cover the bbox, in any CRS (node coordinates are reprojected);
* ``srtm`` - a cached ``srtm_<w>_<s>_<e>_<n>.tif`` clip, else the public 1-arc-second
  AWS "skadi" tiles (``N12E077.hgt.gz``; several tiles merged), gunzipped, clipped to
  the bbox plus ``dem_margin_deg`` and written as a compressed GeoTIFF;
* ``open_meteo`` - the Open-Meteo elevation API (Copernicus 90 m), batched;
* ``synthetic`` - the spec's valley formula ``880 + 30 sin(lat*100) + 15 cos(lon*100)``.

Samples that are nodata or outside ``valid_range_m`` are voids; a source with more than
``max_void_fraction`` voids is rejected, otherwise voids are filled by inverse-distance
weighting of the nearest valid nodes. :func:`enrich_graph` then adds relative elevation
(TPI), distance to drain (:mod:`src.data_pipeline.drains`), clipped edge grades and
steepest-descent flow accumulation / sinks (:mod:`src.data_pipeline.terrain`), returning a
new graph.

Offline, ``refresh_dem`` is ignored with a WARNING (cached SRTM clips / tiles are reused:
nothing could be re-downloaded), and :func:`run_elevation_stage` refuses to overwrite a graph
whose elevations or drains came from real data with synthetic fallbacks unless
``allow_synthetic`` is set. The enriched graph records the enrichment config hash
(:func:`~src.data_pipeline.elevation_settings.enrichment_config_hash`); settings, the hash and
the downgrade check live in :mod:`src.data_pipeline.elevation_settings`.
"""

from __future__ import annotations

import gzip
import math
import tempfile
import zlib
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import rasterio
from rasterio.errors import CRSError, RasterioError
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from scipy.spatial import cKDTree

from src.data_pipeline.drains import compute_drain_distances, expand_bbox, nodes_bbox, validate_coords
from src.data_pipeline.elevation_settings import (  # noqa: F401 - re-exported public API
    DEFAULTS,
    ENRICHMENT_HASH_ATTR,
    OPEN_METEO_MAX_BATCH,
    REAL_DRAIN_SOURCES,
    REAL_ELEVATION_SOURCES,
    ElevationSettings,
    _check_range,
    _number,
    _sources,
    enrichment_config_hash,
    enrichment_drift,
    synthetic_downgrades,
)
from src.data_pipeline.graph_io import GraphFormatError, graph_to_arrays, load_graph, save_graph
from src.data_pipeline.terrain import flow_accumulation, relative_elevation
from src.utils.config import ConfigError, get_section, is_offline, resolve_path
from src.utils.geo import haversine_m, lonlat_to_local_xy, validate_bbox
from src.utils.http import NetworkUnavailable, get_bytes, get_json
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_bytes

LOGGER = get_logger(__name__)

SOURCE_LABELS = {"local": "local_dem", "srtm": "srtm", "open_meteo": "open_meteo", "synthetic": "synthetic"}
SRTM_NODATA = -32768
SRTM_CACHE_PREFIX = "srtm_"
DEM_SUFFIXES = (".tif", ".tiff")
DEFAULT_GRAPH_FILE = "data/interim/bellandur_osm.graphml"
_SRTM_BYTES = {1201 * 1201 * 2, 3601 * 3601 * 2}  # SRTM3 / SRTM1 int16 grids
_GZIP_MAGIC = b"\x1f\x8b"
_GRAPHML_READER_KEYS = ("node_default", "edge_default")
_SOURCE_ERRORS = (NetworkUnavailable, RasterioError, CRSError, ValueError, OSError)


class ElevationError(RuntimeError):
    """A DEM file or SRTM tile is unreadable, corrupt or contains no usable data."""


@dataclass(frozen=True)
class ElevationResult:
    """Node elevations plus provenance (source label, void statistics, DEM file used)."""

    elevation: np.ndarray
    source: str
    n_voids: int
    void_fraction: float
    dem_path: str | None


# --------------------------------------------------------------------------- DEM discovery
def _is_wgs84(crs: Any) -> bool:
    try:
        return crs.to_epsg() == 4326
    except (CRSError, AttributeError):
        return False


def _dem_covers(path: Path, bbox: tuple[float, float, float, float]) -> bool:
    try:
        with rasterio.open(path) as src:
            if src.crs is None or src.count < 1:
                LOGGER.warning("Skipping DEM %s: no CRS or no bands", path.name)
                return False
            target = bbox if _is_wgs84(src.crs) else transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            (left, right), (bottom, top) = sorted(src.bounds[::2]), sorted(src.bounds[1::2])
    except (RasterioError, CRSError, ValueError, OSError) as exc:
        LOGGER.warning("Skipping unreadable DEM %s: %s", path.name, exc)
        return False
    west, south, east, north = target
    tol = 1e-7
    return left <= west + tol and bottom <= south + tol and right >= east - tol and top >= north - tol


def find_local_dem(dem_dir: str | Path, bbox: Sequence[float], *, cached: bool = False) -> Path | None:
    """First GeoTIFF in ``dem_dir`` (sorted by name) covering ``bbox``.

    ``cached=False`` considers user-supplied DEMs only; ``cached=True`` only our
    ``srtm_*`` clips. Unreadable files are skipped with a WARNING.
    """
    directory = Path(dem_dir)
    target = validate_bbox(bbox)
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in DEM_SUFFIXES or path.name.startswith("."):
            continue
        if path.name.startswith(SRTM_CACHE_PREFIX) != cached:
            continue
        if _dem_covers(path, target):
            return path
    return None


# --------------------------------------------------------------------------- SRTM
def srtm_tile_names(bbox: Sequence[float], margin_deg: float = 0.0) -> list[str]:
    """1x1 degree SRTM tile names (e.g. ``N12E077``) intersecting ``bbox`` grown by ``margin_deg``."""
    west, south, east, north = expand_bbox(bbox, margin_deg)
    lat_range = range(math.floor(south), max(math.floor(south), math.ceil(north) - 1) + 1)
    lon_range = range(math.floor(west), max(math.floor(west), math.ceil(east) - 1) + 1)
    return [
        f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        for lat in lat_range
        for lon in lon_range
    ]


def _decode_tile(payload: bytes, tile: str) -> bytes:
    """Gunzip (if needed) and size-check an SRTM ``.hgt`` payload."""
    data = payload
    if payload[:2] == _GZIP_MAGIC:
        try:
            data = gzip.decompress(payload)
        except (OSError, EOFError, zlib.error) as exc:
            raise ElevationError(f"SRTM tile {tile} is not valid gzip: {exc}") from exc
    if len(data) not in _SRTM_BYTES:
        raise ElevationError(
            f"SRTM tile {tile} has unexpected size {len(data)} bytes (expected a 1201^2 or 3601^2 int16 grid)"
        )
    return data


def _refresh_requested(cfg: Mapping[str, Any], settings: ElevationSettings, what: str) -> bool:
    """``refresh_dem`` as it applies to this run: ignored offline (nothing could be re-downloaded)."""
    if not settings.refresh_dem:
        return False
    if is_offline(cfg):
        LOGGER.warning("elevation.refresh_dem ignored offline (cannot re-download): reusing the cached %s", what)
        return False
    return True


def _fetch_tile(cfg: Mapping[str, Any], settings: ElevationSettings, tile: str, dem_dir: Path, workdir: Path) -> Path:
    """Materialise ``<workdir>/<tile>.hgt`` from the raw-tile cache or a download."""
    raw_path = dem_dir / f"{tile}.hgt.gz"
    hgt: bytes | None = None
    if raw_path.exists() and not _refresh_requested(cfg, settings, f"SRTM tile {raw_path.name}"):
        try:
            hgt = _decode_tile(raw_path.read_bytes(), tile)
        except (ElevationError, OSError) as exc:
            LOGGER.warning("Cached SRTM tile %s is unusable (%s); downloading it again", raw_path.name, exc)
    if hgt is None:
        if is_offline(cfg):
            state = "is unusable" if raw_path.exists() else "is not cached"
            raise NetworkUnavailable(f"Offline mode: SRTM tile {tile} {state} in {dem_dir} and cannot be downloaded")
        try:
            url = settings.srtm_url_template.format(lat_band=tile[:3], lon_band=tile[3:], tile=tile)
        except (KeyError, IndexError, ValueError) as exc:
            raise ConfigError(f"elevation.srtm_url_template is invalid: {exc}") from exc
        LOGGER.info("Downloading SRTM tile %s from %s", tile, url)
        payload = get_bytes(
            url, timeout_s=settings.timeout_s, max_retries=settings.max_retries,
            backoff_s=settings.backoff_s, offline=is_offline(cfg),
        )
        hgt = _decode_tile(payload, tile)
        try:
            atomic_write_bytes(raw_path, payload)
        except OSError as exc:
            LOGGER.warning("Could not cache SRTM tile %s: %s", raw_path, exc)
    out = workdir / f"{tile}.hgt"  # the SRTMHGT driver georeferences from this file name
    out.write_bytes(hgt)
    return out


def _snap_bounds(bounds: tuple[float, float, float, float], transform: Any) -> tuple[float, float, float, float]:
    """Grow ``bounds`` outwards to the source pixel grid so merging copies pixels without resampling."""
    west, south, east, north = bounds
    res_x, res_y, x0, y0 = transform.a, -transform.e, transform.c, transform.f
    eps = 1e-9
    return (
        x0 + math.floor((west - x0) / res_x + eps) * res_x,
        y0 - math.ceil((y0 - south) / res_y - eps) * res_y,
        x0 + math.ceil((east - x0) / res_x - eps) * res_x,
        y0 - math.floor((y0 - north) / res_y + eps) * res_y,
    )


def _write_clip(hgt_files: list[Path], clip: tuple[float, float, float, float], out: Path) -> None:
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in hgt_files]
        mosaic, transform = merge(sources, bounds=_snap_bounds(clip, sources[0].transform), nodata=SRTM_NODATA)
    band = mosaic[0].astype(np.int16)
    if band.size == 0 or np.all(band == SRTM_NODATA):
        raise ElevationError(f"SRTM clip for bbox {clip} contains only voids")
    profile = {
        "driver": "GTiff", "width": band.shape[1], "height": band.shape[0], "count": 1, "dtype": "int16",
        "crs": "EPSG:4326", "transform": transform, "nodata": SRTM_NODATA, "compress": "deflate", "predictor": 2,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(band, 1)
        payload = memfile.read()
    atomic_write_bytes(out, payload)


def download_srtm(cfg: Mapping[str, Any], bbox: Sequence[float]) -> Path:
    """Build ``paths.dem_dir/srtm_<w>_<s>_<e>_<n>.tif`` from SRTM tiles (raw tiles cached as ``<tile>.hgt.gz``).

    Raises ``NetworkUnavailable`` (offline / download failure), :class:`ElevationError`
    (corrupt tile, all voids) or ``ValueError`` (bbox needs too many tiles).
    """
    settings = ElevationSettings.from_config(cfg)
    target = validate_bbox(bbox)
    tiles = srtm_tile_names(target, settings.margin_deg)
    if len(tiles) > settings.max_srtm_tiles:
        raise ValueError(
            f"bbox {target} needs {len(tiles)} SRTM tiles (> elevation.max_srtm_tiles={settings.max_srtm_tiles}); "
            "use a smaller region.bbox or provide a local DEM in paths.dem_dir"
        )
    dem_dir = resolve_path(cfg, "dem_dir", "data/raw/dem")
    out = dem_dir / (SRTM_CACHE_PREFIX + "_".join(f"{v:.4f}" for v in target) + ".tif")
    with tempfile.TemporaryDirectory(prefix="namma_flow_srtm_") as tmp:
        hgt_files = [_fetch_tile(cfg, settings, tile, dem_dir, Path(tmp)) for tile in tiles]
        _write_clip(hgt_files, expand_bbox(target, settings.margin_deg), out)
    LOGGER.info("Wrote SRTM clip %s from tile(s) %s", out, ", ".join(tiles))
    return out


def _srtm_dem(cfg: Mapping[str, Any], settings: ElevationSettings, bbox: tuple[float, ...]) -> Path | None:
    dem_dir = resolve_path(cfg, "dem_dir", "data/raw/dem")
    if not _refresh_requested(cfg, settings, "SRTM clip / tiles"):
        cached = find_local_dem(dem_dir, bbox, cached=True)
        if cached is not None:
            LOGGER.info("Using cached SRTM clip %s", cached)
            return cached
    try:
        return download_srtm(cfg, bbox)
    except ConfigError:
        raise
    except (ElevationError, *_SOURCE_ERRORS) as exc:
        LOGGER.warning("SRTM DEM unavailable: %s", exc)
        return None


def _dem_for(name: str, cfg: Mapping[str, Any], settings: ElevationSettings, bbox: tuple) -> Path | None:
    """DEM file for a raster source: ``local`` (user GeoTIFFs) or ``srtm`` (cache -> download)."""
    if name == "local":
        return find_local_dem(resolve_path(cfg, "dem_dir", "data/raw/dem"), bbox)
    return _srtm_dem(cfg, settings, bbox)


def acquire_dem(cfg: Mapping[str, Any], bbox: Sequence[float]) -> Path | None:
    """Local DEM covering ``bbox`` -> cached/downloaded SRTM clip -> ``None`` (honours ``elevation.sources``)."""
    settings = ElevationSettings.from_config(cfg)
    target = validate_bbox(bbox)
    for name in settings.sources:
        path = _dem_for(name, cfg, settings, target) if name in ("local", "srtm") else None
        if path is not None:
            return path
    return None


# --------------------------------------------------------------------------- sampling
def _read_grid(src: Any, window: Window, limits: tuple[float, float] | None) -> np.ndarray:
    data = np.ma.filled(src.read(1, window=window, masked=True).astype(np.float64), np.nan)
    data[~np.isfinite(data)] = np.nan
    return _apply_range(data, limits)


def _bilinear(src: Any, lon: np.ndarray, lat: np.ndarray, limits: tuple[float, float] | None) -> np.ndarray:
    """NaN-aware bilinear interpolation between pixel centres; edge half-pixels are clamped."""
    if src.crs is None:
        raise ElevationError("DEM has no CRS")
    if _is_wgs84(src.crs):
        xs, ys = lon, lat
    else:
        xs, ys = (np.asarray(v, dtype=np.float64) for v in warp_transform("EPSG:4326", src.crs, lon, lat))
    inv = ~src.transform  # world -> fractional (col, row), pixel corners at integers
    cols = inv.a * xs + inv.b * ys + inv.c
    rows = inv.d * xs + inv.e * ys + inv.f
    height, width = src.height, src.width
    inside = np.isfinite(cols) & np.isfinite(rows) & (cols >= 0) & (cols <= width) & (rows >= 0) & (rows <= height)
    out = np.full(lon.shape, np.nan)
    if not inside.any():
        return out
    c = np.clip(cols[inside] - 0.5, 0, width - 1)
    r = np.clip(rows[inside] - 0.5, 0, height - 1)
    c0, r0 = np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)
    c1, r1 = np.minimum(c0 + 1, width - 1), np.minimum(r0 + 1, height - 1)
    wx, wy = c - c0, r - r0
    col_off, row_off = int(c0.min()), int(r0.min())
    window = Window(col_off, row_off, int(c1.max()) - col_off + 1, int(r1.max()) - row_off + 1)
    grid = _read_grid(src, window, limits)
    num = np.zeros(c.shape)
    den = np.zeros(c.shape)
    corners = ((r0, c0, (1 - wx) * (1 - wy)), (r0, c1, wx * (1 - wy)), (r1, c0, (1 - wx) * wy), (r1, c1, wx * wy))
    for rr, cc, weight in corners:
        values = grid[rr - row_off, cc - col_off]
        valid = np.isfinite(values)
        num += np.where(valid, values * weight, 0.0)
        den += np.where(valid, weight, 0.0)
    sampled = np.full(c.shape, np.nan)
    good = den > 1e-12
    sampled[good] = num[good] / den[good]
    out[inside] = sampled
    return out


def sample_dem(
    dem_path: str | Path, lon: Any, lat: Any, *, valid_range: Sequence[float] | None = None
) -> np.ndarray:
    """Bilinear DEM samples at ``(lon, lat)`` (EPSG:4326). Voids -> NaN.

    Voids are nodata/masked pixels, non-finite values, pixels outside ``valid_range`` and
    points outside the raster. Only the window around the points is read.
    """
    lon, lat = validate_coords(lon, lat)
    limits = _check_range(valid_range)
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")
    if lon.size == 0:
        return np.zeros(0, dtype=np.float64)
    try:
        with rasterio.open(path) as src:
            return _bilinear(src, lon, lat, limits)
    except (RasterioError, CRSError) as exc:
        raise ElevationError(f"Could not read DEM {path}: {exc}") from exc


# --------------------------------------------------------------------------- other sources
def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _apply_range(values: np.ndarray, limits: tuple[float, float] | None) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if limits is not None:
        with np.errstate(invalid="ignore"):
            out[(out < limits[0]) | (out > limits[1])] = np.nan
    return out


def fetch_open_meteo_elevation(lon: Any, lat: Any, cfg: Mapping[str, Any]) -> np.ndarray:
    """Open-Meteo elevation API in batches of <= 100 points; out-of-range values -> NaN.

    Raises ``NetworkUnavailable`` offline, on HTTP failure or on a malformed response.
    """
    settings = ElevationSettings.from_config(cfg)
    lon, lat = validate_coords(lon, lat)
    if lon.size == 0:
        return np.zeros(0, dtype=np.float64)
    if is_offline(cfg):
        raise NetworkUnavailable("Offline mode: not calling the Open-Meteo elevation API")
    out = np.empty(lon.size, dtype=np.float64)
    for start in range(0, lon.size, settings.batch_size):
        stop = min(start + settings.batch_size, lon.size)
        params = {
            "latitude": ",".join(f"{v:.6f}" for v in lat[start:stop]),
            "longitude": ",".join(f"{v:.6f}" for v in lon[start:stop]),
        }
        payload = get_json(
            settings.open_meteo_url, params, timeout_s=settings.timeout_s,
            max_retries=settings.max_retries, backoff_s=settings.backoff_s, offline=False,
        )
        values = payload.get("elevation") if isinstance(payload, Mapping) else None
        if not isinstance(values, list) or len(values) != stop - start:
            raise NetworkUnavailable(
                f"Open-Meteo elevation API returned a malformed response (expected {stop - start} values): "
                f"{str(payload)[:200]}"
            )
        out[start:stop] = [_to_float(v) for v in values]
    return _apply_range(out, settings.valid_range)


def synthetic_elevation(lon: Any, lat: Any) -> np.ndarray:
    """Spec fallback valley topography: ``880 + 30 sin(lat*100) + 15 cos(lon*100)`` metres."""
    lon, lat = validate_coords(lon, lat)
    return 880.0 + 30.0 * np.sin(lat * 100.0) + 15.0 * np.cos(lon * 100.0)


def fill_voids(values: Any, lon: Any, lat: Any, k: int = 8) -> np.ndarray:
    """Replace NaN/inf by the inverse-distance-squared mean of the ``k`` nearest valid values (new array)."""
    lon, lat = validate_coords(lon, lat)
    out = np.atleast_1d(np.asarray(values, dtype=np.float64)).copy()
    if out.shape != lon.shape:
        raise ValueError(f"values, lon and lat must have the same shape, got {out.shape} and {lon.shape}")
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    void = ~np.isfinite(out)
    if not void.any():
        return out
    if void.all():
        raise ValueError("Cannot fill voids: there are no valid values to interpolate from")
    x, y = lonlat_to_local_xy(lon, lat)
    xy = np.column_stack([x, y])
    known = out[~void]
    k_eff = min(int(k), known.size)
    dist, idx = cKDTree(xy[~void]).query(xy[void], k=k_eff)
    dist = np.asarray(dist).reshape(-1, k_eff)
    idx = np.asarray(idx).reshape(-1, k_eff)
    weights = 1.0 / np.maximum(dist, 1e-6) ** 2
    out[void] = (weights * known[idx]).sum(axis=1) / weights.sum(axis=1)
    return out


# --------------------------------------------------------------------------- source chain
def _sample_source(
    name: str, lon: np.ndarray, lat: np.ndarray, cfg: Mapping[str, Any], settings: ElevationSettings, bbox: tuple
) -> tuple[np.ndarray | None, str | None]:
    if name == "synthetic":
        return synthetic_elevation(lon, lat), None
    if name == "open_meteo":
        return fetch_open_meteo_elevation(lon, lat, cfg), None
    path = _dem_for(name, cfg, settings, bbox)
    if path is None:
        LOGGER.info("Elevation source '%s': no DEM covering bbox %s", name, bbox)
        return None, None
    return sample_dem(path, lon, lat, valid_range=settings.valid_range), str(path)


def _try_source(
    name: str, lon: np.ndarray, lat: np.ndarray, cfg: Mapping[str, Any], settings: ElevationSettings, bbox: tuple
) -> ElevationResult | None:
    if name not in SOURCE_LABELS:
        LOGGER.warning("Unknown elevation source '%s' (expected one of %s); skipping", name, sorted(SOURCE_LABELS))
        return None
    try:
        values, dem_path = _sample_source(name, lon, lat, cfg, settings, bbox)
    except ConfigError:
        raise
    except (ElevationError, *_SOURCE_ERRORS) as exc:
        LOGGER.warning("Elevation source '%s' failed: %s", name, exc)
        return None
    if values is None:
        return None
    voids = ~np.isfinite(values)
    fraction = float(voids.mean())
    if voids.all() or fraction > settings.max_void_fraction:
        LOGGER.warning(
            "Rejecting elevation source '%s': %.1f%% of nodes are voids (limit %.0f%%)",
            name, 100 * fraction, 100 * settings.max_void_fraction,
        )
        return None
    if voids.any():
        LOGGER.info("Filled %d void node elevation(s) (%.1f%%) from neighbours", int(voids.sum()), 100 * fraction)
        values = fill_voids(values, lon, lat, k=settings.void_fill_k)
    label = SOURCE_LABELS[name]
    LOGGER.info("Elevation source: %s%s", label, f" ({dem_path})" if dem_path else "")
    return ElevationResult(values, label, int(voids.sum()), fraction, dem_path)


def estimate_elevations(
    lon: Any, lat: Any, cfg: Mapping[str, Any], bbox: Sequence[float] | None = None
) -> ElevationResult:
    """Run the ``elevation.sources`` chain; the synthetic formula is the last resort (never raises on data)."""
    settings = ElevationSettings.from_config(cfg)
    lon, lat = validate_coords(lon, lat)
    if lon.size == 0:
        raise ValueError("Elevation lookup needs at least one node")
    target = validate_bbox(bbox) if bbox is not None else nodes_bbox(lon, lat, cfg)
    for name in settings.sources:
        result = _try_source(name, lon, lat, cfg, settings, target)
        if result is not None:
            return result
    LOGGER.warning("Elevation source chain %s exhausted; using the synthetic valley formula", list(settings.sources))
    return ElevationResult(synthetic_elevation(lon, lat), SOURCE_LABELS["synthetic"], 0, 0.0, None)


def node_elevations(lon: Any, lat: Any, cfg: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Contract API: ``(elevation_m [N], source)``, source in ``local_dem|srtm|open_meteo|synthetic``."""
    result = estimate_elevations(lon, lat, cfg)
    return result.elevation, result.source


# --------------------------------------------------------------------------- graph helpers
def _length_or_inf(data: Mapping[str, Any]) -> float:
    value = _to_float(data.get("length"))
    return value if math.isfinite(value) and value > 0 else math.inf


def _as_digraph(G: Any) -> nx.DiGraph:
    """New ``DiGraph`` from any networkx graph (parallel edges -> shortest, undirected -> both ways).

    GraphML-reader artefacts (``node_default``/``edge_default`` dicts that ``nx.read_graphml``
    adds) are dropped: ``graph_io.save_graph`` would stringify them and the networkx writer
    then crashes calling ``.get()`` on the string, breaking any load -> save round trip.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError(f"Expected a networkx graph, got {type(G).__name__}")
    if not G.is_multigraph():
        H = nx.DiGraph(G)
        for key in _GRAPHML_READER_KEYS:
            H.graph.pop(key, None)
        return H
    H = nx.DiGraph()
    H.graph.update({k: v for k, v in G.graph.items() if k not in _GRAPHML_READER_KEYS})
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        pairs = ((u, v),) if G.is_directed() else ((u, v), (v, u))
        for a, b in pairs:
            if H.has_edge(a, b) and _length_or_inf(data) >= _length_or_inf(H.edges[a, b]):
                continue
            H.add_edge(a, b)
            H.edges[a, b].clear()
            H.edges[a, b].update(data)
    return H


def _check_coordinates(G: nx.DiGraph) -> None:
    for node, data in G.nodes(data=True):
        if "x" not in data or "y" not in data:
            raise GraphFormatError(f"Node {node!r} has no x/y coordinates")
        x, y = _to_float(data["x"]), _to_float(data["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise GraphFormatError(f"Node {node!r} needs finite x/y coordinates, got ({data['x']!r}, {data['y']!r})")


def _min_edge_length(cfg: Mapping[str, Any]) -> float:
    value = _to_float(get_section(cfg, "network", {"min_edge_length_m": 1.0}).get("min_edge_length_m"))
    if not (math.isfinite(value) and value > 0):
        raise ConfigError(f"network.min_edge_length_m must be > 0, got {value}")
    return value


def compute_edge_grades(G: nx.DiGraph, cfg: Mapping[str, Any]) -> nx.DiGraph:
    """Return a NEW graph with ``grade = (elev_v - elev_u) / length`` clipped to ``+-max_abs_grade`` on every edge.

    Missing/invalid lengths are replaced by the haversine distance (>= network.min_edge_length_m).
    """
    settings = ElevationSettings.from_config(cfg)
    min_length = _min_edge_length(cfg)
    H = _as_digraph(G)
    missing = [n for n, d in H.nodes(data=True) if not math.isfinite(_to_float(d.get("elevation")))]
    if missing:
        raise ValueError(f"{len(missing)} node(s) lack a finite 'elevation' (e.g. {missing[:3]}); run node_elevations")
    repaired = clipped = 0
    for u, v, data in H.edges(data=True):
        nu, nv = H.nodes[u], H.nodes[v]
        length = _length_or_inf(data)
        if not math.isfinite(length):
            _check_coordinates(H.subgraph((u, v)))
            length = float(haversine_m(_to_float(nu["x"]), _to_float(nu["y"]), _to_float(nv["x"]), _to_float(nv["y"])))
            repaired += 1
        length = max(length, min_length)
        raw = (_to_float(nv["elevation"]) - _to_float(nu["elevation"])) / length
        grade = float(np.clip(raw, -settings.max_abs_grade, settings.max_abs_grade))
        clipped += int(grade != raw)
        data["length"] = float(length)
        data["grade"] = grade
    if repaired:
        LOGGER.warning("%d edge(s) had a missing/invalid length; replaced by the haversine distance", repaired)
    if clipped:
        LOGGER.info("%d edge grade(s) clipped to +-%.2f", clipped, settings.max_abs_grade)
    return H


# --------------------------------------------------------------------------- enrichment / stage
def enrich_graph(G: nx.DiGraph, cfg: Mapping[str, Any]) -> nx.DiGraph:
    """Return a NEW DiGraph with elevation, relative_elevation, dist_to_drain_m, flow_accumulation,
    is_sink on nodes, grade (and repaired length) on edges, and provenance graph attributes."""
    H = _as_digraph(G)
    if H.number_of_nodes() == 0:
        raise GraphFormatError("Graph has no nodes; nothing to enrich")
    _check_coordinates(H)
    settings = ElevationSettings.from_config(cfg)
    arrays = graph_to_arrays(H)
    bbox = nodes_bbox(arrays.lon, arrays.lat, cfg)
    elevation = estimate_elevations(arrays.lon, arrays.lat, cfg, bbox)
    tpi = relative_elevation(arrays.lon, arrays.lat, elevation.elevation, settings.tpi_radius_m)
    drains = compute_drain_distances(arrays.lon, arrays.lat, cfg)
    for i, node in enumerate(arrays.node_ids):
        H.nodes[node].update(
            elevation=float(elevation.elevation[i]),
            relative_elevation=float(tpi[i]),
            dist_to_drain_m=float(drains.distance_m[i]),
        )
    H = compute_edge_grades(H, cfg)
    accumulation, sinks = flow_accumulation(H)
    for i, node in enumerate(arrays.node_ids):
        H.nodes[node].update(flow_accumulation=float(accumulation[i]), is_sink=bool(sinks[i]))
    H.graph.update(
        elevation_source=elevation.source,
        elevation_dem=Path(elevation.dem_path).name if elevation.dem_path else "",
        elevation_voids_filled=int(elevation.n_voids),
        elevation_void_fraction=float(round(elevation.void_fraction, 6)),
        drain_source=drains.source,
        drain_feature_count=int(drains.n_features),
        enriched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{ENRICHMENT_HASH_ATTR: enrichment_config_hash(cfg)},
    )
    LOGGER.info(
        "Enriched %d nodes / %d edges: elevation=%s, drains=%s, %d sinks",
        H.number_of_nodes(), H.number_of_edges(), elevation.source, drains.source, int(sinks.sum()),
    )
    return H


def run_elevation_stage(cfg: Mapping[str, Any], allow_synthetic: bool = False) -> tuple[nx.DiGraph, Path]:
    """Stage 02: load ``paths.graph_file``, enrich it and save it back atomically; returns ``(graph, path)``.

    Raises :class:`ElevationError` (and saves nothing) when the new enrichment would replace
    real elevations (local DEM / SRTM / Open-Meteo) or real OSM drains with a synthetic
    fallback - e.g. offline with the DEM cache deleted - unless ``allow_synthetic``.
    """
    path = resolve_path(cfg, "graph_file", DEFAULT_GRAPH_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Road graph not found at {path}: run python src/data_pipeline/01_extract_network.py first"
        )
    original = load_graph(path)
    enriched = enrich_graph(original, cfg)
    lost = synthetic_downgrades(original, enriched)
    if lost and not allow_synthetic:
        raise ElevationError(
            f"Refusing to overwrite {path}: the new enrichment would degrade real inputs to synthetic "
            f"fallbacks ({'; '.join(lost)}; see the WARNINGs above). The graph was NOT saved. Restore the "
            "DEM / waterway caches or run online, or pass --allow-synthetic to accept the fallback"
        )
    if lost:
        LOGGER.warning("Overwriting %s with synthetic fallbacks (%s) because allow_synthetic is set", path, "; ".join(lost))
    save_graph(enriched, path)
    return enriched, path
