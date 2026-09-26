"""Stage 01 — road-network extraction, cleaning and persistence.

Junctions become graph nodes and drivable street segments directed edges along
which the GNN passes runoff messages. ``paths.graph_file`` (contract schema 2.1) is
produced by a fallback chain that always yields a usable graph: cached file →
OSM place query (rejected if the geocoded polygon is too large, outside
``region.bbox`` or street-less — "Bellandur, …" really geocodes to the *lake*) →
OSM ``region.bbox`` → synthetic street grid (OSM unreachable). Offline, the bbox query is
replayed from the osmnx response cache (``paths.osm_cache_dir``) without touching the
network, so ``--force --offline`` rebuilds the real graph; a real OSM graph on disk is never
replaced by the synthetic grid unless ``allow_synthetic`` is set, while a cached synthetic
grid is rebuilt automatically when online. The cleaned graph is enriched by module B
(``src.data_pipeline.elevation.enrich_graph``) and saved atomically; a graph that is not fully
enriched is never written, and no (re-)enrichment may replace real SRTM / OSM-drain inputs with
synthetic fallbacks (:mod:`src.data_pipeline.enrichment`). A cached graph enriched with other
elevation / drains settings is re-enriched. ``persist=False`` (diagnostics) never writes.

Every Overpass request is bounded (:mod:`src.data_pipeline.overpass_guard`): a mirror that
keeps answering HTTP 429 / 504 is given up after ``network.overpass_max_attempts`` and the next
``network.overpass_urls`` mirror is tried. The OpenStreetMap snapshot the graph was built from
(Overpass ``timestamp_osm_base``) is stored as the graph attribute ``osm_base_utc``;
``network.osm_date`` pins the queries to a past snapshot. Settings and cleaning live in
:mod:`src.data_pipeline.network_settings` / :mod:`src.data_pipeline.network_clean`.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import networkx as nx
import numpy as np

from src.data_pipeline.graph_io import GraphFormatError, load_graph, save_graph
from src.data_pipeline.network_clean import (  # noqa: F401 - re-exported public API
    DEFAULT_HIGHWAY,
    LARGE_COMPONENT_LOSS,
    LIST_SEPARATOR,
    WGS84_CRS,
    _GRAPHML_READER_ATTRS,
    GraphTooSmallError,
    _flatten_value,
    _is_finite_value,
    _parse_oneway,
    _resolve_length,
    _valid_lengths,
    clean_graph,
)
from src.data_pipeline.network_settings import (  # noqa: F401 - re-exported public API
    NETWORK_DEFAULTS,
    OSMNX_NETWORK_TYPES,
    REGION_DEFAULTS,
    NetworkSettings,
    RegionSettings,
    _to_float,
    network_config_hash,
    overpass_query_settings,
)
from src.data_pipeline.overpass_guard import OverpassLog, bounded_osmnx_requests
from src.utils.config import is_offline, resolve_path
from src.utils.geo import UTM_43N_EPSG, bbox_area_km2, haversine_m
from src.utils.http import USER_AGENT, NetworkUnavailable
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_GRAPH_FILE = "data/interim/bellandur_osm.graphml"
DEFAULT_OSM_CACHE_DIR = "data/interim/osm_cache"

SOURCE_PLACE, SOURCE_BBOX, SOURCE_SYNTHETIC = "osm_place", "osm_bbox", "synthetic_grid"
OSM_SOURCES = frozenset({SOURCE_PLACE, SOURCE_BBOX})
ENRICHED_NODE_ATTRS = ("elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink")
ENRICHED_EDGE_ATTRS = ("grade",)
OSM_SNAPSHOT_ATTRS = ("osm_base_utc", "osm_query_utc", "osm_date", "overpass_endpoint")
STAGE_GRAPH_ATTRS = ("crs", "source", "bbox", "created_utc", "network_config_hash", *OSM_SNAPSHOT_ATTRS)
LARGE_BBOX_KM2 = 250.0        # warn before asking Overpass for more than this
# Nominatim feature classes/types that are water bodies, never a locality with streets.
_WATER_CLASSES = frozenset({"water", "waterway"})
_WATER_TYPES = frozenset({"water", "lake", "reservoir", "river", "pond", "wetland", "basin", "canal", "stream"})


class NetworkStageError(RuntimeError):
    """Stage 01 cannot complete (e.g. the elevation module is unavailable or misbehaves)."""


class _PlaceRejected(Exception):
    """Internal: the place-name query produced an unusable result; fall back to the bbox."""


class OsmCacheMiss(NetworkUnavailable):
    """Offline replay: a request is not in the osmnx response cache (nothing was downloaded)."""


# --------------------------------------------------------------------------- synthetic fallback
def synthetic_grid_graph(cfg: Mapping[str, Any]) -> nx.DiGraph:
    """Offline fallback: a regular two-way street grid (rows × cols junctions) over ``region.bbox``.

    Junctions sit at cell centres (strictly inside the bbox, so raster sampling never
    hits the edge). Deterministic: the same config always yields the same graph.
    """
    region = RegionSettings.from_config(cfg)
    net = NetworkSettings.from_config(cfg)
    rows, cols = net.grid_rows, net.grid_cols
    if rows * cols < net.min_nodes:
        raise GraphTooSmallError(
            f"network.synthetic_grid {rows}x{cols} has {rows * cols} junctions, fewer than "
            f"network.min_nodes={net.min_nodes}; enlarge the grid"
        )
    west, south, east, north = region.bbox
    lons = west + (np.arange(cols) + 0.5) * (east - west) / cols
    lats = south + (np.arange(rows) + 0.5) * (north - south) / rows
    node_id = lambda r, c: r * cols + c  # noqa: E731
    G = nx.DiGraph(crs=WGS84_CRS, source=SOURCE_SYNTHETIC, bbox=json.dumps(list(region.bbox)))
    for r in range(rows):
        for c in range(cols):
            degree = (r > 0) + (r < rows - 1) + (c > 0) + (c < cols - 1)
            G.add_node(node_id(r, c), x=float(lons[c]), y=float(lats[r]), street_count=int(degree))
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                _add_grid_street(G, node_id(r, c), node_id(r, c + 1), f"Synthetic Street {r + 1}", r, net)
            if r + 1 < rows:
                _add_grid_street(G, node_id(r, c), node_id(r + 1, c), f"Synthetic Avenue {c + 1}", c, net)
    LOGGER.info("Built synthetic %dx%d street grid: %d nodes, %d edges",
                rows, cols, G.number_of_nodes(), G.number_of_edges())
    return G


def _add_grid_street(G: nx.DiGraph, u: int, v: int, name: str, line: int, net: NetworkSettings) -> None:
    du, dv = G.nodes[u], G.nodes[v]
    length = max(float(haversine_m(du["x"], du["y"], dv["x"], dv["y"])), net.min_edge_length_m)
    highway = "secondary" if line % 5 == 0 else "residential"  # an arterial every fifth line
    attrs = {"length": length, "highway": highway, "oneway": False, "reversed_added": False, "name": name}
    G.add_edge(u, v, **attrs)
    G.add_edge(v, u, **attrs)


# --------------------------------------------------------------------------- OpenStreetMap
def _import_osmnx() -> Any:
    """Import osmnx lazily (heavy import; also the seam tests use to inject a fake)."""
    import osmnx

    return osmnx


@contextmanager
def _osmnx_settings(ox: Any, cache_dir: Path, net: NetworkSettings) -> Iterator[OverpassLog]:
    """Temporarily point osmnx at the project cache/timeout/endpoint, restoring global settings afterwards.

    osmnx embeds the timeout in the Overpass query text, i.e. in its cache key, so an
    integral timeout is passed as ``int`` (osmnx's own default formatting) to share caches.
    Inside, busy-server retries are bounded (:func:`bounded_osmnx_requests`); yields its log.
    """
    timeout = int(net.request_timeout_s) if float(net.request_timeout_s).is_integer() else net.request_timeout_s
    overrides = {"cache_folder": str(cache_dir), "use_cache": True, "requests_timeout": timeout,
                 "log_console": False, "http_user_agent": USER_AGENT, "overpass_url": net.overpass_urls[0]}
    if net.osm_date is not None:  # attic query; only then, so default queries keep their cache keys
        overrides["overpass_settings"] = overpass_query_settings(net.osm_date)
    saved = {key: getattr(ox.settings, key) for key in overrides if hasattr(ox.settings, key)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key in saved:
        setattr(ox.settings, key, overrides[key])
    try:
        with bounded_osmnx_requests(ox, net.retry) as log:
            yield log
    finally:
        for key, value in saved.items():
            setattr(ox.settings, key, value)


def _stamp_snapshot(G: nx.MultiDiGraph, log: OverpassLog, net: NetworkSettings) -> None:
    """Record the OSM snapshot behind ``G`` (F5-07): Overpass ``timestamp_osm_base``, query time, pin.

    ``osm_query_utc`` is set only when Overpass was actually queried (not answered from the cache).
    """
    G.graph["osm_base_utc"] = log.latest_osm_base or "unknown"
    if log.requests_sent:
        G.graph["osm_query_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if net.osm_date is not None:
        G.graph["osm_date"] = net.osm_date


def _place_polygon(gdf: Any) -> tuple[Any, float]:
    """Union the geocoded geometries and return ``(polygon, area_km2)``."""
    import geopandas as gpd

    if gdf is None or len(gdf) == 0:
        raise _PlaceRejected("the geocoder returned no result")
    geometry = gdf.geometry.union_all()
    if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise _PlaceRejected(f"the geocoder returned a {geometry.geom_type}, not a polygon")
    series = gpd.GeoSeries([geometry], crs=gdf.crs or "EPSG:4326")
    try:
        metric = series.estimate_utm_crs()
    except (RuntimeError, ValueError):  # pragma: no cover - pyproj database lookup failure
        metric = UTM_43N_EPSG
    return series.to_crs(4326).iloc[0], float(series.to_crs(metric).area.iloc[0] / 1e6)


def _fetch_place_graph(ox: Any, region: RegionSettings, net: NetworkSettings) -> nx.MultiDiGraph:
    """Try ``graph_from_place``; raise :class:`_PlaceRejected` with the reason on any problem."""
    from shapely.geometry import box

    try:
        gdf = ox.geocode_to_gdf(region.place)
    except Exception as exc:  # noqa: BLE001 - osmnx/Nominatim raise many unrelated types
        raise _PlaceRejected(f"geocoding failed: {exc}") from exc
    polygon, area_km2 = _place_polygon(gdf)
    first = gdf.iloc[0]
    kind = (str(first.get("class", "")).lower(), str(first.get("type", "")).lower())
    if kind[0] in _WATER_CLASSES or kind[1] in _WATER_TYPES:
        raise _PlaceRejected(f"the geocoder matched a water body ({kind[0]}={kind[1]}: "
                             f"{first.get('name') or first.get('display_name')!s}), not a locality with streets")
    if area_km2 > region.max_place_area_km2:
        raise _PlaceRejected(
            f"geocoded polygon covers {area_km2:.1f} km2 > region.max_place_area_km2={region.max_place_area_km2:g}"
        )
    if not polygon.intersects(box(*region.bbox)):
        bounds = tuple(round(b, 4) for b in polygon.bounds)
        raise _PlaceRejected(f"geocoded polygon {bounds} does not intersect region.bbox")
    try:
        G = ox.graph_from_place(region.place, network_type=net.network_type, simplify=net.simplify)
    except Exception as exc:  # noqa: BLE001 - e.g. "Found no graph nodes within the requested polygon"
        raise _PlaceRejected(f"osmnx could not build a graph: {exc}") from exc
    if G is None or G.number_of_nodes() < region.min_place_nodes:
        count = 0 if G is None else G.number_of_nodes()
        raise _PlaceRejected(f"graph has {count} node(s) < region.min_place_nodes={region.min_place_nodes}")
    G.graph["bbox"] = json.dumps([round(float(b), 6) for b in polygon.bounds])
    return G


def _fetch_bbox_graph(ox: Any, region: RegionSettings, net: NetworkSettings) -> nx.MultiDiGraph:
    """Download ``region.bbox``, trying each ``network.overpass_urls`` endpoint in turn."""
    area = bbox_area_km2(region.bbox)
    if area > LARGE_BBOX_KM2:
        LOGGER.warning("region.bbox covers %.0f km2 (large); the Overpass download and the graph will be big", area)
    errors: list[str] = []
    for index, url in enumerate(net.overpass_urls, start=1):
        ox.settings.overpass_url = url  # restored by _osmnx_settings
        LOGGER.info("Overpass mirror %d of %d: %s (road network of region.bbox)", index, len(net.overpass_urls), url)
        try:
            G = ox.graph_from_bbox(bbox=region.bbox, network_type=net.network_type, simplify=net.simplify)
        except Exception as exc:  # noqa: BLE001 - HTTP, Overpass and empty-response errors alike
            LOGGER.warning("Overpass endpoint %s failed for region.bbox: %s", url, exc)
            errors.append(f"{url}: {exc}")
            continue
        if G is None or G.number_of_nodes() == 0:
            raise NetworkUnavailable(f"OSM returned an empty road graph for region.bbox {list(region.bbox)}")
        G.graph["bbox"] = json.dumps(list(region.bbox))
        G.graph["overpass_endpoint"] = url
        return G
    raise NetworkUnavailable(f"OSM download for region.bbox {list(region.bbox)} failed on every Overpass "
                             f"endpoint (network.overpass_urls): {' | '.join(errors)}")


def fetch_osm_graph(cfg: Mapping[str, Any]) -> tuple[nx.MultiDiGraph, str]:
    """Download the raw OSM street network: place query first (if enabled), then ``region.bbox``.

    Returns ``(raw MultiDiGraph, source)`` with ``source`` in ``{"osm_place", "osm_bbox"}``;
    the graph's ``bbox`` attribute holds the queried extent as a JSON list. Raises
    :class:`NetworkUnavailable` when offline, when osmnx is missing, or when the
    bbox download fails.
    """
    region = RegionSettings.from_config(cfg)
    net = NetworkSettings.from_config(cfg)
    if is_offline(cfg):
        raise NetworkUnavailable("Offline mode: not downloading the OSM road network")
    try:
        ox = _import_osmnx()
    except ImportError as exc:
        raise NetworkUnavailable(f"osmnx is not importable ({exc}); install requirements.txt") from exc
    cache_dir = resolve_path(cfg, "osm_cache_dir", DEFAULT_OSM_CACHE_DIR)
    with _osmnx_settings(ox, cache_dir, net) as log:
        if region.use_place_query and region.place:
            try:
                G = _fetch_place_graph(ox, region, net)
                LOGGER.info("OSM place query %r returned %d nodes", region.place, G.number_of_nodes())
                _stamp_snapshot(G, log, net)
                return G, SOURCE_PLACE
            except _PlaceRejected as exc:
                LOGGER.warning("OSM place query %r rejected (%s); falling back to region.bbox", region.place, exc)
        G = _fetch_bbox_graph(ox, region, net)
        _stamp_snapshot(G, log, net)
        LOGGER.info("OSM bbox query returned %d nodes, %d edges (OSM snapshot %s)", G.number_of_nodes(),
                    G.number_of_edges(), G.graph["osm_base_utc"])
        return G, SOURCE_BBOX


@contextmanager
def _cache_only(ox: Any) -> Iterator[None]:
    """Let osmnx answer from its response cache only; the first uncached request raises :class:`OsmCacheMiss`.

    osmnx >= 2.1.1 (the requirements.txt floor) looks an Overpass request up in
    ``settings.cache_folder`` before touching the network and calls ``_http._config_dns`` (a DNS
    lookup) only on a miss, so replacing that hook stops a miss before any network access
    (2.0.x / 2.1.0 call ``_config_dns`` first, so they cannot replay the cache offline).
    """
    http = getattr(ox, "_http", None)
    if http is None or not callable(getattr(http, "_config_dns", None)):
        raise OsmCacheMiss("this osmnx version has no cache-only hook (osmnx._http._config_dns); cannot replay offline")
    original = http._config_dns

    def _refuse(url: str) -> None:
        raise OsmCacheMiss(f"offline: the request to {url} is not in the osmnx cache")

    http._config_dns = _refuse
    try:
        yield
    finally:
        http._config_dns = original


def fetch_cached_osm_graph(cfg: Mapping[str, Any]) -> tuple[nx.MultiDiGraph, str]:
    """Rebuild the raw ``region.bbox`` street network from the osmnx response cache, never the network.

    Returns ``(raw MultiDiGraph, "osm_bbox")``. Raises :class:`OsmCacheMiss` (a
    ``NetworkUnavailable``) when the cache holds no matching Overpass response. The place
    query is not replayed: the geocoder is never contacted offline.
    """
    region = RegionSettings.from_config(cfg)
    net = NetworkSettings.from_config(cfg)
    cache_dir = resolve_path(cfg, "osm_cache_dir", DEFAULT_OSM_CACHE_DIR)
    if not cache_dir.is_dir() or not any(cache_dir.glob("*.json")):
        raise OsmCacheMiss(f"no cached Overpass responses in {cache_dir}")
    try:
        ox = _import_osmnx()
    except ImportError as exc:
        raise OsmCacheMiss(f"osmnx is not importable ({exc}); install requirements.txt") from exc
    with _osmnx_settings(ox, cache_dir, net) as log, _cache_only(ox):
        G = _fetch_bbox_graph(ox, region, net)
        _stamp_snapshot(G, log, net)
    G.graph.pop("overpass_endpoint", None)  # replayed from the cache, not queried
    LOGGER.info("Rebuilt the OSM bbox graph from the osmnx cache %s (%d nodes, %d edges; OSM snapshot %s)",
                cache_dir, G.number_of_nodes(), G.number_of_edges(), G.graph["osm_base_utc"])
    return G, SOURCE_BBOX


# --------------------------------------------------------------------------- stage orchestration
def missing_enrichment(G: nx.DiGraph) -> list[str]:
    """Names of enrichment attributes (schema 2.1) absent or non-finite on any node/edge."""
    missing = []
    for attr in ENRICHED_NODE_ATTRS:
        if G.number_of_nodes() == 0 or not all(_is_finite_value(d.get(attr)) for _, d in G.nodes(data=True)):
            missing.append(attr)
    for attr in ENRICHED_EDGE_ATTRS:
        if not all(_is_finite_value(d.get(attr)) for _, _, d in G.edges(data=True)):
            missing.append(attr)
    return missing


def _load_enricher() -> Callable[[nx.DiGraph, Mapping[str, Any]], nx.DiGraph]:
    """Import module B's ``enrich_graph`` lazily, failing with an actionable message."""
    try:
        from src.data_pipeline.elevation import enrich_graph
    except Exception as exc:  # noqa: BLE001 - ImportError, or any error raised while importing module B
        raise NetworkStageError(
            f"src.data_pipeline.elevation.enrich_graph is unavailable ({type(exc).__name__}: {exc}); "
            "the road graph was NOT saved. Fix the elevation module, then re-run 01_extract_network.py"
        ) from exc
    if not callable(enrich_graph):
        raise NetworkStageError("src.data_pipeline.elevation.enrich_graph is not callable")
    return enrich_graph


def _cache_problem(G: nx.DiGraph, net: NetworkSettings) -> str | None:
    if G.number_of_nodes() < net.min_nodes:
        return f"{G.number_of_nodes()} nodes < network.min_nodes={net.min_nodes}"
    if G.number_of_edges() == 0:
        return "no edges"
    bad = sum(1 for *_, d in G.edges(data=True) if not (_to_float(d.get("length")) > 0))
    return f"{bad} edge(s) without a positive length" if bad else None


def _load_cached_graph(path: Path, net: NetworkSettings) -> nx.DiGraph | None:
    if not path.exists():
        return None
    try:
        G = load_graph(path)
    except (GraphFormatError, OSError) as exc:
        LOGGER.warning("Cached graph %s could not be read (%s); rebuilding it", path, exc)
        return None
    problem = _cache_problem(G, net)
    if problem:
        LOGGER.warning("Cached graph %s is unusable (%s); rebuilding it", path, problem)
        return None
    return G


def _warn_if_stale(G: nx.DiGraph, path: Path, cfg: Mapping[str, Any]) -> None:
    stored = G.graph.get("network_config_hash")
    if stored and stored != network_config_hash(cfg):
        LOGGER.warning("Cached graph %s was built with different region/network settings; reusing it as-is. "
                       "Re-run 01_extract_network.py --force to rebuild it", path)


def _rebuild_synthetic_online(G: nx.DiGraph, path: Path, cfg: Mapping[str, Any]) -> bool:
    """True (with a WARNING) when the cached graph is the synthetic fallback grid and we are online.

    A synthetic grid protects no real data, so it is replaced by OpenStreetMap automatically
    (F5-08: an offline or diagnostics run must not pin a fresh clone to the synthetic grid).
    """
    if G.graph.get("source") != SOURCE_SYNTHETIC or is_offline(cfg):
        return False
    LOGGER.warning("Cached graph %s is the synthetic offline fallback grid, not real OSM streets; online, so "
                   "rebuilding it from OpenStreetMap (it is kept only if OSM is still unavailable)", path)
    return True


def _build_fresh_graph(
    cfg: Mapping[str, Any], net: NetworkSettings, path: Path, protected: str | None
) -> tuple[nx.DiGraph, str]:
    """OSM (online: place → bbox; offline: bbox replayed from the osmnx cache) → synthetic grid.

    ``protected`` is the source of a real OSM graph already at ``path``: then the synthetic
    fallback raises :class:`NetworkStageError` instead of silently replacing it.
    """
    offline = is_offline(cfg)
    try:
        raw, source = fetch_cached_osm_graph(cfg) if offline else fetch_osm_graph(cfg)
        return clean_graph(raw, cfg), source
    except NetworkUnavailable as exc:
        what = "Offline mode: the cached OSM responses are unusable" if offline else "OSM road network unavailable"
        reason = f"{what} ({exc})"
    except GraphTooSmallError as exc:
        reason = f"OSM road network unusable ({exc})"
    if protected:
        raise NetworkStageError(
            f"Refusing to replace the OpenStreetMap road graph {path} (source={protected}) with the synthetic "
            f"grid: {reason}. The graph was NOT changed. Re-run online, drop --force to keep the existing "
            "graph, or pass --allow-synthetic to replace it anyway"
        )
    LOGGER.warning("%s; falling back to a synthetic %dx%d street grid over region.bbox",
                   reason, net.grid_rows, net.grid_cols)
    return synthetic_grid_graph(cfg), SOURCE_SYNTHETIC


def _round_trip(G: nx.DiGraph) -> nx.DiGraph:
    """``G`` exactly as :func:`load_graph` would return it after saving (without touching the project paths)."""
    with tempfile.TemporaryDirectory(prefix="namma_flow_graph_") as tmp:
        target = Path(tmp) / "graph.graphml"
        save_graph(G, target)
        return load_graph(target)


def _enrich(G: nx.DiGraph, cfg: Mapping[str, Any], missing: list[str] | None, drift: str | None) -> nx.DiGraph:
    """Derived attributes only when that suffices (keeps stored SRTM / OSM values), else module B."""
    from src.data_pipeline.enrichment import can_recompute_derived, recompute_derived  # lazy: geo stack

    if missing and can_recompute_derived(G, missing, drift):
        return recompute_derived(G, cfg)
    enrich_graph = _load_enricher()
    enriched = enrich_graph(G, cfg)
    if not isinstance(enriched, nx.DiGraph) or enriched.is_multigraph():
        raise NetworkStageError(f"enrich_graph must return a networkx.DiGraph, got {type(enriched).__name__}")
    return enriched


def _enrich_and_save(
    G: nx.DiGraph, cfg: Mapping[str, Any], path: Path, *, baseline: nx.DiGraph | None = None,
    missing: list[str] | None = None, drift: str | None = None, allow_synthetic: bool = False, persist: bool = True,
) -> nx.DiGraph:
    """Enrich, verify completeness, refuse synthetic downgrades of ``baseline`` (the graph on disk),
    save atomically (``persist``) and return the graph as saved (or as it would be saved)."""
    from src.data_pipeline.enrichment import check_downgrade  # lazy: geo stack

    enriched = _enrich(G, cfg, missing, drift)
    still = missing_enrichment(enriched)
    if still:
        raise NetworkStageError(f"enrich_graph returned a graph without {still}; refusing to save a "
                                f"half-enriched graph to {path}")
    check_downgrade(baseline, enriched, allow_synthetic=allow_synthetic, target=f"the road graph {path}",
                    error=NetworkStageError, hint="Restore the DEM / waterway caches or run online, or pass "
                    "--allow-synthetic to accept the fallback")
    final = enriched.copy()
    final.graph.update({key: G.graph[key] for key in STAGE_GRAPH_ATTRS if key in G.graph})
    for key in _GRAPHML_READER_ATTRS:
        final.graph.pop(key, None)
    final.graph.setdefault("crs", WGS84_CRS)  # a foreign cached graph may lack it; x/y are lon/lat by contract
    for key in ("elevation_source", "drain_source"):
        if not final.graph.get(key):
            LOGGER.warning("enrich_graph did not set graph attribute %r; recording 'unknown'", key)
            final.graph[key] = "unknown"
    if not persist:
        LOGGER.info("Using an in-memory road graph (%d nodes, %d edges); %s was not written", final.number_of_nodes(),
                    final.number_of_edges(), path)
        return _round_trip(final)
    save_graph(final, path)
    LOGGER.info("Saved road graph to %s (%d nodes, %d edges)", path, final.number_of_nodes(), final.number_of_edges())
    return load_graph(path)


def _reuse_cached(cached: nx.DiGraph, cfg: Mapping[str, Any], path: Path, *, allow_synthetic: bool,
                  persist: bool) -> nx.DiGraph:
    """The cached graph as is, or re-enriched when it lacks attributes or its enrichment settings changed."""
    from src.data_pipeline.elevation_settings import enrichment_drift  # lazy: geo stack

    missing = missing_enrichment(cached)
    drift = enrichment_drift(cached, cfg)
    if not missing and not drift:
        LOGGER.info("Using cached road graph %s (%d nodes, %d edges)", path,
                    cached.number_of_nodes(), cached.number_of_edges())
        return cached
    if drift:
        LOGGER.warning("Cached road graph %s: %s; re-enriching it", path, drift)
    else:
        LOGGER.info("Cached road graph %s lacks %s; re-enriching it", path, missing)
    return _enrich_and_save(cached, cfg, path, baseline=cached, missing=missing, drift=drift,
                            allow_synthetic=allow_synthetic, persist=persist)


def extract_network(
    cfg: Mapping[str, Any], force: bool = False, allow_synthetic: bool = False, *, persist: bool = True
) -> nx.DiGraph:
    """Full stage 01: cache → OSM (place → bbox) → synthetic grid; enrich; save ``paths.graph_file``.

    Offline the OSM bbox graph is rebuilt from the osmnx response cache. A readable OSM graph
    already on disk is never replaced by the synthetic grid (``force`` or not) unless
    ``allow_synthetic``; a cached synthetic grid is rebuilt when online. A cached graph that
    lacks enrichment attributes, or whose ``enrichment_config_hash`` differs from the current
    elevation / drains / region settings, is re-enriched; no enrichment may replace real
    elevation / drain inputs of the graph on disk with synthetic fallbacks unless
    ``allow_synthetic``. ``persist=False`` (diagnostics, e.g. calibrate) never writes the graph
    file. Returns the graph exactly as it was (or would be) written to disk. Raises
    :class:`ConfigError` for invalid settings and :class:`NetworkStageError` when the graph
    cannot be enriched or would be degraded (nothing is saved).
    """
    region = RegionSettings.from_config(cfg)
    net = NetworkSettings.from_config(cfg)
    path = resolve_path(cfg, "graph_file", DEFAULT_GRAPH_FILE)
    cached = _load_cached_graph(path, net)
    if cached is not None and not force and not _rebuild_synthetic_online(cached, path, cfg):
        _warn_if_stale(cached, path, cfg)
        return _reuse_cached(cached, cfg, path, allow_synthetic=allow_synthetic, persist=persist)
    existing = None if cached is None else cached.graph.get("source")
    protected = str(existing) if existing in OSM_SOURCES and not allow_synthetic else None
    G, source = _build_fresh_graph(cfg, net, path, protected)
    fresh = G.copy()
    fresh.graph.update(crs=WGS84_CRS, source=source, bbox=G.graph.get("bbox") or json.dumps(list(region.bbox)),
                       created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       network_config_hash=network_config_hash(cfg))
    return _enrich_and_save(fresh, cfg, path, baseline=cached, allow_synthetic=allow_synthetic, persist=persist)


# --------------------------------------------------------------------------- summary (used by the CLI)
def summarize_graph(G: nx.DiGraph, path: Path) -> dict[str, Any]:
    """Headline statistics of a stage-01 graph (used by the CLI summary)."""
    lengths = np.array([_to_float(d.get("length")) for *_, d in G.edges(data=True)], dtype=np.float64)
    summary: dict[str, Any] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_weakly_connected_components(G) if G.number_of_nodes() else 0,
        "source": G.graph.get("source", "unknown"),
        "elevation_source": G.graph.get("elevation_source", "unknown"),
        "drain_source": G.graph.get("drain_source", "unknown"),
        "bbox": G.graph.get("bbox"),
        "created_utc": G.graph.get("created_utc", "unknown"),
        "osm_base_utc": (G.graph.get("osm_base_utc") or "not recorded (graph predates snapshot recording)")
        if G.graph.get("source") in OSM_SOURCES else None,
        "osm_query_utc": G.graph.get("osm_query_utc"),
        "osm_date": G.graph.get("osm_date"),
        "mean_edge_length_m": float(np.nanmean(lengths)) if lengths.size else None,
        "reverse_edges_added": sum(1 for *_, d in G.edges(data=True) if d.get("reversed_added") is True),
        "path": str(path),
    }
    elevations = np.array([_to_float(d.get("elevation")) for _, d in G.nodes(data=True)], dtype=np.float64)
    if elevations.size and np.isfinite(elevations).any():
        summary.update(elevation_min_m=float(np.nanmin(elevations)), elevation_max_m=float(np.nanmax(elevations)),
                       elevation_mean_m=float(np.nanmean(elevations)))
    sinks = [d.get("is_sink") for _, d in G.nodes(data=True)]
    if sinks and all(s is not None for s in sinks):
        summary["sinks"] = sum(1 for s in sinks if s is True or str(s).lower() == "true")
    return summary


_SUMMARY_ROWS = (
    ("nodes", "Nodes (junctions)"), ("edges", "Edges (directed)"), ("components", "Components"),
    ("source", "Network source"), ("elevation_source", "Elevation source"), ("drain_source", "Drain source"),
    ("bbox", "BBox [W,S,E,N]"), ("mean_edge_length_m", "Mean edge length (m)"),
    ("reverse_edges_added", "Reverse edges added"), ("elevation", "Elevation min/mean/max (m)"),
    ("sinks", "Sink junctions"), ("osm_base_utc", "OSM snapshot (UTC)"), ("osm_date", "OSM date pin"),
    ("osm_query_utc", "Overpass queried (UTC)"), ("elapsed_s", "Elapsed (s)"), ("created_utc", "Created (UTC)"),
    ("path", "Output"),
)


def format_summary(summary: Mapping[str, Any]) -> str:
    """Human-readable, aligned summary table (keys absent from ``summary`` are skipped)."""
    values = dict(summary)
    if "elevation_min_m" in values:
        keys = ("elevation_min_m", "elevation_mean_m", "elevation_max_m")
        values["elevation"] = " / ".join(f"{values[k]:.1f}" for k in keys)
    rows = [(label, f"{values[key]:.1f}" if isinstance(values[key], float) else values[key])
            for key, label in _SUMMARY_ROWS if values.get(key) is not None]
    width = max(len(label) for label, _ in rows)
    lines = [f"  {label:<{width}} : {value}" for label, value in rows]
    return "\n".join(["Namma-Flow stage 01 - road network", *lines])
