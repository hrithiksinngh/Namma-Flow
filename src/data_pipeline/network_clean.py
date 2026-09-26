"""Cleaning of raw osmnx graphs into the canonical, GraphML-safe road ``DiGraph`` (schema 2.1).

Split out of :mod:`src.data_pipeline.network`, which re-exports every public name.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import networkx as nx
import numpy as np

from src.data_pipeline.network_settings import NetworkSettings, _to_float
from src.utils.geo import haversine_m
from src.utils.logger import get_logger

LOGGER = get_logger("src.data_pipeline.network")

WGS84_CRS = "epsg:4326"
# networkx.read_graphml adds these dict-valued graph attrs; graph_io.save_graph turns them into
# JSON strings, which then crash networkx's GraphML writer, so they are stripped before saving.
_GRAPHML_READER_ATTRS = ("node_default", "edge_default")
LIST_SEPARATOR = ";"          # OSM's own multi-value separator
DEFAULT_HIGHWAY = "road"      # OSM tag value for "road of unknown classification"
LARGE_COMPONENT_LOSS = 0.10   # warn (not just inform) when cleaning drops > 10 % of junctions
_COORD_ATTRS = frozenset({"x", "y", "lon", "lat"})
_DROPPED_EDGE_ATTRS = frozenset({"geometry", "reversed", "length", "highway", "oneway", "osmid", "reversed_added"})
_TRUE_ONEWAY = frozenset({"yes", "true", "1", "-1", "reverse", "reversible", "alternating"})


class GraphTooSmallError(ValueError):
    """The (cleaned) road graph has fewer junctions than ``network.min_nodes``."""


# --------------------------------------------------------------------------- value normalisation
def _flatten_value(value: Any) -> Any:
    """Make an OSM attribute GraphML-safe: lists → ``a;b`` strings, geometry/NaN → ``None`` (drop)."""
    if value is None or hasattr(value, "geom_type"):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
        parts: list[str] = []
        for item in items:
            flat = _flatten_value(item)
            if flat is not None and str(flat) not in parts:
                parts.append(str(flat))
        return LIST_SEPARATOR.join(parts) if parts else None
    return str(value)


def _parse_oneway(value: Any) -> bool:
    """Interpret the many spellings of OSM ``oneway`` (bool, 'yes', '-1', lists …)."""
    if isinstance(value, (list, tuple, set)):
        return any(_parse_oneway(v) for v in value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_ONEWAY
    number = _to_float(value)
    return bool(math.isfinite(number) and number != 0)


def _valid_lengths(raw: Any) -> list[float]:
    """Finite, positive numbers in an OSM ``length`` value (scalar or list)."""
    candidates = raw if isinstance(raw, (list, tuple)) else [raw]
    return [x for x in (_to_float(c) for c in candidates) if math.isfinite(x) and x > 0]


def _resolve_length(raw: Any, haversine: float, min_len: float, default_len: float) -> float:
    """Edge length: valid OSM length → straight-line length → ``default_len``; never below ``min_len``."""
    valid = _valid_lengths(raw)
    if valid:
        return max(min(valid), min_len)
    if math.isfinite(haversine):
        return max(haversine, min_len)
    return max(default_len, min_len)


def _is_finite_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)) or (isinstance(value, str) and value.strip().lower() in ("true", "false")):
        return True
    return math.isfinite(_to_float(value))


# --------------------------------------------------------------------------- cleaning
def _is_wgs84(crs: Any) -> bool:
    if crs is None or str(crs).strip().lower() in (WGS84_CRS, "wgs84", "wgs 84"):
        return True
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        return CRS.from_user_input(crs).to_epsg() == 4326
    except CRSError as exc:
        raise ValueError(f"Graph CRS {crs!r} is not recognised by pyproj: {exc}") from exc


def _node_positions(G_raw: nx.Graph) -> dict[Any, tuple[float, float]]:
    """Return ``{node: (lon, lat)}`` for nodes with valid coordinates (reprojected if needed)."""
    nodes = list(G_raw.nodes)
    xs = np.array([_to_float(G_raw.nodes[n].get("x")) for n in nodes], dtype=np.float64)
    ys = np.array([_to_float(G_raw.nodes[n].get("y")) for n in nodes], dtype=np.float64)
    crs = G_raw.graph.get("crs")
    if nodes and not _is_wgs84(crs):
        from pyproj import Transformer

        LOGGER.info("Reprojecting %d nodes from %s to EPSG:4326", len(nodes), crs)
        xs, ys = Transformer.from_crs(crs, 4326, always_xy=True).transform(xs, ys)
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(xs) & np.isfinite(ys) & (np.abs(xs) <= 180.0) & (np.abs(ys) <= 90.0)
    dropped = len(nodes) - int(valid.sum())
    if dropped:
        LOGGER.warning("Dropped %d node(s) with missing or invalid x/y coordinates", dropped)
    return {n: (float(x), float(y)) for n, x, y, ok in zip(nodes, xs, ys, valid) if ok}


def _clean_node_attrs(data: Mapping[str, Any], lon: float, lat: float) -> dict[str, Any]:
    out: dict[str, Any] = {"x": lon, "y": lat}
    for key, value in data.items():
        if key in _COORD_ATTRS or key == "geometry":
            continue
        if key == "street_count":
            count = _to_float(value)
            if math.isfinite(count) and count >= 0:
                out[key] = int(count)
            continue
        flat = _flatten_value(value)
        if flat is not None:
            out[str(key)] = flat
    return out


def _clean_edge_attrs(data: Mapping[str, Any], length: float) -> dict[str, Any]:
    highway = _flatten_value(data.get("highway"))
    out: dict[str, Any] = {
        "length": float(length),
        "highway": str(highway) if highway not in (None, "") else DEFAULT_HIGHWAY,
        "oneway": _parse_oneway(data.get("oneway")),
        "reversed_added": False,
    }
    osmid = _flatten_value(data.get("osmid"))
    if osmid is not None:
        out["osmid"] = str(osmid)  # always a string: merged ways give "id1;id2"
    for key, value in data.items():
        if key in _DROPPED_EDGE_ATTRS:
            continue
        flat = _flatten_value(value)
        if flat is not None:
            out[str(key)] = flat
    return out


def _collect_edges(
    G_raw: nx.Graph, positions: Mapping[Any, tuple[float, float]], net: NetworkSettings
) -> dict[tuple[Any, Any], tuple[float, Mapping[str, Any]]]:
    """Map each directed pair to its shortest (length, raw attrs); self-loops are skipped."""
    best: dict[tuple[Any, Any], tuple[float, Mapping[str, Any]]] = {}
    self_loops = repaired = parallel = 0
    for u, v, data in G_raw.edges(data=True):
        if u not in positions or v not in positions:
            continue
        if u == v:
            self_loops += 1
            continue
        (lon_u, lat_u), (lon_v, lat_v) = positions[u], positions[v]
        straight = float(haversine_m(lon_u, lat_u, lon_v, lat_v))
        length = _resolve_length(data.get("length"), straight, net.min_edge_length_m, net.default_edge_length_m)
        repaired += not _valid_lengths(data.get("length"))
        for pair in ((u, v),) if G_raw.is_directed() else ((u, v), (v, u)):
            if pair in best:
                parallel += 1
                if length >= best[pair][0]:
                    continue
            best[pair] = (length, data)
    if self_loops:
        LOGGER.info("Removed %d self-loop edge(s)", self_loops)
    if parallel:
        LOGGER.info("Collapsed %d parallel edge(s), keeping the shortest segment", parallel)
    if repaired:
        LOGGER.warning("Repaired %d edge length(s) that were missing, non-numeric or <= 0 "
                       "(straight-line length used)", repaired)
    return best


def _prune_components(G: nx.DiGraph, keep_largest: bool) -> nx.DiGraph:
    """Drop isolated junctions and (optionally) every component but the largest one."""
    isolates = list(nx.isolates(G))
    if isolates:
        LOGGER.info("Removed %d isolated node(s)", len(isolates))
        G = G.subgraph(set(G.nodes) - set(isolates)).copy()
    if not keep_largest or G.number_of_nodes() == 0:
        return G
    components = list(nx.weakly_connected_components(G))
    if len(components) <= 1:
        return G
    largest = max(components, key=len)
    dropped = G.number_of_nodes() - len(largest)
    log = LOGGER.warning if dropped > LARGE_COMPONENT_LOSS * G.number_of_nodes() else LOGGER.info
    log("Kept the largest weakly connected component (%d nodes); dropped %d node(s) in %d smaller component(s)",
        len(largest), dropped, len(components) - 1)
    return G.subgraph(largest).copy()


def _with_reverse_edges(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy where every u→v has a v→u twin of equal length (runoff ignores one-way rules).

    Opposite one-way ways between the same junctions (dual carriageways) are parallel
    segments for runoff: both directions take the shorter length, so grades negate exactly.
    """
    out = G.copy()
    missing = [(u, v, d) for u, v, d in G.edges(data=True) if not G.has_edge(v, u)]
    for u, v, data in missing:
        out.add_edge(v, u, **{**data, "reversed_added": True})
    longer = [(u, v) for u, v, d in out.edges(data=True) if d["length"] > out.edges[v, u]["length"]]
    for u, v in longer:
        out.edges[u, v]["length"] = out.edges[v, u]["length"]
    if missing or longer:
        LOGGER.info("Bidirectional graph: added %d reverse edge(s), equalised %d opposite-direction length(s)",
                    len(missing), len(longer))
    return out


def clean_graph(G_raw: nx.Graph, cfg: Mapping[str, Any]) -> nx.DiGraph:
    """Turn a raw osmnx graph into the canonical, GraphML-safe road ``DiGraph``.

    Never mutates ``G_raw``. Nodes without valid coordinates, self-loops and isolated
    nodes are removed; parallel edges collapse to the shortest; bad lengths are
    repaired; list attributes are flattened; geometries dropped; only the largest
    weakly connected component is kept (``network.keep_largest_component``); reverse
    edges are added when ``network.bidirectional``. Raises :class:`GraphTooSmallError`
    if fewer than ``network.min_nodes`` junctions survive.
    """
    if not isinstance(G_raw, nx.Graph):
        raise TypeError(f"clean_graph expects a networkx graph, got {type(G_raw).__name__}")
    net = NetworkSettings.from_config(cfg)
    positions = _node_positions(G_raw)

    G = nx.DiGraph()
    graph_attrs = ((k, _flatten_value(v)) for k, v in G_raw.graph.items() if k not in _GRAPHML_READER_ATTRS)
    G.graph.update({str(k): v for k, v in graph_attrs if v is not None})
    G.graph["crs"] = WGS84_CRS
    for node, (lon, lat) in positions.items():
        G.add_node(node, **_clean_node_attrs(G_raw.nodes[node], lon, lat))
    for (u, v), (length, data) in _collect_edges(G_raw, positions, net).items():
        G.add_edge(u, v, **_clean_edge_attrs(data, length))

    G = _prune_components(G, net.keep_largest_component)
    if net.bidirectional:
        G = _with_reverse_edges(G)
    if G.number_of_nodes() < net.min_nodes:
        raise GraphTooSmallError(
            f"Cleaned road graph has {G.number_of_nodes()} junction(s), fewer than network.min_nodes="
            f"{net.min_nodes}; enlarge region.bbox, check the OSM extract, or lower network.min_nodes"
        )
    LOGGER.info("Cleaned road graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G
