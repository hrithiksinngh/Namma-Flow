"""GraphML persistence and conversion of the road graph to dense numpy arrays.

The canonical in-memory representation between stages is a ``networkx.DiGraph``
whose nodes carry ``x`` (lon) / ``y`` (lat) and whose edges carry ``length``.
Stages that do numerical work convert it once to :class:`GraphArrays`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_bytes

LOGGER = get_logger(__name__)

ATTRIBUTE_DIGEST_DECIMALS = 6  # attribute values are rounded to this many decimals before hashing
ATTRIBUTE_DIGEST_TAG = b"namma-flow/graph-attributes/v1"
_SCALAR_TYPES = (bool, int, float, str)
# networkx.read_graphml injects these dict-valued graph attributes; they must not be re-serialised.
_GRAPHML_RESERVED_GRAPH_ATTRS = ("node_default", "edge_default")


class GraphFormatError(ValueError):
    """Raised when a graph file or object does not have the expected structure."""


def _sanitize_value(value: Any) -> Any:
    """Convert an attribute value into something GraphML can store, or ``None`` to drop it."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, _SCALAR_TYPES):
        return value
    if hasattr(value, "wkt"):  # shapely geometry
        return value.wkt
    if isinstance(value, (list, tuple, set, dict)):
        try:
            return json.dumps(sorted(value) if isinstance(value, set) else value, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _sanitize_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in attrs.items():
        converted = _sanitize_value(value)
        if converted is not None:
            clean[str(key)] = converted
    return clean


def _coerce_node_id(node: Any) -> Any:
    if isinstance(node, str):
        try:
            return int(node)
        except ValueError:
            return node
    return node


def save_graph(G: nx.DiGraph, path: str | Path) -> Path:
    """Write ``G`` to GraphML atomically after sanitising attribute types."""
    if G.number_of_nodes() == 0:
        raise GraphFormatError("Refusing to save an empty graph")
    clean = nx.DiGraph()
    graph_attrs = {k: v for k, v in G.graph.items() if k not in _GRAPHML_RESERVED_GRAPH_ATTRS}
    clean.graph.update(_sanitize_attrs(graph_attrs))
    for node, data in G.nodes(data=True):
        clean.add_node(node, **_sanitize_attrs(data))
    for u, v, data in G.edges(data=True):
        clean.add_edge(u, v, **_sanitize_attrs(data))
    # GraphML serialisation is in-memory then atomically moved into place.
    payload = "\n".join(nx.generate_graphml(clean, prettyprint=False)).encode("utf-8")
    return atomic_write_bytes(Path(path), payload)


def _set_edge(G: nx.DiGraph, u: Any, v: Any, data: Mapping[str, Any]) -> None:
    """Add ``u -> v`` with exactly ``data`` as attributes.

    ``G.add_edge(u, v, **data)`` would merge into an existing edge's dict, so a shorter
    parallel segment replacing a longer one would inherit the longer one's extra keys
    (``bridge``, ``name`` variants, a precomputed ``grade``, ...).
    """
    G.add_edge(u, v)
    attrs = G.edges[u, v]
    attrs.clear()
    attrs.update(dict(data))


def load_graph(path: str | Path) -> nx.DiGraph:
    """Read a GraphML file written by :func:`save_graph` (or osmnx) into a ``DiGraph``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    try:
        raw = nx.read_graphml(path)
    except Exception as exc:  # networkx raises several parser-specific types
        raise GraphFormatError(f"Could not parse GraphML {path}: {exc}") from exc

    G = nx.DiGraph()
    G.graph.update({k: v for k, v in raw.graph.items() if k not in _GRAPHML_RESERVED_GRAPH_ATTRS})
    for node, data in raw.nodes(data=True):
        G.add_node(_coerce_node_id(node), **data)
    edges = raw.edges(keys=True, data=True) if raw.is_multigraph() else raw.edges(data=True)
    for edge in edges:
        u, v, data = edge[0], edge[1], edge[-1]
        u, v = _coerce_node_id(u), _coerce_node_id(v)
        if G.has_edge(u, v):
            # Parallel edges from a MultiDiGraph source: keep the shorter segment.
            if float(data.get("length", np.inf)) >= float(G.edges[u, v].get("length", np.inf)):
                continue
        _set_edge(G, u, v, data)
        if not raw.is_directed():
            _set_edge(G, v, u, data)

    for node, data in G.nodes(data=True):
        for coord in ("x", "y"):
            if coord not in data:
                raise GraphFormatError(f"Node {node!r} in {path} has no '{coord}' coordinate")
            data[coord] = float(data[coord])
    LOGGER.debug("Loaded graph %s: %d nodes, %d edges", path, G.number_of_nodes(), G.number_of_edges())
    return G


@dataclass(frozen=True)
class GraphArrays:
    """Dense, index-aligned view of a road graph.

    Node order is ``sorted(G.nodes)`` (by string form) so the same graph always maps
    to the same row order, which checkpoints rely on.
    """

    node_ids: tuple
    lon: np.ndarray
    lat: np.ndarray
    node_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=np.int64))
    edge_attrs: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def node_matrix(self, names: Iterable[str]) -> np.ndarray:
        """Stack node attributes into ``[N, len(names)]`` (float32)."""
        names = list(names)
        missing = [n for n in names if n not in self.node_attrs]
        if missing:
            raise KeyError(f"Graph is missing node attributes {missing}; available: {sorted(self.node_attrs)}")
        if not names:
            return np.zeros((self.num_nodes, 0), dtype=np.float32)
        return np.stack([self.node_attrs[n] for n in names], axis=1).astype(np.float32)

    def edge_matrix(self, names: Iterable[str]) -> np.ndarray:
        """Stack edge attributes into ``[E, len(names)]`` (float32)."""
        names = list(names)
        missing = [n for n in names if n not in self.edge_attrs]
        if missing:
            raise KeyError(f"Graph is missing edge attributes {missing}; available: {sorted(self.edge_attrs)}")
        if not names:
            return np.zeros((self.num_edges, 0), dtype=np.float32)
        return np.stack([self.edge_attrs[n] for n in names], axis=1).astype(np.float32)

    def _topology_digest(self) -> Any:
        digest = hashlib.sha256()
        digest.update(json.dumps([str(n) for n in self.node_ids]).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.edge_index).tobytes())  # int64 from graph_to_arrays
        return digest

    def signature(self) -> dict[str, Any]:
        """Identity of the graph TOPOLOGY (node ids and order, edge_index) - the junction-order check.

        Attribute values (elevation, drain distance, grades, ...) are not covered; use
        :meth:`attributes_signature` for those.
        """
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "sha256": self._topology_digest().hexdigest()[:16],
        }

    def attributes_signature(self, node_names: Sequence[str], edge_names: Sequence[str]) -> str:
        """Content digest (16 hex chars of a sha256) of the topology plus the named attribute arrays.

        Covers node ids, ``edge_index`` and every named node / edge attribute as float64
        rounded to ``ATTRIBUTE_DIGEST_DECIMALS`` decimals (C order; ``-0.0`` and NaN
        canonicalised). Names are de-duplicated and sorted, so the digest depends neither on
        the order of ``node_attrs`` / ``edge_attrs`` nor on the order of the names. It changes
        when stage 02 re-enriches the graph (new DEM, drains or grades) even though the
        topology and :meth:`signature` stay the same. On a graph without edges, edge
        attributes are empty arrays by definition.

        Raises ``KeyError`` when a named attribute is not a numeric attribute of the graph.
        """
        nodes = _digest_names(node_names, "node_names")
        edges = _digest_names(edge_names, "edge_names")
        missing = [f"node attribute {n!r}" for n in nodes if n not in self.node_attrs]
        if self.num_edges:
            missing += [f"edge attribute {n!r}" for n in edges if n not in self.edge_attrs]
        if missing:
            raise KeyError(f"Cannot compute the graph attribute digest: the graph has no numeric {', '.join(missing)} "
                           f"(node attributes: {sorted(self.node_attrs)}; edge attributes: {sorted(self.edge_attrs)}); "
                           "re-run stage 02 (python src/data_pipeline/02_elevation_engine.py)")
        digest = self._topology_digest()
        digest.update(ATTRIBUTE_DIGEST_TAG)
        for kind, names, attrs in (("node", nodes, self.node_attrs), ("edge", edges, self.edge_attrs)):
            for name in names:
                values = attrs.get(name, np.zeros(0, dtype=np.float64))
                digest.update(json.dumps([kind, name, int(np.size(values))]).encode("utf-8"))
                digest.update(_canonical_bytes(values))
        return digest.hexdigest()[:16]


def _digest_names(names: Sequence[str], label: str) -> list[str]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        raise TypeError(f"{label} must be a sequence of attribute names, got {names!r}")
    out = [str(n) for n in names]
    if any(not n for n in out):
        raise ValueError(f"{label} must not contain empty names, got {list(names)!r}")
    return sorted(set(out))


def _canonical_bytes(values: Any) -> bytes:
    """float64 C-order bytes rounded to ``ATTRIBUTE_DIGEST_DECIMALS``; ``-0.0`` -> ``0.0``, one NaN pattern."""
    arr = np.round(np.asarray(values, dtype=np.float64).reshape(-1), ATTRIBUTE_DIGEST_DECIMALS) + 0.0
    arr = np.where(np.isnan(arr), np.nan, arr)
    return np.ascontiguousarray(arr, dtype=np.float64).tobytes()


def _numeric_attr(values: list[Any]) -> np.ndarray | None:
    """Return a float array if every value is numeric (bools count), else ``None``."""
    out = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            out[i] = float(value)
            continue
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "false"):
                out[i] = 1.0 if lowered == "true" else 0.0
                continue
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            return None
    return out


def graph_to_arrays(G: nx.DiGraph) -> GraphArrays:
    """Convert ``G`` to :class:`GraphArrays`; only attributes numeric on every node/edge are kept."""
    if G.number_of_nodes() == 0:
        raise GraphFormatError("Graph has no nodes")
    node_ids = tuple(sorted(G.nodes, key=str))
    index = {node: i for i, node in enumerate(node_ids)}
    node_data = [G.nodes[n] for n in node_ids]
    try:
        lon = np.array([float(d["x"]) for d in node_data], dtype=np.float64)
        lat = np.array([float(d["y"]) for d in node_data], dtype=np.float64)
    except KeyError as exc:
        raise GraphFormatError(f"Every node needs x/y coordinates: missing {exc}") from exc

    node_keys = set().union(*(d.keys() for d in node_data)) - {"x", "y"}
    node_attrs: dict[str, np.ndarray] = {}
    for key in sorted(node_keys):
        values = [d.get(key) for d in node_data]
        if any(v is None for v in values):
            continue
        arr = _numeric_attr(values)
        if arr is not None:
            node_attrs[key] = arr

    edges = sorted(G.edges(data=True), key=lambda e: (index[e[0]], index[e[1]]))
    edge_index = np.array([[index[u] for u, _, _ in edges], [index[v] for _, v, _ in edges]], dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_attrs: dict[str, np.ndarray] = {}
    if edges:
        edge_keys = set().union(*(d.keys() for _, _, d in edges))
        for key in sorted(edge_keys):
            values = [d.get(key) for _, _, d in edges]
            if any(v is None for v in values):
                continue
            arr = _numeric_attr(values)
            if arr is not None:
                edge_attrs[key] = arr

    return GraphArrays(
        node_ids=node_ids,
        lon=lon,
        lat=lat,
        node_attrs=node_attrs,
        edge_index=edge_index,
        edge_attrs=edge_attrs,
    )
