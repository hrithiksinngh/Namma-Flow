"""Static drainage network of the label simulator, derived once from the enriched road graph.

For every junction ``i`` this module computes the catchment ``A_i``, the pondable area, the
street-inlet drain capacity, the multiple-flow-direction (MFD) routing matrix and the
*contributing-area operator* ``K`` that the local drain-surcharge mechanism uses (see
:mod:`src.hydrology.simulator` for the physics and the justification of each term).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays
from src.hydrology.params import HydrologyParams, SimulationError
from src.utils.geo import haversine_m
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

REFERENCE_GRADE = 0.01          # outflow_rate_per_h is the hourly release fraction at a 1 % grade
MIN_CATCHMENT_M2 = 1.0
MIN_SEGMENT_LENGTH_M = 1.0
ELEVATION_ATTR = "elevation"
DRAIN_ATTR = "dist_to_drain_m"
LENGTH_ATTR = "length"


@dataclass(frozen=True)
class DrainageNetwork:
    """Static per-junction hydraulic properties derived once from the road graph.

    ``routing`` is a sparse ``[N, N]`` matrix with ``routing[j, i]`` = share of junction ``i``'s
    outflow delivered to ``j`` (columns of non-sinks sum to 1), so ``inflow = routing @ outflow``.
    ``contributing`` is the row-stochastic sparse ``[N, N]`` operator with
    ``contributing[i, j] = U_ij A_j / sum_j U_ij A_j`` where ``U = (I - routing)^-1`` holds the
    fraction of junction ``j``'s runoff that passes through ``i``: ``contributing @ rain`` is the
    flow-weighted mean rain over each junction's contributing (upstream) area.
    ``contributing_area_m2`` is ``sum_j U_ij A_j``.
    """

    elevation_m: np.ndarray
    area_m2: np.ndarray
    pond_area_m2: np.ndarray
    drain_capacity_m3_h: np.ndarray
    outflow_fraction: np.ndarray
    downhill_slope: np.ndarray
    is_sink: np.ndarray
    spill_storage_m3: np.ndarray | None
    routing: sp.csr_matrix
    contributing: sp.csr_matrix
    contributing_area_m2: np.ndarray
    n_segments: int

    @property
    def num_nodes(self) -> int:
        return int(self.area_m2.size)


def as_arrays(graph: Any) -> GraphArrays:
    """Accept :class:`GraphArrays` or a networkx graph (converted with :func:`graph_to_arrays`)."""
    if isinstance(graph, GraphArrays):
        return graph
    if isinstance(graph, nx.Graph):
        return graph_to_arrays(graph)
    raise TypeError(f"graph must be GraphArrays or a networkx graph, got {type(graph).__name__}")


def _node_elevation(graph: GraphArrays) -> np.ndarray:
    if ELEVATION_ATTR not in graph.node_attrs:
        raise ValueError(
            "Graph nodes have no 'elevation' attribute; enrich the graph first "
            "(python src/data_pipeline/02_elevation_engine.py)"
        )
    elev = np.array(graph.node_attrs[ELEVATION_ATTR], dtype=np.float64, copy=True)
    bad = ~np.isfinite(elev)
    if bad.all():
        raise ValueError("No junction has a finite elevation; cannot route runoff")
    if bad.any():
        LOGGER.warning("%d junctions have non-finite elevation; using the mean of the others", int(bad.sum()))
        elev[bad] = float(elev[~bad].mean())
    return elev


def _edge_lengths(graph: GraphArrays) -> np.ndarray:
    """Length of every directed edge (m): the ``length`` attribute, repaired with haversine where invalid."""
    src, dst = graph.edge_index
    straight = np.maximum(haversine_m(graph.lon[src], graph.lat[src], graph.lon[dst], graph.lat[dst]), 0.0)
    if LENGTH_ATTR not in graph.edge_attrs:
        LOGGER.warning("Graph edges have no 'length' attribute; using straight-line (haversine) lengths")
        length = np.asarray(straight, dtype=np.float64)
    else:
        length = np.array(graph.edge_attrs[LENGTH_ATTR], dtype=np.float64, copy=True)
        bad = ~np.isfinite(length) | (length <= 0)
        if bad.any():
            LOGGER.warning("%d edges have an invalid length; using straight-line lengths", int(bad.sum()))
            length[bad] = straight[bad]
    return np.maximum(length, MIN_SEGMENT_LENGTH_M)


def _segments(graph: GraphArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected street segments ``(lo, hi, length)``; self-loops dropped, duplicates -> shortest."""
    if graph.num_edges == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float64)
    src, dst = graph.edge_index.astype(np.int64)
    length = _edge_lengths(graph)
    keep = src != dst
    lo, hi, length = np.minimum(src, dst)[keep], np.maximum(src, dst)[keep], length[keep]
    key = lo * graph.num_nodes + hi
    order = np.lexsort((length, key))
    key, lo, hi, length = key[order], lo[order], hi[order], length[order]
    first = np.ones(key.size, dtype=bool)
    first[1:] = key[1:] != key[:-1]
    return lo[first], hi[first], length[first]


def _mfd_weights(u: np.ndarray, slope: np.ndarray, n: int, exponent: float) -> np.ndarray:
    """MFD weights ``(slope / max slope of the source node) ** exponent`` (every source has one weight 1).

    Normalising by the steepest downhill slope of each source *before* the power keeps the
    weights in (0, 1] for any exponent, so ``weight / sum`` can never become ``0 / 0`` through
    underflow (fix-round finding R2-07: ``slope ** 100`` underflowed to 0 for gentle slopes).
    """
    steepest = np.zeros(n, dtype=np.float64)
    np.maximum.at(steepest, u, slope)
    return np.power(slope / steepest[u], exponent)


def _routing(
    n: int, elev: np.ndarray, lo: np.ndarray, hi: np.ndarray, length: np.ndarray, params: HydrologyParams
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """MFD routing matrix, hourly outflow fraction, flow-weighted downhill slope and sink mask."""
    u = np.concatenate([lo, hi])
    v = np.concatenate([hi, lo])
    seg = np.concatenate([length, length])
    drop = elev[u] - elev[v]
    down = drop > 0
    u, v, slope = u[down], v[down], drop[down] / seg[down]
    weight = _mfd_weights(u, slope, n, params.routing_exponent)
    total = np.bincount(u, weights=weight, minlength=n)
    share = weight / total[u]
    if not np.all(np.isfinite(share)):
        raise SimulationError(f"non-finite MFD routing shares ({int((~np.isfinite(share)).sum())} edges)")
    routing = sp.csr_matrix((share, (v, u)), shape=(n, n))
    downhill_slope = np.bincount(u, weights=share * slope, minlength=n)
    is_sink = np.bincount(u, minlength=n) == 0
    fraction = params.outflow_rate_per_h * np.maximum(downhill_slope, params.min_routing_grade) / REFERENCE_GRADE
    fraction = np.where(is_sink, 0.0, np.minimum(fraction, 1.0))
    return routing, fraction, downhill_slope, is_sink


def _upstream_operator(routing: sp.csr_matrix) -> sp.csr_matrix:
    """``U = (I - R)^-1 = I + R + R^2 + ...``: ``U[i, j]`` = share of junction ``j``'s water passing ``i``.

    Routing only moves water to strictly lower junctions, so ``R`` is nilpotent (a DAG) and the
    series terminates after at most ``N - 1`` terms; a longer series means a cycle (a bug).
    """
    n = routing.shape[0]
    upstream = sp.identity(n, format="csr", dtype=np.float64)
    term = upstream
    for _ in range(n):
        term = (routing @ term).tocsr()
        term.eliminate_zeros()
        if term.nnz == 0:
            return upstream
        upstream = (upstream + term).tocsr()
    raise SimulationError("routing matrix is not acyclic; the upstream series did not terminate")


def _contributing_operator(routing: sp.csr_matrix, area: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Row-stochastic flow-weighted contributing-area operator and the contributing areas (m^2)."""
    upstream = _upstream_operator(routing)
    weighted = (upstream @ sp.diags(area)).tocsr()
    contributing_area = np.asarray(weighted.sum(axis=1)).reshape(-1)
    operator = (sp.diags(1.0 / contributing_area) @ weighted).tocsr()
    if not np.all(np.isfinite(operator.data)):
        raise SimulationError("non-finite contributing-area weights")
    return operator, contributing_area


def _drain_distance(graph: GraphArrays) -> np.ndarray:
    if DRAIN_ATTR not in graph.node_attrs:
        LOGGER.warning("Graph nodes have no '%s' attribute; assuming every junction is far from drains", DRAIN_ATTR)
        return np.full(graph.num_nodes, np.inf)
    dist = np.array(graph.node_attrs[DRAIN_ATTR], dtype=np.float64, copy=True)
    dist[np.isnan(dist)] = np.inf
    return np.maximum(dist, 0.0)


def build_drainage_network(graph: GraphArrays | nx.Graph, params: HydrologyParams) -> DrainageNetwork:
    """Derive catchments, drain capacities, the routing matrix and the contributing-area operator."""
    arrays = as_arrays(graph)
    if not isinstance(params, HydrologyParams):
        raise TypeError(f"params must be HydrologyParams, got {type(params).__name__}")
    n = arrays.num_nodes
    elev = _node_elevation(arrays)
    lo, hi, length = _segments(arrays)
    half = 0.5 * length
    incident = np.bincount(lo, weights=half, minlength=n) + np.bincount(hi, weights=half, minlength=n)
    area = np.maximum(params.catchment_width_m * incident, MIN_CATCHMENT_M2)
    routing, fraction, slope, is_sink = _routing(n, elev, lo, hi, length, params)
    contributing, contributing_area = _contributing_operator(routing, area)
    near, far = params.drain_capacity_near_mm_h, params.drain_capacity_far_mm_h
    capacity_mm_h = far + (near - far) * np.exp(-_drain_distance(arrays) / params.drain_decay_m)
    pond = params.ponding_fraction * area
    spill = None if params.spill_depth_m is None else params.spill_depth_m * pond
    return DrainageNetwork(
        elevation_m=elev,
        area_m2=area,
        pond_area_m2=pond,
        drain_capacity_m3_h=area * capacity_mm_h / 1000.0,
        outflow_fraction=fraction,
        downhill_slope=slope,
        is_sink=is_sink,
        spill_storage_m3=spill,
        routing=routing,
        contributing=contributing,
        contributing_area_m2=contributing_area,
        n_segments=int(lo.size),
    )
