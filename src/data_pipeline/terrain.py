"""Terrain analysis on the road graph: topographic position index and steepest-descent flow routing.

Pure functions of node coordinates / elevations (no I/O), used by
:func:`src.data_pipeline.elevation.enrich_graph` and re-exported by that module.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from src.data_pipeline.drains import validate_coords
from src.data_pipeline.graph_io import graph_to_arrays
from src.utils.geo import haversine_m, lonlat_to_local_xy

__all__ = ["flow_accumulation", "relative_elevation"]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def relative_elevation(lon: Any, lat: Any, elev: Any, radius_m: float) -> np.ndarray:
    """Topographic position index: elevation minus the mean elevation of nodes within ``radius_m`` (incl. itself)."""
    lon, lat = validate_coords(lon, lat)
    values = np.atleast_1d(np.asarray(elev, dtype=np.float64))
    if values.shape != lon.shape:
        raise ValueError(f"elev must match lon/lat shape {lon.shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("elevations must be finite (fill voids first)")
    radius = _to_float(radius_m)
    if not (math.isfinite(radius) and radius > 0):
        raise ValueError(f"radius_m must be > 0, got {radius_m!r}")
    n = lon.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    centred = values - values.mean()  # better conditioned sums
    x, y = lonlat_to_local_xy(lon, lat)
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(radius, output_type="ndarray")
    sums = centred.copy()
    counts = np.ones(n)
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        sums += np.bincount(i, weights=centred[j], minlength=n) + np.bincount(j, weights=centred[i], minlength=n)
        counts += np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    return centred - sums / counts


def _edge_distances(arrays: Any) -> np.ndarray:
    src, dst = arrays.edge_index
    straight = np.atleast_1d(haversine_m(arrays.lon[src], arrays.lat[src], arrays.lon[dst], arrays.lat[dst]))
    straight = np.maximum(straight, 1e-3)
    length = arrays.edge_attrs.get("length")
    if length is None:
        return straight
    return np.where(np.isfinite(length) & (length > 0), length, straight)


def _steepest_receivers(arrays: Any, elev: np.ndarray) -> np.ndarray:
    """Index of each node's steepest strictly-lower neighbour over the undirected adjacency (-1 = sink)."""
    receivers = np.full(arrays.num_nodes, -1, dtype=np.int64)
    if arrays.num_edges == 0:
        return receivers
    src, dst = arrays.edge_index
    dist = _edge_distances(arrays)
    u, v, d = np.concatenate([src, dst]), np.concatenate([dst, src]), np.concatenate([dist, dist])
    drop = elev[u] - elev[v]
    keep = (drop > 0) & (u != v)
    if not keep.any():
        return receivers
    u, v, slope = u[keep], v[keep], drop[keep] / d[keep]
    order = np.lexsort((v, -slope, u))  # by node, steepest first, lowest neighbour index on ties
    u_sorted, v_sorted = u[order], v[order]
    first = np.ones(u_sorted.size, dtype=bool)
    first[1:] = u_sorted[1:] != u_sorted[:-1]
    receivers[u_sorted[first]] = v_sorted[first]
    return receivers


def flow_accumulation(G: nx.DiGraph) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent (D8-style) flow accumulation over the road graph, in ``graph_to_arrays`` order.

    Returns ``(accumulation [N] float >= 1, is_sink [N] bool)``: accumulation counts the
    nodes (incl. itself) whose descent path passes through a node; sinks have no
    strictly-lower neighbour. O(N log N).
    """
    if not isinstance(G, nx.Graph):
        raise TypeError(f"Expected a networkx graph, got {type(G).__name__}")
    arrays = graph_to_arrays(G)
    elev = arrays.node_attrs.get("elevation")
    if elev is None or not np.isfinite(elev).all():
        raise ValueError("Every node needs a finite numeric 'elevation' before flow accumulation")
    receivers = _steepest_receivers(arrays, elev)
    order = np.lexsort((np.arange(arrays.num_nodes), -elev))  # highest first
    accumulation = np.ones(arrays.num_nodes, dtype=np.float64)
    for node, target in zip(order.tolist(), receivers[order].tolist()):
        if target >= 0:
            accumulation[target] += accumulation[node]
    return accumulation, receivers < 0
