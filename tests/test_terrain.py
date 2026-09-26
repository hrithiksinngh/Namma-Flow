"""Tests for steepest-descent flow routing (``src/data_pipeline/terrain.py``, re-exported by ``elevation``)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.data_pipeline.elevation import flow_accumulation
from src.data_pipeline.graph_io import graph_to_arrays
from tests.conftest import make_grid_graph


# --------------------------------------------------------------------------- flow accumulation
def _line_graph(elevations, spacing_deg=0.001):
    G = nx.DiGraph()
    for i, elev in enumerate(elevations):
        G.add_node(i, x=77.66 + i * spacing_deg, y=12.93, elevation=float(elev))
    for i in range(len(elevations) - 1):
        G.add_edge(i, i + 1, length=100.0)
        G.add_edge(i + 1, i, length=100.0)
    return G


@pytest.mark.unit
def test_flow_accumulation_on_a_slope_and_a_valley():
    acc, sink = flow_accumulation(_line_graph([900, 895, 890, 885, 880]))
    np.testing.assert_array_equal(acc, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(sink, [False, False, False, False, True])
    acc, sink = flow_accumulation(_line_graph([900, 890, 880, 890, 900]))
    np.testing.assert_array_equal(acc, [1, 2, 5, 2, 1])
    np.testing.assert_array_equal(sink, [False, False, True, False, False])


@pytest.mark.unit
def test_flow_accumulation_steepest_descent_ties_flats_and_isolated():
    G = nx.DiGraph()
    G.add_node(0, x=77.66, y=12.93, elevation=900.0)
    G.add_node(1, x=77.661, y=12.93, elevation=899.0)  # 1 m drop over 10 m -> steepest
    G.add_node(2, x=77.66, y=12.931, elevation=880.0)  # 20 m drop over 1000 m
    G.add_node(3, x=77.70, y=12.95, elevation=870.0)  # isolated
    G.add_edge(0, 1, length=10.0)
    G.add_edge(0, 2, length=1000.0)
    acc, sink = flow_accumulation(G)
    np.testing.assert_array_equal(acc, [1, 2, 1, 1])
    np.testing.assert_array_equal(sink, [False, True, True, True])
    flat_acc, flat_sink = flow_accumulation(_line_graph([880, 880, 880]))
    np.testing.assert_array_equal(flat_acc, [1, 1, 1])
    assert flat_sink.all()


@pytest.mark.unit
def test_flow_accumulation_grid_conserves_nodes_and_uses_array_order():
    G = make_grid_graph(8, 8)
    arrays = graph_to_arrays(G)
    acc, sink = flow_accumulation(G)
    assert acc.shape == (arrays.num_nodes,) and sink.dtype == bool
    assert acc.min() >= 1
    assert acc[sink].sum() == pytest.approx(arrays.num_nodes)  # every path ends in exactly one sink
    lowest = int(np.argmin(arrays.node_attrs["elevation"]))
    assert sink[lowest]
    G.nodes[arrays.node_ids[0]].pop("elevation")
    with pytest.raises(ValueError, match="elevation"):
        flow_accumulation(G)
