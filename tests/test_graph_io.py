"""Tests for ``src.data_pipeline.graph_io``: the attribute digest (X1) and parallel-edge loading (R1-09).

(Save / load / array conversion basics are covered in ``tests/test_utils.py``.)
"""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pytest

from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays, load_graph, save_graph
from tests.conftest import make_grid_graph

STATIC = ["elevation", "dist_to_drain_m", "relative_elevation"]
EDGES = ["length", "grade"]


@pytest.fixture
def arrays() -> GraphArrays:
    return graph_to_arrays(make_grid_graph(4, 5))


def _with(arrays: GraphArrays, *, node: dict | None = None, edge: dict | None = None) -> GraphArrays:
    return dataclasses.replace(arrays, node_attrs={**arrays.node_attrs, **(node or {})},
                               edge_attrs={**arrays.edge_attrs, **(edge or {})})


# --------------------------------------------------------------------------- X1: attributes_signature


@pytest.mark.unit
def test_attributes_signature_is_a_deterministic_short_hex(arrays):
    digest = arrays.attributes_signature(STATIC, EDGES)
    assert isinstance(digest, str) and len(digest) == 16 and int(digest, 16) >= 0
    assert digest == graph_to_arrays(make_grid_graph(4, 5)).attributes_signature(STATIC, EDGES)


@pytest.mark.unit
def test_attributes_signature_ignores_dict_and_name_order(arrays):
    reordered = dataclasses.replace(arrays, node_attrs=dict(reversed(list(arrays.node_attrs.items()))),
                                    edge_attrs=dict(reversed(list(arrays.edge_attrs.items()))))
    digest = arrays.attributes_signature(STATIC, EDGES)
    assert reordered.attributes_signature(STATIC, EDGES) == digest
    assert arrays.attributes_signature(list(reversed(STATIC)), ["grade", "length", "grade"]) == digest


@pytest.mark.unit
def test_attributes_signature_tracks_named_attribute_values_only(arrays):
    digest = arrays.attributes_signature(STATIC, EDGES)
    elevation = arrays.node_attrs["elevation"].copy()
    elevation[3] += 1e-3
    assert _with(arrays, node={"elevation": elevation}).attributes_signature(STATIC, EDGES) != digest
    grade = -arrays.edge_attrs["grade"]
    assert _with(arrays, edge={"grade": grade}).attributes_signature(STATIC, EDGES) != digest
    tiny = arrays.node_attrs["dist_to_drain_m"] + 1e-9  # below the 6-decimal rounding
    assert _with(arrays, node={"dist_to_drain_m": tiny}).attributes_signature(STATIC, EDGES) == digest
    other = arrays.node_attrs["street_count"] + 1.0  # not a model attribute
    assert _with(arrays, node={"street_count": other}).attributes_signature(STATIC, EDGES) == digest
    assert arrays.attributes_signature(STATIC[:2], EDGES) != digest  # the attribute set itself counts


@pytest.mark.unit
def test_attribute_changes_keep_the_topology_signature(arrays):
    changed = _with(arrays, node={"elevation": arrays.node_attrs["elevation"] + 150.0})
    assert changed.signature() == arrays.signature()
    assert changed.attributes_signature(STATIC, EDGES) != arrays.attributes_signature(STATIC, EDGES)


@pytest.mark.unit
def test_attributes_signature_covers_topology_and_canonicalises_values(arrays):
    digest = arrays.attributes_signature(STATIC, EDGES)
    other_topology = graph_to_arrays(make_grid_graph(5, 4))
    assert other_topology.attributes_signature(STATIC, EDGES) != digest
    zeros = np.zeros(arrays.num_nodes)
    assert (_with(arrays, node={"relative_elevation": -zeros}).attributes_signature(STATIC, EDGES)
            == _with(arrays, node={"relative_elevation": zeros}).attributes_signature(STATIC, EDGES))
    nan_a = np.full(arrays.num_nodes, np.nan)
    nan_b = np.frombuffer(np.full(arrays.num_nodes, 0x7FF8000000000001, dtype=np.int64).tobytes(), dtype=np.float64)
    assert (_with(arrays, node={"elevation": nan_a}).attributes_signature(STATIC, EDGES)
            == _with(arrays, node={"elevation": nan_b}).attributes_signature(STATIC, EDGES))


@pytest.mark.unit
def test_attributes_signature_survives_a_graphml_round_trip(tmp_path):
    G = make_grid_graph(4, 4)
    path = save_graph(G, tmp_path / "g.graphml")
    assert (graph_to_arrays(load_graph(path)).attributes_signature(STATIC, EDGES)
            == graph_to_arrays(G).attributes_signature(STATIC, EDGES))


@pytest.mark.unit
def test_attributes_signature_missing_attribute_raises_key_error(arrays):
    with pytest.raises(KeyError, match="node attribute 'soil_type'"):
        arrays.attributes_signature([*STATIC, "soil_type"], EDGES)
    with pytest.raises(KeyError, match="edge attribute 'width'"):
        arrays.attributes_signature(STATIC, ["length", "width"])
    with pytest.raises(TypeError, match="sequence of attribute names"):
        arrays.attributes_signature("elevation", EDGES)
    with pytest.raises(ValueError, match="empty names"):
        arrays.attributes_signature(["elevation", ""], EDGES)


@pytest.mark.unit
def test_attributes_signature_on_a_graph_without_edges():
    G = nx.DiGraph()
    G.add_node(1, x=77.66, y=12.93, elevation=880.0, dist_to_drain_m=10.0, relative_elevation=0.0)
    single = graph_to_arrays(G)
    digest = single.attributes_signature(STATIC, EDGES)
    assert len(digest) == 16
    G.nodes[1]["elevation"] = 881.0
    assert graph_to_arrays(G).attributes_signature(STATIC, EDGES) != digest


# --------------------------------------------------------------------------- R1-09: parallel edges


def _two_node_graph(graph: nx.Graph) -> nx.Graph:
    graph.add_node(1, x=77.66, y=12.93)
    graph.add_node(2, x=77.67, y=12.94)
    return graph


@pytest.mark.unit
def test_load_graph_replaces_not_merges_a_longer_parallel_edge(tmp_path):
    G = _two_node_graph(nx.MultiDiGraph())
    G.add_edge(1, 2, length=120.0, name="Service Road", bridge="yes", grade=0.05)
    G.add_edge(1, 2, length=60.0, name="Main Road")
    path = tmp_path / "multi.graphml"
    nx.write_graphml(G, path)
    loaded = load_graph(path)
    assert dict(loaded.edges[1, 2]) == {"length": 60.0, "name": "Main Road"}  # no stale bridge / grade


@pytest.mark.unit
def test_load_graph_keeps_the_shorter_edge_when_it_comes_first(tmp_path):
    G = _two_node_graph(nx.MultiDiGraph())
    G.add_edge(1, 2, length=40.0, name="Short")
    G.add_edge(1, 2, length=90.0, name="Long", tunnel="yes")
    path = tmp_path / "multi_first.graphml"
    nx.write_graphml(G, path)
    assert dict(load_graph(path).edges[1, 2]) == {"length": 40.0, "name": "Short"}


@pytest.mark.unit
def test_load_graph_undirected_multigraph_replaces_both_directions(tmp_path):
    G = _two_node_graph(nx.MultiGraph())
    G.add_edge(1, 2, length=150.0, name="Old", bridge="yes")
    G.add_edge(1, 2, length=70.0, name="New")
    path = tmp_path / "undirected.graphml"
    nx.write_graphml(G, path)
    loaded = load_graph(path)
    assert dict(loaded.edges[1, 2]) == dict(loaded.edges[2, 1]) == {"length": 70.0, "name": "New"}
    loaded.edges[1, 2]["name"] = "changed"
    assert loaded.edges[2, 1]["name"] == "New"  # the mirror owns its own attribute dict
