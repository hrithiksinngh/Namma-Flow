"""Tests for road-graph cleaning (``src/data_pipeline/network_clean.py``, re-exported by ``network``)."""

from __future__ import annotations

import copy
import logging

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.data_pipeline import network
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.data_pipeline.network import GraphTooSmallError, clean_graph
from src.utils.geo import haversine_m, to_utm
from tests.test_network import edge_snapshot, fake_enrich, make_raw_osm_graph, with_overrides


# --------------------------------------------------------------------------- clean_graph


@pytest.mark.unit
def test_clean_graph_returns_digraph_and_does_not_mutate_input(cfg):
    raw = make_raw_osm_graph()
    before_nodes = copy.deepcopy(dict(raw.nodes(data=True)))
    before_edges = edge_snapshot(raw)
    G = clean_graph(raw, cfg)
    assert type(G) is nx.DiGraph
    assert dict(raw.nodes(data=True)) == before_nodes
    assert edge_snapshot(raw) == before_edges
    assert G.graph["crs"] == "epsg:4326"
    assert G.number_of_nodes() == 25


@pytest.mark.unit
def test_clean_graph_keeps_shortest_parallel_edge(cfg):
    raw = make_raw_osm_graph()
    u, v = next(iter((u, v) for u, v, _ in raw.edges(keys=True)))
    raw.add_edge(u, v, length=5.0, highway="primary", oneway=False, osmid=999)
    raw.add_edge(u, v, length=5000.0, highway="trunk", oneway=False, osmid=998)
    G = clean_graph(raw, cfg)
    assert G.edges[u, v]["length"] == pytest.approx(5.0)
    assert G.edges[u, v]["highway"] == "primary"


@pytest.mark.unit
@pytest.mark.parametrize("bad_length", [None, float("nan"), -3.0, 0.0, "abc", float("inf")])
def test_clean_graph_repairs_bad_lengths_with_haversine(cfg, bad_length):
    raw = make_raw_osm_graph()
    u, v, k = next(iter(raw.edges(keys=True)))
    if bad_length is None:
        del raw.edges[u, v, k]["length"]
    else:
        raw.edges[u, v, k]["length"] = bad_length
    G = clean_graph(raw, cfg)
    du, dv = G.nodes[u], G.nodes[v]
    expected = haversine_m(du["x"], du["y"], dv["x"], dv["y"])
    assert G.edges[u, v]["length"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_clean_graph_enforces_min_edge_length(cfg):
    raw = make_raw_osm_graph()
    u, v, k = next(iter(raw.edges(keys=True)))
    raw.edges[u, v, k]["length"] = 0.01
    G = clean_graph(raw, with_overrides(cfg, {"network": {"min_edge_length_m": 2.0, "default_edge_length_m": 25.0}}))
    lengths = np.array([d["length"] for *_, d in G.edges(data=True)])
    assert lengths.min() >= 2.0
    assert G.edges[u, v]["length"] == pytest.approx(2.0)


@pytest.mark.unit
def test_edge_length_resolution_chain():
    assert network._resolve_length(12.5, 30.0, 1.0, 25.0) == pytest.approx(12.5)
    assert network._resolve_length(None, 30.0, 1.0, 25.0) == pytest.approx(30.0)
    assert network._resolve_length("nope", 0.0, 1.0, 25.0) == pytest.approx(1.0)
    assert network._resolve_length(-1, float("nan"), 1.0, 25.0) == pytest.approx(25.0)
    assert network._resolve_length([10.0, 12.0], 30.0, 1.0, 25.0) == pytest.approx(10.0)
    assert network._resolve_length(True, 30.0, 1.0, 25.0) == pytest.approx(30.0)


@pytest.mark.unit
def test_clean_graph_drops_nodes_without_valid_coordinates(cfg, caplog):
    raw = make_raw_osm_graph()
    ids = list(raw.nodes)
    del raw.nodes[ids[0]]["x"]
    raw.nodes[ids[1]]["y"] = float("nan")
    raw.nodes[ids[2]]["x"] = "not-a-number"
    raw.nodes[ids[3]]["y"] = 95.0  # impossible latitude
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = clean_graph(raw, cfg)
    for node in ids[:4]:
        assert node not in G
    assert "coordinates" in caplog.text


@pytest.mark.unit
def test_clean_graph_removes_self_loops_isolates_and_small_components(cfg, caplog):
    raw = make_raw_osm_graph()
    first = next(iter(raw.nodes))
    raw.add_edge(first, first, length=10.0, highway="residential", oneway=False)
    raw.add_node(1, x=77.66, y=12.93)  # isolated junction
    raw.add_node(2, x=77.661, y=12.931)
    raw.add_node(3, x=77.662, y=12.932)
    raw.add_edge(2, 3, length=150.0, highway="service", oneway=True)  # tiny detached component
    with caplog.at_level(logging.INFO, logger="namma_flow"):
        G = clean_graph(raw, cfg)
    assert nx.number_of_selfloops(G) == 0
    assert 1 not in G and 2 not in G and 3 not in G
    assert nx.is_weakly_connected(G)
    assert "component" in caplog.text


@pytest.mark.unit
def test_clean_graph_can_keep_all_components(cfg):
    raw = make_raw_osm_graph()
    raw.add_node(2, x=77.661, y=12.931)
    raw.add_node(3, x=77.662, y=12.932)
    raw.add_edge(2, 3, length=150.0, highway="service", oneway=True)
    G = clean_graph(raw, with_overrides(cfg, {"network": {"keep_largest_component": False}}))
    assert 2 in G and 3 in G
    assert nx.number_weakly_connected_components(G) == 2


@pytest.mark.unit
def test_clean_graph_raises_clear_error_when_too_small(cfg):
    raw = make_raw_osm_graph(3, 3)  # 9 junctions < min_nodes (20)
    with pytest.raises(GraphTooSmallError, match="min_nodes"):
        clean_graph(raw, cfg)
    assert issubclass(GraphTooSmallError, ValueError)
    with pytest.raises(GraphTooSmallError):
        clean_graph(nx.MultiDiGraph(), cfg)


@pytest.mark.unit
def test_clean_graph_rejects_non_graph_input(cfg):
    with pytest.raises(TypeError):
        clean_graph({"nodes": []}, cfg)


@pytest.mark.unit
def test_clean_graph_flattens_lists_and_drops_geometry(cfg):
    raw = make_raw_osm_graph()
    u, v, k = next(iter(raw.edges(keys=True)))
    raw.edges[u, v, k].update(osmid=[11, 12, 11], highway=["residential", "tertiary"],
                              name=["MG Road", "Old Airport Rd"], lanes=["2", "3"], maxspeed=np.int64(40))
    node = next(iter(raw.nodes))
    raw.nodes[node].update(highway=["traffic_signals", "crossing"], geometry=LineString([(0, 0), (1, 1)]))
    G = clean_graph(raw, cfg)
    data = G.edges[u, v]
    assert data["osmid"] == "11;12"
    assert data["highway"] == "residential;tertiary"
    assert data["name"] == "MG Road;Old Airport Rd"
    assert data["lanes"] == "2;3"
    assert data["maxspeed"] == 40 and type(data["maxspeed"]) is int
    assert "geometry" not in data and "reversed" not in data
    assert "geometry" not in G.nodes[node]
    assert G.nodes[node]["highway"] == "traffic_signals;crossing"
    # osmid is always a string on edges so GraphML sees a single attribute type.
    assert all(isinstance(d["osmid"], str) for *_, d in G.edges(data=True) if "osmid" in d)


@pytest.mark.unit
def test_clean_graph_normalises_highway_oneway_and_street_count(cfg):
    raw = make_raw_osm_graph()
    edges = list(raw.edges(keys=True))
    raw.edges[edges[0]].pop("highway")
    raw.edges[edges[1]]["oneway"] = "yes"
    raw.edges[edges[2]]["oneway"] = [False, True]
    raw.edges[edges[3]]["oneway"] = "-1"
    raw.edges[edges[4]]["oneway"] = "no"
    raw.edges[edges[5]]["oneway"] = None
    raw.edges[edges[6]]["name"] = float("nan")
    first = next(iter(raw.nodes))
    raw.nodes[first]["street_count"] = "3"
    second = list(raw.nodes)[1]
    raw.nodes[second]["street_count"] = "lots"
    G = clean_graph(raw, with_overrides(cfg, {"network": {"bidirectional": False}}))
    pick = lambda e: G.edges[e[0], e[1]]  # noqa: E731
    assert pick(edges[0])["highway"] == "road"
    assert pick(edges[1])["oneway"] is True
    assert pick(edges[2])["oneway"] is True
    assert pick(edges[3])["oneway"] is True
    assert pick(edges[4])["oneway"] is False
    assert pick(edges[5])["oneway"] is False
    assert "name" not in pick(edges[6])
    assert G.nodes[first]["street_count"] == 3
    assert "street_count" not in G.nodes[second]


@pytest.mark.unit
def test_clean_graph_adds_reverse_edges_when_bidirectional(cfg):
    raw = make_raw_osm_graph()
    # Make one street one-way by removing its reverse direction.
    u, v, _ = next(iter(raw.edges(keys=True)))
    raw.remove_edge(v, u)
    raw.edges[u, v, 0]["oneway"] = True
    G = clean_graph(raw, cfg)
    for a, b in G.edges:
        assert G.has_edge(b, a)
    assert G.edges[v, u]["reversed_added"] is True
    assert G.edges[u, v]["reversed_added"] is False
    assert G.edges[v, u]["length"] == pytest.approx(G.edges[u, v]["length"])
    assert G.edges[v, u]["oneway"] is True
    assert sum(d["reversed_added"] for *_, d in G.edges(data=True)) == 1


@pytest.mark.unit
def test_clean_graph_equalises_dual_carriageway_lengths(cfg):
    """Real case (6th Main Road): two opposite one-way ways of 70.56 m and 71.13 m join the same
    junctions; with bidirectional=true both directions get the shorter length so grades negate exactly."""
    raw = make_raw_osm_graph()
    u, v, _ = next(iter(raw.edges(keys=True)))
    raw.edges[u, v, 0].update(length=70.56, oneway=True, osmid=781732034)
    raw.edges[v, u, 0].update(length=71.13, oneway=True, osmid=781732035)
    G = clean_graph(raw, cfg)
    assert G.edges[u, v]["length"] == pytest.approx(70.56)
    assert G.edges[v, u]["length"] == pytest.approx(70.56)
    assert G.edges[v, u]["osmid"] == "781732035" and G.edges[v, u]["reversed_added"] is False
    enriched = fake_enrich(G, cfg)
    for a, b, data in enriched.edges(data=True):
        assert enriched.edges[b, a]["grade"] == pytest.approx(-data["grade"], abs=1e-12)
    kept = clean_graph(raw, with_overrides(cfg, {"network": {"bidirectional": False}}))
    assert kept.edges[v, u]["length"] == pytest.approx(71.13)


@pytest.mark.unit
def test_clean_graph_without_bidirectional_keeps_one_way_streets(cfg):
    raw = make_raw_osm_graph()
    u, v, _ = next(iter(raw.edges(keys=True)))
    raw.remove_edge(v, u)
    G = clean_graph(raw, with_overrides(cfg, {"network": {"bidirectional": False}}))
    assert G.has_edge(u, v) and not G.has_edge(v, u)
    assert not any(d["reversed_added"] for *_, d in G.edges(data=True))


@pytest.mark.unit
def test_clean_graph_accepts_undirected_input(cfg):
    raw = nx.Graph(make_raw_osm_graph().to_undirected())
    G = clean_graph(raw, with_overrides(cfg, {"network": {"bidirectional": False}}))
    assert type(G) is nx.DiGraph
    for a, b in G.edges:
        assert G.has_edge(b, a)
    assert not any(d["reversed_added"] for *_, d in G.edges(data=True))


@pytest.mark.unit
def test_clean_graph_reprojects_projected_input(cfg):
    raw = make_raw_osm_graph()
    lon = np.array([d["x"] for _, d in raw.nodes(data=True)])
    lat = np.array([d["y"] for _, d in raw.nodes(data=True)])
    x_m, y_m = to_utm(lon, lat)
    projected = raw.copy()
    projected.graph["crs"] = "EPSG:32643"
    for node, xm, ym in zip(projected.nodes, x_m, y_m):
        projected.nodes[node].update(x=float(xm), y=float(ym), lon=0.0, lat=0.0)
    G = clean_graph(projected, cfg)
    out_lon = np.array([G.nodes[n]["x"] for n in raw.nodes])
    out_lat = np.array([G.nodes[n]["y"] for n in raw.nodes])
    np.testing.assert_allclose(out_lon, lon, atol=1e-7)
    np.testing.assert_allclose(out_lat, lat, atol=1e-7)
    assert "lon" not in G.nodes[next(iter(raw.nodes))]
    assert G.graph["crs"] == "epsg:4326"


@pytest.mark.unit
def test_clean_graph_rejects_unknown_crs(cfg):
    raw = make_raw_osm_graph()
    raw.graph["crs"] = "not-a-crs"
    with pytest.raises(ValueError, match="CRS"):
        clean_graph(raw, cfg)


@pytest.mark.integration
def test_cleaned_graph_round_trips_through_graphml(cfg, tmp_path):
    raw = make_raw_osm_graph()
    u, v, k = next(iter(raw.edges(keys=True)))
    raw.edges[u, v, k]["osmid"] = [1, 2]
    G = clean_graph(raw, cfg)
    path = save_graph(G, tmp_path / "rt.graphml")
    back = load_graph(path)
    assert set(back.nodes) == set(G.nodes)
    assert back.number_of_edges() == G.number_of_edges()
    assert back.edges[u, v]["osmid"] == "1;2"
    assert isinstance(back.edges[u, v]["oneway"], bool)
    assert isinstance(back.edges[u, v]["reversed_added"], bool)
    arrays = graph_to_arrays(back)
    assert "length" in arrays.edge_attrs and "street_count" in arrays.node_attrs
