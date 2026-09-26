"""Guarded re-enrichment of a cached stage-01 graph (F1-02), enrichment-settings drift (F1-03)
and diagnostics runs that never write the graph file (F5-08).

The REAL elevation module runs here, offline with an empty DEM directory: its only elevation
source is the synthetic valley formula, which is exactly the situation in which a real SRTM
graph must never be overwritten silently.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.data_pipeline import elevation_settings, network
from src.data_pipeline.elevation_settings import (
    ENRICHMENT_HASH_ATTR,
    ENRICHMENT_HASH_KEYS,
    enrichment_config_hash,
    enrichment_drift,
)
from src.data_pipeline.enrichment import can_recompute_derived, recompute_derived
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.data_pipeline.network import NetworkStageError, extract_network, missing_enrichment, network_config_hash
from src.data_pipeline.terrain import relative_elevation
from src.utils.config import deep_merge, resolve_path
from src.utils.http import NetworkUnavailable
from tests.conftest import make_grid_graph
from tests.test_network import BBOX, enricher, make_raw_osm_graph  # noqa: F401 - pytest fixture


@pytest.fixture
def log(caplog):
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def _real_cached_graph(cfg: dict, drop: tuple[str, ...] = ()) -> tuple[object, bytes]:
    """A cached OSM graph with 'real' SRTM elevations and OSM drains; ``drop`` node attributes removed."""
    G = make_grid_graph(6, 6)
    G.graph.update(source="osm_bbox", elevation_source="srtm", drain_source="osm", crs="epsg:4326",
                   network_config_hash=network_config_hash(cfg), created_utc="2026-09-23T13:05:12+00:00")
    for _, data in G.nodes(data=True):
        data["elevation"] = float(data["elevation"]) + 7.25  # distinguishable from any synthetic value
        for attr in drop:
            data.pop(attr, None)
    path = resolve_path(cfg, "graph_file")
    save_graph(G, path)
    return G, path.read_bytes()


def _online(cfg: dict, monkeypatch) -> dict:
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    return deep_merge(cfg, {"project": {"offline": False}})


def _no_module_b(monkeypatch) -> None:
    def refuse():
        raise AssertionError("the full elevation & drainage chain must not run for derived attributes")

    monkeypatch.setattr(network, "_load_enricher", refuse)


# --------------------------------------------------------------------------- F1-02
@pytest.mark.integration
def test_missing_derived_attributes_are_recomputed_from_stored_srtm_elevations(cfg, monkeypatch, log):
    original, _ = _real_cached_graph(cfg, drop=("flow_accumulation", "is_sink"))
    _no_module_b(monkeypatch)
    G = extract_network(cfg)
    assert missing_enrichment(G) == []
    assert G.graph["elevation_source"] == "srtm" and G.graph["drain_source"] == "osm"
    saved = load_graph(resolve_path(cfg, "graph_file"))
    assert saved.graph["elevation_source"] == "srtm" and missing_enrichment(saved) == []
    for node, data in original.nodes(data=True):
        assert saved.nodes[node]["elevation"] == pytest.approx(data["elevation"])
        assert saved.nodes[node]["dist_to_drain_m"] == pytest.approx(data["dist_to_drain_m"])
    assert "stored elevations" in log.text


@pytest.mark.integration
def test_full_reenrichment_offline_refuses_to_replace_srtm_with_synthetic(cfg, log):
    _, before = _real_cached_graph(cfg, drop=("dist_to_drain_m",))  # not derivable: needs module B
    with pytest.raises(NetworkStageError, match="Refusing to replace.*elevation_source srtm -> synthetic"):
        extract_network(cfg)
    assert resolve_path(cfg, "graph_file").read_bytes() == before, "the real graph must stay untouched"
    G = extract_network(cfg, allow_synthetic=True)
    assert G.graph["elevation_source"] == "synthetic" and "allow_synthetic" in log.text


@pytest.mark.unit
def test_derived_recompute_matches_the_terrain_functions(cfg, grid_graph):
    H = grid_graph.copy()
    for _, data in H.nodes(data=True):
        data.pop("relative_elevation")
    assert can_recompute_derived(H, ["relative_elevation"], None)
    assert not can_recompute_derived(H, ["relative_elevation"], "settings changed")
    assert not can_recompute_derived(H, ["dist_to_drain_m"], None)
    out = recompute_derived(H, cfg)
    arrays = graph_to_arrays(out)
    expected = relative_elevation(arrays.lon, arrays.lat, arrays.node_attrs["elevation"],
                                  cfg["elevation"]["tpi_radius_m"])
    np.testing.assert_allclose(arrays.node_attrs["relative_elevation"], expected)
    assert "relative_elevation" not in next(iter(H.nodes(data=True)))[1], "input graph must not be mutated"


# --------------------------------------------------------------------------- F1-03 (graph side)
@pytest.mark.integration
def test_enrichment_hash_is_stamped_and_drift_reenriches_the_cached_graph(cfg, log):
    first = extract_network(cfg)  # offline: synthetic grid, real (synthetic-source) enrichment
    assert first.graph[ENRICHMENT_HASH_ATTR] == enrichment_config_hash(cfg)
    assert extract_network(cfg).graph["created_utc"] == first.graph["created_utc"]  # plain cache hit
    wider = deep_merge(cfg, {"elevation": {"tpi_radius_m": 900.0}})
    assert enrichment_drift(first, wider) is not None
    G = extract_network(wider)
    assert "enrichment config hash" in log.text and G.graph[ENRICHMENT_HASH_ATTR] == enrichment_config_hash(wider)
    arrays = graph_to_arrays(G)
    expected = relative_elevation(arrays.lon, arrays.lat, arrays.node_attrs["elevation"], 900.0)
    np.testing.assert_allclose(arrays.node_attrs["relative_elevation"], expected, atol=1e-6)


@pytest.mark.unit
def test_enrichment_hash_covers_exactly_the_enrichment_settings(cfg):
    base = enrichment_config_hash(cfg)
    assert base == enrichment_config_hash(cfg), "stable within a run"
    for fetch_only in ({"elevation": {"request_timeout_s": 5, "max_retries": 9, "refresh_dem": True,
                                      "srtm_url_template": "https://mirror.test/{tile}.hgt.gz"}},
                       {"drains": {"request_timeout_s": 5, "refresh": True}},
                       {"network": {"request_timeout_s": 5, "overpass_urls": ["https://other.test/api"]}},
                       {"weather": {"forecast_days": 7}, "hydrology": {"spill_depth_m": 0.9}}):
        assert enrichment_config_hash(deep_merge(cfg, fetch_only)) == base, fetch_only
    for shaping in ({"elevation": {"tpi_radius_m": 600.0}}, {"elevation": {"max_abs_grade": 0.2}},
                    {"drains": {"exclude_water_values": []}}, {"drains": {"max_distance_m": 900.0}},
                    {"region": {"bbox": [77.60, 12.90, 77.70, 12.99]}}, {"network": {"min_edge_length_m": 3.0}}):
        assert enrichment_config_hash(deep_merge(cfg, shaping)) != base, shaping
    assert set(ENRICHMENT_HASH_KEYS) == {"elevation", "drains", "region", "network"}
    legacy = make_grid_graph(3, 3)
    assert enrichment_drift(legacy, cfg) is None, "graphs from before the hash cannot drift"
    assert elevation_settings.enrichment_settings(cfg)["drains"]["osm_tags"] == cfg["drains"]["osm_tags"]


# --------------------------------------------------------------------------- F5-08
@pytest.mark.integration
def test_persist_false_never_writes_the_graph_file(cfg, log):
    path = resolve_path(cfg, "graph_file")
    G = extract_network(cfg, persist=False)
    assert G.number_of_nodes() == 36 and missing_enrichment(G) == []
    assert not path.exists(), "a diagnostics run must not save the synthetic grid"
    _, before = _real_cached_graph(cfg, drop=("is_sink",))
    H = extract_network(cfg, persist=False)
    assert missing_enrichment(H) == [] and path.read_bytes() == before


@pytest.mark.integration
def test_extract_rebuilds_cached_synthetic_graph_when_online(cfg, enricher, monkeypatch, caplog):
    """F5-08: a cached synthetic grid protects no real data, so online it is replaced by OSM."""
    extract_network(cfg)  # offline: the synthetic grid is cached
    raw = make_raw_osm_graph(6, 6)
    raw.graph["bbox"] = json.dumps(list(BBOX))
    monkeypatch.setattr(network, "fetch_osm_graph", lambda c: (raw, "osm_bbox"))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(_online(cfg, monkeypatch))
    assert "synthetic" in caplog.text and "rebuilding it from OpenStreetMap" in caplog.text
    assert G.graph["source"] == "osm_bbox"
    assert load_graph(resolve_path(cfg, "graph_file")).graph["source"] == "osm_bbox"


@pytest.mark.integration
def test_extract_keeps_synthetic_grid_when_osm_still_unavailable(cfg, enricher, monkeypatch, caplog):
    extract_network(cfg)

    def unavailable(c):
        raise NetworkUnavailable("Overpass busy")

    monkeypatch.setattr(network, "fetch_osm_graph", unavailable)
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(_online(cfg, monkeypatch))
    assert G.graph["source"] == "synthetic_grid" and "Overpass busy" in caplog.text
