"""Tests for the road-network extraction stage (``src/data_pipeline/network.py``).

osmnx is never contacted: a fake osmnx namespace is injected through
``network._import_osmnx`` and module B's ``enrich_graph`` is replaced through
``network._load_enricher`` so these tests are independent of the elevation module.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import yaml
from shapely.geometry import LineString, box

from src.data_pipeline import network
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.data_pipeline.network import (
    ENRICHED_EDGE_ATTRS,
    ENRICHED_NODE_ATTRS,
    GraphTooSmallError,
    NetworkSettings,
    NetworkStageError,
    RegionSettings,
    clean_graph,
    extract_network,
    fetch_osm_graph,
    missing_enrichment,
    synthetic_grid_graph,
)
from src.utils.config import ConfigError, deep_merge, resolve_path
from src.utils.geo import haversine_m, point_in_bbox
from src.utils.http import NetworkUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "data_pipeline" / "01_extract_network.py"
BBOX = (77.655, 12.915, 77.700, 12.950)


# --------------------------------------------------------------------------- helpers


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a new config dict with ``overrides`` deep-merged (never mutates ``cfg``)."""
    return deep_merge(cfg, overrides)


def online(cfg: dict, monkeypatch) -> dict:
    """Config that is *not* offline (the network itself is always faked in these tests)."""
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    return with_overrides(cfg, {"project": {"offline": False}})


def make_raw_osm_graph(rows: int = 5, cols: int = 5, bbox=BBOX) -> nx.MultiDiGraph:
    """An osmnx-like MultiDiGraph: int ids, x/y, street_count, list attributes, geometries."""
    west, south, east, north = bbox
    lons = np.linspace(west + 0.003, east - 0.003, cols)
    lats = np.linspace(south + 0.003, north - 0.003, rows)
    G = nx.MultiDiGraph(crs="epsg:4326", created_with="osmnx 2.1.1", simplified=True)
    nid = lambda r, c: 5_000_000 + r * cols + c  # noqa: E731
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            G.add_node(nid(r, c), x=float(lon), y=float(lat), street_count=4)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr >= rows or cc >= cols:
                    continue
                u, v = nid(r, c), nid(rr, cc)
                du, dv = G.nodes[u], G.nodes[v]
                length = float(haversine_m(du["x"], du["y"], dv["x"], dv["y"]))
                geom = LineString([(du["x"], du["y"]), (dv["x"], dv["y"])])
                attrs = dict(osmid=100 + r * cols + c, highway="residential", oneway=False,
                             reversed=False, length=length, geometry=geom, name="Test Road")
                G.add_edge(u, v, **attrs)
                G.add_edge(v, u, **{**attrs, "reversed": True})
    return G


def fake_enrich(G: nx.DiGraph, cfg: dict) -> nx.DiGraph:
    """Stand-in for module B's ``enrich_graph``: returns a NEW, fully enriched graph."""
    out = copy.deepcopy(G)
    for _, data in out.nodes(data=True):
        elev = 880.0 + 30.0 * np.sin(data["y"] * 100) + 15.0 * np.cos(data["x"] * 100)
        data.update(elevation=float(elev), dist_to_drain_m=100.0, relative_elevation=0.0,
                    flow_accumulation=1.0, is_sink=False)
    for u, v, data in out.edges(data=True):
        data["grade"] = float((out.nodes[v]["elevation"] - out.nodes[u]["elevation"]) / data["length"])
    out.graph.update(elevation_source="synthetic", drain_source="synthetic_line")
    return out


class EnricherSpy:
    """Callable wrapper counting calls to the fake enricher."""

    def __init__(self, func=fake_enrich) -> None:
        self.func = func
        self.calls = 0

    def __call__(self, G, cfg):
        self.calls += 1
        return self.func(G, cfg)


@pytest.fixture
def enricher(monkeypatch) -> EnricherSpy:
    spy = EnricherSpy()
    monkeypatch.setattr(network, "_load_enricher", lambda: spy)
    return spy


class FakeOsmnx:
    """Minimal fake of the osmnx 2.x API surface used by ``fetch_osm_graph``."""

    def __init__(self, *, place_polygon=None, place_graph=None, place_error=None, geocode_error=None,
                 place_class="boundary", place_type="administrative", bbox_graph=None, bbox_error=None,
                 failing_endpoints=()) -> None:
        self.settings = SimpleNamespace(cache_folder="./cache", use_cache=False, requests_timeout=180,
                                        log_console=False, http_user_agent="osmnx-default",
                                        overpass_url="https://overpass-api.de/api")
        self.place_class, self.place_type = place_class, place_type
        self.failing_endpoints = set(failing_endpoints)
        self.place_polygon = place_polygon if place_polygon is not None else box(77.66, 12.92, 77.69, 12.945)
        self.place_graph = place_graph
        self.place_error = place_error
        self.geocode_error = geocode_error
        self.bbox_graph = bbox_graph
        self.bbox_error = bbox_error
        self.calls: list[tuple[str, tuple, dict]] = []
        self.settings_seen: list[dict] = []

    def _record(self, name, args, kwargs) -> None:
        self.calls.append((name, args, kwargs))
        self.settings_seen.append(dict(vars(self.settings)))

    def called(self, name: str) -> list[tuple[tuple, dict]]:
        return [(a, k) for n, a, k in self.calls if n == name]

    def geocode_to_gdf(self, query, **kwargs):
        self._record("geocode_to_gdf", (query,), kwargs)
        if self.geocode_error is not None:
            raise self.geocode_error
        return gpd.GeoDataFrame({"display_name": [str(query)], "class": [self.place_class],
                                 "type": [self.place_type], "name": [str(query).split(",")[0]]},
                                geometry=[self.place_polygon], crs="EPSG:4326")

    def graph_from_place(self, query, **kwargs):
        self._record("graph_from_place", (query,), kwargs)
        if self.place_error is not None:
            raise self.place_error
        return self.place_graph if self.place_graph is not None else make_raw_osm_graph(8, 8)

    def graph_from_bbox(self, bbox, **kwargs):
        self._record("graph_from_bbox", (), {"bbox": bbox, **kwargs})
        if self.settings.overpass_url in self.failing_endpoints:
            raise ConnectionError(f"[Errno 61] Connection refused: {self.settings.overpass_url}")
        if self.bbox_error is not None:
            raise self.bbox_error
        return self.bbox_graph if self.bbox_graph is not None else make_raw_osm_graph(6, 6)


def install_fake_osmnx(monkeypatch, fake: FakeOsmnx) -> FakeOsmnx:
    monkeypatch.setattr(network, "_import_osmnx", lambda: fake)
    return fake


def edge_snapshot(G: nx.MultiDiGraph) -> list[str]:
    """Order-independent textual snapshot of every edge and its attributes."""
    return sorted(f"{u}->{v}#{k}:{sorted(d.items(), key=lambda kv: kv[0])!r}"
                  for u, v, k, d in G.edges(keys=True, data=True))


def load_cli():
    """Import the numbered stage-01 script (not importable by name) as a module."""
    spec = importlib.util.spec_from_file_location("extract_network_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(cfg: dict, path: Path) -> Path:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.write_text(yaml.safe_dump(clean), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- settings


@pytest.mark.unit
def test_settings_defaults_when_sections_missing():
    minimal = {"project": {"region": "Somewhere"}, "paths": {}}
    net = NetworkSettings.from_config(minimal)
    reg = RegionSettings.from_config(minimal)
    assert net.network_type == "drive" and net.simplify and net.bidirectional
    assert net.min_nodes >= 2 and net.min_edge_length_m > 0
    assert net.grid_rows >= 2 and net.grid_cols >= 2
    assert reg.place == "Somewhere" and len(reg.bbox) == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"network": {"min_nodes": 1}},
        {"network": {"min_nodes": "many"}},
        {"network": {"min_edge_length_m": 0}},
        {"network": {"min_edge_length_m": float("nan")}},
        {"network": {"default_edge_length_m": 0.5, "min_edge_length_m": 1.0}},
        {"network": {"network_type": "boat"}},
        {"network": {"simplify": "yes"}},
        {"network": {"bidirectional": 1}},
        {"network": {"request_timeout_s": -5}},
        {"network": {"synthetic_grid": {"rows": 1, "cols": 5}}},
        {"network": {"synthetic_grid": {"rows": 2.5, "cols": 5}}},
        {"network": {"synthetic_grid": {"rows": True, "cols": 5}}},
        {"network": {"synthetic_grid": "20x20"}},
        {"network": {"overpass_urls": []}},
        {"network": {"overpass_urls": ["ftp://mirror.example/api"]}},
        {"network": {"overpass_urls": 42}},
        {"region": {"max_place_area_km2": 0}},
        {"region": {"min_place_nodes": 0}},
        {"region": {"use_place_query": "true"}},
    ],
)
def test_settings_reject_invalid_values(cfg, overrides):
    bad = with_overrides(cfg, overrides)
    settings_cls = RegionSettings if "region" in overrides else NetworkSettings
    with pytest.raises(ConfigError):
        settings_cls.from_config(bad)


@pytest.mark.unit
def test_region_settings_bad_bbox_and_blank_place(cfg):
    with pytest.raises(ConfigError):
        RegionSettings.from_config(with_overrides(cfg, {"region": {"bbox": [77.7, 12.9, 77.6, 12.95]}}))
    blank = RegionSettings.from_config(with_overrides(cfg, {"project": {"region": "   "}}))
    assert blank.place is None


@pytest.mark.unit
def test_large_bbox_logs_warning(cfg, monkeypatch, caplog):
    install_fake_osmnx(monkeypatch, FakeOsmnx())
    big = with_overrides(online(cfg, monkeypatch),
                         {"region": {"bbox": [77.3, 12.7, 77.9, 13.2], "use_place_query": False}})
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        fetch_osm_graph(big)
    assert "large" in caplog.text.lower()


# --------------------------------------------------------------------------- synthetic grid


@pytest.mark.unit
def test_synthetic_grid_structure(cfg):
    G = synthetic_grid_graph(cfg)  # conftest: 6 x 6
    rows = cols = 6
    assert G.number_of_nodes() == rows * cols
    assert G.number_of_edges() == 2 * (rows * (cols - 1) + cols * (rows - 1))
    lon = np.array([d["x"] for _, d in G.nodes(data=True)])
    lat = np.array([d["y"] for _, d in G.nodes(data=True)])
    assert point_in_bbox(lon, lat, BBOX).all()
    assert nx.is_weakly_connected(G)
    for a, b, data in G.edges(data=True):
        assert G.has_edge(b, a)
        da, db = G.nodes[a], G.nodes[b]
        assert data["length"] == pytest.approx(haversine_m(da["x"], da["y"], db["x"], db["y"]))
        assert data["reversed_added"] is False and data["oneway"] is False
        assert isinstance(data["highway"], str)
    assert G.graph["source"] == "synthetic_grid"
    assert json.loads(G.graph["bbox"]) == list(BBOX)
    assert G.graph["crs"] == "epsg:4326"
    assert all(d["street_count"] in (2, 3, 4) for _, d in G.nodes(data=True))


@pytest.mark.unit
def test_synthetic_grid_is_deterministic_and_rectangular(cfg):
    cfg2 = with_overrides(cfg, {"network": {"synthetic_grid": {"rows": 4, "cols": 7}}})
    a, b = synthetic_grid_graph(cfg2), synthetic_grid_graph(cfg2)
    assert list(a.nodes(data=True)) == list(b.nodes(data=True))
    assert list(a.edges(data=True)) == list(b.edges(data=True))
    assert a.number_of_nodes() == 28


@pytest.mark.unit
def test_synthetic_grid_too_small_for_min_nodes(cfg):
    small = with_overrides(cfg, {"network": {"synthetic_grid": {"rows": 3, "cols": 3}}})
    with pytest.raises(GraphTooSmallError, match="synthetic_grid"):
        synthetic_grid_graph(small)


# --------------------------------------------------------------------------- fetch_osm_graph


@pytest.mark.unit
def test_fetch_refuses_when_offline(cfg, monkeypatch):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx())
    with pytest.raises(NetworkUnavailable, match="[Oo]ffline"):
        fetch_osm_graph(cfg)
    assert fake.calls == []


@pytest.mark.unit
def test_fetch_uses_place_graph_when_acceptable(cfg, monkeypatch):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx())
    G, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_place"
    assert G.number_of_nodes() == 64
    (args, kwargs), = fake.called("graph_from_place")
    assert args[0] == cfg["project"]["region"]
    assert kwargs["network_type"] == "drive" and kwargs["simplify"] is True
    assert json.loads(G.graph["bbox"]) == pytest.approx([77.66, 12.92, 77.69, 12.945])
    assert fake.called("graph_from_bbox") == []


@pytest.mark.unit
def test_fetch_rejects_water_body_geocode_before_overpass(cfg, monkeypatch, caplog):
    """Real observed case: osmnx picks 'Bellandur Lake' (class=water, type=lake) for the place query."""
    lake = box(77.643, 12.928, 77.680, 12.945)
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(place_polygon=lake, place_class="water", place_type="lake"))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        _, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_bbox"
    assert fake.called("graph_from_place") == []
    assert "water body" in caplog.text


@pytest.mark.unit
def test_fetch_falls_back_to_bbox_when_place_has_no_nodes(cfg, monkeypatch, caplog):
    """If a polygon slips through the pre-checks, osmnx's 'no graph nodes' error also falls back."""
    lake = box(77.66, 12.93, 77.68, 12.945)
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(
        place_polygon=lake, place_error=ValueError("Found no graph nodes within the requested polygon")))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_bbox"
    (_, kwargs), = fake.called("graph_from_bbox")
    assert kwargs["bbox"] == BBOX
    assert kwargs["network_type"] == "drive" and kwargs["simplify"] is True
    assert json.loads(G.graph["bbox"]) == list(BBOX)
    assert "no graph nodes" in caplog.text.lower()


@pytest.mark.unit
def test_fetch_rejects_oversized_place_polygon(cfg, monkeypatch, caplog):
    huge = box(77.3, 12.7, 77.9, 13.2)  # ~3,600 km2, e.g. geocoder matched the whole city
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(place_polygon=huge))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        _, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_bbox"
    assert fake.called("graph_from_place") == []
    assert "max_place_area_km2" in caplog.text


@pytest.mark.unit
def test_fetch_rejects_place_outside_bbox(cfg, monkeypatch, caplog):
    elsewhere = box(75.0, 15.0, 75.02, 15.02)  # a same-named locality in another district
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(place_polygon=elsewhere))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        _, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_bbox"
    assert fake.called("graph_from_place") == []
    assert "region.bbox" in caplog.text


@pytest.mark.unit
def test_fetch_rejects_tiny_place_graph(cfg, monkeypatch, caplog):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(place_graph=make_raw_osm_graph(3, 3)))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        _, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_bbox"
    assert "min_place_nodes" in caplog.text
    assert len(fake.called("graph_from_bbox")) == 1


@pytest.mark.unit
def test_fetch_falls_back_when_geocoding_fails(cfg, monkeypatch):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(geocode_error=TypeError("Nominatim geocoder returned 0 results")))
    _, source = fetch_osm_graph(online(cfg, monkeypatch))
    assert source == "osm_bbox"
    assert fake.called("graph_from_place") == []


@pytest.mark.unit
def test_fetch_skips_place_query_when_disabled(cfg, monkeypatch):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx())
    cfg2 = with_overrides(online(cfg, monkeypatch), {"region": {"use_place_query": False}})
    _, source = fetch_osm_graph(cfg2)
    assert source == "osm_bbox"
    assert fake.called("geocode_to_gdf") == []


@pytest.mark.unit
def test_fetch_raises_network_unavailable_when_bbox_fails(cfg, monkeypatch):
    install_fake_osmnx(monkeypatch, FakeOsmnx(
        place_error=ConnectionError("dns failure"), bbox_error=ConnectionError("dns failure")))
    with pytest.raises(NetworkUnavailable, match="bbox") as info:
        fetch_osm_graph(online(cfg, monkeypatch))
    assert "dns failure" in str(info.value)


@pytest.mark.unit
def test_fetch_tries_overpass_mirrors_in_order(cfg, monkeypatch, caplog):
    primary, mirror = "https://overpass-api.de/api", "https://maps.mail.ru/osm/tools/overpass/api"
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx(failing_endpoints={primary}))
    cfg2 = with_overrides(online(cfg, monkeypatch), {"region": {"use_place_query": False},
                                                     "network": {"overpass_urls": [primary, mirror + "/"]}})
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G, source = fetch_osm_graph(cfg2)
    assert source == "osm_bbox" and G.number_of_nodes() == 36
    assert [s["overpass_url"] for s in fake.settings_seen] == [primary, mirror]
    assert "Connection refused" in caplog.text
    assert fake.settings.overpass_url == primary  # global osmnx setting restored


@pytest.mark.unit
def test_fetch_reports_every_failed_mirror(cfg, monkeypatch):
    urls = ["https://a.example/api", "https://b.example/api"]
    install_fake_osmnx(monkeypatch, FakeOsmnx(failing_endpoints=set(urls)))
    cfg2 = with_overrides(online(cfg, monkeypatch), {"region": {"use_place_query": False},
                                                     "network": {"overpass_urls": urls}})
    with pytest.raises(NetworkUnavailable, match="every Overpass endpoint") as info:
        fetch_osm_graph(cfg2)
    assert all(u in str(info.value) for u in urls)


@pytest.mark.unit
def test_fractional_timeout_is_passed_through(cfg, monkeypatch):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx())
    fetch_osm_graph(with_overrides(online(cfg, monkeypatch), {"network": {"request_timeout_s": 90.5}}))
    assert fake.settings_seen[0]["requests_timeout"] == 90.5


@pytest.mark.unit
def test_fetch_raises_network_unavailable_on_empty_bbox_graph(cfg, monkeypatch):
    install_fake_osmnx(monkeypatch, FakeOsmnx(place_error=ValueError("no nodes"), bbox_graph=nx.MultiDiGraph()))
    with pytest.raises(NetworkUnavailable, match="empty"):
        fetch_osm_graph(online(cfg, monkeypatch))


@pytest.mark.unit
def test_fetch_configures_and_restores_osmnx_settings(cfg, monkeypatch):
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx())
    original = dict(vars(fake.settings))
    fetch_osm_graph(online(cfg, monkeypatch))
    seen = fake.settings_seen[0]
    assert Path(seen["cache_folder"]) == resolve_path(cfg, "osm_cache_dir")
    assert seen["use_cache"] is True
    # Integral timeouts are passed as int: osmnx writes them into the query text (= cache key).
    assert seen["requests_timeout"] == 180 and type(seen["requests_timeout"]) is int
    assert seen["overpass_url"] == "https://overpass-api.de/api"
    assert dict(vars(fake.settings)) == original
    assert resolve_path(cfg, "osm_cache_dir").is_dir()


@pytest.mark.unit
def test_fetch_reports_missing_osmnx(cfg, monkeypatch):
    def broken():
        raise ImportError("No module named 'osmnx'")

    monkeypatch.setattr(network, "_import_osmnx", broken)
    with pytest.raises(NetworkUnavailable, match="osmnx"):
        fetch_osm_graph(online(cfg, monkeypatch))


@pytest.mark.unit
def test_import_osmnx_returns_module():
    assert hasattr(network._import_osmnx(), "graph_from_bbox")


# --------------------------------------------------------------------------- enrichment checks


@pytest.mark.unit
def test_missing_enrichment_detects_partial_and_non_finite(grid_graph):
    assert missing_enrichment(grid_graph) == []
    G = grid_graph.copy()
    first = next(iter(G.nodes))
    del G.nodes[first]["dist_to_drain_m"]
    G.nodes[list(G.nodes)[1]]["elevation"] = float("nan")
    u, v = next(iter(G.edges))
    del G.edges[u, v]["grade"]
    missing = missing_enrichment(G)
    assert "dist_to_drain_m" in missing and "elevation" in missing and "grade" in missing
    bare = nx.DiGraph()
    bare.add_node(1, x=77.66, y=12.93)
    assert set(missing_enrichment(bare)) >= set(ENRICHED_NODE_ATTRS)
    bare.add_edge(1, 1)
    assert set(missing_enrichment(bare)) >= set(ENRICHED_EDGE_ATTRS)


# --------------------------------------------------------------------------- extract_network


@pytest.mark.integration
def test_extract_offline_builds_enriched_synthetic_graph(cfg, enricher, caplog):
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(cfg)
    path = resolve_path(cfg, "graph_file")
    assert path.exists()
    assert enricher.calls == 1
    assert G.number_of_nodes() == 36
    assert G.graph["source"] == "synthetic_grid"
    assert G.graph["elevation_source"] == "synthetic"
    assert G.graph["drain_source"] == "synthetic_line"
    assert G.graph["crs"] == "epsg:4326"
    assert json.loads(G.graph["bbox"]) == list(BBOX)
    assert "T" in G.graph["created_utc"]
    assert missing_enrichment(G) == []
    assert "offline" in caplog.text.lower()
    arrays = graph_to_arrays(G)
    assert {"elevation", "dist_to_drain_m", "relative_elevation"} <= set(arrays.node_attrs)
    assert {"length", "grade"} <= set(arrays.edge_attrs)


@pytest.mark.integration
def test_extract_reuses_enriched_cache_without_rebuilding(cfg, enricher, monkeypatch):
    first = extract_network(cfg)
    monkeypatch.setattr(network, "synthetic_grid_graph", lambda c: pytest.fail("cache should be used"))
    monkeypatch.setattr(network, "fetch_osm_graph", lambda c: pytest.fail("cache should be used"))
    second = extract_network(cfg)
    assert enricher.calls == 1
    assert sorted(first.nodes) == sorted(second.nodes)
    assert first.graph["created_utc"] == second.graph["created_utc"]


@pytest.mark.integration
def test_extract_reenriches_cached_graph_missing_attributes(cfg, enricher):
    path = resolve_path(cfg, "graph_file")
    bare = synthetic_grid_graph(cfg)
    save_graph(bare, path)
    G = extract_network(cfg)
    assert enricher.calls == 1
    assert missing_enrichment(G) == []
    assert missing_enrichment(load_graph(path)) == []


@pytest.mark.integration
def test_extract_force_rebuilds(cfg, enricher):
    extract_network(cfg)
    extract_network(cfg, force=True)
    assert enricher.calls == 2


@pytest.mark.integration
def test_extract_rebuilds_corrupt_cache(cfg, enricher, caplog):
    path = resolve_path(cfg, "graph_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<graphml><<< definitely not xml", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(cfg)
    assert G.number_of_nodes() == 36
    assert "could not be read" in caplog.text
    assert load_graph(path).number_of_nodes() == 36


@pytest.mark.integration
def test_extract_rebuilds_cache_that_is_too_small(cfg, enricher, caplog):
    path = resolve_path(cfg, "graph_file")
    tiny = fake_enrich(synthetic_grid_graph(with_overrides(
        cfg, {"network": {"min_nodes": 2, "synthetic_grid": {"rows": 2, "cols": 2}}})), cfg)
    save_graph(tiny, path)
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(cfg)
    assert G.number_of_nodes() == 36
    assert "unusable" in caplog.text


@pytest.mark.integration
def test_extract_warns_on_stale_cache_but_keeps_it(cfg, enricher, caplog):
    extract_network(cfg)
    changed = with_overrides(cfg, {"region": {"bbox": [77.60, 12.90, 77.70, 12.99]}})
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(changed)
    assert enricher.calls == 1
    assert json.loads(G.graph["bbox"]) == list(BBOX)
    assert "--force" in caplog.text


@pytest.mark.integration
def test_extract_online_uses_osm_graph(cfg, enricher, monkeypatch):
    raw = make_raw_osm_graph(6, 6)
    raw.graph["bbox"] = json.dumps(list(BBOX))
    monkeypatch.setattr(network, "fetch_osm_graph", lambda c: (raw, "osm_bbox"))
    G = extract_network(online(cfg, monkeypatch))
    assert G.graph["source"] == "osm_bbox"
    assert G.number_of_nodes() == 36
    assert json.loads(G.graph["bbox"]) == list(BBOX)
    assert G.graph["network_config_hash"]
    assert missing_enrichment(G) == []


@pytest.mark.integration
def test_extract_falls_back_to_synthetic_when_osm_unavailable(cfg, enricher, monkeypatch, caplog):
    def unavailable(c):
        raise NetworkUnavailable("Overpass timed out")

    monkeypatch.setattr(network, "fetch_osm_graph", unavailable)
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(online(cfg, monkeypatch))
    assert G.graph["source"] == "synthetic_grid"
    assert "Overpass timed out" in caplog.text


@pytest.mark.integration
def test_extract_falls_back_to_synthetic_when_osm_graph_too_small(cfg, enricher, monkeypatch, caplog):
    monkeypatch.setattr(network, "fetch_osm_graph", lambda c: (make_raw_osm_graph(3, 3), "osm_bbox"))
    with caplog.at_level(logging.WARNING, logger="namma_flow"):
        G = extract_network(online(cfg, monkeypatch))
    assert G.graph["source"] == "synthetic_grid"
    assert "min_nodes" in caplog.text


@pytest.mark.integration
def test_extract_fails_clearly_when_elevation_module_missing(cfg, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.data_pipeline.elevation", None)
    with pytest.raises(NetworkStageError, match="enrich_graph"):
        extract_network(cfg)
    assert not resolve_path(cfg, "graph_file").exists()


@pytest.mark.integration
def test_extract_rejects_half_enriched_result(cfg, monkeypatch):
    def lazy(G, c):
        out = G.copy()
        for _, d in out.nodes(data=True):
            d["elevation"] = 880.0
        return out

    monkeypatch.setattr(network, "_load_enricher", lambda: lazy)
    with pytest.raises(NetworkStageError, match="dist_to_drain_m"):
        extract_network(cfg)
    assert not resolve_path(cfg, "graph_file").exists()


@pytest.mark.integration
def test_extract_rejects_enricher_returning_wrong_type(cfg, monkeypatch):
    monkeypatch.setattr(network, "_load_enricher", lambda: (lambda G, c: None))
    with pytest.raises(NetworkStageError, match="DiGraph"):
        extract_network(cfg)


@pytest.mark.integration
def test_extract_validates_config_before_any_work(cfg, enricher):
    with pytest.raises(ConfigError):
        extract_network(with_overrides(cfg, {"network": {"min_nodes": 0}}))
    assert enricher.calls == 0


@pytest.mark.unit
def test_load_enricher_imports_module_b_or_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.data_pipeline.elevation", None)
    with pytest.raises(NetworkStageError, match="02_elevation_engine|elevation"):
        network._load_enricher()
    fake_module = SimpleNamespace(enrich_graph=fake_enrich)
    monkeypatch.setitem(sys.modules, "src.data_pipeline.elevation", fake_module)
    assert network._load_enricher() is fake_enrich
    monkeypatch.setitem(sys.modules, "src.data_pipeline.elevation", SimpleNamespace())
    with pytest.raises(NetworkStageError):
        network._load_enricher()


# --------------------------------------------------------------------------- summary & CLI


@pytest.mark.unit
def test_summarize_graph(cfg, grid_graph):
    G = grid_graph.copy()
    G.graph.update(source="synthetic_grid", elevation_source="synthetic", drain_source="synthetic_line",
                   bbox=json.dumps(list(BBOX)))
    summary = network.summarize_graph(G, Path("/tmp/x.graphml"))
    assert summary["nodes"] == 36 and summary["edges"] == G.number_of_edges()
    assert summary["source"] == "synthetic_grid"
    assert summary["elevation_min_m"] <= summary["elevation_max_m"]
    text = network.format_summary(summary)
    assert "Nodes" in text and "/tmp/x.graphml" in text and "synthetic" in text
    bare = nx.DiGraph()
    bare.add_node(1, x=77.66, y=12.93)
    assert network.summarize_graph(bare, Path("g.graphml"))["elevation_source"] == "unknown"


@pytest.mark.integration
def test_cli_main_success(cfg, enricher, tmp_path, capsys):
    config_path = write_config(cfg, tmp_path / "cfg.yaml")
    code = load_cli().main(["--config", str(config_path), "--offline", "--force"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Nodes" in out and "36" in out and "synthetic_grid" in out
    assert str(resolve_path(cfg, "graph_file")) in out


@pytest.mark.integration
def test_cli_main_handles_missing_config(tmp_path, capsys):
    code = load_cli().main(["--config", str(tmp_path / "missing.yaml")])
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("ERROR:") and len(err.strip().splitlines()) == 1


@pytest.mark.integration
def test_cli_main_handles_stage_errors(cfg, tmp_path, monkeypatch, capsys):
    config_path = write_config(cfg, tmp_path / "cfg.yaml")

    def boom(c, force=False, allow_synthetic=False):
        raise NetworkStageError("enrich_graph is unavailable")

    cli = load_cli()
    monkeypatch.setattr(network, "extract_network", boom)
    assert cli.main(["--config", str(config_path)]) == 1
    assert "enrich_graph" in capsys.readouterr().err

    def unexpected(c, force=False, allow_synthetic=False):
        raise KeyError("surprise")

    monkeypatch.setattr(network, "extract_network", unexpected)
    assert cli.main(["--config", str(config_path)]) == 1
    assert "surprise" in capsys.readouterr().err

    def interrupted(c, force=False, allow_synthetic=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(network, "extract_network", interrupted)
    assert cli.main(["--config", str(config_path)]) == 130


@pytest.mark.integration
def test_cli_offline_flag_forces_offline(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    config_path = write_config(with_overrides(cfg, {"project": {"offline": False}}), tmp_path / "cfg.yaml")
    seen = {}

    def capture(c, force=False, allow_synthetic=False):
        seen.update(offline=c["project"]["offline"], force=force, allow_synthetic=allow_synthetic)
        return synthetic_grid_graph(c)

    monkeypatch.setattr(network, "extract_network", capture)
    assert load_cli().main(["--config", str(config_path), "--offline"]) == 0
    assert seen == {"offline": True, "force": False, "allow_synthetic": False}
    assert load_cli().main(["--config", str(config_path), "--offline", "--force", "--allow-synthetic"]) == 0
    assert seen == {"offline": True, "force": True, "allow_synthetic": True}


@pytest.mark.e2e
def test_cli_script_runs_from_project_root(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--force" in result.stdout and "--offline" in result.stdout
    bad = subprocess.run([sys.executable, str(SCRIPT), "--config", str(tmp_path / "nope.yaml")],
                         cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert bad.returncode == 1
    assert "Traceback" not in bad.stderr
    assert "ERROR:" in bad.stderr


# --------------------------------------------------------------------------- real network (opt-in)


@pytest.mark.network
def test_real_osm_bbox_fetch(cfg):
    real = with_overrides(cfg, {"project": {"offline": False}})
    raw, source = fetch_osm_graph(real)
    G = clean_graph(raw, real)
    assert source in {"osm_place", "osm_bbox"}
    assert 300 < G.number_of_nodes() < 5000
    assert nx.is_weakly_connected(G)
