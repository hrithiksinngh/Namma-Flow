"""Offline rebuild of the OSM road graph from the osmnx cache, and no silent downgrade to the
synthetic grid (review finding R1-03). Helpers and fakes are shared with ``tests/test_network.py``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data_pipeline import network
from src.data_pipeline.network import NetworkSettings, NetworkStageError, RegionSettings, extract_network
from src.utils.config import resolve_path
from src.utils.http import NetworkUnavailable
from tests.test_network import (  # noqa: F401 - enricher is a fixture
    BBOX,
    FakeOsmnx,
    enricher,
    install_fake_osmnx,
    load_cli,
    online,
    with_overrides,
    write_config,
)


# --------------------------------------------------------------------------- offline rebuild / no silent downgrade
class CachedFakeOsmnx(FakeOsmnx):
    """Fake osmnx whose Overpass responses come from a 'cache'; a miss goes through ``_http._config_dns``
    first, exactly like osmnx 2.x (which is where the cache-only guard intercepts it)."""

    def __init__(self, cached_urls=(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.cached_urls = set(cached_urls)
        self.network_hits: list[str] = []
        self._http = SimpleNamespace(_config_dns=lambda url: self.network_hits.append(url))

    def graph_from_bbox(self, bbox, **kwargs):
        if self.settings.overpass_url not in self.cached_urls:
            self._http._config_dns(self.settings.overpass_url)
        return super().graph_from_bbox(bbox, **kwargs)


def seed_osm_cache(cfg: dict) -> Path:
    cache_dir = resolve_path(cfg, "osm_cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "0123abcd.json").write_text("{}")
    return cache_dir


def save_osm_graph(cfg: dict, enricher, monkeypatch) -> bytes:
    """Build the 'real' OSM graph online (fake osmnx) and return the saved file's bytes."""
    install_fake_osmnx(monkeypatch, FakeOsmnx(place_error=ValueError("lake")))
    G = extract_network(online(cfg, monkeypatch))
    assert G.graph["source"] == "osm_bbox"
    monkeypatch.setenv("NAMMA_FLOW_OFFLINE", "1")
    return resolve_path(cfg, "graph_file").read_bytes()


@pytest.mark.integration
def test_offline_force_rebuilds_the_osm_graph_from_the_osmnx_cache(cfg, enricher, monkeypatch):
    """R1-03: ``01 --force --offline`` replays the cached Overpass response instead of a synthetic grid."""
    save_osm_graph(cfg, enricher, monkeypatch)
    seed_osm_cache(cfg)
    primary = cfg["network"]["overpass_urls"][0]
    fake = install_fake_osmnx(monkeypatch, CachedFakeOsmnx(cached_urls={primary}))
    G = extract_network(cfg, force=True)
    assert G.graph["source"] == "osm_bbox" and G.number_of_nodes() == 36
    assert fake.network_hits == [] and not fake.called("geocode_to_gdf"), "offline: no network, no geocoder"
    assert callable(fake._http._config_dns) and fake._http._config_dns("x") is None, "hook restored"


@pytest.mark.integration
def test_offline_force_refuses_to_replace_the_osm_graph(cfg, enricher, monkeypatch):
    """R1-03: without usable cached responses the real graph is kept unless --allow-synthetic."""
    before = save_osm_graph(cfg, enricher, monkeypatch)
    path = resolve_path(cfg, "graph_file")
    with pytest.raises(NetworkStageError, match="allow-synthetic"):
        extract_network(cfg, force=True)
    assert path.read_bytes() == before
    seed_osm_cache(cfg)
    fake = install_fake_osmnx(monkeypatch, CachedFakeOsmnx(cached_urls=()))
    with pytest.raises(NetworkStageError, match="not in the osmnx cache"):
        extract_network(cfg, force=True)
    assert path.read_bytes() == before and fake.network_hits == []
    G = extract_network(cfg, force=True, allow_synthetic=True)
    assert G.graph["source"] == "synthetic_grid"


@pytest.mark.integration
def test_online_outage_never_silently_replaces_the_osm_graph(cfg, enricher, monkeypatch):
    before = save_osm_graph(cfg, enricher, monkeypatch)

    def unavailable(c):
        raise NetworkUnavailable("Overpass timed out")

    monkeypatch.setattr(network, "fetch_osm_graph", unavailable)
    with pytest.raises(NetworkStageError, match="Overpass timed out"):
        extract_network(online(cfg, monkeypatch), force=True)
    assert resolve_path(cfg, "graph_file").read_bytes() == before


@pytest.mark.e2e
def test_cli_offline_force_exits_1_and_keeps_the_osm_graph(cfg, enricher, monkeypatch, tmp_path, capsys):
    before = save_osm_graph(cfg, enricher, monkeypatch)
    config_path = write_config(cfg, tmp_path / "cfg.yaml")
    assert load_cli().main(["--config", str(config_path), "--offline", "--force"]) == 1
    assert "Refusing to replace" in capsys.readouterr().err
    assert resolve_path(cfg, "graph_file").read_bytes() == before


class _FakeOverpassResponse:
    def __init__(self, payload: dict, url: str) -> None:
        self._payload, self.url = payload, url
        self.content = json.dumps(payload).encode()
        self.text, self.status_code, self.reason, self.ok = self.content.decode(), 200, "OK", True

    def json(self) -> dict:
        return self._payload


def _overpass_grid(rows: int = 4, cols: int = 4) -> dict:
    west, south, east, north = BBOX
    nid = lambda r, c: 1000 + r * cols + c  # noqa: E731
    nodes = [{"type": "node", "id": nid(r, c), "lat": south + (r + 1) * (north - south) / (rows + 1),
              "lon": west + (c + 1) * (east - west) / (cols + 1)} for r in range(rows) for c in range(cols)]
    ways = [{"type": "way", "id": 1 + r, "nodes": [nid(r, c) for c in range(cols)],
             "tags": {"highway": "residential"}} for r in range(rows)]
    ways += [{"type": "way", "id": 100 + c, "nodes": [nid(r, c) for r in range(rows)],
              "tags": {"highway": "tertiary"}} for c in range(cols)]
    return {"version": 0.6, "generator": "test", "elements": nodes + ways}


@pytest.mark.integration
def test_real_osmnx_cache_is_replayed_offline_without_network(cfg, monkeypatch):
    """R1-03 with the real osmnx: a response cached by an online bbox query is replayed offline."""
    import osmnx as ox
    import requests

    online_cfg = online(cfg, monkeypatch)
    region, net = RegionSettings.from_config(online_cfg), NetworkSettings.from_config(online_cfg)
    cache_dir = resolve_path(online_cfg, "osm_cache_dir")
    monkeypatch.setattr(requests, "post", lambda url, data=None, **k: _FakeOverpassResponse(_overpass_grid(), url))
    monkeypatch.setattr(ox._http, "_config_dns", lambda url: None)
    monkeypatch.setattr(ox._overpass, "_get_overpass_pause", lambda *a, **k: 0)
    with network._osmnx_settings(ox, cache_dir, net):
        seeded = network._fetch_bbox_graph(ox, region, net)
    assert list(cache_dir.glob("*.json"))
    monkeypatch.undo()
    monkeypatch.setenv("NAMMA_FLOW_OFFLINE", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("offline replay touched the network")

    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.setattr(requests, "get", no_network)
    raw, source = network.fetch_cached_osm_graph(cfg)
    assert source == "osm_bbox" and raw.number_of_nodes() == seeded.number_of_nodes() > 0
    original_hook = ox._http._config_dns
    assert original_hook.__name__ == "_config_dns", "the osmnx DNS hook is restored after the replay"
    other = with_overrides(cfg, {"network": {"request_timeout_s": 99}})  # different query text -> cache miss
    with pytest.raises(NetworkUnavailable, match="not in the osmnx cache"):
        network.fetch_cached_osm_graph(other)
