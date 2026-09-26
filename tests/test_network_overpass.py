"""Bounded Overpass retries (F5-01) and the recorded OpenStreetMap snapshot (F5-07).

Two local HTTP servers stand in for the Overpass mirrors of ``network.overpass_urls``:
mirror A answers every query with HTTP 504 (a busy server) and mirror B answers with a small
valid Overpass JSON document. The REAL osmnx request code runs against them (osmnx itself
retries 429/504 forever); the stages must give mirror A up after
``network.overpass_max_attempts`` and move to mirror B within a bounded time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from src.data_pipeline import drains, network
from src.data_pipeline.drains import load_cached_waterways
from src.data_pipeline.network import extract_network, fetch_cached_osm_graph, fetch_osm_graph, format_summary
from src.data_pipeline.overpass_guard import (
    OverpassLog,
    RetryPolicy,
    ServerBusy,
    bounded_osmnx_requests,
)
from src.utils.config import ConfigError, deep_merge, resolve_path
from tests.test_network import EnricherSpy

OSM_BASE = "2026-09-23T12:50:49Z"
STATUS_TEXT = ("Connected as: 1\nCurrent time: 2026-09-24T00:00:00Z\nAnnounced endpoint: none\n"
               "Rate limit: 2\n2 slots available now.\nCurrently running queries:\n")
LONS = [77.662, 77.670, 77.678, 77.686]
LATS = [12.922, 12.930, 12.938, 12.946]
TEST_BUDGET_S = 15.0


# --------------------------------------------------------------------------- fake Overpass mirrors
def _road_elements() -> list[dict]:
    """A 4 x 4 street grid (one OSM way per row and per column) inside region.bbox."""
    nodes = [{"type": "node", "id": 1 + r * 4 + c, "lat": LATS[r], "lon": LONS[c]} for r in range(4) for c in range(4)]
    rows = [{"type": "way", "id": 101 + r, "nodes": [1 + r * 4 + c for c in range(4)],
             "tags": {"highway": "residential", "name": f"Row {r}"}} for r in range(4)]
    cols = [{"type": "way", "id": 201 + c, "nodes": [1 + r * 4 + c for r in range(4)],
             "tags": {"highway": "residential", "name": f"Column {c}"}} for c in range(4)]
    return nodes + rows + cols


def _drain_elements() -> list[dict]:
    nodes = [{"type": "node", "id": 900 + i, "lat": 12.918 + 0.01 * i, "lon": 77.675} for i in range(4)]
    return nodes + [{"type": "way", "id": 990, "nodes": [900, 901, 902, 903],
                     "tags": {"waterway": "drain", "name": "Test rajakaluve"}}]


class _Mirror:
    """A local Overpass stand-in: ``busy`` answers 504 to every query, otherwise valid JSON."""

    def __init__(self, busy: bool) -> None:
        self.busy = busy
        self.hits: list[float] = []
        self.started = time.monotonic()
        mirror = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep pytest output clean
                return

            def _send(self, code: int, body: bytes, kind: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self._send(200, STATUS_TEXT.encode(), "text/plain")

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                query = parse_qs(self.rfile.read(length).decode()).get("data", [""])[0]
                mirror.hits.append(round(time.monotonic() - mirror.started, 2))
                if mirror.busy:
                    self._send(504, b"<html>Gateway Timeout</html>", "text/html")
                    return
                elements = _drain_elements() if "waterway" in query else _road_elements()
                doc = {"version": 0.6, "generator": "fake-overpass",
                       "osm3s": {"timestamp_osm_base": OSM_BASE, "copyright": "test"}, "elements": elements}
                self._send(200, json.dumps(doc).encode(), "application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def mirrors():
    busy, good = _Mirror(busy=True), _Mirror(busy=False)
    yield busy, good
    busy.close()
    good.close()


@pytest.fixture
def online_cfg(cfg, monkeypatch, mirrors):
    """Online config whose Overpass mirrors are the two local servers (busy first)."""
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    busy, good = mirrors
    return deep_merge(cfg, {
        "project": {"offline": False},
        "region": {"use_place_query": False},
        "network": {"overpass_urls": [busy.url, good.url], "overpass_max_attempts": 2,
                    "overpass_retry_pause_s": 0.2, "request_timeout_s": 20},
        "drains": {"request_timeout_s": 20},
    })


@pytest.fixture
def log(caplog):
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


# --------------------------------------------------------------------------- F5-01: bounded retries
@pytest.mark.integration
def test_busy_mirror_is_given_up_and_the_next_mirror_is_used(online_cfg, mirrors, log):
    busy, good = mirrors
    started = time.monotonic()
    G, source = fetch_osm_graph(online_cfg)
    elapsed = time.monotonic() - started
    assert elapsed < TEST_BUDGET_S, f"stage did not move to mirror B within a bounded time ({elapsed:.1f} s)"
    assert len(busy.hits) == 2, f"mirror A must be tried exactly overpass_max_attempts times, got {busy.hits}"
    assert len(good.hits) == 1 and source == "osm_bbox" and G.number_of_nodes() >= 4
    assert G.graph["overpass_endpoint"] == good.url
    text = log.text
    assert "Overpass mirror 1 of 2" in text and "Overpass mirror 2 of 2" in text
    assert "HTTP 504" in text and "retrying" in text and "gave up" in text


@pytest.mark.integration
def test_waterway_query_skips_the_busy_mirror_and_records_the_snapshot(online_cfg, mirrors, log):
    busy, good = mirrors
    started = time.monotonic()
    frame = drains.fetch_waterways(online_cfg, (77.64, 12.90, 77.71, 12.96))
    assert time.monotonic() - started < TEST_BUDGET_S
    assert len(busy.hits) == 2 and len(good.hits) == 1 and len(frame) == 1
    path = resolve_path(online_cfg, "waterways_file")
    meta = json.loads(path.read_text())["namma_flow"]
    assert meta["osm_base_utc"] == OSM_BASE
    cached, covers = load_cached_waterways(path, (77.65, 12.91, 77.70, 12.95), online_cfg["drains"]["osm_tags"])
    assert covers and len(cached) == 1


@pytest.mark.integration
def test_every_mirror_busy_raises_network_unavailable_quickly(online_cfg, mirrors):
    busy, _ = mirrors
    only_busy = deep_merge(online_cfg, {"network": {"overpass_urls": [busy.url], "overpass_max_attempts": 3}})
    started = time.monotonic()
    with pytest.raises(network.NetworkUnavailable, match="every Overpass endpoint"):
        fetch_osm_graph(only_busy)
    assert time.monotonic() - started < TEST_BUDGET_S and len(busy.hits) == 3


@pytest.mark.integration
def test_osmnx_globals_are_restored_after_the_stage(online_cfg):
    import osmnx

    before = (osmnx._overpass._overpass_request, osmnx._overpass.requests,
              osmnx._nominatim._nominatim_request, osmnx._nominatim.requests, osmnx.settings.overpass_url)
    fetch_osm_graph(online_cfg)
    after = (osmnx._overpass._overpass_request, osmnx._overpass.requests,
             osmnx._nominatim._nominatim_request, osmnx._nominatim.requests, osmnx.settings.overpass_url)
    assert after == before


@pytest.mark.unit
def test_osmnx_style_recursive_retry_is_bounded_without_the_requests_proxy():
    """If osmnx's own 429/504 recursion is reached (proxy bypassed), the wrapper still stops it."""
    calls: list[int] = []
    module = SimpleNamespace()

    def recursive_request(data):  # mimics osmnx: on a busy answer, pause and call the module global again
        calls.append(1)
        return module._overpass_request(data)

    module._overpass_request = recursive_request
    fake_ox = SimpleNamespace(_overpass=module, settings=SimpleNamespace(overpass_url="http://busy.test/api"))
    with bounded_osmnx_requests(fake_ox, RetryPolicy(max_attempts=2, retry_pause_s=0.0)):
        with pytest.raises(ServerBusy, match="gave up"):
            module._overpass_request({"data": "q"})
    assert len(calls) == 2
    assert module._overpass_request is recursive_request, "the original request function must be restored"


@pytest.mark.unit
def test_guard_records_snapshots_and_is_a_noop_for_fakes_without_private_api():
    fake_ox = SimpleNamespace(settings=SimpleNamespace(overpass_url="x"))
    with bounded_osmnx_requests(fake_ox, RetryPolicy()) as log:
        assert isinstance(log, OverpassLog) and log.latest_osm_base is None
    log = OverpassLog()
    log.record("a", {"osm3s": {"timestamp_osm_base": "2026-01-01T00:00:00Z"}})
    log.record("b", {"osm3s": {"timestamp_osm_base": OSM_BASE}})
    log.record("b", {"elements": []})
    assert log.latest_osm_base == OSM_BASE and log.endpoints == ["a", "b"]


@pytest.mark.unit
@pytest.mark.parametrize("overrides, match", [
    ({"overpass_max_attempts": 0}, "overpass_max_attempts"),
    ({"overpass_max_attempts": True}, "overpass_max_attempts"),
    ({"overpass_retry_pause_s": -1}, "overpass_retry_pause_s"),
    ({"overpass_retry_pause_s": "soon"}, "overpass_retry_pause_s"),
    ({"osm_date": "yesterday"}, "osm_date"),
])
def test_retry_and_snapshot_settings_are_validated(cfg, overrides, match):
    with pytest.raises(ConfigError, match=match):
        network.NetworkSettings.from_config(deep_merge(cfg, {"network": overrides}))


# --------------------------------------------------------------------------- F5-07: OSM snapshot
@pytest.mark.integration
def test_stage01_records_the_osm_snapshot_in_graph_and_summary(online_cfg, mirrors, monkeypatch):
    monkeypatch.setattr(network, "_load_enricher", lambda: EnricherSpy())
    G = extract_network(deep_merge(online_cfg, {"network": {"min_nodes": 4}}))
    assert G.graph["source"] == "osm_bbox"
    assert G.graph["osm_base_utc"] == OSM_BASE and G.graph["osm_query_utc"].endswith("+00:00")
    summary = network.summarize_graph(G, resolve_path(online_cfg, "graph_file"))
    assert summary["osm_base_utc"] == OSM_BASE
    assert "OSM snapshot (UTC)" in format_summary(summary) and OSM_BASE in format_summary(summary)


@pytest.mark.integration
def test_offline_replay_recovers_the_snapshot_from_the_osmnx_cache(online_cfg, mirrors, monkeypatch):
    fetch_osm_graph(online_cfg)  # caches mirror B's answer in paths.osm_cache_dir
    busy, good = mirrors
    hits = (len(busy.hits), len(good.hits))
    monkeypatch.setenv("NAMMA_FLOW_OFFLINE", "1")
    G, source = fetch_cached_osm_graph(deep_merge(online_cfg, {"project": {"offline": True}}))
    assert (len(busy.hits), len(good.hits)) == hits, "offline replay must not contact any server"
    assert source == "osm_bbox" and G.graph["osm_base_utc"] == OSM_BASE
    assert "osm_query_utc" not in G.graph and "overpass_endpoint" not in G.graph


@pytest.mark.unit
def test_osm_date_pins_the_query_and_counts_in_the_network_hash(cfg, monkeypatch):
    from tests.test_network import FakeOsmnx, install_fake_osmnx

    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    pinned = deep_merge(cfg, {"project": {"offline": False}, "region": {"use_place_query": False},
                              "network": {"osm_date": OSM_BASE}})
    fake = install_fake_osmnx(monkeypatch, FakeOsmnx())
    fake.settings.overpass_settings = "[out:json][timeout:{timeout}]{maxsize}"
    G, _ = fetch_osm_graph(pinned)
    seen = fake.settings_seen[-1]["overpass_settings"]
    assert seen.endswith(f'[date:"{OSM_BASE}"]') and G.graph["osm_date"] == OSM_BASE
    assert fake.settings.overpass_settings == "[out:json][timeout:{timeout}]{maxsize}", "setting restored"
    assert network.network_config_hash(pinned) != network.network_config_hash(cfg)
    fetch_only = deep_merge(cfg, {"network": {"overpass_max_attempts": 5, "overpass_retry_pause_s": 1}})
    assert network.network_config_hash(fetch_only) == network.network_config_hash(cfg)


@pytest.mark.unit
def test_busy_nominatim_geocoder_is_bounded_too():
    """The place query geocodes through Nominatim, which osmnx also retries forever on 429/504."""
    sleeps: list[float] = []
    real_requests = SimpleNamespace(get=lambda url, **kw: SimpleNamespace(status_code=429, reason="Too Many Requests"))
    module = SimpleNamespace(requests=real_requests)

    def nominatim_request(params, request_type="search"):
        return module.requests.get("https://nominatim.test/search", params=params)

    module._nominatim_request = nominatim_request
    fake_ox = SimpleNamespace(_nominatim=module, settings=SimpleNamespace(nominatim_url="https://nominatim.test/"))
    with bounded_osmnx_requests(fake_ox, RetryPolicy(max_attempts=3, retry_pause_s=1.5), sleep=sleeps.append):
        with pytest.raises(ServerBusy, match="HTTP 429.*gave up .* after 3 attempt"):
            module._nominatim_request({"q": "Bellandur"})
    assert sleeps == [1.5, 1.5]
    assert module.requests is real_requests and module._nominatim_request is nominatim_request
