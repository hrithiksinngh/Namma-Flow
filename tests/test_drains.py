"""Tests for ``src.data_pipeline.drains`` (OSM waterways, distance-to-drain, fallbacks)."""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from src.data_pipeline import drains
from src.data_pipeline.drains import (
    DrainResult,
    clean_waterways,
    compute_drain_distances,
    distance_to_drains,
    drain_distances,
    expand_bbox,
    fetch_waterways,
    load_cached_waterways,
    nodes_bbox,
    save_waterways_cache,
    synthetic_drain_distance,
    validate_coords,
)
from src.utils.config import ConfigError, deep_merge
from src.utils.geo import haversine_m
from src.utils.http import NetworkUnavailable

BBOX = (77.655, 12.915, 77.700, 12.950)
DRAIN_LON = 77.675


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture
def log(caplog):
    """Capture records from the project logger (it does not propagate to root)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


@pytest.fixture
def online_cfg(cfg, monkeypatch):
    """Config in online mode, with any real HTTP request turned into a test failure."""
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)

    def _blocked(*_args, **_kwargs):
        raise AssertionError("a unit test attempted a real network request")

    monkeypatch.setattr(requests.Session, "request", _blocked)
    return deep_merge(cfg, {"project": {"offline": False}})


def _osm_like_frame(geometries, **columns) -> gpd.GeoDataFrame:
    """GeoDataFrame shaped like ``osmnx.features_from_bbox`` output (MultiIndex element/id)."""
    index = pd.MultiIndex.from_tuples([("way", 100 + i) for i in range(len(geometries))], names=["element", "id"])
    return gpd.GeoDataFrame(columns, geometry=list(geometries), crs="EPSG:4326", index=index)


def _drain_line_frame() -> gpd.GeoDataFrame:
    line = LineString([(DRAIN_LON, 12.90), (DRAIN_LON, 12.97)])
    return _osm_like_frame([line], waterway=["drain"], name=["Test rajakaluve"])


def _lake_polygon() -> Polygon:
    return Polygon([(77.660, 12.925), (77.670, 12.925), (77.670, 12.935), (77.660, 12.935)])


# --------------------------------------------------------------------------- coordinate helpers
@pytest.mark.unit
def test_validate_coords_accepts_scalars_and_returns_copies():
    lon_in = np.array([77.66, 77.67])
    lon, lat = validate_coords(lon_in, [12.92, 12.93])
    assert lon.dtype == np.float64 and lat.shape == (2,)
    lon[0] = 0.0
    assert lon_in[0] == 77.66  # caller's array untouched
    lon_s, lat_s = validate_coords(77.66, 12.92)
    assert lon_s.shape == (1,) and lat_s.shape == (1,)
    empty_lon, empty_lat = validate_coords([], [])
    assert empty_lon.size == 0 and empty_lat.size == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "lon, lat, message",
    [
        ([77.6, 77.7], [12.9], "same shape"),
        ([77.6, np.nan], [12.9, 12.9], "finite"),
        ([200.0], [12.9], "longitude"),
        ([77.6], [95.0], "latitude"),
        (["a"], [12.9], "numeric"),
        ([[77.6, 77.7]], [[12.9, 12.9]], "1-D"),
    ],
)
def test_validate_coords_rejects_bad_input(lon, lat, message):
    with pytest.raises(ValueError, match=message):
        validate_coords(lon, lat)


@pytest.mark.unit
def test_nodes_bbox_prefers_region_bbox_when_it_contains_all_nodes(cfg):
    lon = np.array([77.66, 77.69])
    lat = np.array([12.92, 12.94])
    assert nodes_bbox(lon, lat, cfg) == pytest.approx(BBOX)


@pytest.mark.unit
def test_nodes_bbox_uses_padded_node_extent_otherwise(cfg):
    lon = np.array([77.60, 77.61])
    lat = np.array([12.90, 12.91])
    west, south, east, north = nodes_bbox(lon, lat, cfg)
    assert west < 77.60 and east > 77.61 and south < 12.90 and north > 12.91
    # A single node still yields a non-degenerate bbox.
    w1, s1, e1, n1 = nodes_bbox(np.array([77.6]), np.array([12.9]), {"region": {}})
    assert w1 < e1 and s1 < n1
    with pytest.raises(ValueError):
        nodes_bbox(np.array([]), np.array([]), cfg)


@pytest.mark.unit
def test_expand_bbox_clamps_to_world():
    assert expand_bbox(BBOX, 0.01) == pytest.approx((77.645, 12.905, 77.71, 12.96))
    assert expand_bbox((179.99, 89.99, 180.0, 90.0), 0.5)[2:] == (180.0, 90.0)
    with pytest.raises(ValueError):
        expand_bbox(BBOX, -1.0)


# --------------------------------------------------------------------------- synthetic fallback
@pytest.mark.unit
def test_synthetic_drain_distance_is_east_west_metres(cfg):
    lon = np.array([DRAIN_LON, DRAIN_LON + 0.01, DRAIN_LON - 0.02])
    lat = np.full(3, 12.93)
    dist = synthetic_drain_distance(lon, lat, cfg)
    assert dist[0] == pytest.approx(0.0, abs=1e-6)
    expected = 0.01 * np.pi / 180.0 * 6_371_008.8 * np.cos(np.radians(12.93))
    assert dist[1] == pytest.approx(expected, rel=1e-3)
    assert dist[2] == pytest.approx(2 * expected, rel=1e-3)
    assert np.all(dist >= 0)


@pytest.mark.unit
def test_synthetic_drain_distance_clips_and_defaults(cfg):
    far = synthetic_drain_distance(np.array([78.5]), np.array([12.93]), cfg)
    assert far[0] == pytest.approx(cfg["drains"]["max_distance_m"])
    no_line = deep_merge(cfg, {"region": {"drain_fallback_lon": None}})
    centre_lon = (BBOX[0] + BBOX[2]) / 2
    assert synthetic_drain_distance(np.array([centre_lon]), np.array([12.93]), no_line)[0] == pytest.approx(0.0)
    with pytest.raises(ConfigError):
        bad_line = deep_merge(cfg, {"region": {"drain_fallback_lon": "x"}})
        synthetic_drain_distance(np.array([77.6]), np.array([12.9]), bad_line)
    assert synthetic_drain_distance(np.array([]), np.array([]), cfg).size == 0


# --------------------------------------------------------------------------- geometry cleaning
@pytest.mark.unit
def test_clean_waterways_filters_repairs_and_flattens():
    bowtie = Polygon([(77.660, 12.920), (77.670, 12.930), (77.670, 12.920), (77.660, 12.930)])
    tiny_pond = Polygon([(77.690, 12.940), (77.69001, 12.940), (77.69001, 12.94001), (77.690, 12.94001)])
    raw = _osm_like_frame(
        [
            Point(77.66, 12.92),
            LineString([(77.66, 12.92), (77.67, 12.93)]),
            MultiLineString([[(77.68, 12.92), (77.68, 12.93)], [(77.685, 12.92), (77.685, 12.93)]]),
            bowtie,
            tiny_pond,
            None,
        ],
        waterway=[None, "drain", ["stream", "canal"], None, None, None],
        natural=[None, None, None, "water", "water", None],
    )
    assert not raw.geometry.iloc[3].is_valid
    clean = clean_waterways(raw, min_polygon_area_m2=1000.0)
    assert clean.crs.to_epsg() == 4326
    assert set(clean.geom_type) <= {"LineString", "Polygon"}
    assert clean.geometry.is_valid.all()
    assert (clean["kind"] == "line").sum() == 3  # one line + two exploded multiline parts
    assert (clean["kind"] == "polygon").sum() == 2  # bowtie repaired into two triangles, pond dropped
    assert "stream;canal" in set(clean["waterway"].dropna())
    assert {"element", "osmid"} <= set(clean.columns)
    assert clean.index.is_unique


@pytest.mark.unit
def test_clean_waterways_handles_empty_missing_crs_and_other_crs():
    assert len(clean_waterways(None)) == 0
    assert len(clean_waterways(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))) == 0
    no_crs = gpd.GeoDataFrame(geometry=[LineString([(77.66, 12.92), (77.67, 12.93)])])
    assert clean_waterways(no_crs).crs.to_epsg() == 4326
    utm = gpd.GeoDataFrame(geometry=[LineString([(77.66, 12.92), (77.67, 12.93)])], crs="EPSG:4326").to_crs(32643)
    back = clean_waterways(utm)
    assert back.crs.to_epsg() == 4326
    assert back.geometry.iloc[0].coords[0][0] == pytest.approx(77.66, abs=1e-7)
    # Idempotent, and tolerant of index level names that are also columns.
    once = clean_waterways(_drain_line_frame())
    assert clean_waterways(once).equals(once)
    clash = _drain_line_frame()
    clash["element"] = "way"
    assert len(clean_waterways(clash)) == 1


@pytest.mark.unit
def test_clean_waterways_keeps_lake_parts_of_nested_collections(cfg):
    """R1-06: make_valid -> GEOMETRYCOLLECTION(MULTIPOLYGON, LINESTRING) must keep the polygons."""
    import shapely

    spiked_bowtie = Polygon([
        (77.660, 12.920), (77.670, 12.930), (77.670, 12.920), (77.660, 12.930), (77.660, 12.925),
        (77.655, 12.925), (77.660, 12.925),
    ])
    repaired = shapely.make_valid(spiked_bowtie)
    assert repaired.geom_type == "GeometryCollection"
    assert "MultiPolygon" in {g.geom_type for g in repaired.geoms}
    clean = clean_waterways(_osm_like_frame([spiked_bowtie], natural=["water"]))
    assert (clean["kind"] == "polygon").sum() == 2, list(clean.geom_type)
    inside_lobe = distance_to_drains(np.array([77.6625]), np.array([12.925]), clean, cfg)
    assert inside_lobe[0] == 0.0


@pytest.mark.unit
def test_wastewater_tanks_are_not_storm_drains(cfg):
    """R1-08: water=wastewater (treatment tanks) and pools do not count as drains."""
    tank = Polygon([(77.6800, 12.9300), (77.6810, 12.9300), (77.6810, 12.9310), (77.6800, 12.9310)])
    frame = _osm_like_frame(
        [tank, LineString([(DRAIN_LON, 12.90), (DRAIN_LON, 12.97)])],
        natural=["water", None], water=["wastewater", None], waterway=[None, "drain"],
    )
    node = (np.array([77.6805]), np.array([12.9305]))
    with_tank = distance_to_drains(*node, frame, deep_merge(cfg, {"drains": {"exclude_water_values": []}}))
    without = distance_to_drains(*node, frame, cfg)
    assert with_tank[0] == 0.0
    assert without[0] == pytest.approx(synthetic_drain_distance(*node, cfg)[0], rel=5e-3)
    lon, lat = _grid_nodes()
    query = expand_bbox(nodes_bbox(lon, lat, cfg), cfg["drains"]["bbox_margin_deg"])
    save_waterways_cache(clean_waterways(frame), drains.resolve_path(cfg, "waterways_file"), query,
                         cfg["drains"]["osm_tags"])
    result = compute_drain_distances(lon, lat, cfg)
    assert result.source == "osm" and result.n_features == 1
    with pytest.raises(ConfigError, match="exclude_water_values"):
        distance_to_drains(*node, frame, deep_merge(cfg, {"drains": {"exclude_water_values": 5}}))


@pytest.mark.unit
def test_user_osm_tags_replace_the_default_filter(online_cfg, monkeypatch):
    """R1-07: narrowing drains.osm_tags must not silently re-add the default natural=water key."""
    import osmnx

    seen = []
    monkeypatch.setattr(osmnx, "features_from_bbox", lambda bbox, tags: seen.append(tags) or _drain_line_frame())
    narrowed = {**online_cfg, "drains": {**online_cfg["drains"], "osm_tags": {"waterway": ["drain", "canal"]}}}
    fetch_waterways(narrowed, BBOX)
    assert seen == [{"waterway": ["drain", "canal"]}]
    without_section = {k: v for k, v in online_cfg.items() if k != "drains"}
    fetch_waterways(without_section, BBOX)
    assert seen[-1] == drains.DEFAULTS["osm_tags"]


# --------------------------------------------------------------------------- distances
@pytest.mark.unit
def test_distance_to_line_matches_synthetic_line(cfg):
    lon = np.array([DRAIN_LON, DRAIN_LON + 0.005, DRAIN_LON - 0.015])
    lat = np.array([12.93, 12.93, 12.94])
    dist = distance_to_drains(lon, lat, _drain_line_frame(), cfg)
    np.testing.assert_allclose(dist, synthetic_drain_distance(lon, lat, cfg), rtol=5e-3, atol=0.5)


@pytest.mark.unit
def test_distance_to_polygon_is_zero_inside_and_boundary_distance_outside(cfg):
    lake = gpd.GeoDataFrame({"natural": ["water"]}, geometry=[_lake_polygon()], crs="EPSG:4326")
    lon = np.array([77.665, 77.675, 77.665])
    lat = np.array([12.930, 12.930, 12.9345])
    dist = distance_to_drains(lon, lat, lake, cfg)
    assert dist[0] == 0.0  # inside the lake
    assert dist[1] == pytest.approx(haversine_m(77.670, 12.930, 77.675, 12.930), rel=5e-3)
    assert dist[2] == 0.0  # inside, close to the boundary


@pytest.mark.unit
def test_distance_to_drains_clips_and_handles_crs_and_empty(cfg, log):
    lines = _drain_line_frame()
    far = distance_to_drains(np.array([77.9]), np.array([12.93]), lines, cfg)
    assert far[0] == pytest.approx(cfg["drains"]["max_distance_m"])
    near = distance_to_drains(np.array([DRAIN_LON + 0.001]), np.array([12.93]), lines.to_crs(32643), cfg)
    assert near[0] == pytest.approx(108.4, rel=0.01)
    no_crs = gpd.GeoDataFrame(geometry=list(lines.geometry))
    assert distance_to_drains(np.array([DRAIN_LON]), np.array([12.93]), no_crs, cfg)[0] < 1.0
    assert "no CRS" in log.text
    assert distance_to_drains(np.array([]), np.array([]), lines, cfg).size == 0
    with pytest.raises(ValueError, match="no line or polygon"):
        distance_to_drains(np.array([77.66]), np.array([12.93]), gpd.GeoDataFrame(geometry=[], crs=4326), cfg)
    with pytest.raises(ValueError, match="no line or polygon"):
        distance_to_drains(np.array([77.66]), np.array([12.93]), None, cfg)
    points_only = gpd.GeoDataFrame(geometry=[Point(77.66, 12.93)], crs=4326)
    with pytest.raises(ValueError, match="no line or polygon"):
        distance_to_drains(np.array([77.66]), np.array([12.93]), points_only, cfg)
    with pytest.raises(ConfigError):
        zero_clip = deep_merge(cfg, {"drains": {"max_distance_m": 0}})
        distance_to_drains(np.array([77.66]), np.array([12.93]), lines, zero_clip)


# --------------------------------------------------------------------------- cache
@pytest.mark.unit
def test_cache_round_trip_and_coverage(tmp_path):
    path = tmp_path / "ww.geojson"
    clean = clean_waterways(_drain_line_frame())
    save_waterways_cache(clean, path, BBOX, {"waterway": ["drain"]})
    payload = json.loads(path.read_text())
    assert payload["type"] == "FeatureCollection"
    assert payload["namma_flow"]["query_bbox"] == list(BBOX)
    loaded, covers = load_cached_waterways(path, (77.66, 12.92, 77.69, 12.94), {"waterway": ["drain"]})
    assert covers and len(loaded) == 1 and loaded.crs.to_epsg() == 4326
    _, covers_bigger = load_cached_waterways(path, (77.60, 12.92, 77.69, 12.94), {"waterway": ["drain"]})
    assert not covers_bigger
    _, covers_other_tags = load_cached_waterways(path, BBOX, {"waterway": ["river"]})
    assert not covers_other_tags


@pytest.mark.unit
def test_cache_missing_corrupt_empty_and_foreign(tmp_path, log):
    assert load_cached_waterways(tmp_path / "missing.geojson") == (None, False)
    corrupt = tmp_path / "corrupt.geojson"
    corrupt.write_text("{not json")
    assert load_cached_waterways(corrupt) == (None, False)
    assert "unreadable" in log.text
    wrong_type = tmp_path / "wrong.geojson"
    wrong_type.write_text(json.dumps({"type": "Feature"}))
    assert load_cached_waterways(wrong_type) == (None, False)
    empty = tmp_path / "empty.geojson"
    save_waterways_cache(clean_waterways(None), empty, BBOX, {"natural": ["water"]})
    frame, covers = load_cached_waterways(empty, BBOX, {"natural": ["water"]})
    assert covers and len(frame) == 0
    foreign = tmp_path / "foreign.geojson"  # plain GeoJSON written by another tool: no metadata
    foreign.write_text(clean_waterways(_drain_line_frame()).to_json())
    frame, covers = load_cached_waterways(foreign, BBOX, {"natural": ["water"]})
    assert covers and len(frame) == 1


# --------------------------------------------------------------------------- OSM fetch (mocked)
@pytest.mark.unit
def test_fetch_waterways_offline_raises(cfg):
    with pytest.raises(NetworkUnavailable):
        fetch_waterways(cfg, BBOX)


@pytest.mark.unit
def test_fetch_waterways_filters_and_caches(online_cfg, monkeypatch):
    import osmnx

    calls = []
    previous_cache = osmnx.settings.cache_folder

    def fake_features(bbox, tags):
        calls.append((bbox, tags, osmnx.settings.cache_folder, osmnx.settings.requests_timeout))
        return _osm_like_frame(
            [LineString([(DRAIN_LON, 12.90), (DRAIN_LON, 12.97)]), Point(77.66, 12.93), _lake_polygon()],
            waterway=["drain", None, None],
            natural=[None, "water", "water"],
        )

    monkeypatch.setattr(osmnx, "features_from_bbox", fake_features)
    frame = fetch_waterways(online_cfg, BBOX)
    assert len(calls) == 1
    bbox, tags, cache_folder, timeout = calls[0]
    assert bbox == BBOX and tags["waterway"] == ["drain", "canal", "stream", "river", "ditch"]
    assert cache_folder.endswith("osm_cache") and timeout == online_cfg["drains"]["request_timeout_s"]
    assert osmnx.settings.cache_folder == previous_cache  # global osmnx settings restored
    assert sorted(frame["kind"]) == ["line", "polygon"]
    cache_path = drains.resolve_path(online_cfg, "waterways_file")
    cached, covers = load_cached_waterways(cache_path, BBOX, tags)
    assert covers and len(cached) == 2


@pytest.mark.unit
def test_fetch_waterways_empty_response_is_not_cached(online_cfg, monkeypatch):
    """R1-11: an empty Overpass answer may be transient: every mirror is tried and nothing is cached."""
    import osmnx
    from osmnx._errors import InsufficientResponseError

    urls = []

    def no_features(bbox, tags):
        urls.append(osmnx.settings.overpass_url)
        raise InsufficientResponseError("No matching features")

    monkeypatch.setattr(osmnx, "features_from_bbox", no_features)
    frame = fetch_waterways(online_cfg, BBOX)
    assert len(frame) == 0
    assert urls == list(online_cfg["network"]["overpass_urls"])
    assert not drains.resolve_path(online_cfg, "waterways_file").exists()


@pytest.mark.unit
def test_empty_answer_tries_the_next_mirror(online_cfg, monkeypatch):
    """R1-11: the first mirror answering 'no elements' (e.g. a timeout remark) must not end the query."""
    import osmnx
    from osmnx._errors import InsufficientResponseError

    first, second = online_cfg["network"]["overpass_urls"][:2]

    def flaky(bbox, tags):
        if osmnx.settings.overpass_url == first:
            raise InsufficientResponseError("remark: runtime error: Query timed out")
        return _drain_line_frame()

    monkeypatch.setattr(osmnx, "features_from_bbox", flaky)
    assert len(fetch_waterways(online_cfg, BBOX)) == 1
    cached, covers = load_cached_waterways(drains.resolve_path(online_cfg, "waterways_file"), BBOX, None)
    assert covers and len(cached) == 1
    assert second != first


@pytest.mark.integration
def test_empty_cache_is_requeried_online_and_good_cache_is_kept(online_cfg, monkeypatch, log):
    """R1-11: a legacy empty cache is not authoritative online; an empty refresh keeps a good cache."""
    import osmnx
    from osmnx._errors import InsufficientResponseError

    lon, lat = _grid_nodes()
    path = drains.resolve_path(online_cfg, "waterways_file")
    tags = online_cfg["drains"]["osm_tags"]
    save_waterways_cache(clean_waterways(None), path, (70.0, 10.0, 80.0, 15.0), tags)
    monkeypatch.setattr(osmnx, "features_from_bbox", lambda bbox, tags: _drain_line_frame())
    _, source = drain_distances(lon, lat, online_cfg)
    assert source == "osm", "an empty cached layer must be re-queried online"

    def empty(bbox, tags):
        raise InsufficientResponseError("No matching features")

    monkeypatch.setattr(osmnx, "features_from_bbox", empty)
    before = path.read_text()
    _, source = drain_distances(lon, lat, deep_merge(online_cfg, {"drains": {"refresh": True}}))
    assert source == "osm" and path.read_text() == before
    assert "keeping the existing cache" in log.text


@pytest.mark.unit
def test_fetch_waterways_wraps_transport_errors(online_cfg, monkeypatch):
    import osmnx

    def broken(bbox, tags):
        raise requests.ConnectionError("overpass down")

    monkeypatch.setattr(osmnx, "features_from_bbox", broken)
    with pytest.raises(NetworkUnavailable, match="overpass down"):
        fetch_waterways(online_cfg, BBOX)
    with pytest.raises(ConfigError):
        fetch_waterways(deep_merge(online_cfg, {"drains": {"osm_tags": "waterway"}}), BBOX)
    with pytest.raises(ConfigError):
        fetch_waterways(deep_merge(online_cfg, {"drains": {"osm_tags": {"waterway": 5}}}), BBOX)


@pytest.mark.unit
def test_fetch_waterways_survives_unwritable_cache(online_cfg, monkeypatch, log):
    import osmnx

    monkeypatch.setattr(osmnx, "features_from_bbox", lambda bbox, tags: _drain_line_frame())

    def cannot_write(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(drains, "atomic_write_text", cannot_write)
    assert len(fetch_waterways(online_cfg, BBOX)) == 1
    assert "Could not cache" in log.text


# --------------------------------------------------------------------------- resolution chain
def _grid_nodes():
    lon, lat = np.meshgrid(np.linspace(77.66, 77.69, 5), np.linspace(12.92, 12.945, 4))
    return lon.ravel(), lat.ravel()


@pytest.mark.integration
def test_drain_distances_offline_without_cache_uses_synthetic_line(cfg, log):
    lon, lat = _grid_nodes()
    dist, source = drain_distances(lon, lat, cfg)
    assert source == "synthetic_line"
    np.testing.assert_allclose(dist, synthetic_drain_distance(lon, lat, cfg))
    assert "synthetic" in log.text


@pytest.mark.integration
def test_drain_distances_uses_covering_cache_offline(cfg):
    lon, lat = _grid_nodes()
    query = expand_bbox(nodes_bbox(lon, lat, cfg), cfg["drains"]["bbox_margin_deg"])
    cache_path = drains.resolve_path(cfg, "waterways_file")
    save_waterways_cache(clean_waterways(_drain_line_frame()), cache_path, query, cfg["drains"]["osm_tags"])
    result = compute_drain_distances(lon, lat, cfg)
    assert isinstance(result, DrainResult)
    assert result.source == "osm" and result.n_features == 1
    np.testing.assert_allclose(result.distance_m, synthetic_drain_distance(lon, lat, cfg), rtol=5e-3, atol=0.5)


@pytest.mark.integration
def test_drain_distances_empty_or_stale_cache(cfg, log):
    lon, lat = _grid_nodes()
    path = drains.resolve_path(cfg, "waterways_file")
    tags = cfg["drains"]["osm_tags"]
    save_waterways_cache(clean_waterways(None), path, (70.0, 10.0, 80.0, 15.0), tags)
    _, source = drain_distances(lon, lat, cfg)
    assert source == "synthetic_line"
    # A cache for a smaller bbox is still better than nothing when offline.
    save_waterways_cache(clean_waterways(_drain_line_frame()), path, (77.67, 12.92, 77.68, 12.93), tags)
    _, source = drain_distances(lon, lat, cfg)
    assert source == "osm"
    assert "does not cover" in log.text


@pytest.mark.integration
def test_drain_distances_refetches_stale_cache_online(online_cfg, monkeypatch):
    import osmnx

    lon, lat = _grid_nodes()
    path = drains.resolve_path(online_cfg, "waterways_file")
    save_waterways_cache(clean_waterways(None), path, (77.67, 12.92, 77.68, 12.93), online_cfg["drains"]["osm_tags"])
    calls = []

    def fake(bbox, tags):
        calls.append(bbox)
        return _drain_line_frame()

    monkeypatch.setattr(osmnx, "features_from_bbox", fake)
    _, source = drain_distances(lon, lat, online_cfg)
    assert source == "osm" and len(calls) == 1
    _, source = drain_distances(lon, lat, online_cfg)  # cache now covers: no second query
    assert source == "osm" and len(calls) == 1
    refresh = deep_merge(online_cfg, {"drains": {"refresh": True}})
    drain_distances(lon, lat, refresh)
    assert len(calls) == 2


@pytest.mark.integration
def test_drain_distances_fetch_failure_falls_back(online_cfg, monkeypatch, log):
    import osmnx

    def broken(bbox, tags):
        raise TimeoutError("slow overpass")

    monkeypatch.setattr(osmnx, "features_from_bbox", broken)
    lon, lat = _grid_nodes()
    _, source = drain_distances(lon, lat, online_cfg)
    assert source == "synthetic_line"
    # With --refresh-drains and a failing network, an existing cache is used rather than nothing.
    path = drains.resolve_path(online_cfg, "waterways_file")
    save_waterways_cache(clean_waterways(_drain_line_frame()), path, BBOX, online_cfg["drains"]["osm_tags"])
    _, source = drain_distances(lon, lat, deep_merge(online_cfg, {"drains": {"refresh": True}}))
    assert source == "osm"
    assert "slow overpass" in log.text


@pytest.mark.unit
def test_drain_distances_empty_input(cfg):
    dist, source = drain_distances(np.array([]), np.array([]), cfg)
    assert dist.size == 0 and source == "synthetic_line"


# --------------------------------------------------------------------------- real network (opt-in)
@pytest.mark.network
def test_real_waterways_near_bellandur(cfg, monkeypatch):
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    online = deep_merge(cfg, {"project": {"offline": False}})
    frame = fetch_waterways(online, (77.650, 12.915, 77.690, 12.945))
    assert len(frame) > 0 and (frame["kind"] == "polygon").any()
