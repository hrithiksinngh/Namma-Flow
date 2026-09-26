"""Tests for ``src.data_pipeline.elevation`` (DEM chain, sampling, TPI, flow routing, enrichment, CLI)."""

from __future__ import annotations

import gzip
import importlib.util
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import rasterio
import requests
import yaml
from rasterio.transform import from_origin

from src.data_pipeline import elevation
from src.data_pipeline.drains import clean_waterways, expand_bbox, nodes_bbox, save_waterways_cache
from src.data_pipeline.elevation import (
    ElevationError,
    ElevationResult,
    acquire_dem,
    compute_edge_grades,
    download_srtm,
    enrich_graph,
    estimate_elevations,
    fetch_open_meteo_elevation,
    fill_voids,
    find_local_dem,
    flow_accumulation,
    node_elevations,
    relative_elevation,
    run_elevation_stage,
    sample_dem,
    srtm_tile_names,
    synthetic_elevation,
)
from src.data_pipeline.graph_io import GraphFormatError, graph_to_arrays, load_graph, save_graph
from src.utils.config import ConfigError, deep_merge
from src.utils.geo import to_utm
from src.utils.http import NetworkUnavailable
from tests.conftest import make_grid_graph

pytestmark = pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")  # rasterio/affine internals

BBOX = (77.655, 12.915, 77.700, 12.950)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "src" / "data_pipeline" / "02_elevation_engine.py"


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture
def log(caplog):
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


@pytest.fixture
def online_cfg(cfg, monkeypatch):
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)

    def _blocked(*_args, **_kwargs):
        raise AssertionError("a unit test attempted a real network request")

    monkeypatch.setattr(requests.Session, "request", _blocked)
    return deep_merge(cfg, {"project": {"offline": False}})


def plane(lon, lat):
    """Linear elevation surface: bilinear interpolation reproduces it exactly."""
    return 880.0 + 400.0 * (np.asarray(lon) - 77.65) - 300.0 * (np.asarray(lat) - 12.9)


def write_plane_dem(path: Path, bounds=(77.64, 12.90, 77.72, 12.96), res=0.001, nodata=-9999.0, void=None) -> Path:
    """GeoTIFF (EPSG:4326) with ``plane`` evaluated at pixel centres; ``void`` = boolean mask of nodata."""
    west, south, east, north = bounds
    width, height = round((east - west) / res), round((north - south) / res)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    data = plane(west + (cols + 0.5) * res, north - (rows + 0.5) * res).astype(np.float32)
    if void is not None:
        data[void(rows, cols)] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(west, north, res, res), nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def fake_srtm_tile(tile: str, size: int = 1201, value=None) -> bytes:
    """Gzipped SRTM3-sized .hgt whose elevation rises 1 m every 10 columns (big-endian int16)."""
    cols = np.arange(size)
    grid = np.broadcast_to(850 + cols // 10, (size, size)).astype(">i2").copy()
    if value is not None:
        grid[:] = value
    grid[0, 0] = -32768  # a void, like real SRTM
    return gzip.compress(grid.tobytes(), compresslevel=1)


def load_cli():
    spec = importlib.util.spec_from_file_location("elevation_engine_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- synthetic / fill
@pytest.mark.unit
def test_synthetic_elevation_matches_spec_formula():
    lon = np.array([77.66, 77.69])
    lat = np.array([12.92, 12.94])
    expected = 880.0 + 30.0 * np.sin(lat * 100) + 15.0 * np.cos(lon * 100)
    np.testing.assert_allclose(synthetic_elevation(lon, lat), expected)
    assert synthetic_elevation(77.66, 12.92).shape == (1,)


@pytest.mark.unit
def test_fill_voids_inverse_distance_and_no_mutation():
    lon = np.array([77.660, 77.661, 77.662, 77.6615, 77.700])
    lat = np.full(5, 12.93)
    values = np.array([880.0, np.nan, 884.0, np.nan, 900.0])
    original = values.copy()
    filled = fill_voids(values, lon, lat, k=2)
    np.testing.assert_array_equal(values, original)
    assert np.isfinite(filled).all()
    assert filled[1] == pytest.approx(882.0, abs=1e-6)  # midway between 880 and 884
    assert 882.0 < filled[3] < 884.0
    np.testing.assert_array_equal(fill_voids(original[[0, 2]], lon[[0, 2]], lat[[0, 2]]), original[[0, 2]])


@pytest.mark.unit
def test_fill_voids_edge_cases():
    same_point = fill_voids(np.array([870.0, np.nan]), np.array([77.66, 77.66]), np.array([12.93, 12.93]))
    assert same_point[1] == pytest.approx(870.0)
    with pytest.raises(ValueError, match="no valid"):
        fill_voids(np.array([np.nan, np.nan]), np.array([77.66, 77.67]), np.array([12.93, 12.93]))
    with pytest.raises(ValueError, match="same shape"):
        fill_voids(np.array([1.0, 2.0]), np.array([77.66]), np.array([12.93]))
    with pytest.raises(ValueError):
        fill_voids(np.array([1.0, np.nan]), np.array([77.66, 77.67]), np.array([12.93, 12.93]), k=0)
    assert fill_voids(np.array([]), np.array([]), np.array([])).size == 0


# --------------------------------------------------------------------------- DEM sampling
@pytest.mark.unit
def test_sample_dem_bilinear_is_exact_on_a_plane(tmp_path):
    dem = write_plane_dem(tmp_path / "plane.tif")
    rng = np.random.default_rng(0)
    lon = rng.uniform(77.6405, 77.7195, 200)
    lat = rng.uniform(12.9005, 12.9595, 200)
    np.testing.assert_allclose(sample_dem(dem, lon, lat), plane(lon, lat), atol=1e-3)


@pytest.mark.unit
def test_sample_dem_raster_edges_and_outside(tmp_path):
    dem = write_plane_dem(tmp_path / "plane.tif")
    # Outer half-pixel ring: clamped to the edge pixel centres (no NaN, no index error).
    lon = np.array([77.64, 77.72, 77.6802, 77.6802, 77.72])
    lat = np.array([12.93, 12.93, 12.90, 12.96, 12.96])
    values = sample_dem(dem, lon, lat)
    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(plane(77.6405, 12.93), abs=1e-3)
    assert values[4] == pytest.approx(plane(77.7195, 12.9595), abs=1e-3)
    outside = sample_dem(dem, np.array([77.60, 77.68]), np.array([12.93, 13.5]))
    assert np.isnan(outside).all()
    assert sample_dem(dem, np.array([]), np.array([])).size == 0


@pytest.mark.unit
def test_sample_dem_voids_and_valid_range(tmp_path):
    dem = write_plane_dem(tmp_path / "holes.tif", void=lambda r, c: (r >= 20) & (r < 30) & (c >= 20) & (c < 30))
    centre_lon, centre_lat = 77.64 + 0.025, 12.96 - 0.025  # middle of the void block
    # A quarter of the way from the last valid pixel centre (col 19) towards the first void one (col 20):
    # the void neighbour's weight is dropped and the rest renormalised => value of column 19.
    edge_lon = 77.64 + 0.01975
    values = sample_dem(dem, np.array([centre_lon, edge_lon]), np.array([centre_lat, centre_lat]))
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(plane(77.6595, centre_lat), abs=1e-3)
    ranged = sample_dem(dem, np.array([77.645, 77.715]), np.array([12.93, 12.93]), valid_range=(880.0, 1200.0))
    assert np.isnan(ranged[0]) and np.isfinite(ranged[1])


@pytest.mark.unit
def test_sample_dem_projected_raster(tmp_path):
    x0, y0 = to_utm(77.64, 12.96)
    res = 30.0
    width, height = 300, 250
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs = float(x0) + (cols + 0.5) * res
    ys = float(y0) - (rows + 0.5) * res
    data = (900.0 + 0.001 * (xs - float(x0)) - 0.002 * (float(y0) - ys)).astype(np.float64)
    path = tmp_path / "utm.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float64",
                       crs="EPSG:32643", transform=from_origin(float(x0), float(y0), res, res)) as dst:
        dst.write(data, 1)
    lon = np.array([77.66, 77.67, 77.68])
    lat = np.array([12.94, 12.93, 12.92])
    ux, uy = to_utm(lon, lat)
    expected = 900.0 + 0.001 * (ux - float(x0)) - 0.002 * (float(y0) - uy)
    np.testing.assert_allclose(sample_dem(path, lon, lat), expected, atol=1e-3)


@pytest.mark.unit
def test_sample_dem_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        sample_dem(tmp_path / "missing.tif", np.array([77.66]), np.array([12.93]))
    garbage = tmp_path / "garbage.tif"
    garbage.write_bytes(b"not a tiff")
    with pytest.raises(ElevationError):
        sample_dem(garbage, np.array([77.66]), np.array([12.93]))
    dem = write_plane_dem(tmp_path / "plane.tif")
    with pytest.raises(ValueError):
        sample_dem(dem, np.array([77.66, 77.67]), np.array([12.93]))
    with pytest.raises(ValueError):
        sample_dem(dem, np.array([77.66]), np.array([12.93]), valid_range=(900.0, 800.0))


# --------------------------------------------------------------------------- DEM discovery / SRTM
@pytest.mark.unit
def test_srtm_tile_names():
    assert srtm_tile_names(BBOX, 0.01) == ["N12E077"]
    assert srtm_tile_names((77.99, 12.5, 78.01, 12.6)) == ["N12E077", "N12E078"]
    assert srtm_tile_names((77.2, 12.5, 77.4, 13.0)) == ["N12E077"]  # north edge exactly on a tile line
    assert srtm_tile_names((-0.5, -0.5, 0.5, 0.5)) == ["S01W001", "S01E000", "N00W001", "N00E000"]


@pytest.mark.unit
def test_find_local_dem_coverage_crs_and_corrupt_files(tmp_path, log):
    dem_dir = tmp_path / "dem"
    write_plane_dem(dem_dir / "small.tif", bounds=(77.66, 12.92, 77.68, 12.94))
    assert find_local_dem(dem_dir, BBOX) is None
    (dem_dir / "broken.tif").write_bytes(b"garbage")
    assert find_local_dem(dem_dir, BBOX) is None
    assert "Skipping unreadable DEM" in log.text
    big = write_plane_dem(dem_dir / "city.tiff")
    assert find_local_dem(dem_dir, BBOX) == big
    assert find_local_dem(dem_dir, BBOX, cached=True) is None  # user DEMs are not SRTM cache files
    write_plane_dem(dem_dir / "srtm_cache.tif")
    assert find_local_dem(dem_dir, BBOX, cached=True).name == "srtm_cache.tif"
    assert find_local_dem(tmp_path / "does-not-exist", BBOX) is None
    # A projected DEM is accepted when its bounds cover the bbox after reprojection.
    x0, y0 = to_utm(77.60, 13.00)
    with rasterio.open(dem_dir / "utm.tif", "w", driver="GTiff", width=400, height=400, count=1, dtype="float32",
                       crs="EPSG:32643", transform=from_origin(float(x0), float(y0), 30.0, 30.0)) as dst:
        dst.write(np.full((400, 400), 880.0, dtype=np.float32), 1)
    (dem_dir / "city.tiff").unlink()
    assert find_local_dem(dem_dir, BBOX).name == "utm.tif"


@pytest.mark.unit
def test_acquire_dem_offline_prefers_local_and_respects_sources(cfg):
    dem_dir = Path(cfg["paths"]["dem_dir"])
    assert acquire_dem(cfg, BBOX) is None
    local = write_plane_dem(dem_dir / "local.tif")
    assert acquire_dem(cfg, BBOX) == local
    assert acquire_dem(deep_merge(cfg, {"elevation": {"sources": ["srtm", "synthetic"]}}), BBOX) is None


@pytest.mark.unit
def test_srtm_download_clip_cache_and_refresh(online_cfg, monkeypatch):
    urls = []

    def fake_get_bytes(url, params=None, **kwargs):
        urls.append(url)
        return fake_srtm_tile(url.rsplit("/", 1)[-1].split(".")[0])

    monkeypatch.setattr(elevation, "get_bytes", fake_get_bytes)
    path = acquire_dem(online_cfg, BBOX)
    assert urls == ["https://s3.amazonaws.com/elevation-tiles-prod/skadi/N12/N12E077.hgt.gz"]
    assert path.name == "srtm_77.6550_12.9150_77.7000_12.9500.tif"
    assert (path.parent / "N12E077.hgt.gz").exists()  # raw tile cached for reuse
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326 and src.nodata == -32768
        assert src.bounds.left <= 77.645 + 1e-3 and src.bounds.top >= 12.96 - 1e-3
        assert src.width < 200  # clipped, not the whole 1-degree tile
    lon = np.array([77.66, 77.69])
    expected = 850 + np.floor((lon - 77.0) * 1200 / 10)
    np.testing.assert_allclose(sample_dem(path, lon, np.array([12.93, 12.93])), expected, atol=1.01)
    # Second call: cached GeoTIFF is reused without any download.
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: pytest.fail("should use the cache"))
    assert acquire_dem(online_cfg, BBOX) == path
    # --refresh-dem: ignore cached GeoTIFF and raw tile, download again.
    monkeypatch.setattr(elevation, "get_bytes", fake_get_bytes)
    acquire_dem(deep_merge(online_cfg, {"elevation": {"refresh_dem": True}}), BBOX)
    assert len(urls) == 2


@pytest.mark.unit
def test_srtm_clip_rebuilt_offline_from_cached_raw_tile(online_cfg, monkeypatch):
    monkeypatch.setattr(elevation, "get_bytes", lambda url, **k: fake_srtm_tile("N12E077"))
    first = download_srtm(online_cfg, BBOX)
    first.unlink()  # e.g. the bbox changed: a new clip is needed, but the raw tile is cached
    offline = deep_merge(online_cfg, {"project": {"offline": True}})
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: pytest.fail("offline: no download"))
    other_bbox = (77.60, 12.90, 77.65, 12.95)
    rebuilt = acquire_dem(offline, other_bbox)
    assert rebuilt is not None and rebuilt.name.startswith("srtm_77.6000")
    assert np.isfinite(sample_dem(rebuilt, np.array([77.62]), np.array([12.92]))).all()


@pytest.mark.unit
def test_srtm_multi_tile_merge(online_cfg, monkeypatch):
    values = {"N12E077": 880, "N12E078": 890}
    def fake_get_bytes(url, params=None, **kwargs):
        tile = url.rsplit("/", 1)[-1][:7]
        return fake_srtm_tile(tile, value=values[tile])

    monkeypatch.setattr(elevation, "get_bytes", fake_get_bytes)
    bbox = (77.990, 12.50, 78.010, 12.52)
    path = download_srtm(online_cfg, bbox)
    sampled = sample_dem(path, np.array([77.995, 78.005]), np.array([12.51, 12.51]))
    np.testing.assert_allclose(sampled, [880.0, 890.0])


@pytest.mark.unit
def test_srtm_failures_fall_through(online_cfg, monkeypatch, log):
    def not_found(url, params=None, **kwargs):
        raise NetworkUnavailable(f"{url} returned HTTP 404")

    monkeypatch.setattr(elevation, "get_bytes", not_found)
    assert acquire_dem(online_cfg, BBOX) is None
    assert "HTTP 404" in log.text
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: b"\x1f\x8b corrupt gzip")
    assert acquire_dem(online_cfg, BBOX) is None
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: gzip.compress(b"short"))
    with pytest.raises(ElevationError, match="unexpected size"):
        download_srtm(online_cfg, BBOX)
    with pytest.raises(ValueError, match="tiles"):
        download_srtm(online_cfg, (70.0, 10.0, 80.0, 15.0))
    with pytest.raises(NetworkUnavailable):
        download_srtm(deep_merge(online_cfg, {"project": {"offline": True}}), BBOX)


@pytest.mark.unit
def test_srtm_all_void_tile_is_rejected(online_cfg, monkeypatch):
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: fake_srtm_tile("N12E077", value=-32768))
    with pytest.raises(ElevationError, match="only voids"):
        download_srtm(online_cfg, BBOX)


@pytest.mark.unit
def test_srtm_corrupt_cached_raw_tile_is_redownloaded(online_cfg, monkeypatch):
    dem_dir = Path(online_cfg["paths"]["dem_dir"])
    dem_dir.mkdir(parents=True)
    (dem_dir / "N12E077.hgt.gz").write_bytes(b"broken")
    urls = []
    monkeypatch.setattr(elevation, "get_bytes", lambda url, **k: urls.append(url) or fake_srtm_tile("N12E077"))
    assert download_srtm(online_cfg, BBOX).exists()
    assert len(urls) == 1


@pytest.mark.unit
def test_offline_refresh_dem_reuses_the_cached_srtm_clip(online_cfg, monkeypatch, log):
    """R1-03: offline, --refresh-dem cannot re-download, so the cached clip is used (WARNING)."""
    monkeypatch.setattr(elevation, "get_bytes", lambda url, **k: fake_srtm_tile("N12E077"))
    clip = acquire_dem(online_cfg, BBOX)
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: pytest.fail("offline: no download"))
    offline_refresh = deep_merge(online_cfg, {"project": {"offline": True}, "elevation": {"refresh_dem": True}})
    assert acquire_dem(offline_refresh, BBOX) == clip
    assert "refresh_dem ignored offline" in log.text
    lon, lat = np.array([77.66, 77.69]), np.array([12.93, 12.93])
    assert estimate_elevations(lon, lat, offline_refresh, BBOX).source == "srtm"


@pytest.mark.unit
def test_offline_refresh_dem_rebuilds_the_clip_from_the_cached_tile(online_cfg, monkeypatch, log):
    """R1-03: the raw tile on disk is used (not reported as 'not cached') when the clip is gone."""
    monkeypatch.setattr(elevation, "get_bytes", lambda url, **k: fake_srtm_tile("N12E077"))
    download_srtm(online_cfg, BBOX).unlink()
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: pytest.fail("offline: no download"))
    offline_refresh = deep_merge(online_cfg, {"project": {"offline": True}, "elevation": {"refresh_dem": True}})
    rebuilt = download_srtm(offline_refresh, BBOX)
    assert rebuilt.exists() and "not cached" not in log.text
    (rebuilt.parent / "N12E077.hgt.gz").unlink()
    rebuilt.unlink()
    with pytest.raises(NetworkUnavailable, match="is not cached"):
        download_srtm(offline_refresh, BBOX)


# --------------------------------------------------------------------------- Open-Meteo
@pytest.mark.unit
def test_fetch_open_meteo_elevation_batches_and_voids(online_cfg, monkeypatch):
    calls = []

    def fake_get_json(url, params=None, **kwargs):
        lats = params["latitude"].split(",")
        calls.append(len(lats))
        return {"elevation": [870.0 + i for i in range(len(lats) - 1)] + [-9999.0]}

    monkeypatch.setattr(elevation, "get_json", fake_get_json)
    cfg = deep_merge(online_cfg, {"elevation": {"open_meteo_batch_size": 3}})
    lon = np.linspace(77.66, 77.69, 7)
    lat = np.linspace(12.92, 12.94, 7)
    values = fetch_open_meteo_elevation(lon, lat, cfg)
    assert calls == [3, 3, 1]
    assert np.isnan(values[[2, 5, 6]]).all()  # last point of every batch is out of range
    assert values[0] == 870.0 and values[3] == 870.0


@pytest.mark.unit
@pytest.mark.parametrize("payload", [{"elevation": [1.0]}, {"error": True}, ["x"], {"elevation": "abc"}])
def test_fetch_open_meteo_elevation_malformed(online_cfg, monkeypatch, payload):
    monkeypatch.setattr(elevation, "get_json", lambda *a, **k: payload)
    with pytest.raises(NetworkUnavailable, match="malformed"):
        fetch_open_meteo_elevation(np.array([77.66, 77.67]), np.array([12.93, 12.93]), online_cfg)


@pytest.mark.unit
def test_fetch_open_meteo_elevation_offline(cfg):
    with pytest.raises(NetworkUnavailable):
        fetch_open_meteo_elevation(np.array([77.66]), np.array([12.93]), cfg)
    assert fetch_open_meteo_elevation(np.array([]), np.array([]), cfg).size == 0


# --------------------------------------------------------------------------- source chain
def _nodes(n_side: int = 6):
    lon, lat = np.meshgrid(np.linspace(77.66, 77.69, n_side), np.linspace(12.92, 12.945, n_side))
    return lon.ravel(), lat.ravel()


@pytest.mark.integration
def test_node_elevations_offline_falls_back_to_synthetic(cfg, log):
    lon, lat = _nodes()
    elev, source = node_elevations(lon, lat, cfg)
    assert source == "synthetic"
    np.testing.assert_allclose(elev, synthetic_elevation(lon, lat))


@pytest.mark.integration
def test_node_elevations_local_dem_with_few_voids_is_filled(cfg):
    lon, lat = _nodes()
    dem_dir = Path(cfg["paths"]["dem_dir"])
    # Void block around the first node only.
    write_plane_dem(dem_dir / "local.tif", void=lambda r, c: (r >= 37) & (r <= 42) & (c >= 18) & (c <= 22))
    result = estimate_elevations(lon, lat, cfg)
    assert isinstance(result, ElevationResult)
    assert result.source == "local_dem" and result.dem_path.endswith("local.tif")
    assert result.n_voids == 1 and result.void_fraction == pytest.approx(1 / 36)
    assert np.isfinite(result.elevation).all()
    np.testing.assert_allclose(result.elevation[1:], plane(lon[1:], lat[1:]), atol=1e-3)
    assert abs(result.elevation[0] - plane(lon[0], lat[0])) < 10.0


@pytest.mark.integration
def test_node_elevations_rejects_mostly_void_source(cfg, log):
    lon, lat = _nodes()
    write_plane_dem(Path(cfg["paths"]["dem_dir"]) / "holes.tif", void=lambda r, c: r < 45)
    elev, source = node_elevations(lon, lat, cfg)
    assert source == "synthetic"
    assert "voids" in log.text


@pytest.mark.integration
def test_node_elevations_open_meteo_and_unknown_sources(online_cfg, monkeypatch, log):
    def fake_get_json(url, params=None, **kwargs):
        return {"elevation": [875.5] * len(params["latitude"].split(","))}

    monkeypatch.setattr(elevation, "get_json", fake_get_json)
    cfg = deep_merge(online_cfg, {"elevation": {"sources": ["bogus", "open_meteo", "synthetic"]}})
    lon, lat = _nodes()
    elev, source = node_elevations(lon, lat, cfg)
    assert source == "open_meteo" and np.all(elev == 875.5)
    assert "Unknown elevation source 'bogus'" in log.text


@pytest.mark.integration
def test_node_elevations_exhausted_chain_and_bad_config(cfg, log):
    lon, lat = _nodes()
    elev, source = node_elevations(lon, lat, deep_merge(cfg, {"elevation": {"sources": ["srtm", "open_meteo"]}}))
    assert source == "synthetic"
    assert "exhausted" in log.text
    with pytest.raises(ConfigError):
        node_elevations(lon, lat, deep_merge(cfg, {"elevation": {"sources": 5}}))
    with pytest.raises(ConfigError):
        node_elevations(lon, lat, deep_merge(cfg, {"elevation": {"valid_range_m": [900, 800]}}))
    with pytest.raises(ValueError, match="at least one node"):
        node_elevations(np.array([]), np.array([]), cfg)
    string_sources = deep_merge(cfg, {"elevation": {"sources": "synthetic"}})
    assert node_elevations(lon, lat, string_sources)[1] == "synthetic"


# --------------------------------------------------------------------------- grades / TPI
@pytest.mark.unit
def test_compute_edge_grades_values_clipping_and_immutability(cfg):
    G = make_grid_graph(3, 3)
    for _, _, data in G.edges(data=True):
        data["grade"] = 99.0
    first = next(iter(G.nodes))
    G.nodes[first]["elevation"] = 2000.0  # absurd spike => clipped grades
    H = compute_edge_grades(G, cfg)
    assert H is not G
    assert all(d["grade"] == 99.0 for _, _, d in G.edges(data=True))
    max_grade = cfg["elevation"]["max_abs_grade"]
    for u, v, data in H.edges(data=True):
        raw = (H.nodes[v]["elevation"] - H.nodes[u]["elevation"]) / data["length"]
        assert data["grade"] == pytest.approx(float(np.clip(raw, -max_grade, max_grade)))
        assert H.edges[v, u]["grade"] == pytest.approx(-data["grade"])
    assert max(abs(d["grade"]) for _, _, d in H.edges(data=True)) == pytest.approx(max_grade)


@pytest.mark.unit
def test_compute_edge_grades_repairs_lengths_and_validates(cfg, log):
    G = make_grid_graph(2, 2)
    u, v = next(iter(G.edges))
    G.edges[u, v]["length"] = float("nan")
    G.edges[v, u].pop("length")
    H = compute_edge_grades(G, cfg)
    assert H.edges[u, v]["length"] > 100 and H.edges[v, u]["length"] == pytest.approx(H.edges[u, v]["length"])
    assert "missing/invalid length" in log.text
    G.nodes[u].pop("elevation")
    with pytest.raises(ValueError, match="elevation"):
        compute_edge_grades(G, cfg)
    with pytest.raises(ConfigError):
        compute_edge_grades(make_grid_graph(2, 2), deep_merge(cfg, {"elevation": {"max_abs_grade": -1}}))


@pytest.mark.unit
def test_relative_elevation_bowl_flat_and_validation():
    lon, lat = np.meshgrid(np.linspace(77.660, 77.664, 5), np.linspace(12.930, 12.934, 5))
    lon, lat = lon.ravel(), lat.ravel()
    x = (lon - lon.mean()) * 1e4
    y = (lat - lat.mean()) * 1e4
    bowl = 870.0 + 5.0 * np.hypot(x, y)  # cone: the apex is the sharpest local depression
    tpi = relative_elevation(lon, lat, bowl, radius_m=200.0)
    centre = 12
    assert tpi[centre] < 0 and tpi[centre] == tpi.min()
    assert np.argsort(tpi)[0] == centre and tpi[centre] < np.sort(tpi)[1] - 1.0
    np.testing.assert_allclose(relative_elevation(lon, lat, np.full(25, 880.0), 300.0), 0.0, atol=1e-9)
    np.testing.assert_allclose(relative_elevation(lon, lat, bowl, 1.0), 0.0, atol=1e-9)  # nobody else within 1 m
    # Hand-computed: 3 collinear nodes 100 m apart, radius 150 m.
    lon3 = 77.66 + np.array([0.0, 1.0, 2.0]) * 100.0 / 108_500.0
    tpi3 = relative_elevation(lon3, np.full(3, 12.93), np.array([10.0, 13.0, 19.0]), 150.0)
    np.testing.assert_allclose(tpi3, [10 - 11.5, 13 - 14.0, 19 - 16.0], atol=1e-6)
    assert relative_elevation(np.array([]), np.array([]), np.array([]), 300.0).size == 0
    with pytest.raises(ValueError):
        relative_elevation(lon, lat, bowl, 0.0)
    with pytest.raises(ValueError):
        relative_elevation(lon, lat, bowl[:3], 100.0)
    with pytest.raises(ValueError):
        relative_elevation(lon, lat, np.full(25, np.nan), 100.0)


# --------------------------------------------------------------------------- enrichment
@pytest.mark.integration
def test_enrich_graph_offline_adds_every_attribute(cfg, tmp_path):
    G = make_grid_graph(6, 6)
    for node in G.nodes:
        for key in ("elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink"):
            G.nodes[node].pop(key)
    snapshot = {n: dict(d) for n, d in G.nodes(data=True)}
    H = enrich_graph(G, cfg)
    assert H is not G and {n: dict(d) for n, d in G.nodes(data=True)} == snapshot
    assert H.graph["elevation_source"] == "synthetic" and H.graph["drain_source"] == "synthetic_line"
    assert H.number_of_nodes() == 36 and H.number_of_edges() == G.number_of_edges()
    for _, data in H.nodes(data=True):
        assert np.isfinite(data["elevation"]) and data["dist_to_drain_m"] >= 0
        assert data["flow_accumulation"] >= 1 and isinstance(data["is_sink"], bool)
    for u, v, data in H.edges(data=True):
        assert abs(data["grade"]) <= cfg["elevation"]["max_abs_grade"]
        assert H.edges[v, u]["grade"] == pytest.approx(-data["grade"])
    path = save_graph(H, tmp_path / "enriched.graphml")
    arrays = graph_to_arrays(load_graph(path))
    for key in ("elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink"):
        assert key in arrays.node_attrs
    assert {"length", "grade"} <= set(arrays.edge_attrs)
    summarize_enrichment = load_cli().summarize_enrichment
    summary = summarize_enrichment(H)
    assert summary["nodes"] == 36 and summary["elevation_source"] == "synthetic"
    assert summary["n_sinks"] == int(arrays.node_attrs["is_sink"].sum())
    # Regression: networkx.read_graphml injects node_default/edge_default dicts that GraphML
    # cannot re-serialise; load_graph strips them and re-enriching a loaded graph saves cleanly.
    reloaded = load_graph(path)
    assert "node_default" not in reloaded.graph
    again = enrich_graph(reloaded, cfg)
    assert "node_default" not in again.graph
    save_graph(again, tmp_path / "again.graphml")
    unenriched = make_grid_graph(2, 2)
    for node in unenriched.nodes:
        unenriched.nodes[node].pop("flow_accumulation")
    with pytest.raises(ValueError, match="not enriched"):
        summarize_enrichment(unenriched)


@pytest.mark.integration
def test_enrich_graph_with_local_dem_and_cached_drains(cfg):
    G = make_grid_graph(5, 5)
    write_plane_dem(Path(cfg["paths"]["dem_dir"]) / "city.tif")
    arrays = graph_to_arrays(G)
    query = expand_bbox(nodes_bbox(arrays.lon, arrays.lat, cfg), cfg["drains"]["bbox_margin_deg"])
    import geopandas as gpd
    from shapely.geometry import LineString

    line = LineString([(77.675, 12.9), (77.675, 12.97)])
    lines = gpd.GeoDataFrame({"waterway": ["drain"]}, geometry=[line], crs=4326)
    save_waterways_cache(clean_waterways(lines), Path(cfg["paths"]["waterways_file"]), query, cfg["drains"]["osm_tags"])
    H = enrich_graph(G, cfg)
    assert H.graph["elevation_source"] == "local_dem" and H.graph["drain_source"] == "osm"
    assert H.graph["drain_feature_count"] == 1
    out = graph_to_arrays(H)
    np.testing.assert_allclose(out.node_attrs["elevation"], plane(out.lon, out.lat), atol=1e-3)


@pytest.mark.unit
def test_enrich_graph_input_validation(cfg):
    with pytest.raises(GraphFormatError, match="no nodes"):
        enrich_graph(nx.DiGraph(), cfg)
    G = make_grid_graph(2, 2)
    G.nodes[next(iter(G.nodes))].pop("x")
    with pytest.raises(GraphFormatError):
        enrich_graph(G, cfg)
    with pytest.raises(TypeError):
        enrich_graph("not a graph", cfg)
    bad = make_grid_graph(2, 2)
    bad.nodes[next(iter(bad.nodes))]["y"] = float("nan")
    with pytest.raises(GraphFormatError, match="finite"):
        enrich_graph(bad, cfg)


@pytest.mark.unit
def test_enrich_graph_accepts_multigraph_and_single_node(cfg):
    M = nx.MultiDiGraph()
    M.add_node(1, x=77.66, y=12.93)
    M.add_node(2, x=77.661, y=12.93)
    M.add_edge(1, 2, length=150.0, highway="primary")
    M.add_edge(1, 2, length=110.0, highway="service")
    M.add_edge(2, 1, length=110.0)
    H = enrich_graph(M, cfg)
    assert isinstance(H, nx.DiGraph) and not H.is_multigraph()
    assert H.edges[1, 2]["length"] == 110.0 and H.edges[1, 2]["highway"] == "service"
    U = nx.Graph()
    U.add_edge("a", "b", length=50.0)
    U.nodes["a"].update(x=77.66, y=12.93)
    U.nodes["b"].update(x=77.6605, y=12.93)
    HU = enrich_graph(U, cfg)
    assert HU.has_edge("a", "b") and HU.has_edge("b", "a")
    single = nx.DiGraph()
    single.add_node(7, x=77.67, y=12.93)
    S = enrich_graph(single, cfg)
    assert S.nodes[7]["is_sink"] is True and S.nodes[7]["relative_elevation"] == 0.0


# --------------------------------------------------------------------------- stage + CLI
def _write_config(cfg: dict, path: Path) -> Path:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.write_text(yaml.safe_dump(clean, sort_keys=False))
    return path


@pytest.mark.integration
def test_run_elevation_stage_missing_graph(cfg):
    with pytest.raises(FileNotFoundError, match="01_extract_network"):
        run_elevation_stage(cfg)


@pytest.mark.e2e
def test_cli_enriches_graph_in_place(cfg, tmp_path, capsys):
    graph_path = Path(cfg["paths"]["graph_file"])
    save_graph(make_grid_graph(4, 4), graph_path)
    config_path = _write_config(cfg, tmp_path / "config.yaml")
    cli = load_cli()
    assert cli.main(["--config", str(config_path), "--offline", "--refresh-dem", "--refresh-drains"]) == 0
    out = capsys.readouterr().out
    assert "Elevation source" in out and "synthetic" in out and "Sinks" in out
    enriched = load_graph(graph_path)
    assert enriched.graph["elevation_source"] == "synthetic"
    assert all("flow_accumulation" in d for _, d in enriched.nodes(data=True))


def _real_provenance_graph(path: Path) -> nx.DiGraph:
    G = make_grid_graph(4, 4)
    G.graph.update(elevation_source="srtm", drain_source="osm")
    save_graph(G, path)
    return load_graph(path)


@pytest.mark.integration
def test_stage_refuses_to_degrade_real_elevations_to_synthetic(cfg, log):
    """R1-03: offline without the DEM/waterway caches, stage 02 must not overwrite SRTM/OSM data."""
    graph_path = Path(cfg["paths"]["graph_file"])
    _real_provenance_graph(graph_path)
    before = graph_path.read_bytes()
    with pytest.raises(ElevationError, match="allow-synthetic"):
        run_elevation_stage(cfg)
    assert graph_path.read_bytes() == before
    enriched, _ = run_elevation_stage(cfg, allow_synthetic=True)
    assert enriched.graph["elevation_source"] == "synthetic"
    assert load_graph(graph_path).graph["drain_source"] == "synthetic_line"


@pytest.mark.e2e
def test_cli_offline_refresh_dem_keeps_srtm_and_guards_the_graph(online_cfg, tmp_path, capsys, monkeypatch):
    """R1-03 repro: ``02 --offline --refresh-dem`` with the tile cached keeps SRTM elevations."""
    monkeypatch.setattr(elevation, "get_bytes", lambda url, **k: fake_srtm_tile("N12E077"))
    acquire_dem(online_cfg, BBOX)
    monkeypatch.setattr(elevation, "get_bytes", lambda *a, **k: pytest.fail("offline: no download"))
    cfg = deep_merge(online_cfg, {"project": {"offline": True}})
    graph_path = Path(cfg["paths"]["graph_file"])
    G = make_grid_graph(4, 4)
    G.graph.update(elevation_source="srtm")
    save_graph(G, graph_path)
    config_path = _write_config(cfg, tmp_path / "config.yaml")
    cli = load_cli()
    assert cli.main(["--config", str(config_path), "--offline", "--refresh-dem"]) == 0
    assert load_graph(graph_path).graph["elevation_source"] == "srtm"
    for tif in Path(cfg["paths"]["dem_dir"]).iterdir():
        tif.unlink()
    assert cli.main(["--config", str(config_path), "--offline"]) == 1
    assert "allow-synthetic" in capsys.readouterr().err
    assert load_graph(graph_path).graph["elevation_source"] == "srtm"
    assert cli.main(["--config", str(config_path), "--offline", "--allow-synthetic"]) == 0
    assert load_graph(graph_path).graph["elevation_source"] == "synthetic"


@pytest.mark.e2e
def test_cli_handled_failures(cfg, tmp_path, capsys):
    cli = load_cli()
    config_path = _write_config(cfg, tmp_path / "config.yaml")
    assert cli.main(["--config", str(config_path)]) == 1
    assert "run python src/data_pipeline/01_extract_network.py first" in capsys.readouterr().err
    assert cli.main(["--config", str(tmp_path / "nope.yaml")]) == 1
    graph_path = Path(cfg["paths"]["graph_file"])
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text("<graphml>broken")
    assert cli.main(["--config", str(config_path)]) == 1
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.e2e
def test_cli_unexpected_error_is_one_line(cfg, tmp_path, capsys, monkeypatch):
    cli = load_cli()
    config_path = _write_config(cfg, tmp_path / "config.yaml")
    monkeypatch.setattr(cli, "run_elevation_stage", lambda cfg, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main(["--config", str(config_path)]) == 1
    err = capsys.readouterr().err
    assert "boom" in err and "Traceback" not in err


# --------------------------------------------------------------------------- real network (opt-in)
@pytest.mark.network
def test_real_open_meteo_elevation(cfg, monkeypatch):
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    online = deep_merge(cfg, {"project": {"offline": False}})
    values = fetch_open_meteo_elevation(np.array([77.67]), np.array([12.93]), online)
    assert 850 < values[0] < 950
