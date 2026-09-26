"""Shared pytest fixtures.

All fixtures are offline and synthetic: no test may touch the network unless it is
marked ``@pytest.mark.network`` (those are skipped unless NAMMA_FLOW_NETWORK_TESTS=1).
Module-specific fixtures belong in the module's own test file.
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config

BBOX = (77.655, 12.915, 77.700, 12.950)
TZ = "Asia/Kolkata"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("NAMMA_FLOW_NETWORK_TESTS", "").lower() in {"1", "true", "yes"}:
        return
    skip = pytest.mark.skip(reason="network test (set NAMMA_FLOW_NETWORK_TESTS=1 to run)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Force offline mode for every test so an accidental network call fails fast."""
    if os.environ.get("NAMMA_FLOW_NETWORK_TESTS", "").lower() not in {"1", "true", "yes"}:
        monkeypatch.setenv("NAMMA_FLOW_OFFLINE", "1")


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    """Real config with every data/artifact path redirected into ``tmp_path``."""
    paths = {
        "dem_dir": tmp_path / "raw/dem",
        "weather_dir": tmp_path / "raw/weather",
        "labels_dir": tmp_path / "raw/labels",
        "interim_dir": tmp_path / "interim",
        "processed_dir": tmp_path / "processed",
        "graph_file": tmp_path / "interim/test.graphml",
        "waterways_file": tmp_path / "interim/waterways.geojson",
        "weather_file": tmp_path / "raw/weather/weather.csv",
        "flood_reports_file": tmp_path / "raw/labels/flood_reports.csv",
        "train_dataset": tmp_path / "processed/train_dataset.pt",
        "val_dataset": tmp_path / "processed/val_dataset.pt",
        "test_dataset": tmp_path / "processed/test_dataset.pt",
        "checkpoint_dir": tmp_path / "checkpoints",
        "reports_dir": tmp_path / "reports",
        "osm_cache_dir": tmp_path / "interim/osm_cache",
    }
    overrides = {
        "project": {"offline": True},
        "paths": {k: str(v) for k, v in paths.items()},
        "network": {"synthetic_grid": {"rows": 6, "cols": 6}},
        "weather": {"start_date": "2021-09-01", "end_date": "2022-10-31"},
    }
    return load_config(overrides=overrides)


def make_grid_graph(rows: int = 6, cols: int = 6, bbox=BBOX, seed: int = 0) -> nx.DiGraph:
    """Bidirectional street grid with valley-shaped elevation and all enrichment attributes."""
    rng = np.random.default_rng(seed)
    west, south, east, north = bbox
    lons = np.linspace(west + 0.002, east - 0.002, cols)
    lats = np.linspace(south + 0.002, north - 0.002, rows)
    G = nx.DiGraph(crs="epsg:4326", source="test_grid")
    node_id = lambda r, c: 1000 + r * cols + c  # noqa: E731
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            # Valley along the middle column: lowest at the centre, draining south.
            elevation = 870.0 + 2.5 * abs(c - (cols - 1) / 2) + 0.8 * r + rng.normal(0, 0.2)
            G.add_node(
                node_id(r, c),
                x=float(lon),
                y=float(lat),
                street_count=4,
                elevation=float(round(elevation, 2)),
                dist_to_drain_m=float(abs(c - (cols - 1) / 2) * 400.0),
                relative_elevation=0.0,
                flow_accumulation=1.0,
                is_sink=False,
            )
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr >= rows or cc >= cols:
                    continue
                u, v = node_id(r, c), node_id(rr, cc)
                du, dv = G.nodes[u], G.nodes[v]
                length = float(
                    np.hypot((du["x"] - dv["x"]) * 108_000, (du["y"] - dv["y"]) * 111_000)
                )
                for a, b in ((u, v), (v, u)):
                    grade = (G.nodes[b]["elevation"] - G.nodes[a]["elevation"]) / length
                    G.add_edge(a, b, length=length, grade=float(grade), highway="residential", oneway=False)
    mean_elev = np.mean([d["elevation"] for _, d in G.nodes(data=True)])
    for _, data in G.nodes(data=True):
        data["relative_elevation"] = float(data["elevation"] - mean_elev)
    return G


@pytest.fixture
def grid_graph() -> nx.DiGraph:
    return make_grid_graph()


@pytest.fixture
def hourly_index():
    """Four days of hourly timestamps in local time."""
    return pd.date_range("2022-09-03 00:00", periods=96, freq="h", tz=TZ)


@pytest.fixture
def storm_series(hourly_index) -> pd.Series:
    """Areal rainfall with a dry spell, a moderate event and a cloudburst (mm/h)."""
    rain = np.zeros(len(hourly_index))
    rain[20:24] = [2.0, 5.0, 3.0, 1.0]
    rain[60:66] = [10.0, 35.0, 45.0, 25.0, 10.0, 5.0]
    return pd.Series(rain, index=hourly_index, name="precipitation_mm")
