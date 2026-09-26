"""Shared helpers and fixtures of the stage-04 dataset tests (``test_dataset*.py``).

The hydrology simulator (module D) is replaced by a deterministic fake injected into
``sys.modules`` so the tests do not depend on its calibration. Weather comes from a
hand-written schema-2.2 CSV (a 4/18/8 mm storm every 96 h at 15:00) so window selection is
predictable. The default record spans 2021-08 .. 2023-06: train = 2021, val = 2022,
test = 2023. Every dataset path (including ``paths.test_dataset``) points into ``tmp_path``.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.graph_io import save_graph
from src.data_pipeline.weather import save_weather_csv
from src.utils.config import deep_merge
from tests.conftest import TZ, make_grid_graph

FLOOD_MM_H = 10.0
START, END = "2021-08-01", "2023-06-30"
VAL_YEARS, TEST_YEARS = [2022], [2023]


@pytest.fixture
def log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def fake_simulate(graph, rain, cfg):
    """Deterministic stand-in for module D: flooded where the node's hourly rain >= 10 mm."""
    rain = np.asarray(rain, dtype=np.float32)
    assert rain.shape == (rain.shape[0], graph.num_nodes)
    return (rain >= FLOOD_MM_H).astype(np.uint8), (rain / 40.0).astype(np.float32)


@pytest.fixture
def fake_simulator(monkeypatch):
    """Install a fake ``src.hydrology.simulator`` and count its calls."""
    module = types.ModuleType("src.hydrology.simulator")
    calls = []

    def simulate_labels(graph, rain, cfg):
        calls.append(rain.shape)
        return fake_simulate(graph, rain, cfg)

    module.simulate_labels = simulate_labels
    module.calls = calls
    monkeypatch.setitem(sys.modules, "src.hydrology.simulator", module)
    return module


def storm_record(start: str, end: str, every_h: int = 96, profile=(4.0, 18.0, 8.0),
                 source: str = "open_meteo") -> pd.DataFrame:
    """Hourly schema-2.2 frame: dry except a storm with ``profile`` every ``every_h`` hours at 15:00."""
    index = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h", tz=TZ, name="timestamp")
    rain = np.zeros(len(index))
    first = int(np.flatnonzero(index.hour == 15)[0])
    for s in range(first, len(index) - len(profile), every_h):
        rain[s: s + len(profile)] = profile
    return pd.DataFrame({"precipitation_mm": rain, "is_imputed": False, "source": source}, index=index)


def setup_config(cfg: dict, *, start: str = START, end: str = END, graph=None, dataset=None, **sections) -> dict:
    """Write a 5x5 grid graph + storm weather CSV into the tmp paths and return the adjusted config."""
    processed = Path(cfg["paths"]["processed_dir"])
    out = deep_merge(cfg, {
        "paths": {"test_dataset": str(processed / "test_dataset.pt")},
        "weather": {"start_date": start, "end_date": end},
        "dataset": {"val_years": VAL_YEARS, "test_years": TEST_YEARS, **(dataset or {})}, **sections,
    })
    save_graph(graph if graph is not None else make_grid_graph(5, 5), out["paths"]["graph_file"])
    save_weather_csv(storm_record(start, end), out["paths"]["weather_file"])
    return out


def split_years(payload: dict) -> set[int]:
    """Local calendar years of the first hour of every window of ``payload``."""
    starts = payload["window_starts"].numpy()
    stamps = pd.to_datetime(payload["timestamps"].numpy()[starts], unit="s", utc=True).tz_convert(TZ)
    return set(stamps.year)
