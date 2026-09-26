"""Shared fixtures and helpers of the training test files (``test_training*.py``).

A tiny dataset is built ONCE per test module with the real stage-04 code
(:func:`build_datasets`, features, rain field) on a 4x4 ``make_grid_graph`` and a storm
record spanning three seasons, so it has a TRAIN (2021), VALIDATION (2022) and held-out
TEST (2023) split. The hydrology simulator is replaced by a deterministic rule (flooded
where the junction's hourly rain >= 10 mm) so labels are learnable and do not depend on
module D's calibration. Every test trains a very small model for 1-4 epochs in ~a second.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline.dataset import build_datasets
from src.data_pipeline.graph_io import save_graph
from src.data_pipeline.weather import save_weather_csv
from src.utils.config import deep_merge, load_config
from tests.conftest import TZ, make_grid_graph

FLOOD_MM_H = 10.0
DATA = {"weather": {"start_date": "2021-10-01", "end_date": "2023-05-31"},
        "dataset": {"val_years": [2022], "test_years": [2023], "season_months": [5, 10], "seq_len": 12,
                    "warmup_steps": 3}}
SMALL = {"model": {"hidden_dim": 8, "heads": 1, "batch_size": 4, "epochs": 2, "learning_rate": 0.01},
         "training": {"device": "cpu", "windows_per_epoch": 8, "micro_batch_size": 4, "early_stopping_patience": 5,
                      "baseline": {"max_samples": 4000, "max_train_windows": 20, "hist_gbdt": {"max_iter": 15}},
                      # X6 has its own tests (tests/test_areal_skill.py); disabled here to keep the suite fast
                      "areal_skill": {"enabled": False}}}
SEQ_LEN, WARMUP, N_NODES = 12, 3, 16
N_TRAIN, N_VAL, N_TEST = 20, 22, 11  # windows of the shared build (asserted by the fixture)
PATH_NAMES = {"graph_file": "grid.graphml", "weather_file": "weather.csv", "train_dataset": "train.pt",
              "val_dataset": "val.pt", "test_dataset": "test.pt", "reports_dir": "reports",
              "checkpoint_dir": "checkpoints", "flood_reports_file": "reports.csv", "waterways_file": "ww.geojson",
              "osm_cache_dir": "osm", "dem_dir": "dem", "weather_dir": "weather", "interim_dir": "interim",
              "processed_dir": "processed", "labels_dir": "labels"}
SHARED_KEYS = ("train_dataset", "val_dataset", "test_dataset", "graph_file", "weather_file")


def fake_simulator() -> types.ModuleType:
    module = types.ModuleType("src.hydrology.simulator")

    def simulate_labels(graph, rain, cfg):
        rain = np.asarray(rain, dtype=np.float32)
        return (rain >= FLOOD_MM_H).astype(np.uint8), (rain / 40.0).astype(np.float32)

    module.simulate_labels = simulate_labels
    return module


def write_inputs(cfg: dict) -> None:
    """4x4 grid graph + an hourly storm record (4/18/8 mm every 4 days at 15:00)."""
    save_graph(make_grid_graph(4, 4), cfg["paths"]["graph_file"])
    start, end = cfg["weather"]["start_date"], cfg["weather"]["end_date"]
    index = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h", tz=TZ, name="timestamp")
    rain = np.zeros(len(index))
    first = int(np.flatnonzero(index.hour == 15)[0])
    for s in range(first, len(index) - 3, 96):
        rain[s: s + 3] = (4.0, 18.0, 8.0)
    frame = pd.DataFrame({"precipitation_mm": rain, "is_imputed": False, "source": "open_meteo"}, index=index)
    save_weather_csv(frame, cfg["paths"]["weather_file"])


def build_shared(root: Path) -> dict[str, str]:
    """Build the tiny train/val/test datasets under ``root`` (real stage-04 code, fake simulator)."""
    paths = {key: str(root / name) for key, name in PATH_NAMES.items()}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NAMMA_FLOW_OFFLINE", "1")
        patch.setitem(sys.modules, "src.hydrology.simulator", fake_simulator())
        cfg = load_config(overrides={"project": {"offline": True}, "paths": paths,
                                     "network": {"synthetic_grid": {"rows": 6, "cols": 6}}, **DATA})
        write_inputs(cfg)
        summary = build_datasets(cfg)
    counts = {split: info["n_windows"] for split, info in summary["splits"].items()}
    assert counts == {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}, counts
    return {key: paths[key] for key in SHARED_KEYS}


@pytest.fixture(scope="module")
def shared_data(tmp_path_factory) -> dict[str, str]:
    return build_shared(tmp_path_factory.mktemp("training_data"))


@pytest.fixture
def tcfg(cfg, shared_data) -> dict:
    """Test config: shared datasets, per-test checkpoint/report dirs (tmp_path), tiny model."""
    return deep_merge(cfg, {"paths": shared_data, **DATA, **SMALL})


@pytest.fixture
def log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def with_(cfg: dict, **sections) -> dict:
    return deep_merge(cfg, sections)


def dirs(cfg: dict, root: Path) -> dict:
    return with_(cfg, paths={"checkpoint_dir": str(root / "ckpt"), "reports_dir": str(root / "reports")})


def modified_payload(source: str, target: Path, **changes) -> str:
    payload = torch.load(source, weights_only=True)
    payload.update(changes)
    torch.save(payload, target)
    return str(target)


def warnings_text(log) -> str:
    return "\n".join(r.getMessage() for r in log.records if r.levelno >= logging.WARNING)
