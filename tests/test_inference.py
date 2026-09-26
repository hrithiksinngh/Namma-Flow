"""Tests for the inference package: scenarios, predictors and prediction results.

Everything runs offline on a 4 x 4 synthetic street grid with a tiny, untrained GATv2-GRU
checkpoint written with the exact training checkpoint keys (format v2 of the fix-round
contract X3: Platt ``calibration``, ``graph_attributes_sha256``, ``finalized_utc``; ``version=1``
gives the legacy temperature-only format).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import weather
from src.data_pipeline.dataset import FloodSequenceDataset, dataset_config_hash
from src.data_pipeline.features import FeatureScaler, dynamic_feature_names
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.data_pipeline.rain_field import downscale_rainfall
from src.inference import scenarios as sc
from src.inference.predictor import (
    CheckpointMismatch,
    FloodPredictor,
    GraphNotReady,
    ModelNotReady,
    PhysicsPredictor,
    PredictionError,
    load_checkpoint_file,
    load_predictor,
)
from src.models.stgcn import build_model
from src.utils.config import ConfigError, deep_merge, resolve_path
from src.utils.runtime import atomic_torch_save
from tests.conftest import make_grid_graph

TZ = "Asia/Kolkata"
STATIC = ["elevation", "dist_to_drain_m", "relative_elevation"]
EDGES = ["length", "grade"]
WINDOWS = [3, 6, 12, 24]
NOW = pd.Timestamp("2024-10-10 09:30", tz=TZ)
PLATT = {"method": "platt", "slope": 0.8, "intercept": -0.5}


# --------------------------------------------------------------------------- fixtures & helpers


@pytest.fixture
def log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def _warned(log, text: str) -> bool:
    return any(text in r.getMessage() for r in log.records if r.levelno >= logging.WARNING)


@pytest.fixture
def icfg(cfg):
    """Config with a saved 4 x 4 grid graph (and small inference knobs)."""
    save_graph(make_grid_graph(4, 4), cfg["paths"]["graph_file"])
    return deep_merge(cfg, {"inference": {"mc_dropout_samples": 3, "batch_windows": 4}})


@pytest.fixture
def graph(icfg):
    return graph_to_arrays(load_graph(icfg["paths"]["graph_file"]))


def make_checkpoint(cfg, graph, *, hidden: int = 8, dropout: float = 0.3, seed: int = 0, version: int = 2,
                    **overrides) -> dict:
    """A finalized checkpoint dict with every key of the training contract (untrained weights).

    ``version=2`` (default) carries a Platt calibration and the graph attribute digest; ``version=1``
    is the legacy format (``p = sigmoid(logit / temperature)``, no digest). The RNG state is in the
    ``weights_only``-safe builtin form the trainer writes.
    """
    torch.manual_seed(seed)
    model = build_model({"architecture": "gatv2_gru", "node_in_dim": 8, "edge_dim": 2, "hidden_dim": hidden,
                         "heads": 1, "dropout": dropout})
    dynamic = dynamic_feature_names(WINDOWS)
    samples = np.random.default_rng(seed).gamma(0.5, 3.0, size=(400, len(dynamic)))
    scaler = FeatureScaler.fit(graph.node_matrix(STATIC), samples, graph.edge_attrs["length"], 0.3,
                               static_names=STATIC, dynamic_names=dynamic)
    checkpoint = {
        "format_version": version, "model_state": model.state_dict(), "architecture": "gatv2_gru",
        "model_config": model.get_config(), "feature_names": [*STATIC, *dynamic], "static_feature_names": STATIC,
        "rolling_windows_h": WINDOWS, "lookback_hours": 24, "seq_len": 24, "warmup_steps": 6,
        "scaler": scaler.to_dict(), "graph_signature": graph.signature(), "node_ids": list(graph.node_ids),
        "temperature": 1.5 if version == 1 else 1.0 / PLATT["slope"], "threshold": 0.4,
        "metrics": {"pr_auc": 0.42, "roc_auc": 0.9}, "epoch": 3,
        "best_score": 0.42, "history": [{"epoch": 1}], "optimizer_state": {}, "scheduler_state": {},
        "rng_state": {"numpy": {"name": "MT19937", "keys": [1, 2, 3], "pos": 0, "has_gauss": 0,
                                "cached_gaussian": 0.0}, "torch": torch.get_rng_state()},
        "config_hash": "cfg", "dataset_config_hash": dataset_config_hash(cfg), "created_utc": "2024-01-01T00:00:00",
        "finalized_utc": "2024-01-01T00:05:00+00:00",
    }
    if version >= 2:
        checkpoint["calibration"] = dict(PLATT)
        checkpoint["graph_attributes_sha256"] = graph.attributes_signature(STATIC, EDGES)
    checkpoint.update(overrides)
    return checkpoint


def platt(logits: torch.Tensor) -> np.ndarray:
    """Reference Platt probabilities of :data:`PLATT` (independent of the library implementation)."""
    return torch.sigmoid(logits * PLATT["slope"] + PLATT["intercept"]).numpy()


def save_checkpoint(cfg, checkpoint: dict) -> Path:
    path = resolve_path(cfg, "checkpoint_dir") / "best.pt"
    atomic_torch_save(checkpoint, path)
    return path


@pytest.fixture
def ckpt_path(icfg, graph):
    return save_checkpoint(icfg, make_checkpoint(icfg, graph))


@pytest.fixture
def predictor(icfg, ckpt_path):
    return FloodPredictor.from_artifacts(icfg)


def storm_record() -> pd.DataFrame:
    """Schema-2.2 record 2022-09-01 .. 2022-09-20 with three distinct events."""
    index = pd.date_range("2022-09-01 00:00", "2022-09-20 23:00", freq="h", tz=TZ, name="timestamp")
    rain = pd.Series(0.0, index=index)
    rain.loc["2022-09-05 14:00":"2022-09-05 20:00"] = [2, 10, 25, 30, 12, 4, 1]
    rain.loc["2022-09-12 03:00":"2022-09-12 05:00"] = [5, 8, 3]
    rain.loc["2022-09-14 00:00":"2022-09-16 12:00"] = 0.5          # long light event ...
    rain.loc["2022-09-16 06:00":"2022-09-16 08:00"] = [6, 9, 4]     # ... with a burst
    return pd.DataFrame({"precipitation_mm": rain.to_numpy(), "is_imputed": False, "source": "open_meteo"},
                        index=index)


@pytest.fixture
def record_cfg(icfg):
    weather.save_weather_csv(storm_record(), icfg["paths"]["weather_file"])
    return icfg


def custom(cfg, hours: int = 96, storm_at: int = 60, target_start: int = 48, peak: float = 20.0) -> sc.Scenario:
    index = pd.date_range("2024-09-01", periods=hours, freq="h", tz=TZ)
    rain = np.zeros(hours)
    rain[storm_at: storm_at + 4] = [peak / 4, peak, peak / 2, peak / 8]
    return sc.custom_scenario(cfg, pd.Series(rain, index=index), target_start=target_start)


# --------------------------------------------------------------------------- settings & Scenario


@pytest.mark.unit
def test_settings_defaults(cfg):
    settings = sc.InferenceSettings.from_config(cfg)
    assert settings.horizons_h == (12, 24, 36, 48) and settings.max_horizon_h == 48
    assert [n for n, _ in settings.risk_tiers] == ["Low", "Moderate", "High", "Severe"]
    assert len(settings.design_storms) == 3 and settings.design_storms[2].duration_h == 6
    assert sc.InferenceSettings.from_config({}).batch_windows == sc.DEFAULTS["batch_windows"]


@pytest.mark.unit
@pytest.mark.parametrize("override, message", [
    ({"risk_tiers": {"low": 0.1, "high": 0.5}}, "start at 0.0"),
    ({"risk_tiers": {"low": 0.0, "high": 0.0}}, "distinct"),
    ({"risk_tiers": []}, "non-empty mapping"),
    ({"horizons_h": []}, "must not be empty"),
    ({"horizons_h": [0]}, "horizons_h"),
    ({"design_storm_shape": "square"}, "design_storm_shape"),
    ({"num_threads": 0}, "num_threads"),
    ({"batch_windows": True}, "batch_windows"),
    ({"design_storms": [{"total_mm": 10}]}, "duration_h"),
    ({"design_storms": "storm"}, "design_storms"),
    ({"horizons_h": [400]}, "max_scenario_hours"),
    ({"checkpoint_name": " "}, "checkpoint_name"),
    ({"physics_softness_m": -1}, "physics_softness_m"),
])
def test_settings_validation(cfg, override, message):
    with pytest.raises(ConfigError, match=message):
        sc.InferenceSettings.from_config(deep_merge(cfg, {"inference": override}))


@pytest.mark.unit
def test_scenario_is_validated_copied_and_frozen():
    index = pd.date_range("2024-10-01", periods=4, freq="h", tz=TZ)
    areal = np.array([0.0, 1.0, 2.0, 0.5])
    scenario = sc.Scenario("s", "custom", index, areal, None, 1, "d", "custom")
    areal[0] = 99.0
    assert scenario.areal_mm[0] == 0.0 and not scenario.areal_mm.flags.writeable
    assert scenario.n_target_hours == 3 and scenario.history_hours == 1
    assert scenario.rain_total_mm() == 3.5 and scenario.rain_total_mm(1) == 1.0
    assert list(scenario.target_timestamps) == list(index[1:])
    frame = scenario.to_frame()
    assert frame["is_target"].tolist() == [False, True, True, True] and not frame["is_forecast"].any()
    with pytest.raises(AttributeError):
        scenario.name = "other"


@pytest.mark.unit
@pytest.mark.parametrize("change, message", [
    ({"kind": "tomorrow"}, "kind"),
    ({"timestamps": pd.date_range("2024-10-01", periods=4, freq="h")}, "tz-aware"),
    ({"timestamps": pd.DatetimeIndex(["2024-10-01 00:00", "2024-10-01 02:00", "2024-10-01 03:00",
                                      "2024-10-01 04:00"]).tz_localize(TZ)}, "gap-free"),
    ({"timestamps": pd.DatetimeIndex([], tz=TZ)}, "at least one"),
    ({"timestamps": list(range(4))}, "DatetimeIndex"),
    ({"areal_mm": [0.0, 1.0]}, "has 2 values"),
    ({"areal_mm": [0.0, np.nan, 1.0, 0.0]}, "finite"),
    ({"areal_mm": [0.0, -1.0, 1.0, 0.0]}, "finite"),
    ({"areal_mm": ["a", "b", "c", "d"]}, "numeric"),
    ({"is_forecast": [True]}, "is_forecast"),
    ({"target_start": 4}, "target_start"),
    ({"target_start": True}, "target_start"),
    ({"target_start": 1.0}, "target_start"),
])
def test_scenario_rejects_bad_input(change, message):
    fields = dict(name="s", kind="custom", timestamps=pd.date_range("2024-10-01", periods=4, freq="h", tz=TZ),
                  areal_mm=np.zeros(4), is_forecast=None, target_start=0)
    fields.update(change)
    with pytest.raises(ValueError, match=message):
        sc.Scenario(**fields)


# --------------------------------------------------------------------------- forecast & design storms


def fake_forecast_frame(now: pd.Timestamp, future_rain: float = 12.0, days_ahead: int = 3) -> pd.DataFrame:
    index = pd.date_range(now.floor("D") - pd.Timedelta(days=2), periods=(2 + days_ahead) * 24, freq="h", tz=TZ,
                          name="timestamp")
    rain = np.zeros(len(index))
    rain[index > now + pd.Timedelta(hours=5)] = 0.0
    rain[(index > now + pd.Timedelta(hours=10)) & (index <= now + pd.Timedelta(hours=13))] = future_rain
    return pd.DataFrame({"precipitation_mm": rain, "is_imputed": False, "source": "forecast",
                         "is_forecast": index > now.floor("h")}, index=index)


@pytest.mark.unit
def test_forecast_scenario_targets_the_next_hour(cfg, monkeypatch):
    monkeypatch.setattr(sc.weather, "fetch_forecast", lambda cfg, now=None: fake_forecast_frame(NOW))
    scenario = sc.forecast_scenario(cfg, now=NOW)
    assert scenario.kind == "forecast" and scenario.source == "forecast"
    assert scenario.timestamps[scenario.target_start] == NOW.floor("h") + pd.Timedelta(hours=1)
    assert scenario.n_target_hours == 48 and scenario.is_forecast[scenario.target_start:].all()
    assert not scenario.is_forecast[: scenario.target_start].any()
    assert scenario.rain_total_mm() == pytest.approx(36.0)


@pytest.mark.unit
def test_forecast_scenario_failures(cfg, monkeypatch, log):
    with pytest.raises(sc.WeatherUnavailable, match="Offline"):
        sc.forecast_scenario(cfg)  # the real client refuses offline
    past = fake_forecast_frame(NOW).assign(is_forecast=False)
    monkeypatch.setattr(sc.weather, "fetch_forecast", lambda cfg, now=None: past)
    with pytest.raises(sc.WeatherUnavailable, match="no hours after"):
        sc.forecast_scenario(cfg, now=NOW)
    monkeypatch.setattr(sc.weather, "fetch_forecast", lambda cfg, now=None: fake_forecast_frame(NOW))
    assert sc.forecast_scenario(cfg, now=NOW, hours=12).n_target_hours == 12
    short = fake_forecast_frame(NOW, days_ahead=1)
    monkeypatch.setattr(sc.weather, "fetch_forecast", lambda cfg, now=None: short)
    scenario = sc.forecast_scenario(cfg, now=NOW)
    assert scenario.n_target_hours < 48 and _warned(log, "Forecast covers only")


@pytest.mark.unit
def test_design_storm_scenario(cfg):
    scenario = sc.design_storm_scenario(cfg, 80.0, 3, storm_start_offset_h=6, horizon_h=48, now=NOW, rain_scale=1.0)
    assert scenario.kind == "design_storm" and scenario.target_start == 48 and scenario.n_target_hours == 48
    assert scenario.rain_total_mm() == pytest.approx(80.0, abs=1e-9)
    assert scenario.areal_mm[:48].sum() == 0.0
    wet = np.flatnonzero(scenario.areal_mm > 0)
    assert wet[0] == 48 + 6 and wet[-1] <= 48 + 6 + 2
    assert scenario.timestamps[48] == NOW.floor("h") + pd.Timedelta(hours=1)
    assert scenario.is_forecast[48:].all() and not scenario.is_forecast[:48].any()
    scaled = sc.design_storm_scenario(cfg, 80.0, 3, now=NOW)
    assert scaled.rain_total_mm() == pytest.approx(80.0 * 0.14) and "areal-equivalent" in scaled.description
    assert scaled.name == "Design storm: 80 mm in 3 h"


@pytest.mark.unit
@pytest.mark.parametrize("kwargs, error, message", [
    ({"total_mm": 50, "duration_h": 10, "storm_start_offset_h": 45}, sc.ScenarioError, "does not fit"),
    ({"total_mm": 50, "duration_h": 3, "horizon_h": 500}, sc.ScenarioError, "max_scenario_hours"),
    ({"total_mm": -5, "duration_h": 3}, ValueError, "total_mm"),
    ({"total_mm": 50, "duration_h": 0}, ValueError, "duration_h"),
    ({"total_mm": 50, "duration_h": 3, "storm_start_offset_h": -1}, ValueError, "storm_start_offset_h"),
    ({"total_mm": 50, "duration_h": 3, "rain_scale": 0}, ValueError, "rain_scale"),
])
def test_design_storm_validation(cfg, kwargs, error, message):
    with pytest.raises(error, match=message):
        sc.design_storm_scenario(cfg, now=NOW, **kwargs)


@pytest.mark.unit
def test_preset_scenarios(cfg):
    by_index = sc.preset_scenario(cfg, 1, now=NOW)
    assert by_index.name.startswith("Heavy downpour")
    by_name = sc.preset_scenario(cfg, "cloud", now=NOW, rain_scale=1.0)
    assert by_name.name.startswith("Cloudburst") and by_name.rain_total_mm() == pytest.approx(130.0)
    assert sc.preset_scenario(cfg, "0", now=NOW).name.startswith("Moderate")
    for bad in ("nothing", 7, ""):
        with pytest.raises(sc.ScenarioError, match="preset"):
            sc.preset_scenario(cfg, bad, now=NOW)
    with pytest.raises(sc.ScenarioError, match="empty"):
        sc.preset_scenario(deep_merge(cfg, {"inference": {"design_storms": None}}), 0, now=NOW)


# --------------------------------------------------------------------------- historical replays & events


@pytest.mark.unit
def test_historical_scenario_basic(record_cfg):
    scenario = sc.historical_scenario(record_cfg, "2022-09-05 12:00", hours=24)
    assert scenario.kind == "historical" and scenario.source == "open_meteo"
    assert scenario.timestamps[scenario.target_start] == pd.Timestamp("2022-09-05 12:00", tz=TZ)
    assert scenario.history_hours == 48 and scenario.n_target_hours == 24
    assert scenario.rain_total_mm() == pytest.approx(84.0) and not scenario.is_forecast.any()
    utc = sc.historical_scenario(record_cfg, pd.Timestamp("2022-09-05 06:30", tz="UTC"), hours=6)
    assert utc.timestamps[utc.target_start] == pd.Timestamp("2022-09-05 12:00", tz=TZ)  # 12:00 IST, floored


@pytest.mark.unit
def test_historical_scenario_edges(record_cfg, log):
    tail = sc.historical_scenario(record_cfg, "2022-09-20 12:00", hours=48)
    assert tail.n_target_hours == 12 and _warned(log, "replay truncated")
    head = sc.historical_scenario(record_cfg, "2022-09-01 05:00", hours=6)
    assert head.history_hours == 5 and _warned(log, "Only 5 h of history")
    for start in ("2021-01-01", "2023-01-01"):
        with pytest.raises(sc.ScenarioError, match="outside the weather record"):
            sc.historical_scenario(record_cfg, start)
    with pytest.raises(sc.ScenarioError, match="ISO"):
        sc.historical_scenario(record_cfg, "not a date")
    with pytest.raises(sc.ScenarioError, match="max_scenario_hours"):
        sc.historical_scenario(record_cfg, "2022-09-05", hours=1000)
    with pytest.raises(ValueError, match="hours"):
        sc.historical_scenario(record_cfg, "2022-09-05", hours=0)


@pytest.mark.unit
def test_historical_scenario_missing_or_corrupt_record(icfg):
    with pytest.raises(sc.ScenarioError, match="03_weather_ingestion"):
        sc.historical_scenario(icfg, "2022-09-05")
    path = Path(icfg["paths"]["weather_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("garbage,columns\n1,2\n")
    with pytest.raises(sc.ScenarioError, match="unreadable"):
        sc.historical_scenario(icfg, "2022-09-05")


@pytest.mark.integration
def test_historical_replay_reproduces_the_training_rain_field(record_cfg, graph, log):
    """History starting inside a rain event is extended to the event start (exact field)."""
    scenario = sc.historical_scenario(record_cfg, "2022-09-16 06:00", hours=24)
    assert scenario.timestamps[0] == pd.Timestamp("2022-09-14 00:00", tz=TZ)   # event start, not 09-14 06:00
    assert scenario.history_hours == 54
    record = storm_record()
    areal, index = weather.areal_series(record)
    full = downscale_rainfall(areal, index, graph.lon, graph.lat, record_cfg)
    part = downscale_rainfall(scenario.areal_mm, scenario.timestamps, graph.lon, graph.lat, record_cfg)
    first = index.get_loc(scenario.timestamps[0])
    np.testing.assert_array_equal(part, full[first: first + scenario.n_hours])
    naive_first = index.get_loc(pd.Timestamp("2022-09-14 06:00", tz=TZ))   # what a plain 48 h history would use
    stop = first + scenario.n_hours
    naive = downscale_rainfall(areal[naive_first:stop], index[naive_first:stop], graph.lon, graph.lat, record_cfg)
    assert not np.allclose(naive[-24:], full[first + scenario.n_hours - 24: first + scenario.n_hours])
    later = sc.historical_scenario(record_cfg, "2022-09-16 10:00", hours=6)   # history starts mid-event
    assert later.timestamps[0] == pd.Timestamp("2022-09-14 00:00", tz=TZ) and later.history_hours == 58
    dry = sc.historical_scenario(record_cfg, "2022-09-08 00:00", hours=6)     # history starts in a dry spell
    assert dry.history_hours == 48
    limited = deep_merge(record_cfg, {"inference": {"max_event_backfill_h": 3}})
    capped = sc.historical_scenario(limited, "2022-09-16 10:00", hours=6)
    assert capped.history_hours == 48 and _warned(log, "longer than inference.max_event_backfill_h")


@pytest.mark.unit
def test_list_notable_events(record_cfg, icfg, log):
    events = sc.list_notable_events(record_cfg, top_k=5)
    assert list(events.columns) == list(sc.NOTABLE_COLUMNS)
    assert events["rank"].tolist() == list(range(1, len(events) + 1))
    assert events["total_mm"].is_monotonic_decreasing and events.iloc[0]["total_mm"] == pytest.approx(84.0)
    assert events.iloc[0]["peak_mm_h"] == 30.0
    gaps = np.diff(np.sort([t.value // 3_600_000_000_000 for t in events["start"]]))
    assert (gaps >= 72).all()
    assert len(sc.list_notable_events(record_cfg, top_k=1)) == 1
    with pytest.raises(ValueError, match="top_k"):
        sc.list_notable_events(record_cfg, top_k=0)
    empty = sc.list_notable_events(deep_merge(icfg, {"paths": {"weather_file": "/nonexistent/w.csv"}}))
    assert empty.empty and list(empty.columns) == list(sc.NOTABLE_COLUMNS) and _warned(log, "No notable events")


@pytest.mark.unit
def test_custom_scenario_cleaning(cfg, log):
    index = pd.date_range("2024-10-01", periods=6, freq="h", tz=TZ)
    rain = pd.Series([0.0, np.nan, -2.0, 4.0, 1.0, 2.0], index=index).drop(index[4])
    rain = pd.concat([rain, pd.Series([3.0], index=[index[5]])])  # duplicate (keep last)
    scenario = sc.custom_scenario(cfg, rain, target_start=2)
    assert scenario.areal_mm.tolist() == [0.0, 0.0, 0.0, 4.0, 0.0, 3.0]
    assert _warned(log, "NaN/non-finite/negative") and _warned(log, "missing hours") and _warned(log, "duplicate")
    frame = pd.DataFrame({"precipitation_mm": [1.0, 2.0, 3.0], "is_forecast": [False, True, True]},
                         index=pd.date_range("2024-10-01", periods=3, freq="h"))
    assert sc.custom_scenario(cfg, frame).target_start == 1
    assert sc.custom_scenario(cfg, frame[["precipitation_mm"]]).target_start == 2   # history_hours clipped
    for bad, message in ((np.zeros(3), "Series or DataFrame"), (pd.Series([1.0, 2.0]), "DatetimeIndex"),
                         (frame.rename(columns={"precipitation_mm": "x"}), "column"),
                         (pd.Series([1.0], index=pd.DatetimeIndex(["2024-10-01 00:30"])), "whole hours"),
                         (pd.Series([], dtype=float, index=pd.DatetimeIndex([])), "no valid")):
        with pytest.raises(sc.ScenarioError, match=message):
            sc.custom_scenario(cfg, bad)
    with pytest.raises(sc.ScenarioError, match="target_start"):
        sc.custom_scenario(cfg, frame, target_start=3)


# --------------------------------------------------------------------------- loading the GNN


@pytest.mark.unit
def test_missing_or_broken_checkpoint_raises_model_not_ready(icfg, graph, tmp_path):
    with pytest.raises(ModelNotReady, match="train.py"):
        FloodPredictor.from_artifacts(icfg)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ModelNotReady, match="unreadable"):
        FloodPredictor.from_artifacts(icfg, checkpoint_path=bad)
    torch.save([1, 2, 3], bad)
    with pytest.raises(ModelNotReady, match="not a checkpoint dict"):
        load_checkpoint_file(bad)
    partial = make_checkpoint(icfg, graph)
    del partial["scaler"]
    torch.save(partial, bad)
    with pytest.raises(ModelNotReady, match="missing keys"):
        load_checkpoint_file(bad)
    for version in (3, 0, True):
        torch.save(make_checkpoint(icfg, graph, format_version=version), bad)
        with pytest.raises(ModelNotReady, match="format_version"):
            load_checkpoint_file(bad)
    with pytest.raises(ModelNotReady, match="No trained model"):
        load_checkpoint_file(tmp_path)  # a directory


@pytest.mark.unit
def test_v2_checkpoint_loads_with_platt_calibration(icfg, ckpt_path, predictor):
    assert predictor.kind == "gnn" and predictor.threshold == pytest.approx(0.4)
    assert predictor.calibration == PLATT and predictor.temperature == pytest.approx(1.25)
    assert predictor.metrics["pr_auc"] == 0.42 and predictor.history_needed == 30 and not predictor.model.training
    card = predictor.describe()
    assert card["architecture"] == "gatv2_gru" and card["epoch"] == 3 and card["format_version"] == 2
    assert card["calibration"] == PLATT and card["finalized_utc"].startswith("2024-01-01")
    assert card["checkpoint"] == "best.pt"                         # never an absolute path (R5-09)
    assert card["dataset_config_hash"] == dataset_config_hash(icfg)


@pytest.mark.unit
def test_graph_mismatch_raises_checkpoint_mismatch(icfg, ckpt_path):
    save_graph(make_grid_graph(5, 5), icfg["paths"]["graph_file"])
    with pytest.raises(CheckpointMismatch, match="different road graph"):
        FloodPredictor.from_artifacts(icfg)
    with pytest.raises(ModelNotReady):  # subclass: callers can fall back on ModelNotReady alone
        FloodPredictor.from_artifacts(icfg)


@pytest.mark.unit
@pytest.mark.parametrize("override, error, message", [
    ({"feature_names": [*STATIC, "precip_mm_h", "x", "y", "z", "w"]}, CheckpointMismatch, "inconsistent"),
    ({"rolling_windows_h": [3, 6]}, CheckpointMismatch, "inconsistent"),
    ({"rolling_windows_h": [0]}, ModelNotReady, "rolling windows"),
    ({"scaler": {"version": 1}}, ModelNotReady, "scaler"),
    ({"seq_len": 4, "warmup_steps": 6}, ModelNotReady, "seq_len"),
    ({"seq_len": "abc"}, ModelNotReady, "integers"),
    ({"node_ids": ["a"] * 16}, CheckpointMismatch, "junction order"),
    ({"model_config": "gatv2"}, ModelNotReady, "mapping"),
    ({"model_config": {"architecture": "transformer"}}, ModelNotReady, "model_config is invalid"),
])
def test_inconsistent_checkpoints(icfg, graph, override, error, message):
    save_checkpoint(icfg, make_checkpoint(icfg, graph, **override))
    with pytest.raises(error, match=message):
        FloodPredictor.from_artifacts(icfg)


@pytest.mark.unit
def test_optional_checkpoint_keys_are_derived(icfg, graph):
    checkpoint = make_checkpoint(icfg, graph)
    for key in ("architecture", "static_feature_names", "metrics", "history", "optimizer_state", "rng_state"):
        del checkpoint[key]
    checkpoint["model_config"] = {k: v for k, v in checkpoint["model_config"].items() if k != "architecture"}
    save_checkpoint(icfg, checkpoint)
    predictor = FloodPredictor.from_artifacts(icfg)
    assert predictor.static_feature_names == STATIC and predictor.architecture == "gatv2_gru"
    assert predictor.metrics == {}


@pytest.mark.unit
def test_weight_and_dimension_mismatch(icfg, graph):
    checkpoint = make_checkpoint(icfg, graph)
    checkpoint["model_config"] = {**checkpoint["model_config"], "hidden_dim": 16}
    save_checkpoint(icfg, checkpoint)
    with pytest.raises(CheckpointMismatch, match="weights do not fit"):
        FloodPredictor.from_artifacts(icfg)
    checkpoint["model_config"] = {**checkpoint["model_config"], "node_in_dim": 5}
    save_checkpoint(icfg, checkpoint)
    with pytest.raises(CheckpointMismatch, match="node features"):
        FloodPredictor.from_artifacts(icfg)


@pytest.mark.unit
def test_graph_without_static_features(icfg, ckpt_path):
    G = make_grid_graph(4, 4)
    for _, data in G.nodes(data=True):
        del data["dist_to_drain_m"]
    save_graph(G, icfg["paths"]["graph_file"])
    with pytest.raises(CheckpointMismatch, match="dist_to_drain_m"):
        FloodPredictor.from_artifacts(icfg)


@pytest.mark.unit
def test_invalid_calibration_values_fall_back(icfg, graph, log):
    save_checkpoint(icfg, make_checkpoint(icfg, graph, version=1, temperature=-1.0, threshold=1.5))
    predictor = FloodPredictor.from_artifacts(icfg)
    assert predictor.temperature == 1.0 and predictor.threshold == 0.5
    assert predictor.calibration == {"method": "none", "slope": 1.0, "intercept": 0.0}
    assert _warned(log, "temperature") and _warned(log, "threshold")
    save_checkpoint(icfg, make_checkpoint(icfg, graph, version=1, temperature="hot", threshold=None))
    assert FloodPredictor.from_artifacts(icfg).threshold == 0.5
    broken = {"method": "platt", "slope": -2.0, "intercept": 0.0}     # v2 with an unusable dict -> its temperature
    save_checkpoint(icfg, make_checkpoint(icfg, graph, calibration=broken, temperature=2.0))
    fallback = FloodPredictor.from_artifacts(icfg)
    assert fallback.calibration["slope"] == pytest.approx(0.5) and _warned(log, "calibration is invalid")
    save_checkpoint(icfg, make_checkpoint(icfg, graph, calibration={"method": "magic"}, temperature=None))
    assert FloodPredictor.from_artifacts(icfg).calibration["slope"] == 1.0


@pytest.mark.unit
def test_config_drift_is_warned(icfg, graph, log):
    save_checkpoint(icfg, make_checkpoint(icfg, graph, dataset_config_hash="0000"))
    FloodPredictor.from_artifacts(icfg)
    assert _warned(log, "configuration (rain field / hydrology / features) changed")


@pytest.mark.unit
def test_missing_graph_raises_graph_not_ready(cfg, tmp_path):
    with pytest.raises(GraphNotReady, match="01_extract_network"):
        PhysicsPredictor.from_graph(cfg)
    path = Path(cfg["paths"]["graph_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<graphml>broken")
    with pytest.raises(GraphNotReady, match="unreadable"):
        PhysicsPredictor.from_graph(cfg)
    G = make_grid_graph(3, 3)
    for _, data in G.nodes(data=True):
        del data["elevation"]
    save_graph(G, path)
    with pytest.raises(GraphNotReady, match="hydrology simulator"):
        PhysicsPredictor.from_graph(cfg)


# --------------------------------------------------------------------------- GNN predictions


def _dataset_payload(graph, predictor, rain: np.ndarray, epoch: np.ndarray, starts: list[int]) -> dict:
    """A FloodSequenceDataset payload over the scenario's hours (the training code path)."""
    edge_raw = graph.edge_matrix(["length", "grade"])
    return {
        "format_version": 1, "split": "val", "node_ids": list(graph.node_ids),
        "lon": torch.from_numpy(graph.lon.copy()), "lat": torch.from_numpy(graph.lat.copy()),
        "edge_index": torch.from_numpy(graph.edge_index.copy()),
        "edge_attr": torch.from_numpy(predictor.scaler.transform_edges(edge_raw[:, 0], edge_raw[:, 1])),
        "edge_attr_raw": torch.from_numpy(edge_raw), "static_raw": torch.from_numpy(graph.node_matrix(STATIC)),
        "static_feature_names": STATIC, "feature_names": predictor.feature_names, "rain": torch.from_numpy(rain),
        "labels": torch.zeros(rain.shape, dtype=torch.uint8), "depth": torch.zeros(rain.shape, dtype=torch.float16),
        "timestamps": torch.from_numpy(epoch), "timezone": TZ, "window_starts": torch.tensor(starts),
        "seq_len": 24, "warmup_steps": 6, "lookback_hours": 24, "rolling_windows_h": WINDOWS,
        "scaler": predictor.scaler.to_dict(), "label_source": "simulated", "flood_threshold_m": 0.15,
        "graph_signature": graph.signature(), "config_hash": "x", "stats": {},
    }


@pytest.mark.integration
def test_predictions_match_the_training_window_path(icfg, graph, predictor):
    """Each scored hour equals sigmoid(logit / T) of the FloodSequenceDataset window covering it (the exact
    training field: one member with the base rain-field seed)."""
    scenario = dataclasses.replace(custom(icfg, hours=90, storm_at=55, target_start=40), exact_field=True)
    result = predictor.predict(scenario)
    assert result.prob.shape == (50, 16) and result.prob_std is None and result.depth_physics is None
    rain = downscale_rainfall(scenario.areal_mm, scenario.timestamps, graph.lon, graph.lat, icfg)
    epoch = scenario.timestamps.as_unit("s").asi8.astype(np.int64)
    starts = [40 - 6, 40 + 18 - 6]  # windows 0 and 1 (window 2 needs trailing padding)
    dataset = FloodSequenceDataset(_dataset_payload(graph, predictor, rain, epoch, starts))
    for k in range(2):
        item = dataset[k]
        with torch.no_grad():
            logits, _ = predictor.model.forward_sequence(item["x"], dataset.edge_index, dataset.edge_attr)
        expected = platt(logits)[6:]
        np.testing.assert_allclose(result.prob[k * 18: (k + 1) * 18], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.node_rain, rain[40:])
    assert list(result.timestamps) == list(scenario.target_timestamps)


@pytest.mark.integration
def test_batching_and_trailing_padding_do_not_change_results(icfg, graph, ckpt_path):
    scenario = custom(icfg, hours=100, storm_at=60, target_start=40)
    one = FloodPredictor.from_artifacts(deep_merge(icfg, {"inference": {"batch_windows": 1}})).predict(scenario)
    many = FloodPredictor.from_artifacts(deep_merge(icfg, {"inference": {"batch_windows": 8}})).predict(scenario)
    np.testing.assert_allclose(one.prob, many.prob, rtol=1e-5, atol=1e-6)
    shorter = sc.Scenario("cut", "custom", scenario.timestamps[:70], scenario.areal_mm[:70], None, 40)
    cut = FloodPredictor.from_artifacts(icfg).predict(shorter)  # causal model: later hours cannot matter
    np.testing.assert_allclose(cut.prob, many.prob[:30], rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_short_history_is_padded_with_a_warning(predictor, icfg, log):
    scenario = custom(icfg, hours=30, storm_at=5, target_start=4)
    result = predictor.predict(scenario)
    assert result.n_hours == 26 and np.isfinite(result.prob).all()
    assert _warned(log, "padding with dry hours")
    first_hour = predictor.predict(sc.Scenario("t0", "custom", scenario.timestamps, scenario.areal_mm, None, 0))
    assert first_hour.n_hours == 30


@pytest.mark.integration
def test_mc_dropout_mean_std_and_determinism(predictor, icfg):
    scenario = custom(icfg)
    plain = predictor.predict(scenario)
    mc = predictor.predict(scenario, mc_samples=4)
    again = predictor.predict(scenario, mc_samples=4)
    assert mc.prob_std is not None and mc.prob_std.shape == mc.prob.shape and mc.prob_std.max() > 0
    np.testing.assert_array_equal(mc.prob, again.prob)
    np.testing.assert_array_equal(mc.prob_std, again.prob_std)
    assert mc.metadata["mc_samples"] == 4 and plain.metadata["mc_samples"] == 0
    assert not predictor.model.training and not any(m.training for m in predictor.model.modules())
    np.testing.assert_array_equal(predictor.predict(scenario).prob, plain.prob)  # eval mode restored
    assert not np.allclose(mc.prob, plain.prob)


@pytest.mark.integration
def test_concurrent_predictions_do_not_interfere(predictor, icfg):
    """The app shares one predictor across sessions: an MC run must not leak dropout into others."""
    import threading

    scenario = custom(icfg)
    expected = predictor.predict(scenario).prob
    results: dict[str, np.ndarray] = {}

    def run(name: str, samples: int) -> None:
        results[name] = predictor.predict(scenario, mc_samples=samples).prob

    threads = [threading.Thread(target=run, args=(f"mc{i}", 3)) for i in range(2)]
    threads += [threading.Thread(target=run, args=(f"plain{i}", 0)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for i in range(3):
        np.testing.assert_array_equal(results[f"plain{i}"], expected)
    np.testing.assert_array_equal(results["mc0"], results["mc1"])


@pytest.mark.unit
def test_mc_dropout_without_dropout_gives_zero_std(icfg, graph, log):
    save_checkpoint(icfg, make_checkpoint(icfg, graph, dropout=0.0))
    exact = dataclasses.replace(custom(icfg), exact_field=True)             # no rain-field spread either
    result = FloodPredictor.from_artifacts(icfg).predict(exact, mc_samples=2)
    assert float(result.prob_std.max()) == pytest.approx(0.0, abs=1e-6) and _warned(log, "no dropout layer")


@pytest.mark.unit
@pytest.mark.parametrize("samples", [-1, True, 2.5, 10_000])
def test_mc_samples_validation(predictor, icfg, samples):
    with pytest.raises(ValueError, match="mc_samples"):
        predictor.predict(custom(icfg), mc_samples=samples)


@pytest.mark.unit
def test_predict_rain_validation(predictor):
    index = pd.date_range("2024-10-01", periods=10, freq="h", tz=TZ)
    good = np.zeros((10, 16), dtype=np.float32)
    cases = [
        (np.zeros((10, 15)), index, 0, "junctions"), (np.zeros((0, 16)), index[:0], 0, "T = 0"),
        (np.full((10, 16), np.nan), index, 0, "finite"), (-good - 1, index, 0, "finite"),
        (good, index[:5], 0, "DatetimeIndex of length"), (good, index, 10, "target_start"),
        (good, index, True, "target_start"), ([["a"] * 16] * 10, index, 0, "numeric"),
    ]
    for rain, stamps, start, message in cases:
        with pytest.raises(PredictionError, match=message):
            predictor.predict_rain(rain, stamps, start)
    with pytest.raises(TypeError, match="Scenario"):
        predictor.predict("tomorrow")


@pytest.mark.integration
def test_gnn_with_physics_depth(predictor, icfg):
    result = predictor.predict(custom(icfg, peak=40.0), include_physics=True)
    assert result.depth_physics is not None and result.depth_physics.shape == result.prob.shape
    assert result.metadata["architecture"] == "gatv2_gru" and result.metadata["temperature"] == pytest.approx(1.25)
    assert result.metadata["calibration"] == PLATT and result.metadata["checkpoint_epoch"] == 3


# --------------------------------------------------------------------------- physics baseline & selection


@pytest.mark.integration
def test_physics_predictor(icfg, log):
    physics = PhysicsPredictor.from_graph(icfg)
    dry = physics.predict(dataclasses.replace(custom(icfg, peak=0.0), exact_field=True))
    assert dry.prob.max() < 0.01 and dry.prob_std is None and physics.threshold == 0.5
    design = sc.design_storm_scenario(icfg, 80.0, 3, now=NOW, rain_scale=1.0)
    storm = physics.predict(dataclasses.replace(design, exact_field=True), mc_samples=5)   # one field: p <-> depth
    assert storm.prob.max() > 0.9 and storm.depth_physics.max() > 0.15 and _warned(log, "deterministic")
    table = storm.node_table(48)
    assert (table["above_threshold"] == (table["max_depth_m"] >= 0.15)).all()
    assert storm.metadata["predictor"] == "physics" and physics.metrics == {}


@pytest.mark.unit
def test_load_predictor_fallback(icfg, graph, log):
    predictor, reason = load_predictor(icfg)
    assert isinstance(predictor, PhysicsPredictor) and "No trained model" in reason
    assert _warned(log, "Falling back to the physics baseline")
    save_checkpoint(icfg, make_checkpoint(icfg, graph))
    predictor, reason = load_predictor(icfg)
    assert isinstance(predictor, FloodPredictor) and reason is None
    physics, reason = load_predictor(icfg, prefer="physics")
    assert isinstance(physics, PhysicsPredictor) and reason is None
    with pytest.raises(ValueError, match="prefer"):
        load_predictor(icfg, prefer="oracle")
