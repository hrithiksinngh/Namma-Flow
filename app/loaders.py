"""Cached loaders of the Namma-Flow dashboard (config, predictors, scenarios, predictions, files).

Every loader is keyed on its inputs plus the identity (path, mtime, size) of the files it reads,
so an edited config, a re-trained checkpoint or a refreshed weather record is picked up on the
next rerun. Arguments whose names start with ``_`` are not hashed by Streamlit (predictor,
scenario and config objects); an explicit key argument stands in for them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import components as ui
import map_layers as ml
from src.data_pipeline.weather import WeatherUnavailable
from src.inference import predictor as pr
from src.inference import scenarios as sc
from src.inference.results import PredictionResult
from src.utils.config import load_config


def stamp(path: Path) -> tuple[str, int, int]:
    """Cache key of a file's identity: (path, mtime_ns, size); missing files give (path, 0, -1)."""
    try:
        info = Path(path).stat()
    except OSError:
        return str(path), 0, -1
    return str(path), info.st_mtime_ns, info.st_size


def digest(value: Any) -> str:
    """Short stable hash of a JSON-serialisable value (config cache keys)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def scenario_id(scenario: sc.Scenario) -> str:
    """Cache key of a scenario: kind, name, target start, first hour, length and the rain / field hashes."""
    hashed = hashlib.sha256(np.ascontiguousarray(scenario.areal_mm).tobytes())
    hashed.update(np.ascontiguousarray(scenario.is_forecast).tobytes())
    head = f"{scenario.kind}|{scenario.name}|{scenario.target_start}|{scenario.timestamps[0].isoformat()}"
    return f"{head}|{scenario.n_hours}|{scenario.field_shift_h}|{int(scenario.exact_field)}|{hashed.hexdigest()[:16]}"


@st.cache_data(show_spinner=False, max_entries=8)
def cached_config(path: str, file_stamp: tuple, offline_env: str) -> dict:
    return load_config(path)


@st.cache_resource(show_spinner="Loading the road graph…", max_entries=4)
def physics_predictor(cfg_key: str, graph_stamp: tuple, _cfg: dict) -> pr.PhysicsPredictor:
    return pr.PhysicsPredictor.from_graph(_cfg)


@st.cache_resource(show_spinner="Loading the trained model…", max_entries=4)
def gnn_predictor(cfg_key: str, graph_stamp: tuple, checkpoint_stamp: tuple,
                  _cfg: dict) -> tuple[pr.FloodPredictor | None, str | None, str | None]:
    """``(GNN, None, None)``, or ``(None, what went wrong, how to fix it)`` when the model is missing, not
    finalized or does not fit the graph / data (the fix is chosen by the error type)."""
    try:
        return pr.FloodPredictor.from_artifacts(_cfg), None, None
    except pr.ModelNotReady as exc:
        what, hint = pr.remediation(exc)
        return None, what, hint


@st.cache_data(show_spinner="Fetching the live Open-Meteo forecast…", max_entries=4)
def forecast(cfg_key: str, bucket: int, hours: int, _cfg: dict) -> tuple[sc.Scenario | None, str | None]:
    """The live forecast scenario or the failure reason (failures are cached too, so a dead API
    does not stall every rerun; the sidebar's Refresh button clears this cache)."""
    try:
        return sc.forecast_scenario(_cfg, hours=hours), None
    except (WeatherUnavailable, sc.ScenarioError, ValueError) as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False, max_entries=4)
def events(cfg_key: str, weather_stamp: tuple, top_k: int, _cfg: dict) -> pd.DataFrame:
    return sc.list_notable_events(_cfg, top_k=top_k)


@st.cache_data(show_spinner=False, max_entries=4)
def record_blocks(cfg_key: str, weather_stamp: tuple, _cfg: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
    """``[(first, last)]`` hour of every stored block of the weather record (None when there is no record)."""
    try:
        record = sc.load_weather_record(_cfg)
    except sc.ScenarioError as exc:
        ui.LOGGER.warning("%s", exc)
        return None
    return sc.block_spans(record) or None


@st.cache_data(show_spinner="Preparing the replay…", max_entries=16)
def historical(cfg_key: str, weather_stamp: tuple, start: str, hours: int, _cfg: dict) -> sc.Scenario:
    return sc.historical_scenario(_cfg, start, hours)


@st.cache_data(show_spinner=False, max_entries=16)
def design(cfg_key: str, total_mm: float, duration_h: int, offset_h: int, rain_scale: float, now: str,
           hours: int, _cfg: dict) -> sc.Scenario:
    return sc.design_storm_scenario(_cfg, total_mm, duration_h, offset_h, hours, now=now, rain_scale=rain_scale)


@st.cache_data(show_spinner="Predicting junction flood probabilities…", max_entries=24)
def predict(predictor_id: str, scenario_key: str, mc_samples: int, _predictor: Any,
            _scenario: sc.Scenario) -> PredictionResult:
    """Predict with the simulated depth and each rain-field member's probabilities (for the share at risk)."""
    return _predictor.predict(_scenario, mc_samples=mc_samples, include_physics=True, keep_members=True)


@st.cache_data(show_spinner=False, max_entries=8)
def geojson_bytes(prediction_id: str, hours: int, threshold: float, _result: PredictionResult) -> bytes:
    return json.dumps(_result.to_geojson(hours, include_timeline=True), separators=(",", ":")).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=16)
def json_file(path: str, file_stamp: tuple) -> dict | None:
    return ui.read_json_file(path)


@st.cache_data(show_spinner=False, max_entries=16)
def csv_file(path: str, file_stamp: tuple) -> pd.DataFrame | None:
    return ui.read_csv_file(path)


@st.cache_data(show_spinner=False, max_entries=4)
def waterways(path: str, file_stamp: tuple, exclude_water: tuple[str, ...] = ()) -> ml.WaterLayerData | None:
    return ml.load_waterways(path, exclude_water=exclude_water)
