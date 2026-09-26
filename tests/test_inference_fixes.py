"""Regression tests for the fix-round findings of the inference package.

R3-03 / R5-01 (safe checkpoint loading), R4-02 / R5-05 (non-finalized checkpoints), X4
(Platt calibration, graph attribute digest), R4-03 (clock-independent design-storm rain
field), R4-04 (replays starting on a dry gap hour inside an event), R4-06 / R5-07 (preset
storm offset), R4-07 (insufficient history) and R5-09 (no absolute paths in exports).
Shares the offline fixtures of :mod:`tests.test_inference`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import weather
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.data_pipeline.rain_field import downscale_rainfall
from src.inference import scenarios as sc
from src.inference.predictor import (
    CheckpointMismatch,
    FloodPredictor,
    ModelNotReady,
    PhysicsPredictor,
    load_checkpoint_file,
    load_predictor,
)
from src.inference.results import display_path
from src.utils.config import ConfigError, deep_merge, project_root, resolve_path
from tests.conftest import make_grid_graph
from tests.test_inference import (  # noqa: F401 - fixtures are used by name
    NOW,
    TZ,
    _warned,
    ckpt_path,
    custom,
    graph,
    icfg,
    log,
    make_checkpoint,
    predictor,
    record_cfg,
    save_checkpoint,
)


# --------------------------------------------------------------------------- checkpoint loading (R3-03, R4-02, X4)


class _Payload:
    """A pickle that runs code when unpickled by the full unpickler (R3-03 / R5-01)."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


@pytest.mark.unit
def test_malicious_checkpoint_is_refused_without_running_code(icfg, graph, tmp_path):
    """Regression R3-03 / R5-01: the safe loader's rejection must not fall back to the full unpickler."""
    marker = tmp_path / "pwned"
    evil = tmp_path / "evil.pt"
    torch.save({"format_version": 2, "payload": _Payload(marker)}, evil)
    with pytest.raises(ModelNotReady, match="not a Namma-Flow checkpoint"):
        load_checkpoint_file(evil)
    assert not marker.exists()
    legacy = make_checkpoint(icfg, graph, rng_state={"legacy": np.random.get_state()})  # numpy objects
    torch.save(legacy, evil)
    with pytest.raises(ModelNotReady, match=r"safe \(weights-only\) loader refused"):
        FloodPredictor.from_artifacts(icfg, checkpoint_path=evil)
    predictor, reason = load_predictor(icfg, checkpoint_path=evil)
    assert isinstance(predictor, PhysicsPredictor) and "not a Namma-Flow checkpoint" in reason and not marker.exists()


@pytest.mark.unit
def test_v1_checkpoint_uses_its_temperature(icfg, graph, log):
    save_checkpoint(icfg, make_checkpoint(icfg, graph, version=1))
    predictor = FloodPredictor.from_artifacts(icfg)
    assert predictor.calibration == {"method": "temperature", "slope": pytest.approx(1 / 1.5), "intercept": 0.0}
    assert predictor.temperature == pytest.approx(1.5)
    assert _warned(log, "records no graph attribute digest")       # v1: warning only, still served
    x = np.random.default_rng(1).normal(size=(1, 24, 16, 8)).astype(np.float32)
    with torch.no_grad():
        logits, _ = predictor.model.forward_sequence(torch.from_numpy(x[0]), predictor.edge_index,
                                                     predictor.edge_attr, return_logits=True)
    np.testing.assert_allclose(predictor._forward(x)[0], torch.sigmoid(logits / 1.5).numpy(), rtol=1e-5, atol=1e-6)
    assert predictor.predict(custom(icfg)).prob.shape == (48, 16)


@pytest.mark.unit
def test_non_finalized_checkpoint_is_not_served(icfg, graph, log):
    """Regression R4-02 / R5-05: a per-epoch (in-progress / interrupted) file must not be served as the model."""
    for missing in ({"finalized_utc": None}, {"finalized_utc": ""}):
        save_checkpoint(icfg, make_checkpoint(icfg, graph, **missing))
        with pytest.raises(ModelNotReady, match="checkpoint is not finalized"):
            FloodPredictor.from_artifacts(icfg)
    checkpoint = make_checkpoint(icfg, graph)
    del checkpoint["finalized_utc"]
    save_checkpoint(icfg, checkpoint)
    predictor, reason = load_predictor(icfg)
    assert isinstance(predictor, PhysicsPredictor) and "not finalized" in reason and "--resume" in reason
    with pytest.raises(ModelNotReady, match="given checkpoint is not finalized"):
        FloodPredictor(icfg, graph, checkpoint)


@pytest.mark.unit
def test_graph_attribute_digest_mismatch_raises_checkpoint_mismatch(icfg, graph, log):
    """X4: same topology, re-enriched static attributes -> CheckpointMismatch with a rebuild hint."""
    save_checkpoint(icfg, make_checkpoint(icfg, graph))
    G = make_grid_graph(4, 4)
    for _, data in G.nodes(data=True):
        data["elevation"] = float(data["elevation"]) + 0.5            # a new DEM; topology unchanged
    save_graph(G, icfg["paths"]["graph_file"])
    assert graph_to_arrays(load_graph(icfg["paths"]["graph_file"])).signature() == graph.signature()
    with pytest.raises(CheckpointMismatch, match="rebuild the datasets and retrain"):
        FloodPredictor.from_artifacts(icfg)
    predictor, reason = load_predictor(icfg)
    assert isinstance(predictor, PhysicsPredictor) and "attributes" in reason
    save_checkpoint(icfg, make_checkpoint(icfg, graph, graph_attributes_sha256=None))
    assert FloodPredictor.from_artifacts(icfg).kind == "gnn" and _warned(log, "no graph attribute digest")


@pytest.mark.unit
def test_exported_metadata_has_no_absolute_paths(predictor, icfg):
    """Regression R5-09: the GeoJSON run metadata keeps the checkpoint's file name and identity only."""
    collection = predictor.predict(custom(icfg)).to_geojson(12)
    text = json.dumps(collection)
    run = collection["metadata"]["run"]
    assert run["checkpoint"] == "best.pt" and run["checkpoint_finalized_utc"].startswith("2024-01-01")
    assert str(resolve_path(icfg, "checkpoint_dir")) not in text and str(Path.home()) not in text




# --------------------------------------------------------------------------- design-storm rain field (R4-03)


def _field(scenario: sc.Scenario, graph, cfg, axis: str = "field") -> np.ndarray:
    stamps = scenario.field_timestamps if axis == "field" else scenario.timestamps
    return downscale_rainfall(scenario.areal_mm, stamps, graph.lon, graph.lat, cfg)


@pytest.mark.integration
def test_design_storm_junction_field_is_clock_and_offset_independent(icfg, graph):
    """Regression R4-03: the same what-if storm gives the same junction field at any wall-clock hour."""
    a = sc.design_storm_scenario(icfg, 130.0, 6, 6, 48, now="2026-09-23T20:30", rain_scale=1.0)
    b = sc.design_storm_scenario(icfg, 130.0, 6, 6, 48, now="2026-09-23T21:30", rain_scale=1.0)
    c = sc.design_storm_scenario(icfg, 130.0, 6, 0, 48, now="2026-09-24T09:10", rain_scale=1.0)
    assert a.timestamps[0] != b.timestamps[0] and np.array_equal(a.areal_mm, b.areal_mm)
    assert a.field_timestamps[48 + 6] == pd.Timestamp("2000-01-01 00:00", tz=TZ) == c.field_timestamps[48]
    assert not np.allclose(_field(a, graph, icfg, "real"), _field(b, graph, icfg, "real"))  # the old behaviour
    np.testing.assert_array_equal(_field(a, graph, icfg), _field(b, graph, icfg))
    np.testing.assert_array_equal(_field(a, graph, icfg)[54:60], _field(c, graph, icfg)[48:54])
    physics = PhysicsPredictor.from_graph(icfg)
    first, second = physics.predict(a), physics.predict(b)
    np.testing.assert_array_equal(first.prob, second.prob)
    assert list(first.top_k(5, 48)["node_id"]) == list(second.top_k(5, 48)["node_id"])
    assert first.metadata["rain_field_shift_h"] == a.field_shift_h and "canonical" in first.metadata["rain_field"]
    preset = sc.preset_scenario(icfg, "cloud", now="2026-09-23T20:30")
    assert preset.field_shift_h and preset.name.startswith("Cloudburst")


@pytest.mark.unit
def test_rain_field_axis_by_scenario_kind(icfg, monkeypatch):
    from tests.test_inference import fake_forecast_frame

    monkeypatch.setattr(sc.weather, "fetch_forecast", lambda cfg, now=None: fake_forecast_frame(NOW))
    forecast = sc.forecast_scenario(icfg, now=NOW)
    assert forecast.field_shift_h == 0 and forecast.field_timestamps.equals(forecast.timestamps)
    assert "averaged over stochastic rain-field realisations" in forecast.rain_field_note and not forecast.notes
    assert not forecast.exact_field                                            # F1-01: the field is unknown
    short = sc.forecast_scenario(deep_merge(icfg, {"inference": {"horizons_h": [12, 96]}}), now=NOW)
    assert short.notes and "covers only" in short.notes[0]
    moved = sc.design_storm_scenario(deep_merge(icfg, {"inference": {"design_storm_field_anchor": "2010-06-01T05:00"}}),
                                     30.0, 3, 6, 48, now=NOW)
    assert moved.field_timestamps[54] == pd.Timestamp("2010-06-01 05:00", tz=TZ)
    for bad in ("tomorrow", "2000-01-01T00:30", "2000-01-01T00:00+05:30"):
        with pytest.raises(ConfigError, match="design_storm_field_anchor"):
            sc.InferenceSettings.from_config(deep_merge(icfg, {"inference": {"design_storm_field_anchor": bad}}))
    index = pd.date_range("2024-10-01", periods=3, freq="h", tz=TZ)
    assert sc.Scenario("s", "custom", index, np.zeros(3), None, 0, notes="one").notes == ("one",)
    assert "only corridor-average rain" in sc.Scenario("s", "design_storm", index, np.zeros(3), None, 0).rain_field_note
    assert "canonical storm-anchored" in moved.rain_field_note
    assert "exact training field" in sc.Scenario("s", "historical", index, np.zeros(3), None, 0,
                                                 exact_field=True).rain_field_note
    for bad in (True, 1.5, "2"):
        with pytest.raises(ValueError, match="field_shift_h"):
            sc.Scenario("s", "custom", index, np.zeros(3), None, 0, field_shift_h=bad)


# --------------------------------------------------------------------------- replays (R4-04)


def gap_record() -> pd.DataFrame:
    """An event with a 2 h dry gap (<= rainfall_field.max_dry_gap_h) on 2022-09-10."""
    index = pd.date_range("2022-09-01 00:00", "2022-09-20 23:00", freq="h", tz=TZ, name="timestamp")
    rain = pd.Series(0.0, index=index)
    rain.loc["2022-09-10 10:00":"2022-09-10 12:00"] = [4.0, 9.0, 3.0]
    rain.loc["2022-09-10 15:00":"2022-09-10 18:00"] = [6.0, 12.0, 5.0, 1.0]   # 13:00-14:00 dry, same event
    return pd.DataFrame({"precipitation_mm": rain.to_numpy(), "is_imputed": False, "source": "open_meteo"},
                        index=index)


@pytest.mark.integration
def test_replay_starting_on_a_dry_gap_hour_reproduces_the_training_field(icfg, graph):
    """Regression R4-04: the in-progress event is found even when the history's first hour is a dry gap hour."""
    record = gap_record()
    weather.save_weather_csv(record, icfg["paths"]["weather_file"])
    scenario = sc.historical_scenario(icfg, "2022-09-12 13:00", hours=6)       # history would start 09-10 13:00
    assert scenario.timestamps[0] == pd.Timestamp("2022-09-10 10:00", tz=TZ) and not scenario.notes
    areal, index = weather.areal_series(record)
    full = downscale_rainfall(areal, index, graph.lon, graph.lat, icfg)
    first = index.get_loc(scenario.timestamps[0])
    np.testing.assert_array_equal(_field(scenario, graph, icfg), full[first: first + scenario.n_hours])
    later = sc.historical_scenario(icfg, "2022-09-12 14:00", hours=6)          # second gap hour
    assert later.timestamps[0] == pd.Timestamp("2022-09-10 10:00", tz=TZ)
    after = sc.historical_scenario(icfg, "2022-09-12 19:00", hours=6)          # dry hour after the event
    assert after.timestamps[0] == pd.Timestamp("2022-09-10 19:00", tz=TZ)


@pytest.mark.unit
def test_backfill_limit_is_surfaced_as_a_scenario_note(record_cfg, log):
    capped = sc.historical_scenario(deep_merge(record_cfg, {"inference": {"max_event_backfill_h": 3}}),
                                    "2022-09-16 10:00", hours=6)
    assert capped.history_hours == 48 and len(capped.notes) == 1 and "max_event_backfill_h" in capped.notes[0]
    assert _warned(log, "longer than inference.max_event_backfill_h")


# --------------------------------------------------------------------------- presets (R4-06 / R5-07)


@pytest.mark.unit
def test_preset_scenario_passes_the_storm_offset_through(icfg):
    default = sc.preset_scenario(icfg, "cloud", now=NOW)
    assert np.flatnonzero(default.areal_mm > 0)[0] == 48 + 6
    early = sc.preset_scenario(icfg, "cloud", horizon_h=8, now=NOW, storm_start_offset_h=0)
    assert early.n_target_hours == 8 and np.flatnonzero(early.areal_mm > 0)[0] == 48
    assert "starting 0 h into a 8 h horizon" in early.description
    with pytest.raises(sc.ScenarioError, match="does not fit"):
        sc.preset_scenario(icfg, "cloud", horizon_h=8, now=NOW)


# --------------------------------------------------------------------------- insufficient history (R4-07)


@pytest.mark.integration
def test_short_history_is_flagged_in_metadata(icfg, graph, log):
    save_checkpoint(icfg, make_checkpoint(icfg, graph))
    short = deep_merge(icfg, {"inference": {"history_hours": 12}})
    predictor = FloodPredictor.from_artifacts(short)
    assert _warned(log, "inference.history_hours=12 is shorter than the 30 h")
    result = predictor.predict(sc.design_storm_scenario(short, 80.0, 3, now=NOW))
    assert result.metadata["padded_history_h"] == 18 and result.metadata["history_needed_h"] == 30
    assert predictor.predict(custom(icfg)).metadata["padded_history_h"] == 0
    physics = PhysicsPredictor.from_graph(short).predict(sc.design_storm_scenario(short, 80.0, 3, now=NOW))
    assert physics.metadata["padded_history_h"] == 0 and physics.metadata["history_needed_h"] == 0


@pytest.mark.unit
def test_display_path_is_relative_or_a_file_name(tmp_path):
    assert display_path(project_root() / "artifacts" / "checkpoints" / "best.pt") == "artifacts/checkpoints/best.pt"
    assert display_path(tmp_path / "x" / "best.pt") == "best.pt" and display_path(None) is None


@pytest.mark.unit
def test_block_max_takes_the_peak_of_each_warning_window():
    from src.inference.backtest import block_max

    values = np.array([[0.1, 0.0], [0.9, 0.2], [0.3, 0.8], [0.0, 0.1], [0.5, 0.0]])
    np.testing.assert_allclose(block_max(values, 2), [[0.9, 0.2], [0.3, 0.8], [0.5, 0.0]])
    np.testing.assert_allclose(block_max(values, 24), [[0.9, 0.8]])
    assert block_max(np.zeros((0, 3)), 6).shape == (0, 3)
    with pytest.raises(ValueError):
        block_max(values, 0)


@pytest.mark.unit
def test_warning_window_metrics_forgive_timing_errors():
    from src.inference.backtest import _score

    labels = np.zeros((12, 2), dtype=np.uint8)
    labels[3, 0] = 1
    prob = np.zeros((12, 2))
    prob[5, 0] = 0.9  # forecast two hours late: an hourly miss, but a hit inside the 6 h window
    scores = _score(labels, prob, 0.5)
    assert scores["junction_hours"]["recall"] == 0.0
    assert scores["junction_block_6h"]["recall"] == 1.0
    assert scores["junction_block_24h"]["precision"] == 1.0


# --------------------------------------------------------------------------- areal serving threshold


@pytest.mark.unit
def test_ensemble_predictions_use_the_areal_serving_threshold(icfg, graph):
    """Forecast / design-storm probabilities are averaged over rain fields: they get their own threshold."""
    save_checkpoint(icfg, make_checkpoint(icfg, graph, areal_threshold=0.123))
    predictor = FloodPredictor.from_artifacts(icfg)
    assert predictor.areal_threshold == pytest.approx(0.123)
    assert predictor.threshold_for_members(1) == pytest.approx(0.4)  # exact training field (replays)
    assert predictor.threshold_for_members(8) == pytest.approx(0.123)
    design = sc.design_storm_scenario(icfg, 40.0, 3, horizon_h=12)
    result = predictor.predict(design, field_members=3)
    assert result.threshold == pytest.approx(0.123) and result.metadata["threshold_kind"] == "areal_ensemble"
    single = predictor.predict(design, field_members=1)
    assert single.threshold == pytest.approx(0.4) and single.metadata["threshold_kind"] == "exact_field"


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "bad", 0.0, 1.5, float("nan")])
def test_missing_or_invalid_areal_threshold_falls_back_to_the_model_threshold(icfg, graph, value):
    save_checkpoint(icfg, make_checkpoint(icfg, graph, areal_threshold=value))
    predictor = FloodPredictor.from_artifacts(icfg)
    assert predictor.areal_threshold is None
    assert predictor.threshold_for_members(8) == pytest.approx(0.4)


@pytest.mark.unit
def test_physics_predictor_keeps_its_threshold_for_ensembles(icfg, graph):
    physics = PhysicsPredictor.from_graph(icfg)
    assert physics.threshold_for_members(8) == physics.threshold == 0.5


@pytest.mark.unit
def test_backtest_scores_each_row_at_its_serving_threshold():
    """Ensemble rows are served with the areal threshold, so the backtest must score them with it."""
    import inspect

    from src.inference import backtest

    source = inspect.getsource(backtest._run_row)
    assert "_score(labels, result.prob, result.threshold)" in source
    assert "_score(labels, physics.prob, physics.threshold)" in source
