"""Tests for the hydrology calibration diagnostics (``src.hydrology.calibrate``, ``label_diagnostics``).

Covers criteria (a)-(h) of the fix-round contract, the rain-field heterogeneity target, the
CLI and a read-only regression guard that the shipped defaults pass on the real record.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_pipeline.graph_io import graph_to_arrays
from src.hydrology import calibrate, label_diagnostics
from src.hydrology.simulator import DEFAULTS, SimulationError, UrbanDrainageSimulator
from src.utils.config import resolve_path
from tests.conftest import TZ
from tests.test_hydrology import grid_arrays  # noqa: F401 - shared fixture


def _toy_record(hours: int = 24 * 40, nodes: int = 4):
    stamps = pd.date_range("2022-08-20", periods=hours, freq="h", tz=TZ)
    areal = np.zeros(hours)
    labels = np.zeros((hours, nodes), dtype=np.uint8)
    storm = stamps.get_loc(pd.Timestamp("2022-09-05 02:00", tz=TZ))
    areal[storm - 2 : storm + 1] = [4.0, 10.0, 6.0]
    labels[storm, :2] = 1
    small = 5 * 24
    areal[small] = 2.0
    labels[small + 1, 3] = 1  # a flood although the 6 h rain is only 2 mm
    return stamps, areal, labels


@pytest.mark.unit
def test_compute_statistics_on_toy_record():
    stamps, areal, labels = _toy_record()
    static = {
        "relative_elevation": np.array([-2.0, -1.0, 1.0, 2.0]),
        "flow_accumulation": np.array([10.0, 5.0, 1.0, 1.0]),
        "is_sink": np.array([1.0, 1.0, 0.0, 0.0]),
    }
    stats = calibrate.compute_statistics(labels, areal, stamps, static, max_dry_gap_h=2)
    assert stats.dry_flood_node_hours == 1
    assert stats.sept2022_peak_fraction == pytest.approx(0.5)
    assert stats.season_rate == pytest.approx(3 / labels.size)
    assert stats.n_events == 2 and stats.events_with_flooding == 2
    assert stats.correlations["relative_elevation"]["spearman"] < 0
    checks = stats.checks()
    assert checks["b_no_floods_below_4mm_6h"] is False
    lines = "\n".join(stats.format_lines())
    assert "Sept 4-5 2022" in lines and "relative_elevation" in lines
    assert stats.to_dict()["checks"] == checks


@pytest.mark.unit
def test_compute_statistics_without_sept_2022_and_constant_features():
    stamps = pd.date_range("2019-06-01", periods=48, freq="h", tz=TZ)
    labels = np.zeros((48, 3), dtype=np.uint8)
    static = {"is_sink": np.zeros(3)}
    stats = calibrate.compute_statistics(labels, np.zeros(48), stamps, static)
    assert stats.sept2022_peak_fraction is None
    assert stats.correlations["is_sink"]["spearman"] is None
    assert stats.checks()["c_sept2022_peak_10_to_50pct"] is None
    assert stats.events_per_year_with_flooding == 0.0
    with pytest.raises(ValueError):
        calibrate.compute_statistics(labels[:10], np.zeros(48), stamps, static)


@pytest.mark.unit
def test_parse_overrides():
    assert calibrate.parse_overrides(["spill_depth_m=null", "catchment_width_m=12", "rainfall_field.n_cells=4"]) == {
        "hydrology": {"spill_depth_m": None, "catchment_width_m": 12},
        "rainfall_field": {"n_cells": 4},
    }
    assert calibrate.parse_overrides([]) == {}
    for bad in (["oops"], ["=3"], ["k=[1, 2"]):
        with pytest.raises(ValueError):
            calibrate.parse_overrides(bad)


@pytest.mark.integration
def test_run_calibration_offline_synthetic(cfg):
    stats = calibrate.run_calibration(cfg)
    assert stats.n_nodes == 36
    assert stats.n_hours > 0
    assert 0.0 <= stats.season_rate <= 1.0
    assert stats.runtime_s["simulation"] >= 0.0


@pytest.mark.integration
def test_calibration_is_read_only_on_a_fresh_clone(cfg):
    """F5-08: a diagnostics run never saves a synthetic graph / weather record at the project paths."""
    graph_file, weather_file = resolve_path(cfg, "graph_file"), resolve_path(cfg, "weather_file")
    assert not graph_file.exists() and not weather_file.exists()
    stats = calibrate.run_calibration(cfg, compare_uniform=False)
    assert stats.n_nodes == 36 and stats.provenance["source"] == "synthetic_grid"
    assert not graph_file.exists(), "calibrate must use an in-memory graph when no real one exists"
    assert not weather_file.exists(), "calibrate must not write the weather cache"


@pytest.mark.integration
def test_cli_main_success_and_failure(cfg, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_json = tmp_path / "calib.json"
    code = calibrate.main(["--config", str(config_path), "--offline", "--json", str(out_json),
                           "--set", "flood_depth_threshold_m=0.2"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "flooded node-hours" in printed and out_json.exists()
    assert calibrate.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert calibrate.main(["--config", str(config_path), "--set", "ponding_fraction=-2"]) == 1
    assert "error" in capsys.readouterr().err


@pytest.mark.unit
def test_rank1_r2_is_one_for_a_basin_switch_times_a_fixed_map():
    hours = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    nodes = np.array([1, 1, 0, 0, 1], dtype=bool)
    labels = np.outer(hours, nodes)
    assert label_diagnostics.rank1_r2(labels) == pytest.approx(1.0)


@pytest.mark.unit
def test_rank1_r2_is_low_when_floods_move_between_events():
    labels = np.zeros((40, 8), dtype=bool)
    for t in range(0, 40, 4):
        labels[t, (t // 4) % 8] = True   # each event floods a different junction
    r2 = label_diagnostics.rank1_r2(labels)
    assert r2 is not None and r2 < 0.2
    assert label_diagnostics.rank1_r2(np.zeros((5, 3), dtype=bool)) is None
    assert label_diagnostics.rank1_r2(np.zeros((0, 3), dtype=bool)) is None
    with pytest.raises(ValueError):
        label_diagnostics.rank1_r2(np.zeros(5))


@pytest.mark.unit
def test_label_jaccard_and_flooded_share():
    a = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=bool)
    b = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype=bool)
    assert label_diagnostics.label_jaccard(a, b) == pytest.approx(3 / 4)
    assert label_diagnostics.label_jaccard(np.zeros((2, 2)), np.zeros((2, 2))) is None
    with pytest.raises(ValueError):
        label_diagnostics.label_jaccard(a, b[:2])
    share = label_diagnostics.flooded_share_stats(a)
    assert share.n_flood_hours == 2 and share.median == pytest.approx(2 / 3) and share.max == pytest.approx(2 / 3)
    assert label_diagnostics.flooded_share_stats(np.zeros((3, 2))).median is None
    assert label_diagnostics.flooded_share_stats(np.zeros((3, 0))).n_flood_hours == 0


@pytest.mark.unit
def test_rain_field_heterogeneity_statistics():
    rain = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
    areal = rain.mean(axis=1)
    stats = label_diagnostics.rain_field_heterogeneity(rain, areal)
    assert stats.n_wet_hours == 2 and stats.wet_threshold_mm_h == 1.0
    assert stats.cv_median == pytest.approx(0.5)            # CVs 0 and 1
    assert stats.max_over_mean_median == pytest.approx(1.5)  # ratios 1 and 2
    empty = label_diagnostics.rain_field_heterogeneity(rain * 0.1, areal * 0.1)
    assert empty.n_wet_hours == 0 and empty.cv_median is None
    with pytest.raises(ValueError):
        label_diagnostics.rain_field_heterogeneity(rain, areal[:2])


@pytest.mark.unit
def test_separable_lookup_pr_auc_on_a_separable_record():
    stamps = pd.date_range("2021-05-01", "2022-11-30 23:00", freq="h", tz=TZ)
    rng = np.random.default_rng(0)
    areal = np.where(rng.random(len(stamps)) < 0.05, rng.gamma(1.0, 4.0, len(stamps)), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(areal)])
    idx = np.arange(1, areal.size + 1)
    basin = csum[idx] - csum[np.maximum(idx - 5, 0)]
    susceptible = np.array([True, True, False, False, False])
    labels = (basin[:, None] > 10.0) & susceptible[None, :]
    out = label_diagnostics.separable_lookup_pr_auc(labels, areal, stamps)
    assert out["eval_year"] == 2022 and out["n_pos"] > 0 and out["pr_auc"] > 0.95
    one_year = label_diagnostics.separable_lookup_pr_auc(labels[:2000], areal[:2000], stamps[:2000])
    assert one_year["pr_auc"] is None


@pytest.mark.unit
def test_compute_statistics_reports_criteria_f_g_h():
    stamps, areal, labels = _toy_record()
    static = {"is_sink": np.array([1.0, 1.0, 0.0, 0.0])}
    uniform = labels.copy()
    uniform[:, 2] = labels[:, 0]
    uniform[:, 3] = 0
    stats = calibrate.compute_statistics(labels, areal, stamps, static, uniform_labels=uniform)
    assert stats.rank1_r2 is not None and 0.0 <= stats.rank1_r2 <= 1.0
    assert stats.uniform_jaccard == pytest.approx(2 / 4)
    assert stats.flooded_share["n_flood_hours"] == 2 and stats.flooded_share["median"] == pytest.approx(0.375)
    checks = stats.checks()
    assert set(checks) == set(calibrate.CHECK_NAMES)
    assert checks["g_uniform_rain_jaccard_below_0_75"] is True
    assert checks["h_median_flooded_share_below_25pct"] is False
    assert checks["rf_rain_field_heterogeneous"] is None
    text = "\n".join(stats.format_lines())
    assert "(f)" in text and "(g)" in text and "(h)" in text and "(rf)" in text
    with pytest.raises(ValueError, match="uniform_labels"):
        calibrate.compute_statistics(labels, areal, stamps, static, uniform_labels=uniform[:10])
    no_uniform = calibrate.compute_statistics(labels, areal, stamps, static)
    assert no_uniform.uniform_jaccard is None and no_uniform.checks()["g_uniform_rain_jaccard_below_0_75"] is None


@pytest.mark.unit
def test_rain_field_check_uses_cv_range_and_peak_ratio():
    stamps, areal, labels = _toy_record()
    base = calibrate.compute_statistics(labels, areal, stamps, {"is_sink": np.array([1.0, 1.0, 0.0, 0.0])})
    ok = dataclasses.replace(base, rain_field={"cv_median": 0.45, "max_over_mean_p95": 2.5})
    flat = dataclasses.replace(base, rain_field={"cv_median": 0.08, "max_over_mean_p95": 1.4})
    assert ok.checks()["rf_rain_field_heterogeneous"] is True
    assert flat.checks()["rf_rain_field_heterogeneous"] is False


@pytest.mark.integration
def test_simulate_and_score_compares_against_uniform_rain(cfg, grid_arrays):
    stamps = pd.date_range("2022-09-01", periods=24 * 20, freq="h", tz=TZ)
    rng = np.random.default_rng(4)
    areal = np.where(rng.random(len(stamps)) < 0.15, rng.gamma(1.2, 5.0, len(stamps)), 0.0)
    from src.data_pipeline.rain_field import downscale_rainfall

    rain = downscale_rainfall(areal, stamps, grid_arrays.lon, grid_arrays.lat, cfg)
    stats = calibrate.simulate_and_score(grid_arrays, rain, areal, stamps, cfg)
    assert stats.uniform_jaccard is None or 0.0 <= stats.uniform_jaccard <= 1.0
    assert "uniform_simulation" in stats.runtime_s
    assert stats.rain_field["n_wet_hours"] == int((areal >= 1.0).sum())
    quick = calibrate.simulate_and_score(grid_arrays, rain, areal, stamps, cfg, compare_uniform=False)
    assert quick.uniform_jaccard is None and "uniform_simulation" not in quick.runtime_s


@pytest.mark.unit
def test_simulate_and_score_raises_on_non_finite_budget(cfg, grid_arrays, monkeypatch):
    stamps = pd.date_range("2022-09-01", periods=4, freq="h", tz=TZ)
    rain = np.zeros((4, grid_arrays.num_nodes), dtype=np.float32)
    original = UrbanDrainageSimulator.run

    def nan_run(self, rain, initial_storage_m3=None):
        result = original(self, rain, initial_storage_m3)
        return dataclasses.replace(result, mass_balance=dataclasses.replace(result.mass_balance, inflow_m3=float("nan")))

    monkeypatch.setattr(UrbanDrainageSimulator, "run", nan_run)
    with pytest.raises(SimulationError):
        calibrate.simulate_and_score(grid_arrays, rain, np.zeros(4), stamps, cfg, compare_uniform=False)


@pytest.mark.integration
def test_cli_no_uniform_flag(cfg, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_path.write_text(yaml.safe_dump(clean), encoding="utf-8")
    assert calibrate.main(["--config", str(config_path), "--offline", "--no-uniform"]) == 0
    assert "n/a (not computed)" in capsys.readouterr().out


_REAL_GRAPH = Path(__file__).resolve().parents[1] / "data/interim/bellandur_osm.graphml"


_REAL_WEATHER = Path(__file__).resolve().parents[1] / "data/raw/weather/open_meteo_hourly.csv"


@pytest.mark.integration
@pytest.mark.skipif(not (_REAL_GRAPH.exists() and _REAL_WEATHER.exists()), reason="real graph / weather record absent")
def test_calibrated_defaults_meet_every_criterion_on_the_real_record(cfg):
    """Regression guard for R2-01: the shipped defaults pass (a)-(h) and the rain-field target (read-only)."""
    from src.data_pipeline.graph_io import load_graph
    from src.data_pipeline.rain_field import DEFAULTS as RAIN_DEFAULTS
    from src.data_pipeline.rain_field import downscale_rainfall
    from src.data_pipeline.weather import areal_series
    from src.data_pipeline.weather_schema import read_weather_csv

    cfg = {"project": cfg["project"], "region": cfg["region"],
           "hydrology": dict(DEFAULTS), "rainfall_field": dict(RAIN_DEFAULTS)}
    arrays = graph_to_arrays(load_graph(_REAL_GRAPH))
    areal, stamps = areal_series(read_weather_csv(_REAL_WEATHER, TZ))
    rain = downscale_rainfall(areal, pd.DatetimeIndex(stamps), arrays.lon, arrays.lat, cfg)
    stats = calibrate.simulate_and_score(arrays, rain, np.asarray(areal), pd.DatetimeIndex(stamps), cfg)
    failed = [name for name, ok in stats.checks().items() if ok is not True]
    assert not failed, "\n".join(stats.format_lines())
