"""Tests for the stochastic convective rainfall downscaling (``src.data_pipeline.rain_field``)."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import rain_field
from src.data_pipeline.rain_field import (
    DEFAULTS,
    RainFieldParams,
    downscale_rainfall,
    find_rain_events,
)
from src.hydrology.label_diagnostics import rain_field_heterogeneity
from src.utils.config import ConfigError, deep_merge
from tests.conftest import BBOX, TZ


@pytest.fixture
def warn_log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def _nodes(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    west, south, east, north = BBOX
    return rng.uniform(west, east, n), rng.uniform(south, north, n)


def _with_field(cfg: dict, **overrides) -> dict:
    return deep_merge(cfg, {"rainfall_field": overrides})


def _areal(storm_series: pd.Series) -> tuple[np.ndarray, pd.DatetimeIndex]:
    return np.array(storm_series.to_numpy(dtype=float), copy=True), pd.DatetimeIndex(storm_series.index)


# --------------------------------------------------------------------------- params


@pytest.mark.unit
def test_params_from_config_reads_section(cfg):
    params = RainFieldParams.from_config(cfg)
    section = {**DEFAULTS, **cfg["rainfall_field"]}
    assert params.enabled is True
    assert params.n_cells == section["n_cells"]
    assert params.cell_radius_m == tuple(float(v) for v in section["cell_radius_m"])
    assert params.advection_speed_mps == tuple(float(v) for v in section["advection_speed_mps"])
    assert params.multiplier_clip == tuple(float(v) for v in section["multiplier_clip"])
    assert params.background_fraction == pytest.approx(section["background_fraction"])
    assert params.cell_lifetime_h == pytest.approx(section["cell_lifetime_h"])
    assert params.normalise_intensity is bool(section["normalise_intensity"])
    assert params.max_dry_gap_h == 2
    assert params.seed == 42
    assert params.half_life_s == pytest.approx(0.5 * params.cell_lifetime_h * 3600)


@pytest.mark.unit
def test_config_yaml_mirrors_calibrated_defaults(cfg):
    """The calibrated rain-field defaults live in the module and in config/config.yaml; keep them in sync."""
    section = cfg["rainfall_field"]
    assert set(section) == set(DEFAULTS)
    for key, value in DEFAULTS.items():
        assert section[key] == (pytest.approx(value) if not isinstance(value, bool) else value), key


@pytest.mark.unit
def test_params_accept_scalar_ranges_and_missing_section(cfg):
    params = RainFieldParams.from_config(_with_field(cfg, cell_radius_m=1500, advection_speed_mps=2))
    assert params.cell_radius_m == (1500.0, 1500.0)
    assert params.advection_speed_mps == (2.0, 2.0)
    bare = {key: value for key, value in cfg.items() if key != "rainfall_field"}
    assert RainFieldParams.from_config(bare).n_cells == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"multiplier_clip": [1.2, 3.0]},
        {"multiplier_clip": [0.2, 0.9]},
        {"cell_radius_m": [0.0, 100.0]},
        {"cell_radius_m": [900.0, 800.0]},
        {"advection_speed_mps": [-1.0, 2.0]},
        {"background_fraction": 1.5},
        {"n_cells": -1},
        {"n_cells": 2.5},
        {"max_dry_gap_h": -1},
        {"substeps_per_hour": 0},
        {"seed": -3},
        {"cell_radius_m": [1.0, 2.0, 3.0]},
        {"cell_radius_m": "wide"},
        {"cell_lifetime_h": 0.1},
        {"cell_lifetime_h": float("nan")},
        {"cell_lifetime_h": True},
        {"normalise_intensity": "yes"},
        {"stratiform_rate_mm_h": 0.0},
        {"stratiform_rate_mm_h": -2.0},
        {"stratiform_rate_mm_h": "fast"},
        {"n_cells": float("inf")},
        {"amplitude_range": [0.5, 1.5]},
    ],
)
def test_params_reject_invalid_values(cfg, overrides):
    with pytest.raises(ConfigError):
        RainFieldParams.from_config(_with_field(cfg, **overrides))


# --------------------------------------------------------------------------- events


@pytest.mark.unit
def test_find_rain_events_merges_short_dry_gaps(hourly_index):
    rain = np.zeros(len(hourly_index))
    rain[[5, 6, 9]] = 1.0          # gap of 2 dry hours -> same event
    rain[[20, 24]] = 2.0           # gap of 3 dry hours -> separate events
    events = find_rain_events(rain, hourly_index, max_dry_gap_h=2)
    assert events.num_events == 3
    assert events.start_idx.tolist() == [5, 20, 24]
    assert events.end_idx.tolist() == [9, 20, 24]
    assert (events.event_of_hour[5:10] == 0).all()
    assert events.event_of_hour[4] == -1 and events.event_of_hour[10] == -1
    expected_start = int(hourly_index[5].tz_convert("UTC").value // 3_600_000_000_000)
    assert events.start_hour[0] == expected_start


@pytest.mark.unit
def test_find_rain_events_handles_empty_and_dry(hourly_index):
    empty = find_rain_events(np.zeros(0), hourly_index[:0])
    assert empty.num_events == 0 and empty.event_of_hour.shape == (0,)
    dry = find_rain_events(np.zeros(len(hourly_index)), hourly_index)
    assert dry.num_events == 0 and (dry.event_of_hour == -1).all()


@pytest.mark.unit
def test_find_rain_events_counts_missing_timestamps_as_dry(hourly_index):
    stamps = hourly_index[[0, 1, 5, 6]]  # 3 missing hours between index 1 and 5
    events = find_rain_events(np.array([1.0, 1.0, 1.0, 1.0]), stamps, max_dry_gap_h=2)
    assert events.num_events == 2
    merged = find_rain_events(np.array([1.0, 1.0, 1.0, 1.0]), stamps, max_dry_gap_h=3)
    assert merged.num_events == 1


# --------------------------------------------------------------------------- core behaviour


@pytest.mark.unit
def test_downscale_shape_dtype_and_mass_conservation(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(400)
    rain = downscale_rainfall(areal, stamps, lon, lat, cfg)
    assert rain.shape == (len(areal), 400)
    assert rain.dtype == np.float32
    assert np.isfinite(rain).all() and (rain >= 0).all()
    wet = areal > 0
    node_mean = rain[wet].astype(np.float64).mean(axis=1)
    np.testing.assert_allclose(node_mean, areal[wet], rtol=1e-4)
    assert (rain[~wet] == 0).all()


@pytest.mark.unit
def test_downscale_produces_spatial_structure_within_clip(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(500)
    rain = downscale_rainfall(areal, stamps, lon, lat, cfg)
    wet = areal > 0
    multipliers = rain[wet] / areal[wet, None]
    assert multipliers.std(axis=1).max() > 0.05, "at least one wet hour should be non-uniform"
    lo, hi = cfg["rainfall_field"]["multiplier_clip"]
    assert multipliers.min() >= lo * 0.95 and multipliers.max() <= hi * 1.05


@pytest.mark.unit
def test_downscale_is_deterministic_and_seed_sensitive(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(200)
    first = downscale_rainfall(areal, stamps, lon, lat, cfg)
    second = downscale_rainfall(areal, stamps, lon, lat, cfg)
    np.testing.assert_array_equal(first, second)
    other = downscale_rainfall(areal, stamps, lon, lat, _with_field(cfg, seed=7))
    assert not np.allclose(first, other)


@pytest.mark.unit
def test_downscale_is_consistent_for_sub_ranges(cfg, storm_series):
    """A sub-range starting in a dry spell (or ending mid-event) reproduces the full-series field."""
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(150)
    full = downscale_rainfall(areal, stamps, lon, lat, cfg)
    for lo, hi in ((10, 96), (40, 70), (50, 63), (0, 62)):
        part = downscale_rainfall(areal[lo:hi], stamps[lo:hi], lon, lat, cfg)
        np.testing.assert_allclose(part, full[lo:hi], rtol=1e-6, atol=1e-7)


@pytest.mark.unit
def test_downscale_disabled_broadcasts(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(30)
    rain = downscale_rainfall(areal, stamps, lon, lat, _with_field(cfg, enabled=False))
    np.testing.assert_allclose(rain, np.repeat(areal[:, None], 30, axis=1).astype(np.float32))


@pytest.mark.unit
@pytest.mark.parametrize("overrides", [{"n_cells": 0}, {"background_fraction": 1.0}])
def test_downscale_uniform_when_no_convective_component(cfg, storm_series, overrides):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(40)
    rain = downscale_rainfall(areal, stamps, lon, lat, _with_field(cfg, **overrides))
    np.testing.assert_allclose(rain, np.repeat(areal[:, None], 40, axis=1), rtol=1e-6)


@pytest.mark.unit
def test_downscale_single_node_equals_areal(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    rain = downscale_rainfall(areal, stamps, np.array([77.68]), np.array([12.93]), cfg)
    assert rain.shape == (len(areal), 1)
    np.testing.assert_allclose(rain[:, 0], areal, rtol=1e-6)


@pytest.mark.unit
def test_downscale_empty_inputs(cfg, hourly_index):
    lon, lat = _nodes(5)
    out = downscale_rainfall(np.zeros(0), hourly_index[:0], lon, lat, cfg)
    assert out.shape == (0, 5) and out.dtype == np.float32
    no_nodes = downscale_rainfall(np.ones(3), hourly_index[:3], np.zeros(0), np.zeros(0), cfg)
    assert no_nodes.shape == (3, 0)


@pytest.mark.unit
def test_downscale_all_zero_series(cfg, hourly_index):
    lon, lat = _nodes(20)
    out = downscale_rainfall(np.zeros(len(hourly_index)), hourly_index, lon, lat, cfg)
    assert out.shape == (len(hourly_index), 20)
    assert not out.any()


@pytest.mark.unit
def test_downscale_nan_and_negative_areal_become_zero(cfg, storm_series, warn_log):
    areal, stamps = _areal(storm_series)
    areal[61] = np.nan
    areal[21] = -4.0
    areal[62] = np.inf
    lon, lat = _nodes(25)
    rain = downscale_rainfall(areal, stamps, lon, lat, cfg)
    assert np.isfinite(rain).all()
    assert not rain[[21, 61, 62]].any()
    messages = " ".join(r.getMessage() for r in warn_log.records if r.levelno == logging.WARNING)
    assert "non-finite" in messages and "negative" in messages


@pytest.mark.unit
def test_downscale_does_not_mutate_inputs(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    areal[3] = np.nan
    lon, lat = _nodes(10)
    areal_copy, lon_copy, lat_copy = areal.copy(), lon.copy(), lat.copy()
    downscale_rainfall(areal, stamps, lon, lat, cfg)
    np.testing.assert_array_equal(areal, areal_copy)
    np.testing.assert_array_equal(lon, lon_copy)
    np.testing.assert_array_equal(lat, lat_copy)


@pytest.mark.unit
def test_downscale_accepts_lists_and_naive_timestamps(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(12)
    aware = downscale_rainfall(areal, stamps, lon, lat, cfg)
    naive = downscale_rainfall(list(areal), stamps.tz_localize(None), lon.tolist(), lat.tolist(), cfg)
    np.testing.assert_array_equal(aware, naive)
    utc = downscale_rainfall(areal, stamps.tz_convert("UTC"), lon, lat, cfg)
    np.testing.assert_array_equal(aware, utc)


@pytest.mark.unit
def test_downscale_input_validation(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(10)
    with pytest.raises(ValueError, match="timestamps"):
        downscale_rainfall(areal[:-1], stamps, lon, lat, cfg)
    with pytest.raises(ValueError, match="lon"):
        downscale_rainfall(areal, stamps, lon[:-1], lat, cfg)
    with pytest.raises(ValueError, match="1-D"):
        downscale_rainfall(areal[:, None], stamps, lon, lat, cfg)
    bad_lon = lon.copy()
    bad_lon[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        downscale_rainfall(areal, stamps, bad_lon, lat, cfg)
    with pytest.raises(ValueError, match="increasing"):
        downscale_rainfall(areal, stamps[::-1], lon, lat, cfg)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        downscale_rainfall(areal, np.arange(len(areal)), lon, lat, cfg)
    with pytest.raises(ValueError, match="numeric"):
        downscale_rainfall(np.array(["a"] * len(areal)), stamps, lon, lat, cfg)


def _mean_lag1_correlation(rain: np.ndarray) -> float:
    corr = [np.corrcoef(rain[i], rain[i + 1])[0, 1] for i in range(len(rain) - 1)
            if rain[i].std() > 1e-3 and rain[i + 1].std() > 1e-3]
    return float(np.mean(corr)) if corr else 0.0


@pytest.mark.unit
def test_downscale_fields_are_temporally_coherent_for_slow_storms(cfg):
    """With long-lived (6 h) cells, slow advection keeps consecutive hours correlated; fast advection does not."""
    stamps = pd.date_range("2022-09-04 12:00", periods=8, freq="h", tz=TZ)
    areal = np.full(8, 10.0)
    lon, lat = _nodes(400, seed=11)
    slow, fast = [], []
    for seed in range(4):
        base = {"background_fraction": 0.1, "seed": seed, "cell_lifetime_h": 6.0}
        slow_cfg = _with_field(cfg, advection_speed_mps=[0.2, 0.2], **base)
        fast_cfg = _with_field(cfg, advection_speed_mps=[6.0, 6.0], **base)
        slow.append(_mean_lag1_correlation(downscale_rainfall(areal, stamps, lon, lat, slow_cfg)))
        fast.append(_mean_lag1_correlation(downscale_rainfall(areal, stamps, lon, lat, fast_cfg)))
    assert np.mean(slow) > 0.6
    assert np.mean(slow) > np.mean(fast) + 0.2


@pytest.mark.unit
def test_nodes_outside_bbox_are_supported(cfg, storm_series):
    areal, stamps = _areal(storm_series)
    lon, lat = _nodes(50)
    lon = np.concatenate([lon, [77.9, 77.3]])
    lat = np.concatenate([lat, [13.2, 12.6]])
    rain = downscale_rainfall(areal, stamps, lon, lat, cfg)
    wet = areal > 0
    np.testing.assert_allclose(rain[wet].mean(axis=1), areal[wet], rtol=1e-4)


@pytest.mark.integration
def test_downscale_year_of_hours_is_fast(cfg):
    """One year x 1000 nodes with a realistic wet fraction stays well inside the speed budget."""
    stamps = pd.date_range("2022-01-01", periods=8760, freq="h", tz=TZ)
    rng = np.random.default_rng(0)
    areal = np.where(rng.random(8760) < 0.3, rng.gamma(0.6, 3.0, 8760), 0.0)
    lon, lat = _nodes(1000)
    start = time.perf_counter()
    rain = downscale_rainfall(areal, stamps, lon, lat, cfg)
    elapsed = time.perf_counter() - start
    assert rain.shape == (8760, 1000)
    wet = areal > 0
    np.testing.assert_allclose(rain[wet].astype(np.float64).mean(axis=1), areal[wet], rtol=1e-4)
    assert elapsed < 15.0, f"downscaling took {elapsed:.1f}s"


# --------------------------------------------------------------------------- fix round: heterogeneity (R2-01)


def _convective_record(n_events: int = 90) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Many short convective events (areal 1.5-6 mm/h) separated by dry spells."""
    stamps = pd.date_range("2021-06-01", periods=n_events * 14, freq="h", tz=TZ)
    areal = np.zeros(len(stamps))
    for k in range(n_events):
        areal[k * 14 + 3 : k * 14 + 7] = [2.0, 6.0, 4.0, 1.5]
    return areal, stamps


def _default_field_cfg(cfg: dict) -> dict:
    return {"region": cfg["region"], "project": cfg["project"], "rainfall_field": dict(DEFAULTS)}


@pytest.mark.unit
def test_default_field_is_genuinely_heterogeneous(cfg):
    """Regression for R2-01: the old field had a within-hour CV of ~0.08 and max/mean ~1.4."""
    areal, stamps = _convective_record()
    lon, lat = _nodes(400, seed=5)
    rain = downscale_rainfall(areal, stamps, lon, lat, _default_field_cfg(cfg))
    stats = rain_field_heterogeneity(rain, areal)
    assert stats.n_wet_hours == int((areal >= 1.0).sum())
    assert 0.3 <= stats.cv_median <= 0.7, stats
    assert stats.max_over_mean_p95 >= 2.0, stats
    wet = areal > 0
    np.testing.assert_allclose(rain[wet].astype(np.float64).mean(axis=1), areal[wet], rtol=1e-4)


@pytest.mark.unit
def test_intensity_normalisation_makes_every_convective_hour_structured(cfg):
    areal, stamps = _convective_record(30)
    lon, lat = _nodes(300, seed=6)
    base = _default_field_cfg(cfg)
    on = downscale_rainfall(areal, stamps, lon, lat, base)
    off = downscale_rainfall(areal, stamps, lon, lat, deep_merge(base, {"rainfall_field": {"normalise_intensity": False}}))
    cv_on = rain_field_heterogeneity(on, areal).cv_median
    cv_off = rain_field_heterogeneity(off, areal).cv_median
    assert cv_on > cv_off
    wet = areal > 0
    assert (on[wet].std(axis=1) > 0.01 * areal[wet]).all(), "no wet hour may collapse to a uniform field"


@pytest.mark.unit
def test_normalise_intensity_scales_each_hour_to_its_strongest_cell():
    intensity = np.array([[0.1, 0.2, 0.05], [0.0, 0.0, 0.0], [0.5, 0.25, 0.0]])
    strongest = np.array([0.8, 0.9, 0.5])
    out = rain_field._normalise_intensity(intensity, strongest)
    np.testing.assert_allclose(out, [[0.4, 0.8, 0.2], [0.0, 0.0, 0.0], [0.5, 0.25, 0.0]])


@pytest.mark.unit
def test_cell_generations_form_a_partition_of_unity():
    """Overlapping cell life cycles always add up to one cell-equivalent per slot."""
    half = 3600.0
    t = np.linspace(-3600.0, 40_000.0, 997)
    for phase_value in (0.0, 0.3, 0.999):
        phase = np.full(t.size, phase_value)
        (k0, peak0, env0), (k1, peak1, env1) = rain_field._alive_generations(t, phase, half)
        np.testing.assert_allclose(env0 + env1, 1.0, atol=1e-12)
        assert (k1 == k0 + 1).all() and (peak0 <= t + 1e-9).all() and (peak1 >= t - 1e-9).all()


@pytest.mark.unit
def test_generation_draws_do_not_depend_on_how_many_are_drawn():
    """The consistency guarantee relies on word i of a SeedSequence state being independent of n."""
    long = rain_field._uniform_words((42, 123456, 2), 40)
    short = rain_field._uniform_words((42, 123456, 2), 8)
    np.testing.assert_array_equal(long[:8], short)
    assert ((long >= 0) & (long < 1)).all()


@pytest.mark.unit
def test_long_event_sub_ranges_are_consistent(cfg):
    """An event spanning many cell generations reproduces exactly when a sub-range starts at its start."""
    stamps = pd.date_range("2022-09-04 00:00", periods=60, freq="h", tz=TZ)
    areal = np.zeros(60)
    areal[5:41] = np.linspace(1.0, 9.0, 36)
    lon, lat = _nodes(120, seed=8)
    full = downscale_rainfall(areal, stamps, lon, lat, cfg)
    for lo, hi in ((5, 15), (0, 30), (2, 60), (5, 41)):
        part = downscale_rainfall(areal[lo:hi], stamps[lo:hi], lon, lat, cfg)
        np.testing.assert_allclose(part, full[lo:hi], rtol=1e-6, atol=1e-7)


@pytest.mark.unit
def test_cells_are_placed_around_the_corridor_not_on_a_torus(cfg):
    params = RainFieldParams.from_config(cfg)
    lon, lat = _nodes(50)
    x, y, domain = rain_field._node_domain(lon, lat, cfg, params)
    margin = params.domain_margin_factor * params.cell_radius_m[1]
    assert domain.x_min <= x.min() - margin + 1.5 and domain.y_min <= y.min() - margin + 1.5
    assert domain.x_min + domain.width >= x.max() + margin - 1.5
    events = find_rain_events(np.array([1.0, 1.0, 0.0]), pd.date_range("2022-01-01", periods=3, freq="h", tz=TZ))
    cells = rain_field._draw_event_cells(events.start_hour, np.array([3600.0]), params, domain)
    assert (cells.peak_x >= domain.x_min).all() and (cells.peak_x <= domain.x_min + domain.width).all()
    assert (cells.peak_y >= domain.y_min).all() and (cells.peak_y <= domain.y_min + domain.height).all()
    lo_r, hi_r = params.cell_radius_m
    assert (cells.radius >= lo_r).all() and (cells.radius <= hi_r).all()


@pytest.mark.unit
def test_stratiform_share_decreases_with_rain_rate(cfg):
    params = RainFieldParams.from_config(_with_field(cfg, background_fraction=0.2, stratiform_rate_mm_h=2.0))
    share = params.background_for(np.array([0.0, 2.0, 8.0, 18.0]))[:, 0]
    np.testing.assert_allclose(share, [0.2, 0.1, 0.04, 0.02])
    constant = RainFieldParams.from_config(_with_field(cfg, stratiform_rate_mm_h=None))
    np.testing.assert_allclose(constant.background_for(np.array([0.5, 20.0]))[:, 0], constant.background_fraction)


@pytest.mark.unit
def test_intense_hours_are_more_localised_than_light_ones(cfg):
    """With a rate-dependent stratiform share an intense hour concentrates in convective cores."""
    stamps = pd.date_range("2022-06-01", periods=60 * 12, freq="h", tz=TZ)
    light, heavy = np.zeros(len(stamps)), np.zeros(len(stamps))
    for k in range(60):
        light[k * 12 + 2 : k * 12 + 5] = 1.5
        heavy[k * 12 + 2 : k * 12 + 5] = 15.0
    lon, lat = _nodes(300, seed=9)
    base = _default_field_cfg(cfg)
    cv_light = rain_field_heterogeneity(downscale_rainfall(light, stamps, lon, lat, base), light).cv_median
    cv_heavy = rain_field_heterogeneity(downscale_rainfall(heavy, stamps, lon, lat, base), heavy).cv_median
    assert cv_heavy > cv_light + 0.05
