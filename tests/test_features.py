"""Tests for the single node/edge feature definition (``src.data_pipeline.features``)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.data_pipeline.features import (
    FeatureScaler,
    build_node_features,
    dynamic_feature_names,
    dynamic_features,
    dynamic_features_at,
    feature_names,
    rolling_feature_name,
    rolling_sums,
)
from src.utils.config import ConfigError, deep_merge

WINDOWS = (3, 6, 12, 24)
STATIC_NAMES = ("elevation", "dist_to_drain_m", "relative_elevation")


@pytest.fixture
def log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def _rain(t: int = 60, n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rain = rng.gamma(0.6, 4.0, size=(t, n)).astype(np.float32)
    rain[rng.random((t, n)) < 0.6] = 0.0
    return rain


def _naive_rolling(rain: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(rain, dtype=np.float64)
    for t in range(rain.shape[0]):
        out[t] = rain[max(0, t - window + 1): t + 1].sum(axis=0)
    return out


def _scaler(n_static: int = 3, windows=WINDOWS, seed: int = 1) -> FeatureScaler:
    rng = np.random.default_rng(seed)
    static = rng.normal(880, 10, size=(40, n_static))
    dyn = rng.gamma(0.5, 3.0, size=(500, 1 + len(windows)))
    lengths = rng.uniform(5, 400, size=30)
    return FeatureScaler.fit(
        static, dyn, lengths, 0.3,
        static_names=STATIC_NAMES[:n_static], dynamic_names=dynamic_feature_names(windows),
    )


# --------------------------------------------------------------------------- names


@pytest.mark.unit
def test_feature_names_follow_contract_order(cfg):
    assert feature_names(cfg) == [
        "elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink",
        "precip_mm_h", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm",
    ]
    assert len(feature_names(cfg)) == cfg["model"]["node_in_dim"]


@pytest.mark.unit
def test_feature_names_follow_configured_windows(cfg):
    custom = deep_merge(cfg, {"dataset": {"static_features": ["elevation"], "rolling_windows_h": [2, 48]}})
    assert feature_names(custom) == ["elevation", "precip_mm_h", "rain_2h_mm", "rain_48h_mm"]
    assert rolling_feature_name(6) == "rain_6h_mm"
    assert dynamic_feature_names([]) == ["precip_mm_h"]


@pytest.mark.unit
def test_feature_names_defaults_when_section_missing():
    assert feature_names({"project": {}, "paths": {}})[:3] == list(STATIC_NAMES)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset",
    [
        {"static_features": "elevation"},
        {"static_features": ["elevation", "elevation"]},
        {"static_features": ["elevation", ""]},
        {"static_features": ["precip_mm_h"]},
        {"rolling_windows_h": [3, 3]},
        {"rolling_windows_h": [0]},
        {"rolling_windows_h": [2.5]},
        {"rolling_windows_h": "6"},
    ],
)
def test_feature_names_reject_bad_config(dataset):
    with pytest.raises(ConfigError):
        feature_names({"dataset": dataset})


# --------------------------------------------------------------------------- rolling sums


@pytest.mark.unit
def test_rolling_sums_match_naive_trailing_sums_including_current_hour():
    rain = _rain()
    sums = rolling_sums(rain, WINDOWS)
    assert sums.shape == (60, 5, 4)
    assert sums.dtype == np.float32
    for k, window in enumerate(WINDOWS):
        np.testing.assert_allclose(sums[..., k], _naive_rolling(rain, window), rtol=1e-5, atol=1e-4)
    # window 1 is the hour itself; the first hour of every window is partial (only hour 0)
    np.testing.assert_allclose(rolling_sums(rain, [1])[..., 0], rain, atol=1e-6)
    np.testing.assert_allclose(sums[0, :, 3], rain[0], atol=1e-6)


@pytest.mark.unit
def test_rolling_sums_are_exactly_zero_after_a_dry_spell():
    rain = np.zeros((50, 3), dtype=np.float32)
    rain[5] = [1e3, 0.1, 7.3]
    sums = rolling_sums(rain, [3, 24])
    assert np.all(sums[30:] == 0.0)
    assert np.all(sums >= 0.0)


@pytest.mark.unit
def test_rolling_sums_edge_shapes():
    assert rolling_sums(np.zeros((0, 4), np.float32), WINDOWS).shape == (0, 4, 4)
    assert rolling_sums(np.zeros((5, 0), np.float32), WINDOWS).shape == (5, 0, 4)
    assert rolling_sums(np.ones((5, 2), np.float32), []).shape == (5, 2, 0)
    # a window longer than the record is a partial sum everywhere
    np.testing.assert_allclose(rolling_sums(np.ones((4, 1)), [10])[:, 0, 0], [1, 2, 3, 4])


@pytest.mark.unit
def test_rolling_sums_do_not_mutate_input_and_treat_bad_values_as_zero(log):
    rain = _rain(10, 2)
    rain[3, 0] = np.nan
    rain[4, 1] = -5.0
    original = rain.copy()
    sums = rolling_sums(rain, [3])
    np.testing.assert_array_equal(rain, original)
    assert np.isfinite(sums).all() and (sums >= 0).all()
    assert "non-finite/negative" in log.text


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((2, 2, 2))])
def test_rolling_sums_reject_wrong_rank(bad):
    with pytest.raises(ValueError, match=r"\[T, N\]"):
        rolling_sums(bad, WINDOWS)


@pytest.mark.unit
@pytest.mark.parametrize("windows", [[0], [-3], [2.5], ["x"], [True]])
def test_rolling_sums_reject_bad_windows(windows):
    with pytest.raises(ValueError, match="window"):
        rolling_sums(np.zeros((3, 2)), windows)


# --------------------------------------------------------------------------- dynamic features


@pytest.mark.unit
def test_dynamic_features_return_last_t_hours_of_history():
    rain = _rain(48, 4)
    dyn = dynamic_features(rain, 24, WINDOWS)
    assert dyn.shape == (24, 4, 5)
    assert dyn.dtype == np.float32
    np.testing.assert_allclose(dyn[..., 0], rain[24:], atol=1e-6)
    np.testing.assert_allclose(dyn[..., 1:], rolling_sums(rain, WINDOWS)[24:], atol=1e-5)
    # with a full lookback the 24 h sum at the first target hour sees exactly 24 hours of history
    np.testing.assert_allclose(dyn[0, :, 4], rain[1:25].sum(axis=0), rtol=1e-5)


@pytest.mark.unit
def test_dynamic_features_zero_targets_and_validation():
    assert dynamic_features(np.zeros((6, 3)), 6, WINDOWS).shape == (0, 3, 5)
    with pytest.raises(ValueError, match="lookback"):
        dynamic_features(np.zeros((6, 3)), 7, WINDOWS)
    with pytest.raises(ValueError, match="lookback"):
        dynamic_features(np.zeros((6, 3)), -1, WINDOWS)
    with pytest.raises(ValueError, match="lookback"):
        dynamic_features(np.zeros((6, 3)), 1.5, WINDOWS)


@pytest.mark.unit
@pytest.mark.parametrize("node_chunk", [1, 2, 128])
def test_dynamic_features_at_matches_windowed_computation(node_chunk):
    rain = _rain(200, 5, seed=4)
    hours = np.array([30, 31, 100, 199])
    at = dynamic_features_at(rain, hours, WINDOWS, node_chunk=node_chunk)
    assert at.shape == (4, 5, 5) and at.dtype == np.float32
    for k, h in enumerate(hours):
        expected = dynamic_features(rain[h - 24: h + 1], 24, WINDOWS)[0]
        np.testing.assert_allclose(at[k], expected, rtol=1e-5, atol=1e-5)
    assert dynamic_features_at(rain, np.array([], dtype=int), WINDOWS).shape == (0, 5, 5)


@pytest.mark.unit
def test_dynamic_features_at_validation():
    with pytest.raises(ValueError, match=r"\[T, N\]"):
        dynamic_features_at(np.zeros(5), [0], WINDOWS)
    with pytest.raises(ValueError, match="hours"):
        dynamic_features_at(np.zeros((5, 2)), [5], WINDOWS)
    with pytest.raises(ValueError, match="hours"):
        dynamic_features_at(np.zeros((5, 2)), [-1], WINDOWS)


# --------------------------------------------------------------------------- scaler


@pytest.mark.unit
def test_scaler_fit_statistics_follow_the_contract():
    rng = np.random.default_rng(3)
    static = rng.normal([880, 900, 0], [5, 300, 2], size=(100, 3))
    dyn = rng.gamma(0.5, 3.0, size=(1000, 5))
    lengths = rng.uniform(1, 500, 80)
    scaler = FeatureScaler.fit(static, dyn, lengths, 0.3)
    np.testing.assert_allclose(scaler.static_mean, static.mean(axis=0))
    np.testing.assert_allclose(scaler.static_std, static.std(axis=0))
    np.testing.assert_allclose(scaler.dynamic_mean, np.log1p(dyn).mean(axis=0))
    np.testing.assert_allclose(scaler.dynamic_std, np.log1p(dyn).std(axis=0))
    assert scaler.edge_len_mean == pytest.approx(np.log1p(lengths).mean())
    assert scaler.edge_len_std == pytest.approx(np.log1p(lengths).std())
    assert scaler.static_names == ("static_0", "static_1", "static_2")
    assert scaler.dynamic_names == ("dynamic_0", "dynamic_1", "dynamic_2", "dynamic_3", "dynamic_4")
    assert scaler.feature_names == [*scaler.static_names, *scaler.dynamic_names]
    z = scaler.transform_static(static)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-4)
    zd = scaler.transform_dynamic(dyn)
    np.testing.assert_allclose(zd.mean(axis=0), 0, atol=1e-5)


@pytest.mark.unit
def test_scaler_accepts_multidimensional_dynamic_samples():
    dyn = np.random.default_rng(0).gamma(0.5, 3.0, size=(10, 7, 5))
    a = FeatureScaler.fit(np.ones((7, 1)), dyn, np.ones(3), 0.3)
    b = FeatureScaler.fit(np.ones((7, 1)), dyn.reshape(-1, 5), np.ones(3), 0.3)
    assert a == b


@pytest.mark.unit
def test_scaler_floors_std_and_zeroes_nan(log):
    static = np.array([[5.0, np.nan], [5.0, np.nan], [5.0, np.nan]])
    dyn = np.zeros((20, 5))
    scaler = FeatureScaler.fit(static, dyn, np.array([]), 0.3)
    assert scaler.static_std[0] == pytest.approx(1e-6)
    assert scaler.static_mean[1] == 0.0 and scaler.static_std[1] == 1.0  # all-NaN column
    assert (scaler.edge_len_mean, scaler.edge_len_std) == (0.0, 1.0)      # graph without edges
    assert "no finite values" in log.text
    out = scaler.transform_static(np.array([[5.0, np.nan], [np.inf, 2.0]]))
    assert out.dtype == np.float32
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0 and out[1, 0] == 0.0
    assert out[1, 1] == pytest.approx(2.0)
    dyn_out = scaler.transform_dynamic(np.array([[np.nan, 0.0, 0.0, 0.0, 0.0]]))
    assert np.all(dyn_out == 0.0)


@pytest.mark.unit
def test_scaler_fit_without_dynamic_samples_warns_and_uses_identity(log):
    scaler = FeatureScaler.fit(np.ones((3, 1)), np.zeros((0, 5)), np.ones(2), 0.3)
    np.testing.assert_array_equal(scaler.dynamic_mean, np.zeros(5))
    np.testing.assert_array_equal(scaler.dynamic_std, np.ones(5))
    assert "no dynamic samples" in log.text


@pytest.mark.unit
def test_scaler_transform_edges_normalises_length_and_grade():
    lengths = np.array([10.0, 100.0, 1000.0])
    grade = np.array([-0.3, 0.0, 0.6])
    scaler = FeatureScaler.fit(np.ones((2, 1)), np.ones((4, 5)), lengths, 0.3)
    edges = scaler.transform_edges(lengths, grade)
    assert edges.shape == (3, 2) and edges.dtype == np.float32
    np.testing.assert_allclose(edges[:, 0].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(edges[:, 1], [-1.0, 0.0, 1.0])  # clipped to [-1, 1]
    assert scaler.transform_edges(np.array([]), np.array([])).shape == (0, 2)
    with pytest.raises(ValueError, match="same length"):
        scaler.transform_edges(lengths, grade[:2])


@pytest.mark.unit
def test_scaler_dynamic_clips_negative_values_before_log():
    scaler = FeatureScaler.fit(np.ones((2, 1)), np.ones((4, 5)), np.ones(2), 0.3)
    neg = scaler.transform_dynamic(np.full((1, 5), -3.0))
    zero = scaler.transform_dynamic(np.zeros((1, 5)))
    np.testing.assert_array_equal(neg, zero)


@pytest.mark.unit
def test_scaler_round_trips_through_json_dict():
    scaler = _scaler()
    data = scaler.to_dict()
    restored = FeatureScaler.from_dict(json.loads(json.dumps(data)))
    assert restored == scaler
    assert restored != _scaler(seed=9)
    assert scaler != "not a scaler"
    x = np.random.default_rng(0).normal(880, 10, size=(6, 3))
    np.testing.assert_array_equal(restored.transform_static(x), scaler.transform_static(x))


@pytest.mark.unit
def test_scaler_is_immutable():
    scaler = _scaler()
    with pytest.raises(AttributeError):
        scaler.max_abs_grade = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        scaler.static_mean[0] = 1.0


@pytest.mark.unit
def test_scaler_fit_does_not_mutate_inputs():
    static = np.array([[1.0, np.nan], [3.0, 4.0]])
    dyn = np.array([[1.0, -1.0, 0.0, 2.0, np.nan]])
    before = (static.copy(), dyn.copy())
    FeatureScaler.fit(static, dyn, np.array([3.0]), 0.3)
    np.testing.assert_array_equal(static, before[0])
    np.testing.assert_array_equal(dyn, before[1])


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("static_mean"), "missing"),
        (lambda d: d.update(static_std=[1.0]), "static"),
        (lambda d: d.update(dynamic_std=[1.0, 1.0, 0.0, 1.0, 1.0]), "positive"),
        (lambda d: d.update(max_abs_grade=0.0), "max_abs_grade"),
        (lambda d: d.update(edge_len_std=-1.0), "edge_len_std"),
        (lambda d: d.update(static_mean=["a", 1.0, 2.0]), "numeric"),
        (lambda d: d.update(version=99), "version"),
    ],
)
def test_scaler_from_dict_validates(mutate, match):
    data = _scaler().to_dict()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        FeatureScaler.from_dict(data)


@pytest.mark.unit
def test_scaler_from_dict_rejects_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        FeatureScaler.from_dict([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(static=np.ones(3)), "static"),
        (dict(dynamic_samples=np.ones(5)), "dynamic"),
        (dict(edge_length=np.ones((2, 2))), "edge_length"),
        (dict(max_abs_grade=0.0), "max_abs_grade"),
        (dict(max_abs_grade=float("nan")), "max_abs_grade"),
        (dict(static_names=("a",)), "static_names"),
        (dict(dynamic_names=("a", "b")), "dynamic_names"),
    ],
)
def test_scaler_fit_validates_inputs(kwargs, match):
    args = dict(static=np.ones((4, 2)), dynamic_samples=np.ones((6, 5)), edge_length=np.ones(3), max_abs_grade=0.3)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FeatureScaler.fit(**args)


@pytest.mark.unit
def test_scaler_transform_rejects_wrong_feature_count():
    scaler = _scaler()
    with pytest.raises(ValueError, match="static"):
        scaler.transform_static(np.ones((4, 2)))
    with pytest.raises(ValueError, match="dynamic"):
        scaler.transform_dynamic(np.ones((4, 3)))


# --------------------------------------------------------------------------- node features


@pytest.mark.unit
def test_build_node_features_matches_manual_pipeline():
    scaler = _scaler()
    static = np.random.default_rng(5).normal(880, 10, size=(5, 3)).astype(np.float32)
    rain = _rain(48, 5)
    x = build_node_features(static, rain, 24, WINDOWS, scaler)
    assert x.shape == (24, 5, 8) and x.dtype == np.float32
    np.testing.assert_allclose(x[3, :, :3], scaler.transform_static(static), atol=1e-6)
    np.testing.assert_allclose(x[..., 3:], scaler.transform_dynamic(dynamic_features(rain, 24, WINDOWS)), atol=1e-6)
    assert np.isfinite(x).all()


@pytest.mark.unit
def test_build_node_features_is_deterministic_and_pure():
    scaler = _scaler()
    static = np.ones((5, 3), np.float32)
    rain = _rain(30, 5)
    before = rain.copy()
    a = build_node_features(static, rain, 6, WINDOWS, scaler)
    b = build_node_features(static, rain, 6, WINDOWS, scaler)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(rain, before)


@pytest.mark.unit
def test_build_node_features_validates_shapes():
    scaler = _scaler()
    with pytest.raises(ValueError, match="nodes"):
        build_node_features(np.ones((4, 3)), _rain(30, 5), 6, WINDOWS, scaler)
    with pytest.raises(ValueError, match="static"):
        build_node_features(np.ones(5), _rain(30, 5), 6, WINDOWS, scaler)
    with pytest.raises(ValueError, match="rolling"):
        build_node_features(np.ones((5, 3)), _rain(30, 5), 6, (3, 6), scaler)
    with pytest.raises(ValueError, match="static"):
        build_node_features(np.ones((5, 2)), _rain(30, 5), 6, WINDOWS, scaler)
    with pytest.raises(TypeError, match="FeatureScaler"):
        build_node_features(np.ones((5, 3)), _rain(30, 5), 6, WINDOWS, scaler.to_dict())  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_node_features_zero_target_hours_and_single_node():
    scaler = _scaler()
    assert build_node_features(np.ones((1, 3)), np.zeros((24, 1)), 24, WINDOWS, scaler).shape == (0, 1, 8)
    assert build_node_features(np.ones((1, 3)), np.zeros((25, 1)), 24, WINDOWS, scaler).shape == (1, 1, 8)


@pytest.mark.unit
def test_scaler_log_transforms_heavy_tailed_static_features_only():
    static = np.array([[870.0, 1.0, 0.0], [880.0, 9.0, 1.0], [890.0, 99.0, 0.0]])
    names = ["elevation", "flow_accumulation", "is_sink"]
    scaler = FeatureScaler.fit(static, np.ones((4, 5)), np.array([10.0, 20.0]), 0.3,
                               static_names=names, dynamic_names=[f"d{i}" for i in range(5)])
    assert scaler.static_log == (False, True, False)
    z = scaler.transform_static(static)
    logged = np.log1p(static[:, 1])
    expected = (logged - logged.mean()) / logged.std()
    np.testing.assert_allclose(z[:, 1], expected, rtol=1e-5)
    np.testing.assert_allclose(z[:, 0], (static[:, 0] - 880.0) / static[:, 0].std(), rtol=1e-5)
    restored = FeatureScaler.from_dict(scaler.to_dict())
    assert restored == scaler and restored.static_log == (False, True, False)


@pytest.mark.unit
def test_scaler_dict_without_static_log_is_backward_compatible():
    scaler = FeatureScaler.fit(np.array([[1.0], [3.0]]), np.ones((2, 5)), np.array([5.0]), 0.3,
                               static_names=["flow_accumulation"], dynamic_names=[f"d{i}" for i in range(5)])
    legacy = {k: v for k, v in scaler.to_dict().items() if k != "static_log"}
    assert FeatureScaler.from_dict(legacy).static_log == (False,)
