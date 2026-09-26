"""Tests for the hydrology label simulator (``src.hydrology``): parameters, drainage network and physics.

Observed-report fusion lives in ``tests/test_hydrology_observed.py`` and the calibration
diagnostics (criteria a-h) in ``tests/test_hydrology_calibration.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import pytest

from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays
from src.hydrology.simulator import (
    DEFAULTS,
    DrainageNetwork,
    HydrologyParams,
    SimulationError,
    SimulationResult,
    UrbanDrainageSimulator,
    build_drainage_network,
    physics_flood_probability,
    simulate_labels,
    surcharge_factor,
    tailwater_gate,
)
from src.utils.config import ConfigError
from tests.conftest import TZ, make_grid_graph

# --------------------------------------------------------------------------- helpers


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def warn_log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def _warnings(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def _params(**overrides) -> HydrologyParams:
    return HydrologyParams.from_config({"hydrology": overrides})


def _arrays(
    elev,
    edges=(),
    lengths=None,
    dist=None,
    lon=None,
    lat=None,
    bidirectional: bool = True,
    node_attrs: dict | None = None,
) -> GraphArrays:
    """Small hand-made :class:`GraphArrays` (edges given once; reversed copies added when bidirectional)."""
    elev = np.asarray(elev, dtype=np.float64)
    n = elev.size
    lon = np.linspace(77.660, 77.661 + 0.001 * n, n) if lon is None else np.asarray(lon, dtype=np.float64)
    lat = np.full(n, 12.93) if lat is None else np.asarray(lat, dtype=np.float64)
    pairs = list(edges)
    lens = list(lengths) if lengths is not None else None
    if bidirectional:
        pairs = pairs + [(v, u) for u, v in pairs]
        lens = lens + lens if lens is not None else None
    edge_index = np.array(pairs, dtype=np.int64).T.reshape(2, -1)
    attrs = {"elevation": elev, "dist_to_drain_m": np.full(n, 1000.0) if dist is None else np.asarray(dist, float)}
    if node_attrs is not None:
        attrs = node_attrs
    edge_attrs = {"length": np.asarray(lens, dtype=np.float64)} if lens is not None else {}
    return GraphArrays(
        node_ids=tuple(range(n)),
        lon=lon,
        lat=lat,
        node_attrs=attrs,
        edge_index=edge_index,
        edge_attrs=edge_attrs,
    )


@pytest.fixture
def grid_arrays(grid_graph) -> GraphArrays:
    return graph_to_arrays(grid_graph)


@pytest.fixture
def grid_rain(grid_arrays, storm_series) -> np.ndarray:
    """Storm hyetograph with a smooth spatial multiplier (node mean preserved)."""
    rng = np.random.default_rng(7)
    mult = rng.uniform(0.4, 2.2, grid_arrays.num_nodes)
    mult = mult / mult.mean()
    return (storm_series.to_numpy()[:, None] * mult[None, :]).astype(np.float32)


# --------------------------------------------------------------------------- parameters


@pytest.mark.unit
def test_params_from_config_reads_every_key(cfg):
    params = HydrologyParams.from_config(cfg)
    section = cfg["hydrology"]
    assert set(DEFAULTS) == {f.name for f in dataclasses.fields(HydrologyParams)}
    assert set(section) <= set(DEFAULTS), "config.yaml hydrology keys must all be known parameters"
    for field in dataclasses.fields(HydrologyParams):
        expected = section.get(field.name, DEFAULTS[field.name])
        actual = getattr(params, field.name)
        assert actual is None if expected is None else actual == pytest.approx(expected), field.name


@pytest.mark.unit
def test_params_missing_section_uses_defaults():
    params = HydrologyParams.from_config({})
    as_dict = params.to_dict()
    for key, value in DEFAULTS.items():
        assert as_dict[key] is None if value is None else as_dict[key] == pytest.approx(value), key
    assert isinstance(params.antecedent_window_h, int)
    assert isinstance(params.surcharge_window_h, int)


@pytest.mark.unit
def test_params_to_dict_round_trip():
    params = _params(catchment_width_m=12.5, spill_depth_m=None)
    again = HydrologyParams.from_config({"hydrology": params.to_dict()})
    assert again == params


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"catchment_width_m": 0.0},
        {"catchment_width_m": -3},
        {"runoff_coeff_dry": 1.2},
        {"runoff_coeff_dry": 0.9, "runoff_coeff_wet": 0.5},
        {"ponding_fraction": 0.0},
        {"ponding_fraction": 1.5},
        {"surcharge_min_factor": -0.1},
        {"flood_depth_threshold_m": 0.0},
        {"flood_depth_threshold_m": float("nan")},
        {"spill_depth_m": 0.1, "flood_depth_threshold_m": 0.15},
        {"antecedent_window_h": -1},
        {"antecedent_window_h": 2.5},
        {"surcharge_window_h": 0},
        {"routing_exponent": -1.0},
        {"drain_capacity_far_mm_h": "fast"},
        {"infiltration_mm_h": True},
        {"drain_decay_m": None},
        {"surcharge_local_weight": 1.5},
        {"surcharge_local_weight": -0.1},
        {"tailwater_threshold_mm": -1.0},
        {"tailwater_ramp_mm": 0.0},
        {"tailwater_ramp_mm": "wide"},
    ],
)
def test_params_invalid_values_raise(overrides):
    with pytest.raises(ConfigError):
        _params(**overrides)


@pytest.mark.unit
def test_params_direct_construction_is_validated():
    params = HydrologyParams.from_config({})
    with pytest.raises(ConfigError):
        dataclasses.replace(params, ponding_fraction=-1.0)


@pytest.mark.unit
def test_params_unknown_key_warns(warn_log):
    _params(catchmnet_width_m=10.0)
    assert "catchmnet_width_m" in _warnings(warn_log)


@pytest.mark.unit
def test_params_section_must_be_mapping():
    with pytest.raises(ConfigError):
        HydrologyParams.from_config({"hydrology": [1, 2]})


# --------------------------------------------------------------------------- drainage network


@pytest.mark.unit
def test_catchment_area_is_width_times_half_incident_length():
    arrays = _arrays([3.0, 2.0, 1.0], edges=[(0, 1), (1, 2)], lengths=[100.0, 60.0])
    net = build_drainage_network(arrays, _params(catchment_width_m=30.0))
    np.testing.assert_allclose(net.area_m2, [30 * 50, 30 * (50 + 30), 30 * 30])
    assert net.n_segments == 2


@pytest.mark.unit
def test_chain_routes_downhill_and_bottom_is_sink():
    arrays = _arrays([3.0, 2.0, 1.0], edges=[(0, 1), (1, 2)], lengths=[100.0, 100.0])
    net = build_drainage_network(arrays, _params(outflow_rate_per_h=0.6))
    assert net.is_sink.tolist() == [False, False, True]
    routing = net.routing.toarray()
    assert routing[1, 0] == pytest.approx(1.0)
    assert routing[2, 1] == pytest.approx(1.0)
    assert routing[:, 2].sum() == 0.0
    # grade 1 % -> outflow fraction == outflow_rate_per_h; sinks keep their water
    np.testing.assert_allclose(net.outflow_fraction, [0.6, 0.6, 0.0])


@pytest.mark.unit
@pytest.mark.parametrize("exponent, expected", [(1.0, [2 / 3, 1 / 3]), (0.0, [0.5, 0.5]), (2.0, [0.8, 0.2])])
def test_mfd_split_proportional_to_slope_power(exponent, expected):
    # node 0 at 10 m, neighbour 1 two metres lower, neighbour 2 one metre lower, both 100 m away
    arrays = _arrays([10.0, 8.0, 9.0], edges=[(0, 1), (0, 2)], lengths=[100.0, 100.0])
    net = build_drainage_network(arrays, _params(routing_exponent=exponent))
    routing = net.routing.toarray()
    np.testing.assert_allclose(routing[[1, 2], 0], expected)
    np.testing.assert_allclose(routing.sum(axis=0)[~net.is_sink], 1.0)


@pytest.mark.unit
def test_outflow_fraction_uses_min_grade_and_saturates():
    # flat-ish chain (0.01 % grade) and a steep chain (10 % grade)
    flat = _arrays([1.01, 1.0], edges=[(0, 1)], lengths=[100.0])
    steep = _arrays([11.0, 1.0], edges=[(0, 1)], lengths=[100.0])
    params = _params(outflow_rate_per_h=0.6, min_routing_grade=0.0005)
    assert build_drainage_network(flat, params).outflow_fraction[0] == pytest.approx(0.6 * 0.0005 / 0.01)
    assert build_drainage_network(steep, params).outflow_fraction[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_one_way_edges_self_loops_and_duplicates_are_normalised():
    edges = [(0, 1), (1, 0), (1, 2), (2, 2)]
    arrays = _arrays([3.0, 2.0, 1.0], edges=edges, lengths=[80.0, 40.0, 50.0, 10.0], bidirectional=False)
    net = build_drainage_network(arrays, _params(catchment_width_m=10.0))
    assert net.n_segments == 2  # self-loop dropped, the duplicate 0-1 pair collapsed to the shorter length
    np.testing.assert_allclose(net.area_m2, [10 * 20, 10 * (20 + 25), 10 * 25])


@pytest.mark.unit
def test_missing_length_attribute_falls_back_to_haversine(warn_log):
    arrays = _arrays([2.0, 1.0], edges=[(0, 1)], lon=[77.66, 77.661], lat=[12.93, 12.93])
    net = build_drainage_network(arrays, _params(catchment_width_m=1.0))
    assert net.area_m2[0] == pytest.approx(108.4 / 2, rel=0.02)
    assert "length" in _warnings(warn_log)


@pytest.mark.unit
def test_invalid_lengths_are_repaired(warn_log):
    arrays = _arrays([2.0, 1.0, 0.5], edges=[(0, 1), (1, 2)], lengths=[np.nan, -5.0],
                     lon=[77.66, 77.661, 77.662], lat=[12.93] * 3)
    net = build_drainage_network(arrays, _params(catchment_width_m=1.0))
    assert np.all(np.isfinite(net.area_m2)) and np.all(net.area_m2 > 1.0)
    assert "invalid" in _warnings(warn_log)


@pytest.mark.unit
def test_missing_elevation_raises_clear_error():
    arrays = _arrays([1.0, 2.0], edges=[(0, 1)], lengths=[10.0], node_attrs={"dist_to_drain_m": np.zeros(2)})
    with pytest.raises(ValueError, match="elevation"):
        build_drainage_network(arrays, _params())


@pytest.mark.unit
def test_non_finite_elevation_is_filled_with_warning(warn_log):
    arrays = _arrays([3.0, np.nan, 1.0], edges=[(0, 1), (1, 2)], lengths=[10.0, 10.0])
    net = build_drainage_network(arrays, _params())
    assert np.all(np.isfinite(net.elevation_m))
    assert "elevation" in _warnings(warn_log)
    all_nan = _arrays([np.nan, np.nan], edges=[(0, 1)], lengths=[10.0])
    with pytest.raises(ValueError, match="finite"):
        build_drainage_network(all_nan, _params())


@pytest.mark.unit
def test_missing_drain_distance_uses_far_capacity(warn_log):
    arrays = _arrays([2.0, 1.0], edges=[(0, 1)], lengths=[100.0], node_attrs={"elevation": np.array([2.0, 1.0])})
    params = _params(drain_capacity_far_mm_h=5.0, drain_capacity_near_mm_h=50.0, catchment_width_m=10.0)
    net = build_drainage_network(arrays, params)
    np.testing.assert_allclose(net.drain_capacity_m3_h, net.area_m2 * 5.0 / 1000.0)
    assert "dist_to_drain_m" in _warnings(warn_log)


@pytest.mark.unit
def test_drain_capacity_decays_with_distance():
    arrays = _arrays([2.0, 1.0, 0.5], edges=[(0, 1), (1, 2)], lengths=[100.0, 100.0], dist=[0.0, 400.0, np.inf])
    params = _params(drain_capacity_near_mm_h=30.0, drain_capacity_far_mm_h=8.0, drain_decay_m=400.0)
    net = build_drainage_network(arrays, params)
    per_mm = net.drain_capacity_m3_h / net.area_m2 * 1000.0
    np.testing.assert_allclose(per_mm, [30.0, 8.0 + 22.0 * np.exp(-1.0), 8.0])


@pytest.mark.unit
def test_build_rejects_wrong_types():
    with pytest.raises(TypeError):
        build_drainage_network("graph", _params())
    with pytest.raises(TypeError):
        UrbanDrainageSimulator(_arrays([1.0]), {"catchment_width_m": 3})


@pytest.mark.unit
def test_simulator_accepts_networkx_graph(grid_graph):
    sim = UrbanDrainageSimulator(grid_graph, _params())
    assert sim.num_nodes == grid_graph.number_of_nodes()
    assert isinstance(sim.network, DrainageNetwork)


# --------------------------------------------------------------------------- surcharge


@pytest.mark.unit
def test_surcharge_factor_ramp():
    params = _params(surcharge_window_h=2, surcharge_threshold_mm=10.0, surcharge_ramp_mm=10.0,
                     surcharge_min_factor=0.2)
    basin = np.array([0.0, 5.0, 5.0, 10.0, 10.0, 20.0, 0.0, 0.0])
    # trailing 2 h sums: 0, 5, 10, 15, 20, 30, 20, 0
    np.testing.assert_allclose(surcharge_factor(basin, params), [1, 1, 1, 0.6, 0.2, 0.2, 0.2, 1])
    assert surcharge_factor(np.zeros(0), params).shape == (0,)


# --------------------------------------------------------------------------- simulation: edge cases


@pytest.mark.unit
def test_zero_hours_returns_empty_result(grid_arrays):
    sim = UrbanDrainageSimulator(grid_arrays, _params())
    init = np.full(grid_arrays.num_nodes, 2.0)
    result = sim.run(np.zeros((0, grid_arrays.num_nodes), dtype=np.float32), initial_storage_m3=init)
    assert isinstance(result, SimulationResult)
    assert result.depth_m.shape == (0, grid_arrays.num_nodes) and result.depth_m.dtype == np.float32
    assert result.flooded.shape == (0, grid_arrays.num_nodes) and result.flooded.dtype == bool
    np.testing.assert_allclose(result.final_storage_m3, init)
    assert result.mass_balance.relative_error == 0.0


@pytest.mark.unit
def test_single_node_without_edges():
    arrays = _arrays([900.0])
    params = _params(runoff_coeff_dry=1.0, runoff_coeff_wet=1.0, infiltration_mm_h=0.0,
                     drain_capacity_far_mm_h=0.0, drain_capacity_near_mm_h=0.0, ponding_fraction=0.1,
                     spill_depth_m=None)
    rain = np.array([[10.0], [5.0], [0.0]], dtype=np.float32)
    result = UrbanDrainageSimulator(arrays, params).run(rain)
    # area floors at 1 m2: depth = cumulative rain / ponding fraction
    np.testing.assert_allclose(result.depth_m[:, 0], [0.1, 0.15, 0.15], rtol=1e-6)
    assert result.flooded[:, 0].tolist() == [False, True, True]


@pytest.mark.unit
def test_graph_without_edges_ponds_locally():
    arrays = _arrays([900.0, 901.0, 902.0])
    rain = np.full((4, 3), 20.0, dtype=np.float32)
    result = UrbanDrainageSimulator(arrays, _params()).run(rain)
    assert result.depth_m.shape == (4, 3)
    assert np.all(np.isfinite(result.depth_m)) and np.all(result.depth_m >= 0)
    assert result.mass_balance.relative_error < 1e-9


@pytest.mark.unit
def test_all_zero_rain_gives_no_water(grid_arrays):
    result = UrbanDrainageSimulator(grid_arrays, _params()).run(np.zeros((48, grid_arrays.num_nodes)))
    assert not result.flooded.any()
    assert float(result.depth_m.max()) == 0.0
    assert result.mass_balance.inflow_m3 == 0.0


@pytest.mark.unit
def test_nan_and_negative_rain_are_zeroed_with_warning(grid_arrays, grid_rain, warn_log):
    dirty = grid_rain.copy()
    dirty[3, :4] = np.nan
    dirty[5, 2] = -4.0
    dirty[6, 1] = np.inf
    clean = dirty.copy()
    clean[~np.isfinite(clean) | (clean < 0)] = 0.0
    sim = UrbanDrainageSimulator(grid_arrays, _params())
    got = sim.run(dirty)
    want = sim.run(clean)
    np.testing.assert_array_equal(got.depth_m, want.depth_m)
    text = _warnings(warn_log)
    assert "non-finite" in text and "negative" in text


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 36, 1), (0,)])
def test_rain_shape_mismatch_raises(grid_arrays, shape):
    with pytest.raises(ValueError, match="rain"):
        UrbanDrainageSimulator(grid_arrays, _params()).run(np.zeros(shape))


@pytest.mark.unit
def test_non_numeric_rain_raises(grid_arrays):
    bad = np.full((2, grid_arrays.num_nodes), "x", dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        UrbanDrainageSimulator(grid_arrays, _params()).run(bad)


@pytest.mark.unit
@pytest.mark.parametrize("initial", [np.zeros(3), -np.ones(36), np.full(36, np.nan), np.zeros((36, 1)) + 1])
def test_invalid_initial_storage_raises(grid_arrays, initial):
    with pytest.raises(ValueError, match="initial_storage"):
        UrbanDrainageSimulator(grid_arrays, _params()).run(np.zeros((2, 36)), initial_storage_m3=initial)


@pytest.mark.unit
def test_inputs_are_not_mutated(grid_arrays, grid_rain):
    rain = grid_rain.copy()
    rain[0, 0] = np.nan
    before = rain.copy()
    init = np.full(grid_arrays.num_nodes, 3.0)
    init_before = init.copy()
    UrbanDrainageSimulator(grid_arrays, _params()).run(rain, initial_storage_m3=init)
    np.testing.assert_array_equal(rain, before)
    np.testing.assert_array_equal(init, init_before)


# --------------------------------------------------------------------------- simulation: physics


@pytest.mark.unit
def test_mass_balance_closes_with_spill_and_surcharge(grid_arrays, grid_rain):
    params = _params(surcharge_threshold_mm=10.0, surcharge_ramp_mm=20.0, spill_depth_m=0.3,
                     drain_capacity_far_mm_h=2.0, drain_capacity_near_mm_h=6.0)
    init = np.linspace(0.0, 5.0, grid_arrays.num_nodes)
    result = UrbanDrainageSimulator(grid_arrays, params).run(grid_rain * 3, initial_storage_m3=init)
    mb = result.mass_balance
    assert mb.boundary_outflow_m3 > 0 and mb.drained_m3 > 0
    lhs = mb.inflow_m3
    rhs = (mb.final_storage_m3 - mb.initial_storage_m3) + mb.drained_m3 + mb.boundary_outflow_m3
    assert abs(lhs - rhs) <= 1e-6 * lhs
    assert mb.relative_error < 1e-6
    assert mb.final_storage_m3 == pytest.approx(result.final_storage_m3.sum())
    assert mb.initial_storage_m3 == pytest.approx(init.sum())


@pytest.mark.unit
def test_inflow_equals_rain_volume_when_fully_impervious(grid_arrays, grid_rain):
    params = _params(runoff_coeff_dry=1.0, runoff_coeff_wet=1.0, infiltration_mm_h=0.0)
    sim = UrbanDrainageSimulator(grid_arrays, params)
    result = sim.run(grid_rain)
    expected = float((grid_rain.astype(np.float64) * sim.network.area_m2[None, :]).sum() / 1000.0)
    assert result.mass_balance.inflow_m3 == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_flooded_matches_depth_threshold(grid_arrays, grid_rain):
    params = _params(flood_depth_threshold_m=0.05)
    result = UrbanDrainageSimulator(grid_arrays, params).run(grid_rain * 4)
    np.testing.assert_array_equal(result.flooded, result.depth_m >= np.float32(0.05))
    assert result.flooded.any()


@pytest.mark.unit
def test_water_collects_in_the_valley(grid_arrays, grid_rain):
    uniform = np.repeat(grid_rain.mean(axis=1, keepdims=True), grid_arrays.num_nodes, axis=1) * 3
    sim = UrbanDrainageSimulator(grid_arrays, _params(spill_depth_m=None))
    peak = sim.run(uniform).depth_m.max(axis=0)
    sinks = sim.network.is_sink
    assert sinks.any() and (~sinks).any()
    assert peak[sinks].mean() > peak[~sinks].mean()


@pytest.mark.unit
def test_more_rain_never_means_less_water(grid_arrays, grid_rain):
    sim = UrbanDrainageSimulator(grid_arrays, _params(spill_depth_m=None))
    low = sim.run(grid_rain).depth_m
    high = sim.run(grid_rain * 2).depth_m
    assert np.all(high >= low - 1e-6)
    assert high.max() > low.max()


@pytest.mark.unit
def test_nearby_drain_reduces_ponding():
    arrays = _arrays([1.0, 1.0], dist=[0.0, 5000.0])  # two isolated junctions
    params = _params(drain_capacity_near_mm_h=30.0, drain_capacity_far_mm_h=5.0, spill_depth_m=None,
                     surcharge_threshold_mm=1e6)
    rain = np.full((6, 2), 20.0)
    depth = UrbanDrainageSimulator(arrays, params).run(rain).depth_m
    assert depth[-1, 0] == 0.0
    assert depth[-1, 1] > 0.5


@pytest.mark.unit
def test_surcharge_slows_drainage_in_long_storms():
    arrays = _arrays([1.0])
    rain = np.full((12, 1), 12.0)
    base = dict(drain_capacity_far_mm_h=10.0, drain_capacity_near_mm_h=10.0, spill_depth_m=None,
                surcharge_window_h=6, surcharge_ramp_mm=10.0, surcharge_min_factor=0.25)
    free = UrbanDrainageSimulator(arrays, _params(surcharge_threshold_mm=1e6, **base)).run(rain)
    surcharged = UrbanDrainageSimulator(arrays, _params(surcharge_threshold_mm=20.0, **base)).run(rain)
    assert surcharged.depth_m[-1, 0] > free.depth_m[-1, 0]
    assert surcharged.mass_balance.drained_m3 < free.mass_balance.drained_m3


@pytest.mark.unit
def test_wet_antecedent_conditions_increase_runoff():
    arrays = _arrays([1.0])
    params = _params(runoff_coeff_dry=0.5, runoff_coeff_wet=1.0, antecedent_window_h=24,
                     antecedent_saturation_mm=20.0, infiltration_mm_h=0.0, drain_capacity_far_mm_h=0.0,
                     drain_capacity_near_mm_h=0.0, spill_depth_m=None)
    sim = UrbanDrainageSimulator(arrays, params)
    burst = np.zeros((30, 1))
    burst[-1] = 10.0
    wet = burst.copy()
    wet[10:15] = 4.0  # 20 mm within the previous 24 h saturates the catchment
    dry_gain = sim.run(burst).mass_balance.inflow_m3
    wet_run = sim.run(wet)
    wet_gain = wet_run.depth_m[-1, 0] - wet_run.depth_m[-2, 0]
    assert dry_gain == pytest.approx(0.5 * 10.0 / 1000.0)
    assert wet_gain * params.ponding_fraction == pytest.approx(1.0 * 10.0 / 1000.0)


@pytest.mark.unit
def test_spill_caps_depth():
    arrays = _arrays([1.0])
    params = _params(spill_depth_m=0.4, drain_capacity_far_mm_h=0.0, drain_capacity_near_mm_h=0.0)
    result = UrbanDrainageSimulator(arrays, params).run(np.full((10, 1), 50.0))
    assert result.depth_m.max() == pytest.approx(0.4, rel=1e-6)
    assert result.mass_balance.boundary_outflow_m3 > 0


@pytest.mark.unit
def test_initial_storage_drains_away(grid_arrays):
    sim = UrbanDrainageSimulator(grid_arrays, _params())
    init = sim.network.pond_area_m2 * 1.0  # 1 m of water everywhere
    result = sim.run(np.zeros((72, grid_arrays.num_nodes)), initial_storage_m3=init)
    assert result.depth_m[0].max() > 0.15
    assert result.depth_m[-1].max() < result.depth_m[0].max()
    assert result.final_storage_m3.sum() < init.sum()


@pytest.mark.unit
def test_deterministic_and_chunk_invariant(grid_arrays, grid_rain):
    params = _params(surcharge_threshold_mm=15.0)
    full = UrbanDrainageSimulator(grid_arrays, params).run(grid_rain)
    again = UrbanDrainageSimulator(grid_arrays, params).run(grid_rain)
    chunked = UrbanDrainageSimulator(grid_arrays, params, chunk_hours=7).run(grid_rain)
    np.testing.assert_array_equal(full.depth_m, again.depth_m)
    np.testing.assert_allclose(chunked.depth_m, full.depth_m, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(chunked.final_storage_m3, full.final_storage_m3, rtol=1e-9)


@pytest.mark.unit
def test_invalid_chunk_hours():
    with pytest.raises(ValueError):
        UrbanDrainageSimulator(_arrays([1.0]), _params(), chunk_hours=0)


@pytest.mark.integration
def test_speed_large_graph():
    rows = cols = 32  # 1024 junctions
    arrays = graph_to_arrays(make_grid_graph(rows, cols))
    hours = 8760
    rng = np.random.default_rng(1)
    rain = np.where(rng.random((hours, 1)) < 0.1, rng.gamma(0.8, 4.0, (hours, 1)), 0.0)
    rain = (rain * rng.uniform(0.3, 2.0, (1, arrays.num_nodes))).astype(np.float32)
    sim = UrbanDrainageSimulator(arrays, _params())
    started = time.perf_counter()
    result = sim.run(rain)
    elapsed = time.perf_counter() - started
    assert result.depth_m.shape == (hours, arrays.num_nodes)
    # contract: 61 000 h x 1 000 nodes in < 60 s  ->  8 760 h must take well under 60 * 8760 / 61000 s
    assert elapsed < 60.0 * hours / 61_000


# --------------------------------------------------------------------------- simulate_labels / probability / config


@pytest.mark.unit
def test_simulate_labels_types_and_consistency(cfg, grid_arrays, grid_rain):
    labels, depth = simulate_labels(grid_arrays, grid_rain * 3, cfg)
    assert labels.dtype == np.uint8 and depth.dtype == np.float32
    assert labels.shape == depth.shape == grid_rain.shape
    threshold = HydrologyParams.from_config(cfg).flood_depth_threshold_m
    np.testing.assert_array_equal(labels.astype(bool), depth >= np.float32(threshold))
    assert set(np.unique(labels)).issubset({0, 1})


@pytest.mark.unit
def test_simulate_labels_without_hydrology_section(grid_arrays, grid_rain):
    labels, depth = simulate_labels(grid_arrays, grid_rain, {"project": {"timezone": TZ}})
    assert labels.shape == grid_rain.shape


@pytest.mark.unit
def test_physics_probability_logistic():
    depth = np.array([0.0, 0.1, 0.15, 0.2, 1.0, 50.0, -50.0])
    prob = physics_flood_probability(depth, 0.15, softness_m=0.05)
    assert prob.dtype == np.float32 and prob.shape == depth.shape
    assert prob[2] == pytest.approx(0.5)
    assert np.all(np.diff(prob[:6]) >= 0)
    assert prob[5] == pytest.approx(1.0) and prob[6] == pytest.approx(0.0)
    assert prob[0] == pytest.approx(1.0 / (1.0 + np.exp(3.0)), rel=1e-5)


@pytest.mark.unit
def test_physics_probability_nan_and_validation(warn_log):
    prob = physics_flood_probability(np.array([[np.nan, 0.3]], dtype=np.float32), 0.15)
    assert np.isfinite(prob).all() and prob.shape == (1, 2)
    assert "NaN" in _warnings(warn_log)
    with pytest.raises(ValueError):
        physics_flood_probability(np.zeros(3), 0.15, softness_m=0.0)
    with pytest.raises(ValueError):
        physics_flood_probability(np.zeros(3), float("nan"))
    with pytest.raises(ValueError):
        physics_flood_probability(np.array(["a"]), 0.15)
    assert physics_flood_probability(0.15, 0.15).shape == ()


@pytest.mark.unit
def test_config_yaml_mirrors_calibrated_defaults(cfg):
    """The calibrated defaults live in both the module and config/config.yaml; keep them in sync."""
    section = cfg["hydrology"]
    assert set(section) == set(DEFAULTS)
    for key, value in DEFAULTS.items():
        assert section[key] is None if value is None else section[key] == pytest.approx(value), key
    assert cfg["labels"]["min_report_severity"] == 1


# --------------------------------------------------------------------------- fix round: routing (R2-07)


def _gentle_fan(exponent: float) -> tuple[GraphArrays, HydrologyParams]:
    """Node 0 drains to three neighbours over gentle slopes (2e-5 .. 6e-5): slope**200 underflows to 0."""
    arrays = _arrays([10.0, 9.998, 9.996, 9.994], edges=[(0, 1), (0, 2), (0, 3)], lengths=[100.0, 100.0, 100.0])
    return arrays, _params(routing_exponent=exponent)


@pytest.mark.unit
@pytest.mark.parametrize("exponent", [1.0, 100.0, 200.0, 1000.0])
def test_large_routing_exponent_never_produces_nan_shares(exponent):
    arrays, params = _gentle_fan(exponent)
    net = build_drainage_network(arrays, params)
    shares = net.routing.toarray()[:, 0]
    assert np.isfinite(net.routing.data).all()
    assert shares.sum() == pytest.approx(1.0)
    assert shares[3] == shares.max()  # the steepest neighbour always keeps a finite weight
    if exponent >= 100:
        assert shares[3] == pytest.approx(1.0)


@pytest.mark.unit
def test_large_routing_exponent_keeps_depths_and_budget_finite():
    arrays, params = _gentle_fan(200.0)
    result = UrbanDrainageSimulator(arrays, params).run(np.full((6, 4), 30.0))
    assert np.isfinite(result.depth_m).all()
    assert result.mass_balance.is_finite and result.mass_balance.relative_error < 1e-9


@pytest.mark.unit
def test_non_finite_budget_raises_instead_of_returning_dry_labels(grid_arrays):
    huge = np.full((3, grid_arrays.num_nodes), np.finfo(np.float64).max)
    with np.errstate(all="ignore"), pytest.raises(SimulationError, match="non-finite"):
        UrbanDrainageSimulator(grid_arrays, _params()).run(huge)


# --------------------------------------------------------------------------- fix round: local surcharge and tailwater gate (R2-01)


@pytest.mark.unit
def test_contributing_operator_on_a_chain():
    arrays = _arrays([3.0, 2.0, 1.0], edges=[(0, 1), (1, 2)], lengths=[100.0, 60.0])
    net = build_drainage_network(arrays, _params(catchment_width_m=30.0))
    a0, a1, a2 = net.area_m2
    np.testing.assert_allclose(net.contributing_area_m2, [a0, a0 + a1, a0 + a1 + a2])
    expected = np.array([[1.0, 0.0, 0.0], [a0, a1, 0.0], [a0, a1, a2]])
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(net.contributing.toarray(), expected)


@pytest.mark.unit
def test_contributing_operator_is_row_stochastic_on_a_grid(grid_arrays):
    net = build_drainage_network(grid_arrays, _params())
    k = net.contributing.toarray()
    np.testing.assert_allclose(k.sum(axis=1), 1.0)
    assert (np.diag(k) > 0).all() and (k >= 0).all()
    assert (net.contributing_area_m2 >= net.area_m2 - 1e-9).all()
    assert net.contributing_area_m2[net.is_sink].max() > net.area_m2.max()  # sinks collect upstream area


def _two_valleys() -> GraphArrays:
    """Valley A (nodes 0-2 -> sink 2) and valley B (nodes 3-5 -> sink 5), not connected."""
    return _arrays([3.0, 2.0, 1.0, 3.0, 2.0, 1.0], edges=[(0, 1), (1, 2), (3, 4), (4, 5)],
                   lengths=[100.0] * 4, dist=[2000.0] * 6)


_VALLEY_PARAMS = dict(drain_capacity_near_mm_h=5.0, drain_capacity_far_mm_h=5.0, surcharge_window_h=6,
                      surcharge_threshold_mm=30.0, surcharge_ramp_mm=10.0, surcharge_min_factor=0.0,
                      tailwater_threshold_mm=0.0, tailwater_ramp_mm=0.1, infiltration_mm_h=0.0,
                      ponding_fraction=0.02, spill_depth_m=None, outflow_rate_per_h=1.0)


@pytest.mark.unit
def test_drains_far_from_the_cloudburst_keep_working():
    """Regression for R2-01: heavy rain on valley A must not surcharge valley B's drains."""
    rain = np.zeros((8, 6))
    rain[:6, :3] = 12.0   # 72 mm over 6 h on A: its drains surcharge
    rain[:6, 3:] = 4.0    # 24 mm on B: below the threshold, 4 mm/h < 5 mm/h capacity
    local = UrbanDrainageSimulator(_two_valleys(), _params(surcharge_local_weight=1.0, **_VALLEY_PARAMS)).run(rain)
    basin = UrbanDrainageSimulator(_two_valleys(), _params(surcharge_local_weight=0.0, **_VALLEY_PARAMS)).run(rain)
    assert local.flooded[:, 2].any() and not local.flooded[:, 3:].any()
    assert basin.flooded[:, 5].any(), "the legacy basin-wide switch surcharges every drain at once"


@pytest.mark.unit
def test_same_areal_rain_in_different_subcatchments_floods_different_junctions():
    sim = UrbanDrainageSimulator(_two_valleys(), _params(**_VALLEY_PARAMS))
    on_a, on_b = np.zeros((8, 6)), np.zeros((8, 6))
    on_a[:6, :3], on_b[:6, 3:] = 14.0, 14.0
    assert on_a.mean() == on_b.mean()
    flooded_a = sim.run(on_a).flooded.any(axis=0)
    flooded_b = sim.run(on_b).flooded.any(axis=0)
    assert flooded_a[2] and not flooded_a[5]
    assert flooded_b[5] and not flooded_b[2]


@pytest.mark.unit
def test_downstream_junction_surcharges_from_upstream_rain():
    """A sink whose own rain is light still surcharges when its contributing area is hammered."""
    params = _params(**{**_VALLEY_PARAMS, "surcharge_threshold_mm": 20.0})
    sim = UrbanDrainageSimulator(_two_valleys(), params)
    block = np.zeros((6, 6))
    block[:, :2] = 10.0   # upstream of sink 2 only
    loading = sim._surcharge_loading(block)
    assert loading[0, 2] > 0.0 and loading[0, 5] == 0.0
    factor, _ = sim._surcharge(block, sim_carry(sim))
    assert factor[-1, 2] < 1.0 and factor[-1, 5] == 1.0


def sim_carry(sim: UrbanDrainageSimulator):
    from src.hydrology.simulator import _Carry

    lead = sim.params.surcharge_window_h - 1
    return _Carry(np.zeros((sim.params.antecedent_window_h, sim.num_nodes)), np.zeros((lead, sim.num_nodes)),
                  np.zeros((lead, 1)))


@pytest.mark.unit
def test_tailwater_gate_blocks_backwater_over_a_dry_corridor():
    """A burst on one valley of an otherwise dry corridor drains freely (criterion (b) mechanism)."""
    rain = np.zeros((9, 6))
    rain[:7, :3] = 4.8    # below the 5 mm/h inlet capacity, but 28.8 mm per 6 h overloads the local drains
    local = {**_VALLEY_PARAMS, "surcharge_threshold_mm": 20.0, "surcharge_ramp_mm": 5.0}
    gated = UrbanDrainageSimulator(_two_valleys(), _params(**{**local, "tailwater_threshold_mm": 20.0}))
    open_gate = UrbanDrainageSimulator(_two_valleys(), _params(**local))
    assert not gated.run(rain).flooded[:, 2].any()
    assert open_gate.run(rain).flooded[:, 2].any()


@pytest.mark.unit
def test_tailwater_gate_ramp_and_history():
    params = _params(surcharge_window_h=2, tailwater_threshold_mm=4.0, tailwater_ramp_mm=4.0)
    basin = np.array([0.0, 2.0, 2.0, 4.0, 4.0, 0.0, 0.0])
    # trailing 2 h sums: 0, 2, 4, 6, 8, 4, 0
    np.testing.assert_allclose(tailwater_gate(basin, params), [0, 0, 0, 0.5, 1.0, 0, 0])
    np.testing.assert_allclose(tailwater_gate(basin[3:], params, history_mm=basin[:3]), [0.5, 1.0, 0, 0])


@pytest.mark.unit
def test_surcharge_factor_per_junction_with_history_matches_one_block():
    params = _params(surcharge_window_h=3, surcharge_threshold_mm=5.0, surcharge_ramp_mm=5.0, surcharge_min_factor=0.1)
    rng = np.random.default_rng(3)
    loading = rng.uniform(0.0, 4.0, (12, 4))
    whole = surcharge_factor(loading, params)
    split = np.concatenate([surcharge_factor(loading[:5], params),
                            surcharge_factor(loading[5:], params, history_mm=loading[:5])])
    np.testing.assert_allclose(split, whole)
    assert whole.shape == (12, 4) and whole.min() >= 0.1 - 1e-12 and whole.max() <= 1.0
    with pytest.raises(ValueError):
        surcharge_factor(loading[:, :, None], params)
    with pytest.raises(ValueError):
        surcharge_factor(loading, params, history_mm=np.zeros((2, 3)))


@pytest.mark.unit
def test_local_surcharge_run_is_chunk_invariant_and_mass_conserving(grid_arrays, grid_rain):
    params = _params(surcharge_threshold_mm=8.0, surcharge_ramp_mm=6.0, tailwater_threshold_mm=2.0,
                     tailwater_ramp_mm=3.0, surcharge_local_weight=0.7)
    full = UrbanDrainageSimulator(grid_arrays, params).run(grid_rain * 2)
    hourly = UrbanDrainageSimulator(grid_arrays, params, chunk_hours=1).run(grid_rain * 2)
    np.testing.assert_allclose(hourly.depth_m, full.depth_m, rtol=1e-6, atol=1e-7)
    assert full.mass_balance.relative_error < 1e-9
    assert full.flooded.any()
