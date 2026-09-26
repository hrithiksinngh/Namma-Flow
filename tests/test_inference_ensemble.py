"""Regression tests for the rain-field ensemble serving (contract X5; findings F1-01 and F3-01).

Training labels come from ONE stochastic junction rain field per event; the model is trained on
that exact field. Replays reproduce it (``Scenario.exact_field``); forecasts, design storms and
custom series know only corridor-average rain, so predictors average over K independent field
realisations (member k at offset o: ``rainfall_field.seed = base + (o + k) * stride``).
Shares the offline fixtures of :mod:`tests.test_inference`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.rain_field import downscale_rainfall
from src.inference import field_ensemble as fe
from src.inference import scenarios as sc
from src.inference.predictor import FloodPredictor, PhysicsPredictor
from src.inference.results import PredictionResult
from src.utils.config import ConfigError, deep_merge, load_config
from tests.test_inference import (  # noqa: F401 - fixtures are used by name
    NOW,
    TZ,
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

BASE_SEED = 42
STRIDE = 7919
# The shipped default ensemble size (config/config.yaml inference.field_members); tests follow the config.
MEMBERS = int(load_config()["inference"]["field_members"])


def _seeded(cfg: dict, seed: int) -> dict:
    return deep_merge(cfg, {"rainfall_field": {"seed": seed}})


def _member_rain(scenario: sc.Scenario, graph, cfg: dict, seeds) -> np.ndarray:
    return np.stack([downscale_rainfall(scenario.areal_mm, scenario.field_timestamps, graph.lon, graph.lat,
                                        _seeded(cfg, seed)) for seed in seeds])


# --------------------------------------------------------------------------- plans & settings


@pytest.mark.unit
def test_field_plan_seeds_offsets_and_exact_field(icfg):
    settings = sc.InferenceSettings.from_config(icfg)
    assert (settings.field_members, settings.field_seed_stride) == (MEMBERS, STRIDE)
    exact = fe.field_plan(icfg, settings, exact_field=True, field_members=5, field_seed_offset=3)
    assert (exact.members, exact.seeds, exact.exact, exact.offset) == (1, (BASE_SEED,), True, 0)
    default = fe.field_plan(icfg, settings, exact_field=False)
    assert default.members == MEMBERS and default.seeds == tuple(BASE_SEED + k * STRIDE for k in range(MEMBERS))
    assert default.seeds[0] == BASE_SEED                           # offset 0 / member 0 = the old single draw
    forecast = fe.field_plan(icfg, settings, exact_field=False, field_members=3, field_seed_offset=1)
    assert forecast.seeds == (BASE_SEED + STRIDE, BASE_SEED + 2 * STRIDE, BASE_SEED + 3 * STRIDE)
    assert BASE_SEED not in forecast.seeds                         # offset >= 1 never reuses the label field
    assert forecast.metadata() == {"field_members": 3, "field_seed_offset": 1, "field_seeds": list(forecast.seeds),
                                   "exact_field": False}
    uniform = fe.field_plan(deep_merge(icfg, {"rainfall_field": {"enabled": False}}), settings, exact_field=False)
    assert uniform.members == 1 and uniform.uniform and "uniform junction rain" in fe.rain_field_note(uniform)
    with pytest.raises(ValueError, match="field_members"):
        fe.field_plan(icfg, settings, exact_field=False, field_members=0)
    with pytest.raises(ValueError, match="exceeds"):
        fe.field_plan(icfg, settings, exact_field=False, field_members=65)
    with pytest.raises(ValueError, match="field_seed_offset"):
        fe.field_plan(icfg, settings, exact_field=False, field_seed_offset=-1)
    assert fe.passes_for(8, 0) == 8 and fe.passes_for(8, 20) == 20 and fe.passes_for(8, 3) == 8


@pytest.mark.unit
@pytest.mark.parametrize("override, message", [
    ({"field_members": 0}, "field_members"), ({"field_members": True}, "field_members"),
    ({"field_members": 65}, "field_members"), ({"field_seed_stride": 0}, "field_seed_stride"),
])
def test_field_ensemble_settings_are_validated(cfg, override, message):
    with pytest.raises(ConfigError, match=message):
        sc.InferenceSettings.from_config(deep_merge(cfg, {"inference": override}))


@pytest.mark.unit
def test_rain_field_notes_name_the_ensemble(icfg):
    settings = sc.InferenceSettings.from_config(icfg)
    ensemble = fe.field_plan(icfg, settings, exact_field=False)
    assert fe.rain_field_note(ensemble).startswith(f"junction rain averaged over {ensemble.members} stochastic rain-field realisations")
    assert "canonical storm-anchored" in fe.rain_field_note(ensemble, design_storm=True)
    assert fe.rain_field_note(fe.field_plan(icfg, settings, exact_field=True)).startswith("exact training field")
    single = fe.field_plan(icfg, settings, exact_field=False, field_members=1)
    assert "one stochastic rain-field realisation" in fe.rain_field_note(single)


@pytest.mark.unit
def test_member_rain_uses_each_member_seed(icfg, graph):
    scenario = custom(icfg)
    plan = fe.field_plan(icfg, sc.InferenceSettings.from_config(icfg), exact_field=False, field_members=3,
                         field_seed_offset=1)
    rain = fe.member_rain(icfg, plan, scenario.areal_mm, scenario.field_timestamps, graph.lon, graph.lat)
    assert rain.shape == (3, scenario.n_hours, graph.num_nodes) and rain.dtype == np.float32
    np.testing.assert_array_equal(rain, _member_rain(scenario, graph, icfg, plan.seeds))
    assert not np.allclose(rain[0], rain[1])                        # independent realisations
    np.testing.assert_allclose(rain.mean(axis=2)[:, 60], [scenario.areal_mm[60]] * 3, rtol=1e-5)  # areal kept


# --------------------------------------------------------------------------- the GNN ensemble


@pytest.mark.integration
def test_gnn_ensemble_is_the_mean_of_the_member_predictions(icfg, graph, predictor):
    """F1-01: forecast / what-if probabilities marginalise over K independent rain fields."""
    scenario = custom(icfg)
    result = predictor.predict(scenario, field_members=3, field_seed_offset=1, keep_members=True)
    seeds = [BASE_SEED + k * STRIDE for k in (1, 2, 3)]
    rain = _member_rain(scenario, graph, icfg, seeds)
    singles = np.stack([predictor.predict_rain(rain[k], scenario.timestamps, scenario.target_start).prob
                        for k in range(3)])
    np.testing.assert_allclose(result.prob, singles.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.prob_std, singles.std(axis=0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(result.member_prob, singles, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.node_rain, rain[:, scenario.target_start:].mean(axis=0), rtol=1e-6)
    meta = result.metadata
    assert (meta["field_members"], meta["field_seed_offset"], meta["passes"], meta["mc_samples"]) == (3, 1, 3, 0)
    assert meta["field_seeds"] == seeds and meta["exact_field"] is False
    assert meta["rain_field"].startswith("junction rain averaged over 3 stochastic rain-field realisations")
    direct = predictor.predict_rain(rain, scenario.timestamps, scenario.target_start)   # [K, T, N] input
    np.testing.assert_allclose(direct.prob, result.prob, rtol=1e-6)
    assert direct.metadata["field_members"] == 3


@pytest.mark.integration
def test_exact_field_scenarios_use_the_training_field(icfg, graph, predictor, record_cfg):
    """Replays keep the exact training field (one member, the base seed): emulation of the teacher."""
    replay = sc.historical_scenario(record_cfg, "2022-09-05 12:00", hours=12)
    assert replay.exact_field
    result = predictor.predict(replay, field_members=5, field_seed_offset=2, keep_members=True)
    assert result.prob_std is None and result.metadata["field_seeds"] == [BASE_SEED]
    assert result.member_prob is None and result.at_risk_range(12) is None     # one field: no spread to show
    assert result.metadata["rain_field"].startswith("exact training field")
    single = predictor.predict(dataclasses.replace(replay, exact_field=False), field_members=1)
    np.testing.assert_array_equal(result.prob, single.prob)          # offset 0 / member 0 = the training seed


@pytest.mark.integration
def test_default_ensemble_for_forecast_like_scenarios(icfg, predictor):
    result = predictor.predict(custom(icfg))
    assert result.metadata["field_members"] == MEMBERS and result.metadata["passes"] == MEMBERS
    assert result.prob_std is not None and result.prob_std.max() > 0 and result.member_prob is None
    assert result.summary(48)["field_members"] == MEMBERS and result.summary(48)["junctions_at_risk_range"] is None


@pytest.mark.integration
def test_mc_dropout_passes_cycle_through_the_members(icfg, graph):
    """pass i uses member i % K; without dropout layers the MC passes only replay the members."""
    save_checkpoint(icfg, make_checkpoint(icfg, graph, dropout=0.0))
    model = FloodPredictor.from_artifacts(icfg)
    scenario = custom(icfg)
    members = model.predict(scenario, field_members=2, keep_members=True)
    mc = model.predict(scenario, field_members=2, mc_samples=5, keep_members=True)
    assert mc.metadata["passes"] == 5 and mc.metadata["mc_samples"] == 5
    p0, p1 = members.member_prob
    np.testing.assert_allclose(mc.prob, (3 * p0 + 2 * p1) / 5, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mc.member_prob, members.member_prob, rtol=1e-5, atol=1e-6)
    few = model.predict(scenario, field_members=3, mc_samples=2)
    assert few.metadata["passes"] == 3                               # max(members, mc_samples)


@pytest.mark.integration
def test_mc_dropout_with_members_is_deterministic(icfg, predictor):
    scenario = custom(icfg)
    first = predictor.predict(scenario, field_members=2, mc_samples=4)
    again = predictor.predict(scenario, field_members=2, mc_samples=4)
    np.testing.assert_array_equal(first.prob, again.prob)
    np.testing.assert_array_equal(first.prob_std, again.prob_std)
    assert not predictor.model.training


@pytest.mark.integration
def test_member_batching_does_not_change_results(icfg, graph, ckpt_path):
    scenario = custom(icfg, hours=100, storm_at=60, target_start=40)
    one = FloodPredictor.from_artifacts(deep_merge(icfg, {"inference": {"batch_windows": 1}}))
    many = FloodPredictor.from_artifacts(deep_merge(icfg, {"inference": {"batch_windows": 8}}))
    a = one.predict(scenario, field_members=3, keep_members=True)
    b = many.predict(scenario, field_members=3, keep_members=True)
    np.testing.assert_allclose(a.prob, b.prob, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a.member_prob, b.member_prob, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_gnn_physics_depth_is_the_member_mean(icfg, graph, predictor):
    scenario = custom(icfg, peak=40.0)
    result = predictor.predict(scenario, field_members=2, include_physics=True)
    physics = PhysicsPredictor(icfg, graph)
    rain = _member_rain(scenario, graph, icfg, [BASE_SEED, BASE_SEED + STRIDE])
    depth = np.mean([physics._physics_depth(r)[scenario.target_start:] for r in rain], axis=0)
    np.testing.assert_allclose(result.depth_physics, depth, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- the physics ensemble


@pytest.mark.integration
def test_physics_ensemble_averages_probability_and_depth(icfg, graph):
    physics = PhysicsPredictor.from_graph(icfg)
    scenario = sc.design_storm_scenario(icfg, 80.0, 3, now=NOW, rain_scale=1.0)
    result = physics.predict(scenario, field_members=3, keep_members=True)
    rain = _member_rain(scenario, graph, icfg, [BASE_SEED + k * STRIDE for k in range(3)])
    t0 = scenario.target_start
    singles = [physics.predict_rain(r, scenario.timestamps, t0) for r in rain]
    np.testing.assert_allclose(result.prob, np.mean([s.prob for s in singles], axis=0), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result.depth_physics, np.mean([s.depth_physics for s in singles], axis=0),
                               rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result.prob_std, np.std([s.prob for s in singles], axis=0), rtol=1e-4, atol=1e-6)
    assert result.metadata["passes"] == 3 and result.member_prob.shape == (3, *result.prob.shape)


# --------------------------------------------------------------------------- design storms (F3-01)


@pytest.mark.integration
def test_design_storms_rank_on_the_ensemble_mean_with_shared_members(icfg, predictor):
    """F3-01: what-if junction answers no longer hinge on one arbitrary draw; members are common to all storms."""
    moderate = sc.preset_scenario(icfg, "moderate", now=NOW)
    heavy = sc.preset_scenario(icfg, "heavy", now=NOW)
    a = predictor.predict(moderate, keep_members=True)
    b = predictor.predict(heavy, keep_members=True)
    assert a.metadata["field_seeds"] == b.metadata["field_seeds"]    # common random numbers across storms
    assert "canonical storm-anchored" in a.metadata["rain_field"]
    single = predictor.predict(heavy, field_members=1)               # the old single-draw answer
    assert not np.allclose(b.prob, single.prob)
    spread = b.at_risk_range(48)
    assert spread is not None and spread[0] <= spread[1] <= b.n_nodes
    share = b.node_table(48)["field_share_at_risk"]
    assert share.between(0, 1).all() and set(np.round(share * 8).astype(int)) <= set(range(9))


# --------------------------------------------------------------------------- PredictionResult members


def _result(member_prob, threshold=0.5) -> PredictionResult:
    member_prob = np.asarray(member_prob, dtype=np.float32)
    k, n_hours, n_nodes = member_prob.shape
    return PredictionResult(
        timestamps=pd.date_range("2024-10-01", periods=n_hours, freq="h", tz=TZ), prob=member_prob.mean(axis=0),
        prob_std=member_prob.std(axis=0), node_rain=np.zeros((n_hours, n_nodes)), depth_physics=None,
        node_ids=tuple(range(n_nodes)), lon=np.linspace(77.6, 77.7, n_nodes), lat=np.full(n_nodes, 12.9),
        static=pd.DataFrame({"node_id": range(n_nodes)}), threshold=threshold, horizons_h=(1, 2),
        scenario_name="unit", member_prob=member_prob)


@pytest.mark.unit
def test_result_member_share_and_range():
    members = [[[0.9, 0.1, 0.0], [0.2, 0.6, 0.0]],     # member 0: junctions 0 and 1 at risk (at some hour)
               [[0.1, 0.1, 0.0], [0.7, 0.2, 0.0]],     # member 1: junction 0
               [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]     # member 2: none
    result = _result(members)
    assert result.n_members == 3
    assert result.members_at_risk(2).tolist() == [2, 1, 0]
    assert result.at_risk_range(2) == (0, 2) and result.at_risk_range(2, hour_index=0) == (0, 1)
    assert result.node_table(2)["field_share_at_risk"].round(3).tolist() == [0.667, 0.333, 0.0]
    summary = result.summary(2)
    assert summary["junctions_at_risk_range"] == [0, 2] and summary["field_members"] == 3
    assert "field_share_at_risk" in result.to_geojson(2)["features"][0]["properties"]
    plain = dataclasses.replace(result, member_prob=None)
    assert plain.members_at_risk(2) is None and plain.at_risk_range(2) is None
    assert plain.node_table(2)["field_share_at_risk"].isna().all()


@pytest.mark.unit
def test_result_member_prob_validation():
    good = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="member_prob"):
        dataclasses.replace(_result(good), member_prob=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="member_prob"):
        dataclasses.replace(_result(good), member_prob=np.zeros((2, 5, 3)))
    with pytest.raises(ValueError, match="member_prob"):
        dataclasses.replace(_result(good), member_prob=np.full((2, 2, 3), 1.5))
    assert _result(good).member_prob.flags.writeable is False


@pytest.mark.unit
def test_scenario_exact_field_flag_is_validated():
    index = pd.date_range("2024-10-01", periods=3, freq="h", tz=TZ)
    assert sc.Scenario("s", "custom", index, np.zeros(3), None, 0).exact_field is False
    assert sc.Scenario("s", "historical", index, np.zeros(3), None, 0, exact_field=np.bool_(True)).exact_field is True
    with pytest.raises(ValueError, match="exact_field"):
        sc.Scenario("s", "custom", index, np.zeros(3), None, 0, exact_field="yes")


@pytest.mark.integration
def test_predict_rain_rejects_bad_member_shapes(predictor):
    index = pd.date_range("2024-10-01", periods=10, freq="h", tz=TZ)
    with pytest.raises(ValueError, match="members"):
        predictor.predict_rain(np.zeros((2, 10, 15)), index, 0)
    with pytest.raises(ValueError, match="members"):
        predictor.predict_rain(np.zeros((1, 2, 10, 16)), index, 0)
    with pytest.raises(ValueError, match="length 10"):
        predictor.predict_rain(np.zeros((2, 10, 16)), index[:5], 0)
