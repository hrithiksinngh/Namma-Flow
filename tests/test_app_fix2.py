"""Regression tests of the dashboard for the final fix round.

X7 / F3-05 (skill KPI per mode: "given junction rain" for replays, the rain-field-ensemble areal
skill for forecast / what-if; the areal-skill table; the backtest ceiling row and warning-window
columns), F3-02 (typed remediation hints), F3-01 (share of rain fields at risk), F3-03 (physics
backtests), F3-06 (the pipeline's own water-exclusion predicate), F4-10 (offline graph build
wording / no silent OSM replacement), F5-09 (provenance-true header, synthetic-data banner) and
F1-04 (stored-block replay picker). Reuses the synthetic world of :mod:`tests.test_app`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import weather
from src.data_pipeline.graph_io import load_graph, save_graph
from src.inference import scenarios as sc
from src.inference.predictor import CheckpointMismatch, ModelNotFinalized, ModelNotReady, remediation
from src.inference.results import PredictionResult
from src.utils.config import deep_merge, resolve_path
from tests.conftest import make_grid_graph
from tests.test_app import (  # noqa: F401 - fixtures are used by name
    BACKTEST,
    METRICS,
    TZ,
    _fresh_caches,
    assert_clean,
    env,
    gnn_env,
    install_model,
    kpis,
    run_app,
    skill_kpi,
    storm_record,
    text_of,
    write_config,
)

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
import banners  # noqa: E402
import components as ui  # noqa: E402
import map_layers as ml  # noqa: E402
import metrics_view as mv  # noqa: E402

sys.path.remove(str(APP_DIR))

CHECKPOINT_STAMPS = {"created_utc": "2024-01-01T00:00:00+00:00", "finalized_utc": "2024-01-01T00:05:00+00:00"}
ROW = {"pr_auc": 0.6, "roc_auc": 0.95, "f2": 0.5, "precision": 0.4, "recall": 0.6, "csi": 0.3, "brier": 0.01,
       "ece": 0.02, "threshold": 0.4}
AREAL = {"status": "ok", "split": "test", "years": [2024], "members": 8, "threshold": 0.4,
         "checkpoint": {"run_id": "r1", "epoch": 3, **CHECKPOINT_STAMPS},
         "rows": {"exact_field": {**ROW, "pr_auc": 0.95}, "field_ensemble": {**ROW, "pr_auc": 0.61},
                  "single_other_field": {**ROW, "pr_auc": 0.46}, "uniform_areal": {**ROW, "pr_auc": 0.63},
                  "physics_other_field": {**ROW, "pr_auc": 0.39}},
         "note": "x", "created_utc": "2024-01-02T00:00:00+00:00"}
CEILING_BACKTEST = {**BACKTEST, "leads": {
    "lead_0h": {"lead_hours": 0, "field": {"exact": True, "members": 1, "seed_offset": 0},
                "rain": {"correlation": 1.0}, "junction_hours": {"pr_auc": 0.95},
                "junction_block_24h": {"pr_auc": 0.93, "recall": 0.9, "precision": 0.8},
                "physics_baseline": {"role": "teacher", "junction_hours": {"pr_auc": 1.0}}},
    "areal_ceiling": {"lead_hours": 0, "role": "areal_ceiling", "field": {"exact": False, "members": 8,
                                                                          "seed_offset": 1},
                      "rain": {"correlation": 1.0}, "junction_hours": {"pr_auc": 0.6},
                      "junction_block_6h": {"pr_auc": 0.62},
                      "junction_block_24h": {"pr_auc": 0.66, "recall": 0.7, "precision": 0.5},
                      "physics_baseline": {"role": "areal_baseline", "junction_hours": {"pr_auc": 0.45}}},
    "lead_24h": {"lead_hours": 24, "field": {"exact": False, "members": 8, "seed_offset": 1},
                 "rain": {"correlation": 0.2}, "junction_hours": {"pr_auc": 0.06},
                 "physics_baseline": {"role": "forecast_baseline", "junction_hours": {"pr_auc": 0.05}}}}}


def fake_forecast(cfg, now=None, hours=None):
    index = pd.date_range("2024-10-10 00:00", periods=96, freq="h", tz=TZ)
    rain = np.zeros(96)
    rain[60:64] = [4.0, 12.0, 6.0, 1.0]
    return sc.Scenario("Live forecast (issued 2024-10-11 23:00)", "forecast", index, rain, np.arange(96) >= 48, 48,
                       "Open-Meteo forecast: 23.0 mm over the next 48 h", "forecast")


def write_report(env_, name: str, payload: dict) -> None:
    path = resolve_path(env_.cfg, "reports_dir") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- X7 / F3-05: skill per mode (unit)


@pytest.mark.unit
def test_areal_skill_summary_prefers_the_matching_report():
    summary = mv.areal_skill_summary(AREAL, METRICS, CHECKPOINT_STAMPS)
    assert summary["field_ensemble_pr_auc"] == 0.61 and summary["members"] == 8
    assert summary["source"] == "areal_skill.json"
    stale = {**AREAL, "checkpoint": {"created_utc": "other", "finalized_utc": "other"}}
    stored = {"areal_skill": {"field_ensemble_pr_auc": 0.58, "uniform_areal_pr_auc": 0.6, "exact_field_pr_auc": 0.95,
                              "members": 8}}
    fallback = mv.areal_skill_summary(stale, stored, CHECKPOINT_STAMPS)
    assert fallback["field_ensemble_pr_auc"] == 0.58 and fallback["source"] == "metrics"
    assert mv.areal_skill_summary(stale, METRICS, CHECKPOINT_STAMPS) is None
    assert mv.areal_skill_summary({"status": "unavailable"}, None, CHECKPOINT_STAMPS) is None
    assert mv.areal_skill_summary(AREAL, None, None) is None                     # cannot verify the checkpoint


@pytest.mark.unit
def test_pr_auc_kpi_per_mode():
    replay = mv.pr_auc_kpi(METRICS, "gnn")
    assert replay.label == "PR-AUC given junction rain · test 2024" and replay.delta == "-0.13 vs GBDT"
    assert "exact synthetic junction rain" in replay.help
    areal = mv.pr_auc_kpi(METRICS, "gnn", exact_field=False, areal=mv.areal_skill_summary(AREAL, METRICS,
                                                                                         CHECKPOINT_STAMPS))
    assert areal.label == "PR-AUC given areal rain · test 2024" and areal.value == "0.61" and areal.delta is None
    assert "8 random junction rain fields" in areal.help and "0.42" in areal.help and "0.63" in areal.help
    caveat = mv.pr_auc_kpi(METRICS, "gnn", exact_field=False)
    assert caveat.label == "PR-AUC given junction rain · test 2024" and caveat.value == "0.42"
    assert caveat.delta == "not this mode's skill" and caveat.delta_color == "off"
    assert "python -m src.training.areal_skill" in caveat.help
    # replays: the simulator runs on the label field -> no independent score
    assert mv.pr_auc_kpi(METRICS, "physics", exact_field=True).value == "teacher"
    # forecast / what-if: averaged over rain fields it is a genuine predictor (V3-01)
    assert mv.pr_auc_kpi(METRICS, "physics", exact_field=False).value == "n/a"
    scored = mv.pr_auc_kpi(METRICS, "physics", exact_field=False,
                           areal={"physics_field_ensemble_pr_auc": 0.69, "members": 32})
    assert scored.value == "0.69" and "areal rain" in scored.label


@pytest.mark.unit
def test_areal_skill_table_title_and_notes():
    table = mv.areal_skill_table(AREAL)
    assert list(table.columns) == ["Junction rain given to the model", *mv.METRIC_COLUMNS]
    labels = table["Junction rain given to the model"].tolist()
    assert labels[0].startswith("Exact junction field") and labels[1].endswith("8 fields")
    assert table["PR-AUC"].tolist() == ["0.95", "0.61", "0.46", "0.63", "0.39"]
    assert mv.areal_skill_title(AREAL) == "Skill when only corridor-average rain is known (test 2024)"
    assert mv.areal_skill_note(AREAL, CHECKPOINT_STAMPS) is None
    assert "different checkpoint" in mv.areal_skill_note(AREAL, {"created_utc": "z", "finalized_utc": "z"})
    assert "Not computed yet" in mv.areal_skill_note(None, CHECKPOINT_STAMPS)
    assert "weather record missing" in mv.areal_skill_note({"status": "unavailable",
                                                            "reason": "weather record missing"}, None)
    assert mv.areal_skill_table({"status": "unavailable"}).empty and mv.areal_skill_table(None).empty
    partial = {**AREAL, "rows": {**AREAL["rows"], "physics_other_field": {"status": "unavailable",
                                                                         "reason": "MemoryError: x"}}}
    last = mv.areal_skill_table(partial).iloc[-1]
    assert "unavailable (MemoryError: x)" in last.iloc[0] and last["PR-AUC"] == "n/a"


@pytest.mark.unit
def test_backtest_table_has_the_ceiling_row_and_warning_windows():
    table = mv.backtest_table(CEILING_BACKTEST)
    assert list(table.index) == ["0 h ahead", "Perfect areal forecast (ensemble)", "24 h ahead"]
    assert table.loc["0 h ahead", "Rain field"] == "exact (label field)"
    assert table.loc["Perfect areal forecast (ensemble)", "Rain field"] == "ensemble of 8"
    assert table.loc["Perfect areal forecast (ensemble)", "PR-AUC"] == "0.6"
    assert table.loc["Perfect areal forecast (ensemble)", "24 h window PR-AUC"] == "0.66"
    assert table.loc["Perfect areal forecast (ensemble)", "6 h window PR-AUC"] == "0.62"
    assert table.loc["Perfect areal forecast (ensemble)", "Physics PR-AUC"] == "0.45"
    assert table.loc["0 h ahead", "Physics PR-AUC"] == mv.TEACHER_LABEL
    assert table.loc["24 h ahead", "24 h window recall"] == "n/a"
    old = mv.backtest_table(BACKTEST)                                    # reports without field info
    assert old.loc["24 h ahead", "Rain field"] == "one field"


@pytest.mark.unit
def test_stale_note_says_to_rebuild_the_datasets_first():
    """F3-02: retraining alone reuses the stale datasets, so the note names the dataset builder."""
    note = mv.stale_model_note({"dataset": {"stale": True}}, "zzz")
    assert "04_dataset_builder.py --force" in note and "train.py" in note
    assert note.index("04_dataset_builder") < note.index("train.py")


@pytest.mark.unit
def test_remediation_hint_is_chosen_by_error_type():
    what, hint = remediation(CheckpointMismatch("graph re-enriched (digest a vs b); ignored"))
    assert "04_dataset_builder.py --force" in hint and what.startswith("graph re-enriched")
    what, hint = remediation(ModelNotFinalized("The best.pt checkpoint is not finalized; --resume it"))
    assert "--resume" in hint and what == "The best.pt checkpoint is not finalized"
    what, hint = remediation(ModelNotReady("No trained model at best.pt; train one"))
    assert "train.py" in hint and what == "No trained model at best.pt"
    custom = CheckpointMismatch("lacks elevation; re-run stage 02", hint="re-run stage 02")
    assert remediation(custom) == ("lacks elevation", "re-run stage 02")


@pytest.mark.unit
def test_build_kpis_reports_the_range_across_rain_fields():
    members = np.array([[[0.9, 0.1]], [[0.2, 0.7]], [[0.0, 0.0]]], dtype=np.float32)
    result = PredictionResult(
        timestamps=pd.date_range("2024-10-01", periods=1, freq="h", tz=TZ), prob=members.mean(axis=0),
        prob_std=members.std(axis=0), node_rain=np.zeros((1, 2)), depth_physics=None, node_ids=(1, 2),
        lon=np.array([77.6, 77.7]), lat=np.array([12.9, 12.9]), static=pd.DataFrame({"node_id": [1, 2]}),
        threshold=0.3, horizons_h=(1,), scenario_name="unit", member_prob=members)
    tiles = ui.build_kpis(result, hours=1, peak_mode=True, hour_index=0, metrics=METRICS, predictor_kind="gnn",
                          exact_field=False, areal=mv.areal_skill_summary(AREAL, METRICS, CHECKPOINT_STAMPS))
    assert "the 3 rain-field realisations put 0–1 junctions at risk" in tiles[0].help
    assert tiles[0].delta == "50% of junctions · 0–1 across fields"
    assert tiles[4].label.startswith("PR-AUC given areal rain")


# --------------------------------------------------------------------------- F5-09 / F4-10 (unit)


@pytest.mark.unit
def test_header_caption_and_synthetic_banner_follow_the_provenance():
    osm = {"source": "osm_bbox", "elevation_source": "srtm", "drain_source": "osm"}
    demo = {"source": "synthetic_grid", "elevation_source": "synthetic", "drain_source": "synthetic_line"}
    assert "OpenStreetMap road graph" in banners.header_caption(osm, "gnn")
    assert "synthetic demo street grid" in banners.header_caption(demo, "gnn")
    assert "OpenStreetMap" not in banners.header_caption(demo, "gnn")
    assert "hydrology simulator" in banners.header_caption(osm, "physics")
    index = pd.date_range("2024-10-01", periods=3, freq="h", tz=TZ)
    replay = sc.Scenario("r", "historical", index, np.zeros(3), None, 0, source="synthetic", exact_field=True)
    real = sc.Scenario("r", "historical", index, np.zeros(3), None, 0, source="open_meteo", exact_field=True)
    assert banners.synthetic_inputs(osm, real) == [] and banners.synthetic_notice([]) is None
    found = banners.synthetic_inputs(demo, replay)
    assert found == ["the street grid", "the elevation", "the rain record"]
    notice = banners.synthetic_notice(found)
    assert notice.level == "warning" and "The street grid, the elevation and the rain record are synthetic" in \
        notice.message and "no physical meaning" in notice.message
    assert "The elevation is synthetic" in banners.synthetic_notice(["the elevation"]).message


# --------------------------------------------------------------------------- F3-06 (unit)


WATER = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"water": "wastewater"},
     "geometry": {"type": "Polygon", "coordinates": [[[77.66, 12.92], [77.661, 12.92], [77.661, 12.921]]]}},
    {"type": "Feature", "properties": {"water": "wastewater;pond"},
     "geometry": {"type": "Polygon", "coordinates": [[[77.67, 12.92], [77.671, 12.92], [77.671, 12.921]]]}},
    {"type": "Feature", "properties": {"water": ["pond", "Swimming_Pool"]},
     "geometry": {"type": "Polygon", "coordinates": [[[77.68, 12.92], [77.681, 12.92], [77.681, 12.921]]]}},
    {"type": "Feature", "properties": {"waterway": "drain"},
     "geometry": {"type": "LineString", "coordinates": [[77.66, 12.92], [77.67, 12.93]]}},
]}


@pytest.mark.unit
def test_water_exclusion_uses_the_pipelines_predicate():
    """F3-06: ';'-joined and list-valued water tags are matched exactly like drains._drop_excluded_water."""
    from src.data_pipeline import drains

    flags = ml.excluded_water_flags(["wastewater", "wastewater;pond", ["pond", "Swimming_Pool"], None],
                                    ("wastewater", "swimming_pool"))
    assert flags.tolist() == [True, True, True, False]
    assert ml.excluded_water_flags(["wastewater"], ()).tolist() == [False]
    layer = ml.parse_waterways(WATER, exclude_water=("wastewater",))
    assert len(layer.features) == 2 and "2 treatment tanks / pools excluded, as in the model" in layer.note
    as_string = drains._exclude_values({"exclude_water_values": "wastewater"})     # a string config is valid
    assert len(ml.parse_waterways(WATER, exclude_water=tuple(as_string)).features) == 2


# --------------------------------------------------------------------------- the app (AppTest)


@pytest.mark.e2e
def test_app_forecast_mode_shows_areal_skill_and_ceiling_row(gnn_env, monkeypatch):
    """X7 / F3-05: forecast mode shows the areal (rain-field-ensemble) skill; the About panel explains it."""
    write_report(gnn_env, "areal_skill.json", AREAL)
    write_report(gnn_env, "backtest.json", CEILING_BACKTEST)
    monkeypatch.setattr(sc, "forecast_scenario", fake_forecast)
    at = run_app()
    assert_clean(at)
    skill = skill_kpi(at)
    assert skill.label == "PR-AUC given areal rain · test 2024" and skill.value == "0.61" and not skill.proto.delta
    assert ":gray-badge[32 rain fields]" in text_of(at.markdown)
    assert "Skill when only corridor-average rain is known (test 2024)" in text_of(at.markdown)
    areal = next(d.value for d in at.dataframe if "Junction rain given to the model" in d.value.columns)
    assert areal["PR-AUC"].tolist()[:2] == ["0.95", "0.61"]
    backtest = next(d.value for d in at.dataframe if "Physics PR-AUC" in d.value.columns)
    assert "Perfect areal forecast (ensemble)" in backtest.index and "24 h window PR-AUC" in backtest.columns
    captions = text_of(at.caption)
    assert "Perfect areal forecast (ensemble)" in captions and "emulates the teacher" in captions
    assert "averaged over 32 stochastic rain-field realisations" in captions
    at_risk = at.metric[0]
    assert "rain-field realisations put" in at_risk.help
    top = next(d.value for d in at.dataframe if "Junction" in d.value.columns and "Rank" in d.value.columns)
    assert "Fields at risk" in top.columns and top["Fields at risk"].between(0, 1).all()


@pytest.mark.e2e
def test_app_design_mode_without_areal_skill_keeps_the_caveat(gnn_env):
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("design").run()
    assert_clean(at)
    skill = skill_kpi(at)
    assert skill.label == "PR-AUC given junction rain · test 2024" and skill.proto.delta == "not this mode's skill"
    assert "Not computed yet" in text_of(at.caption)


@pytest.mark.e2e
def test_app_replay_uses_the_exact_field_kpi(gnn_env):
    write_report(gnn_env, "areal_skill.json", AREAL)
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("historical").run()
    assert_clean(at)
    skill = skill_kpi(at)
    assert skill.label == "PR-AUC given junction rain · test 2024" and skill.proto.delta == "-0.13 vs GBDT"
    assert "rain fields]" not in text_of(at.markdown) and "exact training field" in text_of(at.caption)


@pytest.mark.e2e
@pytest.mark.parametrize("overrides, fix", [
    ({"graph_attributes_sha256": "0" * 16}, "04_dataset_builder.py --force && python src/training/train.py"),
    ({"finalized_utc": None}, "train.py --resume"),
])
def test_app_fallback_banner_gives_the_right_fix(env, overrides, fix):
    """F3-02: the banner keeps the type-specific remediation instead of 'Train the GNN'."""
    install_model(env, **overrides)
    at = run_app()
    assert_clean(at)
    warnings = text_of(at.warning)
    assert "Showing the physics baseline" in warnings and fix in warnings and "Train the GNN with" not in warnings


@pytest.mark.e2e
def test_app_physics_backtest_report_is_flagged(gnn_env):
    """F3-03: an old physics report in backtest.json is not presented as the GNN's backtest."""
    physics = {**BACKTEST, "predictor": {"kind": "physics", "label": "Physics baseline", "threshold": 0.5},
               "leads": {"lead_0h": {**BACKTEST["leads"]["lead_0h"], "role": "teacher"}}}
    write_report(gnn_env, "backtest.json", physics)
    at = run_app()
    assert_clean(at)
    assert "the physics baseline driven by archived" in text_of(at.markdown)
    assert "This backtest scored the **physics baseline**" in text_of(at.caption)
    backtest = next(d.value for d in at.dataframe if "Physics PR-AUC" in d.value.columns)
    assert backtest.loc["0 h ahead", "PR-AUC"] == mv.TEACHER_LABEL


@pytest.mark.e2e
def test_app_synthetic_demo_graph_header_and_banner(env):
    """F5-09: the header and a banner say the data are synthetic."""
    graph = make_grid_graph(4, 4)
    graph.graph.update(source="synthetic_grid", elevation_source="synthetic", drain_source="synthetic_line")
    save_graph(graph, env.cfg["paths"]["graph_file"])
    at = run_app()
    assert_clean(at)
    captions = text_of(at.caption)
    assert "synthetic demo street grid" in captions and "over the OpenStreetMap road graph" not in captions
    assert "Synthetic demo data" in text_of(at.warning)


@pytest.mark.e2e
def test_app_real_graph_has_no_synthetic_banner(env):
    graph = make_grid_graph(4, 4)
    graph.graph.update(source="osm_bbox", elevation_source="srtm", drain_source="osm")
    save_graph(graph, env.cfg["paths"]["graph_file"])
    at = run_app()
    assert_clean(at)
    assert "over the OpenStreetMap road graph" in text_of(at.caption)
    assert "Synthetic demo data" not in text_of(at.warning)


@pytest.mark.e2e
def test_app_map_excludes_water_like_the_pipeline_with_a_string_config(env):
    """F3-06: drains.exclude_water_values as a string is parsed by the drains module, not split into letters."""
    graph = make_grid_graph(4, 4)
    graph.graph.update(source="osm_bbox", elevation_source="srtm", drain_source="osm")
    save_graph(graph, env.cfg["paths"]["graph_file"])
    Path(env.cfg["paths"]["waterways_file"]).write_text(json.dumps(WATER), encoding="utf-8")
    write_config(deep_merge(env.cfg, {"drains": {"exclude_water_values": "wastewater"}}), env.path)
    at = run_app()
    assert_clean(at)
    assert "2 treatment tanks / pools excluded, as in the model" in text_of(at.caption)
    write_config(deep_merge(env.cfg, {"drains": {"exclude_water_values": [1, 2]}}), env.path)
    at = run_app()
    assert not at.exception and "drains.exclude_water_values" in text_of(at.error)


@pytest.mark.e2e
def test_app_setup_page_describes_the_offline_build(env):
    """F4-10: the setup page says the build replays the osmnx cache when one exists, else a synthetic grid."""
    Path(env.cfg["paths"]["graph_file"]).unlink()
    at = run_app()
    assert not at.exception
    text = text_of(at.markdown)
    assert "replayed from the osmnx response cache" in text and "else a **synthetic street grid**" in text
    assert "never replaced by the synthetic grid" in text


@pytest.mark.unit
def test_offline_build_never_replaces_a_readable_osm_graph(tmp_path, monkeypatch):
    """F4-10: allow_synthetic only when there is no graph file or it is itself synthetic."""
    missing = tmp_path / "missing.graphml"
    assert banners.may_replace_graph(missing)
    for source, allowed in (("synthetic_grid", True), ("osm_bbox", False), ("osm_place", False)):
        graph = make_grid_graph(3, 3)
        graph.graph["source"] = source
        path = tmp_path / f"{source}.graphml"
        save_graph(graph, path)
        assert load_graph(path).graph["source"] == source
        assert banners.may_replace_graph(path) is allowed
    broken = tmp_path / "broken.graphml"
    broken.write_text("<not graphml", encoding="utf-8")
    assert banners.may_replace_graph(broken)


@pytest.mark.e2e
def test_app_replay_picker_lists_stored_blocks_and_refuses_gaps(env):
    """F1-04: the replay caption lists the stored blocks and a start in a gap is refused, not zero-filled."""
    later = storm_record()
    later.index = later.index + pd.Timedelta(days=61)
    weather.save_weather_csv(pd.concat([storm_record(), later]), env.cfg["paths"]["weather_file"])
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("historical").run()
    assert_clean(at)
    caption = text_of(at.sidebar.caption)
    assert "2022-09-01 → 2022-09-20, 2022-11-01 → 2022-11-20" in caption and "cannot be replayed" in caption
    at.sidebar.selectbox(key="nf_event").set_value("__custom__").run()
    at.sidebar.date_input(key="nf_day").set_value(pd.Timestamp("2022-10-10").date()).run()
    assert not at.exception and "falls in a gap of the weather record" in text_of(at.error)
