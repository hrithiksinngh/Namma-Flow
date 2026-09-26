"""Unit tests of the dashboard's pure helpers: ``app/map_layers.py`` (layers, colours, water),
``app/components.py`` (settings, formatting, KPIs, frames, charts, provenance) and
``app/metrics_view.py`` (headline split wording, baselines, backtest and model-card tables).

The Streamlit app itself is exercised end to end in :mod:`tests.test_app`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydeck as pdk
import pytest

from src.inference import scenarios as sc
from src.inference.results import PredictionResult
from src.utils.config import ConfigError, project_root
from tests.conftest import BBOX
from tests.test_app import APP_DIR, BACKTEST, LEGACY_METRICS, METRICS, PLATT, TIERS, TZ, WATERWAYS, ml, ui

sys.path.insert(0, str(APP_DIR))
import metrics_view as mv  # noqa: E402

sys.path.remove(str(APP_DIR))


def make_result(prob: np.ndarray, *, std: np.ndarray | None = None, threshold: float = 0.4,
                areal: np.ndarray | None = None) -> PredictionResult:
    t, n = prob.shape
    index = pd.date_range("2022-09-05 12:00", periods=t, freq="h", tz=TZ)
    static = pd.DataFrame({"node_id": list(range(n)), "elevation": np.linspace(870, 880, n),
                           "dist_to_drain_m": np.full(n, 100.0), "relative_elevation": np.linspace(-2, 2, n)})
    return PredictionResult(timestamps=index, prob=prob, prob_std=std, node_rain=np.ones((t, n)),
                            depth_physics=np.full((t, n), 0.1), node_ids=tuple(range(n)),
                            lon=np.linspace(77.66, 77.69, n), lat=np.linspace(12.92, 12.94, n), static=static,
                            threshold=threshold, horizons_h=(2, 4), scenario_name="test", risk_tiers=TIERS,
                            predictor="gnn", areal_mm=None if areal is None else areal)


# --------------------------------------------------------------------------- map_layers: colours & tiers


@pytest.mark.unit
@pytest.mark.parametrize("value, text", [(0.82, "82%"), (0.043, "4.3%"), (3.6e-5, "3.6e-05"), (0.0, "0%"),
                                         (None, "n/a"), (float("nan"), "n/a"), ("x", "n/a"), (1.0, "100%")])
def test_format_probability(value, text):
    assert ml.format_probability(value) == text


@pytest.mark.unit
def test_clean_probabilities_and_ramp():
    p = ml.clean_probabilities([np.nan, -1.0, 0.5, 2.0, np.inf])
    assert p.tolist() == [0.0, 0.0, 0.5, 1.0, 0.0]
    with pytest.raises(ValueError, match="numeric"):
        ml.clean_probabilities(["a"])
    colours = ml.ramp_rgb([0.0, 0.5, 1.0])
    assert colours.dtype == np.uint8 and colours.shape == (3, 3)
    assert tuple(colours[0]) == ml.RAMP_STOPS[0][1] and tuple(colours[-1]) == ml.RAMP_STOPS[-1][1]
    assert colours[0, 0] < colours[-1, 0] and colours[0, 1] > colours[-1, 1]   # green -> red


@pytest.mark.unit
def test_tiers_boundaries_palette_and_validation():
    names = ml.tier_names([0.0, 0.2499, 0.25, 0.5, 0.74, 0.75, 1.0, np.nan], TIERS)
    assert names.tolist() == ["Low", "Low", "Moderate", "High", "High", "Severe", "Severe", "Low"]
    palette = ml.tier_palette(TIERS)
    assert list(palette) == ["Low", "Moderate", "High", "Severe"]
    assert palette["Low"] == ml.RAMP_STOPS[0][1] and palette["Severe"] == ml.RAMP_STOPS[-1][1]
    assert ml.tier_palette([("Only", 0.0)]) == {"Only": ml.RAMP_STOPS[0][1]}
    assert ml.validate_tiers(None) == ml.DEFAULT_TIERS
    assert ml.validate_tiers([("B", 0.5), ("A", 0.0)]) == (("A", 0.0), ("B", 0.5))
    for bad in ([("A", 0.1)], [], [("A", 0.0), ("B", 0.0)], [("A", 0.0), ("B", 1.0)], [("A", "x")], [1]):
        with pytest.raises(ValueError):
            ml.validate_tiers(bad)


@pytest.mark.unit
def test_color_scale_tiers_and_relative():
    tiers = ml.ColorScale("tiers", TIERS)
    assert tiers.heights([0.0, 0.5, 1.0]).tolist() == [0.0, 50.0, 100.0]
    assert tuple(tiers.rgb([0.8])[0]) == ml.tier_palette(TIERS)["Severe"]
    assert tiers.rgba([0.1], alpha=300)[0, 3] == 255
    assert [label for label, _ in tiers.legend()][0].startswith("Low (0%–25%)")
    assert "risk tier" in tiers.describe()
    relative = ml.ColorScale("relative", TIERS, vmax=0.01)
    assert relative.heights([0.005, 0.02]).tolist() == [50.0, 100.0]
    assert tuple(relative.rgb([0.01])[0]) == ml.RAMP_STOPS[-1][1]
    legend = relative.legend()
    assert legend[0][0] == "p = 0" and legend[-1][0] == "p = 1.0%" and len(legend) == 5
    assert "relative" in relative.describe().lower()
    for kwargs in ({"mode": "rainbow"}, {"mode": "relative", "vmax": 0.0}, {"mode": "relative", "vmax": 2.0},
                   {"mode": "relative", "vmax": float("nan")}):
        with pytest.raises(ValueError):
            ml.ColorScale(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("mode, top, threshold, expected, vmax", [
    ("auto", 0.6, 0.4, "tiers", 1.0),
    ("auto", 3e-5, 3.6e-5, "relative", 3.6e-5),     # every junction in the lowest tier
    ("auto", 0.02, 1e-4, "relative", 0.02),
    ("relative", 0.0, 0.0, "relative", 1.0),       # dry day, degenerate threshold
    ("relative", np.nan, 0.3, "relative", 0.3),
    ("tiers", 1e-6, 0.4, "tiers", 1.0),
])
def test_resolve_color_scale(mode, top, threshold, expected, vmax):
    scale = ml.resolve_color_scale(mode, top, threshold, TIERS)
    assert scale.mode == expected and math.isclose(scale.vmax, vmax)


@pytest.mark.unit
def test_resolve_color_scale_rejects_unknown_mode_and_single_tier():
    with pytest.raises(ValueError, match="colour scale"):
        ml.resolve_color_scale("pastel", 0.5, 0.5)
    assert ml.resolve_color_scale("auto", 0.9, 0.5, [("Only", 0.0)]).mode == "relative"


# --------------------------------------------------------------------------- map_layers: frames


def node_table(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": [101, 102, 103][:n], "lon": [77.66, 77.67, 77.68][:n], "lat": [12.92, 12.93, 12.94][:n],
        "elevation": [870.0, np.nan, 880.0][:n], "relative_elevation": [-1.5, 0.5, np.nan][:n],
        "dist_to_drain_m": [120.0, 800.0, np.nan][:n], "max_prob": [0.9, 0.3, 0.0][:n],
        "peak_time": [pd.Timestamp("2022-09-05 17:00", tz=TZ), pd.NaT, pd.NaT][:n],
        "max_depth_m": [0.31, np.nan, 0.0][:n], "peak_rain_mm_h": [30.0, 2.0, np.nan][:n],
    })


@pytest.mark.unit
def test_junction_frame_columns_colours_and_tooltips():
    table = node_table()
    original = table.copy()
    scale = ml.ColorScale("tiers", TIERS)
    frame = ml.junction_frame(table, [0.9, 0.3, np.nan], scale, std=[0.05, np.nan, np.nan], when="peak")
    assert list(frame.columns) == list(ml.JUNCTION_COLUMNS)
    assert frame["node_id"].tolist() == ["101", "102", "103"]
    assert frame["tier"].tolist() == ["Severe", "Moderate", "Low"]
    assert frame["height"].tolist() == pytest.approx([90.0, 30.0, 0.0])
    assert all(len(c) == 4 for c in frame["color"])
    first, second = frame["tooltip"][0], frame["tooltip"][1]
    assert first.split("\n")[0] == "Junction 101" and "<" not in first
    assert "Flood probability (peak): 90% ± 5.0%" in first and "local depression" in first
    assert "Simulated peak depth: 0.31 m" in first and "Peak: Mon 05 Sep 17:00" in first
    assert "±" not in second and "Elevation" not in second and "Peak:" not in second
    pd.testing.assert_frame_equal(table, original)          # the caller's table is not modified


@pytest.mark.unit
def test_junction_frame_validation():
    scale = ml.ColorScale("tiers")
    with pytest.raises(TypeError):
        ml.junction_frame([1, 2], [0.1], scale)
    with pytest.raises(ValueError, match="missing columns"):
        ml.junction_frame(pd.DataFrame({"node_id": [1]}), [0.1], scale)
    with pytest.raises(ValueError, match="probabilities"):
        ml.junction_frame(node_table(), [0.1], scale)
    with pytest.raises(ValueError, match="std"):
        ml.junction_frame(node_table(), [0.1, 0.2, 0.3], scale, std=[0.1])
    tooltip = ml.junction_frame(node_table(1).assign(node_id=["<b>x</b>"]), [0.1], scale)["tooltip"][0]
    assert tooltip.startswith("Junction <b>x</b>\n")     # plain text: the browser shows it verbatim


@pytest.mark.unit
def test_undirected_segments_dedupes_and_validates():
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 2, 0]])
    assert ml.undirected_segments(edges, 4).tolist() == [[0, 1], [0, 3], [1, 2]]
    assert ml.undirected_segments(np.zeros((2, 0)), 3).shape == (0, 2)
    assert ml.undirected_segments(np.array([[1], [1]]), 3).shape == (0, 2)
    with pytest.raises(ValueError, match="shape"):
        ml.undirected_segments(np.zeros((3, 2)), 3)
    with pytest.raises(ValueError, match="outside"):
        ml.undirected_segments(np.array([[0], [5]]), 3)


@pytest.mark.unit
def test_road_frame_colours_by_mean_endpoint_probability():
    scale = ml.ColorScale("tiers", TIERS)
    lon, lat = np.array([77.66, 77.67, 77.68]), np.array([12.92, 12.93, 12.94])
    frame = ml.road_frame(lon, lat, np.array([[0, 1], [1, 2]]), [0.9, 0.5, 0.0], scale)
    assert list(frame.columns) == list(ml.ROAD_COLUMNS)
    assert frame["prob"].tolist() == pytest.approx([0.7, 0.25])
    assert frame["source"][0] == [77.66, 12.92] and frame["target"][1] == [77.68, 12.94]
    assert frame["color"][0][3] == ml.ROAD_ALPHA and "70%" in frame["tooltip"][0]
    empty = ml.road_frame(lon, lat, np.zeros((0, 2), dtype=int), [0.1, 0.2, 0.3], scale)
    assert empty.empty and list(empty.columns) == list(ml.ROAD_COLUMNS)
    with pytest.raises(ValueError, match="align"):
        ml.road_frame(lon, lat, np.array([[0, 1]]), [0.1], scale)
    with pytest.raises(ValueError, match="unknown"):
        ml.road_frame(lon, lat, np.array([[0, 7]]), [0.1, 0.2, 0.3], scale)


# --------------------------------------------------------------------------- map_layers: water


@pytest.mark.unit
def test_parse_waterways_geometry_types_and_labels():
    data = ml.parse_waterways(WATERWAYS)
    assert data.source == "osm" and data.n_lines == 3 and data.n_polygons == 2
    labels = [f.label for f in data.features]
    assert "ORR drain (Drain)" in labels and "Bellandur Lake (Lake)" in labels and "Stream" in labels
    lake = next(f for f in data.features if f.label.startswith("Bellandur"))
    assert len(lake.coords) == 2                               # exterior ring + hole
    lines, polygons = ml.water_frames(data)
    assert len(lines) == 3 and len(polygons) == 2 and lines["path"][0] == [[77.66, 12.92], [77.67, 12.93]]
    empty_lines, empty_polygons = ml.water_frames(None)
    assert empty_lines.empty and empty_polygons.empty
    for bad in ({"type": "Feature"}, {"type": "FeatureCollection", "features": {}}, []):
        with pytest.raises(ValueError):
            ml.parse_waterways(bad)


@pytest.mark.unit
def test_parse_waterways_caps_feature_count(monkeypatch):
    monkeypatch.setattr(ml, "MAX_WATER_FEATURES", 2)
    assert len(ml.parse_waterways(WATERWAYS).features) == 2


@pytest.mark.unit
def test_load_waterways_missing_corrupt_and_valid(tmp_path):
    assert ml.load_waterways(tmp_path / "missing.geojson") is None
    corrupt = tmp_path / "corrupt.geojson"
    corrupt.write_text("{not json", encoding="utf-8")
    assert ml.load_waterways(corrupt) is None
    wrong = tmp_path / "wrong.geojson"
    wrong.write_text(json.dumps({"type": "Feature"}), encoding="utf-8")
    assert ml.load_waterways(wrong) is None
    good = tmp_path / "good.geojson"
    good.write_text(json.dumps(WATERWAYS), encoding="utf-8")
    assert ml.load_waterways(good).n_lines == 3


@pytest.mark.unit
def test_synthetic_drain_line():
    data = ml.synthetic_drain_line(77.675, BBOX)
    assert data.source == "synthetic_line" and data.n_lines == 1
    assert data.features[0].coords == ((77.675, BBOX[1]), (77.675, BBOX[3]))
    assert "outside" in ml.synthetic_drain_line(78.5, BBOX).note
    with pytest.raises(ValueError):
        ml.synthetic_drain_line("east", BBOX)
    with pytest.raises(ValueError):
        ml.synthetic_drain_line(77.6, [1, 2, 3])


# --------------------------------------------------------------------------- map_layers: camera & deck


@pytest.mark.unit
def test_zoom_and_view_state():
    assert ml.zoom_for_extent(0.0, 0.0, 12.9) == 15.0
    city = ml.zoom_for_extent(0.045, 0.035, 12.93)
    assert 12.0 < city < 14.5
    assert ml.zoom_for_extent(0.045, 0.035, 12.93) > ml.zoom_for_extent(0.45, 0.35, 12.93)
    assert ml.zoom_for_extent(300.0, 150.0, 0.0) == 3.0 and ml.zoom_for_extent(1e-7, 0.0, 0.0) == 17.0
    view = ml.view_state([77.655, 77.70, np.nan], [12.915, 12.95, 12.0], pitch=30, bearing=-10)
    assert math.isclose(view.longitude, 77.6775) and math.isclose(view.latitude, 12.9325)
    assert view.pitch == 30 and view.bearing == -10
    single = ml.view_state([77.6], [12.9])
    assert single.zoom == 15.0
    with pytest.raises(ValueError, match="finite"):
        ml.view_state([np.nan], [np.nan])


@pytest.mark.unit
def test_map_options_validation():
    assert ml.MapOptions().pitch == 45.0 and ml.MapOptions(three_d=False).pitch == 0.0
    for kwargs in ({"map_style": "satellite"}, {"column_radius_m": 0}, {"elevation_scale": -1},
                   {"pitch_deg": 90}, {"bearing_deg": "north"}):
        with pytest.raises(ValueError):
            ml.MapOptions(**kwargs)


def deck_inputs():
    scale = ml.ColorScale("tiers", TIERS)
    table = node_table()
    junctions = ml.junction_frame(table, [0.9, 0.3, 0.0], scale)
    roads = ml.road_frame(table["lon"], table["lat"], np.array([[0, 1]]), [0.9, 0.3, 0.0], scale)
    return junctions, roads, ml.parse_waterways(WATERWAYS)


@pytest.mark.unit
def test_build_deck_layers_provider_and_selection():
    junctions, roads, water = deck_inputs()
    deck = ml.build_deck(junctions, ml.MapOptions(map_style="dark"), roads=roads, water=water, selected_node=102)
    ids = [layer.id for layer in deck.layers]
    assert ids == [ml.WATER_BODY_LAYER_ID, ml.WATER_LINE_LAYER_ID, ml.ROAD_LAYER_ID, ml.JUNCTION_LAYER_ID,
                   ml.SELECTED_LAYER_ID]
    assert deck.map_provider == "carto" and deck.layers[3].type == "ColumnLayer"
    assert deck.initial_view_state.pitch == 45.0
    text = deck.to_json()
    spec = json.loads(text)
    assert "dark-matter" in spec["mapStyle"] and deck._tooltip == ml.TOOLTIP and "html" not in deck._tooltip
    assert isinstance(deck, pdk.Deck) and "\n" not in text and ", " not in text[:200]
    assert set(spec["layers"][3]["data"][0]) == set(ml.JUNCTION_LAYER_COLUMNS)   # only what the browser needs
    assert set(spec["layers"][2]["data"][0]) == set(ml.ROAD_LAYER_COLUMNS)
    flat = ml.build_deck(junctions, ml.MapOptions(three_d=False, show_roads=False, show_drains=False), roads=roads,
                         water=water, selected_node="unknown")
    assert [layer.id for layer in flat.layers] == [ml.JUNCTION_LAYER_ID]
    assert flat.layers[0].type == "ScatterplotLayer" and flat.initial_view_state.pitch == 0.0
    view = pdk.ViewState(latitude=1, longitude=2, zoom=3)
    assert ml.build_deck(junctions, ml.MapOptions(), water=ml.WaterLayerData((), "osm"),
                         view=view).initial_view_state is view


@pytest.mark.unit
def test_selected_node_id_from_event_shapes():
    event = {"selection": {"indices": {"junctions": [0]}, "objects": {"junctions": [{"node_id": "42"}]}}}
    assert ml.selected_node_id(event) == "42"
    attribute_style = SimpleNamespace(selection=SimpleNamespace(objects={"junctions": [{"node_id": 7}]}))
    assert ml.selected_node_id(attribute_style) == "7"
    for empty in (None, {}, {"selection": None}, {"selection": {"objects": None}},
                  {"selection": {"objects": {"roads": [{"node_id": 1}]}}},
                  {"selection": {"objects": {"junctions": []}}},
                  {"selection": {"objects": {"junctions": ["x"]}}}, SimpleNamespace(selection=None)):
        assert ml.selected_node_id(empty) is None


# --------------------------------------------------------------------------- components: settings & formatting


@pytest.mark.unit
def test_app_settings_defaults_and_config(cfg):
    settings = ui.AppSettings.from_config(cfg)
    assert settings.map_style in ml.MAP_STYLES and settings.top_k == cfg["app"]["top_k"]
    assert ui.AppSettings.from_config({}).map_height_px == ui.DEFAULTS["map_height_px"]
    custom = ui.AppSettings.from_config({"app": {"map_style": "DARK", "default_mode": "design", "top_k": 5.0}})
    assert custom.map_style == "dark" and custom.default_mode == "design" and custom.top_k == 5


@pytest.mark.unit
@pytest.mark.parametrize("override, message", [
    ({"map_style": "neon"}, "app.map_style"), ({"default_mode": "live"}, "app.default_mode"),
    ({"color_scale": "rainbow"}, "app.color_scale"), ({"top_k": 0}, "app.top_k"), ({"top_k": 2.5}, "integer"),
    ({"top_k": True}, "app.top_k"), ({"elevation_scale": "tall"}, "app.elevation_scale"),
    ({"pitch_deg": 90}, "app.pitch_deg"), ({"forecast_ttl_s": 10}, "app.forecast_ttl_s"),
    ({"column_radius_m": float("nan")}, "app.column_radius_m"),
])
def test_app_settings_validation(override, message):
    with pytest.raises(ConfigError, match=message):
        ui.AppSettings.from_config({"app": override})


@pytest.mark.unit
def test_value_time_and_hour_formatting():
    assert ui.fmt_value(0.4213) == "0.421" and ui.fmt_value(7.1e-4) == "7.10e-04" and ui.fmt_value(0) == "0"
    assert ui.fmt_value(None) == "n/a" and ui.fmt_value(float("inf")) == "n/a" and ui.fmt_value("x") == "n/a"
    ts = pd.Timestamp("2022-09-05 17:00", tz=TZ)
    assert ui.fmt_time(ts) == "Mon 05 Sep, 17:00" and ui.fmt_time(pd.NaT) == "" and ui.fmt_time(None) == ""
    assert ui.hour_label(ts, 0) == "Mon 05 Sep 17:00 · lead 1 h"


@pytest.mark.unit
def test_fmt_number():
    assert ui.fmt_number(871.26, "{:.1f} m") == "871.3 m" and ui.fmt_number(-1.5, "{:+.2f}") == "-1.50"
    assert ui.fmt_number(float("nan"), "{:.1f}") == "n/a" and ui.fmt_number(None, "{:.1f}") == "n/a"
    assert ui.fmt_number("x", "{:.1f}") == "n/a"


@pytest.mark.unit
def test_stale_model_note_prefers_the_current_config_hash():
    trained_on = {"dataset": {"dataset_config_hash": "aaa", "stale": False}}
    assert "aaa vs current bbb" in mv.stale_model_note(trained_on, "bbb")
    assert mv.stale_model_note(trained_on, "aaa") is None
    assert mv.stale_model_note(trained_on, None) is None                  # falls back to the recorded flag
    assert "retrain" in mv.stale_model_note({"dataset": {"stale": True}}, "zzz")
    assert mv.stale_model_note({"dataset": "broken"}, "x") is None
    assert mv.stale_model_note(trained_on, "aaa", trained_hash="ccc").startswith("The model was trained")  # R4-02
    assert mv.stale_model_note({"dataset": {"stale": True}}, "ccc", trained_hash="ccc") is None


@pytest.mark.unit
@pytest.mark.parametrize("model", [3.6e-5, 0.4, 0.5, float("nan"), 1.0, 0.0])
def test_threshold_options(model):
    options = ui.threshold_options(model)
    assert options == sorted(options) and len(options) == len(set(options))
    assert options[0] == pytest.approx(1e-6) and options[-1] == 0.95
    expected = model if 0 < model < 1 else 0.5
    assert expected in options


@pytest.mark.unit
def test_threshold_label():
    assert ui.threshold_label(0.4, 0.4) == "40% (model)"
    assert ui.threshold_label(0.5, 0.4) == "50%" and ui.threshold_label(3.6e-5, 3.6e-5) == "3.6e-05 (model)"


# --------------------------------------------------------------------------- components: KPIs & frames


@pytest.mark.unit
def test_build_kpis_peak_and_hour_modes():
    prob = np.array([[0.1, 0.5, 0.0], [0.2, 0.9, 0.45], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    result = make_result(prob, areal=np.array([1.0, 4.0, 0.5, 0.0]))
    peak = ui.build_kpis(result, hours=4, peak_mode=True, hour_index=0, metrics=METRICS, predictor_kind="gnn")
    by_label = {k.label: k for k in peak}
    assert by_label["Junctions at risk"].value == "2 / 3"
    assert by_label["Max flood probability"].value == "90%" and by_label["Max flood probability"].delta == "Severe"
    assert by_label["Peak hour"].value == "Mon 13:00" and by_label["Peak hour"].delta == "lead 2 h"
    assert by_label["Rain over horizon"].value == "5.5 mm" and "4.0 mm/h" in by_label["Rain over horizon"].delta
    skill = by_label["PR-AUC given junction rain · test 2024"]
    assert skill.value == "0.42" and skill.delta == "-0.13 vs GBDT" and skill.delta_color == "normal"
    legacy = ui.build_kpis(result, hours=4, peak_mode=True, hour_index=0, metrics=LEGACY_METRICS,
                           predictor_kind="gnn", dataset_cfg={"val_years": [2021]})
    assert legacy[4].label == "PR-AUC given junction rain · val" and legacy[4].delta == "+0.22 vs logistic"  # no years
    hour = ui.build_kpis(result, hours=2, peak_mode=False, hour_index=5, metrics=None, predictor_kind="gnn")
    assert hour[0].label == "At risk at 13:00" and hour[0].value == "2 / 3"
    assert hour[4].value == "n/a"
    physics = ui.build_kpis(result, hours=99, peak_mode=True, hour_index=0, metrics=METRICS, predictor_kind="physics")
    assert physics[4].value == "teacher" and "next 4 h" in physics[0].help
    no_base = ui.build_kpis(result, hours=4, peak_mode=True, hour_index=0, metrics={"pr_auc": 0.3},
                            predictor_kind="gnn")
    assert no_base[4].delta is None


@pytest.mark.unit
def test_build_kpis_dry_run_has_no_alerts():
    result = make_result(np.zeros((3, 2), dtype=np.float32))
    values = {k.label: k.value for k in ui.build_kpis(result, hours=3, peak_mode=True, hour_index=0, metrics=None,
                                                      predictor_kind="gnn")}
    assert values["Junctions at risk"] == "0 / 2" and values["Max flood probability"] == "0%"


@pytest.mark.unit
def test_resolve_junction_state_machine():
    order = [30, 10, 20]
    state: dict = {}
    keys = {"map_key": "map", "table_key": "table", "junction_key": "junction"}
    assert ui.resolve_junction(state, order, **keys) == 30                  # default: the riskiest
    state["junction"] = 20
    assert ui.resolve_junction(state, order, **keys) == 20                  # the previous choice is kept
    state["map"] = {"selection": {"objects": {"junctions": [{"node_id": "10"}]}}}
    assert ui.resolve_junction(state, order, **keys) == 10                  # a new map click wins
    state["junction"] = 30
    assert ui.resolve_junction(state, order, **keys) == 30                  # ... but is not re-applied
    state[ui.TOPK_IDS_KEY] = [20, 10]
    state["table"] = {"selection": {"rows": [0]}}
    assert ui.resolve_junction(state, order, **keys) == 20                  # a new table row wins
    state["junction"] = 10
    assert ui.resolve_junction(state, order, **keys) == 10
    state["table"] = {"selection": {"rows": [9]}}                          # out-of-range row is ignored
    assert ui.resolve_junction(state, order, **keys) == 10
    state["junction"] = 99                                                  # unknown junction (graph changed)
    assert ui.resolve_junction(state, order, **keys) == 30
    state["table"] = SimpleNamespace(selection=SimpleNamespace(rows=["bad"]))
    assert ui.resolve_junction(state, order, **keys) == 30
    with pytest.raises(ValueError):
        ui.resolve_junction({}, [], **keys)


@pytest.mark.unit
def test_rain_frame_categories_per_scenario_kind():
    index = pd.date_range("2024-09-01", periods=6, freq="h", tz=TZ)
    forecast = sc.Scenario("f", "forecast", index, [0, 1, 2, 3, 0, 0], [False, False, True, True, True, True], 2,
                           "", "forecast")
    frame = ui.rain_frame(forecast)
    assert frame["category"].tolist()[:3] == ["Recent (model analysis)", "Recent (model analysis)", "Forecast"]
    assert frame["time"].dt.tz is None and frame["time"][0] == pd.Timestamp("2024-09-01 00:00")
    assert frame["is_target"].tolist() == [False, False, True, True, True, True]
    assert ui.rain_categories("historical", "synthetic", np.zeros(2), np.zeros(2)) == ["Synthetic record"] * 2
    assert ui.rain_categories("historical", "open_meteo", np.zeros(1), np.zeros(1)) == ["Observed record"]
    assert ui.rain_categories("design_storm", "design_storm", np.zeros(2), np.array([False, True])) == [
        "Dry lead-in", "Design storm"]
    assert ui.rain_categories("custom", "custom", np.zeros(1), np.zeros(1)) == ["Rain"]


@pytest.mark.unit
def test_junction_timeline_tier_counts_and_horizon_table():
    prob = np.array([[0.1, 0.6], [0.3, 0.2], [0.0, 0.9], [0.2, 0.1]], dtype=np.float32)
    std = np.full_like(prob, 0.05)
    result = make_result(prob, std=std)
    line = ui.junction_timeline(result, 1, 3)
    assert len(line) == 3 and line["prob"].tolist() == pytest.approx([0.6, 0.2, 0.9])
    assert line["hi"].tolist() == pytest.approx([0.65, 0.25, 0.95]) and line["depth_m"].tolist() == pytest.approx(
        [0.1] * 3)
    no_std = ui.junction_timeline(make_result(prob), 0, 10)
    assert len(no_std) == 4 and (no_std["lo"] == no_std["hi"]).all()
    with pytest.raises(IndexError):
        ui.junction_timeline(result, 5, 2)
    counts = ui.tier_counts([0.1, 0.3, 0.9, 0.95], TIERS)
    assert counts["junctions"].tolist() == [1, 1, 0, 2] and counts["color"][0].startswith("#")
    table = ui.horizon_table(result)
    assert table["Horizon"].tolist() == ["2 h", "4 h"] and table["At risk"].tolist() == [1, 1]
    short = make_result(prob[:1])
    assert ui.horizon_table(short).empty


@pytest.mark.unit
def test_charts_serialise_to_vega_lite():
    prob = np.array([[0.1, 0.6], [0.3, 0.2], [0.0, 0.9]], dtype=np.float32)
    result = make_result(prob, std=np.full_like(prob, 0.05))
    index = pd.date_range("2022-09-05 09:00", periods=6, freq="h", tz=TZ)
    scenario = sc.Scenario("s", "historical", index, [0, 1, 5, 2, 0, 0], None, 3, "", "open_meteo")
    charts = [
        ui.hyetograph_chart(ui.rain_frame(scenario), marker=index[4], target_span=(index[3], index[5])),
        ui.hyetograph_chart(ui.rain_frame(scenario), marker=None, target_span=None),
        ui.timeline_chart(ui.junction_timeline(result, 1, 3), threshold=0.4, marker=result.timestamps[1]),
        ui.timeline_chart(ui.junction_timeline(make_result(prob * 1e-4), 1, 3), threshold=1e-5, marker=pd.NaT),
        ui.tier_chart(ui.tier_counts([0.1, 0.9], TIERS)),
        ui.reliability_chart(mv.reliability_table(METRICS)),
        ui.history_chart(pd.DataFrame({"epoch": [1, 2], "pr_auc": [0.1, 0.2], "roc_auc": [0.5, None]})),
    ]
    for chart in charts:
        spec = chart.to_dict()
        assert spec.get("layer") or spec.get("mark")
    assert '"format": ".1e"' in json.dumps(charts[3].to_dict())


# --------------------------------------------------------------------------- components: model & provenance


@pytest.mark.unit
def test_read_json_and_csv_files(tmp_path):
    assert ui.read_json_file(tmp_path / "none.json") is None and ui.read_csv_file(tmp_path / "none.csv") is None
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "ok.json").write_text('{"a": 1}', encoding="utf-8")
    assert ui.read_json_file(tmp_path / "bad.json") is None and ui.read_json_file(tmp_path / "list.json") is None
    assert ui.read_json_file(tmp_path / "ok.json") == {"a": 1}
    (tmp_path / "ok.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    assert ui.read_csv_file(tmp_path / "ok.csv").shape == (1, 2) and ui.read_csv_file(tmp_path / "empty.csv") is None


@pytest.mark.unit
def test_comparison_reliability_and_backtest_tables():
    table = mv.comparison_table(METRICS)
    assert list(table.columns) == ["Model", "Split", *mv.METRIC_COLUMNS]
    assert list(zip(table["Model"], table["Split"])) == [
        ("Namma-Flow GNN", "test 2024"), ("Gradient-boosted trees (no graph)", "test 2024"),
        ("Logistic regression (no graph)", "test 2024"), ("Namma-Flow GNN", "val 2022"),
        ("Gradient-boosted trees (no graph)", "val 2022"), ("Logistic regression (no graph)", "val 2022")]
    assert table["PR-AUC"].tolist() == ["0.42", "0.55", "0.2", "0.47", "0.6", "0.25"]
    assert table.loc[2, "Recall"] == "n/a"
    assert mv.comparison_table(None).empty
    legacy = mv.comparison_table(LEGACY_METRICS)
    assert list(legacy["Model"]) == ["Namma-Flow GNN", "Logistic regression (no graph)"]
    assert set(legacy["Split"]) == {"val"}                                       # no years recorded
    assert list(mv.comparison_table({"pr_auc": 0.1, "baseline_logreg": {"status": "failed"}})["Model"]) == [
        "Namma-Flow GNN"]
    reliability = mv.reliability_table(METRICS)
    assert reliability["count"].tolist() == [100, 5]
    assert mv.reliability_table({"reliability": {"bin_centers": 3}}).empty and mv.reliability_table(None).empty
    backtest = mv.backtest_table(BACKTEST)
    assert list(backtest.index) == ["0 h ahead", "24 h ahead"]
    assert backtest.loc["0 h ahead", "Physics PR-AUC"] == "teacher (= labels)"      # R4-05 (older report: no role)
    assert backtest.loc["24 h ahead", "Physics PR-AUC"] == "0.05" and backtest.loc["24 h ahead", "ROC-AUC"] == "n/a"
    assert mv.backtest_table({"status": "unavailable"}).empty and mv.backtest_table(None).empty


@pytest.mark.unit
def test_headline_split_wording_and_strongest_baseline():
    """R3-05 / R5-08 / R2-03: explicit test/validation wording and the strongest graph-free baseline."""
    assert mv.evaluation_split(METRICS) == "test" and mv.evaluation_split(LEGACY_METRICS) == "validation"
    assert mv.evaluation_split({"evaluation_split": "bogus"}) == "validation"
    assert mv.headline_wording(METRICS) == ("held-out test year 2024; model selection and calibration used "
                                            "validation year 2022")
    assert "optimistic" in mv.headline_wording(LEGACY_METRICS, {"val_years": [2022, 2024]})
    assert "validation years," in mv.headline_wording(LEGACY_METRICS, {"val_years": [2022, 2024]})  # not guessed
    no_years = {"evaluation_split": "test", "pr_auc": 0.3}
    assert mv.split_years(no_years, "test", {"test_years": 2024}) == [2024]            # v2 without years: config
    assert mv.split_years({}, "test", {"test_years": 2024}) == [] and mv.split_years(None, "test") == []
    assert mv.split_years({"dataset": {"val_years": ["2021", "x", 2021]}}, "validation") == [2021]
    assert "validation years 2022, 2024" in mv.headline_wording(
        {"dataset": {"years": {"validation": [2024, 2022]}}})
    assert mv.baseline_metrics(METRICS, "lookup", "test") is None                      # failed baseline
    assert mv.split_label("test", []) == "test years" and mv.split_short(None, "validation") == "val"
    assert mv.strongest_baseline(METRICS) == ("hist_gbdt", 0.55)
    derived = {**METRICS, "strongest_baseline": None}
    assert mv.strongest_baseline(derived) == ("hist_gbdt", 0.55)                # derived from baselines[*][test]
    assert mv.strongest_baseline(LEGACY_METRICS) == ("logreg", 0.2) and mv.strongest_baseline({}) is None
    custom = {"evaluation_split": "test", "baselines": {"lookup": {"status": "ok", "test": {"pr_auc": 0.9}}}}
    assert mv.baseline_names(custom)[-1] == "lookup" and mv.strongest_baseline(custom) == ("lookup", 0.9)
    assert mv.baseline_label("lookup") == "lookup (no graph)"
    assert mv.split_metrics(METRICS, "validation")["pr_auc"] == 0.47
    assert mv.split_metrics(LEGACY_METRICS, "test") is None
    kpi = mv.pr_auc_kpi(METRICS, "gnn")
    assert "held-out test year 2024" in kpi.help and "junction rain field" in kpi.help and "0.55" in kpi.help
    assert mv.pr_auc_kpi({"pr_auc": float("nan")}, "gnn").value == "n/a"


@pytest.mark.unit
def test_calibration_wording_and_model_card():
    describe = {"threshold": 0.4, "calibration": PLATT, "temperature": 1.25, "seq_len": 24, "warmup_steps": 6,
                "checkpoint": "artifacts/checkpoints/best.pt", "format_version": 2,
                "finalized_utc": "2024-01-01T00:05:00+00:00"}
    assert mv.calibration_label(METRICS, describe) == "Platt: σ(0.8 · logit − 0.5)"
    assert "Platt-calibrated on the validation year 2022" in mv.calibration_caption(METRICS, describe)
    temperature = {"calibration": {"method": "temperature", "slope": 0.5, "intercept": 0.0}}
    assert mv.calibration_label(None, temperature) == "Temperature: σ(logit / 2)"
    assert "temperature-calibrated" in mv.calibration_caption(LEGACY_METRICS, None)   # v1 metrics: T = 1.5
    identity = {"calibration": {"method": "none", "slope": 1.0, "intercept": 0.0}}
    assert mv.calibration_label(None, identity) == "None (σ(logit))"
    assert "not calibrated" in mv.calibration_caption(None, {"temperature": 1.0})
    card = dict(mv.model_card(METRICS, describe))
    assert card["Parameters"] == "1,234" and card["Alert threshold"].startswith("0.4")
    assert card["Calibration"].startswith("Platt") and card["ECE (test 2024)"] == "0.02"
    assert card["Checkpoint"] == "artifacts/checkpoints/best.pt (format v2)"
    assert card["Trained (UTC)"].startswith("2024-01-01T00:05")
    assert dict(mv.model_card(None, None))["Architecture"] == "n/a"


@pytest.mark.unit
def test_notes_model_card_and_provenance():
    assert "retrain" in mv.stale_model_note(METRICS) and mv.stale_model_note({"dataset": {"stale": False}}) is None
    assert mv.stale_model_note(None) is None
    note = mv.backtest_note(BACKTEST, {"created_utc": "new"})
    assert "epoch 1" in note and "different checkpoint" in note
    assert "different" not in mv.backtest_note(BACKTEST, None) and mv.backtest_note({}, None) is None
    index = pd.date_range("2024-09-01", periods=3, freq="h", tz=TZ)
    scenario = sc.Scenario("s", "design_storm", index, [0, 1, 0], None, 1, "desc", "design_storm")
    graph = {"source": "osm_bbox", "elevation_source": "srtm", "drain_source": "synthetic_line", "nodes": 10,
             "edges": 30}
    meta = {"sources": {"open_meteo": 5}, "start": "2018-01-01T00:00", "end": "2024-12-31T23:00", "rows": 5}
    rows = dict(ui.provenance_rows(graph, meta, scenario, "simulated", 0))
    assert "bounding-box" in rows["Road graph"] and "10 junctions" in rows["Road graph"]
    assert "SRTM" in rows["Elevation"] and "Synthetic" in rows["Drains"]
    assert "2018-01-01 → 2024-12-31" in rows["Weather record"] and "design storm" in rows["This scenario"].lower()
    assert "none" in rows["Flood labels"]
    sparse = dict(ui.provenance_rows({}, {"sources": {}}, scenario, None, 3))
    assert "unknown" in sparse["Weather record"] and "3 observed" in sparse["Flood labels"]
    assert dict(ui.provenance_rows({}, None, scenario, None, None))["Weather record"] == "n/a"


@pytest.mark.unit
def test_tooltip_and_ring_edge_cases():
    table = node_table(1).assign(peak_time=["not a time"])
    tooltip = ml.junction_frame(table, [0.5], ml.ColorScale("tiers"))["tooltip"][0]
    assert "Peak:" not in tooltip
    weird = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": "abc"}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [7]}},
        {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [3]}}]}
    assert ml.parse_waterways(weird).features == ()


@pytest.mark.unit
def test_hyetograph_shading_covers_exactly_the_target_bars():
    """Regression R4-08: bars of hour-ending t span [t, t + 1 h); the shading must be [t0, t_end + 1 h)."""
    index = pd.date_range("2022-09-05 00:00", periods=12, freq="h", tz=TZ)
    start, end = ui.target_band((index[6], index[11]))
    assert start == pd.Timestamp("2022-09-05 06:00") and end == pd.Timestamp("2022-09-05 12:00")
    scenario = sc.Scenario("s", "historical", index, [1.0] * 6 + [5.0] * 6, None, 6, "", "open_meteo")
    spec = ui.hyetograph_chart(ui.rain_frame(scenario), marker=index[8], target_span=(index[6], index[11])).to_dict()
    datasets = spec["datasets"].values()
    span = next(rows for rows in datasets if rows and "start" in rows[0])[0]
    assert span["start"].startswith("2022-09-05T06:00") and span["end"].startswith("2022-09-05T12:00")
    rule = next(rows for rows in spec["datasets"].values() if rows and set(rows[0]) == {"time"})[0]
    assert rule["time"].startswith("2022-09-05T08:30")                              # centre of the 08:00 bar


@pytest.mark.unit
def test_public_text_and_history_notices():
    """R5-09 / R4-07: messages lose machine paths; dry-padded history becomes a visible warning."""
    root, home = str(project_root()), str(Path.home())
    text = ui.public_text(f"No trained model at {root}/artifacts/checkpoints/best.pt; see {home}/notes and {root}")
    assert text == "No trained model at artifacts/checkpoints/best.pt; see ~/notes and ."
    assert ui.show_path(Path(root) / "config" / "config.yaml") == "config/config.yaml"
    assert ui.history_notices({"padded_history_h": 0}) == [] and ui.history_notices({}) == []
    notice = ui.history_notices({"padded_history_h": 18, "history_hours": 12, "history_needed_h": 30})[0]
    assert notice.level == "warning" and "needs 30 h" in notice.message and "missing 18 h" in notice.message


def test_waterways_layer_excludes_treatment_tanks_like_the_model():
    from app.map_layers import parse_waterways

    ring = [[77.66, 12.92], [77.661, 12.92], [77.661, 12.921], [77.66, 12.92]]
    collection = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"natural": "water", "water": "lake", "name": "Bellandur"},
         "geometry": {"type": "Polygon", "coordinates": [ring]}},
        {"type": "Feature", "properties": {"natural": "water", "water": "wastewater"},
         "geometry": {"type": "Polygon", "coordinates": [ring]}},
    ]}
    layer = parse_waterways(collection, exclude_water=("wastewater",))
    assert len(layer.features) == 1
    assert "1 treatment tanks / pools excluded" in layer.note
    assert len(parse_waterways(collection).features) == 2
