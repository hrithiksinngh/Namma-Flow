"""End-to-end tests of the Streamlit dashboard (``app/app.py``) with ``streamlit.testing.v1.AppTest``.

The app runs headless against a tiny synthetic world written into ``tmp_path`` (4 x 4 street
grid, a September-2022 weather record, optionally an untrained GATv2-GRU checkpoint in the
finalized format v2 with a Platt calibration and the graph attribute digest, a metrics.json
with held-out TEST and VALIDATION splits and graph-free baselines, history and backtest),
selected through ``$NAMMA_FLOW_CONFIG``. Everything runs offline. Unit tests of the pure
helpers (``map_layers``, ``components``, ``metrics_view``) are in ``tests/test_app_components.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
import torch
import yaml
from streamlit.testing.v1 import AppTest

from src.data_pipeline import weather
from src.data_pipeline.dataset import dataset_config_hash
from src.data_pipeline.features import FeatureScaler, dynamic_feature_names
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.inference import scenarios as sc
from src.inference.predictor import PhysicsPredictor, PredictionError
from src.inference.results import PredictionResult
from src.models.stgcn import build_model
from src.utils.config import deep_merge, project_root, resolve_path
from src.utils.runtime import atomic_torch_save
from tests.conftest import make_grid_graph

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
APP_FILE = APP_DIR / "app.py"
sys.path.insert(0, str(APP_DIR))
import components as ui  # noqa: E402
import map_layers as ml  # noqa: E402

sys.path.remove(str(APP_DIR))

TZ = "Asia/Kolkata"
STATIC = ["elevation", "dist_to_drain_m", "relative_elevation"]
EDGES = ["length", "grade"]
WINDOWS = [3, 6, 12, 24]
TIERS = (("Low", 0.0), ("Moderate", 0.25), ("High", 0.5), ("Severe", 0.75))
PLATT = {"method": "platt", "slope": 0.8, "intercept": -0.5}


# --------------------------------------------------------------------------- synthetic world


def storm_record() -> pd.DataFrame:
    """Schema-2.2 hourly record 2022-09-01 .. 2022-09-20 with two distinct storms."""
    index = pd.date_range("2022-09-01 00:00", "2022-09-20 23:00", freq="h", tz=TZ, name="timestamp")
    rain = pd.Series(0.0, index=index)
    rain.loc["2022-09-05 14:00":"2022-09-05 20:00"] = [2, 10, 25, 30, 12, 4, 1]
    rain.loc["2022-09-14 03:00":"2022-09-14 05:00"] = [5, 8, 3]
    return pd.DataFrame({"precipitation_mm": rain.to_numpy(), "is_imputed": False, "source": "open_meteo"},
                        index=index)


def make_checkpoint(cfg: dict, graph, **overrides) -> dict:
    """A finalized, untrained format-v2 checkpoint with every key of the training contract (X3)."""
    torch.manual_seed(0)
    model = build_model({"architecture": "gatv2_gru", "node_in_dim": 8, "edge_dim": 2, "hidden_dim": 8,
                         "heads": 1, "dropout": 0.3})
    dynamic = dynamic_feature_names(WINDOWS)
    samples = np.random.default_rng(0).gamma(0.5, 3.0, size=(400, len(dynamic)))
    scaler = FeatureScaler.fit(graph.node_matrix(STATIC), samples, graph.edge_attrs["length"], 0.3,
                               static_names=STATIC, dynamic_names=dynamic)
    checkpoint = {
        "format_version": 2, "model_state": model.state_dict(), "architecture": "gatv2_gru",
        "model_config": model.get_config(), "feature_names": [*STATIC, *dynamic], "static_feature_names": STATIC,
        "rolling_windows_h": WINDOWS, "lookback_hours": 24, "seq_len": 24, "warmup_steps": 6,
        "scaler": scaler.to_dict(), "graph_signature": graph.signature(), "node_ids": list(graph.node_ids),
        "graph_attributes_sha256": graph.attributes_signature(STATIC, EDGES), "calibration": dict(PLATT),
        "temperature": 1.25, "threshold": 0.4, "metrics": {"pr_auc": 0.42, "roc_auc": 0.9, "created_utc": "x"},
        "epoch": 3, "best_score": 0.42, "history": [], "optimizer_state": {}, "scheduler_state": {},
        "rng_state": {}, "config_hash": "cfg", "dataset_config_hash": dataset_config_hash(cfg),
        "created_utc": "2024-01-01T00:00:00+00:00", "finalized_utc": "2024-01-01T00:05:00+00:00",
    }
    checkpoint.update(overrides)
    return checkpoint


SPLIT_METRICS = {"pr_auc": 0.42, "roc_auc": 0.9, "f2": 0.5, "precision": 0.3, "recall": 0.7, "csi": 0.25,
                 "brier": 0.01, "ece": 0.02, "threshold": 0.4, "pos_rate": 0.01}
METRICS = {  # metrics.json format v2: headline = held-out TEST split
    "status": "completed", "evaluation_split": "test", **SPLIT_METRICS,
    "validation": {**SPLIT_METRICS, "pr_auc": 0.47, "f2": 0.55}, "test": dict(SPLIT_METRICS),
    "calibration": dict(PLATT), "temperature": 1.25, "threshold_metric": "f2", "best_epoch": 3, "epochs_run": 5,
    "created_utc": "x",
    "reliability": {"bin_centers": [0.05, 0.15, 0.25], "mean_predicted": [0.01, None, 0.3],
                    "observed_freq": [0.02, None, 0.25], "counts": [100, 0, 5]},
    "baselines": {"logreg": {"status": "ok", "validation": {"pr_auc": 0.25, "roc_auc": 0.82},
                             "test": {"pr_auc": 0.2, "roc_auc": 0.8}},
                  "hist_gbdt": {"status": "ok", "validation": {"pr_auc": 0.6, "roc_auc": 0.97},
                                "test": {"pr_auc": 0.55, "roc_auc": 0.95, "f2": 0.6}},
                  "lookup": {"status": "failed", "reason": "MemoryError"}},
    "baseline_logreg": {"status": "ok", "pr_auc": 0.2, "roc_auc": 0.8, "f2": 0.3, "threshold": 0.6},
    "strongest_baseline": {"name": "hist_gbdt", "pr_auc": 0.55},
    "dataset": {"label_source": "simulated", "stale": True, "dataset_config_hash": "aaa",
                "current_dataset_config_hash": "bbb",
                "years": {"train": [2018, 2019, 2020, 2021, 2023], "validation": [2022], "test": [2024]}},
    "model": {"architecture": "gatv2_gru", "n_parameters": 1234}, "training": {"mean_epoch_time_s": 12.0},
}
LEGACY_METRICS = {  # format v1: validation metrics only, one logistic baseline
    "status": "completed", **SPLIT_METRICS, "temperature": 1.5, "best_epoch": 3, "epochs_run": 5,
    "threshold_metric": "f2", "created_utc": "x",
    "baseline_logreg": {"status": "ok", "pr_auc": 0.2, "roc_auc": 0.8, "f2": 0.3, "threshold": 0.6},
    "dataset": {"label_source": "simulated", "stale": True, "dataset_config_hash": "aaa",
                "current_dataset_config_hash": "bbb"},
}
BACKTEST = {"status": "ok", "start": "2024-10-01T00:00:00+05:30", "end": "2024-10-31T23:00:00+05:30",
            "predictor": {"label": "Namma-Flow GNN", "epoch": 1, "threshold": 0.3, "created_utc": "old"},
            "leads": {"lead_0h": {"lead_hours": 0, "rain": {"correlation": 1.0},
                                  "junction_hours": {"pr_auc": 0.11, "roc_auc": 0.88, "recall": 0.0, "precision": None},
                                  "physics_baseline": {"junction_hours": {"pr_auc": 1.0}}},
                      "lead_24h": {"lead_hours": 24, "rain": {"correlation": 0.2},
                                   "junction_hours": {"pr_auc": 0.03},
                                   "physics_baseline": {"role": "forecast_baseline",
                                                        "junction_hours": {"pr_auc": 0.05}}}, "broken": "x"}}


WATERWAYS = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"waterway": "drain", "name": "ORR drain"},
     "geometry": {"type": "LineString", "coordinates": [[77.66, 12.92], [77.67, 12.93]]}},
    {"type": "Feature", "properties": {"natural": "water", "water": "lake", "name": "Bellandur Lake"},
     "geometry": {"type": "Polygon", "coordinates": [[[77.66, 12.92], [77.67, 12.92], [77.67, 12.93], [77.66, 12.92]],
                                                     [[77.665, 12.922], [77.666, 12.922], [77.666, 12.923]]]}},
    {"type": "Feature", "properties": {"waterway": "stream"},
     "geometry": {"type": "MultiLineString", "coordinates": [[[77.6, 12.9], [77.61, 12.91]], [[77.62, 12.9]]]}},
    {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [
        [[[77.7, 12.9], [77.71, 12.9], [77.71, 12.91]]]]}},
    {"type": "Feature", "properties": {}, "geometry": {"type": "GeometryCollection", "geometries": [
        {"type": "LineString", "coordinates": [[77.6, 12.9], [77.6, 13.0]]}]}},
    {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [77.6, 12.9]}},
    {"type": "Feature", "properties": {}, "geometry": {"type": "LineString",
                                                      "coordinates": [[999, 12.9], ["x", 1], [77.6]]}},
    {"type": "Feature", "properties": {}, "geometry": None},
    "not-a-feature",
]}


def write_config(cfg: dict, path: Path) -> Path:
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    path.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.delenv("NAMMA_FLOW_DEBUG", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def env(cfg, tmp_path, monkeypatch) -> SimpleNamespace:
    """Config file + 4 x 4 graph + weather record in tmp_path (no trained model)."""
    cfg = deep_merge(cfg, {"inference": {"batch_windows": 4, "mc_dropout_samples": 3},
                           "app": {"notable_events": 5, "top_k": 5}})
    save_graph(make_grid_graph(4, 4), cfg["paths"]["graph_file"])
    weather.save_weather_csv(storm_record(), cfg["paths"]["weather_file"])
    path = write_config(cfg, tmp_path / "config.yaml")
    monkeypatch.setenv("NAMMA_FLOW_CONFIG", str(path))
    return SimpleNamespace(cfg=cfg, path=path, tmp=tmp_path)


def install_model(env: SimpleNamespace, metrics: dict | None = None, **overrides) -> SimpleNamespace:
    """Write best.pt (+ metrics.json, backtest.json, training_history.csv) into ``env``."""
    graph = graph_to_arrays(load_graph(env.cfg["paths"]["graph_file"]))
    atomic_torch_save(make_checkpoint(env.cfg, graph, **overrides), resolve_path(env.cfg, "checkpoint_dir") / "best.pt")
    reports = resolve_path(env.cfg, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(METRICS if metrics is None else metrics), encoding="utf-8")
    (reports / "backtest.json").write_text(json.dumps(BACKTEST), encoding="utf-8")
    pd.DataFrame({"epoch": [1, 2, 3], "pr_auc": [0.1, 0.3, 0.42], "roc_auc": [0.6, 0.8, 0.9]}).to_csv(
        reports / "training_history.csv", index=False)
    env.graph = graph
    return env


@pytest.fixture
def gnn_env(env) -> SimpleNamespace:
    """``env`` plus a finalized v2 best.pt, metrics.json, training_history.csv and backtest.json."""
    return install_model(env)


def run_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=120).run()


def text_of(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def kpis(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def skill_kpi(at: AppTest):
    """The model-skill KPI tile (its label names the rain it is given and the split, e.g.
    'PR-AUC given junction rain · test 2024')."""
    return next(m for m in at.metric if "PR-AUC" in m.label)


# --------------------------------------------------------------------------- the app (AppTest)


@pytest.mark.e2e
def test_app_without_model_shows_physics_banner_and_forecast_fallback(env):
    at = run_app()
    assert_clean(at)
    warnings = text_of(at.warning)
    assert "physics baseline" in warnings and "Live forecast unavailable" in warnings
    assert "most recent notable storm" in warnings
    values = kpis(at)
    assert list(values) == ["Junctions at risk", "Max flood probability", "Peak hour", "Rain over horizon",
                            "Model PR-AUC"]
    assert values["Model PR-AUC"] == "teacher" and values["Junctions at risk"].endswith("/ 16")
    assert at.sidebar.radio(key="nf_predictor_unavailable").disabled
    assert len(at.get("download_button")) == 3
    assert "No trained GNN is loaded" in text_of(at.info)             # metrics of an unloaded model are not shown
    assert "Model evaluation" not in text_of(at.markdown)


@pytest.mark.e2e
def test_app_with_model_renders_gnn_dashboard(gnn_env):
    """X4 / R3-05 / R2-03: TEST-split headline with explicit wording and the delta vs the strongest baseline."""
    at = run_app()
    assert_clean(at)
    assert at.sidebar.radio(key="nf_predictor").value == "gnn"
    skill = skill_kpi(at)
    assert skill.label == "PR-AUC given junction rain · test 2024" and skill.value == "0.42"   # fallback replay
    assert skill.proto.delta == "-0.13 vs GBDT"                         # 0.42 vs gradient boosting's 0.55
    assert "held-out test year 2024; model selection and calibration used validation year 2022" in skill.help
    assert "junction rain field" in skill.help
    assert "physics baseline" not in text_of(at.warning)
    assert "older data configuration" not in text_of(at.warning) + text_of(at.info)  # the checkpoint's hash is current
    markdown = text_of(at.markdown)
    assert "Model evaluation — held-out test year 2024; model selection and calibration used validation year 2022" \
        in markdown and "Forecast backtest" in markdown
    table = next(d.value for d in at.dataframe if "Split" in d.value.columns)
    assert table["Split"].tolist()[:3] == ["test 2024"] * 3 and "val 2022" in table["Split"].tolist()
    assert list(table["Model"])[:2] == ["Namma-Flow GNN", "Gradient-boosted trees (no graph)"]
    captions = text_of(at.caption)
    assert "given the junction rain field" in captions and "Platt-calibrated on the validation year 2022" in captions
    card = next(d.value for d in at.dataframe if list(d.value.columns) == ["Item", "Value"])
    assert card.set_index("Item").loc["Calibration", "Value"].startswith("Platt")
    backtest = next(d.value for d in at.dataframe if "Physics PR-AUC" in d.value.columns)
    assert backtest.loc["0 h ahead", "Physics PR-AUC"] == "teacher (= labels)"   # R4-05
    assert backtest.loc["24 h ahead", "Physics PR-AUC"] == "0.05"
    assert at.selectbox(key="nf_junction").value in gnn_env.graph.node_ids
    assert at.sidebar.select_slider(key="nf_alert_gnn_0.4").value == 0.4


@pytest.mark.e2e
def test_app_stale_model_note_uses_the_checkpoint_hash(env):
    """R4-02: the stale-data check uses the loaded checkpoint's dataset_config_hash."""
    install_model(env, dataset_config_hash="0123456789abcdef")
    at = run_app()
    assert_clean(at)
    assert "older data configuration (dataset hash 0123456789abcdef" in text_of(at.info)
    assert "0123456789abcdef" in text_of(at.warning)                   # and in the About panel


@pytest.mark.e2e
def test_app_legacy_metrics_are_worded_as_validation(env):
    """R5-08: validation-only metrics are labelled as tuned on validation (optimistic), not 'held-out'."""
    install_model(env, metrics=LEGACY_METRICS, format_version=1, calibration=None, graph_attributes_sha256=None,
                  temperature=1.5)
    at = run_app()
    assert_clean(at)
    skill = skill_kpi(at)
    assert skill.label == "PR-AUC given junction rain · val" and skill.proto.delta == "+0.22 vs logistic"
    assert "optimistic" in skill.help and "held-out" not in skill.help
    assert "optimistic" in text_of(at.markdown)
    assert "temperature-calibrated" in text_of(at.caption)


@pytest.mark.e2e
def test_app_mode_switching(gnn_env):
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("historical").run()
    assert_clean(at)
    events = at.sidebar.selectbox(key="nf_event").options
    assert len(events) == 3 and "Pick a date" in events[-1]          # two storms + custom
    assert "Historical replay 2022-09-0" in text_of(at.markdown)
    at.sidebar.selectbox(key="nf_event").set_value("__custom__").run()
    assert_clean(at)
    at.sidebar.slider(key="nf_start_hour").set_value(6).run()
    assert_clean(at)
    assert "06:00" in text_of(at.markdown)
    at.sidebar.radio(key="nf_mode").set_value("design").run()
    assert_clean(at)
    assert "Design storm" in text_of(at.markdown)
    at.sidebar.selectbox(key="nf_storm").set_value("__custom__").run()
    at.sidebar.slider(key="nf_total").set_value(120.0).run()
    at.sidebar.slider(key="nf_duration").set_value(6).run()
    assert_clean(at)
    assert "120 mm in 6 h" in text_of(at.markdown)
    at.sidebar.number_input(key="nf_rain_scale").set_value(1.0).run()
    assert_clean(at)
    assert float(kpis(at)["Rain over horizon"].split()[0]) == pytest.approx(120.0, abs=0.2)


@pytest.mark.e2e
def test_app_time_alert_uncertainty_and_map_controls(gnn_env):
    at = run_app()
    at.sidebar.segmented_control(key="nf_horizon").set_value(12).run()
    assert_clean(at)
    assert at.metric[0].help and "next 12 h" in at.metric[0].help
    at.sidebar.toggle(key="nf_peak").set_value(False).run()
    slider = next(s for s in at.sidebar.select_slider if s.key.startswith("nf_hour_"))
    slider.set_value(3).run()
    assert_clean(at)
    assert at.metric[0].label.startswith("At risk at ")
    at.sidebar.select_slider(key="nf_alert_gnn_0.4").set_value(1e-6).run()
    assert_clean(at)
    assert kpis(at)[at.metric[0].label].startswith("16 /")            # every junction clears a tiny threshold
    at.sidebar.toggle(key="nf_mc").set_value(True).run()
    assert_clean(at)
    assert "MC dropout × 3" in text_of(at.markdown)
    at.sidebar.radio(key="nf_color").set_value("tiers").run()
    at.sidebar.toggle(key="nf_3d").set_value(False).run()
    at.sidebar.selectbox(key="nf_basemap").set_value("dark").run()
    at.sidebar.toggle(key="nf_roads").set_value(False).run()
    at.sidebar.toggle(key="nf_drains").set_value(False).run()
    assert_clean(at)
    node = gnn_env.graph.node_ids[5]
    at.selectbox(key="nf_junction").set_value(node).run()
    assert_clean(at)
    assert at.selectbox(key="nf_junction").value == node
    at.sidebar.radio(key="nf_predictor").set_value("physics").run()
    assert_clean(at)
    assert kpis(at)["Model PR-AUC"] == "teacher" and at.sidebar.toggle(key="nf_mc").disabled


@pytest.mark.e2e
def test_app_live_forecast_success(gnn_env, monkeypatch):
    def fake_forecast(cfg, now=None, hours=None):
        index = pd.date_range("2024-10-10 00:00", periods=96, freq="h", tz=TZ)
        rain = np.zeros(96)
        rain[60:64] = [4.0, 12.0, 6.0, 1.0]
        flags = np.arange(96) >= 48
        return sc.Scenario("Live forecast (issued 2024-10-11 23:00)", "forecast", index, rain, flags, 48,
                           "Open-Meteo forecast: 23.0 mm over the next 48 h", "forecast")

    monkeypatch.setattr(sc, "forecast_scenario", fake_forecast)
    at = run_app()
    assert_clean(at)
    assert "Live forecast unavailable" not in text_of(at.warning)
    assert "Live forecast (issued 2024-10-11 23:00)" in text_of(at.markdown)
    assert kpis(at)["Rain over horizon"] == "23.0 mm"
    at.sidebar.button(key="nf_refresh").click().run()
    assert_clean(at)


@pytest.mark.e2e
def test_app_without_weather_record(env):
    Path(env.cfg["paths"]["weather_file"]).unlink()
    at = run_app()
    assert not at.exception
    assert "no weather record to replay" in text_of(at.warning)
    at.sidebar.radio(key="nf_mode").set_value("historical").run()
    assert not at.exception and "No weather record to replay" in text_of(at.error)
    at.sidebar.radio(key="nf_mode").set_value("design").run()
    assert_clean(at)
    assert len(at.metric) == 5


@pytest.mark.e2e
def test_app_without_graph_offers_setup_and_demo_build(env):
    graph_file = Path(env.cfg["paths"]["graph_file"])
    graph_file.unlink()
    at = run_app()
    assert not at.exception
    assert "No usable road graph" in text_of(at.error)
    assert "01_extract_network.py" in text_of(at.get("code"))
    at.button(key="nf_build_demo").click().run()
    assert_clean(at)
    assert graph_file.is_file() and load_graph(graph_file).graph["source"] == "synthetic_grid"
    assert kpis(at)["Junctions at risk"].endswith("/ 36")                # 6 x 6 synthetic grid


@pytest.mark.e2e
def test_app_demo_build_failure_is_reported(env, monkeypatch):
    Path(env.cfg["paths"]["graph_file"]).unlink()
    from src.data_pipeline import network

    def broken(cfg, force=False, **kwargs):
        raise ValueError("disk full")

    monkeypatch.setattr(network, "extract_network", broken)
    at = run_app()
    at.button(key="nf_build_demo").click().run()
    assert not at.exception and "Could not build the demo graph: ValueError: disk full" in text_of(at.error)


@pytest.mark.e2e
def test_app_invalid_and_missing_config(env, monkeypatch):
    write_config(deep_merge(env.cfg, {"app": {"map_style": "neon"}}), env.path)
    at = run_app()
    assert not at.exception and "app.map_style" in text_of(at.error)
    monkeypatch.setenv("NAMMA_FLOW_CONFIG", str(env.tmp / "nope.yaml"))
    at = run_app()
    assert not at.exception and "Config file not found" in text_of(at.error)


@pytest.mark.e2e
@pytest.mark.parametrize("kind, expected", [
    ("corrupt", "unreadable"), ("mismatch", "different road graph"),
    ("unfinalized", "checkpoint is not finalized"),              # R4-02 / R5-05: in-progress per-epoch file
    ("attributes", "attributes (elevation"),                     # X4: graph re-enriched after training
])
def test_app_unusable_checkpoint_falls_back_to_physics(env, kind, expected):
    path = resolve_path(env.cfg, "checkpoint_dir") / "best.pt"
    graph = graph_to_arrays(load_graph(env.cfg["paths"]["graph_file"]))
    if kind == "corrupt":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a checkpoint")
    elif kind == "mismatch":
        atomic_torch_save(make_checkpoint(env.cfg, graph_to_arrays(make_grid_graph(3, 3))), path)
    elif kind == "unfinalized":
        atomic_torch_save(make_checkpoint(env.cfg, graph, finalized_utc=None, kind="best"), path)
    else:
        atomic_torch_save(make_checkpoint(env.cfg, graph, graph_attributes_sha256="0" * 16), path)
    at = run_app()
    assert_clean(at)
    warnings = text_of(at.warning)
    assert "physics baseline" in warnings and expected in warnings
    assert "No trained GNN is loaded" in text_of(at.info)


@pytest.mark.e2e
def test_app_prediction_failure_is_reported(env, monkeypatch):
    def failing_predict(self, scenario, mc_samples=0, **kwargs):
        raise PredictionError("node rain must be finite")

    monkeypatch.setattr(PhysicsPredictor, "predict", failing_predict)
    at = run_app()
    assert not at.exception and "Prediction failed" in text_of(at.error)


@pytest.mark.e2e
def test_app_unexpected_error_is_contained(env, monkeypatch):
    """R5-09: the traceback stays in the server log unless NAMMA_FLOW_DEBUG=1."""
    def boom(*args, **kwargs):
        raise KeyError(f"surprise in {project_root() / 'app'}")

    monkeypatch.setattr(ui, "build_kpis", boom)
    at = run_app()
    assert not at.exception and "Something went wrong" in text_of(at.error)
    assert not at.get("code") and "Technical details" not in [e.label for e in at.get("expandable")]
    assert str(project_root()) not in text_of(at.error) and "surprise in app" in text_of(at.error)
    monkeypatch.setenv("NAMMA_FLOW_DEBUG", "1")
    at = run_app()
    assert "KeyError" in text_of(at.get("code"))


def deck_spec(at: AppTest) -> dict:
    return json.loads(at.get("deck_gl_json_chart")[0].proto.json)


@pytest.mark.e2e
def test_app_map_layers_with_synthetic_drain_line(gnn_env):
    at = run_app()
    assert_clean(at)
    spec = deck_spec(at)
    ids = [layer["id"] for layer in spec["layers"]]
    assert ids == [ml.WATER_LINE_LAYER_ID, ml.ROAD_LAYER_ID, ml.JUNCTION_LAYER_ID, ml.SELECTED_LAYER_ID]
    assert spec["mapProvider"] == "carto" and "positron" in spec["mapStyle"]
    assert len(spec["layers"][2]["data"]) == 16 and not at.get("link_button")   # test grid: no OSM node ids


@pytest.mark.e2e
def test_app_osm_waterways_and_osm_links(env):
    graph = make_grid_graph(4, 4)
    graph.graph.update(source="osm_bbox", elevation_source="srtm", drain_source="osm")
    save_graph(graph, env.cfg["paths"]["graph_file"])
    Path(env.cfg["paths"]["waterways_file"]).write_text(json.dumps(WATERWAYS), encoding="utf-8")
    at = run_app()
    assert_clean(at)
    ids = [layer["id"] for layer in deck_spec(at)["layers"]]
    assert ids[:2] == [ml.WATER_BODY_LAYER_ID, ml.WATER_LINE_LAYER_ID]
    assert "openstreetmap.org/node/" in at.get("link_button")[0].proto.url
    provenance = at.dataframe[-1].value
    assert provenance["Value"].str.contains("OpenStreetMap \\(bounding-box query\\)").any()
    assert provenance["Value"].str.contains("OpenStreetMap drains").any()


@pytest.mark.e2e
def test_app_replay_truncated_at_record_end(env):
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("historical").run()
    at.sidebar.selectbox(key="nf_event").set_value("__custom__").run()
    at.sidebar.date_input(key="nf_day").set_value(pd.Timestamp("2022-09-20").date()).run()
    at.sidebar.slider(key="nf_start_hour").set_value(22).run()
    assert_clean(at)
    assert "Only 2 h of this scenario are available." in text_of(at.sidebar.caption)
    assert "next 2 h" in at.metric[0].help
    at.sidebar.slider(key="nf_start_hour").set_value(23).run()      # the very last hour of the record
    assert_clean(at)
    assert "Only 1 h of this scenario are available." in text_of(at.sidebar.caption)
    assert not [s for s in at.sidebar.select_slider if str(s.key).startswith("nf_hour_")]


@pytest.mark.e2e
def test_app_scenario_failure_is_reported(env, monkeypatch):
    def broken(*args, **kwargs):
        raise sc.ScenarioError("The storm does not fit in horizon_h=48")

    monkeypatch.setattr(sc, "design_storm_scenario", broken)
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("design").run()
    assert not at.exception and "Could not build the scenario" in text_of(at.error)


@pytest.mark.e2e
def test_app_uses_checkpoint_metrics_without_metrics_file_and_clears_caches(gnn_env):
    (resolve_path(gnn_env.cfg, "reports_dir") / "metrics.json").unlink()
    at = run_app()
    assert_clean(at)
    assert skill_kpi(at).value == "0.42" and "No training metrics found" not in text_of(at.info)
    at.sidebar.button(key="nf_clear").click().run()
    assert_clean(at)
    assert len(at.metric) == 5


@pytest.mark.e2e
def test_app_empty_prediction_is_reported(env, monkeypatch):
    def empty(self, scenario, mc_samples=0, **kwargs):
        n = self.num_nodes
        return PredictionResult(timestamps=pd.DatetimeIndex([], tz=TZ), prob=np.zeros((0, n)), prob_std=None,
                                node_rain=np.zeros((0, n)), depth_physics=None, node_ids=self.graph.node_ids,
                                lon=self.graph.lon, lat=self.graph.lat, static=self.static, threshold=0.5,
                                horizons_h=(12,), scenario_name="empty")

    monkeypatch.setattr(PhysicsPredictor, "predict", empty)
    at = run_app()
    assert not at.exception and "prediction is empty" in text_of(at.info)


@pytest.mark.e2e
def test_app_honours_inference_thread_setting(gnn_env, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    write_config(deep_merge(gnn_env.cfg, {"inference": {"num_threads": 2}}), gnn_env.path)
    at = run_app()
    assert_clean(at)
    assert calls and set(calls) == {2}




@pytest.mark.e2e
def test_app_shows_no_absolute_paths(gnn_env):
    """R5-09: config and checkpoint paths are shown relative (or by name), never absolute."""
    at = run_app()
    assert_clean(at)
    assert "Config: `config.yaml`" in text_of(at.sidebar.caption)
    card = next(d.value for d in at.dataframe if list(d.value.columns) == ["Item", "Value"])
    assert card.set_index("Item").loc["Checkpoint", "Value"] == "best.pt (format v2)"
    page = " ".join(text_of(getattr(at, kind)) for kind in ("markdown", "caption", "info", "warning", "error"))
    assert str(gnn_env.tmp) not in page and str(project_root()) not in page


@pytest.mark.e2e
def test_app_design_storm_rain_field_and_short_history_notice(env):
    """R4-03 (canonical field caption) and R4-07 (dry-padded history is surfaced, not only logged)."""
    write_config(deep_merge(env.cfg, {"inference": {"history_hours": 12}}), env.path)
    install_model(env)
    at = run_app()
    at.sidebar.radio(key="nf_mode").set_value("design").run()
    assert_clean(at)
    assert "averaged over 32 stochastic rain-field realisations drawn on the canonical storm-anchored" in text_of(
        at.caption)
    warnings = text_of(at.warning)
    assert "Short rain history" in warnings and "needs 30 h" in warnings and "missing 18 h" in warnings
    at.sidebar.radio(key="nf_predictor").set_value("physics").run()
    assert_clean(at)
    assert "Short rain history" not in text_of(at.warning)


@pytest.mark.e2e
def test_app_renders_scenario_notes(gnn_env, monkeypatch):
    def short_forecast(cfg, now=None, hours=None):
        index = pd.date_range("2024-10-10 00:00", periods=60, freq="h", tz=TZ)
        return sc.Scenario("Live forecast (issued 2024-10-11 23:00)", "forecast", index, np.zeros(60),
                           np.arange(60) >= 48, 48, "Open-Meteo forecast", "forecast",
                           notes=("The forecast covers only the next 12 of the requested 48 hours.",))

    monkeypatch.setattr(sc, "forecast_scenario", short_forecast)
    at = run_app()
    assert_clean(at)
    assert "covers only the next 12" in text_of(at.info)
    assert "junction rain averaged over 32 stochastic rain-field realisations" in text_of(at.caption)
