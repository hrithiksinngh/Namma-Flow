"""Tests for prediction results (tables / exports), the forecast backtest and the CLI.

Shares the offline fixtures (4 x 4 grid graph, tiny untrained checkpoint, storm record) of
:mod:`tests.test_inference`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_pipeline import weather
from src.inference import backtest as bt
from src.inference import predict as cli
from src.inference.predictor import PhysicsPredictor, PredictionResult
from src.inference.results import NODE_TABLE_COLUMNS, json_safe
from src.utils.config import resolve_path
from tests.test_inference import (  # noqa: F401 - fixtures are used by name
    TZ,
    _warned,
    ckpt_path,
    graph,
    icfg,
    log,
    predictor,
    record_cfg,
)


# --------------------------------------------------------------------------- helpers


def make_result(n_hours: int = 5, n_nodes: int = 3, **overrides) -> PredictionResult:
    prob = np.linspace(0.0, 1.0, n_hours * n_nodes, dtype=np.float32).reshape(n_hours, n_nodes)
    fields = dict(
        timestamps=pd.date_range("2024-10-01", periods=n_hours, freq="h", tz=TZ), prob=prob, prob_std=prob / 10,
        node_rain=np.ones((n_hours, n_nodes)), depth_physics=None, node_ids=tuple(range(n_nodes)),
        lon=np.linspace(77.66, 77.69, n_nodes), lat=np.full(n_nodes, 12.93),
        static=pd.DataFrame({"node_id": range(n_nodes), "elevation": 880.0, "dist_to_drain_m": 100.0,
                             "relative_elevation": -1.0}),
        threshold=0.5, horizons_h=(2, 4, 12), scenario_name="unit", areal_mm=np.arange(n_hours, dtype=float),
        is_forecast=np.ones(n_hours, dtype=bool),
    )
    fields.update(overrides)
    return PredictionResult(**fields)


# --------------------------------------------------------------------------- PredictionResult


@pytest.mark.unit
def test_result_horizons_and_tiers():
    result = make_result()
    np.testing.assert_allclose(result.horizon_max(2), result.prob[1])
    np.testing.assert_allclose(result.horizon_max(100), result.prob[-1])   # clipped to T
    assert result.horizon_peak_index(3).tolist() == [2, 2, 2]
    tiers = result.risk_tier([0.0, 0.2499, 0.25, 0.5, 0.75, 1.0, np.nan, 2.0, -1.0])
    assert tiers.tolist() == ["Low", "Low", "Moderate", "High", "Severe", "Severe", "Low", "Severe", "Low"]
    for bad in (0, -3, True, 1.5):
        with pytest.raises(ValueError, match="hours"):
            result.horizon_max(bad)


@pytest.mark.unit
def test_result_tables_and_summary():
    result = make_result(depth_physics=np.full((5, 3), 0.2))
    table = result.node_table(4)
    assert list(table.columns) == list(NODE_TABLE_COLUMNS) and len(table) == 3
    assert table["max_prob"].tolist() == pytest.approx(result.prob[3].tolist())
    assert (table["peak_time"] == result.timestamps[3]).all()
    assert table["max_depth_m"].tolist() == pytest.approx([0.2] * 3)
    assert table["rain_total_mm"].eq(4.0).all() and table["prob_std"].tolist() == pytest.approx(
        (result.prob[3] / 10).tolist())
    top = result.top_k(2, hours=5)
    assert top["rank"].tolist() == [1, 2] and top["max_prob"].is_monotonic_decreasing
    summary = result.summary(5)
    assert summary["junctions_at_risk"] == int((result.prob.max(0) >= 0.5).sum())
    assert summary["rain_total_mm"] == 10.0 and summary["hours"] == 5 and set(summary["horizons"]) == {2, 4}
    assert sum(summary["tier_counts"].values()) == 3
    with pytest.raises(ValueError, match="k must"):
        result.top_k(0)


@pytest.mark.unit
def test_result_exports(tmp_path):
    result = make_result()
    collection = result.to_geojson(5, include_timeline=True)
    text = json.dumps(collection, allow_nan=False)
    assert collection["type"] == "FeatureCollection" and len(collection["features"]) == 3
    feature = collection["features"][0]
    assert feature["geometry"]["type"] == "Point" and len(feature["properties"]["prob_timeline"]) == 5
    assert feature["properties"]["tier"] in {"Low", "Moderate", "High", "Severe"} and "NaN" not in text
    assert collection["metadata"]["summary"]["hours"] == 5
    path = result.write_geojson(tmp_path / "out/p.geojson")
    assert json.loads(path.read_text())["features"][1]["properties"]["max_depth_m"] is None
    csv = pd.read_csv(result.to_csv(tmp_path / "t.csv", hours=2))
    assert len(csv) == 3 and csv["peak_time"].str.contains("+05:30", regex=False).all()


@pytest.mark.unit
def test_empty_result_is_supported(tmp_path):
    result = make_result(n_hours=0, prob_std=None, areal_mm=np.zeros(0), is_forecast=np.zeros(0, bool))
    assert result.horizon_max(48).tolist() == [0.0, 0.0, 0.0]
    assert result.horizon_peak_index(48).tolist() == [-1, -1, -1]
    table = result.node_table()
    assert table["peak_time"].isna().all() and table["prob_std"].isna().all()
    summary = result.summary()
    assert summary["start"] is None and summary["peak_time"] is None and summary["horizons"] == {}
    json.dumps(result.to_geojson(), allow_nan=False)
    result.to_csv(tmp_path / "empty.csv")


@pytest.mark.unit
@pytest.mark.parametrize("override, message", [
    ({"prob": np.full((5, 3), 1.5)}, "probabilities"),
    ({"prob": np.zeros((4, 3))}, "shape"),
    ({"node_rain": np.zeros((5, 2))}, "node_rain"),
    ({"prob_std": np.zeros(3)}, "prob_std"),
    ({"lon": np.zeros(2)}, "lon"),
    ({"static": pd.DataFrame({"a": [1]})}, "static"),
    ({"threshold": 2.0}, "threshold"),
    ({"horizons_h": ()}, "horizons_h"),
    ({"risk_tiers": (("High", 0.5),)}, "risk_tiers"),
    ({"timestamps": [1, 2, 3, 4, 5]}, "DatetimeIndex"),
    ({"areal_mm": np.zeros(4)}, "areal_mm"),
])
def test_result_validation(override, message):
    with pytest.raises(ValueError, match=message):
        make_result(**override)


@pytest.mark.unit
def test_result_is_immutable():
    result = make_result()
    with pytest.raises(ValueError):
        result.prob[0, 0] = 0.3
    with pytest.raises(AttributeError):
        result.threshold = 0.1


# --------------------------------------------------------------------------- backtest


@pytest.mark.unit
def test_json_safe():
    value = {"a": np.float32(1.5), "b": [np.nan, np.inf, np.int64(3)], "c": pd.Timestamp("2024-01-01", tz=TZ),
             "d": pd.NaT, "e": np.array([True, False]), "f": Path("/x"), 7: (1, 2)}
    assert json_safe(value) == {"a": 1.5, "b": [None, None, 3], "c": "2024-01-01T00:00:00+05:30", "d": None,
                                "e": [True, False], "f": "/x", "7": [1, 2]}




def fake_previous_runs(obs_peak: float = 30.0):
    def fetch(cfg, start, end, lead_days=(1, 2), lat=None, lon=None):
        index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h", name="timestamp")
        obs = np.zeros(len(index))
        storm = (index >= pd.Timestamp("2024-10-02 15:00", tz=TZ)) & (index <= pd.Timestamp("2024-10-02 18:00", tz=TZ))
        obs[storm] = [obs_peak / 3, obs_peak, obs_peak / 2, obs_peak / 6][: int(storm.sum())]
        return pd.DataFrame({"precip_lead_0": obs, "precip_lead_24h": np.roll(obs, 1) * 0.8,
                             "precip_lead_48h": obs * 0.3, "is_imputed": False}, index=index)
    return fetch


@pytest.mark.unit
def test_classification_metrics_reference_values():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.6, 0.7, 0.1, 0.3])
    m = bt.classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3) and m["f2"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(7 / 9) and 0 < m["pr_auc"] <= 1
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    none = bt.classification_metrics(np.zeros(4), np.full(4, 0.1), 0.5)
    assert none["pr_auc"] is None and none["recall"] is None and none["precision"] is None and none["f2"] is None
    silent = bt.classification_metrics([1, 0], [0.1, 0.1], 0.5)
    assert silent["recall"] == 0.0 and silent["precision"] is None and silent["f2"] == 0.0
    with pytest.raises(ValueError, match="values"):
        bt.classification_metrics([1, 0], [0.5], 0.5)
    with pytest.raises(ValueError, match="probabilities"):
        bt.classification_metrics([1, 0], [0.5, 1.5], 0.5)
    with pytest.raises(ValueError, match="threshold"):
        bt.classification_metrics([1, 0], [0.5, 0.5], 2.0)


@pytest.mark.unit
def test_rain_skill():
    skill = bt.rain_skill(np.array([0.0, 2.0, 5.0, 0.0]), np.array([0.0, 1.0, 4.0, 1.0]))
    assert skill["total_mm"] == 6.0 and skill["observed_total_mm"] == 7.0 and skill["mae_mm_h"] == 0.75
    assert skill["wet_hour_hit_rate"] == 1.0 and 0 < skill["correlation"] <= 1
    flat = bt.rain_skill(np.zeros(3), np.zeros(3))
    assert flat["correlation"] is None and flat["wet_hour_hit_rate"] is None


@pytest.mark.unit
def test_backtest_offline_is_unavailable(predictor, icfg):
    report = bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-03", predictor)
    assert report["status"] == "unavailable" and "Offline" in report["reason"]
    assert not (resolve_path(icfg, "reports_dir") / "backtest.json").exists()
    assert bt.format_backtest(report)[0].startswith("Backtest unavailable")


@pytest.mark.integration
def test_backtest_report(predictor, icfg, monkeypatch):
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs())
    report = bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-03", predictor)
    assert report["status"] == "ok" and report["n_hours"] == 72 and report["n_nodes"] == 16
    assert report["spinup_hours"] == 72 and report["labels"]["n_positive"] > 0
    assert list(report["leads"]) == ["lead_0h", "areal_ceiling", "lead_24h", "lead_48h"]
    lead = report["leads"]["lead_24h"]
    assert lead["lead_hours"] == 24 and lead["junction_hours"]["threshold"] == pytest.approx(0.4)
    assert lead["rain"]["total_mm"] == pytest.approx(0.8 * report["leads"]["lead_0h"]["rain"]["total_mm"])
    assert "physics_baseline" in lead and lead["physics_baseline"]["junction_hours"]["pr_auc"] is not None
    saved = json.loads(Path(report["report_path"]).read_text())
    assert saved["leads"]["lead_0h"]["junction_hours"]["n"] == 72 * 16 and saved["predictor"]["kind"] == "gnn"
    assert saved["predictor"]["checkpoint"] == "best.pt"                      # no absolute paths (R5-09)
    lines = bt.format_backtest(report)
    assert any(line.startswith("lead_48h") for line in lines)


@pytest.mark.integration
def test_backtest_lead0_physics_is_labelled_as_the_teacher(predictor, icfg, monkeypatch):
    """Regression R4-05: the lead-0 physics run is the label generator (PR-AUC 1.0), not a baseline."""
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs())
    report = bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-03", predictor, save=False)
    lead0, lead24 = report["leads"]["lead_0h"], report["leads"]["lead_24h"]
    teacher = lead0["physics_baseline"]
    assert teacher["role"] == "teacher" and teacher["junction_hours"]["pr_auc"] == 1.0
    assert "label generator" in lead0["physics_baseline"]["note"]
    assert lead24["physics_baseline"]["role"] == "forecast_baseline"
    assert bt.is_teacher(lead0) and not bt.is_teacher(lead24) and not bt.is_teacher({"lead_hours": 0})
    assert bt.is_teacher({"lead_hours": 0, "physics_baseline": {"junction_hours": {}}})   # older reports
    lines = bt.format_backtest(report)
    row = next(line for line in lines if line.startswith("lead_0h"))
    assert row.endswith(bt.TEACHER_LABEL) and any("label generator" in line for line in lines)
    assert not next(line for line in lines if line.startswith("lead_24h")).endswith(bt.TEACHER_LABEL)


@pytest.mark.integration
def test_backtest_physics_predictor_and_dry_period(icfg, monkeypatch):
    physics = PhysicsPredictor.from_graph(icfg)
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs(obs_peak=0.0))
    report = bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-02 12:00", physics, save=False)
    assert report["status"] == "ok" and report["n_hours"] == 37 and "report_path" not in report
    assert "physics_baseline" not in report["leads"]["lead_0h"] and "No simulated floods" in report["note"]
    assert report["leads"]["lead_0h"]["junction_hours"]["pr_auc"] is None
    assert any("Note" in line for line in bt.format_backtest(report))


@pytest.mark.unit
def test_backtest_api_failures(predictor, icfg, monkeypatch):
    def broken(*args, **kwargs):
        raise weather.WeatherUnavailable("HTTP 502")
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", broken)
    assert bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-02", predictor)["reason"] == "HTTP 502"
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", lambda *a, **k: pd.DataFrame({"x": [1.0]}))
    assert bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-02", predictor)["status"] == "unavailable"


@pytest.mark.unit
def test_backtest_period_validation(predictor, icfg, log):
    with pytest.raises(ValueError, match="after end"):
        bt.run_forecast_backtest(icfg, "2024-10-05", "2024-10-01", predictor)
    with pytest.raises(ValueError, match="future"):
        bt.backtest_period("2030-01-01", "2030-01-02", TZ, 120, now="2026-01-01")
    with pytest.raises(ValueError, match="max_backtest_days"):
        bt.backtest_period("2024-01-01", "2024-12-31", TZ, 120)
    with pytest.raises(ValueError, match="ISO"):
        bt.backtest_period("soon", "2024-10-01", TZ, 120)
    lo, hi = bt.backtest_period("2025-12-30", "2026-02-01", TZ, 120, now=pd.Timestamp("2026-01-01 10:20", tz=TZ))
    assert hi == pd.Timestamp("2026-01-01 09:00", tz=TZ) and _warned(log, "clamped")
    lo, hi = bt.backtest_period("2024-10-01", "2024-10-01", TZ, 120)
    assert (hi - lo) == pd.Timedelta(hours=23)
    with pytest.raises(TypeError, match="predictor"):
        bt.run_forecast_backtest(icfg, "2024-10-01", "2024-10-02", object())


@pytest.mark.unit
def test_backtest_report_write_failure(icfg, monkeypatch, log):
    def refuse(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(bt, "atomic_write_text", refuse)
    assert bt.write_backtest_report(icfg, {"status": "ok"}) is None and _warned(log, "disk full")


# --------------------------------------------------------------------------- CLI


@pytest.mark.e2e
def test_cli_design_storm_physics(icfg, tmp_path, capsys):
    out = tmp_path / "pred.geojson"
    code = cli.main(["--scenario", "design", "--total-mm", "80", "--duration-h", "3", "--physics", "--out", str(out),
                     "--csv", str(tmp_path / "pred.csv"), "--timeline", "--top", "3"], cfg=icfg)
    printed = capsys.readouterr().out
    assert code == 0 and "Physics baseline" in printed and "Top 3 junctions" in printed
    data = json.loads(out.read_text())
    assert len(data["features"]) == 16 and len(data["features"][0]["properties"]["prob_timeline"]) == 48
    assert (tmp_path / "pred.csv").exists()
    assert cli.main(["--scenario", "design", "--preset", "heavy", "--physics", "--hours", "24"], cfg=icfg) == 0
    assert (resolve_path(icfg, "reports_dir") / "prediction.geojson").exists()


@pytest.mark.e2e
def test_cli_preset_honours_the_storm_offset(icfg, tmp_path, capsys):
    """Regression R4-06 / R5-07: --storm-offset-h is passed through with --preset; mixing flags is refused."""
    out = tmp_path / "preset.geojson"
    argv = ["--scenario", "design", "--preset", "cloud", "--hours", "8", "--storm-offset-h", "0", "--physics",
            "--out", str(out)]
    assert cli.main(argv, cfg=icfg) == 0
    printed = capsys.readouterr().out
    assert "starting 0 h into a 8 h horizon" in printed
    assert "junction rain averaged over 32 stochastic rain-field realisations drawn on the canonical" in printed
    assert cli.main(["--scenario", "design", "--preset", "heavy", "--storm-offset-h", "30", "--physics"],
                    cfg=icfg) == 0
    assert "starting 30 h into a 48 h horizon" in capsys.readouterr().out
    assert cli.main(["--scenario", "design", "--preset", "heavy", "--total-mm", "50", "--physics"], cfg=icfg) == 1
    assert "--preset sets the storm total" in capsys.readouterr().err


@pytest.mark.e2e
def test_cli_reports_dry_padded_history(icfg, ckpt_path, capsys):
    """R4-07: a configured history shorter than the model needs is reported, not silent."""
    short = {**icfg, "inference": {**icfg["inference"], "history_hours": 12}}
    assert cli.main(["--scenario", "design", "--total-mm", "40", "--duration-h", "2"], cfg=short) == 0
    assert "18 h were treated as dry" in capsys.readouterr().out
    run = json.loads((resolve_path(icfg, "reports_dir") / "prediction.geojson").read_text())["metadata"]["run"]
    assert run["padded_history_h"] == 18 and run["checkpoint"] == "best.pt"


@pytest.mark.e2e
def test_cli_gnn_with_mc_dropout(icfg, ckpt_path, capsys):
    assert cli.main(["--scenario", "design", "--total-mm", "40", "--duration-h", "2", "--mc", "--threads", "2"],
                    cfg=icfg) == 0
    printed = capsys.readouterr().out
    assert "Namma-Flow GNN" in printed and "3 MC samples" in printed


@pytest.mark.e2e
def test_cli_handled_failures(icfg, capsys):
    cases = [
        (["--scenario", "design", "--total-mm", "80", "--duration-h", "3"], "--physics"),       # no model
        (["--scenario", "design", "--physics"], "--total-mm"),
        (["--scenario", "historical", "--physics"], "--start"),
        (["--scenario", "historical", "--start", "2022-09-05", "--physics"], "03_weather_ingestion"),
        (["--scenario", "design", "--preset", "tsunami", "--physics"], "preset"),
        (["--scenario", "design", "--total-mm", "5", "--duration-h", "3", "--physics", "--mc", "-2"], "--mc"),
        (["--scenario", "design", "--total-mm", "5", "--duration-h", "3", "--physics", "--threads", "0"], "--threads"),
        (["--scenario", "forecast", "--physics", "--no-fallback"], "Live forecast unavailable"),
        (["--scenario", "forecast", "--physics"], "no weather record"),
        (["--backtest", "2024-10-01", "2024-10-02", "--physics"], "backtest unavailable"),
        (["--backtest", "2024-10-05", "2024-10-01", "--physics"], "after end"),
    ]
    for argv, message in cases:
        assert cli.main(argv, cfg=icfg) == 1, argv
        assert message in capsys.readouterr().err, argv


@pytest.mark.e2e
def test_cli_forecast_falls_back_to_a_notable_event(record_cfg, capsys):
    assert cli.main(["--scenario", "forecast", "--physics", "--offline"], cfg=record_cfg) == 0
    printed = capsys.readouterr().out
    assert "live forecast unavailable" in printed and "Historical replay" in printed
    assert cli.main(["--scenario", "historical", "--start", "2022-09-05T12:00", "--hours", "12", "--physics"],
                    cfg=record_cfg) == 0


@pytest.mark.e2e
def test_cli_config_file_and_graph_errors(icfg, tmp_path, capsys):
    config = {k: v for k, v in icfg.items() if not k.startswith("_")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    argv = ["--config", str(path), "--scenario", "design", "--preset", "0", "--physics", "--top", "0"]
    assert cli.main(argv) == 0
    assert cli.main(["--config", str(tmp_path / "missing.yaml"), "--physics"]) == 1
    assert "Config file not found" in capsys.readouterr().err
    Path(icfg["paths"]["graph_file"]).unlink()
    assert cli.main(["--scenario", "design", "--preset", "0", "--physics"], cfg=icfg) == 1
    assert "01_extract_network" in capsys.readouterr().err
