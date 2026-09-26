"""Regression tests for the backtest's rain-field rows (F1-01) and physics-predictor backtests (F3-03).

* ``lead_0h`` keeps the exact label field (emulation of the teacher);
* ``areal_ceiling`` ("perfect areal forecast (ensemble)") drives the model with the observed areal
  rain but independent rain fields (seed offset 1): the realistic ceiling of any rain forecast;
* the 24 h / 48 h rows use the same offset-1 ensembles, so no forecast row ever reuses the label field;
* a physics-predictor backtest marks its own lead-0 row as the teacher and is written to
  ``backtest_physics.json``, never over the GNN's ``backtest.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.inference import backtest as bt
from src.inference import predict as cli
from src.inference.predictor import PhysicsPredictor
from src.utils.config import deep_merge, resolve_path
from tests.test_inference import ckpt_path, graph, icfg, log, predictor  # noqa: F401 - fixtures
from tests.test_inference_outputs import fake_previous_runs

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
import metrics_view as mv  # noqa: E402

sys.path.remove(str(APP_DIR))

STRIDE = 7919


@pytest.fixture
def small(icfg):
    """Three rain-field members keep the backtest quick."""
    return deep_merge(icfg, {"inference": {"field_members": 3}})


@pytest.mark.integration
def test_backtest_rows_use_exact_field_ceiling_and_offset_ensembles(small, predictor, monkeypatch):
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs())
    report = bt.run_forecast_backtest(small, "2024-10-01", "2024-10-03", predictor, save=False)
    leads = report["leads"]
    assert list(leads) == ["lead_0h", bt.CEILING_KEY, "lead_24h", "lead_48h"]
    assert leads["lead_0h"]["field"] == {"exact": True, "members": 1, "seed_offset": 0}
    for key in (bt.CEILING_KEY, "lead_24h", "lead_48h"):
        assert leads[key]["field"] == {"exact": False, "members": 3, "seed_offset": 1}     # never the label seed
    ceiling = leads[bt.CEILING_KEY]
    assert ceiling["role"] == "areal_ceiling" and ceiling["label"] == bt.CEILING_LABEL
    assert ceiling["rain"]["total_mm"] == pytest.approx(leads["lead_0h"]["rain"]["total_mm"])   # observed areal
    assert ceiling["physics_baseline"]["role"] == "areal_baseline"
    assert leads["lead_0h"]["physics_baseline"]["role"] == "teacher"
    assert leads["lead_0h"]["physics_baseline"]["junction_hours"]["pr_auc"] == 1.0
    assert leads["lead_24h"]["physics_baseline"]["role"] == "forecast_baseline"
    assert report["field_ensemble"]["members"] == 3 and report["field_ensemble"]["seed_offset"] == 1
    assert report["field_ensemble"]["seed_stride"] == STRIDE
    lines = bt.format_backtest(report)
    row = next(line for line in lines if line.startswith(bt.CEILING_KEY))
    assert "ens x3" in row and any(line.startswith("lead_0h") and "exact" in line for line in lines)
    assert any(bt.CEILING_NOTE in line for line in lines)
    override = bt.run_forecast_backtest(small, "2024-10-01", "2024-10-02", predictor, save=False, field_members=2)
    assert override["leads"]["lead_48h"]["field"]["members"] == 2


@pytest.mark.integration
def test_ceiling_row_is_driven_by_other_fields(small, predictor, monkeypatch):
    """The ceiling row differs from lead 0 only by the junction rain field (same areal rain)."""
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs(obs_peak=45.0))
    report = bt.run_forecast_backtest(small, "2024-10-01", "2024-10-03", predictor, save=False)
    exact, ceiling = report["leads"]["lead_0h"], report["leads"][bt.CEILING_KEY]
    assert exact["rain"]["correlation"] == pytest.approx(1.0) and ceiling["rain"]["correlation"] == pytest.approx(1.0)
    assert exact["junction_hours"]["brier"] != pytest.approx(ceiling["junction_hours"]["brier"])
    teacher, areal = exact["physics_baseline"]["junction_hours"], ceiling["physics_baseline"]["junction_hours"]
    assert teacher["pr_auc"] == 1.0 and areal["pr_auc"] < 1.0      # the teacher cannot know another field either


@pytest.mark.integration
def test_physics_backtest_marks_its_lead0_as_the_teacher_and_uses_its_own_file(small, monkeypatch):
    """F3-03: a physics backtest's lead-0 row is the label generator itself, and never overwrites backtest.json."""
    physics = PhysicsPredictor.from_graph(small)
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs())
    report = bt.run_forecast_backtest(small, "2024-10-01", "2024-10-03", physics)
    lead0 = report["leads"]["lead_0h"]
    assert lead0["role"] == "teacher" and "label generator" in lead0["note"] and bt.predictor_is_teacher(lead0)
    assert lead0["junction_hours"]["pr_auc"] == 1.0 and "physics_baseline" not in lead0
    assert not bt.predictor_is_teacher(report["leads"]["lead_24h"])
    reports = resolve_path(small, "reports_dir")
    assert Path(report["report_path"]).name == bt.PHYSICS_REPORT_NAME
    assert (reports / bt.PHYSICS_REPORT_NAME).is_file() and not (reports / bt.REPORT_NAME).exists()
    lines = bt.format_backtest(report)
    row = next(line for line in lines if line.startswith("lead_0h"))
    assert f"{bt.TEACHER_LABEL} (the predictor is the label generator" in row and row.count("1.000") == 1  # rain corr
    assert next(line for line in lines[lines.index(row) + 1:] if line.startswith("lead_0h")).endswith(bt.TEACHER_LABEL)
    table = mv.backtest_table(report)
    assert table.loc["0 h ahead", "PR-AUC"] == bt.TEACHER_LABEL and table.loc["0 h ahead", "Recall"] == bt.TEACHER_LABEL
    assert table.loc["24 h ahead", "PR-AUC"] != bt.TEACHER_LABEL
    assert "physics baseline" in mv.backtest_note(report, {"created_utc": "x"})


@pytest.mark.unit
def test_report_name_by_predictor_kind(icfg):
    assert bt.report_name("physics") == "backtest_physics.json" and bt.report_name("gnn") == "backtest.json"
    path = bt.write_backtest_report(icfg, {"status": "ok", "predictor": {"kind": "physics"}})
    assert path.name == "backtest_physics.json"
    assert bt.write_backtest_report(icfg, {"status": "ok"}).name == "backtest.json"


@pytest.mark.unit
def test_format_backtest_handles_older_reports():
    older = {"status": "ok", "start": "a", "end": "b", "n_hours": 1, "n_nodes": 1,
             "predictor": {"label": "GNN", "threshold": 0.3},
             "labels": {"n_positive": 0, "pos_rate": 0.0, "flooded_hours": 0, "peak_flooded_fraction": 0.0},
             "leads": {"lead_0h": {"lead_hours": 0, "rain": {"total_mm": 1.0, "correlation": None},
                                   "junction_hours": {k: None for k in ("pr_auc", "roc_auc", "recall", "precision",
                                                                        "f2")},
                                   "hourly_any": {"recall": None, "precision": None}},
                       "lead_24h": {"lead_hours": 24, "rain": {"total_mm": 1.0, "correlation": None},
                                    "junction_hours": {k: 0.1 for k in ("pr_auc", "roc_auc", "recall", "precision",
                                                                        "f2")},
                                    "hourly_any": {"recall": None, "precision": None}}}}
    lines = bt.format_backtest(older)
    assert any(line.startswith("lead_0h") and " exact " in line for line in lines)
    assert any(line.startswith("lead_24h") and "1 field" in line for line in lines)
    assert not any(bt.CEILING_NOTE in line for line in lines)


@pytest.mark.e2e
def test_cli_members_flag_and_physics_backtest_file(small, monkeypatch, capsys):
    monkeypatch.setattr(bt.weather, "fetch_previous_runs", fake_previous_runs())
    code = cli.main(["--backtest", "2024-10-01", "2024-10-02", "--physics", "--members", "2"], cfg=small)
    printed = capsys.readouterr().out
    assert code == 0 and "backtest_physics.json" in printed and "ens x2" in printed
    assert cli.main(["--backtest", "2024-10-01", "2024-10-02", "--physics", "--members", "0"], cfg=small) == 1
    assert "--members must be >= 1" in capsys.readouterr().err


@pytest.mark.unit
def test_cli_help_names_the_config_resolution():
    """F4-11: --config defaults to $NAMMA_FLOW_CONFIG, else config/config.yaml (as load_config resolves it)."""
    text = " ".join(cli.build_parser().format_help().split())
    assert "(default: $NAMMA_FLOW_CONFIG or config/config.yaml)" in text
    assert "--members K" in text
