"""Cross-module contract tests written during integration (final fix round).

Each owner tests its side of an interface against a literal spec; these tests pin the two sides
to EACH OTHER, so a drift in one module fails here even when both unit suites still pass:

* X5 <-> X6: the serving rain-field ensemble (``src.inference.field_ensemble``) and the areal-skill
  evaluation (``src.training.areal_fields``) must draw the same member seeds and the same junction
  rain, otherwise the backtest's "perfect areal forecast (ensemble)" ceiling and
  ``areal_skill.json``'s ``field_ensemble`` row would describe different ensembles.
* X6 <-> X7: ``python -m src.training.areal_skill --split validation`` writes the same
  ``areal_skill.json`` as the test-split report; the app's skill tile is tagged with the headline
  (test) split, so it must never show a validation-split score under that tag.
* INFERENCE <-> TOOLING: ``make clean-reports`` deletes every regenerable report the inference CLI
  writes (the physics backtest now has its own ``backtest_physics.json``) and never what training
  publishes with the model.
* DATA <-> APP: every weather ``source`` the schema can produce (incl. the new ``missing`` label of
  zero-filled gap hours) has a human label in the provenance table.
* DATA <-> TRAINING (F3-07 remainder, between the two owners): training made the paths it records
  portable; the stage-04 ``dataset_summary.json`` published next to metrics.json must not embed
  absolute local paths either.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data_pipeline import dataset as dataset_module
from src.data_pipeline.weather_schema import SOURCE_MISSING, SOURCES
from src.inference import backtest
from src.inference.field_ensemble import field_plan, member_rain
from src.inference.predict import DEFAULT_OUTPUT
from src.inference.settings import InferenceSettings
from src.training import areal_fields as fields
from src.utils.config import deep_merge
from tests.conftest import make_grid_graph

PROJECT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT / "app"
sys.path.insert(0, str(APP_DIR))
import components as ui  # noqa: E402
import metrics_view as mv  # noqa: E402

sys.path.remove(str(APP_DIR))

STAMPS = {"created_utc": "2024-01-01T00:00:00+00:00", "finalized_utc": "2024-01-01T00:05:00+00:00"}
ROW = {"pr_auc": 0.6, "roc_auc": 0.95, "f2": 0.5, "precision": 0.4, "recall": 0.6, "threshold": 0.4}
METRICS = {"status": "completed", "evaluation_split": "test", "pr_auc": 0.95, "pos_rate": 0.01,
           "dataset": {"years": {"train": [2021], "validation": [2022], "test": [2024]}}}  # as training writes it


def areal_report(split: str, ensemble: float) -> dict:
    """An ``ok`` areal_skill.json for the checkpoint ``STAMPS`` scored on ``split``."""
    return {"status": "ok", "split": split, "years": [2024 if split == "test" else 2022], "members": 8,
            "threshold": 0.4, "checkpoint": {"run_id": "r1", "epoch": 3, **STAMPS},
            "rows": {"exact_field": {**ROW, "pr_auc": 0.95}, "field_ensemble": {**ROW, "pr_auc": ensemble},
                     "uniform_areal": {**ROW, "pr_auc": ensemble + 0.01}}}


# --------------------------------------------------------------------------- X6 <-> X7: split of the areal score


@pytest.mark.unit
def test_areal_kpi_never_shows_a_validation_report_under_the_test_tag():
    """Regression: a --split validation report (same file) was shown as 'PR-AUC given areal rain · test'."""
    validation = areal_report("validation", 0.27)
    assert mv.areal_skill_summary(validation, METRICS, STAMPS) is None          # no test-split score known
    kpi = mv.pr_auc_kpi(METRICS, "gnn", exact_field=False, areal=mv.areal_skill_summary(validation, METRICS, STAMPS))
    assert kpi.value != "0.27" and kpi.delta == "not this mode's skill"           # the headline with the caveat

    stored = {**METRICS, "areal_skill": {"status": "ok", "split": "test", "members": 8,
                                         "field_ensemble_pr_auc": 0.61, "uniform_areal_pr_auc": 0.63,
                                         "exact_field_pr_auc": 0.95}}
    fallback = mv.areal_skill_summary(validation, stored, STAMPS)                # training's own test summary wins
    assert fallback["field_ensemble_pr_auc"] == 0.61 and fallback["source"] == "metrics"
    kpi = mv.pr_auc_kpi(stored, "gnn", exact_field=False, areal=fallback)
    assert kpi.label == "PR-AUC given areal rain · test 2024" and kpi.value == "0.61"


@pytest.mark.unit
def test_areal_kpi_split_rule_accepts_matching_and_legacy_sources():
    test_report = areal_report("test", 0.62)
    summary = mv.areal_skill_summary(test_report, METRICS, STAMPS)
    assert summary["field_ensemble_pr_auc"] == 0.62 and summary["source"] == "areal_skill.json"
    legacy = {k: v for k, v in test_report.items() if k != "split"}             # no split recorded: accepted
    assert mv.areal_skill_summary(legacy, METRICS, STAMPS)["field_ensemble_pr_auc"] == 0.62
    validation_headline = {**METRICS, "evaluation_split": "validation"}          # no test split built
    assert mv.areal_skill_summary(test_report, validation_headline, STAMPS) is None
    wrong_stored = {**validation_headline, "areal_skill": {"split": "test", "field_ensemble_pr_auc": 0.6}}
    assert mv.areal_skill_summary(None, wrong_stored, STAMPS) is None
    assert mv.areal_skill_title(areal_report("validation", 0.27)) == \
        "Skill when only corridor-average rain is known (val 2022)"              # the About table stays labelled


# --------------------------------------------------------------------------- X5 <-> X6: one ensemble definition


@pytest.mark.unit
@pytest.mark.parametrize("override", [{}, {"inference": {"field_seed_stride": 13}, "rainfall_field": {"seed": 5}}])
def test_serving_ensemble_and_areal_skill_use_the_same_member_seeds(cfg, override):
    config = deep_merge(cfg, override)
    settings = InferenceSettings.from_config(config)
    serving = field_plan(config, settings, False, field_members=6, field_seed_offset=1)   # backtest ceiling / leads
    assert serving.seeds == tuple(fields.field_member_seed(config, k) for k in range(6))
    assert fields.label_seed(config) not in serving.seeds                               # never the label field
    exact = field_plan(config, settings, True)
    assert exact.seeds == (fields.label_seed(config),) and exact.exact


@pytest.mark.integration
def test_serving_member_rain_equals_the_areal_skill_junction_field(cfg, storm_series):
    """Member k of the serving ensemble downscales exactly like the areal-skill field of the same seed."""
    graph = make_grid_graph(4, 4)
    nodes = sorted(graph.nodes, key=str)
    lon = np.array([graph.nodes[n]["x"] for n in nodes])
    lat = np.array([graph.nodes[n]["y"] for n in nodes])
    areal = storm_series.to_numpy(dtype=np.float64)
    timestamps = storm_series.index
    plan = field_plan(cfg, InferenceSettings.from_config(cfg), False, field_members=3, field_seed_offset=1)
    served = member_rain(cfg, plan, areal, timestamps, lon, lat)
    inputs = fields.WeatherInputs(timestamps, areal, np.arange(len(areal)), lon, lat, "fingerprint")
    for k in range(3):
        skill_field = fields.junction_field(inputs, cfg, fields.field_member_seed(cfg, k))
        np.testing.assert_array_equal(served[k], skill_field)
    assert not np.array_equal(served[0], served[1])                                # members differ


# --------------------------------------------------------------------------- INFERENCE <-> TOOLING: report cleanup


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_clean_reports_removes_every_inference_report_and_keeps_the_published_model_outputs(tmp_path):
    """Regression: the physics backtest moved to backtest_physics.json, which clean-reports left behind."""
    reports = tmp_path / "reports"
    out = subprocess.run(["make", "-n", "-s", "clean-reports", f"REPORTS_DIR={reports}"], cwd=PROJECT,
                         capture_output=True, text=True, timeout=30, check=True).stdout
    deleted = {Path(word).name for word in out.split() if word.startswith(str(reports))}
    assert {DEFAULT_OUTPUT, backtest.report_name("gnn"), backtest.report_name("physics")} <= deleted
    assert backtest.report_name("physics") == "backtest_physics.json"
    assert not deleted & {"metrics.json", "areal_skill.json", "training_history.csv", "dataset_summary.json"}


# --------------------------------------------------------------------------- DATA <-> APP: weather source labels


@pytest.mark.unit
@pytest.mark.parametrize("source", [*SOURCES, SOURCE_MISSING, "custom"])
def test_every_weather_source_has_a_provenance_label(source):
    """weather_schema labels reindexed gap hours ``missing``; the provenance table must not show a raw key."""
    assert source in ui.WEATHER_SOURCES and ui.WEATHER_SOURCES[source] != source
    meta = {"sources": {source: 3}, "start": "2024-01-01T00:00", "end": "2024-01-01T02:00", "rows": 3}
    scenario = SimpleNamespace(source=source, description="d", rain_field_note="exact training field")
    rows = dict(ui.provenance_rows({"source": "synthetic_grid"}, meta, scenario, None, 0))
    assert rows["Weather record"].startswith(ui.WEATHER_SOURCES[source])


# --------------------------------------------------------------------------- DATA <-> TRAINING: portable paths


@pytest.mark.unit
def test_dataset_summary_json_records_portable_split_paths(cfg, tmp_path):
    """Regression (F3-07): dataset_summary.json embedded absolute paths (user name, home layout)."""
    inside = PROJECT / "data" / "processed" / "val_dataset.pt"               # never created, only named
    summary = {"config_hash": "abc", "splits": {"train": {"n_windows": 3, "path": str(tmp_path / "train_dataset.pt")},
                                                "val": {"n_windows": 2, "path": str(inside)},
                                                "test": {"n_windows": 1}}}
    dataset_module._write_summary(cfg, summary)
    written = json.loads((Path(cfg["paths"]["reports_dir"]) / "dataset_summary.json").read_text())
    assert written["splits"]["train"]["path"] == "train_dataset.pt"            # outside the project: file name
    assert written["splits"]["val"]["path"] == "data/processed/val_dataset.pt"  # inside: project-relative
    assert written["splits"]["test"] == {"n_windows": 1} and written["config_hash"] == "abc"
    assert summary["splits"]["train"]["path"] == str(tmp_path / "train_dataset.pt")  # caller's dict untouched
    assert str(tmp_path) not in json.dumps(written) and str(PROJECT) not in json.dumps(written)
