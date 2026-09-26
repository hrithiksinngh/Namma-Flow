"""Tests for the areal-only skill evaluation (X6): ``src.training.areal_skill`` / ``areal_fields``.

The tiny train/val/test build of :mod:`tests.training_fixtures` (4x4 grid, labels = junction
rain >= 10 mm/h) is scored with a small trained model. The key properties: the label field is
regenerated exactly from the full weather record; ensemble members never reuse the label seed
(and a member WITH the label seed reproduces the exact-field row); the report schema, the
finalisation integration (published with best.pt) and every "unavailable" path.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data_pipeline.dataset import FloodSequenceDataset, make_loader
from src.data_pipeline.weather import save_weather_csv
from src.hydrology.simulator import physics_flood_probability
from src.training import areal_fields as fields
from src.training import areal_skill as areal
from src.training import finalize as finalize_module
from src.training.checkpoint import load_checkpoint, model_from_checkpoint
from src.training.evaluation import evaluate
from src.training.trainer import TrainResult, train
from src.utils.config import deep_merge, load_config
from tests.training_fixtures import (  # noqa: F401 - pytest fixtures
    DATA,
    FLOOD_MM_H,
    N_NODES,
    N_TEST,
    PATH_NAMES,
    SEQ_LEN,
    SMALL,
    WARMUP,
    log,
    modified_payload,
    shared_data,
    warnings_text,
    with_,
)

MEMBERS = 3
AREAL_ON = {"training": {"areal_skill": {"enabled": True, "members": MEMBERS}}}
METRIC_KEYS = {"pr_auc", "roc_auc", "f2", "recall", "precision", "threshold", "n", "n_pos", "brier", "ece"}


def _config(root: Path, shared: dict, **sections) -> dict:
    paths = {**{key: str(root / name) for key, name in PATH_NAMES.items()}, **shared}
    return load_config(overrides=deep_merge({"project": {"offline": True}, **DATA, **SMALL, "paths": paths},
                                            deep_merge(AREAL_ON, sections)))


@pytest.fixture(scope="module")
def trained(shared_data, tmp_path_factory) -> tuple[dict, TrainResult]:
    """One finished run WITH the areal-skill evaluation in its finalisation."""
    cfg = _config(tmp_path_factory.mktemp("areal"), shared_data)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NAMMA_FLOW_OFFLINE", "1")
        return cfg, train(cfg)


@pytest.fixture
def acfg(trained, tmp_path) -> dict:
    """The trained run's config with a per-test reports directory."""
    return with_(trained[0], paths={"reports_dir": str(tmp_path / "reports")})


@pytest.fixture(scope="module")
def report(trained) -> dict:
    cfg, result = trained
    return areal.compute_areal_skill(cfg, result.best_checkpoint, members=MEMBERS)


def _payload(shared_data: dict, split: str = "test") -> dict:
    return torch.load(shared_data[f"{split}_dataset"], weights_only=True)


# --------------------------------------------------------------------------- field generation


@pytest.mark.unit
def test_member_seeds_follow_the_x5_scheme_and_never_reuse_the_label_seed(trained):
    cfg = trained[0]
    base = cfg["rainfall_field"]["seed"]
    assert [fields.field_member_seed(cfg, k) for k in range(3)] == [base + (1 + k) * 7919 for k in range(3)]
    assert fields.field_member_seed(cfg, 0, offset=0) == fields.label_seed(cfg) == base
    custom = with_(cfg, inference={"field_seed_stride": 5}, rainfall_field={"seed": 10})
    assert fields.field_member_seed(custom, 2) == 10 + 3 * 5
    assert fields.member_config(cfg, 99)["rainfall_field"]["seed"] == 99 and cfg["rainfall_field"]["seed"] == base
    with pytest.raises(ValueError, match="member"):
        fields.field_member_seed(cfg, -1)
    with pytest.raises(fields.ArealSkillError, match="field_seed_stride"):
        fields.field_member_seed(with_(cfg, inference={"field_seed_stride": 0}), 0)


@pytest.mark.integration
def test_label_field_is_reproduced_from_the_full_record(trained, shared_data):
    cfg = trained[0]
    payload = _payload(shared_data)
    inputs = fields.load_weather_inputs(cfg, payload)
    label = fields.junction_field(inputs, cfg, fields.label_seed(cfg))[inputs.rows]
    np.testing.assert_array_equal(label, payload["rain"].numpy())  # exactly the build's field
    other = fields.junction_field(inputs, cfg, fields.field_member_seed(cfg, 0))[inputs.rows]
    wet = inputs.areal[inputs.rows] > 0
    assert not np.allclose(other[wet], label[wet])  # another realisation ...
    np.testing.assert_allclose(other.mean(axis=1), label.mean(axis=1), rtol=1e-4, atol=1e-5)  # ... same areal rain


@pytest.mark.unit
def test_scored_values_selects_the_scored_steps_of_every_window():
    values = np.arange(20 * 2).reshape(20, 2)
    out = fields.scored_values(values, np.array([0, 5]), seq_len=4, warmup_steps=1)
    assert out.tolist() == [2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17]


@pytest.mark.integration
def test_scored_labels_match_the_models_evaluation(trained, shared_data):
    payload = _payload(shared_data)
    y, _, _ = evaluate(model_from_checkpoint(trained[1].best_checkpoint),
                       make_loader(FloodSequenceDataset(payload), 4, False), "cpu")
    gathered = fields.scored_values(payload["labels"].numpy(), payload["window_starts"].numpy(), SEQ_LEN, WARMUP)
    assert gathered.size == y.size == N_TEST * (SEQ_LEN - WARMUP) * N_NODES and gathered.sum() == y.sum()


@pytest.mark.integration
def test_spin_up_window_reproduces_the_full_record_teacher(trained, shared_data):
    """The physics row re-runs the teacher only from 30 days before the split: same depths as the full record."""
    cfg = trained[0]
    payload = _payload(shared_data)
    inputs = fields.load_weather_inputs(cfg, payload)
    arrays = fields.physics_graph(cfg, payload)
    field = fields.junction_field(inputs, cfg, fields.field_member_seed(cfg, 0))
    span = fields.physics_span(inputs)
    assert span.start == max(0, int(inputs.rows.min()) - fields.PHYSICS_SPINUP_H) and span.start > 0
    _, window_depth = fields.teacher_run(arrays, cfg, field[span], inputs.rows - span.start)
    _, full_depth = fields.teacher_run(arrays, cfg, field, inputs.rows)
    np.testing.assert_allclose(window_depth, full_depth, atol=1e-6)


@pytest.mark.unit
def test_uniform_rain_is_the_node_mean(shared_data):
    payload = _payload(shared_data)
    uniform = areal._uniform_rain(payload)
    assert uniform.shape == tuple(payload["rain"].shape) and np.all(uniform == uniform[:, :1])
    np.testing.assert_allclose(uniform[:, 0], payload["rain"].numpy().mean(axis=1), rtol=1e-6)


# --------------------------------------------------------------------------- report


@pytest.mark.integration
def test_report_schema_and_rows(report, trained):
    cfg, result = trained
    ckpt = load_checkpoint(result.best_checkpoint)
    assert report["status"] == "ok" and report["split"] == "test" and report["years"] == [2023]
    assert report["members"] == MEMBERS and report["threshold"] == pytest.approx(ckpt["threshold"])
    # best.pt carries a validation-chosen areal threshold, so the extra row follows the core rows
    assert tuple(report["rows"]) == (*areal.ROWS, areal.AREAL_THRESHOLD_ROW)
    for name in [n for n in areal.ROWS if not n.startswith("physics")]:
        assert METRIC_KEYS <= set(report["rows"][name]), name
        assert report["rows"][name]["threshold"] == pytest.approx(ckpt["threshold"])
    ensemble = report["rows"]["physics_field_ensemble"]
    assert ensemble["status"] == "ok" and ensemble["members"] == MEMBERS and ensemble["threshold"] == 0.5
    assert METRIC_KEYS <= set(ensemble)
    physics = report["rows"]["physics_other_field"]
    assert physics["status"] == "ok" and physics["threshold"] == 0.5 and METRIC_KEYS <= set(physics)
    assert physics["spinup_h"] == fields.PHYSICS_SPINUP_H
    # the fixture labels come from a stand-in teacher (rain >= 10 mm/h), not module D: flagged, not fatal
    assert physics["teacher_label_mismatches"] > 0 and report["checks"]["teacher_labels_reproduced"] is False
    assert report["checkpoint"] == {k: ckpt.get(k) for k in ("run_id", "epoch", "created_utc", "finalized_utc",
                                                             "published_id")}
    assert report["field_seeds"] == [fields.field_member_seed(cfg, k) for k in range(MEMBERS)]
    assert fields.label_seed(cfg) not in report["field_seeds"] and len(report["member_pr_auc"]) == MEMBERS
    assert report["checks"]["label_field_reproduced"] and report["checks"]["label_field_max_abs_diff_mm_h"] == 0.0
    assert report["checks"]["weather_fingerprint_matches"] is True
    assert report["n"] == N_TEST * (SEQ_LEN - WARMUP) * N_NODES and "emulation of the teacher" in report["note"]
    json.dumps(report, allow_nan=False)


@pytest.mark.integration
def test_exact_field_row_is_the_headline_test_metric(report, trained):
    metrics = trained[1].metrics
    exact = report["rows"]["exact_field"]
    for key in ("pr_auc", "roc_auc", "f2", "recall", "precision", "brier"):
        assert exact[key] == pytest.approx(metrics["test"][key], rel=1e-6, abs=1e-9), key  # batch 1 vs 4 rounding
    assert report["rows"]["field_ensemble"]["pr_auc"] != pytest.approx(exact["pr_auc"])


@pytest.mark.integration
def test_one_member_ensemble_equals_the_single_field(trained):
    cfg, result = trained
    single = areal.compute_areal_skill(cfg, result.best_checkpoint, members=1)
    rows = single["rows"]
    assert rows["field_ensemble"]["pr_auc"] == pytest.approx(rows["single_other_field"]["pr_auc"])
    assert single["member_pr_auc"] == pytest.approx([rows["single_other_field"]["pr_auc"]])


def _teacher_module(threshold_m: float) -> types.ModuleType:
    """Stand-in for module D consistent with the fixture labels: depth crosses the threshold at 10 mm/h."""
    module = types.ModuleType("src.hydrology.simulator")

    def simulate_labels(graph, rain, cfg):
        rain = np.asarray(rain, dtype=np.float32)
        return (rain >= FLOOD_MM_H).astype(np.uint8), (rain * threshold_m / FLOOD_MM_H).astype(np.float32)

    module.simulate_labels = simulate_labels
    module.physics_flood_probability = physics_flood_probability
    return module


@pytest.mark.integration
def test_a_member_with_the_label_seed_reproduces_the_exact_row_and_the_teacher(trained, shared_data, monkeypatch):
    """Consistency of the whole procedure: with the label seed instead of an independent one, the 'other field'
    rows collapse onto the exact field, and the re-simulated teacher reproduces its own labels."""
    cfg, result = trained
    threshold = float(_payload(shared_data)["flood_threshold_m"])
    monkeypatch.setitem(sys.modules, "src.hydrology.simulator", _teacher_module(threshold))
    monkeypatch.setattr(fields, "field_member_seed", lambda c, k, offset=1: fields.label_seed(c))
    same = areal.compute_areal_skill(cfg, result.best_checkpoint, members=2)
    rows = same["rows"]
    for key in ("pr_auc", "f2", "recall", "precision"):
        assert rows["single_other_field"][key] == pytest.approx(rows["exact_field"][key]), key
        assert rows["field_ensemble"][key] == pytest.approx(rows["exact_field"][key]), key
    assert rows["physics_other_field"]["pr_auc"] == pytest.approx(1.0)
    assert rows["physics_other_field"]["teacher_label_mismatches"] == 0 and same["checks"]["teacher_labels_reproduced"]
    assert rows["physics_other_field"]["recall"] == 1.0 and rows["physics_other_field"]["precision"] == 1.0
    # every member uses the label seed, so the teacher ensemble is the teacher itself
    assert rows["physics_field_ensemble"]["pr_auc"] == pytest.approx(1.0)


@pytest.mark.integration
def test_results_do_not_depend_on_the_evaluation_batch(trained, report):
    cfg, result = trained
    batched = areal.compute_areal_skill(cfg, result.best_checkpoint, members=MEMBERS, batch_size=4)
    for name in areal.ROWS:
        assert batched["rows"][name]["pr_auc"] == pytest.approx(report["rows"][name]["pr_auc"], rel=1e-5), name


@pytest.mark.integration
def test_validation_split_can_be_scored(trained):
    cfg, result = trained
    val = areal.compute_areal_skill(cfg, result.best_checkpoint, members=1, split="validation")
    assert val["status"] == "ok" and val["split"] == "validation" and val["years"] == [2022]


# --------------------------------------------------------------------------- unavailable paths


def _unavailable(cfg: dict, checkpoint=None, **kwargs) -> str:
    out = areal.compute_areal_skill(cfg, checkpoint, members=1, **kwargs)
    assert out["status"] == "unavailable" and out["rows"] == {} and out["members"] == 1
    json.dumps(out, allow_nan=False)
    return out["reason"]


@pytest.mark.integration
def test_missing_or_unpublished_model_is_unavailable(acfg, trained, tmp_path, log):
    assert "No usable published model" in _unavailable(with_(acfg, paths={"checkpoint_dir": str(tmp_path / "x")}))
    candidate = trained[1].best_checkpoint.parent / "best_candidate.pt"
    assert "not a finalized" in _unavailable(acfg, candidate)
    assert "Areal-only skill is unavailable" in warnings_text(log)


@pytest.mark.integration
def test_missing_dataset_or_weather_is_unavailable(acfg, trained, tmp_path):
    best = trained[1].best_checkpoint
    assert "No test dataset" in _unavailable(with_(acfg, paths={"test_dataset": str(tmp_path / "none.pt")}), best)
    index = pd.date_range("2021-10-01", "2022-03-31 23:00", freq="h", tz="Asia/Kolkata", name="timestamp")
    short = pd.DataFrame({"precipitation_mm": 0.0, "is_imputed": False, "source": "open_meteo"}, index=index)
    save_weather_csv(short, tmp_path / "short.csv")
    cfg = with_(acfg, paths={"weather_file": str(tmp_path / "short.csv")},
                weather={"start_date": "2021-10-01", "end_date": "2022-03-31"})
    assert "lacks" in _unavailable(cfg, best)


@pytest.mark.integration
def test_stale_rain_field_settings_are_unavailable(acfg, trained):
    reason = _unavailable(with_(acfg, rainfall_field={"n_cells": 5}), trained[1].best_checkpoint)
    assert "cannot be reproduced" in reason and "rebuild" in reason


@pytest.mark.integration
def test_model_trained_on_other_inputs_is_unavailable(acfg, trained, shared_data, tmp_path):
    payload = _payload(shared_data)
    path = modified_payload(shared_data["test_dataset"], tmp_path / "test.pt",
                            feature_names=list(reversed(payload["feature_names"])))
    reason = _unavailable(with_(acfg, paths={"test_dataset": path}), trained[1].best_checkpoint)
    assert "trained on other inputs" in reason and "feature_names" in reason


@pytest.mark.integration
def test_physics_row_failure_keeps_the_report(acfg, trained, tmp_path, log):
    out = areal.compute_areal_skill(with_(acfg, paths={"graph_file": str(tmp_path / "none.graphml")}),
                                    trained[1].best_checkpoint, members=1)
    assert out["status"] == "ok" and out["rows"]["physics_other_field"]["status"] == "unavailable"
    assert out["rows"]["physics_field_ensemble"]["status"] == "unavailable"
    assert out["rows"]["field_ensemble"]["pr_auc"] is not None
    assert "physics reference row is unavailable" in warnings_text(log)


@pytest.mark.unit
@pytest.mark.parametrize("kwargs, match", [({"members": 0}, "members"), ({"members": True}, "members"),
                                           ({"split": "train"}, "split"), ({"batch_size": 0}, "batch_size")])
def test_invalid_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        areal.compute_areal_skill({}, None, **{"members": 2, **kwargs})


@pytest.mark.unit
def test_summary_and_writing(tmp_path):
    rows = {name: {"pr_auc": value} for name, value in zip(areal.ROWS, (0.9, 0.6, 0.4, 0.55, 0.5))}
    ok = {"status": "ok", "split": "test", "members": 8, "rows": rows}
    assert areal.summary_of(ok) == {"status": "ok", "split": "test", "members": 8, "exact_field_pr_auc": 0.9,
                                    "field_ensemble_pr_auc": 0.6, "uniform_areal_pr_auc": 0.55,
                                    "areal_threshold": None, "field_ensemble_f2_at_areal_threshold": None,
                                    "physics_field_ensemble_pr_auc": None}
    missing = areal.unavailable_report("no weather", members=4)
    assert areal.summary_of(missing)["reason"] == "no weather" and areal.summary_of(missing)["members"] == 4
    target = areal.write_areal_skill({"paths": {"reports_dir": str(tmp_path)}}, {**ok, "x": float("nan")})
    assert target == tmp_path / areal.REPORT_NAME and json.loads(target.read_text())["x"] is None


# --------------------------------------------------------------------------- finalisation (published with best.pt)


@pytest.mark.integration
def test_finalisation_publishes_the_report_with_the_model(trained):
    cfg, result = trained
    ckpt = load_checkpoint(result.best_checkpoint)
    on_disk = json.loads((Path(cfg["paths"]["reports_dir"]) / areal.REPORT_NAME).read_text())
    assert on_disk["status"] == "ok" and on_disk["members"] == MEMBERS
    assert on_disk["checkpoint"]["published_id"] == ckpt["published_id"]
    assert on_disk["checkpoint"]["finalized_utc"] == ckpt["finalized_utc"]
    summary = result.metrics["areal_skill"]
    assert summary == ckpt["metrics"]["areal_skill"] == areal.summary_of(on_disk)
    assert summary["exact_field_pr_auc"] == pytest.approx(result.metrics["pr_auc"])
    metrics_json = json.loads((Path(cfg["paths"]["reports_dir"]) / "metrics.json").read_text())
    assert metrics_json["areal_skill"] == summary


@pytest.mark.integration
def test_training_report_prints_the_areal_skill(trained):
    from src.training import train as train_cli

    text = train_cli.format_report(trained[1])
    line = next(row for row in text.splitlines() if row.strip().startswith("Areal-only"))
    assert "with only corridor-average rain" in line and f"{MEMBERS}-member" in line
    assert "emulation of the teacher" in trained[1].metrics["note"]


@pytest.mark.integration
def test_a_failing_evaluation_never_fails_training(acfg, tmp_path, monkeypatch, log):
    def broken(*args, **kwargs):
        raise MemoryError("out of RAM")

    monkeypatch.setattr(finalize_module.areal_module, "compute_areal_skill", broken)
    cfg = with_(acfg, paths={"checkpoint_dir": str(tmp_path / "ckpt")})
    result = train(cfg, epochs=1)
    on_disk = json.loads((Path(cfg["paths"]["reports_dir"]) / areal.REPORT_NAME).read_text())
    assert on_disk["status"] == "unavailable" and "out of RAM" in on_disk["reason"]
    assert result.metrics["areal_skill"]["status"] == "unavailable"
    assert on_disk["checkpoint"]["run_id"] == load_checkpoint(result.best_checkpoint)["run_id"]
    assert "areal-only skill evaluation failed" in warnings_text(log)


@pytest.mark.integration
def test_disabled_evaluation_publishes_an_unavailable_report(acfg, tmp_path):
    cfg = with_(acfg, paths={"checkpoint_dir": str(tmp_path / "ckpt")}, training={"areal_skill": {"enabled": False}})
    result = train(cfg, epochs=1)
    on_disk = json.loads((Path(cfg["paths"]["reports_dir"]) / areal.REPORT_NAME).read_text())
    assert on_disk["status"] == "unavailable" and "enabled is false" in on_disk["reason"]
    assert result.metrics["areal_skill"]["field_ensemble_pr_auc"] is None


@pytest.mark.integration
def test_without_a_test_split_the_validation_split_is_scored(acfg, tmp_path):
    cfg = with_(acfg, paths={"checkpoint_dir": str(tmp_path / "ckpt"), "test_dataset": str(tmp_path / "none.pt")},
                training={"areal_skill": {"members": 1}})
    result = train(cfg, epochs=1)
    assert result.metrics["areal_skill"]["split"] == "validation"
    assert result.metrics["areal_skill"]["status"] == "ok"


@pytest.mark.unit
def test_settings_validate_the_areal_skill_section(acfg):
    from src.training.settings import TrainingSettings
    from src.utils.config import ConfigError

    settings = TrainingSettings.from_config(acfg)
    assert settings.areal_skill_enabled is True and settings.areal_skill_members == MEMBERS
    defaults = TrainingSettings.from_config({"training": {}})
    assert defaults.areal_skill_enabled is True and defaults.areal_skill_members == 8
    for bad, match in (({"members": 0}, "members"), ({"enabled": "yes"}, "enabled")):
        with pytest.raises(ConfigError, match=f"training.areal_skill.{match}"):
            TrainingSettings.from_config(with_(acfg, training={"areal_skill": bad}))


# --------------------------------------------------------------------------- CLI


def _yaml(cfg: dict, path: Path) -> str:
    path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}), encoding="utf-8")
    return str(path)


@pytest.mark.e2e
def test_cli_writes_the_report_and_prints_the_rows(acfg, tmp_path, capsys, monkeypatch):
    threads = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    code = areal.main(["--config", _yaml(acfg, tmp_path / "c.yaml"), "--members", "2", "--threads", "1",
                       "--offline"])
    out = capsys.readouterr().out
    assert code == 0 and threads == [1]
    for text in ("exact_field", "field_ensemble", "single_other_field", "uniform_areal", "physics_other_field",
                 "PR-AUC", "Recall", "Precision", "2 rain-field members", "areal_skill.json"):
        assert text in out, text
    written = json.loads((Path(acfg["paths"]["reports_dir"]) / areal.REPORT_NAME).read_text())
    assert written["members"] == 2 and written["status"] == "ok"


@pytest.mark.e2e
def test_cli_unavailable_exits_one_without_writing(acfg, tmp_path, capsys):
    cfg = with_(acfg, paths={"checkpoint_dir": str(tmp_path / "none")})
    assert areal.main(["--config", _yaml(cfg, tmp_path / "c.yaml")]) == 1
    assert "unavailable" in capsys.readouterr().err
    assert not (Path(acfg["paths"]["reports_dir"]) / areal.REPORT_NAME).exists()
    assert areal.main(["--config", str(tmp_path / "missing.yaml")]) == 1


@pytest.mark.unit
@pytest.mark.parametrize("argv", [["--members", "0"], ["--split", "train"], ["--threads", "x"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        areal.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.integration
def test_finalisation_stores_a_validation_chosen_areal_threshold(trained):
    """best.pt carries the alert threshold for ensemble-averaged probabilities, chosen on validation."""
    cfg, result = trained
    ckpt = load_checkpoint(result.best_checkpoint)
    info = ckpt.get("areal_threshold_info")
    assert info is not None and info["split"] in ("val", "validation") and info["members"] == MEMBERS
    assert 0.0 < ckpt["areal_threshold"] < 1.0 and ckpt["areal_threshold"] == pytest.approx(info["threshold"])
    on_disk = json.loads((Path(cfg["paths"]["reports_dir"]) / areal.REPORT_NAME).read_text())
    assert on_disk["areal_threshold"] == pytest.approx(ckpt["areal_threshold"])
    row = on_disk["rows"][areal.AREAL_THRESHOLD_ROW]
    assert row["threshold"] == pytest.approx(ckpt["areal_threshold"])
    assert result.metrics["areal_skill"]["areal_threshold"] == pytest.approx(ckpt["areal_threshold"])


@pytest.mark.integration
def test_areal_serving_threshold_is_the_best_validation_cut(trained, shared_data):
    from src.training.metrics import best_threshold

    cfg, result = trained
    ckpt = load_checkpoint(result.best_checkpoint)
    info = areal.areal_serving_threshold(cfg, ckpt, _payload(shared_data, "val"), members=2, metric="f2")
    assert 0.0 < info["threshold"] < 1.0 and info["members"] == 2 and info["metric"] == "f2"
    assert 0.0 <= info["score"] <= 1.0
    with pytest.raises(ValueError):
        areal.areal_serving_threshold(cfg, ckpt, _payload(shared_data, "val"), members=0)
