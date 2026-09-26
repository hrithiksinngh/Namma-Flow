"""Tests for the training stage's outputs: checkpoint format v2 and its safe loader, RNG
capture, the graph-free baselines, the ``metrics.json`` structure (validation / test /
baselines / strongest baseline) and the ``src/training/train.py`` CLI.

Fixtures (tiny train/val/test datasets built with the real stage-04 code) live in
:mod:`tests.training_fixtures`.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data_pipeline.dataset import FloodSequenceDataset
from src.training import train as train_cli
from src.training.baseline import BaselineConfig, logistic_baseline, run_baselines
from src.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_KEYS,
    CheckpointError,
    capture_rng_state,
    is_finalized,
    load_checkpoint,
    model_from_checkpoint,
    restore_rng_state,
    to_builtin,
    validate_checkpoint,
)
from src.training.evaluation import HEADLINE_KEYS, strongest_baseline
from src.training.reports import publish_final
from src.training.trainer import TrainResult, train
from tests.training_fixtures import (  # noqa: F401 - pytest fixtures
    N_NODES,
    N_TEST,
    N_VAL,
    SEQ_LEN,
    WARMUP,
    log,
    modified_payload,
    shared_data,
    tcfg,
    warnings_text,
    with_,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "training" / "train.py"
SCORED = SEQ_LEN - WARMUP
CONTRACT_KEYS = {
    "model_state", "architecture", "model_config", "feature_names", "static_feature_names", "rolling_windows_h",
    "lookback_hours", "seq_len", "warmup_steps", "scaler", "graph_signature", "graph_attributes_sha256", "node_ids",
    "calibration", "temperature", "threshold", "metrics", "epoch", "best_score", "history", "optimizer_state",
    "scheduler_state", "rng_state", "config_hash", "dataset_config_hash", "created_utc",
}
MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


@pytest.fixture(scope="module")
def trained(shared_data, tmp_path_factory) -> TrainResult:
    """One finished run shared by the read-only tests of this module."""
    from src.utils.config import load_config
    from tests.training_fixtures import DATA, PATH_NAMES, SMALL

    root = tmp_path_factory.mktemp("trained")
    paths = {**{key: str(root / name) for key, name in PATH_NAMES.items()}, **shared_data}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NAMMA_FLOW_OFFLINE", "1")
        cfg = load_config(overrides={"project": {"offline": True}, **DATA, **SMALL, "paths": paths})
        return train(cfg)


# --------------------------------------------------------------------------- checkpoint format v2 (X3)


@pytest.mark.integration
def test_published_checkpoint_is_format_v2_and_finalized(trained, shared_data):
    raw = torch.load(trained.best_checkpoint, map_location="cpu", weights_only=True)  # the safe loader works
    assert CONTRACT_KEYS <= set(raw) and set(CHECKPOINT_KEYS) <= set(raw)
    assert raw["format_version"] == CHECKPOINT_FORMAT_VERSION == 2 and raw["kind"] == "best"
    assert is_finalized(raw) and raw["finalized_utc"]
    cal = raw["calibration"]
    assert set(cal) == {"method", "slope", "intercept"} and cal["method"] == "platt"
    assert 0.05 <= cal["slope"] <= 20.0 and -20.0 <= cal["intercept"] <= 20.0
    assert raw["temperature"] == pytest.approx(1.0 / cal["slope"])
    assert raw["threshold"] == pytest.approx(trained.metrics["threshold"])
    digest = FloodSequenceDataset(shared_data["train_dataset"]).graph_attributes_sha256
    assert raw["graph_attributes_sha256"] == digest is not None
    assert raw["seq_len"] == SEQ_LEN and raw["warmup_steps"] == WARMUP and len(raw["node_ids"]) == N_NODES


@pytest.mark.integration
def test_candidate_and_last_checkpoints_are_never_finalized(trained):
    for name in ("best_candidate.pt", "last.pt"):
        ckpt = load_checkpoint(trained.best_checkpoint.parent / name)
        assert not is_finalized(ckpt) and ckpt["calibration"]["method"] == "none"
        assert ckpt["format_version"] == 2 and ckpt["run_id"] == load_checkpoint(trained.best_checkpoint)["run_id"]


@pytest.mark.unit
def test_checkpoint_key_list_is_the_contract():
    assert set(CHECKPOINT_KEYS) == CONTRACT_KEYS | {"format_version"}


def _minimal_checkpoint(version: int = 1) -> dict:
    ckpt = {"format_version": version, "model_state": {}, "model_config": {}, "feature_names": ["a"],
            "rolling_windows_h": [3], "lookback_hours": 3, "seq_len": 4, "warmup_steps": 1, "scaler": {},
            "graph_signature": {}, "node_ids": [1], "temperature": 1.0, "threshold": 0.5}
    if version >= 2:
        ckpt["calibration"] = {"method": "platt", "slope": 0.5, "intercept": -2.0}
    return ckpt


@pytest.mark.unit
@pytest.mark.parametrize(
    "version, change, match",
    [
        (1, {"format_version": 3}, "format_version"),
        (1, {"format_version": True}, "format_version"),
        (1, {"threshold": 1.5}, "threshold"),
        (1, {"temperature": 0.0}, "temperature"),
        (1, {"feature_names": []}, "feature_names"),
        (1, {"model_state": None}, "model_state"),
        (1, {"warmup_steps": 4}, "warmup_steps"),
        (1, {"format_version": 2}, r"missing keys \['calibration'\]"),
        (2, {"calibration": {"method": "platt", "slope": -1.0, "intercept": 0.0}}, "slope"),
        (2, {"calibration": {"method": "isotonic", "slope": 1.0, "intercept": 0.0}}, "method"),
        (2, {"calibration": {"method": "platt", "slope": 1.0}}, "intercept"),
        (2, {"calibration": 0.5}, "calibration"),
    ],
)
def test_validate_checkpoint_rejects_bad_values(version, change, match):
    with pytest.raises(CheckpointError, match=match):
        validate_checkpoint({**_minimal_checkpoint(version), **change})


@pytest.mark.unit
def test_validate_checkpoint_accepts_v1_and_v2_and_hand_made_checkpoints(log):
    assert validate_checkpoint(_minimal_checkpoint(1))["temperature"] == 1.0  # v1: no calibration dict needed
    assert validate_checkpoint(_minimal_checkpoint(2))["calibration"]["slope"] == 0.5
    ckpt = _minimal_checkpoint(1)
    del ckpt["format_version"]
    assert validate_checkpoint(ckpt)["threshold"] == 0.5
    assert "no format_version" in warnings_text(log)
    with pytest.raises(CheckpointError, match="missing keys"):
        validate_checkpoint({"model_state": {}})
    with pytest.raises(CheckpointError, match="not a checkpoint dict"):
        validate_checkpoint([1, 2])


@pytest.mark.unit
def test_load_checkpoint_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="train a model first"):
        load_checkpoint(tmp_path / "missing.pt")
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a torch file")
    with pytest.raises(CheckpointError, match="corrupt"):
        load_checkpoint(garbage)
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"format_version": 1}, incomplete)
    with pytest.raises(CheckpointError, match="missing keys"):
        load_checkpoint(incomplete)


class _Payload:
    """Pickles to a call of ``open(marker, "w")``: loading it with the full unpickler creates the file."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


@pytest.mark.unit
def test_load_checkpoint_never_runs_the_full_unpickler(tmp_path):
    """R3-03: a file the safe loader rejects must not be retried with weights_only=False."""
    marker = tmp_path / "code_ran"
    evil = tmp_path / "evil.pt"
    torch.save({**_minimal_checkpoint(1), "metrics": _Payload(marker)}, evil)
    with pytest.raises(CheckpointError, match="not a Namma-Flow checkpoint"):
        load_checkpoint(evil)
    assert not marker.exists()
    with pytest.raises(CheckpointError):
        model_from_checkpoint(evil)
    assert not marker.exists()


@pytest.mark.integration
def test_model_from_checkpoint_rejects_mismatched_weights(trained):
    ckpt = load_checkpoint(trained.best_checkpoint)
    broken = {**ckpt, "model_config": {**ckpt["model_config"], "hidden_dim": 5}}
    with pytest.raises(CheckpointError, match="do not fit"):
        model_from_checkpoint(broken)


@pytest.mark.unit
def test_to_builtin_and_rng_round_trip(tmp_path):
    converted = to_builtin({"a": np.float32(1.5), "b": np.arange(3), "c": torch.tensor([1.0, float("nan")]),
                            1: (np.bool_(True), float("inf")), "p": tmp_path, "o": object})
    assert converted["a"] == 1.5 and converted["b"] == [0, 1, 2] and converted["c"] == [1.0, None]
    assert converted["1"] == [True, None] and isinstance(converted["o"], str)
    assert converted["p"] == tmp_path.name  # F3-07: paths outside the project keep only their name
    state = capture_rng_state("cpu")
    assert "mps" not in state  # a CPU run never touches the MPS generator
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    expected = (np.random.rand(), torch.rand(1).item(), random.random())
    restore_rng_state(torch.load(path, weights_only=True))
    assert (np.random.rand(), torch.rand(1).item(), random.random()) == expected


@pytest.mark.unit
@pytest.mark.skipif(not MPS, reason="needs an Apple-silicon MPS device")
def test_mps_rng_state_is_captured_and_restored(tmp_path):
    """R3-08: dropout masks on device=mps continue (not restart) after a resume."""
    torch.manual_seed(0)
    torch.rand(3, device="mps")
    state = capture_rng_state("mps")
    assert "mps" in state and "mps" in capture_rng_state()
    expected = torch.rand(5, device="mps").cpu()
    torch.save(state, tmp_path / "rng.pt")
    torch.manual_seed(0)
    restore_rng_state(torch.load(tmp_path / "rng.pt", weights_only=True))
    torch.testing.assert_close(torch.rand(5, device="mps").cpu(), expected)


@pytest.mark.unit
def test_restore_rng_state_tolerates_missing_or_bad_state(log):
    restore_rng_state(None)
    restore_rng_state({"numpy": {"name": "MT19937"}})
    assert "RNG state" in warnings_text(log)


@pytest.mark.unit
def test_publish_final_writes_both_files_or_only_the_checkpoint(tmp_path, log):
    ckpt, metrics = {"a": torch.ones(2)}, {"pr_auc": 0.5}
    assert publish_final(ckpt, tmp_path / "ckpt" / "best.pt", metrics, tmp_path / "reports" / "metrics.json")
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text()) == metrics
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    assert not publish_final(ckpt, tmp_path / "ckpt" / "best.pt", metrics, blocker / "metrics.json")
    assert "Could not write" in warnings_text(log)
    leftovers = [p.name for p in (tmp_path / "ckpt").iterdir() if p.name != "best.pt"]
    assert leftovers == []  # staged temporary files are cleaned up


# --------------------------------------------------------------------------- metrics.json (X3, R3-05, R2-02)


@pytest.mark.integration
def test_metrics_report_test_split_at_validation_calibration_and_threshold(trained):
    m = trained.metrics
    assert m["evaluation_split"] == "test" and "held-out TEST" in m["note"]
    for split in ("validation", "test"):
        assert set(HEADLINE_KEYS) <= set(m[split])
        assert m[split]["threshold"] == pytest.approx(m["threshold"])  # ONE threshold, chosen on validation
    for key in HEADLINE_KEYS:
        assert m[key] == m["test"][key], key  # the headline block IS the test split
    assert m["test"]["n"] == N_TEST * SCORED * N_NODES and m["validation"]["n"] == N_VAL * SCORED * N_NODES
    assert m["threshold_selected_on"] == "validation" and m["calibration"]["fitted_on"] == "validation"
    assert m["temperature"] == pytest.approx(1.0 / m["calibration"]["slope"])
    json.dumps(m, allow_nan=False)
    on_disk = json.loads((Path(trained.best_checkpoint).parents[1] / "reports" / "metrics.json").read_text())
    assert on_disk == json.loads(json.dumps(m))  # metrics.json and best.pt's metrics are the same record
    assert on_disk["pr_auc"] == pytest.approx(m["pr_auc"]) and on_disk["evaluation_split"] == "test"


@pytest.mark.integration
def test_platt_calibration_matches_the_validation_base_rate(trained):
    """R2-02: with an intercept the calibrated mean probability equals the base rate on the fitting split."""
    val = trained.metrics["validation"]
    assert val["mean_probability"] == pytest.approx(val["pos_rate"], rel=0.1)
    assert val["uncalibrated"]["log_loss"] > val["log_loss"]  # the fit improves the NLL it optimises
    assert trained.metrics["calibration"]["log_loss_calibrated"] <= trained.metrics["calibration"][
        "log_loss_uncalibrated"]


@pytest.mark.integration
def test_metrics_carry_both_baselines_and_the_strongest(trained):
    m = trained.metrics
    assert set(m["baselines"]) == {"logreg", "hist_gbdt"}
    for name, result in m["baselines"].items():
        assert result["status"] == "ok", name
        assert result["validation"]["pr_auc"] is not None and result["test"]["pr_auc"] is not None
        assert result["test"]["threshold"] == pytest.approx(result["threshold"])  # chosen on validation
    best = max(m["baselines"], key=lambda k: m["baselines"][k]["test"]["pr_auc"])
    strongest = m["strongest_baseline"]
    assert strongest["name"] == best and strongest["split"] == "test"
    assert strongest["pr_auc"] == pytest.approx(m["baselines"][best]["test"]["pr_auc"])
    assert strongest["delta_pr_auc"] == pytest.approx(m["pr_auc"] - strongest["pr_auc"])
    legacy = m["baseline_logreg"]
    assert legacy["status"] == "ok" and legacy["split"] == "test"
    assert legacy["pr_auc"] == pytest.approx(m["baselines"]["logreg"]["test"]["pr_auc"])


@pytest.mark.integration
def test_metrics_record_dataset_hashes_and_protocol(trained, shared_data):
    data = trained.metrics["dataset"]
    # F3-07: never an absolute path (the shared data lie outside the project -> file names only)
    assert data["test_path"] == Path(shared_data["test_dataset"]).name and data["test_windows"] == N_TEST
    assert data["train_path"] == Path(shared_data["train_dataset"]).name
    assert data["hashes"]["graph_attributes_sha256"] is not None
    assert set(data["hashes"]) >= {"dataset_config_hash", "graph_signature", "weather_fingerprint", "build_id"}
    training = trained.metrics["training"]
    assert training["micro_batch_size"] == 4 and training["optimizer_steps"] == 4


@pytest.mark.integration
def test_without_a_test_dataset_the_headline_is_validation(tcfg, tmp_path, log):
    result = train(with_(tcfg, paths={"test_dataset": str(tmp_path / "absent.pt")}), epochs=1)
    m = result.metrics
    assert m["evaluation_split"] == "validation" and m["test"] is None and "optimistic" in m["note"]
    assert m["n"] == m["validation"]["n"] == N_VAL * SCORED * N_NODES
    assert m["baselines"]["logreg"]["test"] is None and m["strongest_baseline"]["split"] == "validation"
    assert "No held-out test dataset" in warnings_text(log)


@pytest.mark.integration
def test_test_dataset_with_the_wrong_split_label_is_reported(tcfg, shared_data, tmp_path, log):
    path = modified_payload(shared_data["test_dataset"], tmp_path / "test.pt", split="val")
    train(with_(tcfg, paths={"test_dataset": path}), epochs=1)
    assert "holds split 'val'" in warnings_text(log)


# --------------------------------------------------------------------------- baselines (R2-03)


@pytest.mark.integration
def test_run_baselines_fits_both_models_and_scores_both_splits(shared_data):
    train_ds, val_ds, test_ds = (FloodSequenceDataset(shared_data[f"{s}_dataset"]) for s in ("train", "val", "test"))
    config = BaselineConfig(max_samples=4000, max_train_windows=20, seed=0, gbdt_params={"max_iter": 30})
    out = run_baselines(train_ds, val_ds, test_ds, config)
    assert set(out) == {"logreg", "hist_gbdt"}
    gbdt = out["hist_gbdt"]
    assert gbdt["status"] == "ok" and gbdt["n_iter"] >= 1 and gbdt["calibration"]["method"] == "platt"
    assert gbdt["validation"]["roc_auc"] > 0.95 and gbdt["test"]["roc_auc"] > 0.95  # learns the rain rule
    assert gbdt["test"]["n"] == N_TEST * SCORED * N_NODES
    assert set(out["logreg"]["coefficients"]) == set(train_ds.feature_names)
    json.dumps(out, allow_nan=False)
    only_val = run_baselines(train_ds, val_ds, None, config)
    assert only_val["hist_gbdt"]["test"] is None


@pytest.mark.integration
def test_one_failing_baseline_does_not_hide_the_other(shared_data, monkeypatch, log):
    from src.training import baseline

    def broken(*args):
        raise ValueError("gbdt exploded")

    monkeypatch.setitem(baseline._FITTERS, "hist_gbdt", broken)
    train_ds, val_ds = (FloodSequenceDataset(shared_data[f"{s}_dataset"]) for s in ("train", "val"))
    out = run_baselines(train_ds, val_ds, None, BaselineConfig(max_samples=2000, max_train_windows=10))
    assert out["hist_gbdt"]["status"] == "failed" and "gbdt exploded" in out["hist_gbdt"]["reason"]
    assert out["logreg"]["status"] == "ok"


@pytest.mark.integration
def test_logistic_baseline_keeps_its_flat_result(shared_data):
    train_ds = FloodSequenceDataset(shared_data["train_dataset"])
    val_ds = FloodSequenceDataset(shared_data["val_dataset"])
    out = logistic_baseline(train_ds, val_ds, max_samples=2000, max_train_windows=10, seed=0)
    assert out["status"] == "ok" and out["n_train_windows"] == 10
    assert out["n_train_samples"] <= 2000 and out["n_train_pos"] > 0
    assert set(out["coefficients"]) == set(train_ds.feature_names)
    assert out["coefficients"]["precip_mm_h"] > 0  # more rain -> more flooding
    assert out["roc_auc"] > 0.9
    json.dumps(out)
    with pytest.raises(ValueError, match="max_samples"):
        logistic_baseline(train_ds, val_ds, max_samples=1)


@pytest.mark.unit
def test_baseline_config_validation():
    with pytest.raises(ValueError, match="max_train_windows"):
        BaselineConfig(max_train_windows=1)
    with pytest.raises(ValueError, match="Unknown baseline"):
        BaselineConfig(models=("svm",))


@pytest.mark.unit
def test_strongest_baseline_selection():
    baselines = {"logreg": {"status": "ok", "test": {"pr_auc": 0.4}, "validation": {"pr_auc": 0.6}},
                 "hist_gbdt": {"status": "ok", "test": {"pr_auc": 0.9}, "validation": {"pr_auc": 0.5}},
                 "other": {"status": "failed"}}
    assert strongest_baseline(baselines, "test", 0.7) == {"name": "hist_gbdt", "pr_auc": 0.9, "split": "test",
                                                          "gnn_pr_auc": 0.7, "delta_pr_auc": pytest.approx(-0.2)}
    assert strongest_baseline(baselines, "validation", None)["name"] == "logreg"
    assert strongest_baseline({}, "test", 0.5) == {"name": None, "pr_auc": None, "split": "test", "gnn_pr_auc": 0.5,
                                                   "delta_pr_auc": None}


# --------------------------------------------------------------------------- CLI


def _yaml_config(cfg: dict, path: Path) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.write_text(yaml.safe_dump(clean), encoding="utf-8")
    return str(path)


@pytest.mark.e2e
def test_cli_trains_and_prints_validation_and_test_side_by_side(tcfg, tmp_path, capsys):
    config = _yaml_config(tcfg, tmp_path / "config.yaml")
    code = train_cli.main(["--config", config, "--epochs", "1", "--device", "cpu", "--windows-per-epoch", "4",
                           "--max-val-windows", "0", "--num-threads", "2", "--offline"])
    out = capsys.readouterr().out
    assert code == 0
    for text in ("PR-AUC", "ROC-AUC", "GNN", "LogReg", "HistGBDT", "Headline      : test split", "platt",
                 "years 2023", "strongest baseline", "best.pt"):
        assert text in out, text
    header = next(line for line in out.splitlines() if line.strip().startswith("Metric"))
    assert header.split()[1:] == ["val", "test"] * 3
    assert (tmp_path / "checkpoints" / "best.pt").exists()


@pytest.mark.e2e
def test_cli_resume_flag(tcfg, tmp_path, capsys):
    config = _yaml_config(tcfg, tmp_path / "config.yaml")
    assert train_cli.main(["--config", config, "--epochs", "1"]) == 0
    assert train_cli.main(["--config", config, "--epochs", "2", "--resume"]) == 0
    assert "2 run of 2" in capsys.readouterr().out


def test_cli_handled_failures_exit_one(tcfg, tmp_path, capsys, monkeypatch):
    assert train_cli.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert "ERROR" in capsys.readouterr().err
    config = _yaml_config(with_(tcfg, model={"node_in_dim": 10, "edge_dim": 3}), tmp_path / "bad.yaml")
    assert train_cli.main(["--config", config]) == 1
    assert "edge_dim=3" in capsys.readouterr().err

    def unexpected(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(train_cli, "train", unexpected)
    assert train_cli.main(["--config", config]) == 1
    assert "unexpected KeyError" in capsys.readouterr().err


def test_cli_interrupt_exits_130(tcfg, tmp_path, capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train_cli, "train", interrupted)
    assert train_cli.main(["--config", _yaml_config(tcfg, tmp_path / "c.yaml")]) == 130
    assert "--resume" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize("argv", [["--epochs", "0"], ["--windows-per-epoch", "-1"], ["--epochs", "x"],
                                  ["--device", "tpu"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        train_cli.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.e2e
def test_script_runs_from_the_project_root():
    proc = subprocess.run([sys.executable, str(SCRIPT.relative_to(PROJECT_ROOT)), "--help"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "--resume" in proc.stdout


@pytest.mark.unit
def test_format_report_without_baselines_or_test_split():
    result = TrainResult(Path("best.pt"), Path("last.pt"), 1, {
        "status": "completed", "epochs_run": 1, "epochs_target": 1, "monitor": "val_loss", "pr_auc": None,
        "evaluation_split": "validation", "validation": {"pr_auc": None, "n": 10, "n_pos": 0}, "test": None,
        "baselines": {"logreg": {"status": "unavailable", "reason": "no floods"}, "hist_gbdt": {"status": "disabled"}},
        "dataset": {"stale": True}, "model": {"architecture": "gatv2_gru", "n_parameters": 10}, "training": {}}, [])
    text = train_cli.format_report(result)
    assert "unavailable no floods" in text and "HistGBDT baseline: disabled" in text
    assert "stale" in text and "n/a" in text and "none - the headline metrics fall back to validation" in text
    assert "Areal-only    : unavailable (not computed)" in text
