"""Regression tests of the final review round for the training package.

* F1-05 - splits from different dataset builds are refused (an interrupted rebuild);
* F2-02 - resume across a changed validation set re-scores the best candidate; across a
  dataset rebuild it is refused; the final calibration always uses the full validation split;
* F2-03 - ``training_history.csv`` is published with ``best.pt`` (per-epoch rows are staged);
* F2-04 - a re-finalised ``best.pt`` gets a new ``created_utc`` / ``published_id``;
* F3-07 - no absolute paths in the checkpoints or ``metrics.json``.

Fixtures live in :mod:`tests.training_fixtures`.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import pytest
import torch

from src.training import finalize as finalize_module
from src.training import trainer
from src.training.checkpoint import load_checkpoint, published_id, save_checkpoint
from src.training.reports import HISTORY_PARTIAL_NAME, portable_path
from src.training.splits import BUILD_KEYS
from src.training.trainer import TrainingError, train
from src.utils.config import project_root
from tests.training_fixtures import (  # noqa: F401 - pytest fixtures
    N_VAL,
    dirs,
    log,
    modified_payload,
    shared_data,
    tcfg,
    warnings_text,
    with_,
)

FLAT = {"val_loss": 1.0, "pr_auc": 0.5, "roc_auc": 0.5, "brier": 0.1, "f2": 0.2, "f2_threshold": 0.3,
        "alert_threshold": 0.3, "n_val": 1, "n_val_pos": 1}


def _reports(cfg: dict) -> Path:
    return Path(cfg["paths"]["reports_dir"])


def _history_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- F1-05 mixed builds


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("key", BUILD_KEYS)
def test_splits_from_different_builds_are_refused(tcfg, shared_data, tmp_path, key, split):
    path = modified_payload(shared_data[f"{split}_dataset"], tmp_path / f"{split}.pt", **{key: "other-build"})
    with pytest.raises(TrainingError, match=f"train and {split} datasets come from different dataset builds.*{key}"):
        train(with_(tcfg, paths={f"{split}_dataset": path}), epochs=1)


def test_interrupted_rebuild_with_old_labels_is_refused(tcfg, shared_data, tmp_path):
    """The reviewer's case: a val file of the old build (other config hash, zeroed labels) next to a new train."""
    payload = torch.load(shared_data["val_dataset"], weights_only=True)
    val = modified_payload(shared_data["val_dataset"], tmp_path / "val.pt", config_hash="old-config",
                           weather_fingerprint="old-weather", build_id="old-build",
                           labels=torch.zeros_like(payload["labels"]))
    with pytest.raises(TrainingError, match="different dataset builds"):
        train(with_(tcfg, paths={"val_dataset": val}), epochs=1)


# --------------------------------------------------------------------------- F2-02 resume identity


def test_resume_with_a_changed_validation_subset_rescores_the_best(tcfg, monkeypatch, log):
    """2 epochs scored on 3 val windows (0.9), resumed on all windows: the candidate re-scores at 0.5, so the
    new epochs (0.6, 0.7) improve instead of being judged against the incomparable 0.9."""
    full_scores = iter([0.5, 0.6, 0.7])

    def fake_validate(model, loader, *args):
        value = 0.9 if len(loader.dataset) == 3 else next(full_scores)
        return {**FLAT, "pr_auc": value}

    monkeypatch.setattr(trainer, "_validate", fake_validate)
    first = train(tcfg, epochs=2, max_val_windows=3)
    assert first.metrics["best_score"] == pytest.approx(0.9)
    log.clear()
    resumed = train(tcfg, epochs=4, resume=True)
    assert [r["improved"] for r in resumed.history[2:]] == [True, True]
    assert resumed.metrics["best_epoch"] == 4 and resumed.metrics["best_score"] == pytest.approx(0.7)
    text = warnings_text(log)
    assert "re-scored on the current validation set" in text and "max_val_windows=3" in text


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_rescoring_without_a_usable_candidate_restarts_the_best_score(tcfg, monkeypatch, log, damage):
    monkeypatch.setattr(trainer, "_validate", lambda model, loader, *args: {**FLAT, "pr_auc": 0.9 if len(
        loader.dataset) == 3 else 0.1})
    first = train(tcfg, epochs=1, max_val_windows=3)
    candidate = first.best_checkpoint.parent / "best_candidate.pt"
    if damage == "missing":
        candidate.unlink()
    else:
        candidate.write_bytes(b"not a checkpoint")
    resumed = train(tcfg, epochs=2, resume=True)
    assert resumed.history[1]["improved"] is True  # 0.1 on the full set beats "no comparable best score"
    text = warnings_text(log)
    assert "re-scored" in text and ("is unreadable" in text) == (damage == "corrupt")


def test_resume_on_the_same_validation_set_keeps_the_best_score(tcfg, monkeypatch, log):
    monkeypatch.setattr(trainer, "_validate", lambda *args: dict(FLAT))
    train(tcfg, epochs=1)
    log.clear()
    resumed = train(tcfg, epochs=2, resume=True)
    assert resumed.metrics["best_score"] == pytest.approx(0.5) and resumed.history[1]["improved"] is False
    assert "re-scored" not in warnings_text(log)


def _rebuilt(shared_data: dict, root: Path, **changes) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {f"{s}_dataset": modified_payload(shared_data[f"{s}_dataset"], root / f"{s}.pt", **changes)
            for s in ("train", "val", "test")}


def test_resume_across_a_dataset_rebuild_is_refused(tcfg, shared_data, tmp_path):
    train(tcfg, epochs=1)
    paths = _rebuilt(shared_data, tmp_path / "rebuilt", build_id="new-build")
    with pytest.raises(TrainingError, match="different dataset build.*build_id"):
        train(with_(tcfg, paths=paths), epochs=2, resume=True)
    relabelled = _rebuilt(shared_data, tmp_path / "relabelled", build_id="relabel", config_hash="new-hydrology")
    with pytest.raises(TrainingError, match="config_hash"):
        train(with_(tcfg, paths=relabelled), epochs=2, resume=True)


def test_resume_of_an_old_checkpoint_uses_the_stored_hashes(tcfg, log):
    """A last.pt written before dataset_identity / val_set existed: the dataset hash is still compared and the
    validation set cannot be verified, so the best candidate is re-scored."""
    first = train(tcfg, epochs=1)
    last = load_checkpoint(first.last_checkpoint)
    legacy = {k: v for k, v in last.items() if k not in ("dataset_identity", "val_set")}
    save_checkpoint(legacy, first.last_checkpoint)
    log.clear()
    resumed = train(tcfg, epochs=2, resume=True)
    assert len(resumed.history) == 2 and "unknown (older checkpoint)" in warnings_text(log)
    save_checkpoint({**legacy, "dataset_config_hash": "stale"}, first.last_checkpoint)
    with pytest.raises(TrainingError, match="different dataset build"):
        train(tcfg, epochs=3, resume=True)


def test_checkpoints_store_the_dataset_and_validation_identity(tcfg, shared_data):
    result = train(tcfg, epochs=1, max_val_windows=3)
    last = load_checkpoint(result.last_checkpoint)
    payload = torch.load(shared_data["train_dataset"], weights_only=True)
    assert last["dataset_identity"] == {key: payload.get(key) for key in BUILD_KEYS}
    assert last["val_set"]["max_val_windows"] == 3 and last["val_set"]["windows"] == 3
    assert len(last["val_set"]["digest"]) == 16


# --------------------------------------------------------------------------- F2-03 staged history


def test_training_history_is_published_only_with_the_model(tcfg, monkeypatch):
    first = train(tcfg, epochs=2)
    first_id = load_checkpoint(first.best_checkpoint)["run_id"]
    published = _reports(tcfg) / "training_history.csv"
    before = published.read_text()
    assert len(_history_rows(published)) == 2 and not (_reports(tcfg) / HISTORY_PARTIAL_NAME).exists()
    real, calls = trainer._train_epoch, {"n": 0}

    def fail_second(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TrainingError("stop")
        return real(*args)

    monkeypatch.setattr(trainer, "_train_epoch", fail_second)
    with pytest.raises(TrainingError):
        train(with_(tcfg, model={"learning_rate": 0.05}), epochs=3)
    assert published.read_text() == before  # the aborted run's curve never replaces the published one
    staged = _history_rows(_reports(tcfg) / HISTORY_PARTIAL_NAME)
    assert len(staged) == 1 and float(staged[0]["lr"]) == pytest.approx(0.05)
    assert load_checkpoint(first.best_checkpoint)["run_id"] == first_id  # ... next to the model it belongs to
    monkeypatch.setattr(trainer, "_train_epoch", real)
    second = train(with_(tcfg, model={"learning_rate": 0.05}), epochs=3)
    rows = _history_rows(published)
    assert len(rows) == 3 and float(rows[0]["lr"]) == pytest.approx(0.05)
    assert not (_reports(tcfg) / HISTORY_PARTIAL_NAME).exists()
    assert len(load_checkpoint(second.best_checkpoint)["history"]) == 3  # the whole run, not up to the best epoch


def test_crash_in_finalisation_keeps_the_published_history(tcfg, monkeypatch):
    train(tcfg, epochs=2)
    before = (_reports(tcfg) / "training_history.csv").read_text()

    def crash(*args, **kwargs):
        raise TrainingError("finalisation crashed")

    monkeypatch.setattr(finalize_module, "_baselines", crash)
    with pytest.raises(TrainingError, match="finalisation crashed"):
        train(tcfg, epochs=1)
    assert (_reports(tcfg) / "training_history.csv").read_text() == before


# --------------------------------------------------------------------------- F2-04 published identity


def test_refinalising_gives_the_model_a_new_identity(tcfg, monkeypatch):
    clock = itertools.count()
    monkeypatch.setattr(finalize_module, "now_utc", lambda: f"2026-01-01T00:00:{next(clock):02d}+00:00")
    first = load_checkpoint(train(tcfg, epochs=2).best_checkpoint)
    assert first["created_utc"] == first["finalized_utc"] and first["candidate_created_utc"]
    again = load_checkpoint(train(with_(tcfg, training={"calibration": "temperature", "threshold_metric": "csi"}),
                                  epochs=2, resume=True).best_checkpoint)
    assert again["run_id"] == first["run_id"] and again["candidate_created_utc"] == first["candidate_created_utc"]
    assert again["created_utc"] != first["created_utc"] and again["created_utc"] == again["finalized_utc"]
    assert again["published_id"] != first["published_id"]
    assert published_id(again) == again["published_id"]  # deterministic over the stored content


@pytest.mark.unit
def test_published_id_tracks_weights_calibration_and_threshold():
    ckpt = {"model_state": {"w": torch.ones(3)}, "calibration": {"method": "platt", "slope": 2.0, "intercept": -1.0},
            "threshold": 0.3, "run_id": "r", "epoch": 1}
    base = published_id(ckpt)
    assert base == published_id(dict(ckpt)) and len(base) == 16
    assert published_id({**ckpt, "threshold": 0.4}) != base
    assert published_id({**ckpt, "model_state": {"w": torch.tensor([1.0, 1.0, 1.5])}}) != base
    assert published_id({**ckpt, "calibration": {**ckpt["calibration"], "slope": 3.0}}) != base


# --------------------------------------------------------------------------- F3-07 portable paths


@pytest.mark.unit
def test_portable_path_is_project_relative_or_a_file_name(tmp_path):
    assert portable_path(project_root() / "data" / "processed" / "test_dataset.pt") == "data/processed/test_dataset.pt"
    assert portable_path(tmp_path / "x" / "val.pt") == "val.pt"
    assert portable_path(None) is None and portable_path("") is None


@pytest.mark.integration
def test_published_files_contain_no_absolute_paths(tcfg, shared_data):
    result = train(tcfg, epochs=1)
    secrets = [str(Path(shared_data["train_dataset"]).parent), str(Path.home())]
    files = [result.best_checkpoint, result.last_checkpoint, result.best_checkpoint.parent / "best_candidate.pt",
             _reports(tcfg) / "metrics.json"]
    for path in files:
        raw = path.read_bytes()
        for secret in secrets:
            assert secret.encode() not in raw, (path.name, secret)
    dataset = result.metrics["dataset"]
    assert (dataset["train_path"], dataset["val_path"], dataset["test_path"]) == ("train.pt", "val.pt", "test.pt")
    assert load_checkpoint(result.best_checkpoint)["metrics"]["dataset"]["test_path"] == "test.pt"


def test_full_validation_is_used_for_the_final_reports(tcfg):
    result = train(dirs(tcfg, Path(tcfg["paths"]["reports_dir"]).parent / "full"), epochs=1, max_val_windows=4)
    assert result.metrics["dataset"]["val_windows"] == N_VAL
    assert result.metrics["dataset"]["val_windows_per_epoch"] == 4
