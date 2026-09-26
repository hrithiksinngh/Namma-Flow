"""Tests for the training loop: ``src.training.trainer`` (+ settings, splits, finalisation).

Fixtures (tiny train/val/test datasets built with the real stage-04 code) live in
:mod:`tests.training_fixtures`; checkpoint / baseline / report / CLI tests are in
``tests/test_training_outputs.py``.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler

from src.data_pipeline.dataset import DatasetError, FloodSequenceDataset, collate_windows, make_loader
from src.training import finalize as finalize_module
from src.training import trainer
from src.training.calibration import apply_calibration
from src.training.checkpoint import load_checkpoint, model_from_checkpoint
from src.training.metrics import ranking_metrics
from src.training.settings import Monitor, TrainingSettings
from src.training.trainer import (
    HISTORY_FIELDS,
    TrainingError,
    TrainResult,
    epoch_sampler,
    evaluate,
    load_datasets,
    train,
    window_sampling_weights,
)
from src.utils.config import ConfigError
from tests.training_fixtures import (  # noqa: F401 - pytest fixtures
    DATA,
    N_TEST,
    N_TRAIN,
    N_VAL,
    N_NODES,
    SEQ_LEN,
    SMALL,
    WARMUP,
    dirs,
    fake_simulator,
    log,
    modified_payload,
    shared_data,
    tcfg,
    warnings_text,
    with_,
    write_inputs,
)

SCORED = SEQ_LEN - WARMUP
FLAT = {"val_loss": 1.0, "pr_auc": 0.5, "roc_auc": 0.5, "brier": 0.1, "f2": 0.2, "f2_threshold": 0.3,
        "n_val": 1, "n_val_pos": 1}


def _ckpt_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"])


# --------------------------------------------------------------------------- end to end


@pytest.mark.integration
def test_train_end_to_end_writes_checkpoints_and_reports(tcfg, tmp_path):
    result = train(tcfg)
    assert isinstance(result, TrainResult)
    assert result.best_checkpoint.exists() and result.last_checkpoint.exists()
    assert (_ckpt_dir(tcfg) / "best_candidate.pt").exists()
    assert len(result.history) == 2 and 1 <= result.best_epoch <= 2
    m = result.metrics
    assert m["status"] == "completed" and m["epochs_run"] == 2 and m["evaluation_split"] == "test"
    assert m["n"] == N_TEST * SCORED * N_NODES and m["validation"]["n"] == N_VAL * SCORED * N_NODES
    assert 0.0 <= m["pr_auc"] <= 1.0 and 0.0 <= m["roc_auc"] <= 1.0
    assert 0.05 <= m["temperature"] <= 20.0 and 0.0 < m["threshold"] < 1.0
    assert m["baseline_logreg"]["status"] == "ok" and m["baseline_logreg"]["pr_auc"] is not None
    assert len(m["per_step"]) == SCORED and m["per_step"][0]["step"] == WARMUP
    assert m["reliability"]["n_bins"] == 10 and m["dataset"]["stale"] is False
    assert m["dataset"]["windows"] == {"train": N_TRAIN, "validation": N_VAL, "test": N_TEST}
    assert m["dataset"]["years"] == {"train": [2021], "validation": [2022], "test": [2023]}
    reports = tmp_path / "reports"
    assert json.loads((reports / "metrics.json").read_text())["threshold"] == pytest.approx(m["threshold"])
    with (reports / "training_history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0].keys()) == HISTORY_FIELDS and len(rows) == 2


@pytest.mark.integration
def test_model_learns_the_rain_rule(tcfg):
    result = train(with_(tcfg, model={"epochs": 4}, training={"windows_per_epoch": 16}))
    assert result.metrics["roc_auc"] > 0.85
    assert result.metrics["pr_auc"] > 3 * result.metrics["pos_rate"]  # far better than chance (AP = base rate)


@pytest.mark.integration
def test_best_checkpoint_rebuilds_the_model_and_reproduces_the_test_metrics(tcfg, shared_data):
    result = train(tcfg)
    ckpt = load_checkpoint(result.best_checkpoint)
    model = model_from_checkpoint(ckpt)
    assert not model.training
    test_ds = FloodSequenceDataset(shared_data["test_dataset"])
    y, logits, _ = evaluate(model, make_loader(test_ds, 4, False), "cpu")
    probs = apply_calibration(logits, ckpt["calibration"])
    assert ranking_metrics(y, probs)["pr_auc"] == pytest.approx(ckpt["metrics"]["pr_auc"], abs=1e-6)
    assert ranking_metrics(y, probs)["brier"] == pytest.approx(ckpt["metrics"]["test"]["brier"], rel=1e-5)
    assert model_from_checkpoint(result.best_checkpoint).state_dict().keys() == model.state_dict().keys()


@pytest.mark.integration
def test_training_is_deterministic(tcfg, tmp_path):
    first = train(dirs(tcfg, tmp_path / "a"))
    second = train(dirs(tcfg, tmp_path / "b"))
    for a, b in zip(first.history, second.history):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-6)
        assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-6)


# --------------------------------------------------------------------------- gradient accumulation


def _batch_items(tcfg, n: int = 4) -> tuple[FloodSequenceDataset, list[dict]]:
    train_ds, _ = load_datasets(tcfg)
    return train_ds, [train_ds[i] for i in range(n)]


def _gradients(tcfg, micro: int, items, train_ds) -> list[torch.Tensor]:
    torch.manual_seed(0)
    model = trainer._build_model(with_(tcfg, model={"dropout": 0.0}), train_ds)
    criterion = trainer.build_loss({"name": "focal"}, pos_rate=train_ds.pos_rate)
    settings = TrainingSettings.from_config(with_(tcfg, training={"micro_batch_size": micro}))
    collate = partial(collate_windows, edge_index=train_ds.edge_index, edge_attr=train_ds.edge_attr,
                      num_nodes=train_ds.num_nodes)
    model.train()
    loss, finite = trainer._accumulate(model, items, criterion, settings, torch.device("cpu"), collate)
    assert finite
    return [loss] + [p.grad.detach().clone() for p in model.parameters()]


@pytest.mark.unit
@pytest.mark.parametrize("micro", [1, 2, 3])
def test_micro_batches_accumulate_the_full_batch_gradient(tcfg, micro):
    train_ds, items = _batch_items(tcfg)
    full = _gradients(tcfg, 4, items, train_ds)
    accumulated = _gradients(tcfg, micro, items, train_ds)
    assert accumulated[0] == pytest.approx(full[0], rel=1e-5)  # the batch loss
    for a, b in zip(accumulated[1:], full[1:]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


@pytest.mark.integration
def test_training_with_micro_batches_matches_full_batches(tcfg, tmp_path):
    base = with_(tcfg, model={"dropout": 0.0})
    full = train(dirs(with_(base, training={"micro_batch_size": 4}), tmp_path / "full"))
    micro = train(dirs(with_(base, training={"micro_batch_size": 2}), tmp_path / "micro"))
    for a, b in zip(full.history, micro.history):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-4)
        assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-4)
        assert a["n_steps"] == b["n_steps"] == 2  # 8 windows / batch 4: optimizer steps are unchanged
    assert micro.metrics["training"]["micro_batch_size"] == 2


# --------------------------------------------------------------------------- resume


@pytest.mark.integration
def test_resume_reproduces_an_uninterrupted_run(tcfg, tmp_path):
    straight = train(dirs(tcfg, tmp_path / "straight"), epochs=3)
    split_cfg = dirs(tcfg, tmp_path / "split")
    train(split_cfg, epochs=1)
    resumed = train(split_cfg, epochs=3, resume=True)
    assert [r["epoch"] for r in resumed.history] == [1, 2, 3]
    for a, b in zip(straight.history, resumed.history):
        for key in ("train_loss", "val_loss", "lr"):
            assert a[key] == pytest.approx(b[key], rel=1e-6), key
    assert resumed.metrics["pr_auc"] == pytest.approx(straight.metrics["pr_auc"], rel=1e-6)


@pytest.mark.integration
def test_resume_via_config_flag_after_completion_only_finalises(tcfg, log):
    train(tcfg)
    log.clear()
    again = train(with_(tcfg, training={"resume": True}))
    assert len(again.history) == 2
    assert "Resumed from" in "\n".join(r.getMessage() for r in log.records)
    assert not any(r.getMessage().startswith("Epoch ") for r in log.records)


@pytest.mark.integration
def test_resume_without_last_checkpoint_starts_fresh(tcfg, log):
    result = train(tcfg, epochs=1, resume=True)
    assert len(result.history) == 1
    assert "starting a fresh run" in warnings_text(log)


@pytest.mark.integration
def test_resume_rejects_an_incompatible_checkpoint(tcfg):
    train(tcfg, epochs=1)
    with pytest.raises(TrainingError, match="hidden_dim"):
        train(with_(tcfg, model={"hidden_dim": 12}), epochs=2, resume=True)


def test_resume_with_a_changed_monitor_resets_the_best_score(tcfg, log):
    train(tcfg, epochs=1)
    result = train(with_(tcfg, training={"early_stopping_metric": "roc_auc"}), epochs=2, resume=True)
    assert result.metrics["monitor"] == "roc_auc" and result.history[1]["improved"] is True
    assert "best score, early stopping and the LR plateau counter are reset" in warnings_text(log)


def test_resume_with_a_changed_monitor_mode_resets_the_lr_scheduler(tcfg, monkeypatch):
    """R3-04: pr_auc (max) -> val_loss (min); a steadily falling val loss must never cut the LR."""
    train(tcfg, epochs=1)
    losses = iter([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    monkeypatch.setattr(trainer, "_validate", lambda *args: {**FLAT, "val_loss": next(losses)})
    cfg = with_(tcfg, training={"early_stopping_metric": "val_loss", "lr_scheduler": {"patience": 1}})
    result = train(cfg, epochs=5, resume=True)
    assert [r["improved"] for r in result.history[1:]] == [True] * 4
    assert [r["lr"] for r in result.history] == pytest.approx([0.01] * 5)


def test_resume_applies_the_current_lr_schedule_settings(tcfg, monkeypatch, log):
    """R3-04: lr_scheduler.* changes are applied on resume (with a WARNING), not silently ignored."""
    monkeypatch.setattr(trainer, "_validate", lambda *args: dict(FLAT))
    train(with_(tcfg, training={"lr_scheduler": {"patience": 5}}), epochs=1)
    cfg = with_(tcfg, training={"lr_scheduler": {"patience": 0, "factor": 0.25}, "early_stopping_patience": 9})
    result = train(cfg, epochs=3, resume=True)
    # epoch 2 does not improve -> patience 0 cuts by the NEW factor 0.25 (old config: patience 5, no cut)
    assert [r["lr"] for r in result.history] == pytest.approx([0.01, 0.01, 0.0025])
    assert "lr_scheduler changed" in warnings_text(log)


def test_resume_after_early_stop_continues_only_with_more_patience(tcfg, monkeypatch):
    monkeypatch.setattr(trainer, "_validate", lambda *args: dict(FLAT))
    stopped = train(with_(tcfg, training={"early_stopping_patience": 1}), epochs=6)
    assert len(stopped.history) == 2 and stopped.metrics["status"] == "early_stopped"
    same = train(with_(tcfg, training={"early_stopping_patience": 1}), epochs=6, resume=True)
    assert len(same.history) == 2  # patience still exhausted: nothing to do but finalise
    more = train(with_(tcfg, training={"early_stopping_patience": 5}), epochs=4, resume=True)
    assert [r["epoch"] for r in more.history] == [1, 2, 3, 4] and more.metrics["status"] == "completed"


def test_resume_of_an_early_stopped_run_with_a_new_metric_trains_again(tcfg, monkeypatch):
    """R3-06: resetting the best score for a new metric must also clear early_stopped."""
    monkeypatch.setattr(trainer, "_validate", lambda *args: dict(FLAT))
    stopped = train(with_(tcfg, training={"early_stopping_patience": 1}), epochs=6)
    assert stopped.metrics["status"] == "early_stopped" and len(stopped.history) == 2
    cfg = with_(tcfg, training={"early_stopping_patience": 1, "early_stopping_metric": "val_loss"})
    resumed = train(cfg, epochs=6, resume=True)
    assert len(resumed.history) > 2 and resumed.history[2]["improved"] is True
    assert resumed.metrics["monitor"] == "val_loss" and resumed.metrics["best_score"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- interrupts (R3-01)


def _interrupt_on_call(monkeypatch, n: int):
    """Raise KeyboardInterrupt on the ``n``-th micro-batch loss; returns the real function."""
    real, calls = trainer._batch_loss, {"n": 0}

    def interrupting(*args):
        calls["n"] += 1
        if calls["n"] == n:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(trainer, "_batch_loss", interrupting)
    return real


def _adam_steps(ckpt: dict) -> list[float]:
    return sorted({float(s["step"]) for s in ckpt["optimizer_state"]["state"].values()})


def test_keyboard_interrupt_saves_last_checkpoint_and_can_resume(tcfg, monkeypatch):
    real, calls = trainer._train_epoch, {"n": 0}

    def interrupt_second(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(trainer, "_train_epoch", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        train(tcfg, epochs=3)
    last = load_checkpoint(_ckpt_dir(tcfg) / "last.pt")
    assert last["epoch"] == 1 and last["train_state"]["interrupted"] is True
    monkeypatch.setattr(trainer, "_train_epoch", real)
    resumed = train(tcfg, epochs=3, resume=True)
    assert [r["epoch"] for r in resumed.history] == [1, 2, 3]


def test_interrupt_mid_epoch_keeps_the_end_of_epoch_state(tcfg, tmp_path, monkeypatch):
    """R3-01: Ctrl-C after an optimizer step of epoch 2 must not save mid-epoch weights labelled epoch 1."""
    straight = train(dirs(tcfg, tmp_path / "straight"), epochs=3)
    cfg = dirs(tcfg, tmp_path / "split")
    real = _interrupt_on_call(monkeypatch, 4)  # 2 steps per epoch -> the 2nd step of epoch 2
    with pytest.raises(KeyboardInterrupt):
        train(cfg, epochs=3)
    last = load_checkpoint(_ckpt_dir(cfg) / "last.pt")
    assert last["epoch"] == 1 and last["train_state"]["interrupted"] is True
    assert _adam_steps(last) == [2.0]  # the state after epoch 1, not after 3 steps
    monkeypatch.setattr(trainer, "_batch_loss", real)
    resumed = train(cfg, epochs=3, resume=True)
    for a, b in zip(straight.history, resumed.history):
        for key in ("train_loss", "val_loss", "lr"):
            assert a[key] == pytest.approx(b[key], rel=1e-6), key


def test_interrupt_in_the_first_epoch_replaces_a_stale_last_checkpoint(tcfg, tmp_path, monkeypatch):
    """R3-01: an epoch-0 interrupt leaves THIS run's start state, never an older run's last.pt."""
    cfg = dirs(tcfg, tmp_path / "run")
    old = train(cfg, epochs=2)
    old_id = load_checkpoint(old.last_checkpoint)["run_id"]
    real = _interrupt_on_call(monkeypatch, 2)  # after the first optimizer step of epoch 1
    with pytest.raises(KeyboardInterrupt):
        train(cfg, epochs=3)
    last = load_checkpoint(_ckpt_dir(cfg) / "last.pt")
    assert last["run_id"] != old_id and last["epoch"] == 0 and last["optimizer_state"]["state"] == {}
    monkeypatch.setattr(trainer, "_batch_loss", real)
    resumed = train(cfg, epochs=2, resume=True)
    for a, b in zip(old.history, resumed.history):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-6)


# --------------------------------------------------------------------------- publishing (R3-02, R4-02, R5-05)


def test_best_pt_is_published_only_by_the_finalisation(tcfg, monkeypatch):
    def crash(*args, **kwargs):
        raise TrainingError("finalisation crashed")

    monkeypatch.setattr(finalize_module, "_baselines", crash)
    with pytest.raises(TrainingError, match="finalisation crashed"):
        train(tcfg)
    ckpt_dir = _ckpt_dir(tcfg)
    assert not (ckpt_dir / "best.pt").exists() and not (Path(tcfg["paths"]["reports_dir"]) / "metrics.json").exists()
    candidate = load_checkpoint(ckpt_dir / "best_candidate.pt")
    assert candidate["kind"] == "best_candidate" and "finalized_utc" not in candidate
    assert candidate["calibration"]["method"] == "none" and candidate["temperature"] == 1.0


@pytest.mark.parametrize("failure", [KeyboardInterrupt, TrainingError])
def test_a_failed_new_run_keeps_the_previous_published_model(tcfg, monkeypatch, failure):
    first = train(tcfg)
    before = load_checkpoint(first.best_checkpoint)
    metrics_before = (Path(tcfg["paths"]["reports_dir"]) / "metrics.json").read_text()
    real, calls = trainer._train_epoch, {"n": 0}

    def fail_second(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise failure("stop")
        return real(*args)

    monkeypatch.setattr(trainer, "_train_epoch", fail_second)
    with pytest.raises(failure):
        train(tcfg, epochs=3)
    after = load_checkpoint(first.best_checkpoint)
    assert after["run_id"] == before["run_id"] and after["finalized_utc"] == before["finalized_utc"]
    assert (Path(tcfg["paths"]["reports_dir"]) / "metrics.json").read_text() == metrics_before
    assert load_checkpoint(_ckpt_dir(tcfg) / "best_candidate.pt")["run_id"] != before["run_id"]


def test_stale_candidate_from_another_run_is_ignored(tcfg, monkeypatch, log):
    train(tcfg, epochs=1)
    flat = {k: None for k in FLAT} | {"n_val": 1, "n_val_pos": 1}
    monkeypatch.setattr(trainer, "_validate", lambda *args: dict(flat))
    result = train(tcfg, epochs=1)  # fresh run whose only epoch never "improves"
    assert "belongs to an earlier run" in warnings_text(log)
    assert load_checkpoint(result.best_checkpoint)["run_id"] == load_checkpoint(result.last_checkpoint)["run_id"]


# --------------------------------------------------------------------------- failure handling


@pytest.mark.integration
def test_missing_datasets_are_built_first(cfg, monkeypatch, log):
    monkeypatch.setitem(sys.modules, "src.hydrology.simulator", fake_simulator())
    local = with_(cfg, **DATA, **SMALL)
    write_inputs(local)
    result = train(local, epochs=1)
    assert Path(local["paths"]["train_dataset"]).exists() and Path(local["paths"]["val_dataset"]).exists()
    assert result.metrics["epochs_run"] == 1 and result.metrics["evaluation_split"] == "test"
    assert "building the datasets" in warnings_text(log)


def test_dataset_build_failure_propagates(tcfg, tmp_path, monkeypatch):
    def fail(cfg, force=False):
        raise DatasetError("No training windows: widen weather.start_date")

    monkeypatch.setattr("src.data_pipeline.dataset.build_datasets", fail)
    missing = with_(tcfg, paths={"train_dataset": str(tmp_path / "none.pt")})
    with pytest.raises(DatasetError, match="No training windows"):
        train(missing)


@pytest.mark.parametrize(
    "model, match",
    [({"node_in_dim": 7}, "node_in_dim=7"), ({"edge_dim": 3}, "edge_dim=3"),
     ({"architecture": "lstm"}, "architecture")],
)
def test_model_must_match_the_dataset(tcfg, model, match):
    with pytest.raises(TrainingError, match=match):
        train(with_(tcfg, model=model))


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("field", ["graph_signature", "feature_names", "graph_attributes_sha256"])
def test_inconsistent_splits_are_rejected(tcfg, shared_data, tmp_path, field, split):
    source = shared_data[f"{split}_dataset"]
    payload = torch.load(source, weights_only=True)
    value = {"graph_signature": {**payload["graph_signature"], "sha256": "0" * 16},
             "feature_names": list(reversed(payload["feature_names"])), "graph_attributes_sha256": "f" * 16}[field]
    path = modified_payload(source, tmp_path / f"{split}.pt", **{field: value})
    with pytest.raises(TrainingError, match=f"train and {split} datasets are inconsistent.*{field}"):
        train(with_(tcfg, paths={f"{split}_dataset": path}))


@pytest.mark.integration
def test_validation_without_floods_falls_back_to_val_loss(tcfg, shared_data, tmp_path, log):
    payload = torch.load(shared_data["val_dataset"], weights_only=True)
    val = modified_payload(shared_data["val_dataset"], tmp_path / "val.pt", labels=torch.zeros_like(payload["labels"]))
    result = train(with_(tcfg, paths={"val_dataset": val}))
    m = result.metrics
    assert m["monitor"] == "val_loss" and m["validation"]["pr_auc"] is None
    assert m["threshold"] == 0.5 and m["temperature"] == 1.0 and m["calibration"]["slope"] == 1.0
    assert "use val_loss instead" in warnings_text(log)


@pytest.mark.integration
def test_training_split_without_floods_still_runs(tcfg, shared_data, tmp_path, log):
    payload = torch.load(shared_data["train_dataset"], weights_only=True)
    train_path = modified_payload(shared_data["train_dataset"], tmp_path / "train.pt",
                                  labels=torch.zeros_like(payload["labels"]))
    result = train(with_(tcfg, paths={"train_dataset": train_path}), epochs=1)
    assert result.metrics["baseline_logreg"]["status"] == "unavailable"
    assert result.metrics["baselines"]["hist_gbdt"]["status"] == "unavailable"
    assert "cannot learn floods" in warnings_text(log)


def test_empty_split_is_rejected(tcfg, shared_data, tmp_path):
    val = modified_payload(shared_data["val_dataset"], tmp_path / "val.pt",
                           window_starts=torch.zeros(0, dtype=torch.int64))
    with pytest.raises(TrainingError, match="split is empty"):
        train(with_(tcfg, paths={"val_dataset": val}))


def test_unwritable_reports_dir_only_warns(tcfg, tmp_path, log):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    result = train(with_(tcfg, paths={"reports_dir": str(blocker)}), epochs=1)
    assert result.best_checkpoint.exists() and load_checkpoint(result.best_checkpoint)["finalized_utc"]
    assert "Could not write" in warnings_text(log)


@pytest.mark.integration
def test_stale_dataset_is_reported(tcfg, log):
    result = train(with_(tcfg, hydrology={"flood_depth_threshold_m": 0.3}), epochs=1)
    assert result.metrics["dataset"]["stale"] is True
    assert "datasets are stale" in warnings_text(log)


def test_single_non_finite_step_is_skipped(tcfg, monkeypatch, log):
    real, calls = trainer._batch_loss, {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        loss = real(*args)
        return loss * float("nan") if calls["n"] == 1 else loss

    monkeypatch.setattr(trainer, "_batch_loss", flaky)
    result = train(tcfg, epochs=1)
    assert result.history[0]["n_skipped"] == 1 and result.history[0]["n_steps"] == 1
    assert "non-finite loss" in warnings_text(log)


def test_persistent_non_finite_loss_aborts(tcfg, monkeypatch):
    monkeypatch.setattr(trainer, "_batch_loss", lambda *args: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(TrainingError, match="3 consecutive"):
        train(with_(tcfg, training={"windows_per_epoch": 16}), epochs=1)
    with pytest.raises(TrainingError, match="Every training step"):
        train(tcfg, epochs=1)


def test_out_of_memory_gives_an_actionable_error(tcfg, monkeypatch):
    def oom(*args):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(trainer, "_train_epoch", oom)
    with pytest.raises(TrainingError, match="Out of memory.*micro_batch_size"):
        train(tcfg)


def test_other_runtime_errors_propagate_unchanged(tcfg, monkeypatch):
    def boom(*args):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(trainer, "_train_epoch", boom)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        train(tcfg)


def test_early_stopping_and_lr_schedule(tcfg, monkeypatch):
    monkeypatch.setattr(trainer, "_validate", lambda *args: dict(FLAT))
    cfg = with_(tcfg, model={"epochs": 8}, training={"early_stopping_patience": 2, "lr_scheduler": {"patience": 0}})
    result = train(cfg)
    assert [r["improved"] for r in result.history] == [True, False, False]
    assert result.metrics["status"] == "early_stopped" and result.best_epoch == 1
    # lr used during each epoch: the plateau (patience 0) halves it after epoch 2's non-improvement
    assert [r["lr"] for r in result.history] == pytest.approx([0.01, 0.01, 0.005])


def test_non_finite_validation_logits_do_not_count_as_improvement(tcfg, monkeypatch, log):
    real = trainer.evaluate
    calls = {"n": 0}

    def nan_first(*args, **kwargs):
        calls["n"] += 1
        y, logits, steps = real(*args, **kwargs)
        return (y, np.full_like(logits, np.nan), steps) if calls["n"] == 1 else (y, logits, steps)

    monkeypatch.setattr(trainer, "evaluate", nan_first)
    result = train(tcfg)
    assert result.history[0]["improved"] is False and result.history[0]["pr_auc"] is None
    assert result.history[1]["improved"] is True and result.best_epoch == 2
    assert "non-finite logits" in warnings_text(log)


# --------------------------------------------------------------------------- options


@pytest.mark.integration
def test_alternative_architecture_and_loss(tcfg):
    result = train(with_(tcfg, model={"architecture": "a3t_gcn"}, loss={"name": "weighted_bce"}), epochs=1)
    assert result.metrics["model"]["architecture"] == "a3t_gcn"
    assert model_from_checkpoint(result.best_checkpoint).architecture == "a3t_gcn"


def test_max_val_windows_and_all_training_windows(tcfg, log):
    """F2-02: max_val_windows subsamples the per-epoch validation only; the final calibration, threshold and
    reports always use the full validation split."""
    result = train(tcfg, epochs=1, max_val_windows=3, windows_per_epoch=0)
    m = result.metrics
    assert m["validation"]["n"] == N_VAL * SCORED * N_NODES
    assert m["calibration"]["n"] == N_VAL * SCORED * N_NODES
    assert m["dataset"]["val_windows"] == N_VAL and m["dataset"]["val_windows_per_epoch"] == 3
    assert m["dataset"]["windows"]["validation"] == N_VAL
    assert result.history[0]["pr_auc"] is not None  # the epoch itself was scored on the 3-window subsample
    assert m["test"]["n"] == N_TEST * SCORED * N_NODES  # the test split is always complete
    assert result.history[0]["n_steps"] == math.ceil(N_TRAIN / 4)  # every train window once
    assert "use all 22 validation windows (3 were scored per epoch)" in "\n".join(r.getMessage() for r in log.records)


def test_options_disable_calibration_and_baseline(tcfg, monkeypatch):
    threads = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    cfg = with_(tcfg, training={"calibrate_temperature": False, "baseline": {"enabled": False}, "num_threads": 2,
                                "grad_clip_norm": None, "threshold_metric": "csi"})
    result = train(cfg, epochs=1)
    m = result.metrics
    assert m["temperature"] == 1.0 and m["calibration"]["method"] == "none"
    assert m["baseline_logreg"] == {"status": "disabled"}
    assert m["baselines"] == {"logreg": {"status": "disabled"}, "hist_gbdt": {"status": "disabled"}}
    assert m["strongest_baseline"]["name"] is None and m["threshold_metric"] == "csi" and threads == [2]


@pytest.mark.parametrize("method", ["temperature", "none"])
def test_calibration_methods(tcfg, method):
    result = train(with_(tcfg, training={"calibration": method}), epochs=1)
    calibration = load_checkpoint(result.best_checkpoint)["calibration"]
    assert calibration["method"] == method and calibration["intercept"] == 0.0
    assert result.metrics["temperature"] == pytest.approx(1.0 / calibration["slope"])


def test_baseline_failure_does_not_fail_training(tcfg, monkeypatch, log):
    def broken(*args, **kwargs):
        raise ValueError("solver exploded")

    monkeypatch.setattr(finalize_module, "run_baselines", broken)
    result = train(tcfg, epochs=1)
    assert result.metrics["baseline_logreg"]["status"] == "failed"
    assert result.metrics["baselines"]["hist_gbdt"]["status"] == "failed"
    assert "solver exploded" in warnings_text(log)


# --------------------------------------------------------------------------- evaluate / sampling / settings


def test_evaluate_shapes_steps_temperature_and_mode(tcfg):
    train_ds, val_ds = load_datasets(tcfg)
    model = trainer._build_model(tcfg, train_ds)
    model.train()
    loader = make_loader(val_ds, 3, False)
    y, logits, steps = evaluate(model, loader, "cpu")
    assert y.dtype == np.uint8 and logits.dtype == np.float32 and steps.dtype == np.int32
    assert y.size == logits.size == steps.size == len(val_ds) * SCORED * N_NODES
    assert steps.min() == WARMUP and steps.max() == SEQ_LEN - 1
    _, half, _ = evaluate(model, loader, torch.device("cpu"), temperature=2.0)
    np.testing.assert_allclose(half, logits / 2.0, rtol=1e-6)
    assert model.training  # mode restored
    with pytest.raises(ValueError, match="temperature"):
        evaluate(model, loader, "cpu", temperature=0.0)


def test_evaluate_on_an_empty_loader():
    model = torch.nn.Linear(1, 1)
    y, logits, steps = evaluate(model, [], "cpu")
    assert y.size == logits.size == steps.size == 0


def test_window_sampling_weights_and_epoch_sampler(tcfg):
    weights = window_sampling_weights([True, False, True], 3.0)
    assert weights.tolist() == [3.0, 1.0, 3.0]
    with pytest.raises(ValueError):
        window_sampling_weights([True], 0.0)
    settings = TrainingSettings.from_config(tcfg)
    flags = np.array([True] * 10 + [False] * 90)
    sampler = epoch_sampler(flags, settings, epoch=1)
    assert isinstance(sampler, WeightedRandomSampler) and len(sampler) == 8
    assert list(epoch_sampler(flags, settings, 1)) == list(sampler)  # seeded per epoch
    assert list(epoch_sampler(flags, settings, 2)) != list(sampler)
    many = TrainingSettings.from_config(with_(tcfg, training={"windows_per_epoch": 20000}))
    drawn = np.array(list(epoch_sampler(flags, many, 1)))
    assert 0.2 < flags[drawn].mean() < 0.3  # 10 x 3 / (10 x 3 + 90) = 0.25
    plain = TrainingSettings.from_config(with_(tcfg, training={"windows_per_epoch": None, "positive_oversample": 1}))
    every = epoch_sampler(flags, plain, 1)
    assert isinstance(every, RandomSampler) and sorted(every) == list(range(100))


@pytest.mark.unit
@pytest.mark.parametrize(
    "section, match",
    [
        ({"training": {"windows_per_epoch": -1}}, "windows_per_epoch"),
        ({"training": {"micro_batch_size": 0}}, "micro_batch_size"),
        ({"training": {"positive_oversample": 0}}, "positive_oversample"),
        ({"training": {"early_stopping_metric": "accuracy"}}, "early_stopping_metric"),
        ({"training": {"threshold_metric": "bogus"}}, "threshold_metric"),
        ({"training": {"calibration": "isotonic"}}, "calibration"),
        ({"training": {"lr_scheduler": {"factor": 1.5}}}, "factor"),
        ({"training": {"device": "tpu"}}, "device"),
        ({"training": {"calibrate_temperature": "yes"}}, "calibrate_temperature"),
        ({"training": {"baseline": {"max_samples": 1}}}, "max_samples"),
        ({"training": {"baseline": {"hist_gbdt": {"learning_rate": 0}}}}, "hist_gbdt.learning_rate"),
        ({"model": {"epochs": 0}}, "epochs"),
        ({"model": {"learning_rate": 0}}, "learning_rate"),
        ({"model": {"batch_size": 2.5}}, "batch_size"),
    ],
)
def test_settings_validation(tcfg, section, match):
    with pytest.raises(ConfigError, match=match):
        TrainingSettings.from_config(with_(tcfg, **section))


@pytest.mark.unit
def test_settings_defaults_and_overrides(tcfg):
    bare = {"project": {"seed": 7}, "paths": {}}
    settings = TrainingSettings.from_config(bare)
    assert settings.epochs == 60 and settings.windows_per_epoch == 512 and settings.seed == 7
    assert settings.micro_batch_size == 4 and settings.batch_size == 16 and settings.calibration == "platt"
    assert settings.monitor_mode == "max" and settings.baseline_enabled and settings.gbdt_enabled
    assert settings.calibrate_temperature is True
    over = TrainingSettings.from_config(tcfg, epochs=3, device="CPU", resume=True, windows_per_epoch=0,
                                        max_val_windows=5)
    assert (over.epochs, over.device, over.resume, over.windows_per_epoch, over.max_val_windows) == (3, "cpu", True,
                                                                                                    None, 5)
    capped = TrainingSettings.from_config(with_(tcfg, training={"micro_batch_size": 64}))
    assert capped.micro_batch_size == 4  # never more than model.batch_size
    assert TrainingSettings.from_config(with_(tcfg, training={"micro_batch_size": None})).micro_batch_size == 4


@pytest.mark.unit
def test_monitor_improvement_rules():
    up, down = Monitor("pr_auc", "max", 0.01, relative=False), Monitor("val_loss", "min", 0.01, relative=False)
    assert up.improved(0.5, None) and not up.improved(None, 0.5)
    assert up.improved(0.52, 0.5) and not up.improved(0.505, 0.5)
    assert down.improved(0.4, 0.5) and not down.improved(0.495, 0.5)
    assert not up.improved(float("nan"), None)
    # Relative margins follow the metric's scale: a rare-event PR-AUC rising 1e-5 -> 7e-4 is a real gain
    # that an absolute 0.001 margin would miss (observed on the real Bellandur datasets).
    rel = Monitor("pr_auc", "max", 0.001, relative=True)
    assert rel.improved(7e-4, 1e-5) and not up.improved(7e-4, 1e-5)
    assert rel.improved(0.3004, 0.3) and not rel.improved(0.30029, 0.3)
    assert Monitor("val_loss", "min", 0.1).improved(0.089, 0.1) and not Monitor("val_loss", "min", 0.1).improved(
        0.091, 0.1)
    assert rel.improved(1e-9, 0.0)  # anything above a zero best counts


def test_settings_min_delta_mode(tcfg):
    assert TrainingSettings.from_config(tcfg).min_delta_mode == "rel"
    assert TrainingSettings.from_config(with_(tcfg, training={"min_delta_mode": "ABS"})).min_delta_mode == "abs"
    with pytest.raises(ConfigError, match="min_delta_mode"):
        TrainingSettings.from_config(with_(tcfg, training={"min_delta_mode": "percent"}))
