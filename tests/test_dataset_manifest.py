"""All-or-nothing commit of the split payloads and the dataset manifest (F1-05).

An interrupted rebuild must never leave train from the new build and val/test from the old one
looking like a consistent build, and the CLI must not claim that nothing was written.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.data_pipeline import dataset_manifest
from src.data_pipeline.dataset import build_datasets, dataset_paths, load_payload
from src.data_pipeline.dataset_manifest import MANIFEST_NAME, commit_splits, manifest_problem, read_manifest
from src.utils.config import deep_merge
from tests.dataset_fixtures import fake_simulator, log  # noqa: F401 - pytest fixtures
from tests.dataset_fixtures import setup_config

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "data_pipeline" / "04_dataset_builder.py"


@pytest.fixture
def built(cfg, fake_simulator):
    dcfg = setup_config(cfg)
    return dcfg, build_datasets(dcfg)


def _snapshot(paths: dict[str, Path]) -> dict[str, bytes]:
    files = dict(paths)
    files["manifest"] = paths["train"].with_name(MANIFEST_NAME)
    return {name: path.read_bytes() for name, path in files.items() if path.exists()}


def _leftover_temps(paths: dict[str, Path]) -> list[Path]:
    return sorted(paths["train"].parent.glob(".*.tmp"))


@pytest.mark.integration
def test_build_writes_a_manifest_that_vouches_for_every_split(built):
    cfg, summary = built
    paths = dataset_paths(cfg)
    manifest = read_manifest(paths)
    payloads = {split: load_payload(paths[split]) for split in ("train", "val", "test")}
    assert manifest["build_id"] == summary["build_id"] == payloads["val"]["build_id"]
    assert set(manifest["splits"]) == {"train", "val", "test"}
    for split, entry in manifest["splits"].items():
        assert entry["file"] == paths[split].name and entry["size_bytes"] == paths[split].stat().st_size
        assert len(entry["sha256"]) == 64
    assert manifest["config_hash"] == payloads["train"]["config_hash"]
    assert "/" not in json.dumps(manifest["splits"]), "the manifest stores file names, never absolute paths"
    assert manifest_problem(paths, ("train", "val", "test"),
                            {s: p["build_id"] for s, p in payloads.items()}) is None
    assert build_datasets(cfg)["reused"] is True


@pytest.mark.integration
def test_interrupt_while_writing_leaves_the_previous_build_untouched(built, fake_simulator, monkeypatch):
    cfg, _ = built
    paths = dataset_paths(cfg)
    before = _snapshot(paths)
    calls = []

    def interrupted_save(obj, path):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt
        torch.save(obj, path)

    monkeypatch.setattr(dataset_manifest, "_torch_save", interrupted_save)
    changed = deep_merge(cfg, {"hydrology": {"drain_capacity_far_mm_h": 9.0}})
    with pytest.raises(KeyboardInterrupt):
        build_datasets(changed)
    assert _snapshot(paths) == before, "no split (and not the manifest) may change before the commit"
    assert _leftover_temps(paths) == []


@pytest.mark.integration
def test_interrupt_between_renames_is_detected_and_rebuilt(built, fake_simulator, monkeypatch, log):
    cfg, _ = built
    paths = dataset_paths(cfg)
    real_replace = dataset_manifest.os.replace
    renamed = []

    def interrupted_replace(src, dst):
        if renamed:
            raise KeyboardInterrupt
        renamed.append(dst)
        real_replace(src, dst)

    monkeypatch.setattr(dataset_manifest.os, "replace", interrupted_replace)
    changed = deep_merge(cfg, {"hydrology": {"drain_capacity_far_mm_h": 9.0}})
    with pytest.raises(KeyboardInterrupt):
        build_datasets(changed)
    monkeypatch.setattr(dataset_manifest.os, "replace", real_replace)
    assert not paths["train"].with_name(MANIFEST_NAME).exists(), "the old manifest is removed before renaming"
    assert _leftover_temps(paths) == []
    ids = {split: load_payload(paths[split])["build_id"] for split in ("train", "val", "test")}
    assert len(set(ids.values())) == 2, "the simulated crash really mixed two builds"
    summary = build_datasets(changed)
    assert summary["reused"] is False
    assert any("different builds" in r or "manifest" in r for r in summary["rebuild_reasons"])
    assert len({load_payload(paths[s])["build_id"] for s in ("train", "val", "test")}) == 1
    assert build_datasets(changed)["reused"] is True


@pytest.mark.integration
def test_splits_without_a_manifest_are_rebuilt(built, fake_simulator, log):
    cfg, _ = built
    dataset_paths(cfg)["train"].with_name(MANIFEST_NAME).unlink()
    summary = build_datasets(cfg)
    assert summary["reused"] is False and "no readable dataset manifest" in summary["rebuild_reasons"][0]


@pytest.mark.unit
def test_manifest_problem_reports_each_inconsistency(tmp_path):
    paths = {split: tmp_path / f"{split}.pt" for split in ("train", "val")}
    payloads = {split: {"build_id": "b1", "split": split} for split in paths}
    commit_splits(payloads, paths, extra={"config_hash": "c"})
    ids = {split: "b1" for split in paths}
    assert manifest_problem(paths, ("train", "val"), ids) is None
    assert "different builds" in manifest_problem(paths, ("train", "val"), {"train": "b1", "val": "b0"})
    assert "does not list the test" in manifest_problem({**paths, "test": tmp_path / "t.pt"}, ("test",), {})
    paths["val"].write_bytes(b"changed")
    assert "changed after the build" in manifest_problem(paths, ("train", "val"), ids)
    paths["train"].with_name(MANIFEST_NAME).write_text("{not json")
    assert "no readable dataset manifest" in manifest_problem(paths, ("train",), ids)
    with pytest.raises(ValueError, match="one build"):
        commit_splits({"train": {"build_id": "x"}, "val": {"build_id": "y"}}, paths)


@pytest.mark.unit
def test_cli_interrupt_message_is_accurate(cfg, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("dataset_builder_cli_manifest", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(cli, "build_datasets", interrupted)
    assert cli.main([]) == cli.EXIT_INTERRUPTED
    err = capsys.readouterr().err
    assert "manifest is written last" in err and "no dataset was written" not in err
