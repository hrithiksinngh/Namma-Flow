"""Tests for the offline-demo folder guard (src/utils/demo_dir.py) and its Makefile wiring (F5-02, F5-05).

The Makefile tests run the real Makefile in a sandbox copy of the project (fake pipeline stages that only
record their arguments and touch the re-rooted paths), never against the real data/ or artifacts/.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from src.utils.demo_dir import (
    DEMO_CONFIG_NAME,
    MARKER_KIND,
    MARKER_NAME,
    DemoDirError,
    check_demo_root,
    clean_demo,
    is_within,
    main,
    prepare_demo,
    read_marker,
    rerooted_config,
    verify_demo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = {"project": {"name": "Namma-Flow", "seed": 42},
                 "paths": {"graph_file": "data/interim/g.graphml", "weather_file": "data/raw/weather/w.csv",
                           "checkpoint_dir": "artifacts/checkpoints", "reports_dir": "artifacts/reports",
                           "osm_cache_dir": "/data/interim/osm_cache", "unset": None},
                 "model": {"node_in_dim": 10}}


@dataclass(frozen=True)
class World:
    """A fake $HOME with a project (real data / artifacts / config) and an unrelated sibling folder."""

    home: Path
    project: Path
    precious: Path
    config: Path


def _digest(folder: Path) -> str:
    """Hash of every file path and content below ``folder`` (detects any deletion or change)."""
    sha = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        sha.update(str(path.relative_to(folder)).encode() + b"\0" + path.read_bytes())
    return sha.hexdigest()


def _build_world(base: Path) -> World:
    home = base / "home"
    project = home / "Desktop" / "proj"
    for rel, text in {"data/processed/train_dataset.pt": "real data", "artifacts/checkpoints/best.pt": "real model",
                      "config/config.yaml": yaml.safe_dump(SOURCE_CONFIG)}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    precious = home / "Desktop" / "precious"
    precious.mkdir(parents=True)
    (precious / "thesis.txt").write_text("irreplaceable")
    return World(home=home, project=project, precious=precious, config=project / "config" / "config.yaml")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return _build_world(tmp_path)


# --------------------------------------------------------------------------- check_demo_root


@pytest.mark.unit
@pytest.mark.parametrize("pick", [
    lambda w: "", lambda w: "   ", lambda w: "/", lambda w: w.project, lambda w: w.project.parent,
    lambda w: w.home, lambda w: w.home.parent, lambda w: w.project / "data", lambda w: w.project / "data" / "sub",
    lambda w: w.project / "artifacts", lambda w: w.project / "artifacts" / "new" / "demo",
    lambda w: w.project / "config",
])
def test_check_demo_root_refuses_folders_that_overlap_real_data(world, pick):
    """F5-02: /, $HOME, the project, its ancestors and anything inside data/ artifacts/ config/ are refused."""
    with pytest.raises(DemoDirError):
        check_demo_root(pick(world), world.project, home=world.home)


@pytest.mark.unit
def test_check_demo_root_accepts_dedicated_folders(world, tmp_path):
    assert check_demo_root(tmp_path / "demo", world.project, home=world.home) == (tmp_path / "demo").resolve()
    assert check_demo_root(world.project / "demo", world.project, home=world.home).name == "demo"
    assert check_demo_root(world.home / "demos" / "a", world.project, home=world.home).name == "a"


@pytest.mark.unit
def test_check_demo_root_refuses_home_even_when_the_project_lives_elsewhere(world, tmp_path):
    other_home = tmp_path / "users" / "someone"
    other_home.mkdir(parents=True)
    for root in (other_home, other_home.parent):
        with pytest.raises(DemoDirError, match="home folder"):
            check_demo_root(root, world.project, home=other_home)
    assert check_demo_root(other_home / "demo", world.project, home=other_home).name == "demo"


@pytest.mark.unit
def test_check_demo_root_resolves_symlinked_data_folders(world, tmp_path):
    external = tmp_path / "external" / "namma-data"
    shutil.move(str(world.project / "data"), str(external))
    (world.project / "data").symlink_to(external, target_is_directory=True)
    for root in (tmp_path / "external", external / "sub", world.project / "data" / "x"):
        with pytest.raises(DemoDirError, match="data/ folder"):
            check_demo_root(root, world.project, home=world.home)
    link = tmp_path / "alias"
    link.symlink_to(world.project, target_is_directory=True)
    with pytest.raises(DemoDirError, match="project folder"):
        check_demo_root(link, world.project, home=world.home)


@pytest.mark.unit
def test_check_demo_root_catches_case_variants_on_case_insensitive_disks(world):
    if not (world.project / "DATA").exists():
        pytest.skip("case-sensitive file system")
    with pytest.raises(DemoDirError, match="data/ folder"):
        check_demo_root(world.project / "DATA" / "demo", world.project, home=world.home)


@pytest.mark.unit
def test_check_demo_root_refuses_a_demo_config_that_is_config_itself(tmp_path, world):
    """F5-05: CONFIG=<dir>/config.yaml with DEMO_DIR=<dir> would run the demo on the real paths."""
    own = tmp_path / "deploy"
    own.mkdir()
    (own / DEMO_CONFIG_NAME).write_text("project: {}\npaths: {}\n")
    with pytest.raises(DemoDirError, match="CONFIG itself"):
        check_demo_root(own, world.project, home=world.home, source_config=own / DEMO_CONFIG_NAME)
    with pytest.raises(DemoDirError):
        check_demo_root(world.project / "config", world.project, home=world.home, source_config=world.config)


@pytest.mark.unit
def test_is_within_handles_missing_paths(tmp_path):
    assert is_within(tmp_path / "a" / "b", tmp_path / "a")
    assert not is_within(tmp_path / "ab", tmp_path / "a")
    assert is_within(tmp_path, tmp_path)


# --------------------------------------------------------------------------- prepare / verify


@pytest.mark.unit
def test_rerooted_config_moves_every_path_under_the_demo(tmp_path):
    cfg, entries = rerooted_config(SOURCE_CONFIG, tmp_path)
    assert cfg["project"] == {"name": "Namma-Flow", "seed": 42, "offline": True}
    assert cfg["paths"]["graph_file"] == str(tmp_path / "data/interim/g.graphml")
    assert cfg["paths"]["osm_cache_dir"] == str(tmp_path / "data/interim/osm_cache"), "absolute paths re-rooted too"
    assert cfg["paths"]["unset"] is None and entries == ("artifacts", "data")
    assert SOURCE_CONFIG["project"] == {"name": "Namma-Flow", "seed": 42}, "the caller's config is not mutated"
    for bad, match in (({"x": "../escape"}, "escape"), ({"x": "config.yaml"}, "collides"),
                       ({"x": f"{MARKER_NAME}/a"}, "collides")):
        with pytest.raises(DemoDirError, match=match):
            rerooted_config({"paths": bad}, tmp_path)
    with pytest.raises(DemoDirError, match="mappings"):
        rerooted_config({"paths": ["data"]}, tmp_path)


@pytest.mark.unit
def test_prepare_writes_marker_and_config_and_is_repeatable(world, tmp_path):
    root = tmp_path / "demo"
    before = world.config.read_bytes()
    done = prepare_demo(root, world.config, world.project, home=world.home)
    assert done.entries == ("artifacts", DEMO_CONFIG_NAME, "data")
    written = yaml.safe_load(done.config.read_text())
    assert written["project"]["offline"] is True
    assert all(v is None or Path(v).is_relative_to(done.root) for v in written["paths"].values())
    marker = json.loads((done.root / MARKER_NAME).read_text())
    assert marker["kind"] == MARKER_KIND and marker["entries"] == list(done.entries)
    assert world.config.read_bytes() == before, "the source config is never touched"
    (done.root / "data").mkdir()
    again = prepare_demo(root, world.config, world.project, home=world.home)
    assert again.entries == done.entries and verify_demo(root, world.config, world.project, home=world.home)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prepare_demo(empty, world.config, world.project, home=world.home).root == empty.resolve()


@pytest.mark.unit
def test_prepare_refuses_folders_it_did_not_create(world, tmp_path):
    """F5-02: a non-empty folder without the marker (e.g. ~/Desktop) is never adopted."""
    before = _digest(world.precious)
    with pytest.raises(DemoDirError, match="no .namma-flow-demo marker"):
        prepare_demo(world.precious, world.config, world.project, home=world.home)
    assert _digest(world.precious) == before and not (world.precious / MARKER_NAME).exists()
    a_file = tmp_path / "file"
    a_file.write_text("x")
    with pytest.raises(DemoDirError, match="not a folder"):
        prepare_demo(a_file, world.config, world.project, home=world.home)
    with pytest.raises(DemoDirError, match="not a prepared demo"):
        verify_demo(world.precious, world.config, world.project, home=world.home)


@pytest.mark.unit
def test_prepare_refuses_entries_that_symlink_out_of_the_demo(world, tmp_path):
    root = prepare_demo(tmp_path / "demo", world.config, world.project, home=world.home).root
    (root / "data").symlink_to(world.project / "data", target_is_directory=True)
    with pytest.raises(DemoDirError, match="points outside DEMO_DIR"):
        prepare_demo(root, world.config, world.project, home=world.home)


@pytest.mark.unit
def test_prepare_reports_unreadable_configs(world, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(DemoDirError, match="must hold a mapping"):
        prepare_demo(tmp_path / "demo", bad, world.project, home=world.home)
    with pytest.raises(DemoDirError, match="cannot read CONFIG"):
        prepare_demo(tmp_path / "demo", tmp_path / "missing.yaml", world.project, home=world.home)
    bad.write_text("")
    assert prepare_demo(tmp_path / "demo", bad, world.project, home=world.home).entries == (DEMO_CONFIG_NAME,)


# --------------------------------------------------------------------------- clean


def _populate(root: Path) -> None:
    for rel in ("data/interim/g.graphml", "artifacts/checkpoints/best.pt", "artifacts/reports/metrics.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("demo")


@pytest.mark.unit
def test_clean_removes_only_what_the_demo_created(world, tmp_path):
    root = prepare_demo(tmp_path / "demo", world.config, world.project, home=world.home).root
    _populate(root)
    (root / "notes.txt").write_text("the user's own file")
    done = clean_demo(root, world.project, home=world.home)
    assert done.removed == ("artifacts", DEMO_CONFIG_NAME, "data") and done.kept == ("notes.txt",)
    assert not done.removed_root and (root / "notes.txt").read_text() == "the user's own file"
    assert sorted(p.name for p in root.iterdir()) == ["notes.txt"]
    again = prepare_demo(tmp_path / "demo2", world.config, world.project, home=world.home).root
    _populate(again)
    assert clean_demo(again, world.project, home=world.home).removed_root and not again.exists()
    missing = clean_demo(tmp_path / "never", world.project, home=world.home)
    assert not missing.existed and missing.removed == ()


@pytest.mark.unit
def test_clean_refuses_unmarked_or_corrupt_folders(world, tmp_path):
    before = _digest(world.precious)
    with pytest.raises(DemoDirError, match="nothing was deleted"):
        clean_demo(world.precious, world.project, home=world.home)
    for payload in ("{not json", json.dumps({"kind": "other", "entries": []}), json.dumps([1, 2]),
                    json.dumps({"kind": MARKER_KIND, "entries": ["../thesis.txt"]}),
                    json.dumps({"kind": MARKER_KIND, "entries": ["a/b"]})):
        (world.precious / MARKER_NAME).write_text(payload)
        with pytest.raises(DemoDirError, match="marker"):
            clean_demo(world.precious, world.project, home=world.home)
    (world.precious / MARKER_NAME).unlink()
    assert _digest(world.precious) == before
    assert read_marker(tmp_path / "nowhere") is None
    a_file = tmp_path / "file"
    a_file.write_text("x")
    with pytest.raises(DemoDirError, match="not a folder"):
        clean_demo(a_file, world.project, home=world.home)
    for protected in (world.project / "data", world.project.parent, world.home):
        with pytest.raises(DemoDirError):
            clean_demo(protected, world.project, home=world.home)


@pytest.mark.unit
def test_clean_unlinks_symlinked_entries_without_following_them(world, tmp_path):
    root = prepare_demo(tmp_path / "demo", world.config, world.project, home=world.home).root
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (root / "data").symlink_to(outside, target_is_directory=True)
    clean_demo(root, world.project, home=world.home)
    assert (outside / "keep.txt").read_text() == "keep" and not root.exists()


# --------------------------------------------------------------------------- CLI


@pytest.mark.unit
def test_cli_reports_success_and_one_line_errors(world, tmp_path, capsys):
    common = ["--project", str(world.project), "--home", str(world.home)]
    root = str(tmp_path / "demo")
    assert main(["prepare", "--root", root, "--config", str(world.config), *common]) == 0
    assert "Wrote demo config" in capsys.readouterr().out
    assert main(["check", "--root", root, "--config", str(world.config), *common]) == 0
    assert "Demo folder OK" in capsys.readouterr().out
    assert main(["clean", "--root", root, *common]) == 0
    assert "removed" in capsys.readouterr().out.lower()
    assert main(["clean", "--root", root, *common]) == 0
    assert "Nothing to clean" in capsys.readouterr().out
    assert main(["clean", "--root", str(world.project.parent), *common]) == 1
    assert capsys.readouterr().err.startswith("ERROR: DEMO_DIR=")
    assert main(["clean", "--root", str(tmp_path / "a b"), *common]) == 1
    assert "whitespace" in capsys.readouterr().err


# --------------------------------------------------------------------------- Makefile (sandbox copy)
FAKE_STAGE = '''"""Fake pipeline stage: records its argv and touches every re-rooted path of --config."""
import sys
from pathlib import Path

import yaml

args = sys.argv[1:]
cfg = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
for value in filter(None, cfg["paths"].values()):
    path = Path(value)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    else:
        path.mkdir(parents=True, exist_ok=True)
if "--out" in args:
    out = Path(args[args.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
with (Path(__file__).resolve().parents[2] / "stages.log").open("a") as log:
    log.write(Path(__file__).name + " " + " ".join(args) + "\\n")
'''
STAGES = ("src/data_pipeline/01_extract_network.py", "src/data_pipeline/03_weather_ingestion.py",
          "src/data_pipeline/04_dataset_builder.py", "src/training/train.py", "src/inference/predict.py")


@pytest.fixture
def sandbox(tmp_path: Path) -> World:
    """World whose project holds a copy of the real Makefile, the guard module and fake pipeline stages."""
    if shutil.which("make") is None:  # pragma: no cover
        pytest.skip("make is not installed")
    w = _build_world(tmp_path)
    shutil.copy(PROJECT_ROOT / "Makefile", w.project / "Makefile")
    for rel in ("src/__init__.py", "src/utils/__init__.py", "src/utils/demo_dir.py", "src/utils/runtime.py",
                "src/utils/logger.py"):
        (w.project / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT_ROOT / rel, w.project / rel)
    for rel in STAGES:
        (w.project / rel).parent.mkdir(parents=True, exist_ok=True)
        (w.project / rel).parent.joinpath("__init__.py").touch()
        (w.project / rel).write_text(FAKE_STAGE)
    return w


def _make(w: World, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(w.home), "NAMMA_FLOW_CONFIG": "", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(["make", "--no-print-directory", "-C", str(w.project), f"PY={sys.executable}", *args],
                          env=env, capture_output=True, text=True, timeout=60)


def _real_state(w: World) -> tuple[str, str]:
    return _digest(w.project), _digest(w.precious)


@pytest.mark.integration
@pytest.mark.parametrize("demo_dir", ["..", ".", "data", "artifacts", "config", "HOME", "/", "../precious"])
def test_make_clean_demo_never_deletes_real_folders(sandbox, demo_dir):
    """F5-02 regression: the reviewer's `make clean-demo DEMO_DIR=..` deleted the project and its siblings."""
    before = _real_state(sandbox)
    result = _make(sandbox, "clean-demo", f"DEMO_DIR={sandbox.home if demo_dir == 'HOME' else demo_dir}")
    assert result.returncode != 0 and "ERROR:" in result.stderr, result.stdout + result.stderr
    assert _real_state(sandbox) == before


@pytest.mark.integration
@pytest.mark.parametrize("demo_dir", ["config", ".", "artifacts", "../precious"])
def test_make_offline_demo_refuses_before_any_stage_runs(sandbox, demo_dir):
    """F5-05 regression: DEMO_DIR=config used to skip the guard and train against the real config."""
    before = _real_state(sandbox)
    result = _make(sandbox, "offline-demo", f"DEMO_DIR={demo_dir}")
    assert result.returncode != 0 and "ERROR:" in result.stderr, result.stdout + result.stderr
    assert not (sandbox.project / "stages.log").exists(), "no pipeline stage may run"
    assert _real_state(sandbox) == before


@pytest.mark.integration
def test_make_offline_demo_then_clean_demo_touch_only_the_demo(sandbox, tmp_path):
    demo = tmp_path / "scratch-demo"
    before = _real_state(sandbox)
    result = _make(sandbox, "offline-demo", f"DEMO_DIR={demo}")
    assert result.returncode == 0, result.stdout + result.stderr
    runs = (sandbox.project / "stages.log").read_text().splitlines()
    assert len(runs) == 6, runs
    assert all(f"--config {demo}/config.yaml" in run and "--offline" in run for run in runs)
    assert (demo / "artifacts/reports/design_storm.geojson").exists() and (demo / "data/interim/g.graphml").exists()
    (sandbox.project / "stages.log").unlink()
    assert _real_state(sandbox) == before, "the demo never writes into the project's data/ artifacts/ config/"
    rerun = _make(sandbox, "demo-config", f"DEMO_DIR={demo}")
    assert rerun.returncode == 0 and "Wrote demo config" in rerun.stdout, "the demo config is always regenerated"
    (demo / "notes.txt").write_text("mine")
    cleaned = _make(sandbox, "clean-demo", f"DEMO_DIR={demo}")
    assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
    assert sorted(p.name for p in demo.iterdir()) == ["notes.txt"] and _real_state(sandbox) == before


@pytest.mark.integration
def test_make_dry_run_runs_the_guard_before_the_stages(sandbox, tmp_path):
    lines = _make(sandbox, "-n", "offline-demo", f"DEMO_DIR={tmp_path / 'd'}").stdout.splitlines()
    assert "demo_dir prepare" in lines[0] and "demo_dir check" in lines[1] and "01_extract_network.py" in lines[2]
    assert "demo_dir clean" in _make(sandbox, "-n", "clean-demo").stdout
    assert "rm -rf" not in (PROJECT_ROOT / "Makefile").read_text()


@pytest.mark.integration
def test_make_setup_refuses_a_python_below_the_floor_before_installing(sandbox):
    """F5-03: `make setup` stops with a clear message (no venv, no pip) when the interpreter is too old."""
    fresh = _make(sandbox, "setup", "MIN_PYTHON=99.0", f"SYSTEM_PYTHON={sys.executable}",
                  f"PY={sandbox.project / '.venv/bin/python'}")
    assert fresh.returncode != 0 and "needs Python >= 99.0" in fresh.stderr, fresh.stdout + fresh.stderr
    assert not (sandbox.project / ".venv").exists(), "no virtual environment is created for a too-old Python"
    existing = _make(sandbox, "setup", "MIN_PYTHON=99.0")
    assert existing.returncode != 0 and "needs Python >= 99.0" in existing.stderr
    assert "pip install" not in existing.stdout, "the pinned install never starts"
    planned = _make(sandbox, "-n", "setup").stdout.splitlines()
    assert "3.12" in planned[1] and "pip install -r requirements.txt -c constraints.txt" in planned[2]
