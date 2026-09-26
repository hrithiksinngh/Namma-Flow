"""Safety rails for the offline demo directory (``make demo-config`` / ``offline-demo`` / ``clean-demo``).

The offline demo copies a config with every ``paths.*`` entry re-rooted under a demo directory
(``DEMO_DIR``), runs the whole pipeline there and can later be deleted again. Because the last step
deletes files, the directory is checked before anything is written or removed (F5-02, F5-05):

* refused outright: an empty value, the filesystem root, ``$HOME`` or any of its ancestors, the
  project folder or any of its ancestors, and any folder that lies inside or contains the project's
  ``data/``, ``artifacts/`` or ``config/`` folders (symlinks resolved, case-insensitive file systems
  handled by comparing inodes of existing folders);
* refused when the demo ``config.yaml`` would be the source ``CONFIG`` itself;
* a directory is only adopted when it is new, empty or carries the marker file
  (:data:`MARKER_NAME`) written by a previous ``demo-config``; the marker lists every top-level
  entry the demo creates;
* ``clean`` deletes only the entries listed in the marker, then the marker, then the directory
  itself if nothing else is left in it.

CLI (used by the Makefile; exit 0 on success, 1 with a one-line ``ERROR:`` message otherwise)::

    python -m src.utils.demo_dir prepare --root DEMO_ROOT --config CONFIG [--project DIR]
    python -m src.utils.demo_dir check   --root DEMO_ROOT --config CONFIG [--project DIR]
    python -m src.utils.demo_dir clean   --root DEMO_ROOT [--project DIR]

``clean`` and ``check`` need only the standard library; ``prepare`` needs PyYAML.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MARKER_NAME = ".namma-flow-demo"
MARKER_KIND = "namma-flow-offline-demo"
MARKER_VERSION = 1
DEMO_CONFIG_NAME = "config.yaml"
PROTECTED_DIRS = ("data", "artifacts", "config")
DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


class DemoDirError(ValueError):
    """Raised when a demo directory is unsafe to write to or to delete."""


@dataclass(frozen=True)
class DemoPrepared:
    """Result of :func:`prepare_demo`: the resolved root, the demo config written and the marker entries."""

    root: Path
    config: Path
    entries: tuple[str, ...]


@dataclass(frozen=True)
class DemoCleaned:
    """Result of :func:`clean_demo`: what was removed, what was left alone, whether the root is gone."""

    root: Path
    removed: tuple[str, ...]
    kept: tuple[str, ...]
    removed_root: bool
    existed: bool


# --------------------------------------------------------------------------- path checks


def real_path(path: str | os.PathLike[str]) -> Path:
    """Absolute path with every symlink of its existing prefix resolved (the path need not exist)."""
    return Path(os.path.realpath(os.fspath(path)))


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def is_within(path: Path, parent: Path) -> bool:
    """True when ``path`` is ``parent`` or lies below it (both resolved; existing folders compared by inode)."""
    path, parent = real_path(path), real_path(parent)
    target = _identity(parent)
    for candidate in (path, *path.parents):
        if candidate == parent or (target is not None and _identity(candidate) == target):
            return True
    return False


def _same_file(first: Path, second: Path) -> bool:
    if real_path(first) == real_path(second):
        return True
    one, two = _identity(first), _identity(second)
    return one is not None and one == two


def check_demo_root(
    root: str | os.PathLike[str],
    project: str | os.PathLike[str] = DEFAULT_PROJECT,
    *,
    home: str | os.PathLike[str] | None = None,
    source_config: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the resolved demo root, or raise :class:`DemoDirError` when it could overlap real data."""
    text = os.fspath(root).strip()
    if not text:
        raise DemoDirError("DEMO_DIR is empty; choose a new folder such as /tmp/namma-flow-demo")
    resolved = real_path(text)
    if resolved.parent == resolved:
        raise DemoDirError(f"DEMO_DIR={text} is the filesystem root; choose a new, dedicated folder")
    project_dir = real_path(project)
    if is_within(project_dir, resolved):
        raise DemoDirError(f"DEMO_DIR={text} is the project folder or one of its ancestors ({project_dir}); "
                           "choose a new, dedicated folder")
    for name in PROTECTED_DIRS:
        protected = real_path(project_dir / name)
        if is_within(resolved, protected):
            raise DemoDirError(f"DEMO_DIR={text} lies inside the project's {name}/ folder ({protected})")
        if is_within(protected, resolved):
            raise DemoDirError(f"DEMO_DIR={text} contains the project's {name}/ folder ({protected})")
    home_dir = real_path(home if home is not None else Path.home())
    if is_within(home_dir, resolved):
        raise DemoDirError(f"DEMO_DIR={text} is your home folder or one of its ancestors ({home_dir})")
    if source_config is not None and _same_file(Path(source_config), resolved / DEMO_CONFIG_NAME):
        raise DemoDirError(f"DEMO_DIR={text}: the demo config {resolved / DEMO_CONFIG_NAME} would be CONFIG "
                           "itself, so the demo would run on the real paths")
    return resolved


# --------------------------------------------------------------------------- marker


def _valid_entry(name: Any) -> bool:
    return (isinstance(name, str) and name not in ("", ".", "..") and "/" not in name
            and os.sep not in name and "\x00" not in name)


def read_marker(root: Path) -> tuple[str, ...] | None:
    """Entries listed by ``root``'s marker, ``None`` when there is no marker; raises when it is corrupt."""
    marker = Path(root) / MARKER_NAME
    if not marker.is_file() or marker.is_symlink():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DemoDirError(f"{marker} is not a readable demo marker ({exc}); delete the folder by hand") from exc
    valid = isinstance(payload, Mapping) and payload.get("kind") == MARKER_KIND
    entries = payload.get("entries") if valid else None
    if not isinstance(entries, list) or not all(map(_valid_entry, entries)):
        raise DemoDirError(f"{marker} is not a valid Namma-Flow demo marker; delete the folder by hand")
    return tuple(entries)


def _adoptable_entries(root: Path) -> tuple[str, ...]:
    """Marker entries of an existing demo, ``()`` for a new or empty folder; refuses any other folder."""
    if not root.exists() and not root.is_symlink():
        return ()
    if not root.is_dir():
        raise DemoDirError(f"DEMO_DIR {root} exists and is not a folder")
    entries = read_marker(root)
    if entries is not None:
        return entries
    if any(root.iterdir()):
        raise DemoDirError(f"DEMO_DIR {root} already holds files and was not created by 'make demo-config' "
                           f"(no {MARKER_NAME} marker); choose a new or empty folder")
    return ()


def _check_entries_inside(root: Path, entries: Sequence[str]) -> None:
    for name in entries:
        path = root / name
        if (path.exists() or path.is_symlink()) and not is_within(path, root):
            raise DemoDirError(f"{path} points outside DEMO_DIR (a symlink); the demo would write through it")


# --------------------------------------------------------------------------- config re-rooting


def rerooted_config(cfg: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """A copy of ``cfg`` with ``project.offline = true`` and every ``paths.*`` entry moved under ``root``.

    Returns ``(config, top-level entry names created under root)``. Paths with ``..`` components
    (which would escape ``root``) and paths that collide with the demo config or marker are refused.
    """
    project = cfg.get("project") or {}
    paths = cfg.get("paths") or {}
    if not isinstance(project, Mapping) or not isinstance(paths, Mapping):
        raise DemoDirError("CONFIG sections 'project' and 'paths' must be mappings")
    new_paths: dict[str, Any] = {}
    entries: set[str] = set()
    for key, value in paths.items():
        if value is None:
            new_paths[key] = None
            continue
        parts = Path(str(value).lstrip("/")).parts
        if ".." in parts:
            raise DemoDirError(f"paths.{key}={value!r} contains '..' and would escape DEMO_DIR")
        if parts and parts[0] in (DEMO_CONFIG_NAME, MARKER_NAME):
            raise DemoDirError(f"paths.{key}={value!r} collides with the demo's own {parts[0]}")
        new_paths[key] = str(root.joinpath(*parts))
        if parts:
            entries.add(parts[0])
    new_cfg = {**cfg, "project": {**project, "offline": True}, "paths": new_paths}
    return new_cfg, tuple(sorted(entries))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DemoDirError(f"cannot read CONFIG {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise DemoDirError(f"CONFIG {path} must hold a mapping, got {type(raw).__name__}")
    return dict(raw)


def prepare_demo(
    root: str | os.PathLike[str],
    source_config: str | os.PathLike[str],
    project: str | os.PathLike[str] = DEFAULT_PROJECT,
    *,
    home: str | os.PathLike[str] | None = None,
) -> DemoPrepared:
    """Check ``root``, then write its marker and the re-rooted copy of ``source_config`` (always regenerated)."""
    import yaml

    from src.utils.runtime import atomic_write_text

    resolved = check_demo_root(root, project, home=home, source_config=source_config)
    demo_cfg, created = rerooted_config(_load_yaml(Path(source_config)), resolved)
    entries = tuple(sorted({*_adoptable_entries(resolved), *created, DEMO_CONFIG_NAME}))
    _check_entries_inside(resolved, entries)
    marker = {"kind": MARKER_KIND, "version": MARKER_VERSION, "project": str(real_path(project)),
              "source_config": str(real_path(source_config)), "entries": list(entries)}
    resolved.mkdir(parents=True, exist_ok=True)
    atomic_write_text(resolved / MARKER_NAME, json.dumps(marker, indent=2) + "\n")
    config_path = atomic_write_text(resolved / DEMO_CONFIG_NAME, yaml.safe_dump(demo_cfg, sort_keys=False))
    return DemoPrepared(root=resolved, config=config_path, entries=entries)


def verify_demo(
    root: str | os.PathLike[str],
    source_config: str | os.PathLike[str],
    project: str | os.PathLike[str] = DEFAULT_PROJECT,
    *,
    home: str | os.PathLike[str] | None = None,
) -> Path:
    """Re-check a prepared demo right before the pipeline runs; returns the demo config path."""
    resolved = check_demo_root(root, project, home=home, source_config=source_config)
    config_path = resolved / DEMO_CONFIG_NAME
    if read_marker(resolved) is None or not config_path.is_file():
        raise DemoDirError(f"DEMO_DIR {resolved} is not a prepared demo (run 'make demo-config' first)")
    return config_path


def clean_demo(
    root: str | os.PathLike[str],
    project: str | os.PathLike[str] = DEFAULT_PROJECT,
    *,
    home: str | os.PathLike[str] | None = None,
) -> DemoCleaned:
    """Delete only what ``demo-config`` / ``offline-demo`` created under ``root`` (the marker's entries)."""
    resolved = check_demo_root(root, project, home=home)
    if not resolved.exists():
        return DemoCleaned(root=resolved, removed=(), kept=(), removed_root=False, existed=False)
    if not resolved.is_dir():
        raise DemoDirError(f"DEMO_DIR {resolved} is not a folder")
    entries = read_marker(resolved)
    if entries is None:
        raise DemoDirError(f"DEMO_DIR {resolved} has no {MARKER_NAME} marker, so it was not created by "
                           "'make demo-config'; nothing was deleted (remove it by hand if you are sure)")
    removed = []
    for name in (e for e in entries if e != MARKER_NAME):
        path = resolved / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            continue
        removed.append(name)
    (resolved / MARKER_NAME).unlink()
    kept = tuple(sorted(p.name for p in resolved.iterdir()))
    if not kept:
        resolved.rmdir()
    return DemoCleaned(root=resolved, removed=tuple(removed), kept=kept, removed_root=not kept, existed=True)


# --------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", required=True, help="demo folder (the Makefile's absolute DEMO_DIR)")
    common.add_argument("--project", default=str(DEFAULT_PROJECT), help="project root to protect")
    common.add_argument("--home", default=None, help=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(prog="python -m src.utils.demo_dir", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("prepare", "check DEMO_DIR, then write its marker and demo config"),
                       ("check", "re-check a prepared DEMO_DIR before the pipeline runs")):
        sub = commands.add_parser(name, parents=[common], help=text)
        sub.add_argument("--config", required=True, help="source config to re-root")
    commands.add_parser("clean", parents=[common], help="delete only what the demo created")
    return parser


def _run(args: argparse.Namespace) -> str:
    if any(ch.isspace() for ch in args.root):
        raise DemoDirError(f"DEMO_DIR={args.root!r} contains whitespace, which make splits; choose another folder")
    if args.command == "prepare":
        done = prepare_demo(args.root, args.config, args.project, home=args.home)
        return f"Wrote demo config {done.config} (marker {done.root / MARKER_NAME}: {', '.join(done.entries)})"
    if args.command == "check":
        return f"Demo folder OK: {verify_demo(args.root, args.config, args.project, home=args.home)}"
    done = clean_demo(args.root, args.project, home=args.home)
    if not done.existed:
        return f"Nothing to clean: {done.root} does not exist"
    kept = f"; left untouched: {', '.join(done.kept)}" if done.kept else f"; removed {done.root}"
    return f"Removed {', '.join(done.removed) or 'nothing'} from {done.root}{kept}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        message = _run(args)
    except (DemoDirError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the Makefile tests
    sys.exit(main())
