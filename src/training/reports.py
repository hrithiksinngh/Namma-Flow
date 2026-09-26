"""Report files of a training run and the publication of the final model.

:func:`publish_final` makes ``best.pt``, ``metrics.json`` and the other published reports
(``training_history.csv``, ``areal_skill.json``) appear together: all are fully serialised to
temporary files in their target directories first (the slow, fallible part), then moved into
place with back-to-back atomic renames. A crash before that point leaves the previously
published set untouched; readers never see a truncated file.

During training the per-epoch history goes to the hidden staging file
:data:`HISTORY_PARTIAL_NAME` in the reports directory, never to the published
``training_history.csv``, so an in-progress, crashed or aborted run never shows its curve next
to the previously published model (F2-03).
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from src.utils.config import project_root
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

__all__ = ["HISTORY_FIELDS", "HISTORY_NAME", "HISTORY_PARTIAL_NAME", "discard_partial_history", "history_csv",
           "portable_path", "publish_final", "write_history", "write_json"]


def _read_umask() -> int:
    """Read the process umask once, at import time (toggling it later would race with threads)."""
    current = os.umask(0)
    os.umask(current)
    return current


_UMASK = _read_umask()

HISTORY_FIELDS: tuple[str, ...] = (
    "epoch", "train_loss", "val_loss", "pr_auc", "roc_auc", "f2", "f2_threshold", "brier", "lr", "n_steps",
    "n_skipped", "grad_norm", "train_time_s", "val_time_s", "epoch_time_s", "monitor_value", "improved",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not write %s (%s); the metrics are still stored in best.pt", path, exc)


HISTORY_NAME = "training_history.csv"
HISTORY_PARTIAL_NAME = ".training_history.partial.csv"


def portable_path(path: str | Path | None) -> str | None:
    """``path`` relative to the project root when inside it, else only its file name (F3-07).

    Used for every path recorded in a checkpoint or a published report, so sharing the model
    files never leaks the user name or the directory layout of the training machine.
    """
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root().resolve()).as_posix()
    except (ValueError, OSError):
        return candidate.name or None


def history_csv(history: Sequence[Mapping[str, Any]]) -> str:
    """The per-epoch history as CSV text (columns :data:`HISTORY_FIELDS`)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in history:
        writer.writerow({k: "" if record.get(k) is None else record.get(k) for k in HISTORY_FIELDS})
    return buffer.getvalue()


def write_history(reports_dir: Path, history: Sequence[Mapping[str, Any]]) -> Path | None:
    """Write the in-progress history to the STAGING file (:data:`HISTORY_PARTIAL_NAME`).

    The published ``training_history.csv`` is only replaced by :func:`publish_final`.
    """
    path = Path(reports_dir) / HISTORY_PARTIAL_NAME
    try:
        atomic_write_text(path, history_csv(history))
    except OSError as exc:
        LOGGER.warning("Could not write the training history to %s (%s)", reports_dir, exc)
        return None
    return path


def discard_partial_history(reports_dir: Path) -> None:
    """Remove the staging history once the final one is published."""
    try:
        (Path(reports_dir) / HISTORY_PARTIAL_NAME).unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.debug("Could not remove the staged history in %s (%s)", reports_dir, exc)


def _stage(path: Path, writer: Callable[[Path], None]) -> Path:
    """Write via ``writer`` to a temporary file next to ``path`` and return it (caller renames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".staged")
    os.close(fd)
    tmp = Path(name)
    try:
        writer(tmp)
        os.chmod(tmp, 0o666 & ~_UMASK)  # mkstemp creates 0600 files
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _stage_text(path: Path, text: str) -> Path | None:
    """Stage ``text`` next to ``path``; None (WARNING) when the directory is not writable."""
    try:
        return _stage(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    except OSError as exc:
        LOGGER.warning("Could not write %s (%s); the previous file (if any) is kept", path, exc)
        return None


def publish_final(checkpoint: Mapping[str, Any], checkpoint_path: Path, metrics: Mapping[str, Any],
                  metrics_path: Path, extras: Mapping[Path, str] | None = None) -> bool:
    """Publish ``checkpoint`` (required), ``metrics`` and the ``extras`` text files together.

    ``extras`` maps a target path to its text (e.g. ``training_history.csv``,
    ``areal_skill.json``); like ``metrics.json`` they are best effort. Returns True when
    ``metrics.json`` was published too; an unwritable reports directory only logs a WARNING
    (the metrics are inside the checkpoint as well).
    """
    staged_ckpt = _stage(Path(checkpoint_path), lambda tmp: torch.save(dict(checkpoint), tmp))
    staged: list[tuple[Path, Path]] = []
    try:
        text = json.dumps(metrics, indent=2, allow_nan=False)
    except ValueError as exc:
        LOGGER.warning("Could not serialise the metrics (%s); they are still stored in %s", exc, checkpoint_path)
        text = None
    metrics_tmp = None if text is None else _stage_text(Path(metrics_path), text)
    if metrics_tmp is not None:
        staged.append((metrics_tmp, Path(metrics_path)))
    for target, body in (extras or {}).items():
        tmp = _stage_text(Path(target), body)
        if tmp is not None:
            staged.append((tmp, Path(target)))
    try:
        os.replace(staged_ckpt, checkpoint_path)
        for tmp, target in staged:
            os.replace(tmp, target)
    finally:
        for tmp in (staged_ckpt, *(t for t, _ in staged)):
            if tmp.exists():
                tmp.unlink()
    return metrics_tmp is not None
