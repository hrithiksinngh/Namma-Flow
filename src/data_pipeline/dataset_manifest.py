"""All-or-nothing commit of the stage-04 split payloads plus the dataset manifest (F1-05).

:func:`commit_splits` writes every split payload to a temporary file next to its target first
(an interruption there leaves the previous build untouched and consistent), then removes the
old manifest, renames all temporary files into place and writes the new manifest LAST. The
manifest (``dataset_manifest.json`` beside the train payload) records the build id and, per
split, the file name, size and sha256. :func:`manifest_problem` is the consumer-side check:
split files without a manifest that names their build (an interrupted commit, or files mixed
from different builds) are never treated as one consistent build.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

MANIFEST_NAME = "dataset_manifest.json"
MANIFEST_VERSION = 1
_HASH_CHUNK = 1 << 20


def manifest_path(paths: Mapping[str, Path]) -> Path:
    """Where the manifest of the splits in ``paths`` lives: next to the train payload."""
    return Path(paths["train"]).with_name(MANIFEST_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_save(obj: Any, path: Path) -> None:
    import torch

    torch.save(obj, path)


def commit_splits(
    payloads: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    *,
    extra: Mapping[str, Any] | None = None,
    saver: Callable[[Any, Path], None] | None = None,
) -> dict[str, Any]:
    """Write every payload, then publish them together; returns the manifest that was written.

    ``extra`` adds fields (e.g. the config hash and input fingerprints) to the manifest;
    ``saver(obj, path)`` defaults to ``torch.save``. Temporary files never outlive the call,
    also on failure or KeyboardInterrupt.
    """
    save = saver or _torch_save
    build_ids = {payload.get("build_id") for payload in payloads.values()}
    if len(build_ids) != 1 or None in build_ids:
        raise ValueError("commit_splits needs payloads of exactly one build, got build ids "
                         f"{sorted(map(str, build_ids))}")
    temps: dict[str, Path] = {}
    try:
        for split, payload in payloads.items():
            target = Path(paths[split])
            target.parent.mkdir(parents=True, exist_ok=True)
            temps[split] = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
            save(payload, temps[split])  # a plain open() keeps the umask-derived permissions
        entries = {split: {"file": Path(paths[split]).name, "size_bytes": int(tmp.stat().st_size),
                           "sha256": _sha256(tmp)} for split, tmp in temps.items()}
        manifest = {"manifest_version": MANIFEST_VERSION, "build_id": build_ids.pop(),
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "splits": entries, **dict(extra or {})}
        manifest_file = manifest_path(paths)
        manifest_file.unlink(missing_ok=True)  # from here until the new manifest exists the splits are in transition
        for split, tmp in temps.items():
            os.replace(tmp, paths[split])
        atomic_write_text(manifest_file, json.dumps(manifest, indent=2, default=str))
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)
    LOGGER.info("Committed %s datasets of build %s (manifest %s)", "/".join(payloads), manifest["build_id"],
                manifest_file)
    return manifest


def read_manifest(paths: Mapping[str, Path]) -> dict[str, Any] | None:
    """The manifest as a dict, or ``None`` when it is absent or unreadable (logged)."""
    path = manifest_path(paths)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable dataset manifest %s (%s)", path, exc)
        return None
    return data if isinstance(data, dict) else None


def manifest_problem(paths: Mapping[str, Path], splits: Sequence[str], build_ids: Mapping[str, Any]) -> str | None:
    """Why the split files are not one committed build (``None`` when the manifest vouches for them).

    ``build_ids`` maps each split to the ``build_id`` stored in its payload.
    """
    manifest = read_manifest(paths)
    if manifest is None:
        return (f"no readable dataset manifest {manifest_path(paths).name} (the last build was interrupted before "
                "it completed, or was written by an older builder)")
    entries = manifest.get("splits") if isinstance(manifest.get("splits"), Mapping) else {}
    for split in splits:
        entry = entries.get(split)
        if not isinstance(entry, Mapping):
            return f"the dataset manifest does not list the {split} split"
        if build_ids.get(split) != manifest.get("build_id"):
            return (f"the {split} dataset (build {build_ids.get(split)}) is not the build the manifest records "
                    f"({manifest.get('build_id')}): files from different builds")
        size = Path(paths[split]).stat().st_size if Path(paths[split]).exists() else None
        if size != entry.get("size_bytes"):
            return (f"the {split} dataset file changed after the build was committed "
                    f"(size {size} != {entry.get('size_bytes')})")
    return None
