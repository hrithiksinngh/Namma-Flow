"""Reproducibility, device selection and atomic file writes."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _read_umask() -> int:
    """Read the process umask once, at import time, before any worker threads exist."""
    current = os.umask(0)
    os.umask(current)
    return current


# os.umask is process-global: toggling it at write time would race with other threads.
_UMASK = _read_umask()


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if importable)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def resolve_device(name: str | None = "auto"):
    """Map ``auto|cpu|cuda|mps`` to an available ``torch.device`` (never raises)."""
    import torch

    choice = (name or "auto").lower()
    if choice == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; using CPU")
        choice = "cpu"
    if choice == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        LOGGER.warning("MPS requested but not available; using CPU")
        choice = "cpu"
    if choice == "auto":
        # MPS is not selected automatically: several PyG scatter kernels fall back to CPU
        # there and it was no faster in benchmarks for graphs of this size.
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        # mkstemp creates 0600 files; give artifacts the usual umask-derived permissions.
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    return _atomic_write(Path(path), lambda tmp: tmp.write_bytes(data))


def atomic_write_text(path: str | Path, text: str) -> Path:
    return _atomic_write(Path(path), lambda tmp: tmp.write_text(text, encoding="utf-8"))


def atomic_torch_save(obj: Any, path: str | Path) -> Path:
    import torch

    return _atomic_write(Path(path), lambda tmp: torch.save(obj, tmp))
