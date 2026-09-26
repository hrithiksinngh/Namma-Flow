"""TRAIN / VALIDATION / TEST datasets for training, with consistency checks.

The TRAIN and VALIDATION files (``paths.train_dataset`` / ``paths.val_dataset``) are
required; stage 04 builds all splits when either is missing. The held-out TEST file
(``paths.test_dataset``, written by stage 04 when ``dataset.test_years`` is non-empty) is
optional: when present it must describe the same graph (junction order AND attribute
digest), features, scaler and window geometry as TRAIN, and it is used ONLY for the final
report - never for model selection, calibration or the alert threshold.

All splits must also come from ONE build (same ``build_id``, dataset ``config_hash``, weather
and flood-report fingerprints): an interrupted rebuild can leave a new TRAIN next to old
VAL/TEST files whose labels come from another teacher, so a mixed set is refused (F1-05).
``training.max_val_windows`` subsamples the validation windows scored per EPOCH only; the
final calibration, threshold and reports use the full split (:attr:`Splits.validation_full`).
Recorded dataset paths are project-relative (or file names), never absolute (F3-07).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.data_pipeline import dataset as dataset_module
from src.data_pipeline.dataset import FloodSequenceDataset, dataset_config_hash
from src.training.checkpoint import to_builtin
from src.training.reports import portable_path
from src.training.run_state import TrainingError
from src.utils.config import ConfigError, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

__all__ = ["BUILD_KEYS", "REBUILD_HINT", "Splits", "attributes_digest", "dataset_info", "load_datasets",
           "load_splits", "test_dataset_path", "window_years"]

REBUILD_HINT = "rebuild them with: python src/data_pipeline/04_dataset_builder.py --force"
_GEOMETRY = ("seq_len", "warmup_steps", "lookback_hours", "rolling_windows_h", "num_nodes")
_FINGERPRINTS = ("weather_fingerprint", "reports_fingerprint", "build_id")
# Payload fields that identify ONE dataset build; every split must agree on all of them (F1-05).
BUILD_KEYS = ("build_id", "config_hash", "weather_fingerprint", "reports_fingerprint")


@dataclass(frozen=True)
class Splits:
    """The loaded datasets (``test`` is None when no held-out test file exists).

    ``val`` holds the windows scored every epoch (``training.max_val_windows``); ``val_full``
    is the complete validation split when ``val`` is a subsample (else None).
    """

    train: FloodSequenceDataset
    val: FloodSequenceDataset
    test: FloodSequenceDataset | None = None
    val_full: FloodSequenceDataset | None = None

    @property
    def validation_full(self) -> FloodSequenceDataset:
        """Every validation window: used for the final calibration, threshold and reports."""
        return self.val_full if self.val_full is not None else self.val


def attributes_digest(ds: FloodSequenceDataset) -> str | None:
    """The dataset's graph-attribute digest (X1/X2), or None for payloads built before it existed."""
    value = getattr(ds, "graph_attributes_sha256", None)
    return value if value is not None else ds.payload.get("graph_attributes_sha256")


def test_dataset_path(cfg: Mapping[str, Any]) -> Path | None:
    """The held-out test payload path as stage 04 resolves it (``dataset_paths(cfg)["test"]``:
    ``paths.test_dataset``, or ``test_dataset.pt`` beside a relocated val dataset); None if unknown."""
    resolver = getattr(dataset_module, "dataset_paths", None)
    if resolver is not None:
        try:
            return Path(resolver(cfg)["test"])
        except (ConfigError, KeyError, TypeError) as exc:
            LOGGER.debug("dataset_paths could not resolve the test dataset (%s)", exc)
    try:
        return resolve_path(cfg, "test_dataset")
    except ConfigError:
        return None


def _check_same_build(train_ds: FloodSequenceDataset, other: FloodSequenceDataset, name: str) -> None:
    """Refuse splits from different builds: their labels may come from different teachers (F1-05)."""
    differ = [f"{key} {train_ds.payload.get(key)!r} vs {other.payload.get(key)!r}" for key in BUILD_KEYS
              if train_ds.payload.get(key) != other.payload.get(key)]
    if differ:
        raise TrainingError(f"The train and {name} datasets come from different dataset builds ({'; '.join(differ)}) "
                            f"- e.g. an interrupted rebuild replaced only some split files, so early stopping, "
                            f"calibration and the reported metrics would use labels of another build; "
                            f"{REBUILD_HINT}")


def _check_split(train_ds: FloodSequenceDataset, other: FloodSequenceDataset, name: str) -> None:
    """``other`` must describe the same graph, features, scaler and window geometry as TRAIN."""
    digest_a, digest_b = attributes_digest(train_ds), attributes_digest(other)
    checks = {
        "graph_signature": train_ds.graph_signature == other.graph_signature,
        "graph_attributes_sha256": digest_a is None or digest_b is None or digest_a == digest_b,
        "feature_names": train_ds.feature_names == other.feature_names,
        "scaler": train_ds.scaler == other.scaler,
        "edge_attr": torch.equal(train_ds.edge_attr, other.edge_attr),
        **{attr: getattr(train_ds, attr) == getattr(other, attr) for attr in _GEOMETRY},
    }
    problems = [key for key, ok in checks.items() if not ok]
    if problems:
        raise TrainingError(f"The train and {name} datasets are inconsistent ({', '.join(problems)} differ); "
                            f"{REBUILD_HINT}")
    if (digest_a is None) != (digest_b is None):
        LOGGER.warning("Only one of the train / %s datasets carries a graph-attribute digest; %s", name, REBUILD_HINT)
    _check_same_build(train_ds, other, name)
    split = other.payload.get("split")
    expected = "val" if name == "val" else name
    if split is not None and str(split) != expected:
        LOGGER.warning("The %s dataset file holds split %r; check paths.%s_dataset", name, split, expected)


def _load_test(cfg: Mapping[str, Any], train_ds: FloodSequenceDataset) -> FloodSequenceDataset | None:
    path = test_dataset_path(cfg)
    if path is None or not path.exists():
        LOGGER.warning("No held-out test dataset (%s); the final metrics are reported on the VALIDATION split, "
                       "which also chose the epoch, calibration and threshold (optimistic). Set dataset.test_years "
                       "and %s", path if path is not None else "paths.test_dataset is not configured", REBUILD_HINT)
        return None
    test_ds = FloodSequenceDataset(path)
    _check_split(train_ds, test_ds, "test")
    if len(test_ds) == 0:
        LOGGER.warning("The test dataset %s has no windows; the final metrics use the validation split", path)
        return None
    return test_ds


def load_splits(cfg: Mapping[str, Any], max_val_windows: int | None = None) -> Splits:
    """All splits: TRAIN / VAL from the config paths (built when missing) and the optional TEST."""
    train_path, val_path = resolve_path(cfg, "train_dataset"), resolve_path(cfg, "val_dataset")
    missing = [str(p) for p in (train_path, val_path) if not p.exists()]
    if missing:
        LOGGER.warning("Dataset file(s) %s not found; building the datasets (stage 04) first", ", ".join(missing))
        dataset_module.build_datasets(cfg)
    train_ds = FloodSequenceDataset(train_path)
    val_ds = FloodSequenceDataset(val_path, max_windows=max_val_windows)
    _check_split(train_ds, val_ds, "val")
    val_full = None if max_val_windows is None else FloodSequenceDataset(val_ds.payload)  # no second read
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise TrainingError(f"A dataset split is empty ({len(train_ds)} train / {len(val_ds)} val windows); "
                            f"{REBUILD_HINT}")
    if not train_ds.window_has_flood.any():
        LOGGER.warning("The training windows contain no flooded node-steps: the model cannot learn floods (check the "
                       "hydrology calibration / labels.source and rebuild the datasets)")
    stored, current = train_ds.payload.get("config_hash"), dataset_config_hash(cfg)
    if stored != current:
        LOGGER.warning("The datasets are stale: built with dataset config %s but the current config hashes to %s "
                       "(region/weather/hydrology/labels/dataset settings changed); training continues on the "
                       "existing files - rebuild with: python src/data_pipeline/04_dataset_builder.py", stored, current)
    if val_full is not None and len(val_full) == len(val_ds):
        val_full = None  # max_val_windows >= the split size: nothing was subsampled
    return Splits(train_ds, val_ds, _load_test(cfg, train_ds), val_full)


def load_datasets(cfg: Mapping[str, Any], max_val_windows: int | None = None
                  ) -> tuple[FloodSequenceDataset, FloodSequenceDataset]:
    """TRAIN / VAL datasets (see :func:`load_splits`; kept for callers that ignore the test split)."""
    splits = load_splits(cfg, max_val_windows)
    return splits.train, splits.val


def window_years(ds: FloodSequenceDataset | None) -> list[int] | None:
    """Calendar years (local time) of the windows' last hours."""
    if ds is None or len(ds) == 0:
        return None
    last = np.asarray(ds.window_start_times, dtype=np.int64) + (ds.seq_len - 1) * 3600
    stamps = pd.to_datetime(last, unit="s", utc=True).tz_convert(str(ds.payload.get("timezone") or "UTC"))
    return sorted({int(y) for y in stamps.year})


def _split_info(ds: FloodSequenceDataset | None, path: Path | None) -> dict[str, Any]:
    if ds is None:
        return {"path": portable_path(path), "windows": None, "windows_with_flood": None,
                "pos_rate": None, "years": None, "date_range": None}
    return {"path": portable_path(path), "windows": len(ds),
            "windows_with_flood": int(ds.window_has_flood.sum()), "pos_rate": ds.pos_rate, "years": window_years(ds),
            "date_range": ds.payload.get("stats", {}).get("date_range")}


def dataset_info(cfg: Mapping[str, Any], splits: Splits) -> dict[str, Any]:
    """Provenance of the datasets for ``metrics.json`` (hashes, window counts per split).

    Validation counts are those of the FULL split (the final reports use it); the windows
    scored per epoch are ``val_windows_per_epoch``.
    """
    train_ds, val_ds, test_ds = splits.train, splits.validation_full, splits.test
    stored, current = train_ds.payload.get("config_hash"), dataset_config_hash(cfg)
    parts = {"train": _split_info(train_ds, resolve_path(cfg, "train_dataset")),
             "validation": _split_info(val_ds, resolve_path(cfg, "val_dataset")),
             "test": _split_info(test_ds, test_dataset_path(cfg) if test_ds is not None else None)}
    return to_builtin({
        "train_path": parts["train"]["path"], "val_path": parts["validation"]["path"],
        "test_path": parts["test"]["path"],
        "train_windows": len(train_ds), "val_windows": len(val_ds), "test_windows": parts["test"]["windows"],
        "val_windows_per_epoch": len(splits.val),
        "train_windows_with_flood": parts["train"]["windows_with_flood"],
        "val_windows_with_flood": parts["validation"]["windows_with_flood"],
        "test_windows_with_flood": parts["test"]["windows_with_flood"],
        "train_pos_rate": train_ds.pos_rate, "val_pos_rate": val_ds.pos_rate,
        "test_pos_rate": parts["test"]["pos_rate"],
        "windows": {name: part["windows"] for name, part in parts.items()},
        "years": {name: part["years"] for name, part in parts.items()},
        "date_range": {"train": parts["train"]["date_range"], "val": parts["validation"]["date_range"],
                       "test": parts["test"]["date_range"]},
        "num_nodes": train_ds.num_nodes, "num_edges": train_ds.num_edges,
        "seq_len": train_ds.seq_len, "warmup_steps": train_ds.warmup_steps,
        "label_source": train_ds.payload.get("label_source"),
        "hashes": {"dataset_config_hash": stored, "graph_signature": train_ds.graph_signature,
                   "graph_attributes_sha256": attributes_digest(train_ds),
                   **{key: train_ds.payload.get(key) for key in _FINGERPRINTS}},
        "dataset_config_hash": stored, "current_dataset_config_hash": current, "stale": stored != current,
    })
