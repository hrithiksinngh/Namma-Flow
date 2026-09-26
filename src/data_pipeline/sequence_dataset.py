"""Loading dataset payloads safely and serving their windows to PyTorch.

:func:`load_payload` only ever uses the safe ``weights_only`` unpickler: a payload written
by :func:`src.data_pipeline.dataset.build_datasets` holds nothing but tensors and plain
Python values, so a file the safe loader rejects is corrupt or foreign and is never handed to
the full unpickler (which would run arbitrary code from a malicious file).
:class:`FloodSequenceDataset` computes features on the fly with
:func:`~src.data_pipeline.features.build_node_features`, the same function inference uses.
"""

from __future__ import annotations

import functools
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data_pipeline.dataset_config import BUILD_HINT, FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS, DatasetError
from src.data_pipeline.features import FeatureScaler, build_node_features, validate_windows
from src.data_pipeline.windows import break_counts, window_positive_counts
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

V1_PAYLOAD_KEYS = (
    "format_version", "split", "node_ids", "lon", "lat", "edge_index", "edge_attr", "edge_attr_raw", "static_raw",
    "static_feature_names", "feature_names", "rain", "labels", "depth", "timestamps", "timezone", "window_starts",
    "seq_len", "warmup_steps", "lookback_hours", "rolling_windows_h", "scaler", "label_source", "flood_threshold_m",
    "graph_signature", "config_hash", "stats",
)
INPUT_FINGERPRINT_KEYS = ("graph_attributes_sha256", "weather_fingerprint", "reports_fingerprint")
REQUIRED_PAYLOAD_KEYS = (*V1_PAYLOAD_KEYS, *INPUT_FINGERPRINT_KEYS)  # format_version 2


# --------------------------------------------------------------------------- loading


def _unpickling_reason(exc: BaseException) -> str:
    """The safe loader's own diagnosis, without torch's advice to retry with the full unpickler."""
    for line in str(exc).splitlines():
        if "WeightsUnpickler error" in line:
            return line.strip()
    return type(exc).__name__


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load a dataset payload written by :func:`~src.data_pipeline.dataset.build_datasets`.

    Uses ``torch.load(weights_only=True)`` only. Raises ``FileNotFoundError`` for a missing
    file and :class:`DatasetError` for a corrupt, truncated or foreign file (including one
    holding Python objects the safe loader refuses - it is never retried unsafely).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}; build it with: "
                                "python src/data_pipeline/04_dataset_builder.py")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise DatasetError(f"{path} is not a Namma-Flow dataset: the safe (weights-only) loader refused it "
                           f"({_unpickling_reason(exc)}). The file is corrupt or truncated, or it holds Python objects "
                           f"the dataset builder never writes, so it is not loaded with the full unpickler; "
                           f"{BUILD_HINT}") from exc
    except Exception as exc:  # noqa: BLE001 - torch raises many types for truncated/corrupt files
        raise DatasetError(f"Could not read dataset {path} ({type(exc).__name__}: {exc}); the file is corrupt or "
                           f"truncated - {BUILD_HINT}") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"Dataset {path} holds a {type(payload).__name__}, not a payload dict - {BUILD_HINT}")
    return payload


def _numpy(value: Any, dtype: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def validate_payload(payload: Any) -> None:
    """Check the format version and the keys a payload of that version must carry."""
    if not isinstance(payload, Mapping):
        raise DatasetError(f"A dataset payload must be a mapping, got {type(payload).__name__}")
    version = payload.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise DatasetError(f"Dataset format_version {version!r} is not supported (this code reads "
                           f"{list(SUPPORTED_FORMAT_VERSIONS)}, writes {FORMAT_VERSION}); {BUILD_HINT}")
    required = REQUIRED_PAYLOAD_KEYS if version == FORMAT_VERSION else V1_PAYLOAD_KEYS
    missing = [key for key in required if key not in payload]
    if missing:
        raise DatasetError(f"Dataset payload is missing keys {missing}; {BUILD_HINT}")
    if version != FORMAT_VERSION:
        LOGGER.info("Dataset payload (split %s) has format_version %s: it carries no input fingerprints "
                    "(graph_attributes_sha256 is None); rebuilding it is recommended", payload.get("split"), version)


def _subset(starts: np.ndarray, max_windows: int | None) -> np.ndarray:
    if max_windows is None:
        return starts
    if isinstance(max_windows, bool) or not isinstance(max_windows, (int, np.integer)) or max_windows < 1:
        raise ValueError(f"max_windows must be a positive integer or None, got {max_windows!r}")
    if max_windows >= starts.size:
        return starts
    return starts[np.unique(np.linspace(0, starts.size - 1, int(max_windows)).round().astype(np.int64))]


# --------------------------------------------------------------------------- dataset


class FloodSequenceDataset(Dataset):
    """Windows of a split; items are computed on the fly from the stored node rain.

    ``item = {"x": float32 [T, N, F], "y": float32 [T, N], "mask": bool [T] (False for warm-up
    steps), "start": unix s of the window's first hour, "index": position in this dataset}``.
    Besides the tensors it exposes ``split`` (``train`` / ``val`` / ``test``) and
    ``graph_attributes_sha256`` (the X1 digest of the graph it was built from; ``None`` for a
    format-1 payload).
    """

    def __init__(self, source: str | Path | Mapping[str, Any], max_windows: int | None = None) -> None:
        payload = load_payload(source) if isinstance(source, (str, Path)) else source
        validate_payload(payload)
        self.payload = dict(payload)
        self.split = str(payload["split"])
        self.graph_attributes_sha256 = payload.get("graph_attributes_sha256")
        self.seq_len, self.warmup_steps = int(payload["seq_len"]), int(payload["warmup_steps"])
        self.lookback_hours = int(payload["lookback_hours"])
        self.rolling_windows_h = validate_windows(payload["rolling_windows_h"])
        self._rain, self._labels = _numpy(payload["rain"], np.float32), _numpy(payload["labels"], np.uint8)
        self._timestamps = _numpy(payload["timestamps"], np.int64)
        self._static = _numpy(payload["static_raw"], np.float32)
        self._check_arrays(_numpy(payload["window_starts"], np.int64))
        self._starts = _subset(_numpy(payload["window_starts"], np.int64), max_windows)
        self.scaler = FeatureScaler.from_dict(payload["scaler"])
        self.edge_index = torch.as_tensor(_numpy(payload["edge_index"], np.int64)).reshape(2, -1)
        self.edge_attr = torch.as_tensor(_numpy(payload["edge_attr"], np.float32)).reshape(-1, 2)
        if self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise DatasetError(f"edge_attr has {self.edge_attr.shape[0]} rows for {self.edge_index.shape[1]} edges")
        self.num_nodes, self.num_edges = int(self._rain.shape[1]), int(self.edge_index.shape[1])
        self.feature_names, self.num_features = list(payload["feature_names"]), len(payload["feature_names"])
        self.static_feature_names = list(payload["static_feature_names"])
        self.graph_signature, self.node_ids = dict(payload["graph_signature"]), list(payload["node_ids"])
        self.window_start_times = self._timestamps[self._starts].copy()  # unix s of each window's first hour
        self._mask = torch.arange(self.seq_len) >= self.warmup_steps
        counts = window_positive_counts(self._labels, self._starts, self.seq_len, self.warmup_steps)
        self.window_has_flood = counts > 0
        scored = self._starts.size * (self.seq_len - self.warmup_steps) * self.num_nodes
        self.pos_rate = float(counts.sum() / scored) if scored else 0.0

    def _check_arrays(self, starts: np.ndarray) -> None:
        n_hours, n_nodes = self._rain.shape if self._rain.ndim == 2 else (-1, -1)
        if (self._rain.ndim != 2 or self._labels.shape != (n_hours, n_nodes) or self._timestamps.shape != (n_hours,)
                or n_nodes != len(self.payload["node_ids"]) or self._static.shape[0] != n_nodes):
            raise DatasetError(f"Inconsistent dataset array shapes: rain {self._rain.shape}, labels "
                               f"{self._labels.shape}, timestamps {self._timestamps.shape}, static "
                               f"{self._static.shape}, {len(self.payload['node_ids'])} node ids - {BUILD_HINT}")
        if starts.ndim != 1 or (starts - self.lookback_hours < 0).any() or (starts + self.seq_len > n_hours).any():
            raise DatasetError(f"window_starts must leave lookback_hours={self.lookback_hours} before and seq_len="
                               f"{self.seq_len} hours after each start within {n_hours} stored hours - {BUILD_HINT}")
        breaks = break_counts(self._timestamps)
        if (breaks[starts + self.seq_len - 1] != breaks[starts - self.lookback_hours]).any():
            raise DatasetError(f"Some windows are not contiguous hourly sequences in the stored timestamps - "
                               f"{BUILD_HINT}")

    def __len__(self) -> int:
        return int(self._starts.size)

    def _position(self, i: int) -> int:
        n = len(self)
        index = int(i) + n if int(i) < 0 else int(i)
        if not 0 <= index < n:
            raise IndexError(f"window index {i} out of range for {n} windows")
        return index

    def __getitem__(self, i: int) -> dict[str, Any]:
        index = self._position(i)
        s = int(self._starts[index])
        chunk = self._rain[s - self.lookback_hours: s + self.seq_len]
        x = build_node_features(self._static, chunk, self.lookback_hours, self.rolling_windows_h, self.scaler)
        y = self._labels[s: s + self.seq_len].astype(np.float32)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": self._mask.clone(),
                "start": int(self._timestamps[s]), "index": index}

    # ------------------------------------------------------------------ helpers
    def window_timestamps(self, i: int) -> pd.DatetimeIndex:
        """Local timestamps of the ``seq_len`` hours of window ``i``."""
        s = int(self._starts[self._position(i)])
        stamps = pd.to_datetime(self._timestamps[s: s + self.seq_len], unit="s", utc=True)
        return pd.DatetimeIndex(stamps).tz_convert(str(self.payload["timezone"]))

    def window_depth(self, i: int) -> np.ndarray:
        """Simulated water depth (m) of window ``i`` as float32 ``[T, N]`` (physics baseline)."""
        s = int(self._starts[self._position(i)])
        return _numpy(self.payload["depth"][s: s + self.seq_len], np.float32)


# --------------------------------------------------------------------------- batching


def collate_windows(batch: list[dict], edge_index: Any, edge_attr: Any, num_nodes: int) -> dict[str, Any]:
    """Stack ``B`` windows into one disjoint-union graph per hour: ``x [T, B*N, F]``, ``y [T, B*N]``,
    ``mask [T]``, ``edge_index [2, B*E]`` (window ``b`` offset by ``b*N``), ``edge_attr [B*E, 2]``,
    ``batch_size``, ``num_nodes``, ``start [B]`` and ``index [B]``."""
    if not batch:
        raise ValueError("collate_windows got an empty batch")
    shapes = {tuple(item["x"].shape) for item in batch}
    if len(shapes) != 1:
        raise ValueError(f"all windows in a batch must have the same shape, got {sorted(shapes)}")
    x = torch.stack([torch.as_tensor(item["x"]) for item in batch])
    size, steps, nodes, features = x.shape
    if nodes != num_nodes:
        raise ValueError(f"windows have {nodes} nodes but num_nodes={num_nodes}")
    mask = torch.as_tensor(batch[0]["mask"], dtype=torch.bool)
    if any(not torch.equal(torch.as_tensor(item["mask"], dtype=torch.bool), mask) for item in batch[1:]):
        raise ValueError("all windows in a batch must share the same warm-up mask")
    edges = torch.as_tensor(edge_index, dtype=torch.long).reshape(2, -1)
    attrs = torch.as_tensor(edge_attr, dtype=torch.float32)
    if attrs.shape[0] != edges.shape[1]:
        raise ValueError(f"edge_attr has {attrs.shape[0]} rows but edge_index has {edges.shape[1]} edges")
    offsets = (torch.arange(size, dtype=torch.long) * nodes).repeat_interleave(edges.shape[1])
    y = torch.stack([torch.as_tensor(item["y"], dtype=torch.float32) for item in batch])
    return {
        "x": x.permute(1, 0, 2, 3).reshape(steps, size * nodes, features),
        "y": y.permute(1, 0, 2).reshape(steps, size * nodes), "mask": mask,
        "edge_index": edges.repeat(1, size) + offsets, "edge_attr": attrs.repeat(size, 1),
        "batch_size": size, "num_nodes": nodes,
        "start": torch.tensor([int(item["start"]) for item in batch], dtype=torch.long),
        "index": torch.tensor([int(item["index"]) for item in batch], dtype=torch.long),
    }


def make_loader(dataset: FloodSequenceDataset, batch_size: int, shuffle: bool, sampler: Sampler | None = None,
                num_workers: int = 0) -> DataLoader:
    """DataLoader over ``dataset`` batching windows with :func:`collate_windows`."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if isinstance(num_workers, bool) or not isinstance(num_workers, (int, np.integer)) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, got {num_workers!r}")
    if not all(hasattr(dataset, a) for a in ("edge_index", "edge_attr", "num_nodes")):
        raise TypeError(f"make_loader needs a FloodSequenceDataset, got {type(dataset).__name__}")
    if sampler is not None and shuffle:
        LOGGER.warning("make_loader: shuffle is ignored because a sampler is given")
        shuffle = False
    collate = functools.partial(collate_windows, edge_index=dataset.edge_index, edge_attr=dataset.edge_attr,
                                num_nodes=dataset.num_nodes)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, sampler=sampler,
                      num_workers=int(num_workers), collate_fn=collate)
