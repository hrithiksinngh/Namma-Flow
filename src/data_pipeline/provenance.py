"""Input fingerprints of a dataset build and the reuse (staleness) check of stage 04.

A payload records what it was built from: the dataset config hash and seed, the road-graph
topology (:meth:`GraphArrays.signature`) and attribute digest
(:meth:`GraphArrays.attributes_signature` over the static and edge features), a fingerprint of
the hourly weather record and, when observed reports feed the labels, of the flood-reports
file. :func:`find_reusable` reuses existing payloads only when every one of these matches
the current inputs AND the dataset manifest vouches for the files as one committed build
(:mod:`src.data_pipeline.dataset_manifest`), and otherwise says WHICH input changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data_pipeline import weather
from src.data_pipeline.dataset_config import FORMAT_VERSION, DatasetError, DatasetSettings, dataset_config_hash
from src.data_pipeline.dataset_manifest import manifest_problem
from src.data_pipeline.graph_io import GraphArrays, GraphFormatError, graph_to_arrays, load_graph
from src.data_pipeline.sequence_dataset import load_payload
from src.data_pipeline.windows import local_index
from src.utils.config import resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

FINGERPRINT_CHARS = 16
PRECIP_DECIMALS = 6  # mm; well below any meaningful rain amount, well above CSV round-trip noise
REPORTS_ABSENT = "absent"  # reports_fingerprint when observed labels are configured but the file does not exist


# --------------------------------------------------------------------------- fingerprints


def weather_fingerprint(timestamps: pd.DatetimeIndex, areal_mm: np.ndarray, sources: Mapping[str, int]) -> str:
    """sha256[:16] over the int64 unix timestamps, the float64 areal precipitation and the source counts.

    Precipitation is rounded to ``PRECIP_DECIMALS`` decimals first: the CSV cache round trip
    perturbs values by ~1e-15 mm, and a record fetched (or synthesised) in memory must
    fingerprint the same as the identical record read back from the cache.
    """
    epoch = pd.DatetimeIndex(timestamps).as_unit("s").asi8.astype(np.int64)
    rain = np.round(np.asarray(areal_mm, dtype=np.float64).reshape(-1), PRECIP_DECIMALS) + 0.0  # -0.0 -> 0.0
    if rain.size != epoch.size:
        raise ValueError(f"weather_fingerprint: {rain.size} precipitation values for {epoch.size} timestamps")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(epoch).tobytes())
    digest.update(np.ascontiguousarray(rain).tobytes())
    digest.update(json.dumps(sorted((str(k), int(v)) for k, v in sources.items())).encode("utf-8"))
    return digest.hexdigest()[:FINGERPRINT_CHARS]


def reports_fingerprint(cfg: Mapping[str, Any], label_source: str) -> str | None:
    """sha256[:16] of ``paths.flood_reports_file`` when observed reports feed the labels, else ``None``."""
    if label_source == "simulated":
        return None
    path = resolve_path(cfg, "flood_reports_file")
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return REPORTS_ABSENT
    except OSError as exc:
        raise DatasetError(f"Cannot read the flood reports file {path} ({exc}); labels.source={label_source} "
                           "needs it") from exc
    return hashlib.sha256(data).hexdigest()[:FINGERPRINT_CHARS]


def graph_attributes_digest(arrays: GraphArrays, settings: DatasetSettings) -> str:
    """X1 digest of the graph over ``dataset.static_features`` and ``dataset.edge_features``."""
    return arrays.attributes_signature(settings.static_features, settings.edge_features)


@dataclass(frozen=True)
class WeatherRecord:
    """The hourly areal record a build uses (stage 03 cache / fetch / synthetic fallback)."""

    timestamps: pd.DatetimeIndex
    areal: np.ndarray
    sources: dict[str, int]
    fingerprint: str


def load_weather_record(cfg: Mapping[str, Any], settings: DatasetSettings) -> WeatherRecord:
    """Weather for the configured range via :func:`weather.load_or_fetch_weather`, with its fingerprint."""
    frame = weather.load_or_fetch_weather(cfg)
    areal, ts = weather.areal_series(frame)
    ts = local_index(ts, settings.timezone)
    sources = {str(k): int(v) for k, v in frame["source"].value_counts().items()} if "source" in frame else {}
    return WeatherRecord(ts, areal, sources, weather_fingerprint(ts, areal, sources))


# --------------------------------------------------------------------------- current inputs


@dataclass(frozen=True)
class BuildInputs:
    """Identity of everything a build depends on; ``None`` graph fields mean "cannot be determined"."""

    config_hash: str
    seed: int
    weather_fingerprint: str
    weather_sources: dict[str, int]
    reports_fingerprint: str | None
    graph_signature: dict[str, Any] | None
    graph_attributes_sha256: str | None
    graph_problem: str | None = None


def current_inputs(cfg: Mapping[str, Any], settings: DatasetSettings, record: WeatherRecord,
                   label_source: str, arrays: GraphArrays | None = None) -> BuildInputs:
    """Fingerprints of the inputs as they are now.

    The graph is ``arrays`` when given (the graph stage 04 prepared, possibly re-enriched in
    memory), else it is read from ``paths.graph_file``.
    """
    signature = digest = problem = None
    path = resolve_path(cfg, "graph_file")
    try:
        arrays = graph_to_arrays(load_graph(path)) if arrays is None else arrays
        signature = arrays.signature()
        digest = graph_attributes_digest(arrays, settings)
    except (FileNotFoundError, GraphFormatError, OSError) as exc:
        problem = f"road graph {path} is not readable ({exc})"
    except KeyError as exc:
        problem = f"road graph lacks model attributes ({exc.args[0] if exc.args else exc})"
    return BuildInputs(dataset_config_hash(cfg), settings.seed, record.fingerprint, dict(record.sources),
                       reports_fingerprint(cfg, label_source), signature, digest, problem)


def stale_reasons(payload: Mapping[str, Any], inputs: BuildInputs) -> list[str]:
    """Why ``payload`` does not match ``inputs`` (empty list = current)."""
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        return [f"format_version {version!r} payload (this builder writes {FORMAT_VERSION}; older payloads carry no "
                "input fingerprints)"]
    reasons = []
    if payload.get("seed") != inputs.seed:
        reasons.append(f"seed changed ({payload.get('seed')} -> {inputs.seed})")
    if payload.get("config_hash") != inputs.config_hash:
        reasons.append(f"dataset configuration changed (config hash {payload.get('config_hash')} -> "
                       f"{inputs.config_hash})")
    if inputs.graph_problem:
        reasons.append(inputs.graph_problem)
    elif payload.get("graph_signature") != inputs.graph_signature:
        reasons.append("road graph changed (topology: junctions / street segments differ)")
    elif payload.get("graph_attributes_sha256") != inputs.graph_attributes_sha256:
        reasons.append(f"road graph attributes changed (elevation / drain distance / relative elevation / edge "
                       f"length or grade re-computed by stage 02; digest {payload.get('graph_attributes_sha256')} -> "
                       f"{inputs.graph_attributes_sha256})")
    if payload.get("weather_fingerprint") != inputs.weather_fingerprint:
        built = (payload.get("stats") or {}).get("weather_sources")
        reasons.append(f"weather record changed (fingerprint {payload.get('weather_fingerprint')} -> "
                       f"{inputs.weather_fingerprint}; sources {built} -> {inputs.weather_sources})")
    if payload.get("reports_fingerprint") != inputs.reports_fingerprint:
        reasons.append(f"flood reports changed (fingerprint {payload.get('reports_fingerprint')} -> "
                       f"{inputs.reports_fingerprint})")
    return reasons


def find_reusable(paths: Mapping[str, Path], splits: Sequence[str],
                  inputs: BuildInputs) -> tuple[dict[str, dict] | None, list[str]]:
    """``(payloads, [])`` when every split file exists and matches ``inputs``, else ``(None, reasons)``."""
    missing = [split for split in splits if not paths[split].exists()]
    if missing:
        return None, [f"no existing {'/'.join(missing)} dataset file"]
    payloads: dict[str, dict] = {}
    for split in splits:
        try:
            payload = load_payload(paths[split])
        except (DatasetError, FileNotFoundError) as exc:
            return None, [f"existing {split} dataset is unusable ({exc})"]
        if payload.get("split") != split:
            return None, [f"{paths[split]} holds the {payload.get('split')!r} split, not {split!r}"]
        payloads[split] = payload
    reasons: list[str] = []
    for payload in payloads.values():
        reasons += [reason for reason in stale_reasons(payload, inputs) if reason not in reasons]
    if len({payload.get("build_id") for payload in payloads.values()}) != 1:
        reasons.append(f"the {'/'.join(splits)} files come from different builds (interrupted save?)")
    elif not reasons:
        problem = manifest_problem(paths, splits, {split: p.get("build_id") for split, p in payloads.items()})
        if problem:
            reasons.append(problem)
    return (None, reasons) if reasons else (payloads, [])
