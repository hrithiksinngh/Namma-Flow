"""Stage 04 — training / validation / test dataset builder (and the public dataset API).

:func:`build_datasets` downscales the areal rainfall record (stage 03) onto the enriched road
graph (stages 01/02) and simulates flood labels (module D), both over the FULL series so
hydrologic state carries across the record; selects in-season wet (plus a seeded share of
dry) windows of ``seq_len`` hours with ``lookback_hours`` of history, split by the calendar
year of their last hour into train / val (``dataset.val_years``) / test
(``dataset.test_years``, optional); fits the :class:`~src.data_pipeline.features.FeatureScaler`
on TRAIN hours only; and stores per split only the hours its windows need.

Payloads are reused only when the format, the dataset configuration, the seed, the road
graph (topology AND the static/edge attribute digest), the weather record and - for observed
labels - the flood-reports file are all unchanged and the dataset manifest vouches for the
files (:mod:`src.data_pipeline.provenance`); a rebuild logs which input changed. The graph is
prepared BEFORE that check: a graph lacking model attributes is completed (derived attributes
from the stored elevations when possible) and a graph whose ``enrichment_config_hash`` differs
from the current elevation / drains / region settings is re-enriched, never degrading real
SRTM / OSM inputs to synthetic fallbacks (:mod:`src.data_pipeline.enrichment`). All split
files are committed together, the manifest last (:mod:`src.data_pipeline.dataset_manifest`).
Loading and batching live in :mod:`src.data_pipeline.sequence_dataset` and are re-exported here.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.data_pipeline import rain_field
from src.data_pipeline.dataset_config import (  # noqa: F401 - re-exported public API
    BUILD_HINT,
    DEFAULT_MAX_ABS_GRADE,
    DEFAULTS,
    FORMAT_VERSION,
    HASH_KEYS,
    HASH_SECTIONS,
    SPLITS,
    DatasetError,
    DatasetSettings,
    dataset_config_hash,
    dataset_hash_inputs,
    dataset_paths,
)
from src.data_pipeline.dataset_manifest import (  # noqa: F401 - re-exported public API
    MANIFEST_NAME,
    commit_splits,
    manifest_problem,
    read_manifest,
)
from src.data_pipeline.features import EDGE_FEATURES, FeatureScaler, dynamic_feature_names, dynamic_features_at
from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays, load_graph, save_graph
from src.data_pipeline.provenance import (
    BuildInputs,
    WeatherRecord,
    current_inputs,
    find_reusable,
    graph_attributes_digest,
    load_weather_record,
)
from src.data_pipeline.sequence_dataset import (  # noqa: F401 - re-exported public API
    REQUIRED_PAYLOAD_KEYS,
    V1_PAYLOAD_KEYS,
    FloodSequenceDataset,
    collate_windows,
    load_payload,
    make_loader,
    validate_payload,
)
from src.data_pipeline.windows import (  # noqa: F401 - re-exported public API
    WindowSelection,
    break_counts,
    select_windows,
    window_positive_counts,
)
from src.utils.config import ConfigError, get_section, project_root, resolve_path
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

LABEL_DEFAULTS: dict[str, Any] = {"source": "simulated", "report_snap_radius_m": 150.0, "report_window_hours": 3}
LABEL_SOURCES = ("simulated", "observed", "hybrid")
ENRICHMENT_ATTRS = frozenset({"elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink"})
GRAPH_PROVENANCE_KEYS = ("source", "elevation_source", "drain_source", "osm_base_utc", "enrichment_config_hash")
_DOWNGRADE_HINT = ("Restore the DEM / waterway caches or run online, then re-run stage 02 "
                   "(python src/data_pipeline/02_elevation_engine.py; --allow-synthetic accepts the fallback)")
_FLOAT16_MAX = 65000.0
FORCED = "forced rebuild (--force)"


# --------------------------------------------------------------------------- inputs: graph, labels


def _missing_graph_attrs(arrays: GraphArrays, settings: DatasetSettings) -> list[str]:
    unknown = [f for f in settings.static_features if f not in arrays.node_attrs and f not in ENRICHMENT_ATTRS]
    if unknown:
        raise ConfigError(f"dataset.static_features {unknown} are not node attributes of the road graph and no "
                          f"pipeline stage produces them; available: {sorted(arrays.node_attrs)}")
    missing = [f for f in settings.static_features
               if f not in arrays.node_attrs or not np.isfinite(arrays.node_attrs[f]).all()]
    if arrays.num_edges:
        missing += [f for f in EDGE_FEATURES
                    if f not in arrays.edge_attrs or not np.isfinite(arrays.edge_attrs[f]).all()]
    return missing


def _reenrich(G: nx.DiGraph, cfg: Mapping[str, Any], missing: list[str], drift: str | None,
              path: Path) -> nx.DiGraph:
    """Complete / refresh the graph's enrichment without ever degrading real inputs (F1-02, F1-03)."""
    from src.data_pipeline.enrichment import can_recompute_derived, check_downgrade, recompute_derived

    if can_recompute_derived(G, missing, drift):
        LOGGER.warning("Road graph lacks %s; recomputing them from its stored elevations", missing)
        return recompute_derived(G, cfg)
    if drift:
        LOGGER.warning("Road graph %s: %s; re-enriching it with the elevation & drainage engine (stage 02)", path,
                       drift)
    else:
        LOGGER.warning("Road graph lacks %s; enriching it with the elevation & drainage engine (stage 02)", missing)
    from src.data_pipeline.elevation import enrich_graph

    enriched = enrich_graph(G, cfg)
    check_downgrade(G, enriched, allow_synthetic=False, target=f"the road graph {path}", error=DatasetError,
                    hint=_DOWNGRADE_HINT)
    return enriched


def _prepare_graph(cfg: Mapping[str, Any], settings: DatasetSettings) -> tuple[nx.DiGraph, GraphArrays]:
    """Load ``paths.graph_file`` (building it with stage 01 if absent); complete / refresh its enrichment."""
    from src.data_pipeline.elevation_settings import enrichment_drift

    path = resolve_path(cfg, "graph_file")
    if path.exists():
        G = load_graph(path)
    else:
        LOGGER.warning("Road graph %s not found; running stage 01 (network extraction) first", path)
        from src.data_pipeline.network import extract_network

        G = extract_network(cfg)
    arrays = graph_to_arrays(G)
    missing = _missing_graph_attrs(arrays, settings)
    drift = enrichment_drift(G, cfg)
    if not missing and not drift:
        return G, arrays
    G = _reenrich(G, cfg, missing, drift, path)
    try:
        save_graph(G, path)
        G = load_graph(path)
    except OSError as exc:
        LOGGER.warning("Could not save the enriched graph to %s (%s); continuing in memory", path, exc)
    arrays = graph_to_arrays(G)
    still = _missing_graph_attrs(arrays, settings)
    if still:
        raise DatasetError(f"Road graph still lacks {still} after enrichment; re-run stages 01/02")
    return G, arrays


def _load_simulator() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    try:
        from src.hydrology.simulator import simulate_labels
    except Exception as exc:  # noqa: BLE001 - ImportError or any error raised while importing module D
        raise DatasetError(
            f"The flood-label simulator src.hydrology.simulator is unavailable ({type(exc).__name__}: {exc}); "
            "it is required to build datasets (labels are simulated)"
        ) from exc
    return simulate_labels


def _clean_labels(labels: Any, depth: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Validate simulator output without full-record float64 copies (the record is ~61 k x 1 k)."""
    lab, dep = np.asarray(labels), np.asarray(depth, dtype=np.float32)
    if lab.shape != shape or dep.shape != shape:
        raise DatasetError(f"simulate_labels returned labels {lab.shape} / depth {dep.shape}; expected shape {shape}")
    if lab.dtype.kind == "f":
        lab = np.nan_to_num(lab, nan=0.0, posinf=1.0, neginf=0.0)
    if lab.size and lab.dtype.kind != "b" and (lab.min() < 0 or lab.max() > 1):
        LOGGER.warning("simulate_labels returned non-binary labels; values > 0 are treated as flooded")
    bad = ~(np.isfinite(dep) & (dep >= 0))
    if bad.any():
        LOGGER.warning("simulate_labels returned %d non-finite/negative depth values; set to 0 m", int(bad.sum()))
        dep = np.where(bad, np.float32(0.0), dep)  # new array: never modify the simulator's output in place
    return (lab if lab.dtype == np.uint8 and lab.max(initial=0) <= 1 else (lab > 0).astype(np.uint8)), dep


def _label_source(cfg: Mapping[str, Any]) -> str:
    source = str(get_section(cfg, "labels", LABEL_DEFAULTS)["source"]).strip().lower()
    if source not in LABEL_SOURCES:
        raise ConfigError(f"labels.source must be one of {list(LABEL_SOURCES)}, got {source!r}")
    return source


def _apply_observed(source: str, labels: np.ndarray, ts: pd.DatetimeIndex, arrays: GraphArrays,
                    cfg: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Merge observed flood reports per ``labels.source``; returns ``(labels, effective source)``."""
    if source == "simulated":
        return labels, source
    try:
        from src.hydrology.observed import load_flood_reports, merge_observed_labels
    except Exception as exc:  # noqa: BLE001 - module E must not crash on an unfinished module D
        if source == "observed":
            raise DatasetError(f"labels.source=observed needs src.hydrology.observed ({type(exc).__name__}: {exc})") \
                from exc
        LOGGER.warning("labels.source=hybrid but src.hydrology.observed is unavailable (%s); using simulated labels "
                       "only", exc)
        return labels, "simulated"
    reports = load_flood_reports(resolve_path(cfg, "flood_reports_file"), str(ts.tz))
    if len(reports) == 0:
        LOGGER.warning("labels.source=%s but there are no flood reports in %s", source,
                       resolve_path(cfg, "flood_reports_file"))
    base = np.zeros_like(labels) if source == "observed" else labels
    merged = np.asarray(merge_observed_labels(base, ts, arrays, reports, cfg))
    if merged.shape != labels.shape:
        raise DatasetError(f"merge_observed_labels returned shape {merged.shape}, expected {labels.shape}")
    return (np.nan_to_num(merged) > 0 if merged.dtype.kind == "f" else merged > 0).astype(np.uint8), source


@dataclass(frozen=True)
class _Record:
    """The full hourly record: areal + node rain, labels and depth (row-aligned)."""

    timestamps: pd.DatetimeIndex
    areal: np.ndarray
    rain: np.ndarray
    labels: np.ndarray
    depth: np.ndarray
    label_source: str
    weather_sources: dict[str, int]


def _build_record(cfg: Mapping[str, Any], arrays: GraphArrays, settings: DatasetSettings, weather_record: WeatherRecord,
                  simulate: Callable[..., tuple[np.ndarray, np.ndarray]]) -> _Record:
    source = _label_source(cfg)
    clock = time.perf_counter()
    ts, areal = weather_record.timestamps, weather_record.areal
    needed = settings.lookback_hours + settings.seq_len
    if len(ts) < needed:
        raise DatasetError(f"The weather record has {len(ts)} hours but one window needs lookback_hours + seq_len = "
                           f"{needed}; widen weather.start_date / weather.end_date")
    rain = np.asarray(rain_field.downscale_rainfall(areal, ts, arrays.lon, arrays.lat, cfg), dtype=np.float32)
    LOGGER.info("Rain field: %d hours x %d nodes in %.1f s", len(ts), arrays.num_nodes, time.perf_counter() - clock)
    clock = time.perf_counter()
    labels, depth = _clean_labels(*simulate(arrays, rain, cfg), shape=rain.shape)
    labels, source = _apply_observed(source, labels, ts, arrays, cfg)
    LOGGER.info("Labels (%s): %.3f %% flooded node-hours, simulated in %.1f s", source, 100 * labels.mean(),
                time.perf_counter() - clock)
    return _Record(ts, areal, rain, labels, depth, source, dict(weather_record.sources))


# --------------------------------------------------------------------------- scaler & payloads


def _union_hours(starts: np.ndarray, before: int, after: int, n_total: int) -> np.ndarray:
    """Sorted indices covered by ``[s - before, s + after)`` for every start ``s``."""
    diff = np.zeros(n_total + 1, dtype=np.int64)
    np.add.at(diff, starts - before, 1)
    np.add.at(diff, starts + after, -1)
    return np.flatnonzero(np.cumsum(diff[:-1]) > 0)


def _dynamic_samples(rain: np.ndarray, starts: np.ndarray, settings: DatasetSettings) -> np.ndarray:
    """Raw dynamic features ``[M, N, D]`` at (a seeded subsample of) the TRAIN window hours."""
    hours = _union_hours(starts, 0, settings.seq_len, rain.shape[0])
    budget = max(1, settings.scaler_max_samples // max(rain.shape[1], 1))
    if hours.size > budget:
        hours = np.sort(np.random.default_rng(settings.seed + 1).choice(hours, budget, replace=False))
    return dynamic_features_at(rain, hours, settings.rolling_windows_h)


def _cfg_float(cfg: Mapping[str, Any], section: str, key: str, default: float) -> float:
    return float(get_section(cfg, section, {key: default})[key])


def _iso(epoch_s: int, tz: str) -> str:
    return pd.Timestamp(int(epoch_s), unit="s", tz="UTC").tz_convert(tz).isoformat()


def _split_stats(labels: np.ndarray, areal: np.ndarray, epoch: np.ndarray, local_starts: np.ndarray,
                 settings: DatasetSettings, years: np.ndarray) -> dict[str, Any]:
    seq, warm, n_windows = settings.seq_len, settings.warmup_steps, int(local_starts.size)
    counts = window_positive_counts(labels, local_starts, seq, warm)
    scored = n_windows * (seq - warm) * labels.shape[1]
    cumulative = np.concatenate([[0.0], np.cumsum(areal)])
    window_rain = cumulative[local_starts + seq] - cumulative[local_starts]
    return {
        "n_windows": n_windows, "n_hours": int(labels.shape[0]), "n_nodes": int(labels.shape[1]),
        "pos_rate": float(counts.sum() / scored) if scored else 0.0,
        "n_windows_with_flood": int((counts > 0).sum()),
        "flood_window_rate": float((counts > 0).mean()) if n_windows else 0.0,
        "stored_flood_rate": float(labels.mean()) if labels.size else 0.0,
        "peak_hour_flood_fraction": float(labels.mean(axis=1).max()) if labels.size else 0.0,
        "n_wet_windows": int((window_rain >= settings.wet_window_min_mm - 1e-9).sum()),
        "mean_window_rain_mm": float(window_rain.mean()) if n_windows else 0.0,
        "date_range": [_iso(epoch[local_starts[0]], settings.timezone),
                       _iso(epoch[local_starts[-1] + seq - 1], settings.timezone)] if n_windows else [],
        "years": sorted({int(y) for y in years}),
    }


def _split_payload(split: str, starts: np.ndarray, record: _Record, settings: DatasetSettings,
                   common: Mapping[str, Any], selection: WindowSelection) -> dict[str, Any]:
    """Compact payload of one split: only the hours its windows (plus lookback) need."""
    lookback, seq = settings.lookback_hours, settings.seq_len
    hours = _union_hours(starts, lookback, seq, len(record.timestamps))
    position = np.full(len(record.timestamps), -1, dtype=np.int64)
    position[hours] = np.arange(hours.size)
    local = position[starts]
    epoch = record.timestamps.as_unit("s").asi8[hours].astype(np.int64)
    breaks = break_counts(epoch)
    if (local - lookback < 0).any() or (breaks[local + seq - 1] != breaks[local - lookback]).any():
        raise RuntimeError(f"internal error: {split} windows are not contiguous in the stored hours")
    labels = record.labels[hours]
    stats = _split_stats(labels, record.areal[hours], epoch, local, settings,
                         record.timestamps.year.to_numpy()[starts + seq - 1])
    stats.update(selection=selection.counts(), label_source=record.label_source, weather_sources=record.weather_sources)
    depth = np.clip(record.depth[hours], 0.0, _FLOAT16_MAX).astype(np.float16)
    return {
        **common, "split": split, "stats": stats,
        "rain": torch.from_numpy(np.ascontiguousarray(record.rain[hours])),
        "labels": torch.from_numpy(np.ascontiguousarray(labels)), "depth": torch.from_numpy(depth),
        "timestamps": torch.from_numpy(epoch), "window_starts": torch.from_numpy(local.astype(np.int64)),
    }


def _common_payload(G: nx.DiGraph, arrays: GraphArrays, settings: DatasetSettings, scaler: FeatureScaler,
                    raw: tuple[np.ndarray, np.ndarray], record: _Record, inputs: BuildInputs,
                    cfg: Mapping[str, Any]) -> dict:
    static_raw, edge_raw = raw
    return {
        "format_version": FORMAT_VERSION,
        "node_ids": list(arrays.node_ids),
        "lon": torch.from_numpy(np.array(arrays.lon, dtype=np.float64)),
        "lat": torch.from_numpy(np.array(arrays.lat, dtype=np.float64)),
        "edge_index": torch.from_numpy(np.array(arrays.edge_index, dtype=np.int64).reshape(2, -1)),
        "edge_attr": torch.from_numpy(scaler.transform_edges(edge_raw[:, 0], edge_raw[:, 1])),
        "edge_attr_raw": torch.from_numpy(np.array(edge_raw, dtype=np.float32).reshape(-1, 2)),
        "static_raw": torch.from_numpy(np.array(static_raw, dtype=np.float32)),
        "static_feature_names": list(settings.static_features), "feature_names": list(settings.feature_names),
        "timezone": settings.timezone, "seq_len": settings.seq_len, "warmup_steps": settings.warmup_steps,
        "lookback_hours": settings.lookback_hours, "rolling_windows_h": list(settings.rolling_windows_h),
        "scaler": scaler.to_dict(), "label_source": record.label_source,
        "flood_threshold_m": _cfg_float(cfg, "hydrology", "flood_depth_threshold_m", 0.15),
        "graph_signature": arrays.signature(),
        "graph_attributes_sha256": graph_attributes_digest(arrays, settings),
        "weather_fingerprint": inputs.weather_fingerprint, "reports_fingerprint": inputs.reports_fingerprint,
        "graph_provenance": {k: str(G.graph.get(k, "unknown")) for k in GRAPH_PROVENANCE_KEYS},
        "config_hash": inputs.config_hash, "seed": settings.seed,
        "splits": list(settings.splits), "val_years": list(settings.val_years),
        "test_years": list(settings.test_years),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_id": uuid.uuid4().hex,  # shared by the train/val/test payloads of one build
    }


def _require_windows(selection: WindowSelection, ts: pd.DatetimeIndex, settings: DatasetSettings) -> None:
    span = f"{ts[0]:%Y-%m-%d} -> {ts[-1]:%Y-%m-%d}"
    filters = (f"season_months={list(settings.season_months)}, wet_window_min_mm={settings.wet_window_min_mm}, "
               f"dry_window_keep_frac={settings.dry_window_keep_frac}")
    held_out = f"dataset.val_years={list(settings.val_years)} / dataset.test_years={list(settings.test_years)}"
    if selection.train.size == 0:
        raise DatasetError(f"No training windows: every selected window of the weather record ({span}) falls in "
                           f"{held_out} or was filtered out ({filters}). Include non-held-out years via "
                           "weather.start_date/end_date or change dataset.val_years / test_years / filters")
    if selection.val.size == 0:
        raise DatasetError(f"No validation windows: dataset.val_years={list(settings.val_years)} has no selected "
                           f"window in the weather record ({span}; {filters}). Set dataset.val_years to years inside "
                           "the record, extend weather.start_date/end_date, or relax the filters")
    if settings.has_test_split and selection.test.size == 0:
        raise DatasetError(f"No test windows: dataset.test_years={list(settings.test_years)} has no selected window "
                           f"in the weather record ({span}; {filters}). Set dataset.test_years to years inside the "
                           "record, extend weather.start_date/end_date, or set dataset.test_years: [] (no test split)")


# --------------------------------------------------------------------------- build orchestration


def _summary(payloads: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path], build_time_s: float,
             reused: bool, reasons: list[str]) -> dict[str, Any]:
    train = payloads["train"]
    splits = {split: {**p["stats"], "path": str(paths[split]), "size_bytes": int(paths[split].stat().st_size)}
              for split, p in payloads.items()}
    return {
        "format_version": FORMAT_VERSION, "config_hash": train["config_hash"], "reused": reused,
        "rebuild_reasons": list(reasons),
        "build_time_s": round(float(build_time_s), 2), "created_utc": train.get("created_utc"),
        "build_id": train.get("build_id"),
        "n_nodes": len(train["node_ids"]), "n_edges": int(train["edge_index"].shape[1]),
        "feature_names": list(train["feature_names"]), "label_source": train["label_source"],
        "flood_threshold_m": train["flood_threshold_m"], "seq_len": train["seq_len"],
        "warmup_steps": train["warmup_steps"], "lookback_hours": train["lookback_hours"],
        "val_years": list(train.get("val_years", [])), "test_years": list(train.get("test_years", [])),
        "graph": dict(train.get("graph_provenance", {})),
        "fingerprints": {"graph_signature": dict(train["graph_signature"]).get("sha256"),
                         **{key: train.get(key) for key in ("graph_attributes_sha256", "weather_fingerprint",
                                                            "reports_fingerprint")}},
        "splits": splits,
    }


def _manifest_extra(train: Mapping[str, Any]) -> dict[str, Any]:
    """Build identity recorded in the dataset manifest next to the per-file checksums."""
    return {"format_version": FORMAT_VERSION, "config_hash": train["config_hash"], "seed": train["seed"],
            "fingerprints": {"graph_signature": dict(train["graph_signature"]).get("sha256"),
                             **{key: train.get(key) for key in ("graph_attributes_sha256", "weather_fingerprint",
                                                                "reports_fingerprint")}},
            "graph_provenance": dict(train.get("graph_provenance", {}))}


def _portable_path(path: Any) -> str:
    """``path`` relative to the project root when inside it, else its file name (F3-07: a shared
    report must not leak the user name or the directory layout of the build machine)."""
    candidate = Path(str(path))
    try:
        return candidate.resolve().relative_to(project_root().resolve()).as_posix()
    except (ValueError, OSError):
        return candidate.name or str(path)


def _write_summary(cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    """Write ``reports_dir/dataset_summary.json`` with portable split paths (the in-memory summary
    the CLI prints keeps the full local paths)."""
    path = resolve_path(cfg, "reports_dir") / "dataset_summary.json"
    splits = {name: {**stats, "path": _portable_path(stats["path"])} if stats.get("path") else dict(stats)
              for name, stats in (summary.get("splits") or {}).items()}
    try:
        atomic_write_text(path, json.dumps({**summary, "splits": splits}, indent=2, default=str))
    except OSError as exc:
        LOGGER.warning("Could not write %s (%s)", path, exc)


def _remove_stale_test(settings: DatasetSettings, path: Path) -> None:
    """With ``dataset.test_years: []`` no test split exists, so an old test file must not be evaluated."""
    if settings.has_test_split or not path.exists():
        return
    try:
        path.unlink()
        LOGGER.warning("dataset.test_years is empty: removed the stale test dataset %s from an earlier build", path)
    except OSError as exc:
        LOGGER.warning("dataset.test_years is empty but the stale test dataset %s could not be removed (%s); "
                       "consumers must ignore it", path, exc)


def _log_rebuild(reasons: list[str]) -> None:
    """INFO for a first or forced build, WARNING (naming every changed input) for a stale one."""
    if reasons and reasons[0].startswith((FORCED, "no existing")):
        LOGGER.info("Building the datasets (%s)", "; ".join(reasons))
    else:
        LOGGER.warning("Existing datasets are stale - rebuilding because: %s", "; ".join(reasons))


def _build(cfg: Mapping[str, Any], settings: DatasetSettings, weather_record: WeatherRecord,
           inputs: BuildInputs, graph: tuple[nx.DiGraph, GraphArrays]) -> dict[str, dict]:
    """Build the payloads of every configured split (not yet saved)."""
    simulate = _load_simulator()
    G, arrays = graph
    record = _build_record(cfg, arrays, settings, weather_record, simulate)
    selection = select_windows(record.timestamps, record.areal, settings)
    _require_windows(selection, record.timestamps, settings)
    static_raw = arrays.node_matrix(settings.static_features)
    edge_raw = arrays.edge_matrix(EDGE_FEATURES) if arrays.num_edges else np.zeros((0, 2), dtype=np.float32)
    samples = _dynamic_samples(record.rain, selection.train, settings)  # TRAIN hours only
    scaler = FeatureScaler.fit(static_raw, samples, edge_raw[:, 0],
                               _cfg_float(cfg, "elevation", "max_abs_grade", DEFAULT_MAX_ABS_GRADE),
                               static_names=settings.static_features,
                               dynamic_names=dynamic_feature_names(settings.rolling_windows_h))
    del samples
    common = _common_payload(G, arrays, settings, scaler, (static_raw, edge_raw), record, inputs, cfg)
    return {split: _split_payload(split, selection.starts(split), record, settings, common, selection)
            for split in settings.splits}


def build_datasets(cfg: Mapping[str, Any], force: bool = False) -> dict[str, Any]:
    """Build (or reuse) the train / val (/ test) payloads of :func:`dataset_paths`; returns the summary.

    Raises :class:`ConfigError` for invalid settings and :class:`DatasetError` when the
    simulator is missing, the record is too short or a configured split would be empty.
    """
    clock = time.perf_counter()
    settings = DatasetSettings.from_config(cfg)
    label_source = _label_source(cfg)
    paths = dataset_paths(cfg)
    weather_record = load_weather_record(cfg, settings)
    graph = _prepare_graph(cfg, settings)  # before the reuse check: re-enrichment changes the attribute digest
    inputs = current_inputs(cfg, settings, weather_record, label_source, arrays=graph[1])
    _remove_stale_test(settings, paths["test"])
    existing, reasons = (None, [FORCED]) if force else find_reusable(paths, settings.splits, inputs)
    if existing is not None:
        LOGGER.info("Datasets are up to date (config %s, graph %s, weather %s); reusing %s", inputs.config_hash,
                    inputs.graph_attributes_sha256, inputs.weather_fingerprint,
                    ", ".join(str(paths[s]) for s in settings.splits))
        summary = _summary(existing, paths, time.perf_counter() - clock, reused=True, reasons=[])
        _write_summary(cfg, summary)
        return summary
    _log_rebuild(reasons)
    payloads = _build(cfg, settings, weather_record, inputs, graph)
    commit_splits(payloads, {split: paths[split] for split in payloads}, extra=_manifest_extra(payloads["train"]))
    for split, payload in payloads.items():
        LOGGER.info("Saved %s dataset %s (%d windows)", split, paths[split], payload["stats"]["n_windows"])
    if payloads["train"]["stats"]["n_windows_with_flood"] == 0:
        LOGGER.warning("The training split contains no flooded node-steps; the model cannot learn floods - check "
                       "the hydrology calibration / labels.source")
    summary = _summary(payloads, paths, time.perf_counter() - clock, reused=False, reasons=reasons)
    _write_summary(cfg, summary)
    return summary
