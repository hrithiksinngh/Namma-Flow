"""Observed flood reports (crowd-sourced / BBMP / news) and their fusion with simulated labels.

Reports live in ``paths.flood_reports_file`` (CSV, one row per report)::

    timestamp,lat,lon,radius_m,severity,description
    2022-09-05 09:00,12.9270,77.6780,250,severe,"<what was observed and the source>"

* ``timestamp`` — ISO-8601; values without an offset are local time in ``project.timezone``,
  values with an offset (``+00:00``, ``Z``) are converted. A date-only value means 00:00 local.
* ``lat`` / ``lon`` — WGS84 degrees (required).
* ``radius_m`` — optional extent of the flooded area; blank = ``labels.report_snap_radius_m``.
* ``severity`` — optional ``1|2|3`` or ``minor|moderate|severe`` (synonyms: low/medium/high,
  ankle/knee/waist, major); blank = 1.
* ``description`` — optional free text.

The repository ships a header-only template: no reports are bundled, because fabricated
"observations" would silently contaminate the labels. :func:`merge_observed_labels` snaps each
report to every junction within ``max(radius_m, labels.report_snap_radius_m)`` metres and marks
those junctions flooded for ``+-labels.report_window_hours`` around the report time.
``labels.source`` selects the fusion: ``simulated`` (reports ignored), ``observed`` (labels are
the reports alone) or ``hybrid`` (union of simulated and observed floods).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays
from src.utils.config import ConfigError, get_section
from src.utils.geo import lonlat_to_local_xy
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "source": "simulated",
    "report_snap_radius_m": 150.0,
    "report_window_hours": 3,
    "min_report_severity": 1,
}

LABEL_SOURCES = ("simulated", "observed", "hybrid")
REPORT_COLUMNS = ("timestamp", "lat", "lon", "radius_m", "severity", "description")
REQUIRED_COLUMNS = ("timestamp", "lat", "lon")
DEFAULT_SEVERITY = 1
SEVERITY_NAMES = {1: "minor", 2: "moderate", 3: "severe"}
_SEVERITY_WORDS = {
    "minor": 1, "low": 1, "ankle": 1, "light": 1,
    "moderate": 2, "medium": 2, "knee": 2,
    "severe": 3, "high": 3, "major": 3, "waist": 3, "extreme": 3,
}
_COLUMN_ALIASES = {
    "time": "timestamp", "datetime": "timestamp", "date_time": "timestamp", "reported_at": "timestamp",
    "latitude": "lat", "longitude": "lon", "lng": "lon", "long": "lon",
    "radius": "radius_m", "extent_m": "radius_m", "notes": "description", "comment": "description",
}
_MAX_EXAMPLES = 5


class FloodReportError(ValueError):
    """Raised when a flood-report file cannot be read or lacks required columns."""


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class LabelSettings:
    """Validated ``labels`` configuration."""

    source: str
    report_snap_radius_m: float
    report_window_hours: float
    min_report_severity: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LabelSettings":
        section = get_section(cfg, "labels", DEFAULTS)
        source = str(section["source"]).strip().lower()
        if source not in LABEL_SOURCES:
            raise ConfigError(f"labels.source must be one of {LABEL_SOURCES}, got {section['source']!r}")
        radius = _config_number("report_snap_radius_m", section["report_snap_radius_m"])
        window = _config_number("report_window_hours", section["report_window_hours"])
        severity = _config_number("min_report_severity", section["min_report_severity"])
        if radius <= 0:
            raise ConfigError(f"labels.report_snap_radius_m must be > 0, got {radius}")
        if window < 0:
            raise ConfigError(f"labels.report_window_hours must be >= 0, got {window}")
        if severity not in SEVERITY_NAMES:
            raise ConfigError(f"labels.min_report_severity must be one of {sorted(SEVERITY_NAMES)}, got {severity}")
        return cls(source, radius, window, int(severity))


def _config_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigError(f"labels.{name} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ConfigError(f"labels.{name} must be finite, got {value!r}")
    return float(value)


# --------------------------------------------------------------------------- loading


def empty_reports(tz: str) -> pd.DataFrame:
    """A report frame with the canonical columns / dtypes and no rows."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype=f"datetime64[ns, {tz}]"),
            "lat": pd.Series([], dtype=np.float64),
            "lon": pd.Series([], dtype=np.float64),
            "radius_m": pd.Series([], dtype=np.float64),
            "severity": pd.Series([], dtype=np.int64),
            "description": pd.Series([], dtype=object),
        }
    )


def _check_tz(tz: str) -> str:
    try:
        pd.Timestamp("2000-01-01").tz_localize(tz)
    except Exception as exc:  # pytz / zoneinfo raise different types
        raise ValueError(f"Unknown timezone {tz!r}: {exc}") from exc
    return tz


def _read_raw(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        raise FloodReportError(f"Flood report path {path} is not a regular file")
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True, encoding="utf-8")
    except pd.errors.EmptyDataError:
        LOGGER.warning("Flood report file %s is empty (not even a header); using no reports", path)
        return None
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise FloodReportError(f"Could not parse flood report CSV {path}: {exc}") from exc
    raw.columns = [_COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in raw.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise FloodReportError(
            f"Flood report CSV {path} lacks required columns {missing}; expected {list(REPORT_COLUMNS)}"
        )
    return raw


def _parse_timestamp(value: str, tz: str) -> pd.Timestamp | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        stamp = pd.Timestamp(text)
        if stamp is pd.NaT:
            return None
        return stamp.tz_localize(tz) if stamp.tzinfo is None else stamp.tz_convert(tz)
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_severity(value: str) -> int | None:
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_SEVERITY
    if text in _SEVERITY_WORDS:
        return _SEVERITY_WORDS[text]
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() and int(number) in SEVERITY_NAMES else None


def _parse_optional_radius(value: str) -> float | None:
    """Radius in metres: blank -> NaN (use the default), invalid -> ``None`` (drop the row)."""
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        radius = float(text)
    except ValueError:
        return None
    return radius if math.isfinite(radius) and radius >= 0 else None


def _column(raw: pd.DataFrame, name: str) -> pd.Series:
    return raw[name] if name in raw.columns else pd.Series([""] * len(raw), index=raw.index, dtype=object)


def _parse_rows(raw: pd.DataFrame, tz: str) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Parse every column; return the parsed frame and the CSV line numbers of each problem."""
    stamps = [_parse_timestamp(v, tz) for v in raw["timestamp"]]
    lat = pd.to_numeric(raw["lat"].str.strip(), errors="coerce")
    lon = pd.to_numeric(raw["lon"].str.strip(), errors="coerce")
    radius = [_parse_optional_radius(v) for v in _column(raw, "radius_m")]
    severity = [_parse_severity(v) for v in _column(raw, "severity")]
    lines = np.arange(len(raw)) + 2  # header is line 1
    problems = {
        "unparseable timestamp": lines[[s is None for s in stamps]].tolist(),
        "latitude missing or outside [-90, 90]": lines[~(lat.abs() <= 90).to_numpy()].tolist(),
        "longitude missing or outside [-180, 180]": lines[~(lon.abs() <= 180).to_numpy()].tolist(),
        "negative or non-numeric radius_m": lines[[r is None for r in radius]].tolist(),
        "unknown severity (use 1-3 or minor/moderate/severe)": lines[[s is None for s in severity]].tolist(),
    }
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "lat": lat.to_numpy(dtype=np.float64),
            "lon": lon.to_numpy(dtype=np.float64),
            "radius_m": [np.nan if r is None else r for r in radius],
            "severity": [DEFAULT_SEVERITY if s is None else s for s in severity],
            "description": _column(raw, "description").astype(str).str.strip().to_numpy(),
        },
        index=raw.index,
    )
    return frame, problems


def _log_problems(path: Path, problems: Mapping[str, list[int]], n_dropped: int) -> None:
    if not n_dropped:
        return
    details = "; ".join(
        f"{reason}: lines {lines[:_MAX_EXAMPLES]}{' ...' if len(lines) > _MAX_EXAMPLES else ''}"
        for reason, lines in problems.items()
        if lines
    )
    LOGGER.warning("Flood reports %s: dropped %d invalid rows (%s)", path, n_dropped, details)


def load_flood_reports(path: str | Path, tz: str) -> pd.DataFrame:
    """Read and validate a flood-report CSV into a frame with :data:`REPORT_COLUMNS`.

    A missing file or a header-only template gives an empty frame; invalid rows are dropped
    with one WARNING that lists the reasons and line numbers; exact duplicates (timestamp,
    lat, lon) are removed. Rows are sorted by time and ``timestamp`` is tz-aware in ``tz``.
    Raises :class:`FloodReportError` when the file is unreadable or lacks required columns.
    """
    tz = _check_tz(tz)
    path = Path(path)
    if not path.exists():
        LOGGER.info("No flood report file at %s; observed labels are empty", path)
        return empty_reports(tz)
    raw = _read_raw(path)
    if raw is None or raw.empty:
        return empty_reports(tz)
    frame, problems = _parse_rows(raw, tz)
    bad_lines = set().union(*(set(v) for v in problems.values()))
    keep = ~np.isin(np.arange(len(raw)) + 2, sorted(bad_lines))
    _log_problems(path, problems, int((~keep).sum()))
    frame = frame.loc[keep]
    before = len(frame)
    frame = frame.drop_duplicates(subset=["timestamp", "lat", "lon"], keep="first")
    if len(frame) < before:
        LOGGER.info("Flood reports %s: removed %d duplicate rows", path, before - len(frame))
    if frame.empty:
        return empty_reports(tz)
    frame = frame.assign(timestamp=pd.Series(pd.DatetimeIndex(frame["timestamp"]).tz_convert(tz), index=frame.index))
    frame = frame.astype({"severity": np.int64, "radius_m": np.float64})
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    LOGGER.info("Loaded %d flood reports from %s (%s -> %s)", len(frame), path,
                frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1])
    return frame[list(REPORT_COLUMNS)]


# --------------------------------------------------------------------------- merging


def _as_arrays(graph: Any) -> GraphArrays:
    if isinstance(graph, GraphArrays):
        return graph
    if isinstance(graph, nx.Graph):
        return graph_to_arrays(graph)
    raise TypeError(f"graph must be GraphArrays or a networkx graph, got {type(graph).__name__}")


def _validate_labels(labels: Any, n_nodes: int) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 2 or arr.shape[1] != n_nodes:
        raise ValueError(f"labels must have shape [T, {n_nodes}], got {arr.shape}")
    if arr.dtype.kind not in "biuf" or (arr.size and not ((arr == 0) | (arr == 1)).all()):
        raise ValueError("labels must contain only 0/1 values")
    return arr.astype(np.uint8)  # always a new array


def _validate_timestamps(timestamps: Any, n_hours: int, tz: str) -> np.ndarray:
    """UTC epoch nanoseconds of the label rows (tz-naive values are local time)."""
    index = pd.DatetimeIndex(timestamps)
    if len(index) != n_hours:
        raise ValueError(f"timestamps has {len(index)} entries but labels has {n_hours} rows")
    if index.hasnans:
        raise ValueError("timestamps must not contain NaT")
    index = index.tz_localize(tz) if index.tz is None else index
    ns = index.tz_convert("UTC").as_unit("ns").asi8
    if ns.size > 1 and not np.all(np.diff(ns) > 0):
        raise ValueError("timestamps must be strictly increasing")
    return ns


def _validate_reports(reports: Any, tz: str) -> pd.DataFrame:
    if reports is None:
        return empty_reports(tz)
    if not isinstance(reports, pd.DataFrame):
        raise TypeError(f"reports must be a DataFrame from load_flood_reports, got {type(reports).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in reports.columns]
    if missing:
        raise ValueError(f"reports lack required columns {missing}; use load_flood_reports()")
    return reports


def _report_times_ns(reports: pd.DataFrame, tz: str) -> np.ndarray:
    stamps = pd.DatetimeIndex(pd.to_datetime(reports["timestamp"]))
    stamps = stamps.tz_localize(tz) if stamps.tz is None else stamps
    return stamps.tz_convert("UTC").as_unit("ns").asi8


def _snap_reports(reports: pd.DataFrame, graph: GraphArrays, snap_radius_m: float) -> list[list[int]]:
    """Junction indices within ``max(radius_m, snap_radius_m)`` of each report (local metres)."""
    origin = (float(np.mean(graph.lon)), float(np.mean(graph.lat)))
    node_x, node_y = lonlat_to_local_xy(graph.lon, graph.lat, origin)
    rep_x, rep_y = lonlat_to_local_xy(reports["lon"].to_numpy(float), reports["lat"].to_numpy(float), origin)
    radius = reports["radius_m"].to_numpy(float) if "radius_m" in reports.columns else np.full(len(reports), np.nan)
    radius = np.fmax(np.nan_to_num(radius, nan=0.0), snap_radius_m)
    tree = cKDTree(np.column_stack([node_x, node_y]))
    points = np.column_stack([rep_x, rep_y])
    return [list(hits) for hits in tree.query_ball_point(points, r=radius)]


def merge_observed_labels(
    labels: np.ndarray,
    timestamps: pd.DatetimeIndex,
    graph: GraphArrays | nx.Graph,
    reports: pd.DataFrame | None,
    cfg: Mapping[str, Any],
    *,
    mode: str | None = None,
) -> np.ndarray:
    """Fuse observed reports into labels ``uint8 [T, N]`` according to ``labels.source`` (or ``mode``).

    Returns a new array; the inputs are never modified. Reports below
    ``labels.min_report_severity``, outside the label time range or farther than the snap
    radius from every junction are ignored (counted in a WARNING).
    """
    settings = LabelSettings.from_config(cfg)
    tz = str((cfg.get("project") or {}).get("timezone", "Asia/Kolkata"))
    chosen = (mode or settings.source).strip().lower()
    if chosen not in LABEL_SOURCES:
        raise ValueError(f"mode must be one of {LABEL_SOURCES}, got {mode!r}")
    arrays = _as_arrays(graph)
    base = _validate_labels(labels, arrays.num_nodes)
    hours_ns = _validate_timestamps(timestamps, base.shape[0], tz)
    reports = _validate_reports(reports, tz)
    if chosen == "simulated":
        LOGGER.info("labels.source=simulated: observed reports are not merged")
        return base
    out = np.zeros_like(base) if chosen == "observed" else base
    if "severity" in reports.columns:
        reports = reports[pd.to_numeric(reports["severity"], errors="coerce").fillna(DEFAULT_SEVERITY)
                          >= settings.min_report_severity]
    if reports.empty or base.shape[0] == 0:
        LOGGER.info("No usable flood reports to merge (%s mode)", chosen)
        return out
    _apply_reports(out, hours_ns, arrays, reports, settings, tz)
    return out


def _apply_reports(
    out: np.ndarray,
    hours_ns: np.ndarray,
    graph: GraphArrays,
    reports: pd.DataFrame,
    settings: LabelSettings,
    tz: str,
) -> None:
    """Set ``out[t, nodes] = 1`` for each report's window / snapped junctions (``out`` is ours)."""
    window_ns = int(round(settings.report_window_hours * 3600 * 1e9))
    times = _report_times_ns(reports, tz)
    snapped = _snap_reports(reports, graph, settings.report_snap_radius_m)
    unsnapped = out_of_range = used = 0
    before = int(out.sum())
    for when, nodes in zip(times.tolist(), snapped):
        if not nodes:
            unsnapped += 1
            continue
        lo = int(np.searchsorted(hours_ns, when - window_ns, side="left"))
        hi = int(np.searchsorted(hours_ns, when + window_ns, side="right"))
        if lo >= hi:
            out_of_range += 1
            continue
        out[lo:hi, nodes] = 1
        used += 1
    if unsnapped:
        LOGGER.warning("%d flood reports could not be snapped to a junction within %.0f m",
                       unsnapped, settings.report_snap_radius_m)
    if out_of_range:
        LOGGER.warning("%d flood reports fall outside the labelled time range", out_of_range)
    LOGGER.info("Merged %d flood reports: %d junction-hours newly flagged", used, int(out.sum()) - before)
