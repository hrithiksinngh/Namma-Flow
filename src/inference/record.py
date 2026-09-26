"""The cached hourly weather record as historical replays and the replay picker see it.

The record file (``paths.weather_file``) may hold several disjoint blocks of hours (e.g.
2018-2024 plus one later month). Replays read it with
:func:`~src.data_pipeline.weather_schema.read_weather_rows`, which keeps the stored rows only:
hours between two blocks stay *absent* instead of being bridged with 0 mm, so a replay never
runs the model on fabricated rain. Hours that are stored but are not observations — long runs
of ``is_imputed`` rows (a bridge written by an older version, or a fetch gap) and rows whose
``source`` is ``missing`` — are "gap-filled" (:func:`gap_filled_mask`): replays refuse to start
on them and note them when the window touches them, and the notable-event list marks them.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_pipeline import weather, weather_schema
from src.inference.settings import InferenceSettings, whole
from src.utils.config import ConfigError, get_section, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_TZ = "Asia/Kolkata"
NOTABLE_COLUMNS = ("rank", "start", "end", "peak_time", "total_mm", "peak_mm_h", "label", "source",
                   "gap_filled_hours")
MISSING_SOURCE = "missing"


class ScenarioError(ValueError):
    """Raised when a scenario cannot be built from the given inputs (bad range, missing record, ...)."""


def record_tz(cfg: Mapping[str, Any]) -> str:
    return str((cfg.get("project") or {}).get("timezone") or DEFAULT_TZ)


@functools.lru_cache(maxsize=4)
def _cached_record(path: str, mtime_ns: int, size: int, tz: str) -> pd.DataFrame:
    return weather_schema.read_weather_rows(path, tz)


def load_weather_record(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """The stored rows of ``paths.weather_file`` (schema-2.2 columns; a copy).

    The frame is sorted and unique but may have gaps between stored blocks (see
    :func:`record_blocks`); hours missing from the file are never filled in.
    """
    path = resolve_path(cfg, "weather_file")
    if not path.is_file():
        raise ScenarioError(f"Weather record {path} not found; build it with: "
                            "python src/data_pipeline/03_weather_ingestion.py")
    stat = path.stat()
    try:
        frame = _cached_record(str(path), stat.st_mtime_ns, stat.st_size, record_tz(cfg))
    except (ValueError, OSError) as exc:
        raise ScenarioError(f"Weather record {path} is unreadable ({exc}); rebuild it with: "
                            "python src/data_pipeline/03_weather_ingestion.py --force") from exc
    if len(frame) == 0:
        raise ScenarioError(f"Weather record {path} has no rows; rebuild it with: "
                            "python src/data_pipeline/03_weather_ingestion.py --force")
    return frame.copy()


def record_blocks(record: pd.DataFrame) -> list[tuple[int, int]]:
    """``[(first, last)]`` row positions of the gap-free hourly blocks of ``record``."""
    return weather_schema.contiguous_blocks(pd.DatetimeIndex(record.index))


def block_spans(record: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``[(first hour, last hour)]`` of every stored block (for pickers and captions)."""
    return [(record.index[a], record.index[b]) for a, b in record_blocks(record)]


def _max_fill_gap(cfg: Mapping[str, Any]) -> int:
    value = get_section(cfg, "weather", weather_schema.DEFAULTS)["max_fill_gap_hours"]
    return whole(value, "weather.max_fill_gap_hours", 0, ConfigError)


def gap_filled_mask(record: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    """Rows that are stored but are not observations (long ``is_imputed`` runs, ``source == missing``)."""
    mask = weather_schema.imputed_runs(record, _max_fill_gap(cfg))
    if "source" in record.columns:
        mask = mask | (record["source"].astype(str).str.lower() == MISSING_SOURCE).to_numpy(dtype=bool)
    return mask


def block_containing(record: pd.DataFrame, ts: pd.Timestamp) -> tuple[int, int]:
    """The ``(first, last)`` positions of the block holding hour ``ts``; ScenarioError when it is not stored."""
    spans = block_spans(record)
    first_ts, last_ts = spans[0][0], spans[-1][1]
    if not first_ts <= ts <= last_ts:
        raise ScenarioError(f"start {ts:%Y-%m-%d %H:%M} is outside the weather record "
                            f"{first_ts:%Y-%m-%d %H:%M} -> {last_ts:%Y-%m-%d %H:%M}")
    for (a, b), (lo, hi) in zip(record_blocks(record), spans):
        if lo <= ts <= hi:
            return a, b
    stored = "; ".join(f"{lo:%Y-%m-%d %H:%M} -> {hi:%Y-%m-%d %H:%M}" for lo, hi in spans)
    raise ScenarioError(f"start {ts:%Y-%m-%d %H:%M} falls in a gap of the weather record (those hours were never "
                        f"stored, so there is no rain to replay); stored blocks: {stored}")


def dominant_source(window: pd.DataFrame, gap_filled: np.ndarray) -> str:
    """The most frequent ``source`` among the window's observed (not gap-filled) rows."""
    if "source" not in window.columns or len(window) == 0:
        return "open_meteo"
    observed = window["source"].astype(str)[~gap_filled]
    counts = (observed if len(observed) else window["source"].astype(str)).value_counts()
    return str(counts.index[0]) if len(counts) else "open_meteo"


# --------------------------------------------------------------------------- notable events


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in NOTABLE_COLUMNS})


def _window_totals(areal: np.ndarray, blocks: list[tuple[int, int]], window: int) -> tuple[np.ndarray, np.ndarray]:
    """``(start positions, totals)`` of every ``window``-hour span that lies inside one stored block."""
    starts, totals = [], []
    for a, b in blocks:
        if b - a + 1 < window:
            continue
        csum = np.concatenate([[0.0], np.cumsum(areal[a: b + 1])])
        totals.append(csum[window:] - csum[:-window])
        starts.append(np.arange(a, b - window + 2))
    if not starts:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(starts), np.concatenate(totals)


def _pick_events(starts: np.ndarray, totals: np.ndarray, index: pd.DatetimeIndex, k: int,
                 separation_h: int) -> list[int]:
    """Wettest window starts, greedily, at least ``separation_h`` hours apart (wall-clock)."""
    chosen: list[int] = []
    hours = index.as_unit("s").asi8 // 3600
    for j in np.argsort(-totals, kind="stable"):
        if totals[j] <= 0 or len(chosen) == k:
            break
        i = int(starts[j])
        if all(abs(int(hours[i]) - int(hours[c])) >= separation_h for c in chosen):
            chosen.append(i)
    return chosen


def _event_row(rank: int, i: int, window: int, record: pd.DataFrame, areal: np.ndarray,
               gap_filled: np.ndarray) -> dict[str, Any]:
    index = record.index
    span = slice(i, i + window)
    peak = i + int(np.argmax(areal[span]))
    total = float(areal[span].sum())
    filled = int(gap_filled[span].sum())
    source = dominant_source(record.iloc[span], gap_filled[span])
    flags = (" · synthetic record" if source == "synthetic" else "") + (
        f" · includes {filled} gap-filled h" if filled else "")
    return {
        "rank": rank, "start": index[i], "end": index[i + window - 1], "peak_time": index[peak],
        "total_mm": round(total, 2), "peak_mm_h": round(float(areal[peak]), 2),
        "label": f"{index[i]:%Y-%m-%d %H:%M}: {total:.1f} mm in {window} h (peak {areal[peak]:.1f} mm/h){flags}",
        "source": source, "gap_filled_hours": filled,
    }


def list_notable_events(cfg: Mapping[str, Any], top_k: int = 10) -> pd.DataFrame:
    """The wettest ``inference.notable_event_window_h``-hour periods of the record (wettest first).

    Columns :data:`NOTABLE_COLUMNS`; events are at least ``notable_event_separation_h`` apart
    and never span two stored blocks. ``source`` is the window's dominant record source
    (labels say "synthetic record" for synthetic spans) and ``gap_filled_hours`` counts hours
    that are not observations (see :func:`gap_filled_mask`). ``start`` is a good
    ``historical_scenario`` start. A missing / unreadable record gives an empty frame (WARNING).
    """
    k = whole(top_k, "top_k", 1)
    settings = InferenceSettings.from_config(cfg)
    try:
        record = load_weather_record(cfg)
    except ScenarioError as exc:
        LOGGER.warning("No notable events: %s", exc)
        return _empty_events()
    areal, index = weather.areal_series(record)
    gap_filled = gap_filled_mask(record, cfg)
    window = settings.notable_event_window_h
    starts, totals = _window_totals(np.where(gap_filled, 0.0, areal), record_blocks(record), window)
    chosen = _pick_events(starts, totals, index, k, settings.notable_event_separation_h)
    rows = [_event_row(rank, i, window, record, areal, gap_filled) for rank, i in enumerate(chosen, start=1)]
    return pd.DataFrame(rows, columns=list(NOTABLE_COLUMNS)) if rows else _empty_events()

