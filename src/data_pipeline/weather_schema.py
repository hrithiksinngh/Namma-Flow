"""Weather frame schema (contract section 2.2): settings, timestamp parsing, cleaning, CSV I/O.

Shared by :mod:`src.data_pipeline.weather` (API clients and cache orchestration) and
:mod:`src.data_pipeline.weather_synthetic` (climatology generator and design storms). Import
the public API from :mod:`src.data_pipeline.weather`; this module holds the primitives.

Frame contract: index = tz-aware, strictly increasing, gap-free hourly ``DatetimeIndex``
named ``timestamp`` in ``project.timezone``; columns ``precipitation_mm`` (mm in the hour
ending at the timestamp, >= 0, clipped to ``weather.max_precip_mm_h``), ``is_imputed`` and
``source``; forecast / design-storm frames add ``is_forecast``. Hours that are absent from the
input and only exist because the frame is reindexed to a gap-free range (e.g. the months
between two stored cache blocks) get 0 mm, ``is_imputed=True`` and ``source="missing"``
(:data:`SOURCE_MISSING`) - never the source of the neighbouring rows - so no consumer can
mistake them for observations.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.config import ConfigError, get_section
from src.utils.geo import bbox_center
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger("src.data_pipeline.weather")

PRECIP = "precipitation_mm"
SOURCES = ("open_meteo", "synthetic", "forecast", "design_storm")
SOURCE_MISSING = "missing"  # hours absent from the input, zero-filled by reindexing (never an observation)
STORM_SHAPES = ("chicago", "uniform", "triangular")
DEFAULT_TZ = "Asia/Kolkata"

DEFAULTS: dict[str, Any] = {
    "provider": "open_meteo",
    "archive_url": "https://archive-api.open-meteo.com/v1/archive",
    "forecast_url": "https://api.open-meteo.com/v1/forecast",
    "previous_runs_url": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "chunk_days": 366,
    "timeout_s": 60,
    "max_retries": 3,
    "backoff_s": 2.0,
    "max_fill_gap_hours": 6,
    "max_precip_mm_h": 150.0,
    "bias_correction_factor": 1.0,
    "forecast_past_days": 2,
    "forecast_days": 3,
    "archive_delay_days": 5,          # the reanalysis archive lags real time by ~5 days
    "latitude": None,                 # override the bbox-centre location
    "longitude": None,
    "models": None,                   # Open-Meteo archive model; None = provider "best_match"
    "previous_runs_max_missing_frac": 0.5,
    "synthetic_scale": "areal",       # areal (reanalysis-like, what the pipeline is calibrated on) | point (gauge-like)
}
SYNTHETIC_SCALES = ("areal", "point")

# key: (minimum, minimum is exclusive, must be an integer)
_NUMERIC_RULES: dict[str, tuple[float, bool, bool]] = {
    "chunk_days": (1, False, True),
    "timeout_s": (0, True, False),
    "max_retries": (1, False, True),
    "backoff_s": (0, False, False),
    "max_fill_gap_hours": (0, False, True),
    "max_precip_mm_h": (0, True, False),
    "bias_correction_factor": (0, True, False),
    "forecast_past_days": (0, False, True),
    "forecast_days": (1, False, True),
    "archive_delay_days": (0, False, True),
    "previous_runs_max_missing_frac": (0, False, False),
}
_DATE_ONLY = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")


class WeatherUnavailable(RuntimeError):
    """Raised when live weather (forecast / previous runs) cannot be obtained."""


class WeatherFormatError(ValueError):
    """Raised when a weather CSV exists but cannot be parsed into schema 2.2."""


# --------------------------------------------------------------------------- settings & parsing


def _settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated ``weather`` section merged over :data:`DEFAULTS`."""
    settings = get_section(cfg, "weather", DEFAULTS)
    for key, (minimum, exclusive, integer) in _NUMERIC_RULES.items():
        value = settings[key]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"weather.{key} must be a number, got {value!r}") from exc
        too_small = number <= minimum if exclusive else number < minimum
        if isinstance(value, bool) or not math.isfinite(number) or too_small or (integer and number != int(number)):
            kind = "an integer" if integer else "a number"
            raise ConfigError(f"weather.{key} must be {kind} {'>' if exclusive else '>='} {minimum}, got {value!r}")
        settings[key] = int(number) if integer else number
    if settings["previous_runs_max_missing_frac"] > 1:
        raise ConfigError("weather.previous_runs_max_missing_frac must lie in [0, 1]")
    scale = str(settings["synthetic_scale"]).strip().lower()
    if scale not in SYNTHETIC_SCALES:
        raise ConfigError(f"weather.synthetic_scale must be one of {SYNTHETIC_SCALES}, got {settings['synthetic_scale']!r}")
    settings["synthetic_scale"] = scale
    return settings


def _tz(cfg: Mapping[str, Any]) -> str:
    return str((cfg.get("project") or {}).get("timezone") or DEFAULT_TZ)


def _check_tz(tz: str) -> str:
    try:
        pd.Timestamp("2000-01-01").tz_localize(tz)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Unknown timezone {tz!r}") from exc
    return tz


def _validate_location(lat: Any, lon: Any) -> tuple[float, float]:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"latitude/longitude must be numbers, got {lat!r}, {lon!r}") from exc
    if not (math.isfinite(lat_f) and -90.0 <= lat_f <= 90.0):
        raise ValueError(f"latitude must lie in [-90, 90], got {lat!r}")
    if not (math.isfinite(lon_f) and -180.0 <= lon_f <= 180.0):
        raise ValueError(f"longitude must lie in [-180, 180], got {lon!r}")
    return lat_f, lon_f


def default_location(cfg: Mapping[str, Any]) -> tuple[float, float]:
    """``(lat, lon)`` used for areal weather: ``weather.latitude/longitude`` or the bbox centre."""
    settings = _settings(cfg)
    if settings["latitude"] is not None and settings["longitude"] is not None:
        return _validate_location(settings["latitude"], settings["longitude"])
    bbox = (cfg.get("region") or {}).get("bbox")
    if bbox is None:
        raise ConfigError("region.bbox (or weather.latitude/longitude) is required to locate the weather point")
    lon, lat = bbox_center(bbox)
    return _validate_location(lat, lon)


def _resolve_location(cfg: Mapping[str, Any], lat: Any, lon: Any) -> tuple[float, float]:
    if lat is None or lon is None:
        return default_location(cfg)
    return _validate_location(lat, lon)


def _bound(value: Any, tz: str, *, end: bool) -> pd.Timestamp:
    """Parse a date/time bound to a local, hour-floored timestamp.

    Date-only values (``"2024-12-31"`` or ``datetime.date``) cover the whole day, so an end
    bound becomes 23:00. tz-naive values are local time; tz-aware values are converted.
    """
    date_only = (isinstance(value, str) and bool(_DATE_ONLY.match(value))) or (
        isinstance(value, dt.date) and not isinstance(value, dt.datetime)
    )
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date/time {value!r}") from exc
    if ts is pd.NaT or pd.isna(ts):
        raise ValueError(f"Invalid date/time {value!r}")
    ts = ts.tz_localize(_check_tz(tz)) if ts.tzinfo is None else ts.tz_convert(tz)
    if date_only and end:
        ts = ts + pd.Timedelta(hours=23)
    return ts.floor("h")


def _parse_range(start: Any, end: Any, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo, hi = _bound(start, tz, end=False), _bound(end, tz, end=True)
    if lo > hi:
        raise ValueError(f"start {lo} must not be after end {hi}")
    return lo, hi


def _parse_times(values: Any, tz: str) -> pd.DatetimeIndex:
    """Parse timestamps (ISO strings, mixed offsets, datetimes) into a local DatetimeIndex."""
    if isinstance(values, pd.DatetimeIndex):
        index = values
    else:
        try:
            index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", format="ISO8601"))
        except (ValueError, TypeError):  # mixed UTC offsets
            index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", format="ISO8601", utc=True))
        if len(index) and index.isna().all():
            index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", format="mixed"))
    if index.tz is None:
        return index.tz_localize(_check_tz(tz), ambiguous="NaT", nonexistent="NaT")
    return index.tz_convert(tz)


# --------------------------------------------------------------------------- cleaning (schema 2.2)


def _as_frame(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        frame = df.to_frame(name=df.name if df.name in (PRECIP, "precipitation") else PRECIP)
    elif isinstance(df, pd.DataFrame):
        frame = df.copy()
    else:
        raise TypeError(f"weather data must be a pandas DataFrame or Series, got {type(df).__name__}")
    for column in ("timestamp", "time"):
        if column in frame.columns:
            frame = frame.set_index(column)
            break
    if PRECIP not in frame.columns:
        if "precipitation" not in frame.columns:
            raise ValueError(f"weather frame needs a '{PRECIP}' column; got {list(frame.columns)}")
        frame = frame.rename(columns={"precipitation": PRECIP})
    if not isinstance(frame.index, pd.DatetimeIndex) and frame.index.inferred_type not in ("string", "datetime", "mixed"):
        raise ValueError("weather frame needs a DatetimeIndex or a 'timestamp' column")
    return frame


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    truthy = {"true", "1", "1.0", "yes", "y", "t"}
    return series.map(lambda v: str(v).strip().lower() in truthy).astype(bool)


def _localize_and_order(frame: pd.DataFrame, tz: str) -> pd.DataFrame:
    frame = frame.set_axis(_parse_times(frame.index, tz), axis=0)
    invalid = frame.index.isna()
    if invalid.any():
        LOGGER.warning("Dropped %d rows with unparseable or non-existent timestamps", int(invalid.sum()))
        frame = frame[~invalid]
    duplicated = frame.index.duplicated(keep="last")
    if duplicated.any():
        LOGGER.warning("Dropped %d duplicate timestamps (kept the last occurrence)", int(duplicated.sum()))
        frame = frame[~duplicated]
    if not frame.index.is_monotonic_increasing:
        LOGGER.info("Weather rows were not sorted; sorting by timestamp")
        frame = frame.sort_index(kind="stable")
    off_hour = frame.index != frame.index.floor("h")
    if not off_hour.any():
        return frame
    LOGGER.warning("Aggregated %d sub-hourly timestamps into hour-ending totals", int(off_hour.sum()))
    keys = frame.index.ceil("h")
    grouped: dict[str, pd.Series] = {
        PRECIP: pd.to_numeric(frame[PRECIP], errors="coerce").groupby(keys).sum(min_count=1)
    }
    for column in ("is_imputed", "is_forecast"):
        if column in frame.columns:
            grouped[column] = _as_bool(frame[column]).groupby(keys).any()
    if "source" in frame.columns:
        grouped["source"] = frame["source"].groupby(keys).last()
    return pd.DataFrame(grouped)


def _clean_values(raw: pd.Series, max_precip: float, bias: float, label: str = "precipitation") -> pd.Series:
    """Coerce to float; non-numeric/inf -> NaN, negative -> 0, apply bias, clip (all logged)."""
    values = pd.to_numeric(raw, errors="coerce").astype(float)
    non_numeric = values.isna() & raw.notna()
    non_finite = np.isinf(values)
    if non_numeric.any() or non_finite.any():
        LOGGER.warning(
            "%d non-numeric or non-finite %s values treated as missing",
            int(non_numeric.sum() + non_finite.sum()), label,
        )
        values = values.mask(non_finite)
    negative = values < 0
    if negative.any():
        LOGGER.warning("%d negative %s values set to 0 mm", int(negative.sum()), label)
        values = values.mask(negative, 0.0)
    values = values * bias
    over = values > max_precip
    if over.any():
        LOGGER.warning(
            "%d %s values above max_precip_mm_h=%.1f clipped (max seen %.1f mm/h)",
            int(over.sum()), label, max_precip, float(values.max()),
        )
        values = values.clip(upper=max_precip)
    return values


def _report_gaps(index: pd.DatetimeIndex, missing: np.ndarray, max_gap: int) -> None:
    if not missing.any():
        return
    edges = np.diff(np.concatenate([[0], missing.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    lengths = np.flatnonzero(edges == -1) - starts
    LOGGER.info("Filled %d missing hours in %d gaps with 0 mm (is_imputed=True)", int(missing.sum()), starts.size)
    long = np.flatnonzero(lengths > max_gap)
    if long.size:
        largest = long[np.argsort(-lengths[long])[:3]]
        spans = ", ".join(
            f"{index[starts[i]]:%Y-%m-%d %H:%M} -> {index[starts[i] + lengths[i] - 1]:%Y-%m-%d %H:%M} ({lengths[i]} h)"
            for i in largest
        )
        LOGGER.warning(
            "%d gap(s) longer than max_fill_gap_hours=%d filled with 0 mm (flagged is_imputed); largest: %s",
            long.size, max_gap, spans,
        )


def _schema_frame(
    index: pd.DatetimeIndex, precip: Any, imputed: Any, source: Any, is_forecast: Any = None
) -> pd.DataFrame:
    n = len(index)
    frame = pd.DataFrame(
        {
            PRECIP: np.asarray(precip, dtype=np.float64).reshape(n),
            "is_imputed": np.broadcast_to(np.asarray(imputed, dtype=bool), (n,)).copy(),
            "source": np.broadcast_to(np.asarray(source, dtype=object), (n,)).astype(str),
        },
        index=pd.DatetimeIndex(index, name="timestamp", freq="h"),
    )
    frame["source"] = frame["source"].astype(str)
    if is_forecast is not None:
        frame["is_forecast"] = np.broadcast_to(np.asarray(is_forecast, dtype=bool), (n,)).copy()
    return frame


def _reindexed_sources(frame: pd.DataFrame, full: pd.DatetimeIndex, default_source: str) -> np.ndarray:
    """``source`` per hour of ``full``: stored rows keep theirs (unlabelled stored rows get the most
    common stored source, or ``default_source``); hours absent from ``frame`` get :data:`SOURCE_MISSING`."""
    absent = ~full.isin(frame.index)
    source = np.full(len(full), default_source, dtype=object)
    if "source" in frame.columns:
        known = frame["source"].reindex(full)
        stored = known[~absent]
        fill = stored.mode().iloc[0] if stored.notna().any() else default_source
        source = known.astype(object).where(known.notna(), fill).astype(str).to_numpy(dtype=object)
    source[absent] = SOURCE_MISSING
    return source


def _normalize(
    df: Any,
    *,
    tz: str,
    max_precip: float,
    max_gap: int,
    bias: float,
    default_source: str,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """Shared implementation of :func:`clean_weather` with explicit parameters."""
    frame = _localize_and_order(_as_frame(df), tz)
    precip = _clean_values(frame[PRECIP], max_precip, bias)
    if len(frame.index) == 0 and (start is None or end is None):
        raise ValueError("Weather frame has no valid timestamps and no start/end range was given")
    lo = _bound(start, tz, end=False) if start is not None else frame.index.min()
    hi = _bound(end, tz, end=True) if end is not None else frame.index.max()
    if lo > hi:
        raise ValueError(f"start {lo} must not be after end {hi}")
    full = pd.date_range(lo, hi, freq="h", tz=tz, name="timestamp")
    outside = int(((frame.index < lo) | (frame.index > hi)).sum())
    if outside:
        LOGGER.debug("Dropped %d weather rows outside %s -> %s", outside, lo, hi)
    precip = precip.reindex(full)
    missing = precip.isna().to_numpy()
    _report_gaps(full, missing, max_gap)
    imputed = missing.copy()
    if "is_imputed" in frame.columns:
        imputed |= _as_bool(frame["is_imputed"]).reindex(full, fill_value=False).to_numpy(dtype=bool)
    source = _reindexed_sources(frame, full, default_source)
    forecast = None
    if "is_forecast" in frame.columns:
        forecast = _as_bool(frame["is_forecast"]).reindex(full, fill_value=False).to_numpy(dtype=bool)
    return _schema_frame(full, precip.fillna(0.0).to_numpy(), imputed, source, forecast)


def clean_weather(df: Any, cfg: Mapping[str, Any], start: Any = None, end: Any = None) -> pd.DataFrame:
    """Enforce schema 2.2 on raw hourly rainfall (a DataFrame or Series).

    Accepts a DatetimeIndex (or a ``timestamp``/``time`` column) and a ``precipitation_mm`` or
    ``precipitation`` column. Non-numeric/inf -> missing; negative -> 0; ``bias_correction_factor``
    applied; values above ``max_precip_mm_h`` clipped; duplicates keep the last row; unsorted
    rows sorted; sub-hourly rows summed into hour-ending totals; tz-naive stamps localised to
    ``project.timezone`` and other zones converted; the index is reindexed to the full hourly
    range (``[start, end]`` when given — date-only bounds cover whole days) and every missing
    hour is set to 0 mm with ``is_imputed=True`` (gaps longer than ``max_fill_gap_hours`` are
    logged as WARNING). Never mutates ``df``.
    """
    settings = _settings(cfg)
    return _normalize(
        df,
        tz=_tz(cfg),
        max_precip=settings["max_precip_mm_h"],
        max_gap=settings["max_fill_gap_hours"],
        bias=settings["bias_correction_factor"],
        default_source="open_meteo",
        start=start,
        end=end,
    )


# --------------------------------------------------------------------------- CSV persistence


def save_weather_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a schema-2.2 frame atomically with ISO-8601 timestamps carrying their UTC offset."""
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("save_weather_csv expects a DataFrame indexed by a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("save_weather_csv needs a tz-aware index (schema 2.2)")
    if PRECIP not in df.columns:
        raise ValueError(f"save_weather_csv needs a '{PRECIP}' column")
    stamps = pd.Series(df.index.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out = pd.DataFrame({"timestamp": stamps.str.slice(0, -2) + ":" + stamps.str.slice(-2)})
    out[PRECIP] = df[PRECIP].to_numpy(dtype=np.float64)
    out["is_imputed"] = df["is_imputed"].to_numpy(dtype=bool) if "is_imputed" in df.columns else False
    out["source"] = df["source"].astype(str).to_numpy() if "source" in df.columns else "open_meteo"
    if "is_forecast" in df.columns:
        out["is_forecast"] = df["is_forecast"].to_numpy(dtype=bool)
    return atomic_write_text(Path(path), out.to_csv(index=False))


def _parse_csv(path: str | Path, tz: str) -> pd.DataFrame:
    """Parse a weather CSV into a local-time frame (rows as stored; no reindexing)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weather CSV not found: {path} (run: python src/data_pipeline/03_weather_ingestion.py)"
        )
    try:
        raw = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise WeatherFormatError(f"Could not parse weather CSV {path}: {exc}") from exc
    missing = [c for c in ("timestamp", PRECIP) if c not in raw.columns]
    if missing:
        raise WeatherFormatError(f"Weather CSV {path} lacks columns {missing}; found {list(raw.columns)}")
    if raw.empty:
        raise WeatherFormatError(f"Weather CSV {path} has no rows")
    index = _parse_times(pd.Index(raw["timestamp"].astype(str)), tz)
    if index.isna().all():
        raise WeatherFormatError(f"Weather CSV {path} has no parseable timestamps")
    return raw.drop(columns="timestamp").set_axis(index, axis=0)


def read_weather_csv(
    path: str | Path, tz: str, *, max_precip: float | None = None, max_gap: int | None = None
) -> pd.DataFrame:
    """Read a weather CSV back into a gap-free schema-2.2 frame (no bias is re-applied).

    Cached rows were cleaned when they were written, so values are NOT clipped again unless
    ``max_precip`` is given (a cache built with a higher ``weather.max_precip_mm_h`` reads
    back unchanged). Hours absent from the file (e.g. between two stored blocks) are filled
    with 0 mm, flagged ``is_imputed`` and labelled ``source="missing"`` (:data:`SOURCE_MISSING`),
    never with the stored rows' source; ``max_gap`` (default ``max_fill_gap_hours``) only sets
    the threshold above which such gaps are logged as a WARNING. Replays that must not bridge
    gaps at all should use :func:`read_weather_rows`.
    """
    return _normalize(
        _parse_csv(path, tz),
        tz=tz,
        max_precip=math.inf if max_precip is None else float(max_precip),
        max_gap=int(DEFAULTS["max_fill_gap_hours"]) if max_gap is None else int(max_gap),
        bias=1.0,
        default_source="open_meteo",
    )


def contiguous_blocks(index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """``[(first, last)]`` positions of the gap-free hourly runs of a sorted DatetimeIndex."""
    if len(index) == 0:
        return []
    seconds = index.as_unit("s").asi8
    breaks = np.flatnonzero(np.diff(seconds) != 3600)
    return list(zip(np.r_[0, breaks + 1].tolist(), np.r_[breaks, len(index) - 1].tolist()))


def normalize_blocks(df: Any, *, tz: str, max_precip: float, bias: float, default_source: str) -> pd.DataFrame:
    """Schema-2.2 cleaning of each gap-free run of hours separately (never bridges a gap).

    Unlike :func:`clean_weather`, hours missing between two runs are NOT reindexed and
    zero-filled: a cache holding 2018-2024 plus one later month stays two blocks, so a later
    request covering the gap fetches / generates it instead of reading months of fake zeros.
    The result is sorted and unique but has no ``freq`` when there are several blocks.
    """
    frame = _localize_and_order(_as_frame(df), tz)
    if len(frame.index) == 0:
        raise ValueError("Weather frame has no valid timestamps")
    blocks = [
        _normalize(frame.iloc[first : last + 1], tz=tz, max_precip=max_precip, max_gap=0, bias=bias,
                   default_source=default_source)
        for first, last in contiguous_blocks(pd.DatetimeIndex(frame.index))
    ]
    return blocks[0] if len(blocks) == 1 else pd.concat(blocks)


def read_weather_rows(path: str | Path, tz: str) -> pd.DataFrame:
    """The rows stored in a weather CSV, cleaned per gap-free block (no bridging, no re-clipping).

    Used by the cache orchestration: hours absent from the file stay absent (they are
    missing, not 0 mm), unlike :func:`read_weather_csv`, which returns a gap-free frame.
    """
    return normalize_blocks(_parse_csv(path, tz), tz=tz, max_precip=math.inf, bias=1.0, default_source="open_meteo")


def imputed_runs(frame: pd.DataFrame, min_hours: int) -> np.ndarray:
    """Boolean mask of rows inside runs of consecutive ``is_imputed`` hours longer than ``min_hours``.

    Such rows are gaps filled with 0 mm (e.g. a bridge between two blocks written by an older
    version), not observations; the cache orchestration never treats them as data.
    """
    if "is_imputed" not in frame.columns or len(frame) == 0:
        return np.zeros(len(frame), dtype=bool)
    flags = _as_bool(frame["is_imputed"]).to_numpy(dtype=bool)
    seconds = pd.DatetimeIndex(frame.index).as_unit("s").asi8
    new_run = np.r_[True, (~flags[:-1]) | (~flags[1:]) | (np.diff(seconds) != 3600)]
    run_id = np.cumsum(new_run) - 1
    lengths = np.bincount(run_id, weights=flags.astype(float))
    return flags & (lengths[run_id] > min_hours)
