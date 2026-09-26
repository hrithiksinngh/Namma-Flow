"""Hourly areal rainfall for the pilot corridor (contract schema 2.2) — public API.

Sources, in order of preference:

* **Open-Meteo archive** (reanalysis, no key) fetched in ``weather.chunk_days`` chunks at the
  bbox centre and cached as ``paths.weather_file`` (CSV) plus a ``.meta.json`` sidecar that
  records the location, model, bias factor and stored blocks. The cache never loses rows:
  disjoint ranges are kept as separate gap-free blocks (never bridged with zeros), and long
  zero-filled ``is_imputed`` runs are treated as missing, not as data.
* **Synthetic Bengaluru climatology** (``source="synthetic"``, reanalysis-like areal scale by
  default) when offline or when the API fails — deterministic from the seed, month by month,
  so any sub-range is reproducible. A record mixing both sources is reported at ERROR level.

Also here: the live **forecast** client, the **Previous Runs** client used for forecast
backtests, **design storms** for what-if scenarios and small helpers used downstream.
Schema primitives live in :mod:`src.data_pipeline.weather_schema`, generators in
:mod:`src.data_pipeline.weather_synthetic`; everything public is re-exported here.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data_pipeline.weather_schema import (
    DEFAULT_TZ,
    DEFAULTS,
    PRECIP,
    SOURCE_MISSING,
    SOURCES,
    STORM_SHAPES,
    WeatherFormatError,
    WeatherUnavailable,
    _clean_values,
    _normalize,
    _parse_range,
    _parse_times,
    _resolve_location,
    _settings,
    _tz,
    _validate_location,
    _bound,
    clean_weather,
    contiguous_blocks,
    default_location,
    imputed_runs,
    normalize_blocks,
    read_weather_csv,
    read_weather_rows,
    save_weather_csv,
)
from src.data_pipeline.weather_synthetic import design_storm, generate_synthetic_weather
from src.utils.config import ConfigError, is_offline, resolve_path
from src.utils.http import NetworkUnavailable, get_json
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

__all__ = [
    "DEFAULTS",
    "DEFAULT_TZ",
    "PRECIP",
    "SOURCE_MISSING",
    "SOURCES",
    "STORM_SHAPES",
    "WeatherFormatError",
    "WeatherSummary",
    "WeatherUnavailable",
    "areal_series",
    "clean_weather",
    "default_location",
    "design_storm",
    "fetch_archive",
    "fetch_forecast",
    "fetch_previous_runs",
    "generate_synthetic_weather",
    "load_or_fetch_weather",
    "read_weather_csv",
    "read_weather_rows",
    "save_weather_csv",
    "summarize_weather",
]

# --------------------------------------------------------------------------- Open-Meteo clients


def _payload_frame(payload: Any, variables: Sequence[str], tz: str, url: str) -> pd.DataFrame:
    """Parse an Open-Meteo ``hourly`` payload (local naive times) into a local-time frame."""
    hourly = payload.get("hourly") if isinstance(payload, Mapping) else None
    if not isinstance(hourly, Mapping) or not isinstance(hourly.get("time"), list):
        reason = payload.get("reason") if isinstance(payload, Mapping) else None
        raise NetworkUnavailable(f"{url} returned no hourly data ({reason or str(payload)[:200]})")
    times = hourly["time"]
    data: dict[str, np.ndarray] = {}
    for variable in variables:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(times):
            size = len(values) if isinstance(values, list) else "none"
            raise NetworkUnavailable(f"{url}: hourly '{variable}' has length {size} but {len(times)} timestamps")
        try:
            data[variable] = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            data[variable] = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy(float)
    frame = pd.DataFrame(data, index=_parse_times(pd.Index(times, dtype=object), tz))
    return frame[~frame.index.isna()].sort_index()


def _date_chunks(lo: pd.Timestamp, hi: pd.Timestamp, chunk_days: int) -> Iterator[tuple[dt.date, dt.date]]:
    first, last = lo.date(), hi.date()
    while first <= last:
        chunk_end = min(first + dt.timedelta(days=chunk_days - 1), last)
        yield first, chunk_end
        first = chunk_end + dt.timedelta(days=1)


def _http_kwargs(cfg: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timeout_s": settings["timeout_s"],
        "max_retries": settings["max_retries"],
        "backoff_s": settings["backoff_s"],
        "offline": is_offline(cfg),
    }


def _iter_archive(
    cfg: Mapping[str, Any], settings: Mapping[str, Any], lo: pd.Timestamp, hi: pd.Timestamp, lat: float, lon: float
) -> Iterator[pd.DataFrame]:
    """Yield raw archive frames chunk by chunk (raises NetworkUnavailable on failure)."""
    tz, url = _tz(cfg), settings["archive_url"]
    for first, last in _date_chunks(lo, hi, settings["chunk_days"]):
        params = {
            "latitude": round(lat, 5), "longitude": round(lon, 5),
            "start_date": first.isoformat(), "end_date": last.isoformat(),
            "hourly": "precipitation", "timezone": tz,
        }
        if settings["models"]:
            params["models"] = settings["models"]
        payload = get_json(url, params, **_http_kwargs(cfg, settings))
        frame = _payload_frame(payload, ("precipitation",), tz, url).rename(columns={"precipitation": PRECIP})
        frame["source"] = "open_meteo"
        LOGGER.info("Fetched Open-Meteo archive %s -> %s (%d hours)", first, last, len(frame))
        yield frame.loc[lo:hi]


def fetch_archive(cfg: Mapping[str, Any], start: Any, end: Any, lat: float, lon: float) -> pd.DataFrame:
    """Fetch raw hourly archive precipitation for ``[start, end]`` (chunked by ``chunk_days``).

    Returns an uncleaned local-time frame (``precipitation_mm`` may contain NaN, ``source``).
    Raises ``NetworkUnavailable`` offline, on HTTP/transport failure or on a malformed payload.
    """
    settings, tz = _settings(cfg), _tz(cfg)
    lo, hi = _parse_range(start, end, tz)
    lat, lon = _validate_location(lat, lon)
    frames = list(_iter_archive(cfg, settings, lo, hi, lat, lon))
    raw = pd.concat(frames) if frames else pd.DataFrame({PRECIP: [], "source": []})
    return raw[~raw.index.duplicated(keep="last")].sort_index()


def _now(now: Any, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz=tz) if now is None else pd.Timestamp(now)
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def fetch_forecast(
    cfg: Mapping[str, Any], lat: float | None = None, lon: float | None = None, now: Any = None
) -> pd.DataFrame:
    """Live forecast: ``forecast_past_days`` of recent model rain + ``forecast_days`` ahead.

    Schema 2.2 with ``source="forecast"`` and ``is_forecast = timestamp > now`` (floored to the
    hour). The same ``bias_correction_factor`` as the archive is applied so the model sees the
    distribution it was trained on. Raises :class:`WeatherUnavailable` on any failure.
    """
    settings, tz = _settings(cfg), _tz(cfg)
    lat, lon = _resolve_location(cfg, lat, lon)
    now_ts = _now(now, tz)
    if is_offline(cfg):
        raise WeatherUnavailable("Offline mode: the live Open-Meteo forecast is unavailable")
    url = settings["forecast_url"]
    params = {
        "latitude": round(lat, 5), "longitude": round(lon, 5), "hourly": "precipitation",
        "past_days": settings["forecast_past_days"], "forecast_days": settings["forecast_days"], "timezone": tz,
    }
    try:
        raw = _payload_frame(get_json(url, params, **_http_kwargs(cfg, settings)), ("precipitation",), tz, url)
    except NetworkUnavailable as exc:
        raise WeatherUnavailable(f"Open-Meteo forecast unavailable: {exc}") from exc
    if raw.empty or raw["precipitation"].isna().all():
        raise WeatherUnavailable(f"Open-Meteo forecast returned no precipitation values ({len(raw)} rows)")
    frame = _normalize(
        raw, tz=tz, max_precip=settings["max_precip_mm_h"], max_gap=settings["max_fill_gap_hours"],
        bias=settings["bias_correction_factor"], default_source="forecast",
    )
    frame["source"] = "forecast"
    frame["is_forecast"] = frame.index > now_ts.floor("h")
    return frame


def _validate_leads(lead_days: Any) -> list[int]:
    try:
        leads = [int(d) for d in lead_days if float(d) == int(d)]
        valid = bool(leads) and len(leads) == len(list(lead_days))
    except (TypeError, ValueError):
        valid = False
    if not valid or any(d < 1 or d > 7 for d in leads):
        raise ValueError(f"lead_days must be a non-empty sequence of integers in 1..7, got {lead_days!r}")
    return sorted(set(leads))


def fetch_previous_runs(
    cfg: Mapping[str, Any],
    start: Any,
    end: Any,
    lead_days: Sequence[int] = (1, 2),
    lat: float | None = None,
    lon: float | None = None,
) -> pd.DataFrame:
    """Archived forecasts from the Previous Runs API for forecast-skill backtests.

    Columns: ``precip_lead_0`` (the API's ``precipitation``, the best estimate of what fell),
    ``precip_lead_{24*d}h`` for each ``d`` in ``lead_days`` (issued ``d`` days earlier) and
    ``is_imputed``; gap-free hourly local index over ``[start, end]``. Missing values are set to
    0 and flagged (WARNING); if any column misses more than ``previous_runs_max_missing_frac``
    of the hours (the API only holds data from ~2024) or the request fails, raises
    :class:`WeatherUnavailable`.
    """
    settings, tz = _settings(cfg), _tz(cfg)
    leads = _validate_leads(lead_days)
    lat, lon = _resolve_location(cfg, lat, lon)
    lo, hi = _parse_range(start, end, tz)
    if is_offline(cfg):
        raise WeatherUnavailable("Offline mode: the Open-Meteo Previous Runs API is unavailable")
    names = {"precipitation": "precip_lead_0"}
    names.update({f"precipitation_previous_day{d}": f"precip_lead_{24 * d}h" for d in leads})
    url, frames = settings["previous_runs_url"], []
    try:
        for first, last in _date_chunks(lo, hi, settings["chunk_days"]):
            params = {
                "latitude": round(lat, 5), "longitude": round(lon, 5), "hourly": ",".join(names),
                "start_date": first.isoformat(), "end_date": last.isoformat(), "timezone": tz,
            }
            frames.append(_payload_frame(get_json(url, params, **_http_kwargs(cfg, settings)), list(names), tz, url))
    except NetworkUnavailable as exc:
        raise WeatherUnavailable(f"Open-Meteo Previous Runs API unavailable: {exc}") from exc
    raw = pd.concat(frames).rename(columns=names)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return _previous_runs_frame(raw, list(names.values()), lo, hi, tz, settings)


def _previous_runs_frame(
    raw: pd.DataFrame, columns: list[str], lo: pd.Timestamp, hi: pd.Timestamp, tz: str, settings: Mapping[str, Any]
) -> pd.DataFrame:
    full = pd.date_range(lo, hi, freq="h", tz=tz, name="timestamp")
    data: dict[str, np.ndarray] = {}
    imputed = np.zeros(len(full), dtype=bool)
    for column in columns:
        values = _clean_values(raw[column], settings["max_precip_mm_h"], settings["bias_correction_factor"], column)
        values = values.reindex(full)
        missing = values.isna().to_numpy()
        if missing.mean() > settings["previous_runs_max_missing_frac"]:
            raise WeatherUnavailable(
                f"{column}: {missing.mean():.0%} of hours missing for {lo:%Y-%m-%d}..{hi:%Y-%m-%d}; "
                "the Open-Meteo Previous Runs archive only holds data from ~2024 onwards"
            )
        if missing.any():
            LOGGER.warning("%s: %d of %d hours missing; set to 0 mm (is_imputed)", column, int(missing.sum()), len(full))
        data[column] = values.fillna(0.0).to_numpy(dtype=np.float64)
        imputed |= missing
    frame = pd.DataFrame(data, index=full)
    frame["is_imputed"] = imputed
    return frame


# --------------------------------------------------------------------------- cache orchestration


def _record_range(settings: Mapping[str, Any], tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        lo = _bound(settings["start_date"], tz, end=False)
        hi = _bound(settings["end_date"], tz, end=True)
    except ValueError as exc:
        raise ConfigError(f"weather.start_date / weather.end_date are invalid: {exc}") from exc
    available = pd.Timestamp.now(tz=tz).normalize() - pd.Timedelta(days=settings["archive_delay_days"])
    available = available + pd.Timedelta(hours=23)
    if hi > available:
        LOGGER.warning("weather.end_date %s is not yet available in the archive; using %s", hi, available)
        hi = available
    if lo > hi:
        raise ConfigError(f"weather.start_date {lo} is after the (effective) end date {hi}")
    return lo, hi


def _cache_identity(lat: float, lon: float, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "models": settings["models"],
        "bias_correction_factor": float(settings["bias_correction_factor"]),
    }


@dataclass(frozen=True)
class _CacheState:
    """Result of reading the CSV cache: the stored rows (or None), whether they must be rewritten
    and whether this run may overwrite them (not when they belong to another location, offline)."""

    frame: pd.DataFrame | None
    dirty: bool = False
    writable: bool = True


def _read_cache(path: Path, tz: str, identity: Mapping[str, Any], online: bool, max_precip: float) -> _CacheState:
    """Load the rows stored in the CSV cache (per gap-free block) and reconcile its ``.meta.json``."""
    if not path.exists():
        return _CacheState(None)
    try:
        cache = read_weather_rows(path, tz)
    except (WeatherFormatError, OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable weather cache %s: %s", path, exc)
        return _CacheState(None)
    meta_path = path.with_suffix(".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable cache metadata %s: %s", meta_path, exc)
        meta = None
    if not isinstance(meta, Mapping):
        return _CacheState(cache)
    same_place = (
        abs(float(meta.get("latitude", identity["latitude"])) - identity["latitude"]) < 1e-5
        and abs(float(meta.get("longitude", identity["longitude"])) - identity["longitude"]) < 1e-5
        and meta.get("models") == identity["models"]
    )
    if not same_place:
        if online:
            LOGGER.warning("Weather cache %s was built for another location/model; refetching", path)
            return _CacheState(None)
        LOGGER.warning(
            "Weather cache %s was built for another location/model; using it read-only (offline)", path
        )
    old = float(meta.get("bias_correction_factor", 1.0))
    new = identity["bias_correction_factor"]
    if math.isclose(old, new):
        return _CacheState(cache, writable=same_place)
    LOGGER.warning("Weather cache bias_correction_factor %.3f != configured %.3f; rescaling cached rows", old, new)
    observed = cache["source"] == "open_meteo"
    rescaled = cache[PRECIP].where(~observed, (cache[PRECIP] * new / old).clip(upper=max_precip))
    return _CacheState(cache.assign(**{PRECIP: rescaled}), dirty=True, writable=same_place)


def _write_cache(frame: pd.DataFrame, path: Path, identity: Mapping[str, Any]) -> None:
    blocks = contiguous_blocks(pd.DatetimeIndex(frame.index))
    meta = dict(identity)
    meta.update(
        start=frame.index[0].isoformat(),
        end=frame.index[-1].isoformat(),
        rows=int(len(frame)),
        blocks=[[frame.index[a].isoformat(), frame.index[b].isoformat()] for a, b in blocks],
        sources={str(k): int(v) for k, v in frame["source"].value_counts().items()},
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    try:
        save_weather_csv(frame, path)
        atomic_write_text(path.with_suffix(".meta.json"), json.dumps(meta, indent=2))
        LOGGER.info("Wrote weather cache %s (%d rows in %d block(s))", path, len(frame), len(blocks))
    except OSError as exc:
        LOGGER.warning("Could not write weather cache %s: %s (continuing in memory)", path, exc)


def _hour_blocks(hours: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(hours[a], hours[b]) for a, b in contiguous_blocks(hours)]


def _fetch_hours(
    cfg: Mapping[str, Any], settings: Mapping[str, Any], missing: pd.DatetimeIndex, lat: float, lon: float
) -> pd.DataFrame | None:
    """Fetch and clean the missing hours; stops at the first failure (no retry storms)."""
    frames = []
    try:
        for lo, hi in _hour_blocks(missing):
            for raw in _iter_archive(cfg, settings, lo, hi, lat, lon):
                if not raw.empty:
                    frames.append(clean_weather(raw, cfg))
    except NetworkUnavailable as exc:
        LOGGER.warning("Open-Meteo archive unavailable (%s); falling back to cache / synthetic data", exc)
    if not frames:
        return None
    fetched = pd.concat(frames)
    fetched = fetched[~fetched.index.duplicated(keep="last")]
    return fetched[fetched.index.isin(missing)]


def _synthetic_hours(cfg: Mapping[str, Any], hours: pd.DatetimeIndex, span: str) -> pd.DataFrame:
    LOGGER.warning(
        "%d hours of %s are unavailable from Open-Meteo and the cache; filled with synthetic "
        "Bengaluru climatology (source=synthetic)", len(hours), span,
    )
    frames = [generate_synthetic_weather(lo, hi, cfg) for lo, hi in _hour_blocks(hours)]
    return pd.concat(frames)


@dataclass(frozen=True)
class _CacheView:
    """The cached rows split by trust: ``usable`` excludes long zero-filled (imputed) runs."""

    rows: pd.DataFrame | None
    usable: pd.DataFrame | None
    trusted: pd.DataFrame | None


def _cache_view(cache: pd.DataFrame | None, settings: Mapping[str, Any], online: bool, force: bool) -> _CacheView:
    """Rows the orchestration may reuse. Long ``is_imputed`` runs (legacy zero bridges between
    blocks, long API gaps) are never data; online only ``open_meteo`` rows are trusted as-is."""
    if cache is None:
        return _CacheView(None, None, None)
    bridge = imputed_runs(cache, settings["max_fill_gap_hours"])
    if bridge.any():
        LOGGER.warning("Weather cache: %d zero-filled hours in long imputed runs are treated as missing", int(bridge.sum()))
    usable = cache[~bridge]
    if force:
        return _CacheView(cache, usable, None)
    return _CacheView(cache, usable, usable[usable["source"] == "open_meteo"] if online else usable)


def _fill_record(
    cfg: Mapping[str, Any],
    settings: Mapping[str, Any],
    target: pd.DatetimeIndex,
    view: _CacheView,
    missing: pd.DatetimeIndex,
    location: tuple[float, float],
) -> pd.DataFrame:
    """Fill ``missing`` hours (API -> usable cached row -> synthetic) and merge with the cache.

    Every cached row outside the target is kept, so the CSV never loses data; disjoint ranges
    stay separate blocks (no zero-filled bridge between them, see :func:`normalize_blocks`).
    """
    lo, hi = target[0], target[-1]
    pieces = [] if view.trusted is None else [view.trusted[view.trusted.index.isin(target)]]
    fetched = None if is_offline(cfg) else _fetch_hours(cfg, settings, missing, *location)
    remaining = missing if fetched is None else missing.difference(fetched.index)
    pieces.append(fetched)
    if view.usable is not None and not remaining.empty:
        from_cache = view.usable[view.usable.index.isin(remaining)]
        pieces.append(from_cache)
        remaining = remaining.difference(from_cache.index)
    if not remaining.empty:
        pieces.append(_synthetic_hours(cfg, remaining, f"{lo:%Y-%m-%d} -> {hi:%Y-%m-%d}"))
    if view.rows is not None:
        pieces.append(view.rows[~view.rows.index.isin(target)])
    merged = pd.concat([p for p in pieces if p is not None and not p.empty])
    return normalize_blocks(
        merged, tz=_tz(cfg), max_precip=settings["max_precip_mm_h"], bias=1.0, default_source="open_meteo"
    )


def _real_observations(frame: pd.DataFrame, max_gap: int) -> int:
    return int(((frame["source"] == "open_meteo").to_numpy() & ~imputed_runs(frame, max_gap)).sum())


def _shrink_problem(cache: pd.DataFrame | None, record: pd.DataFrame, max_gap: int) -> str | None:
    """Why writing ``record`` over ``cache`` would lose data (None when it would not)."""
    if cache is None:
        return None
    lost = cache.index.difference(record.index)
    if len(lost):
        return f"{len(lost)} cached hour(s) ({lost[0]:%Y-%m-%d} ...) would be dropped"
    before, after = _real_observations(cache, max_gap), _real_observations(record, max_gap)
    if after < before:
        return f"the new record holds {after} Open-Meteo hours, fewer than the cached {before}"
    return None


def _warn_if_mixed(record: pd.DataFrame) -> None:
    """Loudly report a record that mixes reanalysis and synthetic rain (two input distributions)."""
    counts = record["source"].value_counts()
    if counts.get("open_meteo", 0) == 0 or counts.get("synthetic", 0) == 0:
        return
    synthetic = pd.DatetimeIndex(record.index[(record["source"] == "synthetic").to_numpy()])
    spans = ", ".join(f"{a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M}" for a, b in _hour_blocks(synthetic)[:3])
    LOGGER.error(
        "Weather record %s -> %s mixes %d Open-Meteo hours with %d synthetic hours (%s%s). The hydrology "
        "labels and the model are calibrated on reanalysis rain: re-run python src/data_pipeline/"
        "03_weather_ingestion.py online before building datasets or training",
        record.index[0], record.index[-1], int(counts["open_meteo"]), int(counts["synthetic"]), spans,
        " ..." if len(_hour_blocks(synthetic)) > 3 else "",
    )


def _target_slice(frame: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, tz: str) -> pd.DataFrame:
    """Gap-free copy of ``[lo, hi]`` with ``freq='h'`` (raises if the record does not cover it)."""
    expected = pd.date_range(lo, hi, freq="h", tz=tz, name="timestamp")
    out = frame.loc[lo:hi].copy()
    if not out.index.equals(expected):
        raise WeatherFormatError(f"internal error: weather record does not cover {lo} -> {hi} hour by hour")
    out.index = expected
    return out


def load_or_fetch_weather(cfg: Mapping[str, Any], force: bool = False, *, persist: bool = True) -> pd.DataFrame:
    """Return the hourly areal record for ``[weather.start_date, weather.end_date]``.

    Chain: CSV cache (if it covers the range) -> Open-Meteo archive for the missing hours
    only -> usable cached rows of any source -> synthetic climatology (WARNING; an ERROR is
    logged when the result mixes Open-Meteo and synthetic hours). Online, cached ``synthetic``
    rows count as missing and are replaced by real data; ``force`` refetches everything
    (keeping the cache if the API fails). The merged record - every previously cached row plus
    the new range, stored as separate blocks when they are disjoint - is written atomically to
    ``paths.weather_file`` with a ``.meta.json`` sidecar; a write that would drop cached hours
    or Open-Meteo observations is refused. The end date is clamped to what the archive can
    have (today - ``archive_delay_days``). ``persist=False`` (diagnostics such as calibrate)
    never writes the cache: the record is returned in memory only.
    """
    settings, tz = _settings(cfg), _tz(cfg)
    lo, hi = _record_range(settings, tz)
    path = resolve_path(cfg, "weather_file")
    lat, lon = default_location(cfg)
    identity = _cache_identity(lat, lon, settings)
    online = not is_offline(cfg)
    if force and not online:
        LOGGER.warning("force=True ignored offline: cannot refetch, reusing the cache")
        force = False
    state = _read_cache(path, tz, identity, online, settings["max_precip_mm_h"])
    view = _cache_view(state.frame, settings, online, force)
    target = pd.date_range(lo, hi, freq="h", tz=tz, name="timestamp")
    missing = target if view.trusted is None else target.difference(view.trusted.index)
    if missing.empty:
        LOGGER.info("Weather cache %s covers %s -> %s", path, lo, hi)
        if state.dirty and state.writable and persist:
            _write_cache(state.frame, path, identity)
        return _target_slice(state.frame, lo, hi, tz)
    record = _fill_record(cfg, settings, target, view, missing, (lat, lon))
    result = _target_slice(record, lo, hi, tz)
    _warn_if_mixed(result)
    problem = _shrink_problem(state.frame, record, settings["max_fill_gap_hours"])
    if not persist:
        LOGGER.info("Not writing the weather cache %s (persist=False): the record stays in memory", path)
    elif not state.writable:
        LOGGER.warning("Not overwriting %s: it belongs to another location/model", path)
    elif problem:
        LOGGER.error("Not overwriting weather cache %s: %s (returning the record in memory only)", path, problem)
    else:
        _write_cache(record, path, identity)
    return result


# --------------------------------------------------------------------------- helpers for downstream stages


def areal_series(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return ``(precipitation float64 copy, DatetimeIndex)``; NaN/negative -> 0 with a WARNING."""
    if not isinstance(df, pd.DataFrame) or PRECIP not in df.columns:
        raise ValueError(f"areal_series needs a DataFrame with a '{PRECIP}' column")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("areal_series needs a DatetimeIndex (schema 2.2)")
    values = pd.to_numeric(df[PRECIP], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    bad = ~np.isfinite(values) | (np.nan_to_num(values, nan=0.0) < 0)
    if bad.any():
        LOGGER.warning("areal series: %d missing/non-finite/negative values set to 0 mm", int(bad.sum()))
        values[bad] = 0.0
    return values, pd.DatetimeIndex(df.index)


@dataclass(frozen=True)
class WeatherSummary:
    """Headline statistics of a weather record (printed by the stage-03 CLI)."""

    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    total_mm: float
    annual_mm: dict[int, float]
    full_year_mean_mm: float | None
    wettest_hour: tuple[pd.Timestamp, float] | None
    wettest_days: tuple[tuple[str, float], ...]
    pct_imputed: float
    sources: dict[str, int]

    def format_lines(self) -> list[str]:
        if self.rows == 0:
            return ["rows            : 0 (empty record)"]
        mean = f", mean {self.full_year_mean_mm:.1f} mm/yr over full years" if self.full_year_mean_mm else ""
        hour_ts, hour_mm = self.wettest_hour
        return [
            f"rows            : {self.rows} hourly ({self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M %z})",
            f"total rainfall  : {self.total_mm:.1f} mm{mean}",
            "annual totals   : " + " | ".join(f"{y}: {v:.1f}" for y, v in self.annual_mm.items()),
            f"wettest hour    : {hour_ts:%Y-%m-%d %H:%M} ({hour_mm:.1f} mm)",
            "wettest days    : " + ", ".join(f"{d} ({v:.1f} mm)" for d, v in self.wettest_days),
            f"imputed hours   : {self.pct_imputed:.2f} %",
            "sources         : " + ", ".join(f"{k}={v}" for k, v in self.sources.items()),
        ]


def summarize_weather(df: pd.DataFrame, top_days: int = 5) -> WeatherSummary:
    """Totals per year, wettest hour/days, imputed share and source counts of a schema-2.2 frame."""
    if len(df) == 0:
        return WeatherSummary(0, None, None, 0.0, {}, None, None, (), 0.0, {})
    rain = pd.to_numeric(df[PRECIP], errors="coerce").fillna(0.0)
    years = rain.groupby(rain.index.year)
    annual = {int(y): float(v) for y, v in years.sum().items()}
    full_years = [annual[int(y)] for y, n in years.size().items() if n >= 360 * 24]
    daily = rain.groupby(rain.index.normalize()).sum().nlargest(top_days)
    peak = rain.idxmax()
    return WeatherSummary(
        rows=int(len(df)),
        start=df.index[0],
        end=df.index[-1],
        total_mm=float(rain.sum()),
        annual_mm=annual,
        full_year_mean_mm=float(np.mean(full_years)) if full_years else None,
        wettest_hour=(peak, float(rain.loc[peak])),
        wettest_days=tuple((f"{d:%Y-%m-%d}", round(float(v), 2)) for d, v in daily.items()),
        pct_imputed=float(100.0 * df["is_imputed"].mean()) if "is_imputed" in df.columns else 0.0,
        sources={str(k): int(v) for k, v in df["source"].value_counts().items()} if "source" in df.columns else {},
    )
