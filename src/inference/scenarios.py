"""Prediction scenarios: the areal rainfall that drives a forecast, replay or what-if run.

A :class:`Scenario` is an hourly areal rain series (schema 2.2 values, mm in the hour ending
at each timestamp) split into *history* (hours before ``target_start``, used only to warm up
the model's rolling sums / GRU state and the physics simulator) and *target* hours (the
hours a predictor reports). Four builders cover the app's modes:

* :func:`forecast_scenario` — live Open-Meteo forecast (recent model rain + the next hours).
* :func:`historical_scenario` — replay of the cached weather record (stored hours only; a
  start in a gap between stored blocks, or on gap-filled hours, is refused). The history is
  extended back to the start of any rain event already in progress (also when the replay
  would start on a dry gap hour inside an event), so the junction rain field is exactly the
  one the dataset builder produced for those hours (no train/serve skew).
* :func:`design_storm_scenario` — dry history followed by a synthetic design storm.
* :func:`custom_scenario` — any user-supplied hourly series.

Junction rain field
-------------------
Predictors turn the areal series into junction rain with
:func:`~src.data_pipeline.rain_field.downscale_rainfall`, whose convective cells are seeded by
``rainfall_field.seed`` and the start hour of each rain event. Training labels come from ONE
such field, and the model is trained on that exact field:

* historical replays reproduce it (``Scenario.exact_field = True``): the prediction is the
  model's emulation of the teacher given the junction rain field;
* everywhere else only corridor-average rain is known (``exact_field = False``), so predictors
  average over ``inference.field_members`` independent field realisations (Monte-Carlo
  marginalisation, :mod:`src.inference.field_ensemble`) and report the spread;
* :attr:`Scenario.field_timestamps` is the time axis the field is drawn on: the real hours,
  except for design storms, which use a canonical axis on which every storm starts at
  ``inference.design_storm_field_anchor`` (the same members for every what-if storm and wall
  clock: common random numbers keep storms comparable).

:func:`list_notable_events` lists the wettest periods of the record for the replay picker
(:mod:`src.inference.record`); :class:`InferenceSettings` validates the ``inference`` config
section (:mod:`src.inference.settings`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_pipeline import weather
from src.data_pipeline.rain_field import RainFieldParams, find_rain_events
from src.inference.record import (  # noqa: F401 - re-exported (historical import path)
    NOTABLE_COLUMNS,
    ScenarioError,
    block_containing,
    block_spans,
    dominant_source,
    gap_filled_mask,
    list_notable_events,
    load_weather_record,
)
from src.inference.settings import (  # noqa: F401 - re-exported (historical import path)
    DEFAULTS,
    DesignStormPreset,
    InferenceSettings,
    real as _real,
    whole as _whole,
)
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

KINDS = ("forecast", "historical", "design_storm", "custom")
HOUR = pd.Timedelta(hours=1)
DEFAULT_TZ = "Asia/Kolkata"
PRECIP = weather.PRECIP
WeatherUnavailable = weather.WeatherUnavailable

RAIN_FIELD_NOTES = {
    "exact": "exact training field: junction rain is the rain field the training labels were simulated from",
    "design_storm": "junction rain averaged over stochastic rain-field realisations drawn on the canonical "
                    "storm-anchored time axis shared by every design storm (clock-independent)",
    "ensemble": "junction rain averaged over stochastic rain-field realisations (only corridor-average rain is known, "
                "so the junction pattern is uncertain)",
}


def _tz(cfg: Mapping[str, Any]) -> str:
    return str((cfg.get("project") or {}).get("timezone") or DEFAULT_TZ)


def _timestamp(value: Any, tz: str, name: str) -> pd.Timestamp:
    """Parse ``value`` to a local tz-aware timestamp (naive values are local time)."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{name} must be an ISO date/time, got {value!r}") from exc
    if pd.isna(ts):
        raise ScenarioError(f"{name} must be an ISO date/time, got {value!r}")
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def _now(now: Any, tz: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=tz) if now is None else _timestamp(now, tz, "now")


# --------------------------------------------------------------------------- the scenario object


def _hourly_index(timestamps: Any) -> pd.DatetimeIndex:
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ValueError(f"Scenario.timestamps must be a pandas DatetimeIndex, got {type(timestamps).__name__}")
    if timestamps.tz is None:
        raise ValueError("Scenario.timestamps must be tz-aware (schema 2.2 local time)")
    if len(timestamps) == 0:
        raise ValueError("Scenario.timestamps must contain at least one hour")
    if timestamps.hasnans:
        raise ValueError("Scenario.timestamps must not contain NaT")
    if len(timestamps) > 1:
        steps = np.diff(timestamps.as_unit("s").asi8)
        if not np.all(steps == 3600):
            raise ValueError("Scenario.timestamps must be strictly increasing, gap-free hourly steps")
    return pd.DatetimeIndex(timestamps, name="timestamp", freq="h" if len(timestamps) > 1 else None)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Scenario:
    """Hourly areal rain for one prediction run (see the module docstring).

    ``areal_mm`` is ``float64 [T]`` (finite, >= 0), ``is_forecast`` ``bool [T]`` and
    ``target_start`` the index of the first hour to predict (``0 <= target_start < T``). Arrays
    are copied and made read-only. ``field_shift_h`` shifts the time axis the junction rain field
    is drawn on (:attr:`field_timestamps`; 0 = the real hours) and ``notes`` are caveats a UI
    should show next to the result (e.g. a truncated forecast). ``exact_field`` is True when the
    scenario replays the record the training labels were simulated from, so the predictor uses
    the exact training rain field; otherwise (forecasts, design storms, custom series) only the
    corridor-average rain is known and predictors average over a rain-field ensemble.
    """

    name: str
    kind: str
    timestamps: pd.DatetimeIndex
    areal_mm: np.ndarray
    is_forecast: np.ndarray | None
    target_start: int
    description: str = ""
    source: str = "custom"
    field_shift_h: int = 0
    notes: tuple[str, ...] = ()
    exact_field: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Scenario.kind must be one of {KINDS}, got {self.kind!r}")
        index = _hourly_index(self.timestamps)
        n_hours = len(index)
        try:
            areal = np.array(self.areal_mm, dtype=np.float64, copy=True).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Scenario.areal_mm must be numeric: {exc}") from exc
        if areal.size != n_hours:
            raise ValueError(f"Scenario.areal_mm has {areal.size} values but timestamps has {n_hours}")
        if not np.isfinite(areal).all() or (areal < 0).any():
            raise ValueError("Scenario.areal_mm must be finite and >= 0 (clean the series first)")
        flags = np.zeros(n_hours, dtype=bool) if self.is_forecast is None else self.is_forecast
        forecast = np.array(flags, dtype=bool, copy=True).reshape(-1)
        if forecast.size != n_hours:
            raise ValueError(f"Scenario.is_forecast has {forecast.size} values but timestamps has {n_hours}")
        start = self.target_start
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= int(start) < n_hours:
            raise ValueError(f"Scenario.target_start must be an index in [0, {n_hours - 1}], got {start!r}")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "timestamps", index)
        object.__setattr__(self, "areal_mm", _frozen(areal))
        object.__setattr__(self, "is_forecast", _frozen(forecast))
        object.__setattr__(self, "target_start", int(start))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "source", str(self.source))
        shift = self.field_shift_h
        if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
            raise ValueError(f"Scenario.field_shift_h must be a whole number of hours, got {shift!r}")
        object.__setattr__(self, "field_shift_h", int(shift))
        notes = (self.notes,) if isinstance(self.notes, str) else tuple(self.notes or ())
        object.__setattr__(self, "notes", tuple(str(note) for note in notes))
        if not isinstance(self.exact_field, (bool, np.bool_)):
            raise ValueError(f"Scenario.exact_field must be a bool, got {self.exact_field!r}")
        object.__setattr__(self, "exact_field", bool(self.exact_field))

    @property
    def n_hours(self) -> int:
        return len(self.timestamps)

    @property
    def n_target_hours(self) -> int:
        return self.n_hours - self.target_start

    @property
    def history_hours(self) -> int:
        return self.target_start

    @property
    def target_timestamps(self) -> pd.DatetimeIndex:
        return self.timestamps[self.target_start:]

    @property
    def target_areal_mm(self) -> np.ndarray:
        return self.areal_mm[self.target_start:]

    @property
    def field_timestamps(self) -> pd.DatetimeIndex:
        """The time axis the junction rain field is drawn on (the real hours unless shifted)."""
        return self.timestamps if not self.field_shift_h else self.timestamps + self.field_shift_h * HOUR

    @property
    def rain_field_note(self) -> str:
        """What the junction rain of this scenario represents (the predictor's result metadata
        ``rain_field`` adds the ensemble size)."""
        if self.exact_field:
            return RAIN_FIELD_NOTES["exact"]
        return RAIN_FIELD_NOTES["design_storm" if self.field_shift_h else "ensemble"]

    def rain_total_mm(self, hours: int | None = None) -> float:
        """Areal rain over the first ``hours`` target hours (all of them when None)."""
        target = self.target_areal_mm
        return float(target.sum() if hours is None else target[: max(0, int(hours))].sum())

    def to_frame(self) -> pd.DataFrame:
        """``precipitation_mm``, ``is_forecast`` and ``is_target`` per hour (for hyetographs)."""
        is_target = np.arange(self.n_hours) >= self.target_start
        return pd.DataFrame(
            {PRECIP: self.areal_mm.copy(), "is_forecast": self.is_forecast.copy(), "is_target": is_target},
            index=self.timestamps,
        )


# --------------------------------------------------------------------------- builders


def forecast_scenario(cfg: Mapping[str, Any], now: Any = None, hours: int | None = None) -> Scenario:
    """Live forecast: recent model rain as history, then up to ``hours`` (default
    ``max(horizons_h)``) forecast hours.

    The forecast resolves only corridor-average rain, so predictors average over a rain-field
    ensemble drawn on the real hours (``exact_field=False``; see the module docstring). Raises
    :class:`WeatherUnavailable` (offline, API failure, or no future hours) so the caller can fall
    back (the app replays the most recent notable event).
    """
    settings = InferenceSettings.from_config(cfg)
    wanted = settings.max_horizon_h if hours is None else _whole(hours, "hours", 1)
    frame = weather.fetch_forecast(cfg, now=now)
    flags = frame["is_forecast"].to_numpy(dtype=bool) if "is_forecast" in frame else np.zeros(len(frame), bool)
    future = np.flatnonzero(flags)
    if future.size == 0:
        raise WeatherUnavailable(f"The Open-Meteo forecast has no hours after {_now(now, _tz(cfg)):%Y-%m-%d %H:%M}")
    t0 = int(future[0])
    stop = min(len(frame), t0 + wanted)
    notes: tuple[str, ...] = ()
    if stop - t0 < wanted:
        LOGGER.warning("Forecast covers only %d of the requested %d hours ahead", stop - t0, wanted)
        notes = (f"The forecast covers only the next {stop - t0} of the requested {wanted} hours.",)
    areal, index = weather.areal_series(frame.iloc[:stop])
    issued = index[t0] - HOUR
    total = float(areal[t0:stop].sum())
    return Scenario(
        name=f"Live forecast (issued {issued:%Y-%m-%d %H:%M})", kind="forecast", timestamps=index,
        areal_mm=areal, is_forecast=flags[:stop], target_start=t0, source="forecast",
        description=f"Open-Meteo forecast: {total:.1f} mm over the next {stop - t0} h "
                    f"({t0} h of recent model rain as history)", notes=notes,
    )


def design_storm_scenario(
    cfg: Mapping[str, Any],
    total_mm: float,
    duration_h: int,
    storm_start_offset_h: int | None = 6,
    horizon_h: int = 48,
    now: Any = None,
    rain_scale: float | None = None,
) -> Scenario:
    """What-if: ``inference.history_hours`` dry hours, then ``horizon_h`` target hours whose storm
    (``total_mm`` over ``duration_h``, shape ``inference.design_storm_shape``) starts
    ``storm_start_offset_h`` hours after the first target hour (the next full hour after ``now``).

    ``total_mm`` is a point (rain-gauge) total, like the presets. The model and the hydrology
    were calibrated on ERA5 *areal* rain, which smooths convective storms several-fold (Sept
    2022: ~130 mm at gauges in 24 h vs ~22 mm of ERA5 areal rain over the same 24 h; the default
    0.14 is the 17.9 mm 16:00-20:00 burst / 130 mm), so the hyetograph is
    multiplied by ``rain_scale`` (default ``inference.design_storm_rain_scale``; 1.0 = as given).

    The junction rain-field ensemble is drawn on a canonical time axis on which the storm starts
    at ``inference.design_storm_field_anchor`` (``Scenario.field_shift_h``), so the same storm
    gives the same members whatever ``now`` and the offset are, and every design storm shares
    them (common random numbers); displayed hours stay real.
    """
    settings = InferenceSettings.from_config(cfg)
    tz = _tz(cfg)
    offset = settings.design_storm_offset_h if storm_start_offset_h is None else _whole(
        storm_start_offset_h, "storm_start_offset_h", 0)
    horizon = _whole(horizon_h, "horizon_h", 1)
    duration = _whole(duration_h, "duration_h", 1)
    scale = settings.design_storm_rain_scale if rain_scale is None else _real(rain_scale, "rain_scale", 1e-3, 10.0)
    gauge_total = _real(total_mm, "total_mm", 0.0, 1e4)
    if horizon > settings.max_scenario_hours:
        raise ScenarioError(f"horizon_h={horizon} exceeds inference.max_scenario_hours={settings.max_scenario_hours}")
    if offset + duration > horizon:
        raise ScenarioError(f"The storm (offset {offset} h + duration {duration} h) does not fit in "
                            f"horizon_h={horizon}")
    t0 = _now(now, tz).floor("h") + HOUR
    frame = weather.design_storm(
        gauge_total * scale, duration, start=t0 + offset * HOUR, lead_hours=settings.history_hours + offset,
        horizon_hours=horizon - offset, tz=tz, shape=settings.design_storm_shape,
        peak_position=settings.design_storm_peak_position,
    )
    areal, index = weather.areal_series(frame)
    scaled = "" if scale == 1.0 else f" = {areal.sum():.1f} mm areal-equivalent (x{scale:g})"
    anchor = pd.Timestamp(settings.design_storm_field_anchor).tz_localize(tz)
    shift_h = int((anchor - index[settings.history_hours + offset]) // HOUR)
    return Scenario(
        name=f"Design storm: {gauge_total:g} mm in {duration} h", kind="design_storm", timestamps=index,
        areal_mm=areal, is_forecast=index >= t0, target_start=settings.history_hours, source="design_storm",
        description=(f"{settings.design_storm_shape.title()} hyetograph, {gauge_total:g} mm (gauge){scaled} over "
                     f"{duration} h starting {offset} h into a {horizon} h horizon after {settings.history_hours} dry "
                     f"hours (peak {areal.max():.1f} mm/h)"), field_shift_h=shift_h,
    )


def preset_scenario(cfg: Mapping[str, Any], preset: str | int, horizon_h: int = 48, now: Any = None,
                    rain_scale: float | None = None, storm_start_offset_h: int | None = None) -> Scenario:
    """Design storm from ``inference.design_storms`` by (case-insensitive) name, name prefix or index.

    ``storm_start_offset_h`` (default ``inference.design_storm_offset_h``) is passed through to
    :func:`design_storm_scenario`.
    """
    presets = InferenceSettings.from_config(cfg).design_storms
    if not presets:
        raise ScenarioError("inference.design_storms is empty; pass total_mm / duration_h instead")
    chosen = None
    if (isinstance(preset, (int, np.integer)) and not isinstance(preset, bool)) or str(preset).strip().isdigit():
        index = int(preset)
        chosen = presets[index] if 0 <= index < len(presets) else None
    else:
        key = str(preset).strip().lower()
        matches = ([p for p in presets if p.name.lower() == key]
                   or [p for p in presets if p.name.lower().startswith(key)])
        chosen = matches[0] if len(matches) == 1 else None
    if chosen is None:
        names = ", ".join(f"{i}: {p.name}" for i, p in enumerate(presets))
        raise ScenarioError(f"Unknown or ambiguous design-storm preset {preset!r}; choose one of {names}")
    scenario = design_storm_scenario(cfg, chosen.total_mm, chosen.duration_h, storm_start_offset_h, horizon_h, now,
                                     rain_scale)
    return dataclasses.replace(scenario, name=chosen.name)


# --------------------------------------------------------------------------- historical record


def _event_aligned_start(record: pd.DataFrame, first: int, cfg: Mapping[str, Any], backfill_h: int) -> tuple[int, bool]:
    """``(start, capped)``: the scenario's first row moved back to the start of a rain event in progress there.

    The rain field of an hour depends on the start of the event containing it, so starting a
    replay mid-event would give a different junction field than the dataset builder's
    full-record field. Events are found on a window that reaches ``max_dry_gap_h + 1`` hours
    before the earliest allowed start (so an event start inside the backfill range is exact)
    and ``max_dry_gap_h + 1`` hours past ``first`` — a dry gap hour inside an event belongs to
    it only if rain resumes within the gap, which a window ending at ``first`` cannot see.
    ``capped`` is True when the event began before ``inference.max_event_backfill_h``.
    """
    if first == 0 or backfill_h == 0:
        return first, False
    gap = RainFieldParams.from_config(cfg).max_dry_gap_h
    lo = max(0, first - backfill_h)
    lo_ext = max(0, lo - gap - 1)
    stop = min(len(record), first + gap + 2)
    areal, index = weather.areal_series(record.iloc[lo_ext:stop])
    events = find_rain_events(areal, index, gap, _tz(cfg))
    event = int(events.event_of_hour[first - lo_ext])
    if event < 0:
        return first, False
    event_start = lo_ext + int(events.start_idx[event])
    if event_start >= first:
        return first, False
    if event_start < lo and lo_ext > 0:
        LOGGER.warning("The replay history starts inside a rain event longer than inference.max_event_backfill_h=%d h;"
                       " its junction rain field may differ slightly from the training field", backfill_h)
        return first, True
    LOGGER.info("Replay history extended by %d h to the start of the rain event in progress", first - event_start)
    return event_start, False


def _replay_rows(cfg: Mapping[str, Any], t0: pd.Timestamp, n_hours: int,
                 settings: InferenceSettings) -> tuple[pd.DataFrame, np.ndarray, int, int, list[str]]:
    """``(block, gap_filled, target, stop, notes)``: the stored block holding ``t0`` and the target rows.

    Starts outside the record, in a gap between stored blocks or on gap-filled hours raise
    :class:`ScenarioError` (there is no observed rain to replay there).
    """
    record = load_weather_record(cfg)
    a, b = block_containing(record, t0)
    block = record.iloc[a: b + 1]
    filled = gap_filled_mask(block, cfg)
    target = int(block.index.get_loc(t0))
    if filled[target]:
        raise ScenarioError(f"start {t0:%Y-%m-%d %H:%M} falls on gap-filled hours of the weather record (stored as "
                            "0 mm and flagged is_imputed / source 'missing', not observed rain); pick another start")
    stop = min(len(block), target + n_hours)
    notes: list[str] = []
    if stop - target < n_hours:
        end = block.index[-1]
        LOGGER.warning("The weather record ends %s; replay truncated to %d of %d hours", f"{end:%Y-%m-%d %H:%M}",
                       stop - target, n_hours)
        if b < len(record) - 1:
            notes.append(f"The stored weather block ends {end:%Y-%m-%d %H:%M} (the next hours are missing from the "
                         f"record), so the replay covers only {stop - target} of the requested {n_hours} h.")
    if target < settings.history_hours and a > 0:
        notes.append(f"The replay starts {target} h into a stored block of the weather record; the hours before "
                     "it are missing, so the rain history is shorter than inference.history_hours.")
    return block, filled, target, stop, notes


def historical_scenario(cfg: Mapping[str, Any], start: Any, hours: int = 48) -> Scenario:
    """Replay ``hours`` hours of the weather record from ``start`` (local time when naive).

    Only stored hours are replayed: a start in a gap between stored blocks or on gap-filled
    hours raises :class:`ScenarioError`, and gap-filled hours inside the window are noted.
    History (``inference.history_hours``, extended to the start of a rain event in progress)
    is included; a target period running past the stored block is truncated with a WARNING.
    The scenario reproduces the training rain field (``exact_field=True``).
    """
    settings = InferenceSettings.from_config(cfg)
    tz = _tz(cfg)
    n_hours = _whole(hours, "hours", 1)
    if n_hours > settings.max_scenario_hours:
        raise ScenarioError(f"hours={n_hours} exceeds inference.max_scenario_hours={settings.max_scenario_hours}")
    t0 = _timestamp(start, tz, "start").floor("h")
    block, filled, target, stop, notes = _replay_rows(cfg, t0, n_hours, settings)
    first = max(0, target - settings.history_hours)
    if target - first < settings.history_hours:
        LOGGER.warning("Only %d h of history precede %s in the record (%d wanted)", target - first,
                       f"{t0:%Y-%m-%d %H:%M}", settings.history_hours)
    first, capped = _event_aligned_start(block, first, cfg, settings.max_event_backfill_h)
    if capped:
        notes.append(f"The replay history starts inside a rain event longer than {settings.max_event_backfill_h} h "
                     "(inference.max_event_backfill_h), so junction rain in its first hours may differ slightly from "
                     "the training field.")
    window, window_filled = block.iloc[first:stop], filled[first:stop]
    if window_filled.any():
        in_target = int(filled[target:stop].sum())
        notes.append(f"{int(window_filled.sum())} replayed hours ({in_target} of them predicted) are gap-filled "
                     "hours of the weather record (0 mm flagged is_imputed / missing), not observed rain.")
    areal, index = weather.areal_series(window)
    source = dominant_source(window, window_filled)
    observed = window["source"].astype(str)[~window_filled] if "source" in window else pd.Series(dtype=str)
    note = " (includes synthetic hours)" if (observed == "synthetic").any() and source != "synthetic" else ""
    total = float(areal[target - first:].sum())
    return Scenario(
        name=f"Historical replay {t0:%Y-%m-%d %H:%M}", kind="historical", timestamps=index, areal_mm=areal,
        is_forecast=np.zeros(len(index), dtype=bool), target_start=target - first, source=source,
        description=f"{source} record: {total:.1f} mm over {stop - target} h from {t0:%Y-%m-%d %H:%M}{note}",
        notes=tuple(notes), exact_field=True,
    )


def _as_rain_frame(rain: Any, tz: str) -> pd.DataFrame:
    """``value`` (float, may be NaN) and ``flag`` (is_forecast) columns on a sorted, unique local index."""
    if isinstance(rain, pd.DataFrame):
        column = PRECIP if PRECIP in rain else "precipitation" if "precipitation" in rain else None
        if column is None:
            raise ScenarioError(f"custom rain needs a '{PRECIP}' column, got {list(rain.columns)}")
        series = rain[column]
        flags = rain["is_forecast"].fillna(False).astype(bool) if "is_forecast" in rain else None
    elif isinstance(rain, pd.Series):
        series, flags = rain, None
    else:
        raise ScenarioError(f"custom rain must be a pandas Series or DataFrame, got {type(rain).__name__}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ScenarioError("custom rain must be indexed by a DatetimeIndex")
    index = series.index.tz_localize(tz) if series.index.tz is None else series.index.tz_convert(tz)
    frame = pd.DataFrame({
        "value": pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64),
        "flag": np.zeros(len(series), dtype=bool) if flags is None else flags.to_numpy(dtype=bool),
    }, index=index)
    frame = frame[~frame.index.isna()]
    duplicated = frame.index.duplicated(keep="last")
    if duplicated.any():
        LOGGER.warning("custom rain: %d duplicate timestamps dropped (kept the last)", int(duplicated.sum()))
    return frame[~duplicated].sort_index()


def custom_scenario(
    cfg: Mapping[str, Any],
    rain: pd.Series | pd.DataFrame,
    *,
    target_start: int | None = None,
    name: str = "Custom scenario",
    description: str = "",
) -> Scenario:
    """Scenario from any hourly areal series (NaN/negative → 0 and missing hours → 0, with WARNINGs).

    ``target_start`` defaults to the first ``is_forecast`` hour (DataFrame input) or else to
    ``inference.history_hours`` (clipped to the series).
    """
    settings = InferenceSettings.from_config(cfg)
    frame = _as_rain_frame(rain, _tz(cfg))
    if frame.empty:
        raise ScenarioError("custom rain has no valid timestamps")
    if (frame.index != frame.index.floor("h")).any():
        raise ScenarioError("custom rain timestamps must fall on whole hours")
    values = frame["value"].to_numpy()
    bad = ~np.isfinite(values) | (np.nan_to_num(values, nan=0.0) < 0)
    if bad.any():
        LOGGER.warning("custom rain: %d NaN/non-finite/negative values set to 0 mm", int(bad.sum()))
    clean = pd.Series(np.where(bad, 0.0, values), index=frame.index)
    full = pd.date_range(frame.index[0], frame.index[-1], freq="h", name="timestamp")
    if len(full) > len(frame):
        LOGGER.warning("custom rain: %d missing hours filled with 0 mm", len(full) - len(frame))
    areal = clean.reindex(full, fill_value=0.0).to_numpy(dtype=np.float64)
    flags = frame["flag"].reindex(full, fill_value=False).to_numpy(dtype=bool)
    if target_start is None:
        target_start = int(np.flatnonzero(flags)[0]) if flags.any() else min(settings.history_hours, len(full) - 1)
    if not 0 <= _whole(target_start, "target_start", 0) < len(full):
        raise ScenarioError(f"target_start={target_start} must index the {len(full)} hours of the series")
    return Scenario(name, "custom", full, areal, flags, int(target_start), description, "custom")
