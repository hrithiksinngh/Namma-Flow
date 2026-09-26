"""Tests for weather ingestion, cleaning, caching, synthetic climatology and design storms."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.data_pipeline.weather as weather
from src.data_pipeline.weather import (
    WeatherFormatError,
    WeatherUnavailable,
    areal_series,
    clean_weather,
    fetch_archive,
    fetch_forecast,
    fetch_previous_runs,
    load_or_fetch_weather,
    read_weather_csv,
    save_weather_csv,
    summarize_weather,
)
from src.utils.config import ConfigError, deep_merge, resolve_path
from src.utils.http import NetworkUnavailable
from tests.conftest import TZ

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "src" / "data_pipeline" / "03_weather_ingestion.py"


# --------------------------------------------------------------------------- helpers & fixtures


@pytest.fixture
def warn_log(caplog):
    """Capture records of the project logger (it does not propagate to the root logger)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


def _warnings(caplog) -> str:
    return " | ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def _hourly(start: str, periods: int, values=None, tz: str | None = TZ) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h", tz=tz)
    data = np.zeros(periods) if values is None else np.asarray(values, dtype=float)
    return pd.DataFrame({"precipitation_mm": data}, index=index)


def _rain_for(times: pd.DatetimeIndex) -> list[float]:
    """Deterministic fake rain: 2 mm at 17:00 local on even days, 0 elsewhere."""
    return [2.0 if (t.hour == 17 and t.day % 2 == 0) else 0.0 for t in times]


class FakeOpenMeteo:
    """Stand-in for ``get_json`` that serves archive / forecast / previous-runs payloads."""

    def __init__(self, fail_on_call: int | None = None, null_every: int | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail_on_call = fail_on_call
        self.null_every = null_every

    def __call__(self, url, params=None, **kwargs):
        self.calls.append((url, dict(params or {})))
        if kwargs.get("offline"):
            raise NetworkUnavailable("offline")
        if self.fail_on_call is not None and len(self.calls) >= self.fail_on_call:
            raise NetworkUnavailable("simulated outage")
        if "past_days" in params:
            start = pd.Timestamp("2026-09-21")
            times = pd.date_range(start, periods=24 * 5, freq="h")
        else:
            start, end = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
            times = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for name in params["hourly"].split(","):
            values = _rain_for(times)
            if name.endswith("day1"):
                values = [v * 0.5 for v in values]
            if self.null_every:
                values = [None if i % self.null_every == 0 else v for i, v in enumerate(values)]
            hourly[name] = values
        return {"latitude": 12.9, "longitude": 77.66, "utc_offset_seconds": 19800, "hourly": hourly}


@pytest.fixture
def online_cfg(cfg, monkeypatch):
    """Config that believes it is online; ``weather.get_json`` must be patched by the test."""
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)

    def _refuse(*args, **kwargs):  # pragma: no cover - guard against accidental real HTTP
        raise AssertionError("real HTTP attempted in an offline test")

    monkeypatch.setattr("src.utils.http.requests.get", _refuse)
    return deep_merge(cfg, {"project": {"offline": False}})


@pytest.fixture
def fake_api(monkeypatch):
    fake = FakeOpenMeteo()
    monkeypatch.setattr(weather, "get_json", fake)
    return fake


def _short_range(cfg: dict, start: str = "2022-08-01", end: str = "2022-09-30") -> dict:
    return deep_merge(cfg, {"weather": {"start_date": start, "end_date": end}})


# --------------------------------------------------------------------------- clean_weather


@pytest.mark.unit
def test_clean_weather_enforces_schema(cfg):
    out = clean_weather(_hourly("2022-09-04", 24, np.arange(24) / 10), cfg)
    assert list(out.columns) == ["precipitation_mm", "is_imputed", "source"]
    assert out.index.name == "timestamp" and out.index.freqstr == "h"
    assert str(out.index.tz) == TZ and out.index.is_monotonic_increasing and out.index.is_unique
    assert out["precipitation_mm"].dtype == np.float64
    assert out["is_imputed"].dtype == bool and not out["is_imputed"].any()
    assert set(out["source"]) == {"open_meteo"}


@pytest.mark.unit
def test_clean_weather_repairs_values(cfg, warn_log):
    values = [1.0, -2.0, 500.0, np.inf, 3.0, 0.5]
    frame = _hourly("2022-09-04", 6, values)
    frame["precipitation_mm"] = frame["precipitation_mm"].astype(object)
    frame.iloc[5, 0] = "n/a"
    out = clean_weather(frame, cfg)
    assert out["precipitation_mm"].tolist() == [1.0, 0.0, 150.0, 0.0, 3.0, 0.0]
    assert out["is_imputed"].tolist() == [False, False, False, True, False, True]
    text = _warnings(warn_log)
    assert "negative" in text and "clipped" in text and "non-numeric" in text


@pytest.mark.unit
def test_clean_weather_sorts_dedupes_and_fills_gaps(cfg, warn_log):
    frame = _hourly("2022-09-04", 30, np.ones(30))
    frame = frame.drop(frame.index[3:5])          # 2-hour gap (short)
    frame = frame.drop(frame.index[10:20])        # 10-hour gap (long, > max_fill_gap_hours)
    dup = frame.iloc[[0]].copy()
    dup.iloc[0, 0] = 7.0
    shuffled = pd.concat([frame.iloc[::-1], dup])
    out = clean_weather(shuffled, cfg)
    assert len(out) == 30 and out.index.is_monotonic_increasing
    assert out["precipitation_mm"].iloc[0] == 7.0, "duplicates keep the last occurrence"
    assert out["is_imputed"].sum() == 12
    assert (out.loc[out["is_imputed"], "precipitation_mm"] == 0).all()
    assert "gap" in _warnings(warn_log) and "10 h" in _warnings(warn_log)


@pytest.mark.unit
def test_clean_weather_handles_timezones_and_columns(cfg):
    naive = _hourly("2022-09-04 10:00", 3, [1, 2, 3], tz=None)
    out = clean_weather(naive, cfg)
    assert out.index[0] == pd.Timestamp("2022-09-04 10:00", tz=TZ)
    utc = _hourly("2022-09-04 04:30", 3, [1, 2, 3], tz="UTC")
    assert clean_weather(utc, cfg).index[0] == pd.Timestamp("2022-09-04 10:00", tz=TZ)
    as_columns = pd.DataFrame(
        {"timestamp": ["2022-09-04T10:00:00+05:30", "2022-09-04T11:00:00+05:30"], "precipitation": [1.0, 2.0]}
    )
    out = clean_weather(as_columns, cfg)
    assert out["precipitation_mm"].tolist() == [1.0, 2.0]
    mixed = pd.DataFrame(
        {"timestamp": ["2022-09-04T10:00:00+05:30", "2022-09-04T05:30:00+00:00"], "precipitation_mm": [1.0, 2.0]}
    )
    out = clean_weather(mixed, cfg)
    assert len(out) == 2 and out.index[1] == pd.Timestamp("2022-09-04 11:00", tz=TZ)


@pytest.mark.unit
def test_clean_weather_range_bias_and_subhourly(cfg, warn_log):
    biased = deep_merge(cfg, {"weather": {"bias_correction_factor": 1.5}})
    frame = _hourly("2022-09-04 05:00", 2, [2.0, 4.0])
    out = clean_weather(frame, biased, start="2022-09-04", end="2022-09-04")
    assert len(out) == 24 and out.index[0].hour == 0 and out.index[-1].hour == 23
    assert out.loc["2022-09-04 05:00+05:30", "precipitation_mm"] == 3.0
    assert out["is_imputed"].sum() == 22
    sub = pd.DataFrame(
        {"precipitation_mm": [1.0, 2.0, 4.0]},
        index=pd.DatetimeIndex(["2022-09-04 10:30", "2022-09-04 11:00", "2022-09-04 11:15"]).tz_localize(TZ),
    )
    agg = clean_weather(sub, cfg)
    assert agg["precipitation_mm"].tolist() == [3.0, 4.0]
    assert "sub-hourly" in _warnings(warn_log)


@pytest.mark.unit
def test_clean_weather_empty_and_invalid_inputs(cfg):
    empty = _hourly("2022-09-04", 0)
    out = clean_weather(empty, cfg, start="2022-09-04", end="2022-09-05")
    assert len(out) == 48 and out["is_imputed"].all()
    with pytest.raises(ValueError, match="no valid"):
        clean_weather(empty, cfg)
    with pytest.raises(ValueError, match="precipitation"):
        clean_weather(pd.DataFrame({"rain": [1.0]}, index=pd.date_range("2022", periods=1, tz=TZ)), cfg)
    with pytest.raises(ValueError, match="timestamp"):
        clean_weather(pd.DataFrame({"precipitation_mm": [1.0, 2.0]}), cfg)
    with pytest.raises(TypeError):
        clean_weather([1, 2, 3], cfg)
    with pytest.raises(ValueError, match="start"):
        clean_weather(empty, cfg, start="2022-09-05", end="2022-09-04")
    with pytest.raises(ValueError, match="date"):
        clean_weather(empty, cfg, start="not-a-date", end="2022-09-04")


@pytest.mark.unit
def test_clean_weather_keeps_flags_and_does_not_mutate(cfg):
    frame = _hourly("2022-09-04", 3, [1.0, -1.0, 2.0])
    frame["is_imputed"] = ["True", "False", "False"]
    frame["source"] = "forecast"
    frame["is_forecast"] = [False, True, True]
    before = frame.copy()
    out = clean_weather(frame, cfg)
    pd.testing.assert_frame_equal(frame, before)
    assert out["is_imputed"].tolist() == [True, False, False]
    assert out["is_forecast"].tolist() == [False, True, True]
    assert set(out["source"]) == {"forecast"}


@pytest.mark.unit
def test_invalid_weather_config_raises(cfg):
    for bad in ({"chunk_days": 0}, {"max_precip_mm_h": -1}, {"bias_correction_factor": 0}, {"timeout_s": "x"}):
        with pytest.raises(ConfigError):
            clean_weather(_hourly("2022-09-04", 2), deep_merge(cfg, {"weather": bad}))


# --------------------------------------------------------------------------- CSV round trip


@pytest.mark.unit
def test_csv_round_trip(cfg, tmp_path):
    frame = clean_weather(_hourly("2022-09-04", 48, np.linspace(0, 5, 48)), cfg)
    path = save_weather_csv(frame, tmp_path / "w" / "weather.csv")
    text = path.read_text().splitlines()
    assert text[0] == "timestamp,precipitation_mm,is_imputed,source"
    assert text[1].startswith("2022-09-04T00:00:00+05:30,")
    back = read_weather_csv(path, TZ)
    pd.testing.assert_frame_equal(back, frame, check_freq=True)


@pytest.mark.unit
def test_read_weather_csv_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="03_weather_ingestion"):
        read_weather_csv(tmp_path / "missing.csv", TZ)
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(WeatherFormatError, match="timestamp"):
        read_weather_csv(bad, TZ)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(WeatherFormatError):
        read_weather_csv(empty, TZ)
    garbage = tmp_path / "garbage.csv"
    garbage.write_text("timestamp,precipitation_mm\nnope,1\nnever,2\n")
    with pytest.raises(WeatherFormatError, match="timestamps"):
        read_weather_csv(garbage, TZ)
    naive = tmp_path / "naive.csv"
    naive.write_text("timestamp,precipitation_mm\n2022-09-04 00:00,1.5\n2022-09-04 01:00,0\n")
    out = read_weather_csv(naive, TZ)
    assert out.index[0] == pd.Timestamp("2022-09-04", tz=TZ) and out["precipitation_mm"].iloc[0] == 1.5


@pytest.mark.unit
def test_save_weather_csv_rejects_bad_frames(tmp_path):
    with pytest.raises(ValueError):
        save_weather_csv(pd.DataFrame({"precipitation_mm": [1.0]}), tmp_path / "x.csv")
    naive = _hourly("2022-09-04", 2, tz=None)
    with pytest.raises(ValueError, match="tz-aware"):
        save_weather_csv(naive, tmp_path / "x.csv")


# --------------------------------------------------------------------------- archive client


@pytest.mark.unit
def test_fetch_archive_chunks_and_parses(online_cfg, fake_api):
    small_chunks = deep_merge(online_cfg, {"weather": {"chunk_days": 10}})
    out = fetch_archive(small_chunks, "2022-09-01", "2022-09-25", 12.93, 77.68)
    assert len(fake_api.calls) == 3
    first_url, first_params = fake_api.calls[0]
    assert first_url == online_cfg["weather"]["archive_url"]
    assert first_params["start_date"] == "2022-09-01" and first_params["end_date"] == "2022-09-10"
    assert first_params["timezone"] == TZ and first_params["hourly"] == "precipitation"
    assert fake_api.calls[-1][1]["end_date"] == "2022-09-25"
    assert len(out) == 25 * 24 and str(out.index.tz) == TZ
    assert out.loc["2022-09-02 17:00+05:30", "precipitation_mm"] == 2.0
    assert set(out["source"]) == {"open_meteo"}


@pytest.mark.unit
def test_fetch_archive_offline_and_bad_inputs(cfg, online_cfg, monkeypatch):
    with pytest.raises(NetworkUnavailable):
        fetch_archive(cfg, "2022-09-01", "2022-09-02", 12.93, 77.68)
    with pytest.raises(ValueError, match="latitude"):
        fetch_archive(online_cfg, "2022-09-01", "2022-09-02", 123.0, 77.68)
    with pytest.raises(ValueError):
        fetch_archive(online_cfg, "2022-09-05", "2022-09-02", 12.93, 77.68)
    monkeypatch.setattr(weather, "get_json", lambda *a, **k: {"error": True, "reason": "nope"})
    with pytest.raises(NetworkUnavailable, match="hourly"):
        fetch_archive(online_cfg, "2022-09-01", "2022-09-02", 12.93, 77.68)
    monkeypatch.setattr(
        weather, "get_json", lambda *a, **k: {"hourly": {"time": ["2022-09-01T00:00"], "precipitation": [1, 2]}}
    )
    with pytest.raises(NetworkUnavailable, match="length"):
        fetch_archive(online_cfg, "2022-09-01", "2022-09-01", 12.93, 77.68)


# --------------------------------------------------------------------------- load_or_fetch_weather


@pytest.mark.integration
def test_load_or_fetch_offline_builds_synthetic_cache(cfg, warn_log, monkeypatch):
    monkeypatch.setattr(weather, "get_json", FakeOpenMeteo())
    out = load_or_fetch_weather(_short_range(cfg))
    assert len(out) == 61 * 24 and set(out["source"]) == {"synthetic"}
    assert "synthetic" in _warnings(warn_log)
    path = resolve_path(cfg, "weather_file")
    assert path.exists() and path.with_suffix(".meta.json").exists()
    again = load_or_fetch_weather(_short_range(cfg))
    pd.testing.assert_frame_equal(again, out)


@pytest.mark.integration
def test_load_or_fetch_online_writes_and_reuses_cache(online_cfg, fake_api):
    short = _short_range(online_cfg)
    out = load_or_fetch_weather(short)
    assert set(out["source"]) == {"open_meteo"} and len(out) == 61 * 24
    assert out["precipitation_mm"].sum() == pytest.approx(2.0 * 30)
    meta = json.loads(resolve_path(short, "weather_file").with_suffix(".meta.json").read_text())
    assert meta["bias_correction_factor"] == 1.0 and meta["rows"] == len(out)
    n_calls = len(fake_api.calls)
    cached = load_or_fetch_weather(short)
    assert len(fake_api.calls) == n_calls, "a covering cache must not hit the network"
    pd.testing.assert_frame_equal(cached, out)
    sub = load_or_fetch_weather(_short_range(online_cfg, "2022-08-10", "2022-08-20"))
    assert len(sub) == 11 * 24 and len(fake_api.calls) == n_calls


@pytest.mark.integration
def test_load_or_fetch_extends_partial_cache_and_replaces_synthetic(online_cfg, fake_api, cfg):
    load_or_fetch_weather(_short_range(cfg, "2022-08-01", "2022-08-31"))  # offline: synthetic August
    out = load_or_fetch_weather(_short_range(online_cfg, "2022-08-01", "2022-09-30"))
    assert set(out["source"]) == {"open_meteo"}, "online runs replace synthetic rows"
    fake_api.calls.clear()
    load_or_fetch_weather(_short_range(online_cfg, "2022-08-01", "2022-10-15"))
    assert len(fake_api.calls) == 1
    assert fake_api.calls[0][1]["start_date"] == "2022-10-01"
    assert fake_api.calls[0][1]["end_date"] == "2022-10-15"


@pytest.mark.integration
def test_load_or_fetch_network_failure_falls_back(online_cfg, monkeypatch, warn_log):
    fake = FakeOpenMeteo(fail_on_call=2)
    monkeypatch.setattr(weather, "get_json", fake)
    short = deep_merge(_short_range(online_cfg, "2022-01-01", "2022-03-31"), {"weather": {"chunk_days": 31}})
    out = load_or_fetch_weather(short)
    assert len(fake.calls) == 2, "stop calling the API after the first failure"
    assert set(out["source"]) == {"open_meteo", "synthetic"}
    assert out.loc[:"2022-01-31", "source"].eq("open_meteo").all()
    assert "synthetic" in _warnings(warn_log)


@pytest.mark.integration
def test_load_or_fetch_force_refetches_or_keeps_cache(online_cfg, monkeypatch):
    short = _short_range(online_cfg, "2022-08-01", "2022-08-10")
    good = FakeOpenMeteo()
    monkeypatch.setattr(weather, "get_json", good)
    first = load_or_fetch_weather(short)
    load_or_fetch_weather(short, force=True)
    assert len(good.calls) == 2
    monkeypatch.setattr(weather, "get_json", FakeOpenMeteo(fail_on_call=1))
    kept = load_or_fetch_weather(short, force=True)
    pd.testing.assert_frame_equal(kept, first)


@pytest.mark.integration
def test_load_or_fetch_ignores_corrupt_cache(cfg, warn_log):
    short = _short_range(cfg, "2022-08-01", "2022-08-05")
    path = resolve_path(short, "weather_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("garbage\n\x00\x01")
    out = load_or_fetch_weather(short)
    assert len(out) == 5 * 24
    assert "unreadable" in _warnings(warn_log)


@pytest.mark.integration
def test_load_or_fetch_rescales_on_bias_change(online_cfg, fake_api, warn_log):
    short = _short_range(online_cfg, "2022-08-01", "2022-08-10")
    base = load_or_fetch_weather(short)
    n_calls = len(fake_api.calls)
    doubled = load_or_fetch_weather(deep_merge(short, {"weather": {"bias_correction_factor": 2.0}}))
    assert len(fake_api.calls) == n_calls
    np.testing.assert_allclose(doubled["precipitation_mm"], 2 * base["precipitation_mm"])
    assert "bias" in _warnings(warn_log)
    moved = deep_merge(short, {"weather": {"latitude": 13.1, "longitude": 77.5}})
    load_or_fetch_weather(moved)
    assert len(fake_api.calls) > n_calls, "a cache for another location is refetched"


@pytest.mark.integration
def test_foreign_cache_is_read_only_offline(online_cfg, fake_api, cfg, warn_log):
    """Offline, a cache built for another location is used but never relabelled or overwritten."""
    short = _short_range(online_cfg, "2022-08-01", "2022-08-10")
    load_or_fetch_weather(short)
    path = resolve_path(short, "weather_file")
    before = path.read_text()
    elsewhere = deep_merge(_short_range(cfg, "2022-08-01", "2022-08-15"), {"weather": {"latitude": 13.1, "longitude": 77.5}})
    out = load_or_fetch_weather(elsewhere, force=True)
    assert len(out) == 15 * 24
    assert out.loc[:"2022-08-10", "source"].eq("open_meteo").all()
    assert out.loc["2022-08-11":, "source"].eq("synthetic").all()
    assert path.read_text() == before, "a foreign cache must not be overwritten"
    text = _warnings(warn_log)
    assert "read-only" in text and "Not overwriting" in text and "force" in text


@pytest.mark.integration
def test_cache_metadata_and_write_failures_are_tolerated(cfg, warn_log, monkeypatch):
    short = _short_range(cfg, "2022-08-01", "2022-08-03")
    first = load_or_fetch_weather(short)
    meta = resolve_path(short, "weather_file").with_suffix(".meta.json")
    meta.write_text("{not json")
    pd.testing.assert_frame_equal(load_or_fetch_weather(short), first)
    assert "metadata" in _warnings(warn_log)

    def _fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(weather, "save_weather_csv", _fail)
    longer = load_or_fetch_weather(_short_range(cfg, "2022-08-01", "2022-08-05"))
    assert len(longer) == 5 * 24 and "Could not write" in _warnings(warn_log)
    with pytest.raises(ConfigError):
        load_or_fetch_weather(_short_range(cfg, "yesterday-ish", "2022-08-05"))


def _csv_rows(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve_path(cfg, "weather_file"))


def _meta(cfg: dict) -> dict:
    return json.loads(resolve_path(cfg, "weather_file").with_suffix(".meta.json").read_text())


@pytest.mark.integration
def test_disjoint_range_offline_never_shrinks_the_cache(online_cfg, fake_api, cfg, warn_log):
    """R1-01/R5-03: a range disjoint from the cache must not wipe the cached (real) record."""
    load_or_fetch_weather(_short_range(online_cfg, "2022-08-01", "2022-08-10"))  # 240 real hours
    out = load_or_fetch_weather(_short_range(cfg, "2023-03-01", "2023-03-05"))  # offline, disjoint
    assert len(out) == 5 * 24 and set(out["source"]) == {"synthetic"}
    rows = _csv_rows(cfg)
    assert len(rows) == 240 + 120, "the cache must keep every old row and add the new block"
    assert (rows["source"] == "open_meteo").sum() == 240
    assert not rows["is_imputed"].any(), "no zero-filled bridge between the two blocks"
    meta = _meta(cfg)
    assert meta["rows"] == 360 and meta["sources"] == {"open_meteo": 240, "synthetic": 120}
    assert len(meta["blocks"]) == 2
    back = load_or_fetch_weather(_short_range(cfg, "2022-08-01", "2022-08-10"))
    assert set(back["source"]) == {"open_meteo"} and back.index.freqstr == "h"


@pytest.mark.integration
def test_gap_between_blocks_is_fetched_not_read_as_zeros(online_cfg, fake_api):
    """R1-01: hours between two stored blocks are missing (refetched), never trusted 0 mm."""
    load_or_fetch_weather(_short_range(online_cfg, "2022-08-01", "2022-08-10"))
    load_or_fetch_weather(_short_range(online_cfg, "2022-10-01", "2022-10-05"))
    assert len(_csv_rows(online_cfg)) == 240 + 120
    fake_api.calls.clear()
    out = load_or_fetch_weather(_short_range(online_cfg, "2022-08-01", "2022-10-05"))
    assert [(c[1]["start_date"], c[1]["end_date"]) for c in fake_api.calls] == [("2022-08-11", "2022-09-30")]
    assert out["precipitation_mm"].sum() == pytest.approx(2.0 * _even_days(out.index))
    assert len(_csv_rows(online_cfg)) == len(out) and len(_meta(online_cfg)["blocks"]) == 1


def _even_days(index: pd.DatetimeIndex) -> int:
    return int(sum(1 for t in index if t.hour == 17 and t.day % 2 == 0))


def _write_legacy_bridge_cache(cfg: dict) -> Path:
    """A CSV written by the old code: two real blocks joined by a 15-day zero-filled bridge."""
    index = pd.date_range("2022-08-01", "2022-08-25 23:00", freq="h", tz=TZ, name="timestamp")
    rain = np.array(_rain_for(index), dtype=float)
    bridge = (index >= pd.Timestamp("2022-08-06", tz=TZ)) & (index < pd.Timestamp("2022-08-21", tz=TZ))
    rain[bridge] = 0.0
    frame = pd.DataFrame({"precipitation_mm": rain, "is_imputed": bridge, "source": "open_meteo"}, index=index)
    path = resolve_path(cfg, "weather_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_weather_csv(frame, path)
    return path


@pytest.mark.integration
def test_legacy_zero_bridge_is_never_trusted(online_cfg, fake_api, cfg, warn_log):
    """R5-03: zero-filled is_imputed bridge rows are not data, online or offline."""
    _write_legacy_bridge_cache(cfg)
    offline = load_or_fetch_weather(_short_range(cfg, "2022-08-01", "2022-08-25"))
    bridge = offline.loc["2022-08-06":"2022-08-20"]
    assert set(bridge["source"]) == {"synthetic"} and not bridge["is_imputed"].any()
    assert offline.loc[:"2022-08-05", "source"].eq("open_meteo").all()
    assert "treated as missing" in _warnings(warn_log)
    _write_legacy_bridge_cache(cfg)
    fake_api.calls.clear()
    online = load_or_fetch_weather(_short_range(online_cfg, "2022-08-01", "2022-08-25"))
    assert [(c[1]["start_date"], c[1]["end_date"]) for c in fake_api.calls] == [("2022-08-06", "2022-08-20")]
    assert set(online["source"]) == {"open_meteo"} and not online["is_imputed"].any()
    assert online["precipitation_mm"].sum() == pytest.approx(2.0 * _even_days(online.index))


@pytest.mark.unit
def test_shrinking_write_is_refused(cfg, warn_log, monkeypatch):
    """Defence in depth: a merged record that lost cached hours is never written."""
    load_or_fetch_weather(_short_range(cfg, "2022-08-01", "2022-08-03"))
    before = resolve_path(cfg, "weather_file").read_text()
    real_fill = weather._fill_record
    monkeypatch.setattr(weather, "_fill_record", lambda *a, **k: real_fill(*a, **k).loc["2022-08-02":])
    out = load_or_fetch_weather(_short_range(cfg, "2022-08-02", "2022-08-05"))
    assert len(out) == 4 * 24
    assert resolve_path(cfg, "weather_file").read_text() == before
    assert "would be dropped" in _warnings(warn_log)
    assert weather._shrink_problem(None, out, 6) is None


@pytest.mark.integration
def test_mixed_open_meteo_and_synthetic_record_logs_an_error(online_cfg, monkeypatch, warn_log):
    """R1-04: a record mixing reanalysis and synthetic rain is reported at ERROR level."""
    monkeypatch.setattr(weather, "get_json", FakeOpenMeteo(fail_on_call=2))
    short = deep_merge(_short_range(online_cfg, "2022-01-01", "2022-03-31"), {"weather": {"chunk_days": 31}})
    load_or_fetch_weather(short)
    errors = [r.getMessage() for r in warn_log.records if r.levelno >= logging.ERROR]
    assert any("mixes" in m and "synthetic" in m for m in errors), errors


@pytest.mark.unit
def test_read_weather_csv_does_not_reclip_cached_rows(cfg, tmp_path, warn_log):
    """R1-10: cached rows keep the configured max_precip_mm_h, not the 150 mm/h default."""
    high = deep_merge(cfg, {"weather": {"max_precip_mm_h": 300.0}})
    clean = clean_weather(_hourly("2022-09-04", 3, [0.0, 200.0, 5.0]), high)
    path = save_weather_csv(clean, tmp_path / "w.csv")
    warn_log.clear()
    back = read_weather_csv(path, TZ)
    assert back["precipitation_mm"].tolist() == [0.0, 200.0, 5.0]
    assert "clipped" not in _warnings(warn_log)
    assert read_weather_csv(path, TZ, max_precip=150.0)["precipitation_mm"].max() == 150.0
    target = resolve_path(high, "weather_file")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_weather_csv(clean, target)
    record = load_or_fetch_weather(deep_merge(high, {"weather": {"start_date": "2022-09-04", "end_date": "2022-09-04"}}))
    assert record["precipitation_mm"].iloc[:3].tolist() == [0.0, 200.0, 5.0]


@pytest.mark.unit
def test_clients_wrap_network_errors_and_pass_models(online_cfg, monkeypatch):
    def _down(*args, **kwargs):
        raise NetworkUnavailable("connection refused")

    monkeypatch.setattr(weather, "get_json", _down)
    with pytest.raises(WeatherUnavailable, match="connection refused"):
        fetch_forecast(online_cfg)
    with pytest.raises(WeatherUnavailable, match="connection refused"):
        fetch_previous_runs(online_cfg, "2024-10-20", "2024-10-21")
    fake = FakeOpenMeteo()
    monkeypatch.setattr(weather, "get_json", fake)
    with_model = deep_merge(online_cfg, {"weather": {"models": "era5"}})
    fetch_archive(with_model, "2022-09-01", "2022-09-01", 12.93, 77.68)
    assert fake.calls[0][1]["models"] == "era5"
    monkeypatch.setattr(
        weather,
        "get_json",
        lambda *a, **k: {"hourly": {"time": ["2022-09-01T00:00", "2022-09-01T01:00"], "precipitation": ["1.5", "x"]}},
    )
    raw = fetch_archive(online_cfg, "2022-09-01", "2022-09-01", 12.93, 77.68)
    assert raw["precipitation_mm"].iloc[0] == 1.5 and np.isnan(raw["precipitation_mm"].iloc[1])


@pytest.mark.integration
def test_load_or_fetch_clamps_future_end(cfg, warn_log):
    future = _short_range(cfg, "2026-09-01", "2030-01-01")
    out = load_or_fetch_weather(future)
    assert out.index[-1] < pd.Timestamp.now(tz=TZ)
    assert "not yet available" in _warnings(warn_log)
    with pytest.raises(ConfigError):
        load_or_fetch_weather(_short_range(cfg, "2022-09-01", "2022-08-01"))


# --------------------------------------------------------------------------- forecast & previous runs


@pytest.mark.unit
def test_fetch_forecast_marks_future_hours(online_cfg, fake_api):
    now = pd.Timestamp("2026-09-23 10:40", tz=TZ)
    out = fetch_forecast(online_cfg, now=now)
    params = fake_api.calls[0][1]
    assert params["past_days"] == 2 and params["forecast_days"] == 3
    assert "is_forecast" in out.columns and set(out["source"]) == {"forecast"}
    assert not out.loc[:"2026-09-23 10:00+05:30", "is_forecast"].any()
    assert out.loc["2026-09-23 11:00+05:30":, "is_forecast"].all()
    naive = fetch_forecast(online_cfg, lat=12.95, lon=77.7, now=pd.Timestamp("2026-09-23 10:40"))
    assert naive["is_forecast"].sum() == out["is_forecast"].sum()


@pytest.mark.unit
def test_fetch_forecast_failures_raise_weather_unavailable(cfg, online_cfg, monkeypatch):
    with pytest.raises(WeatherUnavailable, match="ffline"):
        fetch_forecast(cfg)
    monkeypatch.setattr(weather, "get_json", lambda *a, **k: {"hourly": {"time": [], "precipitation": []}})
    with pytest.raises(WeatherUnavailable):
        fetch_forecast(online_cfg)
    monkeypatch.setattr(
        weather, "get_json", lambda *a, **k: {"hourly": {"time": ["2026-09-23T00:00"], "precipitation": [None]}}
    )
    with pytest.raises(WeatherUnavailable, match="no precipitation"):
        fetch_forecast(online_cfg)


@pytest.mark.unit
def test_fetch_previous_runs_columns_and_gaps(online_cfg, monkeypatch, warn_log):
    fake = FakeOpenMeteo(null_every=10)
    monkeypatch.setattr(weather, "get_json", fake)
    out = fetch_previous_runs(online_cfg, "2024-10-20", "2024-10-21")
    assert list(out.columns) == ["precip_lead_0", "precip_lead_24h", "precip_lead_48h", "is_imputed"]
    assert fake.calls[0][1]["hourly"] == "precipitation,precipitation_previous_day1,precipitation_previous_day2"
    assert len(out) == 48 and out.index.freqstr == "h" and out["is_imputed"].sum() == 5
    assert out.loc["2024-10-20 17:00+05:30", "precip_lead_0"] == 2.0
    assert out.loc["2024-10-20 17:00+05:30", "precip_lead_24h"] == 1.0
    assert not out.isna().any().any()
    assert "missing" in _warnings(warn_log)
    single = fetch_previous_runs(online_cfg, "2024-10-20", "2024-10-20", lead_days=[3])
    assert list(single.columns) == ["precip_lead_0", "precip_lead_72h", "is_imputed"]


@pytest.mark.unit
def test_fetch_previous_runs_failures(cfg, online_cfg, monkeypatch):
    with pytest.raises(WeatherUnavailable):
        fetch_previous_runs(cfg, "2024-10-20", "2024-10-21")
    for bad in ((), (0,), (9,), ("x",)):
        with pytest.raises(ValueError, match="lead_days"):
            fetch_previous_runs(online_cfg, "2024-10-20", "2024-10-21", lead_days=bad)
    monkeypatch.setattr(weather, "get_json", FakeOpenMeteo(null_every=1))
    with pytest.raises(WeatherUnavailable, match="2024"):
        fetch_previous_runs(online_cfg, "2020-10-20", "2020-10-21")


# --------------------------------------------------------------------------- areal series & summary


@pytest.mark.unit
def test_areal_series(cfg, warn_log):
    frame = clean_weather(_hourly("2022-09-04", 4, [1.0, 2.0, 3.0, 4.0]), cfg)
    values, index = areal_series(frame)
    assert values.dtype == np.float64 and values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert index.equals(frame.index)
    values[0] = 99.0
    assert frame["precipitation_mm"].iloc[0] == 1.0, "returns a copy"
    dirty = frame.copy()
    dirty["precipitation_mm"] = [np.nan, -1.0, 3.0, 4.0]
    values, _ = areal_series(dirty)
    assert values.tolist() == [0.0, 0.0, 3.0, 4.0] and "areal" in _warnings(warn_log)
    with pytest.raises(ValueError, match="precipitation_mm"):
        areal_series(frame.drop(columns="precipitation_mm"))
    with pytest.raises(TypeError):
        areal_series(pd.DataFrame({"precipitation_mm": [1.0]}))


@pytest.mark.unit
def test_summarize_weather(cfg):
    frame = clean_weather(_hourly("2021-12-31 20:00", 30, np.r_[np.ones(4), np.zeros(20), [0, 0, 9.0, 0, 0, 0]]), cfg)
    summary = summarize_weather(frame)
    assert summary.rows == 30 and summary.total_mm == pytest.approx(13.0)
    assert summary.annual_mm == {2021: pytest.approx(4.0), 2022: pytest.approx(9.0)}
    assert summary.wettest_hour == (pd.Timestamp("2022-01-01 22:00", tz=TZ), 9.0)
    assert summary.wettest_days[0] == ("2022-01-01", 9.0)
    assert summary.pct_imputed == 0.0 and summary.sources == {"open_meteo": 30}
    lines = "\n".join(summary.format_lines())
    assert "2022" in lines and "wettest" in lines.lower()
    empty = summarize_weather(frame.iloc[:0])
    assert empty.rows == 0 and empty.wettest_hour is None
    assert "empty" in empty.format_lines()[0]


# --------------------------------------------------------------------------- CLI


def _load_cli():
    spec = importlib.util.spec_from_file_location("weather_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(cfg: dict, tmp_path: Path) -> Path:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(clean))
    return path


@pytest.mark.e2e
def test_cli_offline_run(cfg, tmp_path, capsys):
    cli = _load_cli()
    config_path = _write_config(cfg, tmp_path)
    code = cli.main(["--config", str(config_path), "--offline", "--start", "2022-06-01", "--end", "2022-07-31"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "rows" in printed and "2022" in printed and "synthetic" in printed
    assert resolve_path(cfg, "weather_file").exists()


@pytest.mark.e2e
def test_cli_handled_failures(cfg, tmp_path, capsys):
    cli = _load_cli()
    config_path = _write_config(cfg, tmp_path)
    assert cli.main(["--config", str(config_path), "--start", "2022-08-01", "--end", "2022-07-01"]) == 1
    assert "error" in capsys.readouterr().err.lower()
    assert cli.main(["--config", str(tmp_path / "nope.yaml")]) == 1


# --------------------------------------------------------------------------- live API (opt-in)


@pytest.mark.network
def test_live_archive_forecast_and_previous_runs(cfg, monkeypatch):
    monkeypatch.delenv("NAMMA_FLOW_OFFLINE", raising=False)
    live = deep_merge(cfg, {"project": {"offline": False}})
    arch = fetch_archive(live, "2022-09-04", "2022-09-05", 12.9325, 77.6775)
    assert len(arch) == 48
    assert len(fetch_forecast(live)) >= 24
    prev = fetch_previous_runs(live, "2024-10-20", "2024-10-21")
    assert len(prev) == 48
