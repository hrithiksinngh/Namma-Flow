"""Weather-schema side of F1-04 (bridge hours are ``source="missing"``, never the mode) and
read-only weather loading for diagnostics runs (F5-08, ``persist=False``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.weather import (
    SOURCE_MISSING,
    clean_weather,
    load_or_fetch_weather,
    read_weather_csv,
    read_weather_rows,
    save_weather_csv,
    summarize_weather,
)
from src.data_pipeline.weather_schema import contiguous_blocks, imputed_runs
from src.utils.config import deep_merge, resolve_path
from tests.conftest import TZ


def _block(start: str, hours: int, source: str = "open_meteo", rain: float = 1.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h", tz=TZ, name="timestamp")
    return pd.DataFrame({"precipitation_mm": rain, "is_imputed": False, "source": source}, index=index)


def _two_block_csv(tmp_path) -> tuple[str, pd.DataFrame]:
    """A multi-block cache (a supported state): 3 days of 2024, then 2 days of 2025 (months apart)."""
    frame = pd.concat([_block("2024-12-29", 72), _block("2025-03-01", 48, rain=2.0)])
    path = tmp_path / "weather.csv"
    save_weather_csv(frame, path)
    return str(path), frame


@pytest.mark.unit
def test_bridged_hours_are_labelled_missing_not_the_stored_source(tmp_path):
    path, stored = _two_block_csv(tmp_path)
    bridged = read_weather_csv(path, TZ)
    gap = ~bridged.index.isin(stored.index)
    assert gap.sum() > 1000 and bridged.index.freqstr == "h"
    assert set(bridged.loc[gap, "source"]) == {SOURCE_MISSING}
    assert bridged.loc[gap, "is_imputed"].all() and (bridged.loc[gap, "precipitation_mm"] == 0).all()
    assert set(bridged.loc[~gap, "source"]) == {"open_meteo"} and not bridged.loc[~gap, "is_imputed"].any()
    assert summarize_weather(bridged).sources[SOURCE_MISSING] == int(gap.sum())
    assert imputed_runs(bridged, 6)[gap].all()


@pytest.mark.unit
def test_read_weather_rows_never_bridges(tmp_path):
    path, stored = _two_block_csv(tmp_path)
    rows = read_weather_rows(path, TZ)
    assert len(rows) == len(stored) and SOURCE_MISSING not in set(rows["source"])
    assert len(contiguous_blocks(pd.DatetimeIndex(rows.index))) == 2


@pytest.mark.unit
def test_clean_weather_gap_and_range_extension_hours_are_missing(cfg):
    frame = pd.concat([_block("2022-09-04 00:00", 5, source="synthetic"), _block("2022-09-04 08:00", 4,
                                                                                 source="synthetic")])
    out = clean_weather(frame, cfg, start="2022-09-04", end="2022-09-04")
    assert len(out) == 24
    stored = out.index.isin(frame.index)
    assert set(out.loc[stored, "source"]) == {"synthetic"}
    assert set(out.loc[~stored, "source"]) == {SOURCE_MISSING}, "neither the mode nor the default source"
    no_source = clean_weather(frame[["precipitation_mm"]], cfg)
    assert set(no_source.loc[~no_source.index.isin(frame.index), "source"]) == {SOURCE_MISSING}
    assert set(no_source.loc[no_source.index.isin(frame.index), "source"]) == {"open_meteo"}


@pytest.mark.unit
def test_stored_rows_without_a_source_still_get_the_stored_mode(cfg):
    frame = _block("2022-09-04", 6, source="synthetic")
    frame.loc[frame.index[2], "source"] = np.nan
    out = clean_weather(frame, cfg)
    assert list(out["source"]) == ["synthetic"] * 6


@pytest.mark.integration
def test_persist_false_never_writes_the_weather_cache(cfg):
    short = deep_merge(cfg, {"weather": {"start_date": "2022-08-01", "end_date": "2022-08-03"}})
    path = resolve_path(short, "weather_file")
    record = load_or_fetch_weather(short, persist=False)  # offline: synthetic climatology in memory
    assert len(record) == 72 and set(record["source"]) == {"synthetic"}
    assert not path.exists() and not path.with_suffix(".meta.json").exists()
    load_or_fetch_weather(short)
    before = path.read_bytes()
    longer = deep_merge(short, {"weather": {"end_date": "2022-08-05"}})
    assert len(load_or_fetch_weather(longer, persist=False)) == 120
    assert path.read_bytes() == before, "an existing cache is never extended by a persist=False run"
