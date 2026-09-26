"""Regression tests for replays of a multi-block / gap-filled weather record (finding F1-04, replay side).

A record may hold several disjoint blocks (e.g. 2018-2024 plus one later month). Replays read the
stored rows only (``weather_schema.read_weather_rows``): a start in a gap is refused instead of
replaying months of fabricated 0 mm "open_meteo" rain, gap-filled rows (long ``is_imputed`` runs,
``source == missing``) are refused / noted, and the notable-event list marks synthetic and
gap-filled spans and never spans two blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import weather
from src.inference import record as rec
from src.inference import scenarios as sc
from tests.test_inference import TZ, icfg  # noqa: F401 - fixtures are used by name


def _block(start: str, end: str, source: str = "open_meteo", storm_at: str | None = None,
           storm: tuple[float, ...] = (4.0, 12.0, 6.0)) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="h", tz=TZ, name="timestamp")
    rain = pd.Series(0.0, index=index)
    if storm_at:
        at = index.get_loc(pd.Timestamp(storm_at, tz=TZ))
        rain.iloc[at: at + len(storm)] = storm
    return pd.DataFrame({"precipitation_mm": rain.to_numpy(), "is_imputed": False, "source": source}, index=index)


def two_blocks() -> pd.DataFrame:
    """2022-09-01..09-20 (observed) and 2022-11-01..11-10 (synthetic), nothing stored in between."""
    return pd.concat([_block("2022-09-01 00:00", "2022-09-20 23:00", storm_at="2022-09-05 14:00"),
                      _block("2022-11-01 00:00", "2022-11-10 23:00", "synthetic", storm_at="2022-11-04 03:00")])


def bridged() -> pd.DataFrame:
    """A legacy cache: the months between two blocks were written as 0 mm 'open_meteo' imputed rows."""
    frame = pd.concat([_block("2022-09-01 00:00", "2022-09-10 23:00", storm_at="2022-09-05 14:00"),
                       _block("2022-09-11 00:00", "2022-09-30 23:00"),
                       _block("2022-10-01 00:00", "2022-10-10 23:00", storm_at="2022-10-04 03:00")])
    gap = (frame.index >= pd.Timestamp("2022-09-11", tz=TZ)) & (frame.index < pd.Timestamp("2022-10-01", tz=TZ))
    frame.loc[gap, "is_imputed"] = True
    return frame


@pytest.fixture
def blocks_cfg(icfg):
    weather.save_weather_csv(two_blocks(), icfg["paths"]["weather_file"])
    return icfg


@pytest.fixture
def bridged_cfg(icfg):
    weather.save_weather_csv(bridged(), icfg["paths"]["weather_file"])
    return icfg


@pytest.mark.unit
def test_record_keeps_stored_blocks_without_bridging(blocks_cfg):
    record = sc.load_weather_record(blocks_cfg)
    assert len(record) == 20 * 24 + 10 * 24                          # no zero-filled bridge rows
    spans = sc.block_spans(record)
    assert [(lo.date().isoformat(), hi.date().isoformat()) for lo, hi in spans] == [
        ("2022-09-01", "2022-09-20"), ("2022-11-01", "2022-11-10")]


@pytest.mark.unit
def test_replay_start_in_a_gap_is_refused(blocks_cfg):
    with pytest.raises(sc.ScenarioError, match="falls in a gap of the weather record") as err:
        sc.historical_scenario(blocks_cfg, "2022-10-10 00:00", hours=24)
    assert "2022-09-01 00:00 -> 2022-09-20 23:00" in str(err.value)
    with pytest.raises(sc.ScenarioError, match="outside the weather record"):
        sc.historical_scenario(blocks_cfg, "2023-01-01 00:00", hours=24)


@pytest.mark.unit
def test_replay_inside_a_later_block_is_clipped_and_noted(blocks_cfg):
    late = sc.historical_scenario(blocks_cfg, "2022-11-01 10:00", hours=24)
    assert late.timestamps[0] == pd.Timestamp("2022-11-01 00:00", tz=TZ)      # history clipped at the block start
    assert late.source == "synthetic" and late.exact_field
    assert any("hours before it are missing" in note for note in late.notes)
    end = sc.historical_scenario(blocks_cfg, "2022-09-20 12:00", hours=48)
    assert end.n_target_hours == 12
    assert any("stored weather block ends 2022-09-20 23:00" in note for note in end.notes)
    last = sc.historical_scenario(blocks_cfg, "2022-11-10 12:00", hours=48)    # the record end: warning only
    assert last.n_target_hours == 12 and not any("block ends" in note for note in last.notes)


@pytest.mark.unit
def test_gap_filled_rows_are_refused_and_noted(bridged_cfg):
    record = sc.load_weather_record(bridged_cfg)
    mask = sc.gap_filled_mask(record, bridged_cfg)
    assert mask.sum() == 20 * 24 and not mask[: 10 * 24].any()
    with pytest.raises(sc.ScenarioError, match="gap-filled hours"):
        sc.historical_scenario(bridged_cfg, "2022-09-20 00:00", hours=24)
    touching = sc.historical_scenario(bridged_cfg, "2022-09-10 12:00", hours=24)
    assert touching.source == "open_meteo"
    assert any("gap-filled hours of the weather record" in note and "12 of them predicted" in note
               for note in touching.notes)
    clean = sc.historical_scenario(bridged_cfg, "2022-09-05 12:00", hours=24)
    assert not any("gap-filled" in note for note in clean.notes)


@pytest.mark.unit
def test_missing_source_rows_count_as_gap_filled(icfg):
    frame = _block("2022-09-01 00:00", "2022-09-10 23:00", storm_at="2022-09-05 14:00")
    frame.loc[frame.index >= pd.Timestamp("2022-09-08", tz=TZ), "source"] = "missing"   # DATA's F1-04 bridge mark
    weather.save_weather_csv(frame, icfg["paths"]["weather_file"])
    record = sc.load_weather_record(icfg)
    assert sc.gap_filled_mask(record, icfg).sum() == 3 * 24
    with pytest.raises(sc.ScenarioError, match="gap-filled"):
        sc.historical_scenario(icfg, "2022-09-09 00:00", hours=12)
    assert rec.dominant_source(record, sc.gap_filled_mask(record, icfg)) == "open_meteo"


@pytest.mark.unit
def test_notable_events_mark_synthetic_spans_and_stay_inside_blocks(blocks_cfg):
    events = sc.list_notable_events(blocks_cfg, top_k=5)
    assert list(events.columns) == list(sc.NOTABLE_COLUMNS)
    by_source = dict(zip(events["source"], events["label"]))
    assert "synthetic record" in by_source["synthetic"] and "synthetic" not in by_source["open_meteo"]
    record = sc.load_weather_record(blocks_cfg)
    for start, end in zip(events["start"], events["end"]):         # never spanning two stored blocks
        assert (end - start) == pd.Timedelta(hours=23) and end in record.index and start in record.index


@pytest.mark.unit
def test_notable_events_skip_bridged_gaps(bridged_cfg):
    events = sc.list_notable_events(bridged_cfg, top_k=5)
    assert len(events) == 2 and (events["gap_filled_hours"] == 0).all()   # 0 mm gap rows are never "events"


@pytest.mark.unit
def test_notable_event_touching_gap_filled_hours_is_labelled(icfg):
    frame = _block("2022-09-01 00:00", "2022-09-10 23:00", storm_at="2022-09-05 20:00")
    frame.loc[(frame.index >= pd.Timestamp("2022-09-05 06:00", tz=TZ))
              & (frame.index < pd.Timestamp("2022-09-05 20:00", tz=TZ)), "is_imputed"] = True
    weather.save_weather_csv(frame, icfg["paths"]["weather_file"])
    events = sc.list_notable_events(icfg, top_k=1)
    assert events.loc[0, "gap_filled_hours"] == 14 and "includes 14 gap-filled h" in events.loc[0, "label"]


@pytest.mark.unit
def test_short_blocks_and_empty_records(icfg):
    frame = pd.concat([_block("2022-09-01 00:00", "2022-09-01 05:00", storm_at="2022-09-01 01:00"),
                       _block("2022-09-03 00:00", "2022-09-03 05:00")])
    weather.save_weather_csv(frame, icfg["paths"]["weather_file"])
    assert sc.list_notable_events(icfg).empty                         # no block is a full 24 h window
    assert rec.dominant_source(frame.iloc[:0], np.zeros(0, dtype=bool)) == "open_meteo"
