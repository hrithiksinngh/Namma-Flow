"""Tests for observed flood reports and their fusion with simulated labels (``src.hydrology.observed``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.graph_io import graph_to_arrays
from src.hydrology.observed import (
    REPORT_COLUMNS,
    FloodReportError,
    LabelSettings,
    empty_reports,
    load_flood_reports,
    merge_observed_labels,
)
from src.utils.config import ConfigError, deep_merge
from tests.conftest import TZ
from tests.test_hydrology import _warnings, grid_arrays, warn_log  # noqa: F401 - shared fixtures


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.unit
def test_load_missing_file_returns_empty_frame(tmp_path):
    frame = load_flood_reports(tmp_path / "nope.csv", TZ)
    assert frame.empty and list(frame.columns) == list(REPORT_COLUMNS)
    assert str(frame["timestamp"].dt.tz) == TZ


@pytest.mark.unit
def test_load_header_only_template(tmp_path):
    path = _write(tmp_path / "r.csv", ",".join(REPORT_COLUMNS) + "\n")
    frame = load_flood_reports(path, TZ)
    assert frame.empty and list(frame.columns) == list(REPORT_COLUMNS)


@pytest.mark.unit
def test_load_zero_byte_file_warns(tmp_path, warn_log):
    path = _write(tmp_path / "r.csv", "")
    assert load_flood_reports(path, TZ).empty
    assert "empty" in _warnings(warn_log)


@pytest.mark.unit
def test_load_valid_rows_and_timezones(tmp_path):
    text = (
        "timestamp,lat,lon,radius_m,severity,description\n"
        "2022-09-05 14:00,12.93,77.68,200,severe,ORR underpass\n"
        "2022-09-05T08:30:00+00:00,12.931,77.681,,2,\n"
        "2022-09-04 22:00,12.94,77.69,,,\n"
    )
    frame = load_flood_reports(_write(tmp_path / "r.csv", text), TZ)
    assert len(frame) == 3
    assert frame["timestamp"].is_monotonic_increasing
    assert str(frame["timestamp"].dt.tz) == TZ
    by_desc = frame.set_index("description")
    assert by_desc.loc["ORR underpass", "severity"] == 3
    assert by_desc.loc["ORR underpass", "radius_m"] == 200.0
    utc_row = frame[frame["lat"] == 12.931].iloc[0]
    assert utc_row["timestamp"] == pd.Timestamp("2022-09-05 14:00", tz=TZ)  # 08:30 UTC == 14:00 IST
    assert np.isnan(utc_row["radius_m"]) and utc_row["severity"] == 2
    assert frame[frame["lat"] == 12.94].iloc[0]["severity"] == 1  # default severity


@pytest.mark.unit
def test_load_drops_bad_rows_with_warning(tmp_path, warn_log):
    text = (
        "Timestamp,Latitude,Longitude,radius_m,severity\n"
        "2022-09-05 14:00,12.93,77.68,,\n"
        "not-a-date,12.93,77.68,,\n"
        "2022-09-05 15:00,95.0,77.68,,\n"
        "2022-09-05 15:00,12.93,abc,,\n"
        "2022-09-05 16:00,12.93,77.68,-10,\n"
        "2022-09-05 17:00,12.93,77.68,,catastrophic\n"
        "2022-09-05 14:00,12.93,77.68,,\n"
        ",12.93,77.68,,\n"
    )
    frame = load_flood_reports(_write(tmp_path / "r.csv", text), TZ)
    assert len(frame) == 1
    text = _warnings(warn_log)
    assert "dropped 6" in text.lower()


@pytest.mark.unit
def test_load_missing_required_column_raises(tmp_path):
    path = _write(tmp_path / "r.csv", "timestamp,lat\n2022-09-05,12.9\n")
    with pytest.raises(FloodReportError, match="lon"):
        load_flood_reports(path, TZ)


@pytest.mark.unit
def test_load_directory_or_bad_tz_raises(tmp_path):
    (tmp_path / "dir.csv").mkdir()
    with pytest.raises(FloodReportError):
        load_flood_reports(tmp_path / "dir.csv", TZ)
    with pytest.raises(ValueError):
        load_flood_reports(tmp_path / "missing.csv", "Mars/Olympus")


@pytest.mark.unit
def test_load_undecodable_file_raises(tmp_path):
    path = tmp_path / "bin.csv"
    path.write_bytes(b"timestamp,lat,lon\n\xff\xfe\x00\x81,1,2\n")
    with pytest.raises(FloodReportError):
        load_flood_reports(path, TZ)


@pytest.mark.unit
def test_repository_template_is_header_only():
    template = Path(__file__).resolve().parents[1] / "data/raw/labels/flood_reports.csv"
    assert template.read_text(encoding="utf-8").strip() == ",".join(REPORT_COLUMNS)
    assert load_flood_reports(template, TZ).empty


def _reports(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "lat", "lon", "radius_m", "severity"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(TZ)
    frame["description"] = ""
    return frame


@pytest.fixture
def merge_setup(grid_arrays, hourly_index):
    labels = np.zeros((len(hourly_index), grid_arrays.num_nodes), dtype=np.uint8)
    node = 7
    return labels, hourly_index, grid_arrays, node


@pytest.mark.unit
def test_merge_sets_window_around_report(cfg, merge_setup):
    labels, stamps, arrays, node = merge_setup
    ts = stamps[40]
    reports = _reports([(ts.tz_localize(None), arrays.lat[node], arrays.lon[node], np.nan, 1)])
    cfg = deep_merge(cfg, {"labels": {"source": "hybrid", "report_window_hours": 3, "report_snap_radius_m": 50.0}})
    out = merge_observed_labels(labels, stamps, arrays, reports, cfg)
    assert out.dtype == np.uint8 and out.shape == labels.shape
    flagged_hours = np.flatnonzero(out[:, node])
    assert flagged_hours.tolist() == list(range(37, 44))
    assert out.sum() == 7  # only the snapped junction (grid spacing >> 50 m)
    assert labels.sum() == 0  # input not mutated


@pytest.mark.unit
def test_merge_radius_covers_neighbours(cfg, merge_setup):
    labels, stamps, arrays, node = merge_setup
    ts = stamps[10].tz_localize(None)
    reports = _reports([(ts, arrays.lat[node], arrays.lon[node], 2000.0, 2)])
    cfg = deep_merge(cfg, {"labels": {"source": "hybrid", "report_window_hours": 0}})
    out = merge_observed_labels(labels, stamps, arrays, reports, cfg)
    assert out[10].sum() > 1 and out.sum() == out[10].sum()


@pytest.mark.unit
def test_merge_observed_mode_replaces_simulated(cfg, merge_setup):
    labels, stamps, arrays, node = merge_setup
    sim = labels.copy()
    sim[0:5, :] = 1
    reports = _reports([(stamps[50].tz_localize(None), arrays.lat[node], arrays.lon[node], np.nan, 3)])
    observed = merge_observed_labels(sim, stamps, arrays, reports, deep_merge(cfg, {"labels": {"source": "observed"}}))
    assert observed[0:5].sum() == 0 and observed[:, node].sum() > 0
    hybrid = merge_observed_labels(sim, stamps, arrays, reports, cfg, mode="hybrid")
    assert hybrid[0:5].all() and hybrid[:, node].sum() > 5
    simulated_cfg = deep_merge(cfg, {"labels": {"source": "simulated"}})
    simulated = merge_observed_labels(sim, stamps, arrays, reports, simulated_cfg)
    np.testing.assert_array_equal(simulated, sim)


@pytest.mark.unit
def test_merge_ignores_far_out_of_range_and_minor_reports(cfg, merge_setup, warn_log):
    labels, stamps, arrays, node = merge_setup
    rows = [
        (stamps[10].tz_localize(None), 13.5, 77.0, np.nan, 3),  # far away from every junction
        (pd.Timestamp("2019-01-01 10:00"), arrays.lat[node], arrays.lon[node], np.nan, 3),  # outside the record
        (stamps[20].tz_localize(None), arrays.lat[node], arrays.lon[node], np.nan, 1),  # below min severity
    ]
    cfg = deep_merge(cfg, {"labels": {"source": "hybrid", "min_report_severity": 2}})
    out = merge_observed_labels(labels, stamps, arrays, _reports(rows), cfg)
    assert out.sum() == 0
    text = _warnings(warn_log)
    assert "could not be snapped" in text and "outside" in text


@pytest.mark.unit
def test_merge_empty_reports_and_naive_timestamps(cfg, merge_setup):
    labels, stamps, arrays, _ = merge_setup
    labels = labels.copy()
    labels[3, 3] = 1
    cfg = deep_merge(cfg, {"labels": {"source": "hybrid"}})
    out = merge_observed_labels(labels, stamps.tz_localize(None), arrays, empty_reports(TZ), cfg)
    np.testing.assert_array_equal(out, labels)
    assert out is not labels
    assert merge_observed_labels(labels, stamps, arrays, None, cfg).sum() == 1


@pytest.mark.unit
def test_merge_validates_inputs(cfg, merge_setup):
    labels, stamps, arrays, _ = merge_setup
    reports = empty_reports(TZ)
    with pytest.raises(ValueError, match="labels"):
        merge_observed_labels(labels[:, :5], stamps, arrays, reports, cfg)
    with pytest.raises(ValueError, match="timestamps"):
        merge_observed_labels(labels, stamps[:-1], arrays, reports, cfg)
    with pytest.raises(ValueError, match="0/1"):
        merge_observed_labels(labels + 2, stamps, arrays, reports, cfg)
    with pytest.raises(ValueError, match="increasing"):
        merge_observed_labels(labels, stamps[::-1], arrays, reports, cfg)
    with pytest.raises(ValueError, match="columns"):
        merge_observed_labels(labels, stamps, arrays, pd.DataFrame({"a": [1]}), cfg)
    with pytest.raises(ValueError, match="mode"):
        merge_observed_labels(labels, stamps, arrays, reports, cfg, mode="guess")


@pytest.mark.unit
def test_label_settings_validation():
    assert LabelSettings.from_config({}).source == "simulated"
    for bad in ({"source": "crowd"}, {"report_snap_radius_m": 0}, {"report_window_hours": -1},
                {"min_report_severity": 5}, {"report_snap_radius_m": "far"}):
        with pytest.raises(ConfigError):
            LabelSettings.from_config({"labels": bad})


@pytest.mark.integration
def test_load_then_merge_round_trip(cfg, merge_setup, tmp_path):
    labels, stamps, arrays, node = merge_setup
    text = "timestamp,lat,lon,radius_m,severity,description\n" + (
        f"{stamps[30]:%Y-%m-%d %H:%M},{arrays.lat[node]},{arrays.lon[node]},,2,test\n"
    )
    reports = load_flood_reports(_write(tmp_path / "r.csv", text), TZ)
    out = merge_observed_labels(labels, stamps, arrays, reports, cfg, mode="observed")
    assert out[30, node] == 1


@pytest.mark.unit
def test_load_severity_forms_and_duplicates(tmp_path, warn_log):
    text = (
        "timestamp,lat,lon,radius_m,severity,description\n"
        "2022-09-05 10:00,12.93,77.68,,2.0,a\n"
        "2022-09-05 10:00,12.93,77.68,,2.0,duplicate of a\n"
        "2022-09-05 11:00,12.93,77.68,,Knee,b\n"
        "2022-09-05 12:00,12.93,77.68,,0,zero is not a flood severity\n"
    )
    frame = load_flood_reports(_write(tmp_path / "r.csv", text), TZ)
    assert frame["severity"].tolist() == [2, 2]
    assert frame["description"].tolist() == ["a", "b"]
    assert "dropped 1" in _warnings(warn_log).lower()


@pytest.mark.unit
def test_merge_accepts_networkx_graph_and_rejects_bad_types(cfg, grid_graph, hourly_index):
    arrays = graph_to_arrays(grid_graph)
    labels = np.zeros((len(hourly_index), arrays.num_nodes), dtype=np.uint8)
    reports = _reports([(hourly_index[5].tz_localize(None), arrays.lat[0], arrays.lon[0], np.nan, 2)])
    out = merge_observed_labels(labels, hourly_index, grid_graph, reports, cfg, mode="hybrid")
    assert out[5, 0] == 1
    with pytest.raises(TypeError):
        merge_observed_labels(labels, hourly_index, grid_graph, [1, 2, 3], cfg, mode="hybrid")
    with pytest.raises(TypeError):
        merge_observed_labels(labels, hourly_index, "graph", reports, cfg, mode="hybrid")
    bad = labels.astype(float)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="0/1"):
        merge_observed_labels(bad, hourly_index, grid_graph, reports, cfg, mode="hybrid")


@pytest.mark.unit
def test_label_settings_reject_non_finite():
    with pytest.raises(ConfigError):
        LabelSettings.from_config({"labels": {"report_window_hours": float("inf")}})
    settings = LabelSettings.from_config({"labels": {"source": " Hybrid ", "report_window_hours": 1.5}})
    assert settings.source == "hybrid" and settings.report_window_hours == 1.5
