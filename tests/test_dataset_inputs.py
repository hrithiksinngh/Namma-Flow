"""Stage-04 input provenance, reuse, the held-out test split and safe payload loading.

Regression tests for the adversarial-review findings owned by the dataset module:

* R1-02 / R5-02 - a re-enriched graph (same topology, new elevation / drains / grades) must
  rebuild the datasets (graph attribute digest, X1);
* R2-06 / R5-04 - a changed weather record or flood-reports file must rebuild them (X2);
* R2-08 - val/test windows must not score the same hour twice;
* R3-03 - :func:`load_payload` must never fall back to the full unpickler;
* X2 - the train / val / test split, fingerprints in every payload, format-1 compatibility.
"""

from __future__ import annotations

import pathlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import provenance
from src.data_pipeline.dataset import (
    FORMAT_VERSION,
    V1_PAYLOAD_KEYS,
    DatasetError,
    DatasetSettings,
    FloodSequenceDataset,
    build_datasets,
    dataset_paths,
    load_payload,
    select_windows,
)
from src.data_pipeline.features import FeatureScaler
from src.data_pipeline.graph_io import graph_to_arrays, load_graph, save_graph
from src.data_pipeline.weather import load_or_fetch_weather, save_weather_csv
from src.utils.config import ConfigError, deep_merge, project_root
from tests.conftest import TZ, make_grid_graph
from tests.dataset_fixtures import fake_simulator, log  # noqa: F401 - pytest fixtures
from tests.dataset_fixtures import END, START, setup_config, split_years, storm_record

STATIC = ("elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink")
EDGES = ("length", "grade")


@pytest.fixture
def dcfg(cfg):
    return setup_config(cfg)


@pytest.fixture
def built(dcfg, fake_simulator):
    return dcfg, build_datasets(dcfg)


def _paths(cfg: dict) -> dict[str, Path]:
    return {split: Path(cfg["paths"][f"{split}_dataset"]) for split in ("train", "val", "test")}


def _payloads(cfg: dict) -> dict[str, dict]:
    return {split: load_payload(path) for split, path in _paths(cfg).items() if path.exists()}


def _index(start: str, hours: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=hours, freq="h", tz=TZ)


def _settings(cfg: dict, **dataset) -> DatasetSettings:
    return DatasetSettings.from_config(deep_merge(cfg, {"dataset": dataset}))


# --------------------------------------------------------------------------- X2: train / val / test split


@pytest.mark.unit
def test_select_windows_three_way_split_by_last_hour(cfg):
    ts = _index("2022-12-20", 24 * 25)
    s = _settings(cfg, val_years=[2022], test_years=[2023], season_months=list(range(1, 13)), wet_window_min_mm=0.0,
                  train_stride_h=1, val_stride_h=1)
    sel = select_windows(ts, np.zeros(len(ts)), s)
    assert sel.train.size == 0
    assert (ts[sel.val].year == 2022).all() and (ts[sel.val + s.seq_len - 1].year == 2022).all()
    assert (ts[sel.test].year == 2023).all() and sel.test.size > 0
    assert sel.n_straddling == s.seq_len - 1  # windows starting 2022-12-31 09:00 .. 23:00 end in 2023
    assert sel.counts()["n_test"] == sel.test.size
    assert set(sel.val).isdisjoint(sel.test)


@pytest.mark.unit
def test_test_windows_use_the_val_stride(cfg):
    ts = _index("2023-06-01", 24 * 20)
    s = _settings(cfg, val_years=[2022], test_years=[2023], wet_window_min_mm=0.0, train_stride_h=6, val_stride_h=12)
    sel = select_windows(ts, np.zeros(len(ts)), s)
    assert sel.test.size > 0 and set(ts[sel.test].hour) == {0, 12}
    assert np.all(np.diff(sel.test) == 12)
    with pytest.raises(KeyError, match="unknown split"):
        sel.starts("holdout")


@pytest.mark.unit
def test_settings_allow_an_empty_test_split(cfg):
    s = _settings(cfg, test_years=[])
    assert s.test_years == () and not s.has_test_split and s.splits == ("train", "val")
    assert _settings(cfg, test_years=None).test_years == ()
    with pytest.raises(ConfigError, match="overlap"):
        _settings(cfg, val_years=[2022, 2024], test_years=[2024])


@pytest.mark.integration
def test_build_writes_train_val_and_test_from_one_build(built):
    cfg, summary = built
    payloads = _payloads(cfg)
    assert set(payloads) == {"train", "val", "test"} and set(summary["splits"]) == {"train", "val", "test"}
    assert {split: split_years(p) for split, p in payloads.items()} == {"train": {2021}, "val": {2022},
                                                                         "test": {2023}}
    assert len({p["build_id"] for p in payloads.values()}) == 1
    assert all(p["splits"] == ["train", "val", "test"] and p["test_years"] == [2023] for p in payloads.values())
    test = FloodSequenceDataset(_paths(cfg)["test"])
    assert test.split == "test" and len(test) == payloads["test"]["stats"]["n_windows"] > 0
    assert test.scaler == FloodSequenceDataset(_paths(cfg)["train"]).scaler
    assert summary["val_years"] == [2022] and summary["test_years"] == [2023]


@pytest.mark.integration
def test_scaler_is_fitted_on_train_only_not_on_val_or_test(built, fake_simulator):
    cfg, _ = built
    before = _payloads(cfg)
    frame = storm_record(START, END)
    held_out = frame.index.year >= 2022
    frame.loc[held_out, "precipitation_mm"] *= 7.0  # only val/test rain changes
    save_weather_csv(frame, cfg["paths"]["weather_file"])
    assert build_datasets(cfg)["reused"] is False
    after = _payloads(cfg)
    assert FeatureScaler.from_dict(after["train"]["scaler"]) == FeatureScaler.from_dict(before["train"]["scaler"])
    assert not torch.equal(after["test"]["rain"], before["test"]["rain"])


@pytest.mark.integration
def test_empty_test_years_writes_no_test_file_and_removes_a_stale_one(built, fake_simulator, log):
    cfg, _ = built
    test_path = _paths(cfg)["test"]
    assert test_path.exists()
    summary = build_datasets(deep_merge(cfg, {"dataset": {"test_years": []}}))
    assert not test_path.exists() and "test" not in summary["splits"] and summary["test_years"] == []
    assert "removed the stale test dataset" in log.text
    assert load_payload(_paths(cfg)["train"])["splits"] == ["train", "val"]
    again = build_datasets(deep_merge(cfg, {"dataset": {"test_years": []}}))
    assert again["reused"] is True and not test_path.exists()


@pytest.mark.integration
def test_missing_test_file_triggers_a_rebuild(built, fake_simulator, log):
    cfg, _ = built
    _paths(cfg)["test"].unlink()
    summary = build_datasets(cfg)
    assert summary["reused"] is False and _paths(cfg)["test"].exists()
    assert any("no existing test dataset file" in reason for reason in summary["rebuild_reasons"])


@pytest.mark.unit
def test_dataset_paths_keep_the_three_splits_together(cfg, tmp_path):
    explicit = deep_merge(cfg, {"paths": {"test_dataset": str(tmp_path / "elsewhere/test.pt")}})
    assert dataset_paths(explicit)["test"] == tmp_path / "elsewhere/test.pt"
    val = Path(cfg["paths"]["val_dataset"])
    # conftest's cfg relocates train/val but leaves the shipped default test path: follow val, never data/processed
    assert dataset_paths(cfg)["test"] == val.with_name("test_dataset.pt")
    no_key = {**cfg, "paths": {k: v for k, v in cfg["paths"].items() if k != "test_dataset"}}
    assert dataset_paths(no_key)["test"] == val.with_name("test_dataset.pt")
    shipped = deep_merge(cfg, {"paths": {"train_dataset": "data/processed/train_dataset.pt",
                                         "val_dataset": "data/processed/val_dataset.pt",
                                         "test_dataset": "data/processed/test_dataset.pt"}})
    assert dataset_paths(shipped)["test"] == project_root() / "data/processed/test_dataset.pt"


# --------------------------------------------------------------------------- R2-08: val/test scored hours tile


def _scored_hours(payload: dict) -> np.ndarray:
    starts, ts = payload["window_starts"].numpy(), payload["timestamps"].numpy()
    warm, seq = payload["warmup_steps"], payload["seq_len"]
    return np.concatenate([ts[s + warm: s + seq] for s in starts])


@pytest.mark.integration
def test_val_and_test_windows_never_score_an_hour_twice(built):
    cfg, _ = built
    for split in ("val", "test"):
        hours = _scored_hours(load_payload(_paths(cfg)[split]))
        assert hours.size > 0 and np.unique(hours).size == hours.size, f"{split} scores some hours twice"


@pytest.mark.unit
def test_settings_warn_when_val_stride_overlaps_scored_steps(cfg, log):
    _settings(cfg, seq_len=24, warmup_steps=6, val_stride_h=12)
    assert "double-counts" in log.text
    log.clear()
    _settings(cfg)
    assert "double-counts" not in log.text


# --------------------------------------------------------------------------- X1 / X2: input fingerprints


@pytest.mark.integration
def test_payloads_record_the_input_fingerprints(built):
    cfg, summary = built
    arrays = graph_to_arrays(load_graph(cfg["paths"]["graph_file"]))
    frame = load_or_fetch_weather(cfg)
    sources = {str(k): int(v) for k, v in frame["source"].value_counts().items()}
    expected_weather = provenance.weather_fingerprint(frame.index, frame["precipitation_mm"].to_numpy(), sources)
    for payload in _payloads(cfg).values():
        assert payload["format_version"] == FORMAT_VERSION == 2
        assert payload["graph_attributes_sha256"] == arrays.attributes_signature(STATIC, EDGES)
        assert payload["weather_fingerprint"] == expected_weather
        assert payload["reports_fingerprint"] is None  # labels.source = simulated
    assert summary["fingerprints"]["weather_fingerprint"] == expected_weather
    val = FloodSequenceDataset(_paths(cfg)["val"])
    assert val.graph_attributes_sha256 == arrays.attributes_signature(STATIC, EDGES)


@pytest.mark.unit
def test_weather_fingerprint_is_sensitive_to_times_values_and_sources():
    ts = _index("2022-06-01", 48)
    rain = np.linspace(0.0, 4.7, 48)
    base = provenance.weather_fingerprint(ts, rain, {"open_meteo": 48})
    assert base == provenance.weather_fingerprint(ts, rain.copy(), {"open_meteo": 48}) and len(base) == 16
    assert base == provenance.weather_fingerprint(ts.tz_convert("UTC"), rain, {"open_meteo": 48})  # same instants
    bumped = rain.copy()
    bumped[10] += 0.1
    assert base != provenance.weather_fingerprint(ts, bumped, {"open_meteo": 48})
    assert base != provenance.weather_fingerprint(ts + pd.Timedelta(hours=1), rain, {"open_meteo": 48})
    assert base != provenance.weather_fingerprint(ts, rain, {"synthetic": 48})
    noisy = rain * (1.0 + 1e-15) + 1e-13  # CSV round-trip noise must not look like a new record
    assert base == provenance.weather_fingerprint(ts, noisy, {"open_meteo": 48})
    with pytest.raises(ValueError, match="timestamps"):
        provenance.weather_fingerprint(ts, rain[:-1], {})


@pytest.mark.integration
def test_weather_fetched_in_memory_matches_its_cached_copy(dcfg, fake_simulator):
    """The first build synthesises (offline) and caches the weather; the next build reads the CSV and reuses."""
    Path(dcfg["paths"]["weather_file"]).unlink()
    first = build_datasets(dcfg)
    assert first["reused"] is False and Path(dcfg["paths"]["weather_file"]).exists()
    again = build_datasets(dcfg)
    assert again["reused"] is True, again["rebuild_reasons"]


@pytest.mark.integration
def test_reuse_when_nothing_changed(built, fake_simulator):
    cfg, first = built
    mtimes = {split: path.stat().st_mtime_ns for split, path in _paths(cfg).items()}
    calls = len(fake_simulator.calls)
    again = build_datasets(cfg)
    assert again["reused"] is True and again["rebuild_reasons"] == []
    assert len(fake_simulator.calls) == calls
    assert {split: path.stat().st_mtime_ns for split, path in _paths(cfg).items()} == mtimes
    assert again["splits"]["test"]["n_windows"] == first["splits"]["test"]["n_windows"]
    forced = build_datasets(cfg, force=True)
    assert forced["reused"] is False and len(fake_simulator.calls) == calls + 1
    assert forced["rebuild_reasons"] == ["forced rebuild (--force)"]


@pytest.mark.integration
def test_rebuild_when_graph_attributes_change_but_topology_does_not(built, fake_simulator, log):
    """R1-02 / R5-02: stage 02 re-enrichment keeps GraphArrays.signature() but must rebuild."""
    cfg, _ = built
    old = load_payload(_paths(cfg)["train"])
    G = load_graph(cfg["paths"]["graph_file"])
    for _, data in G.nodes(data=True):
        data["elevation"] = float(data["elevation"]) + 150.0
        data["dist_to_drain_m"] = 5000.0
    for _, _, data in G.edges(data=True):
        data["grade"] = -float(data["grade"])
    save_graph(G, cfg["paths"]["graph_file"])
    assert graph_to_arrays(load_graph(cfg["paths"]["graph_file"])).signature() == old["graph_signature"]
    summary = build_datasets(cfg)
    assert summary["reused"] is False
    assert any("road graph attributes changed" in reason for reason in summary["rebuild_reasons"])
    assert "road graph attributes changed" in log.text
    new = load_payload(_paths(cfg)["train"])
    assert new["graph_attributes_sha256"] != old["graph_attributes_sha256"]
    np.testing.assert_allclose(new["static_raw"][:, 0].numpy(), old["static_raw"][:, 0].numpy() + 150.0, atol=1e-3)
    assert build_datasets(cfg)["reused"] is True


@pytest.mark.integration
def test_rebuild_when_the_topology_changes(built, fake_simulator, log):
    cfg, _ = built
    save_graph(make_grid_graph(4, 6), cfg["paths"]["graph_file"])
    summary = build_datasets(cfg)
    assert summary["reused"] is False and summary["n_nodes"] == 24
    assert "road graph changed (topology" in log.text


@pytest.mark.integration
def test_rebuild_when_the_weather_record_changes(built, fake_simulator, log):
    """R2-06 / R5-04: same range, new values (e.g. synthetic rows replaced by Open-Meteo) must rebuild."""
    cfg, _ = built
    old = load_payload(_paths(cfg)["val"])
    frame = storm_record(START, END)
    frame["precipitation_mm"] *= 3.0
    save_weather_csv(frame, cfg["paths"]["weather_file"])
    summary = build_datasets(cfg)
    assert summary["reused"] is False
    assert any(reason.startswith("weather record changed") for reason in summary["rebuild_reasons"])
    assert "weather record changed" in log.text
    new = load_payload(_paths(cfg)["val"])
    assert new["weather_fingerprint"] != old["weather_fingerprint"]
    assert float(new["rain"].sum()) == pytest.approx(3.0 * float(old["rain"].sum()), rel=1e-4)


@pytest.mark.integration
def test_rebuild_when_the_weather_sources_change(built, fake_simulator, log):
    cfg, _ = built
    save_weather_csv(storm_record(START, END, source="synthetic"), cfg["paths"]["weather_file"])
    summary = build_datasets(cfg)
    assert summary["reused"] is False and "synthetic" in " ".join(summary["rebuild_reasons"])


def _fake_observed(monkeypatch) -> None:
    module = types.ModuleType("src.hydrology.observed")
    module.load_flood_reports = lambda path, tz: pd.read_csv(path) if Path(path).exists() else pd.DataFrame()

    def merge_observed_labels(labels, timestamps, graph, reports, cfg):
        out = np.array(labels, copy=True)
        out[: min(len(reports), out.shape[0]), 0] = 1
        return out

    module.merge_observed_labels = merge_observed_labels
    monkeypatch.setitem(sys.modules, "src.hydrology.observed", module)


@pytest.mark.integration
def test_rebuild_when_flood_reports_change(dcfg, fake_simulator, monkeypatch, log):
    """R2-06: with labels.source=hybrid a new flood report must rebuild the labels."""
    _fake_observed(monkeypatch)
    cfg = deep_merge(dcfg, {"labels": {"source": "hybrid"}})
    reports = Path(cfg["paths"]["flood_reports_file"])
    reports.parent.mkdir(parents=True, exist_ok=True)
    reports.write_text("timestamp,lat,lon\n2021-09-01T18:00:00+05:30,12.93,77.67\n", encoding="utf-8")
    first = build_datasets(cfg)
    fingerprint = load_payload(_paths(cfg)["train"])["reports_fingerprint"]
    assert first["label_source"] == "hybrid" and fingerprint not in (None, provenance.REPORTS_ABSENT)
    assert build_datasets(cfg)["reused"] is True
    reports.write_text(reports.read_text() + "2022-09-05T03:00:00+05:30,12.92,77.68\n", encoding="utf-8")
    summary = build_datasets(cfg)
    assert summary["reused"] is False and any("flood reports changed" in r for r in summary["rebuild_reasons"])
    reports.unlink()
    assert build_datasets(cfg)["reused"] is False
    assert load_payload(_paths(cfg)["train"])["reports_fingerprint"] == provenance.REPORTS_ABSENT


@pytest.mark.unit
def test_reports_fingerprint_only_when_reports_feed_the_labels(cfg, tmp_path):
    path = Path(cfg["paths"]["flood_reports_file"])
    assert provenance.reports_fingerprint(cfg, "simulated") is None
    assert provenance.reports_fingerprint(cfg, "observed") == provenance.REPORTS_ABSENT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"timestamp,lat,lon\n")
    first = provenance.reports_fingerprint(cfg, "hybrid")
    path.write_bytes(b"timestamp,lat,lon\n2022-01-01T00:00:00+05:30,12.9,77.6\n")
    assert first != provenance.reports_fingerprint(cfg, "hybrid") and len(first) == 16


@pytest.mark.integration
def test_rebuild_when_config_changes(built, fake_simulator, log):
    cfg, _ = built
    summary = build_datasets(deep_merge(cfg, {"dataset": {"wet_window_min_mm": 1.0}}))
    assert summary["reused"] is False
    assert any("dataset configuration changed" in reason for reason in summary["rebuild_reasons"])


@pytest.mark.integration
def test_rebuild_corrupt_files_and_mismatched_builds(built, fake_simulator, log):
    cfg, _ = built
    _paths(cfg)["val"].write_bytes(b"not a torch file")
    assert build_datasets(cfg)["reused"] is False
    assert "rebuilding" in log.text and "unusable" in log.text
    test_path = _paths(cfg)["test"]
    torch.save({**load_payload(test_path), "build_id": "from-another-build"}, test_path)
    summary = build_datasets(cfg)
    assert summary["reused"] is False and "different builds" in log.text
    torch.save(load_payload(_paths(cfg)["val"]), test_path)  # a val payload in the test slot
    assert build_datasets(cfg)["reused"] is False and "holds the 'val' split" in log.text


# --------------------------------------------------------------------------- format-1 payloads


def _as_v1(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key in V1_PAYLOAD_KEYS or key in ("seed", "build_id")} \
        | {"format_version": 1}


@pytest.mark.integration
def test_format_1_payloads_load_but_are_rebuilt(built, fake_simulator, log):
    cfg, _ = built
    for path in _paths(cfg).values():
        torch.save(_as_v1(load_payload(path)), path)
    legacy = FloodSequenceDataset(_paths(cfg)["val"])
    assert legacy.graph_attributes_sha256 is None and legacy.split == "val" and len(legacy) > 0
    assert "format_version 1" in log.text
    summary = build_datasets(cfg)
    assert summary["reused"] is False and "format_version 1" in summary["rebuild_reasons"][0]
    assert load_payload(_paths(cfg)["val"])["format_version"] == 2


@pytest.mark.unit
def test_format_2_payloads_require_the_fingerprints(built):
    cfg, _ = built
    payload = load_payload(_paths(cfg)["val"])
    missing = {k: v for k, v in payload.items() if k != "weather_fingerprint"}
    with pytest.raises(DatasetError, match="weather_fingerprint"):
        FloodSequenceDataset(missing)
    with pytest.raises(DatasetError, match="format_version"):
        FloodSequenceDataset({**payload, "format_version": 3})


# --------------------------------------------------------------------------- R3-03: safe loading only


class _Exploit:
    """Pickles to a call of ``Path.touch(marker)`` - what a malicious .pt file would carry."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return pathlib.Path.touch, (self.marker,)


@pytest.mark.unit
def test_load_payload_never_runs_the_full_unpickler(tmp_path):
    marker = tmp_path / "pwned"
    evil = tmp_path / "evil.pt"
    torch.save({"format_version": FORMAT_VERSION, "split": "val", "payload": _Exploit(marker)}, evil)
    with pytest.raises(DatasetError, match="not a Namma-Flow dataset"):
        load_payload(evil)
    with pytest.raises(DatasetError, match="not a Namma-Flow dataset"):
        FloodSequenceDataset(evil)
    assert not marker.exists(), "the payload's pickled code ran"


@pytest.mark.integration
def test_malicious_dataset_file_is_rebuilt_not_executed(built, fake_simulator, log, tmp_path):
    cfg, _ = built
    marker = tmp_path / "pwned"
    torch.save({"format_version": FORMAT_VERSION, "split": "val", "x": _Exploit(marker)}, _paths(cfg)["val"])
    assert build_datasets(cfg)["reused"] is False
    assert not marker.exists() and "unusable" in log.text


@pytest.mark.unit
def test_load_payload_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="04_dataset_builder"):
        load_payload(tmp_path / "missing.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"\x00garbage")
    with pytest.raises(DatasetError, match="corrupt"):
        load_payload(corrupt)
    truncated = tmp_path / "truncated.pt"
    torch.save({"a": torch.arange(1000)}, truncated)
    truncated.write_bytes(truncated.read_bytes()[:200])
    with pytest.raises(DatasetError, match="corrupt or truncated"):
        load_payload(truncated)
    listing = tmp_path / "list.pt"
    torch.save([1, 2, 3], listing)
    with pytest.raises(DatasetError, match="dict"):
        load_payload(listing)
    legacy = tmp_path / "numpy.pt"
    torch.save({"a": np.arange(3)}, legacy)  # numpy objects need the full unpickler: refused, never retried unsafely
    with pytest.raises(DatasetError, match="not a Namma-Flow dataset") as excinfo:
        load_payload(legacy)
    assert "weights_only=False" not in str(excinfo.value)
