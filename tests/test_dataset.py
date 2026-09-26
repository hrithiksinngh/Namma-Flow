"""Tests for the dataset builder and loaders (``src.data_pipeline.dataset`` + the stage-04 CLI).

The hydrology simulator (module D) is replaced by a deterministic fake injected into
``sys.modules`` so these tests do not depend on its calibration (one test uses the real
simulator when it is importable). Weather comes from a hand-written schema-2.2 CSV so
window selection is predictable (helpers in ``tests/dataset_fixtures.py``). Input
fingerprints, reuse, the test split and safe loading are tested in ``test_dataset_inputs.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data_pipeline import dataset as ds
from src.data_pipeline.dataset import (
    FORMAT_VERSION,
    REQUIRED_PAYLOAD_KEYS,
    DatasetError,
    DatasetSettings,
    FloodSequenceDataset,
    build_datasets,
    collate_windows,
    dataset_config_hash,
    load_payload,
    make_loader,
    select_windows,
    window_positive_counts,
)
from src.data_pipeline.features import FeatureScaler, build_node_features, dynamic_features
from src.data_pipeline.graph_io import graph_to_arrays, load_graph
from src.utils.config import ConfigError, deep_merge
from tests.conftest import TZ, make_grid_graph
from tests.dataset_fixtures import fake_simulator, log  # noqa: F401 - pytest fixtures
from tests.dataset_fixtures import FLOOD_MM_H, fake_simulate, split_years
from tests.dataset_fixtures import setup_config as _setup

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "data_pipeline" / "04_dataset_builder.py"
SEQ, WARM, LOOK = 16, 4, 24  # config defaults: seq_len, warmup_steps, lookback_hours
SCORED = SEQ - WARM


# --------------------------------------------------------------------------- fixtures / helpers


@pytest.fixture
def dcfg(cfg):
    return _setup(cfg)


@pytest.fixture
def built(dcfg, fake_simulator):
    summary = build_datasets(dcfg)
    return dcfg, summary


def _settings(cfg: dict, **dataset) -> DatasetSettings:
    return DatasetSettings.from_config(deep_merge(cfg, {"dataset": dataset}))


# --------------------------------------------------------------------------- settings


@pytest.mark.unit
def test_settings_read_config(cfg):
    s = DatasetSettings.from_config(cfg)
    assert s.seq_len == SEQ and s.warmup_steps == WARM and s.lookback_hours == LOOK
    assert s.scored_steps == SCORED == s.val_stride_h  # val/test scored hours tile exactly (R2-08)
    assert s.rolling_windows_h == (3, 6, 12, 24)
    assert s.season_months == (5, 6, 7, 8, 9, 10, 11)
    assert s.val_years == (2022,) and s.test_years == (2024,)
    assert s.splits == ("train", "val", "test") and s.has_test_split
    assert s.static_features == ("elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation", "is_sink")
    assert s.seed == 42 and s.timezone == TZ
    assert s.feature_names[5:] == ("precip_mm_h", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm")


@pytest.mark.unit
def test_settings_defaults_when_section_missing():
    s = DatasetSettings.from_config({"project": {}, "paths": {}})
    assert s.seq_len == SEQ and s.warmup_steps == WARM and s.val_stride_h == 12 and s.train_stride_h == 6
    assert s.scaler_max_samples == 2_000_000 and s.val_years == (2022,) and s.test_years == (2024,)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset, match",
    [
        ({"season_months": [0]}, "season_months"),
        ({"season_months": []}, "season_months"),
        ({"season_months": "5"}, "season_months"),
        ({"seq_len": 6, "warmup_steps": 6}, "warmup_steps"),
        ({"seq_len": 0}, "seq_len"),
        ({"lookback_hours": 12}, "lookback_hours"),
        ({"train_stride_h": 0}, "train_stride_h"),
        ({"val_stride_h": 1.5}, "val_stride_h"),
        ({"val_years": []}, "val_years"),
        ({"test_years": "2024"}, "test_years"),
        ({"test_years": [0]}, "test_years"),
        ({"val_years": [2022], "test_years": [2022, 2024]}, "overlap"),
        ({"wet_window_min_mm": -1}, "wet_window_min_mm"),
        ({"dry_window_keep_frac": 1.5}, "dry_window_keep_frac"),
        ({"edge_features": ["length"]}, "edge_features"),
        ({"static_features": []}, "static_features"),
        ({"scaler_max_samples": 0}, "scaler_max_samples"),
        ({"seq_len": True}, "seq_len"),
    ],
)
def test_settings_reject_bad_values(cfg, dataset, match):
    with pytest.raises(ConfigError, match=match):
        _settings(cfg, **dataset)


@pytest.mark.unit
def test_dataset_config_hash_tracks_upstream_sections(cfg):
    base = dataset_config_hash(cfg)
    assert base == dataset_config_hash(deep_merge(cfg, {"app": {"top_k": 3}, "model": {"hidden_dim": 8}}))
    for section, change in [("hydrology", {"infiltration_mm_h": 9.0}), ("dataset", {"seq_len": 12}),
                            ("labels", {"source": "hybrid"}), ("rainfall_field", {"seed": 1})]:
        assert dataset_config_hash(deep_merge(cfg, {section: change})) != base


# --------------------------------------------------------------------------- window selection


def _index(start: str, hours: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=hours, freq="h", tz=TZ)


@pytest.mark.unit
def test_select_windows_respects_lookback_season_and_strides(cfg):
    ts = _index("2021-05-01", 24 * 60)
    s = _settings(cfg, val_years=[2030], wet_window_min_mm=0.0, season_months=[5])
    sel = select_windows(ts, np.zeros(len(ts)), s)
    assert sel.val.size == 0
    starts = ts[sel.train]
    assert sel.train.min() >= s.lookback_hours
    assert (sel.train + s.seq_len <= len(ts)).all()
    assert set(starts.month) == {5}
    assert set(starts.hour % 6) == {0}
    assert np.all(np.diff(sel.train) == 6)
    assert sel.n_wet == sel.n_candidates  # wet_window_min_mm = 0 keeps everything


@pytest.mark.unit
def test_select_windows_keeps_wet_and_sampled_dry_windows(cfg):
    ts = _index("2021-06-01", 24 * 30)
    rain = np.zeros(len(ts))
    rain[300:303] = [2.0, 2.0, 1.0]  # exactly 5 mm: counts as wet (>=)
    none = select_windows(ts, rain, _settings(cfg, val_years=[2030], dry_window_keep_frac=0.0))
    covered = [s for s in none.train if s <= 302 and s + SEQ > 300]
    assert list(none.train) == covered and len(covered) > 0
    everything = select_windows(ts, rain, _settings(cfg, val_years=[2030], dry_window_keep_frac=1.0))
    assert everything.train.size > none.train.size
    assert everything.n_dry_kept == everything.n_candidates - everything.n_wet


@pytest.mark.unit
def test_select_windows_is_seeded(cfg):
    ts = _index("2021-06-01", 24 * 60)
    rain = np.zeros(len(ts))
    a = select_windows(ts, rain, _settings(cfg, val_years=[2030], dry_window_keep_frac=0.3))
    b = select_windows(ts, rain, _settings(cfg, val_years=[2030], dry_window_keep_frac=0.3))
    c = select_windows(ts, rain, DatasetSettings.from_config(
        deep_merge(cfg, {"project": {"seed": 7}, "dataset": {"val_years": [2030], "dry_window_keep_frac": 0.3}})))
    np.testing.assert_array_equal(a.train, b.train)
    assert not np.array_equal(a.train, c.train)


@pytest.mark.unit
def test_select_windows_splits_by_last_hour_year_and_drops_straddlers(cfg):
    ts = _index("2021-12-20", 24 * 25)
    s = _settings(cfg, val_years=[2022], season_months=list(range(1, 13)), wet_window_min_mm=0.0,
                  train_stride_h=1, val_stride_h=1)
    sel = select_windows(ts, np.zeros(len(ts)), s)
    first_train, last_train = ts[sel.train], ts[sel.train + SEQ - 1]
    first_val = ts[sel.val]
    assert (first_train.year == 2021).all() and (last_train.year == 2021).all()
    assert (first_val.year == 2022).all()
    assert sel.n_straddling == SEQ - 1  # windows starting 2021-12-31 09:00 .. 23:00 end in 2022
    assert sel.test.size == 0
    assert set(sel.train).isdisjoint(sel.val)


@pytest.mark.unit
def test_select_windows_rejects_windows_across_gaps(cfg):
    ts = _index("2021-06-01", 24 * 10)
    gap_ts = ts.delete(range(100, 103))  # three missing hours
    s = _settings(cfg, val_years=[2030], wet_window_min_mm=0.0, train_stride_h=1)
    sel = select_windows(gap_ts, np.zeros(len(gap_ts)), s)
    for start in sel.train:
        span = gap_ts[start - LOOK: start + SEQ]
        assert (np.diff(span.asi8) == np.diff(span.asi8)[0]).all()
    full = select_windows(ts, np.zeros(len(ts)), s)
    assert sel.train.size < full.train.size


@pytest.mark.unit
def test_select_windows_short_record_and_validation(cfg, log):
    s = DatasetSettings.from_config(cfg)
    short = LOOK + SEQ - 1  # one hour short of a single window
    empty = select_windows(_index("2021-06-01", short), np.zeros(short), s)
    assert empty.train.size == empty.val.size == empty.test.size == empty.n_candidates == 0
    one = select_windows(_index("2021-06-01", short + 1), np.zeros(short + 1), s)
    assert one.n_candidates == 1
    with pytest.raises(ValueError, match="areal"):
        select_windows(_index("2021-06-01", 40), np.zeros(39), s)
    with pytest.raises(ValueError, match="increasing"):
        select_windows(_index("2021-06-01", 40)[::-1], np.zeros(40), s)
    naive = pd.date_range("2021-06-01", periods=200, freq="h")
    rain = np.full(200, np.nan)
    sel = select_windows(naive, rain, _settings(cfg, val_years=[2030], dry_window_keep_frac=1.0))
    assert sel.train.size > 0  # NaN rain is dry, tz-naive stamps are local time


@pytest.mark.unit
def test_window_positive_counts_uses_scored_steps_only():
    labels = np.zeros((40, 3), dtype=np.uint8)
    labels[2, 0] = 1        # warm-up step of window 0 (start 0) -> not scored
    labels[10, :] = 1       # scored step of window 0; warm-up of window 8
    counts = window_positive_counts(labels, np.array([0, 8, 16]), seq_len=12, warmup=4)
    np.testing.assert_array_equal(counts, [3, 0, 0])
    assert window_positive_counts(labels, np.array([], dtype=np.int64), 12, 4).size == 0


# --------------------------------------------------------------------------- build_datasets


@pytest.mark.integration
def test_build_datasets_writes_complete_payloads(built):
    cfg, summary = built
    train = load_payload(cfg["paths"]["train_dataset"])
    val = load_payload(cfg["paths"]["val_dataset"])
    test = load_payload(cfg["paths"]["test_dataset"])
    for split, payload in (("train", train), ("val", val), ("test", test)):
        assert set(REQUIRED_PAYLOAD_KEYS) <= set(payload)
        assert payload["format_version"] == FORMAT_VERSION and payload["split"] == split
        n = len(payload["node_ids"])
        h = payload["rain"].shape[0]
        assert n == 25
        assert payload["rain"].dtype == torch.float32 and payload["rain"].shape == (h, n)
        assert payload["labels"].dtype == torch.uint8 and payload["labels"].shape == (h, n)
        assert payload["depth"].dtype == torch.float16 and payload["depth"].shape == (h, n)
        assert payload["timestamps"].dtype == torch.int64 and payload["timestamps"].shape == (h,)
        assert payload["lon"].dtype == torch.float64 and payload["lon"].shape == (n,)
        assert payload["edge_index"].dtype == torch.int64 and payload["edge_index"].shape[0] == 2
        e = payload["edge_index"].shape[1]
        assert payload["edge_attr"].shape == (e, 2) and payload["edge_attr_raw"].shape == (e, 2)
        assert payload["static_raw"].shape == (n, 5)
        assert payload["feature_names"] == ["elevation", "dist_to_drain_m", "relative_elevation", "flow_accumulation",
                                            "is_sink", "precip_mm_h",
                                            "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"]
        assert payload["config_hash"] == dataset_config_hash(cfg)
        assert payload["label_source"] == "simulated"
        assert payload["flood_threshold_m"] == pytest.approx(0.15)
        assert payload["timezone"] == TZ
        starts = payload["window_starts"].numpy()
        ts = payload["timestamps"].numpy()
        assert payload["seq_len"] == SEQ and payload["warmup_steps"] == WARM
        assert starts.min() >= LOOK and starts.max() + SEQ <= h
        np.testing.assert_array_equal(ts[starts + SEQ - 1] - ts[starts - LOOK],
                                      np.full(starts.size, (LOOK + SEQ - 1) * 3600))
        assert payload["stats"]["n_windows"] == starts.size
        assert payload["stats"]["n_hours"] == h
    assert split_years(val) == {2022} and split_years(test) == {2023} and split_years(train) == {2021}
    assert summary["splits"]["train"]["n_windows"] == train["stats"]["n_windows"]
    assert summary["splits"]["val"]["size_bytes"] == Path(cfg["paths"]["val_dataset"]).stat().st_size
    assert summary["splits"]["test"]["n_windows"] == test["stats"]["n_windows"] > 0
    assert summary["reused"] is False
    report = json.loads((Path(cfg["paths"]["reports_dir"]) / "dataset_summary.json").read_text())
    assert report["splits"]["train"]["n_windows"] == summary["splits"]["train"]["n_windows"]


@pytest.mark.integration
def test_build_datasets_stores_exactly_the_needed_hours_and_consistent_labels(built):
    cfg, _ = built
    payload = load_payload(cfg["paths"]["train_dataset"])
    starts = payload["window_starts"].numpy()
    ts = payload["timestamps"].numpy()
    needed = set()
    for s in starts:
        needed.update(ts[s - LOOK: s + SEQ].tolist())
    assert set(ts.tolist()) == needed
    rain, labels = payload["rain"].numpy(), payload["labels"].numpy()
    np.testing.assert_array_equal(labels, (rain >= FLOOD_MM_H).astype(np.uint8))
    np.testing.assert_allclose(payload["depth"].numpy().astype(np.float32), rain / 40.0, rtol=2e-3, atol=1e-3)
    stats = payload["stats"]
    counts = window_positive_counts(labels, starts, SEQ, WARM)
    assert stats["n_windows_with_flood"] == int((counts > 0).sum()) > 0
    assert stats["pos_rate"] == pytest.approx(counts.sum() / (starts.size * SCORED * 25))


@pytest.mark.integration
def test_scaler_is_fitted_on_train_windows_only(built):
    cfg, _ = built
    train = load_payload(cfg["paths"]["train_dataset"])
    val = load_payload(cfg["paths"]["val_dataset"])
    assert train["scaler"] == val["scaler"] == load_payload(cfg["paths"]["test_dataset"])["scaler"]
    scaler = FeatureScaler.from_dict(train["scaler"])
    rain = train["rain"].numpy().astype(np.float64)
    ts = train["timestamps"].numpy()
    starts = train["window_starts"].numpy()
    hours = sorted({int(h) for s in starts for h in range(s, s + SEQ)})
    samples = []
    for h in hours:  # dynamic features of each train window hour, via the canonical function
        samples.append(dynamic_features(rain[h - 24: h + 1], 24, (3, 6, 12, 24))[0])
    expected = np.log1p(np.concatenate(samples).astype(np.float64))
    np.testing.assert_allclose(scaler.dynamic_mean, expected.mean(axis=0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(scaler.dynamic_std, expected.std(axis=0), rtol=1e-4, atol=1e-5)
    static = train["static_raw"].numpy()
    static = static.astype(np.float64)
    flow = list(train["static_feature_names"]).index("flow_accumulation")
    static[:, flow] = np.log1p(static[:, flow])  # heavy-tailed attribute: log1p before z-scoring
    np.testing.assert_allclose(scaler.static_mean, static.mean(axis=0), rtol=1e-6, atol=1e-4)
    lengths = train["edge_attr_raw"].numpy()[:, 0].astype(np.float64)
    assert scaler.edge_len_mean == pytest.approx(np.log1p(lengths).mean(), rel=1e-5)
    np.testing.assert_allclose(train["edge_attr"].numpy(),
                               scaler.transform_edges(lengths, train["edge_attr_raw"].numpy()[:, 1]), atol=1e-6)
    assert ts.size > 0


@pytest.mark.integration
def test_scaler_subsamples_when_train_hours_exceed_the_budget(dcfg, fake_simulator):
    cfg = deep_merge(dcfg, {"dataset": {"scaler_max_samples": 100}})
    build_datasets(cfg)
    scaler = FeatureScaler.from_dict(load_payload(cfg["paths"]["train_dataset"])["scaler"])
    assert np.isfinite(scaler.dynamic_mean).all() and (scaler.dynamic_std > 0).all()


@pytest.mark.integration
def test_build_datasets_builds_the_graph_when_missing(cfg, fake_simulator, log):
    out = _setup(cfg)
    Path(out["paths"]["graph_file"]).unlink()
    summary = build_datasets(out)
    assert Path(out["paths"]["graph_file"]).exists()
    assert summary["n_nodes"] == 36  # the offline synthetic 6x6 grid from conftest's cfg
    assert summary["graph"]["source"] == "synthetic_grid"
    assert "running stage 01" in log.text


@pytest.mark.integration
def test_build_datasets_enriches_graph_without_static_features(cfg, fake_simulator, log):
    bare = make_grid_graph(5, 5)
    for _, data in bare.nodes(data=True):
        del data["dist_to_drain_m"]
    out = _setup(cfg, graph=bare)
    build_datasets(out)
    reloaded = graph_to_arrays(load_graph(out["paths"]["graph_file"]))
    assert "dist_to_drain_m" in reloaded.node_attrs
    assert "enriching" in log.text


@pytest.mark.integration
def test_build_datasets_rejects_unknown_static_feature(cfg, fake_simulator):
    out = _setup(cfg, dataset={"static_features": ["elevation", "soil_type"]},
                 model={"node_in_dim": 7})
    with pytest.raises(ConfigError, match="soil_type"):
        build_datasets(out)


@pytest.mark.integration
def test_build_datasets_explains_empty_splits(cfg, fake_simulator):
    with pytest.raises(DatasetError, match="val_years"):
        build_datasets(_setup(cfg, dataset={"val_years": [2030]}))
    with pytest.raises(DatasetError, match="No training windows"):
        build_datasets(_setup(cfg, dataset={"val_years": [2021, 2022]}))
    with pytest.raises(DatasetError, match="No test windows.*test_years"):
        build_datasets(_setup(cfg, dataset={"test_years": [2030]}))


@pytest.mark.integration
def test_build_datasets_rejects_too_short_weather(cfg, fake_simulator):
    out = _setup(cfg, start="2022-06-01", end="2022-06-01")
    with pytest.raises(DatasetError, match="lookback_hours"):
        build_datasets(out)


@pytest.mark.integration
def test_build_datasets_requires_the_simulator(dcfg, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.hydrology.simulator", None)  # import raises ImportError
    with pytest.raises(DatasetError, match="simulator"):
        build_datasets(dcfg)


@pytest.mark.integration
def test_build_datasets_validates_simulator_output(dcfg, monkeypatch, log):
    module = types.ModuleType("src.hydrology.simulator")
    module.simulate_labels = lambda g, r, c: (np.zeros((3, 3), np.uint8), np.zeros((3, 3), np.float32))
    monkeypatch.setitem(sys.modules, "src.hydrology.simulator", module)
    with pytest.raises(DatasetError, match="shape"):
        build_datasets(dcfg)

    def noisy(graph, rain, cfg):
        labels, depth = fake_simulate(graph, rain, cfg)
        labels = labels.astype(np.int64) * 2  # non-binary
        depth[0, 0] = np.nan
        depth[1, 0] = -1.0
        return labels, depth

    module.simulate_labels = noisy
    build_datasets(dcfg, force=True)
    payload = load_payload(dcfg["paths"]["train_dataset"])
    assert set(np.unique(payload["labels"].numpy())) <= {0, 1}
    assert torch.isfinite(payload["depth"].float()).all() and (payload["depth"] >= 0).all()
    assert "non-binary" in log.text and "non-finite/negative depth" in log.text


@pytest.mark.integration
def test_build_datasets_rejects_unknown_label_source(dcfg, fake_simulator):
    with pytest.raises(ConfigError, match="labels.source"):
        build_datasets(deep_merge(dcfg, {"labels": {"source": "crowd"}}))


def _fake_observed(monkeypatch, reports: pd.DataFrame) -> types.ModuleType:
    module = types.ModuleType("src.hydrology.observed")
    module.load_flood_reports = lambda path, tz: reports

    def merge_observed_labels(labels, timestamps, graph, reps, cfg):
        out = np.array(labels, copy=True)
        if len(reps):
            out[: min(5, out.shape[0]), 0] = 1
        return out

    module.merge_observed_labels = merge_observed_labels
    monkeypatch.setitem(sys.modules, "src.hydrology.observed", module)
    return module


@pytest.mark.integration
def test_build_datasets_hybrid_and_observed_label_sources(dcfg, fake_simulator, monkeypatch, log):
    reports = pd.DataFrame({"timestamp": [pd.Timestamp("2021-09-01", tz=TZ)], "lat": [12.93], "lon": [77.67]})
    _fake_observed(monkeypatch, reports)
    hybrid = build_datasets(deep_merge(dcfg, {"labels": {"source": "hybrid"}}))
    assert hybrid["label_source"] == "hybrid"
    observed_cfg = deep_merge(dcfg, {"labels": {"source": "observed"}})
    _fake_observed(monkeypatch, reports.iloc[0:0])
    observed = build_datasets(observed_cfg)
    assert observed["label_source"] == "observed"
    assert observed["splits"]["train"]["pos_rate"] == 0.0  # no reports -> no positives
    assert "no flood reports" in log.text


@pytest.mark.integration
def test_build_datasets_observed_module_missing(dcfg, fake_simulator, monkeypatch, log):
    monkeypatch.setitem(sys.modules, "src.hydrology.observed", None)
    with pytest.raises(DatasetError, match="observed"):
        build_datasets(deep_merge(dcfg, {"labels": {"source": "observed"}}))
    summary = build_datasets(deep_merge(dcfg, {"labels": {"source": "hybrid"}}))
    assert summary["label_source"] == "simulated"
    assert "using simulated labels only" in log.text


@pytest.mark.integration
def test_build_datasets_with_real_simulator(dcfg):
    real = pytest.importorskip("src.hydrology.simulator")
    if not hasattr(real, "simulate_labels"):
        pytest.skip("src.hydrology.simulator has no simulate_labels yet")
    summary = build_datasets(dcfg)
    payload = load_payload(dcfg["paths"]["train_dataset"])
    assert payload["labels"].dtype == torch.uint8
    assert summary["splits"]["train"]["n_windows"] > 0


# --------------------------------------------------------------------------- FloodSequenceDataset


@pytest.mark.integration
def test_dataset_items_match_the_feature_pipeline(built):
    cfg, _ = built
    data = FloodSequenceDataset(cfg["paths"]["train_dataset"])
    payload = data.payload
    assert len(data) == payload["stats"]["n_windows"] > 0
    item = data[1]
    assert set(item) == {"x", "y", "mask", "start", "index"}
    assert item["x"].shape == (SEQ, 25, 10) and item["x"].dtype == torch.float32
    assert item["y"].shape == (SEQ, 25) and item["y"].dtype == torch.float32
    assert item["mask"].dtype == torch.bool and item["mask"].tolist() == [False] * WARM + [True] * SCORED
    s = int(payload["window_starts"][1])
    rain = payload["rain"].numpy()
    expected = build_node_features(payload["static_raw"].numpy(), rain[s - LOOK: s + SEQ], LOOK, (3, 6, 12, 24),
                                   data.scaler)
    np.testing.assert_array_equal(item["x"].numpy(), expected)
    np.testing.assert_array_equal(item["y"].numpy(), payload["labels"].numpy()[s: s + SEQ])
    assert item["start"] == int(payload["timestamps"][s]) and item["index"] == 1
    assert torch.equal(data[-1]["x"], data[len(data) - 1]["x"])
    with pytest.raises(IndexError):
        data[len(data)]
    assert data.num_nodes == 25 and data.num_features == 10
    assert data.edge_index.dtype == torch.long and data.edge_attr.shape == (data.num_edges, 2)
    assert data.feature_names == payload["feature_names"]
    assert data.graph_signature == payload["graph_signature"]
    assert data.split == "train" and data.graph_attributes_sha256 == payload["graph_attributes_sha256"]
    assert data.window_depth(1).shape == (SEQ, 25)
    times = data.window_timestamps(1)
    assert len(times) == SEQ and str(times.tz) == TZ
    assert times[0] == pd.Timestamp(item["start"], unit="s", tz="UTC").tz_convert(TZ)
    assert (np.diff(times.as_unit("s").asi8) == 3600).all()


@pytest.mark.integration
def test_dataset_flood_flags_and_subsets(built):
    cfg, _ = built
    data = FloodSequenceDataset(cfg["paths"]["train_dataset"])
    assert data.window_has_flood.dtype == bool and data.window_has_flood.shape == (len(data),)
    assert data.window_has_flood.any()
    assert data.pos_rate == pytest.approx(data.payload["stats"]["pos_rate"])
    subset = FloodSequenceDataset(data.payload, max_windows=3)
    assert len(subset) == 3
    starts = data.payload["window_starts"].numpy()
    assert subset[2]["start"] == int(data.payload["timestamps"][starts[-1]])  # evenly spaced incl. the last
    assert len(FloodSequenceDataset(data.payload, max_windows=10_000)) == len(data)
    with pytest.raises(ValueError, match="max_windows"):
        FloodSequenceDataset(data.payload, max_windows=0)


@pytest.mark.unit
def test_dataset_rejects_invalid_payloads(built):
    cfg, _ = built
    good = load_payload(cfg["paths"]["val_dataset"])
    with pytest.raises(DatasetError, match="format_version"):
        FloodSequenceDataset({**good, "format_version": 99})
    missing = {k: v for k, v in good.items() if k != "window_starts"}
    with pytest.raises(DatasetError, match="window_starts"):
        FloodSequenceDataset(missing)
    with pytest.raises(DatasetError, match="shape"):
        FloodSequenceDataset({**good, "labels": good["labels"][:-1]})
    with pytest.raises(DatasetError, match="lookback"):
        FloodSequenceDataset({**good, "window_starts": torch.tensor([0])})
    shifted = good["timestamps"].clone()
    shifted[int(good["window_starts"][0])] += 60
    with pytest.raises(DatasetError, match="contiguous"):
        FloodSequenceDataset({**good, "timestamps": shifted})
    with pytest.raises(DatasetError, match="mapping"):
        FloodSequenceDataset([1, 2])  # type: ignore[arg-type]


@pytest.mark.unit
def test_dataset_accepts_numpy_payload(built):
    cfg, _ = built
    payload = load_payload(cfg["paths"]["val_dataset"])
    as_numpy = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in payload.items()}
    data = FloodSequenceDataset(as_numpy)
    assert data[0]["x"].shape[1] == 25


# --------------------------------------------------------------------------- collation / loaders


def _item(t: int, n: int, f: int, value: float, start: int) -> dict:
    return {"x": torch.full((t, n, f), value), "y": torch.full((t, n), value), "mask": torch.arange(t) >= 1,
            "start": start, "index": start}


@pytest.mark.unit
def test_collate_windows_offsets_edges_and_stacks_nodes():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    batch = [_item(4, 3, 2, 1.0, 10), _item(4, 3, 2, 2.0, 20)]
    out = collate_windows(batch, edge_index, edge_attr, 3)
    assert out["x"].shape == (4, 6, 2) and out["y"].shape == (4, 6)
    assert torch.all(out["x"][:, :3] == 1.0) and torch.all(out["x"][:, 3:] == 2.0)
    assert out["edge_index"].tolist() == [[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]
    assert out["edge_attr"].shape == (6, 2) and torch.equal(out["edge_attr"][3:], edge_attr)
    assert out["mask"].tolist() == [False, True, True, True]
    assert out["batch_size"] == 2 and out["num_nodes"] == 3
    assert out["start"].tolist() == [10, 20] and out["index"].tolist() == [10, 20]


@pytest.mark.unit
def test_collate_windows_validation():
    edge_index = torch.tensor([[0], [1]])
    edge_attr = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="empty"):
        collate_windows([], edge_index, edge_attr, 3)
    with pytest.raises(ValueError, match="nodes"):
        collate_windows([_item(4, 2, 2, 1.0, 0)], edge_index, edge_attr, 3)
    with pytest.raises(ValueError, match="same shape"):
        collate_windows([_item(4, 3, 2, 1.0, 0), _item(5, 3, 2, 1.0, 0)], edge_index, edge_attr, 3)
    with pytest.raises(ValueError, match="edge_attr"):
        collate_windows([_item(4, 3, 2, 1.0, 0)], edge_index, torch.zeros(2, 2), 3)
    odd = _item(4, 3, 2, 1.0, 0)
    odd["mask"] = torch.ones(4, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask"):
        collate_windows([_item(4, 3, 2, 1.0, 0), odd], edge_index, edge_attr, 3)
    no_edges = collate_windows([_item(2, 3, 2, 1.0, 0)] * 2, torch.zeros(2, 0, dtype=torch.long),
                               torch.zeros(0, 2), 3)
    assert no_edges["edge_index"].shape == (2, 0)


@pytest.mark.integration
def test_make_loader_batches_whole_dataset(built, log):
    cfg, _ = built
    data = FloodSequenceDataset(cfg["paths"]["val_dataset"])
    loader = make_loader(data, batch_size=4, shuffle=False)
    batches = list(loader)
    assert sum(b["batch_size"] for b in batches) == len(data)
    first = batches[0]
    assert first["x"].shape == (SEQ, first["batch_size"] * 25, 10)
    assert first["edge_index"].max() < first["batch_size"] * 25
    sampler = torch.utils.data.WeightedRandomSampler(np.ones(len(data)), num_samples=3, replacement=True)
    sampled = list(make_loader(data, batch_size=2, shuffle=True, sampler=sampler))
    assert sum(b["batch_size"] for b in sampled) == 3
    assert "shuffle is ignored" in log.text
    with pytest.raises(ValueError, match="batch_size"):
        make_loader(data, batch_size=0, shuffle=False)
    with pytest.raises(TypeError, match="FloodSequenceDataset"):
        make_loader([1, 2, 3], batch_size=1, shuffle=False)


# --------------------------------------------------------------------------- summary / CLI


@pytest.mark.unit
def test_format_summary_lists_splits(built):
    _, summary = built
    format_summary = _load_cli().format_summary
    text = format_summary(summary)
    assert "Namma-Flow stage 04" in text
    assert "Train" in text and "Validation" in text and "Test (held out)" in text and "MB" in text
    assert "Fingerprints" in text and summary["fingerprints"]["weather_fingerprint"] in text
    assert "validation 2022; test 2023" in text and "scored 12" in text
    years_label = _load_cli().years_label
    assert years_label([2023, 2018, 2019, 2020, 2020]) == "2018-2020, 2023"
    assert years_label([2022, 2024]) == "2022, 2024" and years_label([]) == "-"
    reused = format_summary({**summary, "reused": True})
    assert "--force" in reused and "weather record" in reused
    rebuilt = format_summary({**summary, "rebuild_reasons": ["weather record changed (x -> y)"]})
    assert "built in" in rebuilt and "weather record changed" in rebuilt
    no_test = format_summary({**summary, "test_years": [],
                              "splits": {k: v for k, v in summary["splits"].items() if k != "test"}})
    assert "not built (dataset.test_years is empty)" in no_test and "test none" in no_test


def _load_cli():
    spec = importlib.util.spec_from_file_location("dataset_builder_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(cfg: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}), encoding="utf-8")
    return path


@pytest.mark.e2e
def test_cli_builds_and_prints_summary(dcfg, fake_simulator, tmp_path, capsys):
    cli = _load_cli()
    config = _write_config(dcfg, tmp_path)
    assert cli.main(["--config", str(config), "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Train" in out and "Validation" in out and "Test (held out)" in out and "built in" in out
    assert cli.main(["--config", str(config)]) == 0
    assert "reused" in capsys.readouterr().out
    assert cli.main(["--config", str(config), "--force"]) == 0


@pytest.mark.e2e
def test_cli_reports_handled_errors(cfg, fake_simulator, tmp_path, capsys):
    cli = _load_cli()
    bad = _write_config(_setup(cfg, dataset={"val_years": [2030]}), tmp_path)
    assert cli.main(["--config", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err and "val_years" in err
    assert cli.main(["--config", str(tmp_path / "nope.yaml")]) == 1
    assert "not found" in capsys.readouterr().err


@pytest.mark.e2e
def test_cli_reports_unexpected_errors(dcfg, tmp_path, capsys, monkeypatch):
    cli = _load_cli()
    config = _write_config(dcfg, tmp_path)

    def boom(cfg, force=False):
        raise TypeError("kaboom")

    monkeypatch.setattr(cli, "build_datasets", boom)
    assert cli.main(["--config", str(config)]) == 1
    assert "unexpected TypeError: kaboom" in capsys.readouterr().err

    def interrupted(cfg, force=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_datasets", interrupted)
    assert cli.main(["--config", str(config)]) == 130


@pytest.mark.unit
def test_module_exports(built):
    assert ds.FORMAT_VERSION == 2
    assert callable(ds.build_datasets) and callable(ds.dataset_paths)
    assert {"graph_attributes_sha256", "weather_fingerprint", "reports_fingerprint"} <= set(REQUIRED_PAYLOAD_KEYS)
