"""Stage-04 graph preparation and the dataset config hash (F1-02, F1-03).

* F1-02: stage 04 completes a graph that lacks derived attributes from its stored elevations
  and never overwrites real SRTM elevations with the synthetic formula.
* F1-03: the dataset config hash covers exactly the keys stage 04 applies itself (stable across
  processes, documented in ``dataset_config``); changed elevation / drains settings are caught
  through the graph's ``enrichment_config_hash`` and re-enrich the graph BEFORE the reuse check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data_pipeline import elevation
from src.data_pipeline.dataset import DatasetError, build_datasets, load_payload
from src.data_pipeline.dataset_config import HASH_KEYS, HASH_SECTIONS, dataset_config_hash, dataset_hash_inputs
from src.data_pipeline.elevation_settings import ENRICHMENT_HASH_ATTR, enrichment_config_hash
from src.data_pipeline.graph_io import graph_to_arrays, load_graph
from src.data_pipeline.terrain import relative_elevation
from src.utils.config import deep_merge
from tests.conftest import make_grid_graph
from tests.dataset_fixtures import fake_simulator, log  # noqa: F401 - pytest fixtures
from tests.dataset_fixtures import setup_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Golden value of the hash of MINIMAL_CFG: pins the hash definition (a change must be deliberate).
MINIMAL_CFG = {"project": {"timezone": "Asia/Kolkata"}, "dataset": {"seq_len": 16}}
MINIMAL_HASH = "b41c934a39fb337d"


def _srtm_graph(drop: tuple[str, ...] = ()):
    G = make_grid_graph(5, 5)
    G.graph.update(source="osm_bbox", elevation_source="srtm", drain_source="osm")
    for _, data in G.nodes(data=True):
        data["elevation"] = float(data["elevation"]) + 3.5
        for attr in drop:
            data.pop(attr, None)
    return G


# --------------------------------------------------------------------------- F1-03: the hash
@pytest.mark.unit
def test_dataset_hash_ignores_fetch_only_and_upstream_keys(cfg):
    base = dataset_config_hash(cfg)
    for change in ({"weather": {"forecast_days": 7}}, {"weather": {"timeout_s": 120}},
                   {"weather": {"forecast_url": "https://x.test"}}, {"weather": {"max_retries": 9}},
                   {"network": {"request_timeout_s": 300}}, {"network": {"overpass_urls": ["https://y.test/api"]}},
                   {"drains": {"request_timeout_s": 300}}, {"elevation": {"request_timeout_s": 5}},
                   {"elevation": {"tpi_radius_m": 600.0}}, {"drains": {"exclude_water_values": []}},
                   {"model": {"hidden_dim": 8}}, {"app": {"top_k": 3}}, {"inference": {"field_members": 4}}):
        assert dataset_config_hash(deep_merge(cfg, change)) == base, change


@pytest.mark.unit
def test_dataset_hash_tracks_every_build_shaping_key(cfg):
    base = dataset_config_hash(cfg)
    for change in ({"rainfall_field": {"seed": 1}}, {"hydrology": {"infiltration_mm_h": 9.0}},
                   {"labels": {"source": "hybrid"}}, {"dataset": {"seq_len": 12}},
                   {"weather": {"start_date": "2020-01-01"}}, {"weather": {"end_date": "2023-12-31"}},
                   {"weather": {"bias_correction_factor": 1.2}}, {"weather": {"max_precip_mm_h": 99.0}},
                   {"weather": {"max_fill_gap_hours": 3}}, {"weather": {"synthetic_scale": "point"}},
                   {"weather": {"latitude": 12.9, "longitude": 77.6}}, {"elevation": {"max_abs_grade": 0.2}},
                   {"region": {"bbox": [77.60, 12.90, 77.70, 12.99]}}, {"project": {"timezone": "UTC"}}):
        assert dataset_config_hash(deep_merge(cfg, change)) != base, change
    inputs = dataset_hash_inputs(cfg)
    assert set(inputs) == set(HASH_SECTIONS) | {f"{name}.keys" for name in HASH_KEYS}
    assert HASH_SECTIONS == ("rainfall_field", "hydrology", "labels", "dataset")


@pytest.mark.unit
def test_dataset_hash_uses_effective_values_and_is_stable_across_processes():
    assert dataset_config_hash(MINIMAL_CFG) == dataset_config_hash(
        deep_merge(MINIMAL_CFG, {"weather": {"bias_correction_factor": 1.0}, "elevation": {"max_abs_grade": 0.3}})
    ), "writing a default value explicitly must not change the hash"
    code = ("from tests.test_dataset_graph import MINIMAL_CFG; "
            "from src.data_pipeline.dataset_config import dataset_config_hash; print(dataset_config_hash(MINIMAL_CFG))")
    values = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True,
                             text=True, timeout=60, check=True)
        values.add(out.stdout.strip())
    assert values == {dataset_config_hash(MINIMAL_CFG)}
    assert dataset_config_hash(MINIMAL_CFG) == MINIMAL_HASH


# --------------------------------------------------------------------------- F1-02: stage 04
@pytest.mark.integration
def test_stage04_recomputes_derived_attributes_and_keeps_srtm(cfg, fake_simulator, monkeypatch, log):
    original = _srtm_graph(drop=("is_sink", "flow_accumulation"))
    out = setup_config(cfg, graph=original)
    monkeypatch.setattr(elevation, "enrich_graph", lambda *a, **k: pytest.fail("module B must not run"))
    summary = build_datasets(out)
    saved = load_graph(out["paths"]["graph_file"])
    assert saved.graph["elevation_source"] == "srtm" and summary["graph"]["elevation_source"] == "srtm"
    for node, data in original.nodes(data=True):
        assert saved.nodes[node]["elevation"] == pytest.approx(data["elevation"])
    assert "is_sink" in graph_to_arrays(saved).node_attrs and "stored elevations" in log.text


@pytest.mark.integration
def test_stage04_refuses_to_replace_srtm_elevations_with_synthetic(cfg, fake_simulator):
    out = setup_config(cfg, graph=_srtm_graph(drop=("dist_to_drain_m",)))
    before = Path(out["paths"]["graph_file"]).read_bytes()
    with pytest.raises(DatasetError, match="elevation_source srtm -> synthetic"):
        build_datasets(out)
    assert Path(out["paths"]["graph_file"]).read_bytes() == before
    assert not Path(out["paths"]["train_dataset"]).exists()


# --------------------------------------------------------------------------- F1-03: stage 04 drift
@pytest.mark.integration
def test_stage04_reenriches_a_graph_enriched_with_other_settings_before_reuse(cfg, fake_simulator, log):
    base = setup_config(cfg, graph=elevation.enrich_graph(make_grid_graph(5, 5), cfg))
    assert load_graph(base["paths"]["graph_file"]).graph[ENRICHMENT_HASH_ATTR] == enrichment_config_hash(base)
    first = build_datasets(base)
    assert build_datasets(base)["reused"] is True
    wider = deep_merge(base, {"elevation": {"tpi_radius_m": 900.0}})
    assert dataset_config_hash(wider) == dataset_config_hash(base), "elevation settings are not in the dataset hash"
    summary = build_datasets(wider)
    assert "enrichment config hash" in log.text
    assert summary["reused"] is False and any("road graph attributes changed" in r for r in summary["rebuild_reasons"])
    saved = load_graph(wider["paths"]["graph_file"])
    assert saved.graph[ENRICHMENT_HASH_ATTR] == enrichment_config_hash(wider)
    arrays = graph_to_arrays(saved)
    expected = relative_elevation(arrays.lon, arrays.lat, arrays.node_attrs["elevation"], 900.0)
    np.testing.assert_allclose(arrays.node_attrs["relative_elevation"], expected, atol=1e-6)
    train = load_payload(wider["paths"]["train_dataset"])
    assert train["graph_attributes_sha256"] != first["fingerprints"]["graph_attributes_sha256"]
    assert train["graph_provenance"][ENRICHMENT_HASH_ATTR] == enrichment_config_hash(wider)
    assert build_datasets(wider)["reused"] is True
