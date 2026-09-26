"""Tests for the shared foundation utilities."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.data_pipeline.graph_io import GraphFormatError, graph_to_arrays, load_graph, save_graph
from src.utils.config import ConfigError, config_hash, deep_merge, get_section, is_offline, load_config, resolve_path
from src.utils.geo import bbox_area_km2, haversine_m, lonlat_to_local_xy, point_in_bbox, validate_bbox
from src.utils.http import NetworkUnavailable, get_json
from src.utils.runtime import atomic_write_text, resolve_device


@pytest.mark.unit
def test_load_config_defaults_and_offline_env(cfg):
    assert cfg["project"]["name"] == "Namma-Flow"
    assert is_offline(cfg)
    assert resolve_path(cfg, "graph_file").name == "test.graphml"


@pytest.mark.unit
def test_config_rejects_bad_bbox_and_feature_mismatch():
    with pytest.raises(ConfigError):
        load_config(overrides={"region": {"bbox": [77.7, 12.9, 77.6, 12.95]}})
    with pytest.raises(ConfigError):
        load_config(overrides={"model": {"node_in_dim": 5}})
    with pytest.raises(ConfigError):
        load_config(path="does/not/exist.yaml")


def _minimal_config(tmp_path: Path, **sections) -> Path:
    import yaml

    body = {"project": {"name": "x", "seed": 42}, "paths": {"graph_file": str(tmp_path / "g.graphml")}, **sections}
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


@pytest.mark.unit
def test_node_in_dim_is_checked_against_the_effective_dataset_defaults(tmp_path):
    """F4-02: omitted dataset keys take the module defaults (5 static features, 4 windows) in the node_in_dim check."""
    from src.data_pipeline.dataset_config import DEFAULTS
    from src.utils.config import effective_dataset_section

    minimal = _minimal_config(tmp_path, dataset={"val_years": [2021]}, model={"node_in_dim": 10})
    assert load_config(minimal)["model"]["node_in_dim"] == 10, "a partial dataset section with the right width loads"
    partial = load_config(minimal, overrides={"dataset": {"rolling_windows_h": [3, 6]}, "model": {"node_in_dim": 8}})
    assert effective_dataset_section(partial)["static_features"] == DEFAULTS["static_features"]
    with pytest.raises(ConfigError, match=r"node_in_dim=10 but the dataset features imply 8"):
        load_config(minimal, overrides={"dataset": {"rolling_windows_h": [3, 6]}})
    no_dataset = _minimal_config(tmp_path, model={"node_in_dim": 5})
    with pytest.raises(ConfigError, match=r"node_in_dim=5 but the dataset features imply 10"):
        load_config(no_dataset)
    with pytest.raises(ConfigError, match="whole number"):
        load_config(minimal, overrides={"model": {"node_in_dim": "ten"}})
    with pytest.raises(ConfigError, match="static_features"):
        load_config(minimal, overrides={"dataset": {"static_features": "elevation"}})
    assert effective_dataset_section({}) == DEFAULTS, "no dataset section = the dataset module's defaults"
    with pytest.raises(ConfigError, match="'model' must be a mapping"):
        load_config(minimal, overrides={"model": "gatv2"})


@pytest.mark.unit
@pytest.mark.parametrize(("dataset", "message"), [
    ({"lookback_hours": 12}, r"lookback_hours \(12\) must cover the longest rolling window \(24 h\)"),
    ({"seq_len": 3}, r"seq_len \(3\) must be > warmup_steps \(4\)"),
    ({"warmup_steps": 16}, r"seq_len \(16\) must be > warmup_steps \(16\)"),
    ({"seq_len": "abc"}, "dataset.seq_len must be a whole number"),
    ({"seq_len": True}, "dataset.seq_len must be a whole number"),
    ({"lookback_hours": 2.5}, "dataset.lookback_hours must be a whole number"),
    ({"rolling_windows_h": "24"}, "rolling_windows_h"),
])
def test_window_checks_use_the_effective_dataset_defaults(tmp_path, dataset, message):
    """F4-02: the seq_len / warmup / lookback checks also see the defaults of keys the YAML omits."""
    minimal = _minimal_config(tmp_path, dataset=dataset)
    with pytest.raises(ConfigError, match=message):
        load_config(minimal)


@pytest.mark.unit
def test_window_checks_accept_whole_floats_and_consistent_partial_sections(tmp_path):
    minimal = _minimal_config(tmp_path, dataset={"seq_len": 16.0, "lookback_hours": 48, "rolling_windows_h": [3, 48]},
                              model={"node_in_dim": 8})
    assert load_config(minimal)["dataset"]["lookback_hours"] == 48


@pytest.mark.unit
def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 5}})
    assert merged == {"a": {"b": 5, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
    assert get_section({"x": {"k": 1}}, "x", {"k": 0, "j": 2}) == {"k": 1, "j": 2}
    assert config_hash({"a": 1}, ("a",)) == config_hash({"a": 1}, ("a",))


@pytest.mark.unit
def test_get_section_replaces_listed_mapping_keys():
    """R1-07: keys named in ``replace`` take the user value verbatim instead of deep-merging."""
    defaults = {"tags": {"waterway": ["drain"], "natural": ["water"]}, "grid": {"rows": 2, "cols": 3}}
    user = {"x": {"tags": {"waterway": ["canal"]}, "grid": {"rows": 5}}}
    merged = get_section(user, "x", defaults)
    assert merged["tags"] == {"waterway": ["canal"], "natural": ["water"]}
    replaced = get_section(user, "x", defaults, replace=("tags",))
    assert replaced["tags"] == {"waterway": ["canal"]}
    assert replaced["grid"] == {"rows": 5, "cols": 3}, "unlisted mappings still merge"
    assert get_section({}, "x", defaults, replace=("tags",))["tags"] == defaults["tags"]
    replaced["tags"]["waterway"].append("drain")
    assert user["x"]["tags"] == {"waterway": ["canal"]}, "never aliases the caller's config"
    narrowed = load_config(overrides={"drains": {"osm_tags": {"waterway": ["drain"]}}})
    assert narrowed["drains"]["osm_tags"] == {"waterway": ["drain"]}, "overrides replace the YAML tag filter"
    assert load_config()["drains"]["osm_tags"]["natural"] == ["water"]


@pytest.mark.unit
def test_geo_helpers():
    assert validate_bbox([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        validate_bbox([1, 2, 3])
    with pytest.raises(ValueError):
        validate_bbox([1, 2, float("nan"), 4])
    assert 15 < bbox_area_km2([77.655, 12.915, 77.700, 12.950]) < 25
    d = haversine_m(77.6, 12.9, 77.6, 13.0)
    assert 11_000 < d < 11_200
    x, y = lonlat_to_local_xy([77.6, 77.61], [12.9, 12.9], origin=(77.6, 12.9))
    assert abs(x[1] - 1084) < 5 and abs(y[1]) < 1e-6
    assert point_in_bbox([77.66, 10.0], [12.92, 12.92], [77.655, 12.915, 77.700, 12.950]).tolist() == [True, False]


@pytest.mark.unit
def test_http_offline_refuses():
    with pytest.raises(NetworkUnavailable):
        get_json("https://example.com", offline=True)


@pytest.mark.unit
def test_graph_roundtrip_and_arrays(tmp_path, grid_graph):
    path = save_graph(grid_graph, tmp_path / "g.graphml")
    loaded = load_graph(path)
    assert loaded.number_of_nodes() == grid_graph.number_of_nodes()
    assert loaded.number_of_edges() == grid_graph.number_of_edges()
    arrays = graph_to_arrays(loaded)
    assert arrays.edge_index.shape == (2, grid_graph.number_of_edges())
    assert {"elevation", "dist_to_drain_m", "relative_elevation"} <= set(arrays.node_attrs)
    assert arrays.node_matrix(["elevation"]).shape == (36, 1)
    assert arrays.edge_matrix(["length", "grade"]).dtype == np.float32
    assert arrays.signature() == graph_to_arrays(grid_graph).signature()
    with pytest.raises(KeyError):
        arrays.node_matrix(["nope"])


@pytest.mark.unit
def test_graph_io_errors(tmp_path):
    with pytest.raises(GraphFormatError):
        save_graph(nx.DiGraph(), tmp_path / "empty.graphml")
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "missing.graphml")
    bad = tmp_path / "bad.graphml"
    bad.write_text("not xml")
    with pytest.raises(GraphFormatError):
        load_graph(bad)


@pytest.mark.unit
def test_runtime_helpers(tmp_path):
    p = atomic_write_text(tmp_path / "sub/a.txt", "hello")
    assert p.read_text() == "hello"
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


@pytest.mark.unit
def test_graph_load_save_roundtrip_twice(tmp_path, grid_graph):
    first = save_graph(grid_graph, tmp_path / "a.graphml")
    second = save_graph(load_graph(first), tmp_path / "b.graphml")
    reloaded = load_graph(second)
    assert "node_default" not in reloaded.graph and "edge_default" not in reloaded.graph
    assert reloaded.number_of_edges() == grid_graph.number_of_edges()


@pytest.mark.unit
def test_atomic_write_uses_umask_permissions(tmp_path):
    import os
    import stat

    path = atomic_write_text(tmp_path / "perm.txt", "x")
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


# --------------------------------------------------------------------------- packaging (R5-06, R5-11, F5-03/04/06)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Lowest releases providing the APIs the code uses (see the comments in requirements.txt). osmnx 2.1.1 is the
# first release that looks an Overpass request up in its cache before calling _http._config_dns (F5-04).
REQUIRED_FLOORS = {"osmnx": "2.1.1", "geopandas": "1.0.0", "streamlit": "1.52.0", "altair": "5.0.0", "pandas": "2.2.0"}
# Opt-in: fail (instead of skip) when the installed packages drift from constraints.txt (F5-06; make check-env).
STRICT_ENV_VAR = "NAMMA_FLOW_STRICT_ENV"


def _requirement_floors(path: Path) -> dict[str, str]:
    floors = {}
    for line in path.read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec:
            name, _, floor = spec.partition(">=")
            floors[name.strip().lower()] = floor.strip()
    return floors


def _constraint_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec:
            name, _, pinned = spec.partition("==")
            assert pinned, f"constraints.txt entry {spec!r} is not an exact pin"
            pins[name.strip().lower()] = pinned.strip()
    return pins


def _pin_drift(pins: dict[str, str], lookup) -> dict[str, tuple[str, str]]:
    """``{name: (installed, pinned)}`` for installed packages whose version differs from the pin."""
    from importlib.metadata import PackageNotFoundError

    from packaging.version import Version

    drift = {}
    for name, pinned in pins.items():
        try:
            installed = lookup(name)
        except PackageNotFoundError:
            continue  # optional / platform-specific package not installed here
        if Version(installed) != Version(pinned):
            drift[name] = (installed, pinned)
    return drift


def _enforce_pins(drift: dict[str, tuple[str, str]], strict: bool) -> None:
    """Fail on drift only when asked (``NAMMA_FLOW_STRICT_ENV=1``); otherwise report it as a skip."""
    if not drift:
        return
    listing = ", ".join(f"{name} {got} (pinned {want})" for name, (got, want) in sorted(drift.items()))
    message = f"installed packages differ from constraints.txt: {listing}"
    if strict:
        pytest.fail(message)
    pytest.skip(f"{message}; set {STRICT_ENV_VAR}=1 (make check-env) to enforce the pins")


def _pyproject() -> dict:
    import tomllib

    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def _python_floor():
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec = SpecifierSet(_pyproject()["project"]["requires-python"])
    floors = [Version(s.version) for s in spec if s.operator == ">="]
    assert len(floors) == 1, f"requires-python {spec} must have exactly one '>=' floor"
    return floors[0]


@pytest.mark.unit
def test_requirement_floors_provide_every_api_used():
    """R5-06 / F5-04: floors must not admit osmnx < 2.1.1 (no offline cache replay), geopandas 0.x, old Streamlit."""
    from packaging.version import Version

    floors = _requirement_floors(PROJECT_ROOT / "requirements.txt")
    for name, needed in REQUIRED_FLOORS.items():
        assert name in floors, f"{name} missing from requirements.txt"
        assert Version(floors[name]) >= Version(needed), f"{name}>={floors[name]} admits versions without the used APIs"


@pytest.mark.unit
def test_constraints_are_exact_pins_above_every_floor():
    """R5-06 / F5-06: structural check, valid in every environment: every requirement pinned, pins >= floors."""
    from packaging.version import Version

    pins = _constraint_pins(PROJECT_ROOT / "constraints.txt")
    for name, floor in _requirement_floors(PROJECT_ROOT / "requirements.txt").items():
        assert name in pins, f"{name} (requirements.txt) is not pinned in constraints.txt"
        assert Version(pins[name]) >= Version(floor), f"{name}=={pins[name]} is below its floor {floor}"


@pytest.mark.unit
def test_installed_packages_match_constraints():
    """F5-06: drift from the pins is reported (skip); it fails only with NAMMA_FLOW_STRICT_ENV=1 (make check-env)."""
    import os
    from importlib.metadata import version

    from src.utils.config import TRUTHY

    drift = _pin_drift(_constraint_pins(PROJECT_ROOT / "constraints.txt"), version)
    _enforce_pins(drift, strict=os.environ.get(STRICT_ENV_VAR, "").strip().lower() in TRUTHY)


@pytest.mark.unit
def test_pin_drift_skips_by_default_and_fails_only_when_strict():
    """F5-06 regression: an off-pin environment (e.g. the no-constraints install) must not turn the suite red."""
    from importlib.metadata import PackageNotFoundError

    installed = {"requests": "2.33.1", "numpy": "2.5.3"}

    def lookup(name: str) -> str:
        if name not in installed:
            raise PackageNotFoundError(name)
        return installed[name]

    drift = _pin_drift({"requests": "2.34.2", "numpy": "2.5.3", "pyarrow": "25.0.1"}, lookup)
    assert drift == {"requests": ("2.33.1", "2.34.2")}, "only installed, off-pin packages count as drift"
    with pytest.raises(pytest.skip.Exception, match="requests 2.33.1 \\(pinned 2.34.2\\)"):
        _enforce_pins(drift, strict=False)
    with pytest.raises(pytest.fail.Exception, match="requests 2.33.1"):
        _enforce_pins(drift, strict=True)
    _enforce_pins({}, strict=True)  # an exactly pinned environment passes in both modes


@pytest.mark.unit
def test_python_floor_matches_the_pins_and_the_makefile():
    """F5-03: requires-python admits no Python the pinned stack cannot install on; Makefile uses the same."""
    import re
    from importlib.metadata import PackageNotFoundError, metadata

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    floor = _python_floor()
    assert floor >= Version("3.12"), "numpy 2.5 / scipy 1.18 / networkx 3.7 / pyproj 3.8 / rasterio 1.5 need 3.12"
    make_floor = re.search(r"^MIN_PYTHON\s*:?=\s*(\S+)\s*$", (PROJECT_ROOT / "Makefile").read_text(), re.M)
    assert make_floor and Version(make_floor.group(1)) == floor, "Makefile MIN_PYTHON must equal requires-python"
    checked = 0
    for name, pinned in _constraint_pins(PROJECT_ROOT / "constraints.txt").items():
        try:
            meta = metadata(name)
        except PackageNotFoundError:
            continue
        if Version(meta["Version"]) != Version(pinned) or not meta.get("Requires-Python"):
            continue  # only the pinned release's own metadata says which Pythons the pin supports
        checked += 1
        assert SpecifierSet(meta["Requires-Python"]).contains(str(floor)), \
            f"{name}=={pinned} needs Python {meta['Requires-Python']}, but requires-python admits {floor}"
    if checked == 0:  # pragma: no cover - no package installed at its pin
        pytest.skip("no package is installed at its constraints.txt pin")


@pytest.mark.unit
def test_installed_osmnx_reads_its_cache_before_dns_setup():
    """F5-04: network._cache_only relies on osmnx answering from the cache before calling _http._config_dns."""
    import ast
    import importlib.util

    spec = importlib.util.find_spec("osmnx")
    assert spec is not None and spec.origin, "osmnx (requirements.txt) is not installed"
    source = (Path(spec.origin).parent / "_overpass.py").read_text(encoding="utf-8")
    function = next((node for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.FunctionDef) and node.name == "_overpass_request"), None)
    body = ast.get_source_segment(source, function) if function is not None else ""
    if "_retrieve_from_cache" not in body or "_config_dns" not in body:  # pragma: no cover - osmnx internals changed
        pytest.skip("osmnx changed its Overpass request internals; re-check the osmnx floor and network._cache_only")
    assert body.index("_retrieve_from_cache") < body.index("_config_dns"), \
        "this osmnx calls _config_dns before its cache lookup: offline replay fails (need osmnx>=2.1.1)"


@pytest.mark.unit
def test_gitignore_keeps_placeholders_and_ignores_downloaded_data(tmp_path):
    """R5-11: the weather .meta.json sidecar is ignored with its CSV; artifacts/.gitkeep survives."""
    import shutil
    import subprocess

    placeholders = ["artifacts", "data/raw/dem", "data/raw/weather", "data/raw/labels", "data/interim", "data/processed"]
    for folder in placeholders:
        assert (PROJECT_ROOT / folder / ".gitkeep").is_file(), f"{folder}/.gitkeep is missing"
    if shutil.which("git") is None:  # pragma: no cover
        pytest.skip("git is not installed")
    shutil.copy(PROJECT_ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    ignored = ["data/raw/weather/open_meteo_hourly.csv", "data/raw/weather/open_meteo_hourly.meta.json",
               "data/raw/dem/N12E077.hgt.gz", "data/raw/dem/srtm_1.tif", "data/interim/bellandur_osm.graphml",
               "data/processed/train_dataset.pt", "artifacts/checkpoints/best.pt", "artifacts/reports/metrics.json"]
    kept = [f"{folder}/.gitkeep" for folder in placeholders] + ["data/raw/labels/flood_reports.csv"]
    for rel in ignored + kept:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    result = subprocess.run(["git", "check-ignore", *ignored, *kept], cwd=tmp_path, capture_output=True, text=True)
    assert sorted(result.stdout.split()) == sorted(ignored)
