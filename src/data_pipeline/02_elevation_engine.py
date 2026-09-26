"""Stage 02 - elevation & drainage engine.

Loads the road graph written by stage 01 (``paths.graph_file``), adds node elevations
(local DEM -> SRTM -> Open-Meteo -> synthetic), relative elevation, distance to the
nearest drain/lake, clipped edge grades and flow accumulation / sinks, then saves the
graph back in place and prints a summary. All logic lives in
:mod:`src.data_pipeline.elevation` / :mod:`src.data_pipeline.drains`; this script only
parses flags and reports.

Usage (from the project root)::

    python src/data_pipeline/02_elevation_engine.py [--config PATH] [--offline]
                                                    [--refresh-dem] [--refresh-drains] [--allow-synthetic]

Offline, ``--refresh-dem`` is ignored with a WARNING (the cached SRTM clip/tiles are reused).
The stage refuses (exit 1, graph untouched) to replace real SRTM elevations or OSM drains with
synthetic fallbacks unless ``--allow-synthetic`` is given.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import argparse  # noqa: E402
from typing import Any, Mapping  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from src.data_pipeline.elevation import ElevationError, run_elevation_stage  # noqa: E402
from src.data_pipeline.graph_io import GraphFormatError, graph_to_arrays  # noqa: E402
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.http import NetworkUnavailable  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("data_pipeline.02_elevation_engine")
HANDLED_ERRORS = (
    ConfigError, FileNotFoundError, GraphFormatError, ElevationError, NetworkUnavailable, ValueError, OSError,
)
ENRICHED_NODE_ATTRS = ("elevation", "relative_elevation", "dist_to_drain_m", "flow_accumulation", "is_sink")


def summarize_enrichment(G: nx.DiGraph) -> dict[str, Any]:
    """Statistics of an enriched graph (elevation, voids, drains, sinks, grades)."""
    arrays = graph_to_arrays(G)
    missing = [k for k in ENRICHED_NODE_ATTRS if k not in arrays.node_attrs]
    if missing:
        raise ValueError(f"Graph is not enriched (missing node attributes {missing}); run enrich_graph first")
    attrs, meta = arrays.node_attrs, G.graph
    elev, tpi, dist = attrs["elevation"], attrs["relative_elevation"], attrs["dist_to_drain_m"]
    sinks = attrs["is_sink"] > 0.5
    grade = arrays.edge_attrs.get("grade", np.zeros(0))
    return {
        "nodes": arrays.num_nodes,
        "edges": arrays.num_edges,
        "elevation_source": str(meta.get("elevation_source", "unknown")),
        "elevation_dem": str(meta.get("elevation_dem", "")),
        "elevation_min_m": float(elev.min()),
        "elevation_max_m": float(elev.max()),
        "elevation_mean_m": float(elev.mean()),
        "elevation_std_m": float(elev.std()),
        "elevation_voids_filled": int(meta.get("elevation_voids_filled", 0)),
        "elevation_void_pct": 100.0 * float(meta.get("elevation_void_fraction", 0.0)),
        "relative_elevation_min_m": float(tpi.min()),
        "relative_elevation_max_m": float(tpi.max()),
        "drain_source": str(meta.get("drain_source", "unknown")),
        "drain_feature_count": int(meta.get("drain_feature_count", 0)),
        "dist_to_drain_median_m": float(np.median(dist)),
        "dist_to_drain_max_m": float(dist.max()),
        "n_sinks": int(sinks.sum()),
        "sink_pct": 100.0 * float(sinks.mean()),
        "max_flow_accumulation": float(attrs["flow_accumulation"].max()),
        "max_abs_grade": float(np.abs(grade).max()) if grade.size else 0.0,
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """Human-readable multi-line summary."""
    dem = f" ({summary['elevation_dem']})" if summary.get("elevation_dem") else ""
    return "\n".join(
        [
            "Namma-Flow stage 02 - elevation & drainage engine",
            f"  Graph             : {summary.get('graph_file', '-')} "
            f"({summary['nodes']} nodes, {summary['edges']} edges)",
            f"  Elevation source  : {summary['elevation_source']}{dem}",
            f"  Elevation (m)     : min {summary['elevation_min_m']:.1f}  max {summary['elevation_max_m']:.1f}  "
            f"mean {summary['elevation_mean_m']:.1f}  std {summary['elevation_std_m']:.1f}",
            f"  Voids filled      : {summary['elevation_voids_filled']} ({summary['elevation_void_pct']:.1f} %)",
            f"  Relative elev (m) : {summary['relative_elevation_min_m']:.1f} .. "
            f"{summary['relative_elevation_max_m']:.1f}",
            f"  Drain source      : {summary['drain_source']} ({summary['drain_feature_count']} features); "
            f"distance median {summary['dist_to_drain_median_m']:.0f} m, max {summary['dist_to_drain_max_m']:.0f} m",
            f"  Sinks             : {summary['n_sinks']} ({summary['sink_pct']:.1f} %); "
            f"max flow accumulation {summary['max_flow_accumulation']:.0f}",
            f"  Max |grade|       : {summary['max_abs_grade']:.3f}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Namma-Flow stage 02: elevation & drainage enrichment of the road graph"
    )
    parser.add_argument(
        "--config", default=None, help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)"
    )
    parser.add_argument("--offline", action="store_true", help="never touch the network (use caches / fallbacks)")
    parser.add_argument(
        "--refresh-dem", action="store_true", help="ignore cached SRTM GeoTIFF/tiles and download again"
    )
    parser.add_argument(
        "--refresh-drains", action="store_true", help="ignore the cached waterways GeoJSON and query OSM again"
    )
    parser.add_argument(
        "--allow-synthetic", action="store_true",
        help="accept synthetic elevation/drain fallbacks even when the graph currently holds real SRTM/OSM data",
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.offline:
        overrides["project"] = {"offline": True}
    if args.refresh_dem:
        overrides["elevation"] = {"refresh_dem": True}
    if args.refresh_drains:
        overrides["drains"] = {"refresh": True}
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=_overrides(args))
        graph, path = run_elevation_stage(cfg, allow_synthetic=args.allow_synthetic)
        summary = {**summarize_enrichment(graph), "graph_file": str(path)}
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except HANDLED_ERRORS as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # last line of defence: one clear line instead of a traceback
        LOGGER.debug("Unexpected failure in stage 02", exc_info=True)
        print(
            f"ERROR: unexpected {type(exc).__name__}: {exc} (set NAMMA_FLOW_LOG_LEVEL=DEBUG for the traceback)",
            file=sys.stderr,
        )
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
