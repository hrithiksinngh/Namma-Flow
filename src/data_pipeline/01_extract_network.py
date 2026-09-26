"""Stage 01 CLI — extract, clean and enrich the OSM road graph.

Usage (from the project root)::

    python src/data_pipeline/01_extract_network.py [--config PATH] [--offline] [--force] [--allow-synthetic]

Fallback chain: cached graph → OSM place query → OSM bbox (Overpass mirrors; offline:
replayed from the osmnx cache) → synthetic street grid. An existing OpenStreetMap graph is
never replaced by the synthetic grid unless ``--allow-synthetic`` is given (exit 1 instead),
and no re-enrichment replaces real SRTM / OSM-drain inputs with synthetic fallbacks; a cached
synthetic grid is rebuilt automatically when online. A busy Overpass mirror (HTTP 429 / 504)
is given up after ``network.overpass_max_attempts`` and the next mirror is tried. The summary
reports the OpenStreetMap snapshot (Overpass ``timestamp_osm_base``) the graph was built from;
``--force --offline`` rebuilds the same graph from the osmnx cache and records it.
All logic lives in :mod:`src.data_pipeline.network`; this
file only parses arguments, maps failures to exit codes and prints the summary.
Exit codes: 0 success, 1 handled failure (one-line ``ERROR:`` on stderr), 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from src.data_pipeline import network  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("data_pipeline.01_extract_network")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="01_extract_network.py",
        description="Extract, clean and enrich the OSM road graph (falls back to a synthetic grid offline).",
    )
    parser.add_argument("--config", default=None,
                        help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)")
    parser.add_argument("--offline", action="store_true",
                        help="never touch the network (rebuild from the osmnx cache, else a synthetic grid)")
    parser.add_argument("--force", action="store_true", help="ignore the cached graph file and rebuild it")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="allow replacing an existing OpenStreetMap graph with the synthetic grid")
    return parser


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())


def main(argv: Sequence[str] | None = None) -> int:
    """Run stage 01 and return the process exit code."""
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        cfg = load_config(args.config, overrides={"project": {"offline": True}} if args.offline else None)
        G = network.extract_network(cfg, force=args.force, allow_synthetic=args.allow_synthetic)
        summary = network.summarize_graph(G, resolve_path(cfg, "graph_file", network.DEFAULT_GRAPH_FILE))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:  # ConfigError, GraphTooSmallError, NetworkStageError, ...
        print(f"ERROR: {_one_line(exc)}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - last resort: one clear line instead of a traceback
        LOGGER.debug("Unexpected failure in stage 01", exc_info=True)
        print(f"ERROR: unexpected {type(exc).__name__}: {_one_line(exc)} "
              "(set NAMMA_FLOW_LOG_LEVEL=DEBUG for the traceback)", file=sys.stderr)
        return 1
    summary["elapsed_s"] = time.perf_counter() - started
    print(network.format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
