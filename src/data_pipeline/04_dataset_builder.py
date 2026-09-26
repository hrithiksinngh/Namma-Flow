"""Stage 04 CLI — build the training / validation datasets.

Usage (from the project root)::

    python src/data_pipeline/04_dataset_builder.py [--config PATH] [--offline] [--force]

Runs right after stage 01: missing weather is fetched (or synthesised offline) and a
missing graph is extracted. Writes the train, validation (``dataset.val_years``) and held-out
test (``dataset.test_years``; optional) datasets. Existing datasets are reused only when the
configuration, the road graph (topology and attributes), the weather record and the flood
reports are unchanged; otherwise the summary says which input changed (``--force`` always
rebuilds). Logic lives in :mod:`src.data_pipeline.dataset`; this script only parses
arguments and prints a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import argparse  # noqa: E402
from typing import Any, Mapping  # noqa: E402

from src.data_pipeline.dataset import DatasetError, build_datasets  # noqa: E402
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("data_pipeline.04_dataset_builder")
EXIT_OK, EXIT_FAILED, EXIT_INTERRUPTED = 0, 1, 130
INTERRUPTED_MESSAGE = (
    "Interrupted. The split files are committed together and the dataset manifest is written last: the "
    "previous datasets are still in place, or an incomplete commit is detected (no matching manifest) and "
    "rebuilt by the next run; training refuses splits from different builds."
)
_HANDLED = (ConfigError, DatasetError, FileNotFoundError, ValueError, RuntimeError, OSError, ImportError)


def years_label(years: list[int]) -> str:
    """Compact year list: ``[2018, 2019, 2020, 2023]`` → ``"2018-2020, 2023"``."""
    runs: list[list[int]] = []
    for year in sorted(set(int(y) for y in years)):
        if runs and year == runs[-1][-1] + 1:
            runs[-1].append(year)
        else:
            runs.append([year])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs) or "-"


SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test (held out)"}


def _split_line(split: str, stats: Mapping[str, Any]) -> str:
    span = years_label(stats.get("years") or [])
    return (f"  {SPLIT_LABELS.get(split, split.capitalize()):<16}: {stats['n_windows']} windows ({span}), "
            f"{stats['n_windows_with_flood']} with floods ({100 * stats['flood_window_rate']:.1f} %), node-step flood "
            f"rate {100 * stats['pos_rate']:.3f} %, {stats['n_hours']} stored hours, "
            f"{stats['size_bytes'] / 1e6:.1f} MB -> {stats['path']}")


def _status(summary: Mapping[str, Any]) -> str:
    if summary.get("reused"):
        return ("reused existing datasets (config, graph, weather record and flood reports unchanged; use --force to "
                "rebuild)")
    reasons = summary.get("rebuild_reasons") or []
    because = f" - {'; '.join(reasons)}" if reasons else ""
    return f"built in {summary.get('build_time_s', 0.0):.1f} s{because}"


def _fingerprint_line(summary: Mapping[str, Any]) -> str:
    prints = summary.get("fingerprints") or {}
    reports = prints.get("reports_fingerprint")
    return (f"  Fingerprints    : graph {prints.get('graph_signature', '?')} (topology) / "
            f"{prints.get('graph_attributes_sha256', '?')} (attributes), "
            f"weather {prints.get('weather_fingerprint', '?')}, "
            f"flood reports {reports if reports is not None else 'n/a (simulated labels)'}")


def _held_out_line(summary: Mapping[str, Any]) -> str:
    test = summary.get("test_years") or []
    test_text = years_label(test) if test else "none (dataset.test_years is empty)"
    return f"  Held-out years  : validation {years_label(summary.get('val_years') or [])}; test {test_text}"


def format_summary(summary: Mapping[str, Any]) -> str:
    """Human-readable summary of :func:`build_datasets` output (windows, flood rates, sizes, fingerprints)."""
    graph = summary.get("graph", {})
    snapshot = graph.get("osm_base_utc")
    snapshot_text = f"; OSM snapshot {snapshot}" if snapshot and snapshot != "unknown" else ""
    lines = [
        "Namma-Flow stage 04 - dataset builder",
        f"  Status          : {_status(summary)}",
        f"  Graph           : {summary['n_nodes']} nodes, {summary['n_edges']} edges ({graph.get('source', '?')}; "
        f"elevation {graph.get('elevation_source', '?')}; drains {graph.get('drain_source', '?')}{snapshot_text})",
        f"  Labels          : {summary['label_source']} (flooded when depth >= {summary['flood_threshold_m']} m)",
        f"  {'Features (' + str(len(summary['feature_names'])) + ')':<16}: {', '.join(summary['feature_names'])}",
        f"  Windows         : seq_len {summary['seq_len']} h (warm-up {summary['warmup_steps']}, scored "
        f"{summary['seq_len'] - summary['warmup_steps']}), lookback {summary['lookback_hours']} h",
        _held_out_line(summary),
        _fingerprint_line(summary),
    ]
    lines += [_split_line(split, stats) for split, stats in summary["splits"].items()]
    if "test" not in summary["splits"]:
        lines.append(f"  {SPLIT_LABELS['test']:<16}: not built (dataset.test_years is empty)")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Namma-Flow stage 04: build train/val/test flood datasets")
    parser.add_argument("--config", default=None,
                        help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)")
    parser.add_argument("--offline", action="store_true", help="never touch the network (cached / synthetic data)")
    parser.add_argument("--force", action="store_true", help="rebuild even if up-to-date datasets exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    overrides = {"project": {"offline": True}} if args.offline else None
    try:
        cfg = load_config(args.config, overrides=overrides)
        summary = build_datasets(cfg, force=args.force)
    except KeyboardInterrupt:
        print(INTERRUPTED_MESSAGE, file=sys.stderr)
        return EXIT_INTERRUPTED
    except _HANDLED as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - last-resort one-line error for the CLI
        LOGGER.debug("Unexpected failure", exc_info=True)
        print(f"ERROR: unexpected {type(exc).__name__}: {exc} (set NAMMA_FLOW_LOG_LEVEL=DEBUG for the traceback)",
              file=sys.stderr)
        return EXIT_FAILED
    print(format_summary(summary))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
