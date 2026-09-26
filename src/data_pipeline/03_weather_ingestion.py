"""Stage 03 — hourly areal rainfall ingestion (Open-Meteo archive -> cache CSV).

Usage (from the project root)::

    python src/data_pipeline/03_weather_ingestion.py [--config PATH] [--offline] [--force]
                                                     [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Fetches (or reuses / extends the cached) hourly precipitation for the configured record,
falls back to the synthetic Bengaluru climatology when offline or when the API fails, writes
``paths.weather_file`` and prints a summary. The cache never shrinks: a ``--start/--end`` range
disjoint from the cached record is stored as a separate block next to it (no zero-filled
bridge), and the hours between blocks are fetched or generated when a later range covers
them. Exit code 0 on success, 1 on a handled failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import argparse  # noqa: E402

from src.data_pipeline.weather import load_or_fetch_weather, summarize_weather  # noqa: E402
from src.utils.config import ConfigError, load_config, resolve_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Namma-Flow stage 03: hourly areal rainfall ingestion")
    parser.add_argument("--config", default=None, help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)")
    parser.add_argument("--offline", action="store_true", help="never touch the network (cache or synthetic only)")
    parser.add_argument("--force", action="store_true", help="refetch the whole range even if the cache covers it")
    parser.add_argument("--start", default=None, help="override weather.start_date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="override weather.end_date (YYYY-MM-DD, inclusive)")
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.offline:
        overrides["project"] = {"offline": True}
    weather = {key: value for key, value in (("start_date", args.start), ("end_date", args.end)) if value}
    if weather:
        overrides["weather"] = weather
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, _overrides(args))
        frame = load_or_fetch_weather(cfg, force=args.force)
        output = resolve_path(cfg, "weather_file")
    except (ConfigError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 1
    print("Namma-Flow weather ingestion")
    for line in summarize_weather(frame).format_lines():
        print(f"  {line}")
    print(f"  output          : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
