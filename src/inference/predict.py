"""Command-line flood prediction and forecast backtests.

Examples (from the project root)::

    python -m src.inference.predict --scenario forecast
    python -m src.inference.predict --scenario design --total-mm 80 --duration-h 3 --mc 20
    python -m src.inference.predict --scenario design --preset cloudburst --storm-offset-h 0 --hours 12 --physics
    python -m src.inference.predict --scenario historical --start 2022-09-04T12:00 --hours 48
    python -m src.inference.predict --backtest 2024-10-01 2024-10-31

``--physics`` uses the hydrology-simulator baseline instead of the trained GNN (also the way to
run before a model exists). Forecasts and design storms know only corridor-average rain, so their
junction rain is averaged over ``--members`` (default ``inference.field_members``) stochastic
rain-field realisations; historical replays use the exact training field. A physics backtest is
written to ``backtest_physics.json`` (the GNN's ``backtest.json`` is never overwritten). Exit code
0 on success, 1 on a handled failure (one-line error).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # allow ``python src/inference/predict.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.weather import WeatherUnavailable  # noqa: E402
from src.inference.backtest import format_backtest, run_forecast_backtest  # noqa: E402
from src.inference.predictor import (  # noqa: E402
    FloodPredictor,
    GraphNotReady,
    ModelNotReady,
    PhysicsPredictor,
    PredictionError,
)
from src.inference.results import PredictionResult  # noqa: E402
from src.inference.scenarios import (  # noqa: E402
    InferenceSettings,
    Scenario,
    ScenarioError,
    design_storm_scenario,
    forecast_scenario,
    historical_scenario,
    list_notable_events,
    preset_scenario,
)
from src.utils.config import ConfigError, load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("src.inference.predict")
DEFAULT_OUTPUT = "prediction.geojson"


class CliError(RuntimeError):
    """A handled failure: printed as one line, exit code 1."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.inference.predict", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    parser.add_argument("--scenario", choices=("forecast", "design", "historical"), default="forecast")
    parser.add_argument("--total-mm", type=float, help="design storm total rainfall (mm)")
    parser.add_argument("--duration-h", type=int, help="design storm duration (hours)")
    parser.add_argument("--preset", help="design storm preset from inference.design_storms (name prefix or index); "
                                         "not combinable with --total-mm / --duration-h")
    parser.add_argument("--storm-offset-h", type=int, default=None,
                        help="hours from the first target hour to the storm start (also for --preset; default "
                             "inference.design_storm_offset_h)")
    parser.add_argument("--rain-scale", type=float, default=None,
                        help="gauge -> areal factor for design storms (default inference.design_storm_rain_scale)")
    parser.add_argument("--start", help="historical replay start (ISO local time, e.g. 2022-09-04T12:00)")
    parser.add_argument("--hours", type=int, default=None, help="target hours (default: max inference.horizons_h)")
    parser.add_argument("--mc", type=int, nargs="?", const=-1, default=0, metavar="N",
                        help="MC-dropout samples for uncertainty (bare --mc: inference.mc_dropout_samples)")
    parser.add_argument("--members", type=int, default=None, metavar="K",
                        help="rain-field realisations averaged when only areal rain is known (forecast, design, "
                             "backtest forecast rows; default inference.field_members; replays use the exact field)")
    parser.add_argument("--physics", action="store_true", help="use the physics baseline instead of the GNN")
    parser.add_argument("--checkpoint", help="checkpoint path (default: paths.checkpoint_dir/best.pt)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    parser.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    parser.add_argument("--out", help=f"GeoJSON output (default: paths.reports_dir/{DEFAULT_OUTPUT})")
    parser.add_argument("--csv", help="also write the per-junction table as CSV")
    parser.add_argument("--timeline", action="store_true", help="include hourly probabilities in the GeoJSON")
    parser.add_argument("--top", type=int, default=10, help="riskiest junctions to print")
    parser.add_argument("--no-fallback", action="store_true",
                        help="fail instead of replaying the latest notable event when the forecast is unavailable")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"), help="run a forecast backtest instead")
    return parser


# --------------------------------------------------------------------------- steps


def _load_cfg(args: argparse.Namespace, cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    base = dict(cfg) if cfg is not None else load_config(args.config)
    if args.offline:
        base["project"] = {**base.get("project", {}), "offline": True}
    return base


def _configure_threads(args: argparse.Namespace, settings: InferenceSettings) -> None:
    threads = args.threads if args.threads is not None else settings.num_threads
    if threads is None:
        return
    if threads < 1:
        raise CliError(f"--threads must be >= 1, got {threads}")
    import torch

    torch.set_num_threads(int(threads))


def _choose_predictor(cfg: Mapping[str, Any], args: argparse.Namespace) -> Any:
    if args.physics:
        return PhysicsPredictor.from_graph(cfg)
    try:
        return FloodPredictor.from_artifacts(cfg, checkpoint_path=args.checkpoint, device=args.device)
    except ModelNotReady as exc:
        raise CliError(f"{exc}. Re-run with --physics to use the physics baseline.") from exc


def _forecast_or_fallback(cfg: Mapping[str, Any], hours: int, no_fallback: bool) -> Scenario:
    try:
        return forecast_scenario(cfg, hours=hours)
    except WeatherUnavailable as exc:
        if no_fallback:
            raise CliError(f"Live forecast unavailable: {exc}") from exc
        events = list_notable_events(cfg, top_k=50)
        if events.empty:
            raise CliError(f"Live forecast unavailable ({exc}) and no weather record to replay instead") from exc
        latest = events.sort_values("start").iloc[-1]
        print(f"NOTE: live forecast unavailable ({exc}).\n      Replaying the most recent notable event instead: "
              f"{latest['label']}")
        return historical_scenario(cfg, latest["start"], hours)


def build_scenario(cfg: Mapping[str, Any], args: argparse.Namespace, settings: InferenceSettings) -> Scenario:
    """The scenario selected by the CLI arguments."""
    hours = args.hours if args.hours is not None else settings.max_horizon_h
    if args.scenario == "forecast":
        return _forecast_or_fallback(cfg, hours, args.no_fallback)
    if args.scenario == "historical":
        if not args.start:
            raise CliError("--scenario historical needs --start (ISO local time, e.g. 2022-09-04T12:00)")
        return historical_scenario(cfg, args.start, hours)
    if args.preset is not None:
        if args.total_mm is not None or args.duration_h is not None:
            raise CliError("--preset sets the storm total and duration; drop --total-mm / --duration-h, or drop "
                           "--preset to define a custom storm")
        return preset_scenario(cfg, args.preset, horizon_h=hours, rain_scale=args.rain_scale,
                               storm_start_offset_h=args.storm_offset_h)
    if args.total_mm is None or args.duration_h is None:
        presets = ", ".join(f"{i}: {p.name}" for i, p in enumerate(settings.design_storms))
        raise CliError(f"--scenario design needs --total-mm and --duration-h, or --preset ({presets})")
    return design_storm_scenario(cfg, args.total_mm, args.duration_h, args.storm_offset_h, hours,
                                 rain_scale=args.rain_scale)


def _mc_samples(args: argparse.Namespace, settings: InferenceSettings) -> int:
    samples = settings.mc_dropout_samples if args.mc == -1 else args.mc
    if samples < 0:
        raise CliError(f"--mc must be >= 0, got {samples}")
    if args.members is not None and args.members < 1:
        raise CliError(f"--members must be >= 1, got {args.members}")
    return samples


def format_prediction(result: PredictionResult, scenario: Scenario, top: int) -> list[str]:
    """Summary lines printed by the CLI."""
    summary = result.summary(result.n_hours or 1)
    meta = result.metadata
    spread = summary.get("junctions_at_risk_range")
    members = int(meta.get("field_members") or 1)
    across = f" (range {spread[0]}-{spread[1]} across the {members} rain fields)" if spread else ""
    lines = [
        f"Scenario   : {scenario.name} [{scenario.kind}, source={scenario.source}]",
        f"             {scenario.description}",
        f"Predictor  : {meta.get('label', result.predictor)} (alert threshold {result.threshold:.3f}, "
        f"{meta.get('mc_samples', 0)} MC samples, {members} rain-field member(s), {meta.get('passes', 1)} pass(es), "
        f"{meta.get('elapsed_s', 0):.2f} s)",
        f"Target     : {summary['start']} -> {summary['end']} ({summary['hours']} h), "
        f"areal rain {summary['rain_total_mm']:.1f} mm (peak {summary['peak_rain_mm_h']:.1f} mm/h)",
        "Horizon    : " + " | ".join(f"{h} h: {v['junctions_at_risk']} at risk, max p {v['max_prob']:.2f}"
                                     for h, v in summary["horizons"].items()),
        f"Overall    : {summary['junctions_at_risk']} of {summary['n_nodes']} junctions >= threshold{across}, max p "
        f"{summary['max_prob']:.3f}, peak hour {summary['peak_time']}",
        "Tiers      : " + ", ".join(f"{k}={v}" for k, v in summary["tier_counts"].items()),
    ]
    lines += [f"Note       : {note}" for note in _notes(result, scenario)]
    if top > 0 and result.n_hours:
        lines += _top_lines(result, top)
    return lines


def _top_lines(result: PredictionResult, top: int) -> list[str]:
    """The riskiest junctions (``spread``: std across rain fields / MC passes; ``share``: rain fields at risk)."""
    table = result.top_k(top, result.n_hours)
    lines = [f"Top {len(table)} junctions:",
             "  rank  node_id        max_p   spread  share  tier      peak_time         elev_m  rel_elev  drain_m"]
    for row in table.itertuples(index=False):
        std = "  n/a " if row.prob_std != row.prob_std else f"{row.prob_std:6.3f}"
        share = "  n/a" if row.field_share_at_risk != row.field_share_at_risk else f"{row.field_share_at_risk:5.0%}"
        lines.append(f"  {row.rank:>4}  {str(row.node_id):<13} {row.max_prob:6.3f} {std}  {share}  {row.tier:<9} "
                     f"{row.peak_time:%m-%d %H:%M}   {row.elevation:7.1f} {row.relative_elevation:8.2f} "
                     f"{row.dist_to_drain_m:8.0f}")
    return lines


def _notes(result: PredictionResult, scenario: Scenario) -> list[str]:
    """Caveats of this run: scenario notes, dry-padded history and what the junction rain represents."""
    notes = list(scenario.notes)
    padded = int(result.metadata.get("padded_history_h") or 0)
    if padded:
        notes.append(f"only {result.metadata.get('history_hours')} h of rain history preceded the first target hour; "
                     f"the model needs {result.metadata.get('history_needed_h')} h, so {padded} h were treated as dry")
    if result.metadata.get("rain_field"):
        notes.append(str(result.metadata["rain_field"]))
    return notes


def _write_outputs(cfg: Mapping[str, Any], args: argparse.Namespace, result: PredictionResult) -> list[Path]:
    out = Path(args.out) if args.out else resolve_path(cfg, "reports_dir") / DEFAULT_OUTPUT
    hours = max(result.n_hours, 1)
    written = [result.write_geojson(out, hours=hours, include_timeline=args.timeline)]
    if args.csv:
        written.append(result.to_csv(args.csv, hours=hours))
    return written


def _run_prediction(cfg: Mapping[str, Any], args: argparse.Namespace, settings: InferenceSettings) -> None:
    clock = time.perf_counter()
    scenario = build_scenario(cfg, args, settings)
    predictor = _choose_predictor(cfg, args)
    result = predictor.predict(scenario, mc_samples=_mc_samples(args, settings), field_members=args.members,
                               keep_members=True)
    for line in format_prediction(result, scenario, args.top):
        print(line)
    for path in _write_outputs(cfg, args, result):
        print(f"Wrote      : {path}")
    print(f"Total time : {time.perf_counter() - clock:.2f} s")


def _run_backtest(cfg: Mapping[str, Any], args: argparse.Namespace) -> None:
    if args.members is not None and args.members < 1:
        raise CliError(f"--members must be >= 1, got {args.members}")
    predictor = _choose_predictor(cfg, args)
    report = run_forecast_backtest(cfg, args.backtest[0], args.backtest[1], predictor, field_members=args.members)
    for line in format_backtest(report):
        print(line)
    if report.get("status") != "ok":
        raise CliError(f"backtest unavailable: {report.get('reason')}")
    if report.get("report_path"):
        print(f"Wrote      : {report['report_path']}")


def main(argv: Sequence[str] | None = None, cfg: Mapping[str, Any] | None = None) -> int:
    """CLI entry point; ``cfg`` (tests) replaces loading ``--config``."""
    args = build_parser().parse_args(argv)
    try:
        config = _load_cfg(args, cfg)
        settings = InferenceSettings.from_config(config)
        _configure_threads(args, settings)
        if args.backtest:
            _run_backtest(config, args)
        else:
            _run_prediction(config, args, settings)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (CliError, ConfigError, ModelNotReady, GraphNotReady, ScenarioError, PredictionError, WeatherUnavailable,
            ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
