"""Stage 05 CLI — train, calibrate and evaluate the Namma-Flow flood GNN.

Usage (from the project root)::

    python src/training/train.py [--config PATH] [--epochs N] [--device auto|cpu|cuda|mps]
                                 [--resume] [--windows-per-epoch N] [--max-val-windows N]
                                 [--num-threads N] [--offline]

Missing datasets are built first (stage 04). Writes ``best_candidate.pt`` / ``last.pt``
during training and publishes ``best.pt`` (in ``paths.checkpoint_dir``) with
``metrics.json``, ``training_history.csv`` and ``areal_skill.json`` (in ``paths.reports_dir``)
together at the end.
Prints the final metrics on the VALIDATION and held-out TEST splits side by side with the
logistic and HistGradientBoosting baselines. Exit codes: 0 success, 1 handled failure
(one-line error), 130 interrupted (progress saved; continue with ``--resume``). Logic lives
in :mod:`src.training.trainer`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import argparse  # noqa: E402
from typing import Any, Mapping  # noqa: E402

from src.data_pipeline.dataset import DatasetError  # noqa: E402
from src.training.checkpoint import CheckpointError  # noqa: E402
from src.training.settings import DEVICES  # noqa: E402
from src.training.trainer import TrainingError, TrainResult, train  # noqa: E402
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("training.train")
EXIT_OK, EXIT_FAILED, EXIT_INTERRUPTED = 0, 1, 130
_HANDLED = (ConfigError, DatasetError, CheckpointError, TrainingError, FileNotFoundError, ValueError, OSError,
            ImportError)
_TABLE_ROWS = (("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("F2 @ threshold", "f2"), ("Precision", "precision"),
               ("Recall", "recall"), ("CSI", "csi"), ("Brier score", "brier"), ("ECE", "ece"), ("Log loss", "log_loss"))
_COLUMNS = (("GNN", None), ("LogReg", "logreg"), ("HistGBDT", "hist_gbdt"))
_CELL = 10


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {value}")
    return value


def _positive_int(text: str) -> int:
    value = _non_negative_int(text)
    if value == 0:
        raise argparse.ArgumentTypeError("expected a number >= 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Namma-Flow: train the spatio-temporal flood GNN")
    parser.add_argument("--config", default=None,
                        help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)")
    parser.add_argument("--epochs", type=_positive_int, default=None, help="total epochs (default: model.epochs)")
    parser.add_argument("--device", choices=DEVICES, default=None, help="training device (default: training.device)")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in paths.checkpoint_dir")
    parser.add_argument("--windows-per-epoch", type=_non_negative_int, default=None,
                        help="windows sampled per epoch; 0 = every window (default: training.windows_per_epoch)")
    parser.add_argument("--max-val-windows", type=_non_negative_int, default=None,
                        help="evenly spaced validation windows to score; 0 = all (default: training.max_val_windows)")
    parser.add_argument("--num-threads", type=_positive_int, default=None,
                        help="torch CPU threads (default: training.num_threads, else torch's default)")
    parser.add_argument("--offline", action="store_true", help="never touch the network (only matters if the "
                                                               "datasets must be built first)")
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.offline:
        overrides["project"] = {"offline": True}
    if args.num_threads is not None:
        overrides["training"] = {"num_threads": args.num_threads}
    return overrides


def _cell(value: Any) -> str:
    """4 decimals, or 3 significant digits for tiny rare-event values (e.g. a Brier score of 2e-05)."""
    if value is None:
        return "n/a"
    return f"{value:.2e}" if 0 < abs(value) < 1e-3 else f"{value:+.4f}" if value < 0 else f"{value:.4f}"


def _split_line(name: str, split: Mapping[str, Any] | None, windows: Any, years: Any, nodes: Any) -> str:
    if not split:
        return f"  {name:<14}: none - the headline metrics fall back to validation (optimistic)"
    year_text = ", ".join(str(y) for y in years) if isinstance(years, (list, tuple)) and years else "?"
    return (f"  {name:<14}: {windows} windows (years {year_text}) x {nodes} junctions, {split.get('n_pos', 0)} "
            f"flooded node-steps of {split.get('n', 0)} ({100 * (split.get('pos_rate') or 0):.4f} %)")


def _columns(m: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any] | None]]:
    """(split label, metrics dict or None) for GNN / baselines x validation / test."""
    baselines = m.get("baselines") or {}
    out = []
    for _, key in _COLUMNS:
        source: Mapping[str, Any] = m if key is None else (baselines.get(key) or {})
        ok = key is None or source.get("status") == "ok"
        for split, short in (("validation", "val"), ("test", "test")):
            out.append((short, (source.get(split) or None) if ok else None))
    return out


def _table(m: Mapping[str, Any]) -> list[str]:
    """Metric rows; two columns (validation, test) per model under a spanning model header."""
    columns = _columns(m)
    lines = ["  " + " " * 16 + "".join(f"{label:>{2 * _CELL}}" for label, _ in _COLUMNS),
             "  " + f"{'Metric':<16}" + "".join(f"{short:>{_CELL}}" for short, _ in columns)]
    for label, key in _TABLE_ROWS:
        cells = "".join(f"{(_cell(split.get(key)) if split else '-'):>{_CELL}}" for _, split in columns)
        lines.append(f"  {label:<16}{cells}")
    return lines


def _baseline_notes(m: Mapping[str, Any]) -> list[str]:
    notes = []
    for label, key in _COLUMNS[1:]:
        result = (m.get("baselines") or {}).get(key) or {}
        if result.get("status") != "ok":
            status = f"{result.get('status', 'n/a')} {result.get('reason', '')}".rstrip()
            notes.append(f"  {label + ' baseline':<14}: {status}")
    return notes


def _areal_line(areal: Mapping[str, Any] | None) -> str:
    """One line on the areal-only skill (X6): what forecasts / design storms can reach at best."""
    areal = areal or {}
    if areal.get("status") != "ok":
        return f"  Areal-only    : unavailable ({areal.get('reason', 'not computed')})"
    return (f"  Areal-only    : PR-AUC {_cell(areal.get('field_ensemble_pr_auc'))} with only corridor-average rain "
            f"({areal.get('members')}-member rain-field ensemble; uniform {_cell(areal.get('uniform_areal_pr_auc'))}"
            f", exact junction field {_cell(areal.get('exact_field_pr_auc'))}) - areal_skill.json")


def format_report(result: TrainResult) -> str:
    """Human-readable final metrics: GNN vs both baselines on the validation and test splits."""
    m: Mapping[str, Any] = result.metrics
    data, model, training = m.get("dataset", {}), m.get("model", {}), m.get("training", {})
    years, windows = data.get("years") or {}, data.get("windows") or {}
    cal = m.get("calibration") or {}
    strongest = m.get("strongest_baseline") or {}
    headline = m.get("evaluation_split", "validation")
    lines = [
        f"Namma-Flow training - {model.get('architecture', '?')} ({model.get('n_parameters', 0):,} parameters) on "
        f"{training.get('device', '?')}",
        f"  Epochs        : {m.get('epochs_run')} run of {m.get('epochs_target')} ({m.get('status')}); best epoch "
        f"{result.best_epoch} by {m.get('monitor')}; mean epoch {training.get('mean_epoch_time_s') or 0:.1f} s; "
        f"{training.get('optimizer_steps', 'n/a')} optimizer steps",
        _split_line("Validation", m.get("validation") or m, windows.get("validation", data.get("val_windows")),
                    years.get("validation"), data.get("num_nodes")),
        _split_line("Test", m.get("test"), windows.get("test"), years.get("test"), data.get("num_nodes")),
        f"  Calibration   : {cal.get('method', 'n/a')} slope {_cell(cal.get('slope'))} intercept "
        f"{_cell(cal.get('intercept'))} (temperature {_cell(m.get('temperature'))}), alert threshold "
        f"{_cell(m.get('threshold'))} (best {m.get('threshold_metric')}) - both chosen on validation",
        *_table(m),
        f"  Headline      : {headline} split; strongest baseline {strongest.get('name') or 'n/a'} PR-AUC "
        f"{_cell(strongest.get('pr_auc'))} (GNN minus baseline {_cell(strongest.get('delta_pr_auc'))})",
        *_baseline_notes(m),
        _areal_line(m.get("areal_skill")),
    ]
    if data.get("stale"):
        lines.append("  WARNING       : datasets are stale for the current config (rebuild with 04_dataset_builder.py)")
    lines += [f"  Best model    : {result.best_checkpoint}", f"  Last state    : {result.last_checkpoint}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=_overrides(args))
        result = train(cfg, epochs=args.epochs, device=args.device, resume=True if args.resume else None,
                       windows_per_epoch=args.windows_per_epoch, max_val_windows=args.max_val_windows)
    except KeyboardInterrupt:
        print("Interrupted; progress was saved to last.pt - continue with: python src/training/train.py --resume",
              file=sys.stderr)
        return EXIT_INTERRUPTED
    except _HANDLED as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - last-resort one-line error for the CLI
        LOGGER.debug("Unexpected failure", exc_info=True)
        print(f"ERROR: unexpected {type(exc).__name__}: {exc} (set NAMMA_FLOW_LOG_LEVEL=DEBUG for the traceback)",
              file=sys.stderr)
        return EXIT_FAILED
    print(format_report(result))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
