"""Forecast-skill backtest: how early and how reliably would the model have warned?

Open-Meteo's Previous Runs API archives what the forecast said 1 and 2 days ahead of every
hour (data from ~2024). For a period ``[start, end]`` the backtest

1. fetches the best-estimate rain (``precip_lead_0``) and the 24 h / 48 h-ahead forecasts,
   plus ``inference.backtest_spinup_h`` hours before ``start`` to spin up the hydrology;
2. builds the "truth": the hydrology simulator (the label teacher) driven by the lead-0 rain
   field (the base ``rainfall_field.seed``), exactly as training labels are made;
3. runs the predictor on four rows and scores each at the predictor's alert threshold
   (junction-hour PR-AUC / ROC-AUC / precision / recall / F1 / F2 / Brier, the same for "any
   junction flooded this hour", junction warning windows of 6 h / 24 h, and the rain-forecast
   skill of the series itself):

   * ``lead_0h`` — the observed rain with the EXACT label field (``exact_field``): emulation of
     the teacher given its own junction rain field;
   * ``areal_ceiling`` — "perfect areal forecast (ensemble)": the same observed areal rain, but
     with ``inference.field_members`` independent rain fields (seed offset 1, never the label
     seed). Forecasts only know corridor-average rain, so this row is what THIS model reaches
     with a perfect areal forecast (the physics column, the teacher averaged over the same
     fields, estimates the best any predictor could reach);
   * ``lead_24h`` / ``lead_48h`` — the archived forecasts with the same offset-1 ensembles;

4. adds the physics baseline driven by the same rain and fields (when the predictor is the
   GNN). At lead 0 that "baseline" is the label generator itself on the very same rain field
   (PR-AUC 1.0 by construction), so it is marked ``role: "teacher"``; the ceiling row's physics
   run is ``role: "areal_baseline"`` (the teacher's own skill given areal rain) and the 24 h /
   48 h ones are ``role: "forecast_baseline"``. When the predictor IS the physics baseline its
   own lead-0 row is the teacher (``role: "teacher"``) and the report is written to
   ``backtest_physics.json`` so it never overwrites the GNN's ``backtest.json``.

Offline mode or an API failure returns ``{"status": "unavailable", "reason": ...}`` (never raises).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_pipeline import weather
from src.data_pipeline.rain_field import downscale_rainfall
from src.hydrology.simulator import simulate_labels
from src.inference.results import json_safe
from src.inference.scenarios import InferenceSettings, Scenario
from src.utils.config import resolve_path
from src.utils.http import NetworkUnavailable
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

LEADS = (("lead_0h", "precip_lead_0", 0), ("lead_24h", "precip_lead_24h", 24), ("lead_48h", "precip_lead_48h", 48))
CEILING_KEY = "areal_ceiling"
CEILING_LABEL = "perfect areal forecast (ensemble)"
REPORT_NAME = "backtest.json"
PHYSICS_REPORT_NAME = "backtest_physics.json"
FORECAST_SEED_OFFSET = 1  # ensemble rows never reuse the label field's seed (offset 0)
WET_HOUR_MM = 1.0  # an hour counts as "wet" for the rain-skill hit rate at or above this areal rain
TEACHER_LABEL = "teacher (= labels)"
TEACHER_NOTE = ("lead 0: the physics simulator on the observed rain is the label generator itself, so its scores are "
                "1.0 by construction - not an independent baseline")
CEILING_NOTE = ("perfect areal forecast (ensemble): the observed corridor-average rain with independent junction rain "
                "fields (the label field is unknown to any forecast) - what this model reaches with a perfect areal "
                "forecast; the physics column (the teacher averaged over the same fields) estimates the best possible")


# --------------------------------------------------------------------------- metrics


def _f_beta(precision: float | None, recall: float | None, beta: float) -> float | None:
    """F-beta; undefined (None) without positives, 0 when positives exist but nothing was alerted."""
    if recall is None:
        return None
    if precision is None:
        return 0.0
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator > 0 else 0.0


def classification_metrics(y_true: Any, y_prob: Any, threshold: float) -> dict[str, Any]:
    """Ranking and thresholded metrics; undefined values (single-class truth, no alerts) are None."""
    y = np.asarray(y_true).astype(bool).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ValueError(f"y_true has {y.size} values but y_prob has {p.size}")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("y_prob must be finite probabilities in [0, 1]")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold!r}")
    n, n_pos = int(y.size), int(y.sum())
    alert = p >= float(threshold)
    tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
    fn = n_pos - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / n_pos if n_pos else None
    both = 0 < n_pos < n
    return {
        "n": n, "n_positive": n_pos, "pos_rate": n_pos / n if n else 0.0, "predicted_pos_rate": float(alert.mean())
        if n else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": n - tp - fp - fn, "threshold": float(threshold),
        "precision": precision, "recall": recall, "f1": _f_beta(precision, recall, 1.0),
        "f2": _f_beta(precision, recall, 2.0),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "brier": float(np.mean((p - y) ** 2)) if n else None,
    }


def rain_skill(observed: np.ndarray, forecast: np.ndarray) -> dict[str, Any]:
    """Areal rain-forecast skill over the target hours (totals, MAE, correlation, wet-hour hits)."""
    obs, fc = np.asarray(observed, dtype=np.float64), np.asarray(forecast, dtype=np.float64)
    wet_obs, wet_fc = obs >= WET_HOUR_MM, fc >= WET_HOUR_MM
    correlation = None
    if obs.size > 1 and obs.std() > 0 and fc.std() > 0:
        correlation = float(np.corrcoef(obs, fc)[0, 1])
    return {
        "total_mm": float(fc.sum()), "observed_total_mm": float(obs.sum()), "peak_mm_h": float(fc.max(initial=0.0)),
        "mae_mm_h": float(np.abs(fc - obs).mean()) if obs.size else 0.0, "correlation": correlation,
        "wet_hours": int(wet_fc.sum()), "observed_wet_hours": int(wet_obs.sum()),
        "wet_hour_hit_rate": float((wet_fc & wet_obs).sum() / wet_obs.sum()) if wet_obs.any() else None,
    }


# Warning windows: a 12-48 h alert is useful when the junction floods at *some* hour of the block,
# so timing errors of a few hours in the rain forecast are not counted as misses.
WARNING_BLOCKS_H = (6, 24)


def block_max(values: np.ndarray, block_h: int) -> np.ndarray:
    """Max over consecutive ``block_h``-hour blocks along axis 0 (the last block may be shorter)."""
    arr = np.asarray(values)
    if block_h < 1:
        raise ValueError(f"block_h must be >= 1, got {block_h}")
    if arr.shape[0] == 0:
        return arr.reshape((0, *arr.shape[1:]))
    starts = np.arange(0, arr.shape[0], block_h)
    return np.maximum.reduceat(arr, starts, axis=0)


def _score(labels: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Junction-hour, hour-level "any junction flooded" and junction warning-window metrics."""
    scores = {
        "junction_hours": classification_metrics(labels, prob, threshold),
        "hourly_any": classification_metrics(labels.any(axis=1), prob.max(axis=1, initial=0.0), threshold),
    }
    for block in WARNING_BLOCKS_H:
        scores[f"junction_block_{block}h"] = classification_metrics(
            block_max(labels.astype(np.uint8), block), block_max(prob, block), threshold)
    return scores


# --------------------------------------------------------------------------- period & report


def _parse_bound(value: Any, tz: str, *, end: bool, name: str) -> pd.Timestamp:
    date_only = isinstance(value, dt.date) and not isinstance(value, dt.datetime)
    if isinstance(value, str):
        date_only = len(value.strip()) == 10
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"backtest {name} must be an ISO date/time, got {value!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"backtest {name} must be an ISO date/time, got {value!r}")
    ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    return (ts + pd.Timedelta(hours=23) if date_only and end else ts).floor("h")


def backtest_period(start: Any, end: Any, tz: str, max_days: int, now: Any = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validated local ``(first, last)`` hours; date-only ``end`` covers its whole day; the end is
    clamped to the last complete hour (the archive holds no observations of the future)."""
    lo = _parse_bound(start, tz, end=False, name="start")
    hi = _parse_bound(end, tz, end=True, name="end")
    reference = pd.Timestamp.now(tz=tz) if now is None else _parse_bound(now, tz, end=False, name="now")
    latest = reference.floor("h") - pd.Timedelta(hours=1)
    if lo > latest:
        raise ValueError(f"backtest start {lo:%Y-%m-%d %H:%M} is in the future")
    if hi > latest:
        LOGGER.warning("backtest end %s is in the future; clamped to %s", f"{hi:%Y-%m-%d %H:%M}",
                       f"{latest:%Y-%m-%d %H:%M}")
        hi = latest
    if lo > hi:
        raise ValueError(f"backtest start {lo:%Y-%m-%d %H:%M} is after end {hi:%Y-%m-%d %H:%M}")
    if (hi - lo) > pd.Timedelta(days=max_days):
        raise ValueError(f"backtest period exceeds inference.max_backtest_days={max_days}; split it into shorter runs")
    return lo, hi


def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    """The report returned when the backtest cannot run (offline / API failure)."""
    LOGGER.warning("Forecast backtest unavailable: %s", reason)
    return {"status": "unavailable", "reason": str(reason), **json_safe(extra),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def report_name(predictor_kind: str) -> str:
    """``backtest.json`` for the GNN, ``backtest_physics.json`` for a physics-baseline backtest."""
    return PHYSICS_REPORT_NAME if predictor_kind == "physics" else REPORT_NAME


def write_backtest_report(cfg: Mapping[str, Any], report: Mapping[str, Any]) -> Path | None:
    """Write ``report`` to ``paths.reports_dir`` atomically (None + WARNING on failure); the file is
    :func:`report_name` of the report's predictor kind."""
    kind = str(((report or {}).get("predictor") or {}).get("kind") or "gnn")
    path = resolve_path(cfg, "reports_dir") / report_name(kind)
    try:
        return atomic_write_text(path, json.dumps(json_safe(report), indent=2, allow_nan=False))
    except OSError as exc:
        LOGGER.warning("Could not write the backtest report %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- the backtest


def _fetch(cfg: Mapping[str, Any], lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    frame = weather.fetch_previous_runs(cfg, lo, hi, lead_days=(1, 2))
    missing = [column for _, column, _ in LEADS if column not in frame]
    if missing or not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise weather.WeatherUnavailable(f"Previous Runs frame lacks {missing or 'rows'}")
    return frame


def _lead_scenario(frame: pd.DataFrame, column: str, lead_h: int, target: int, exact_field: bool) -> Scenario:
    areal, index = weather.areal_series(frame.rename(columns={column: weather.PRECIP})[[weather.PRECIP]])
    what = "observed rain" if lead_h == 0 else f"{lead_h} h-ahead forecast"
    return Scenario(
        name=f"Backtest ({what}{'' if exact_field else ', rain-field ensemble'})", kind="historical",
        timestamps=index, areal_mm=areal, is_forecast=np.full(len(index), lead_h > 0), target_start=target,
        description=f"Open-Meteo Previous Runs {column}", source="previous_runs", exact_field=exact_field,
    )


@dataclass(frozen=True)
class _Row:
    """One scored row of the backtest: which rain series drives it and which rain fields."""

    key: str
    column: str
    lead_h: int
    exact_field: bool
    label: str


ROWS = (
    _Row("lead_0h", "precip_lead_0", 0, True, "observed rain, exact label field"),
    _Row(CEILING_KEY, "precip_lead_0", 0, False, CEILING_LABEL),
    _Row("lead_24h", "precip_lead_24h", 24, False, "24 h-ahead forecast, rain-field ensemble"),
    _Row("lead_48h", "precip_lead_48h", 48, False, "48 h-ahead forecast, rain-field ensemble"),
)


def _predictor_info(predictor: Any) -> dict[str, Any]:
    info = {"kind": getattr(predictor, "kind", "unknown"), "label": getattr(predictor, "label", "unknown"),
            "threshold": float(predictor.threshold)}
    if hasattr(predictor, "describe"):
        info.update({k: v for k, v in predictor.describe().items() if k != "feature_names"})
    return info


def _field_info(result: Any, row: _Row) -> dict[str, Any]:
    meta = getattr(result, "metadata", {}) or {}
    return {"exact": row.exact_field, "members": int(meta.get("field_members") or 1),
            "seed_offset": int(meta.get("field_seed_offset") or 0)}


def _baseline_role(row: _Row) -> dict[str, Any]:
    if row.exact_field:
        return {"role": "teacher", "note": TEACHER_NOTE}
    return {"role": "areal_baseline"} if row.key == CEILING_KEY else {"role": "forecast_baseline"}


def _run_row(frame: pd.DataFrame, target: int, labels: np.ndarray, observed: np.ndarray, row: _Row, predictor: Any,
             baseline: Any | None, members: int | None) -> dict[str, Any]:
    clock = time.perf_counter()
    scenario = _lead_scenario(frame, row.column, row.lead_h, target, row.exact_field)
    fields = {} if row.exact_field else {"field_members": members, "field_seed_offset": FORECAST_SEED_OFFSET}
    result = predictor.predict(scenario, **fields)
    entry = {"lead_hours": row.lead_h, "label": row.label, "field": _field_info(result, row),
             "rain": rain_skill(observed, scenario.target_areal_mm),
             # Score at the threshold the predictor serves this row with: the exact-field threshold for
             # the replayed field, the validation-chosen areal threshold for rain-field ensembles.
             "threshold": float(result.threshold), **_score(labels, result.prob, result.threshold)}
    if row.exact_field and getattr(predictor, "kind", "") == "physics":
        entry.update(role="teacher", note=TEACHER_NOTE)   # the predictor is the label generator on its own field
    elif row.key == CEILING_KEY:
        entry.update(role="areal_ceiling", note=CEILING_NOTE)
    if baseline is not None:
        physics = baseline.predict(scenario, **fields)
        entry["physics_baseline"] = {**_baseline_role(row), **_score(labels, physics.prob, physics.threshold)}
    entry["elapsed_s"] = round(time.perf_counter() - clock, 2)
    node = entry["junction_hours"]
    LOGGER.info("Backtest %s: PR-AUC %s, recall %s, precision %s (%d field member(s), %.1f s)", row.key,
                node["pr_auc"], node["recall"], node["precision"], entry["field"]["members"], entry["elapsed_s"])
    return entry


def _run_leads(frame: pd.DataFrame, target: int, labels: np.ndarray, predictor: Any,
               baseline: Any | None, members: int | None = None) -> dict[str, Any]:
    observed = frame["precip_lead_0"].to_numpy(dtype=np.float64)[target:]
    return {row.key: _run_row(frame, target, labels, observed, row, predictor, baseline, members) for row in ROWS}


def run_forecast_backtest(cfg: Mapping[str, Any], start: Any, end: Any, predictor: Any, *,
                          include_physics_baseline: bool = True, save: bool = True, now: Any = None,
                          field_members: int | None = None) -> dict[str, Any]:
    """Score ``predictor`` on ``[start, end]`` driven by observed and archived-forecast rain.

    ``field_members`` (default ``inference.field_members``) sets the rain-field ensemble of the
    ceiling and forecast rows. Returns the report dict (``status="ok"``) and writes it to
    ``reports_dir/`` :func:`report_name` when ``save``; offline or API failure →
    ``{"status": "unavailable", "reason": ...}``. Invalid periods raise ``ValueError``.
    """
    if not all(hasattr(predictor, attr) for attr in ("predict", "graph", "threshold")):
        raise TypeError(f"predictor must be a FloodPredictor or PhysicsPredictor, got {type(predictor).__name__}")
    settings = InferenceSettings.from_config(cfg)
    tz = str((cfg.get("project") or {}).get("timezone") or "Asia/Kolkata")
    lo, hi = backtest_period(start, end, tz, settings.max_backtest_days, now)
    spinup = pd.Timedelta(hours=settings.backtest_spinup_h)
    clock = time.perf_counter()
    try:
        frame = _fetch(cfg, lo - spinup, hi)
    except (weather.WeatherUnavailable, NetworkUnavailable, ValueError) as exc:
        return unavailable(str(exc), start=lo, end=hi)
    if lo not in frame.index:
        return unavailable(f"Previous Runs data does not include {lo:%Y-%m-%d %H:%M}", start=lo, end=hi)
    target = int(frame.index.get_loc(lo))
    frame = frame.loc[:hi]
    graph = predictor.graph
    observed = frame["precip_lead_0"].to_numpy(dtype=np.float64)
    rain = downscale_rainfall(observed, pd.DatetimeIndex(frame.index), graph.lon, graph.lat, cfg)
    labels = simulate_labels(graph, rain, cfg)[0][target:]
    baseline = None
    if include_physics_baseline and getattr(predictor, "kind", "") != "physics":
        from src.inference.predictor import PhysicsPredictor

        baseline = PhysicsPredictor(cfg, graph, getattr(predictor, "provenance", None))
    members = settings.field_members if field_members is None else field_members
    leads = _run_leads(frame, target, labels, predictor, baseline, members)
    flooded_fraction = labels.mean(axis=1) if labels.size else np.zeros(0)
    ensemble = leads[CEILING_KEY]["field"]
    report = {
        "status": "ok", "start": lo, "end": hi, "n_hours": int(labels.shape[0]), "n_nodes": int(labels.shape[1]),
        "spinup_hours": int(target), "predictor": _predictor_info(predictor),
        "field_ensemble": {"members": ensemble["members"], "seed_offset": ensemble["seed_offset"],
                           "seed_stride": settings.field_seed_stride,
                           "note": "lead_0h uses the exact label field; the ceiling and forecast rows average over "
                                   "independent rain-field realisations (the label field is unknown to forecasts)"},
        "labels": {"source": "hydrology simulator on the lead-0 (best-estimate) rain field",
                   "n_positive": int(labels.sum()), "pos_rate": float(labels.mean()) if labels.size else 0.0,
                   "flooded_hours": int(labels.any(axis=1).sum()),
                   "peak_flooded_fraction": float(flooded_fraction.max(initial=0.0)),
                   "imputed_hours": int(frame["is_imputed"].to_numpy(dtype=bool)[target:].sum())
                   if "is_imputed" in frame else 0},
        "leads": leads, "elapsed_s": round(time.perf_counter() - clock, 2),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if not report["labels"]["n_positive"]:
        report["note"] = "No simulated floods in this period: PR-AUC / ROC-AUC / recall are undefined (None)"
    if save:
        path = write_backtest_report(cfg, report)
        report["report_path"] = None if path is None else str(path)
    return json_safe(report)


def _fmt(value: Any) -> str:
    return "   n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:6.3f}"


def _field_tag(entry: Mapping[str, Any]) -> str:
    """``exact`` (the label field), ``ens xK`` (K independent fields) or ``1 field`` (older reports)."""
    field = entry.get("field")
    if not isinstance(field, Mapping):
        return "exact" if entry.get("lead_hours") == 0 else "1 field"
    return "exact" if field.get("exact") else f"ens x{int(field.get('members') or 1)}"


def _lead_line(key: str, entry: Mapping[str, Any]) -> tuple[str, bool]:
    node, hourly, rain = entry["junction_hours"], entry["hourly_any"], entry["rain"]
    baseline = entry.get("physics_baseline") or {}
    physics = _fmt(baseline.get("junction_hours", {}).get("pr_auc")) if baseline else _fmt(None)
    teacher = bool(baseline) and is_teacher(entry)
    if teacher:
        physics = TEACHER_LABEL
    head = f"{key:<13} {_field_tag(entry):<7} {rain['total_mm']:7.1f} {_fmt(rain['correlation'])} "
    if predictor_is_teacher(entry):
        return head + f"{TEACHER_LABEL} (the predictor is the label generator on the label field)", True
    scores = (f"{_fmt(node['pr_auc'])}  {_fmt(node['roc_auc'])}  {_fmt(node['recall'])} {_fmt(node['precision'])}  "
              f"{_fmt(node['f2'])} | {_fmt(hourly['recall'])}  {_fmt(hourly['precision'])} | {physics}")
    return head + scores, teacher


def _window_line(key: str, entry: Mapping[str, Any]) -> str:
    if predictor_is_teacher(entry):
        return f"{key:<13} {TEACHER_LABEL}"
    cells = []
    for block in WARNING_BLOCKS_H:
        metrics = entry.get(f"junction_block_{block}h") or {}
        cells.append(" / ".join(_fmt(metrics.get(key)) for key in ("pr_auc", "recall", "precision")))
    return f"{key:<13} " + "      ".join(cells)


def _threshold_text(report: Mapping[str, Any]) -> str:
    """The alert threshold(s) the rows were scored at: exact-field and, for ensembles, areal."""
    exact = round(float(report["predictor"]["threshold"]), 6)
    served = {round(float(e["threshold"]), 6) for e in (report.get("leads") or {}).values()
              if isinstance(e, Mapping) and e.get("threshold") is not None}
    others = sorted(t for t in served if abs(t - exact) > 1e-9)
    if not others:
        return f"threshold {exact:.3f}"
    return f"threshold {exact:.3f} (exact field) / " + ", ".join(f"{t:.3f}" for t in others) + " (rain-field ensembles)"


def format_backtest(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for the CLI."""
    if report.get("status") != "ok":
        return [f"Backtest unavailable: {report.get('reason', 'unknown reason')}"]
    labels = report["labels"]
    lines = [
        f"Backtest {report['start']} -> {report['end']} ({report['n_hours']} h x {report['n_nodes']} junctions), "
        f"predictor {report['predictor']['label']} @ {_threshold_text(report)}",
        f"Simulated floods: {labels['n_positive']} junction-hours ({100 * labels['pos_rate']:.3f} %), "
        f"{labels['flooded_hours']} flooded hours, peak {100 * labels['peak_flooded_fraction']:.1f} % of junctions",
        "row           field   rain_mm  corr   PR-AUC  ROC-AUC  recall  precis.  F2     | hourly recall precis. | "
        "physics PR-AUC",
    ]
    teacher = False
    for key, entry in report["leads"].items():
        line, is_row_teacher = _lead_line(key, entry)
        lines.append(line)
        teacher = teacher or is_row_teacher
    lines.append("warning windows (junction floods at any hour of the block):  " + "   ".join(
        f"{block} h block PR-AUC / recall / precision" for block in WARNING_BLOCKS_H))
    lines += [_window_line(key, entry) for key, entry in report["leads"].items()]
    if teacher:
        lines.append(f"Note: {TEACHER_NOTE}.")
    if CEILING_KEY in report["leads"]:
        lines.append(f"Note: {CEILING_KEY} = {CEILING_NOTE}; 'ens xK' rows average K rain-field realisations.")
    if report.get("note"):
        lines.append(f"Note: {report['note']}")
    return lines


def predictor_is_teacher(lead: Mapping[str, Any]) -> bool:
    """Whether the scored predictor of this row IS the label generator (a physics backtest's lead 0)."""
    return isinstance(lead, Mapping) and lead.get("role") == "teacher"


def is_teacher(lead: Mapping[str, Any]) -> bool:
    """Whether a lead's physics baseline is the label generator itself (lead 0; older reports lack ``role``)."""
    baseline = lead.get("physics_baseline") if isinstance(lead, Mapping) else None
    if not isinstance(baseline, Mapping):
        return False
    role = baseline.get("role")
    return role == "teacher" if role is not None else lead.get("lead_hours") == 0
