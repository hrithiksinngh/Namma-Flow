"""Building blocks of the Namma-Flow dashboard: settings, formatting, KPI maths, Altair charts
and the "About the model" / data-provenance panels.

Pure helpers (``AppSettings``, formatting, ``build_kpis``, ``*_frame`` / ``*_table`` builders,
charts, :func:`public_text`) have no Streamlit side effects and are unit tested directly;
``render_*`` functions draw into the current Streamlit container. How evaluation metrics are
read and worded lives in :mod:`metrics_view`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import map_layers as ml
import metrics_view as mv
from src.inference.results import PredictionResult, display_path
from src.inference.scenarios import Scenario
from src.utils.config import ConfigError, get_section, project_root
from src.utils.logger import get_logger

LOGGER = get_logger("app.components")

MODE_LABELS = {
    "forecast": "Live forecast (Open-Meteo)",
    "historical": "Historical replay",
    "design": "What-if design storm",
}
PREDICTOR_LABELS = {"gnn": "Namma-Flow GNN", "physics": "Physics baseline"}
COLOR_SCALE_LABELS = {"auto": "Auto", "tiers": "Risk tiers", "relative": "Relative"}
DEFAULTS: dict[str, Any] = {
    "map_style": "light",
    "column_radius_m": 22,
    "elevation_scale": 4,
    "top_k": 15,
    "map_height_px": 560,
    "default_mode": "forecast",
    "notable_events": 12,
    "color_scale": "auto",
    "pitch_deg": 45,
    "forecast_ttl_s": 1800,
    "forecast_timeout_s": 20,
    "forecast_max_retries": 2,
    "max_mc_samples_ui": 50,
    "custom_storm_max_mm": 250,
    "custom_storm_max_h": 24,
}
RAIN_COLORS = {
    "Observed record": "#4C78A8", "Synthetic record": "#72B7B2", "Recent (model analysis)": "#9ECAE9",
    "Forecast": "#F58518", "Dry lead-in": "#BAB0AC", "Design storm": "#E45756", "Rain": "#4C78A8",
}
GRAPH_SOURCES = {"osm_place": "OpenStreetMap (place query)", "osm_bbox": "OpenStreetMap (bounding-box query)",
                 "synthetic_grid": "Synthetic street grid (offline demo)"}
ELEVATION_SOURCES = {"local_dem": "Local DEM GeoTIFF", "srtm": "SRTM 1-arc-second DEM (AWS terrain tiles)",
                     "open_meteo": "Open-Meteo elevation API", "synthetic": "Synthetic valley formula"}
DRAIN_SOURCES = {"osm": "OpenStreetMap drains, streams & lakes",
                 "synthetic_line": "Synthetic north-south drain line (OSM waterways unavailable)"}
WEATHER_SOURCES = {"open_meteo": "Open-Meteo ERA5 reanalysis archive", "synthetic": "Synthetic Bengaluru climatology",
                   "forecast": "Open-Meteo live forecast", "design_storm": "Synthetic design storm",
                   "custom": "Custom series", "missing": "Missing hours (zero-filled gaps, not observations)"}
CSS = """
<style>
div.block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
.nf-legend {display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; align-items: center; font-size: 0.85rem;}
.nf-swatch {display: inline-block; width: 0.9rem; height: 0.9rem; border-radius: 3px; margin-right: 0.35rem;
            vertical-align: -0.12rem; border: 1px solid rgba(0,0,0,0.15);}
</style>
"""


# --------------------------------------------------------------------------- settings


def _choice(section: Mapping[str, Any], key: str, options: Iterable[str]) -> str:
    value, allowed = str(section[key]).strip().lower(), tuple(options)
    if value not in allowed:
        raise ConfigError(f"app.{key} must be one of {allowed}, got {section[key]!r}")
    return value


def _number(section: Mapping[str, Any], key: str, lo: float, hi: float, integer: bool = False) -> float | int:
    value = section[key]
    ok = not isinstance(value, bool)
    try:
        number = float(value)
    except (TypeError, ValueError):
        ok = False
        number = float("nan")
    ok = ok and math.isfinite(number) and lo <= number <= hi and (not integer or number == int(number))
    if not ok:
        kind = "an integer" if integer else "a number"
        raise ConfigError(f"app.{key} must be {kind} in [{lo:g}, {hi:g}], got {value!r}")
    return int(number) if integer else number


@dataclass(frozen=True)
class AppSettings:
    """Validated ``app`` config section (merged over :data:`DEFAULTS`)."""

    map_style: str
    column_radius_m: float
    elevation_scale: float
    top_k: int
    map_height_px: int
    default_mode: str
    notable_events: int
    color_scale: str
    pitch_deg: float
    forecast_ttl_s: int
    forecast_timeout_s: float
    forecast_max_retries: int
    max_mc_samples_ui: int
    custom_storm_max_mm: float
    custom_storm_max_h: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "AppSettings":
        """Validate ``cfg["app"]``; raises :class:`ConfigError` naming the offending key."""
        s = get_section(cfg, "app", DEFAULTS)
        ints = {key: _number(s, key, lo, hi, integer=True) for key, lo, hi in (
            ("top_k", 1, 200), ("map_height_px", 300, 1600), ("notable_events", 1, 100),
            ("forecast_ttl_s", 60, 86400), ("forecast_max_retries", 1, 10), ("max_mc_samples_ui", 1, 1000),
            ("custom_storm_max_h", 1, 72))}
        floats = {key: _number(s, key, lo, hi) for key, lo, hi in (
            ("column_radius_m", 1, 500), ("elevation_scale", 0.01, 1000), ("pitch_deg", 0, 85),
            ("forecast_timeout_s", 1, 300), ("custom_storm_max_mm", 10, 1000))}
        return cls(map_style=_choice(s, "map_style", ml.MAP_STYLES),
                   default_mode=_choice(s, "default_mode", MODE_LABELS),
                   color_scale=_choice(s, "color_scale", ml.COLOR_SCALES), **ints, **floats)


# --------------------------------------------------------------------------- formatting


format_prob = ml.format_probability
fmt_value = mv.fmt_value
Kpi = mv.Kpi
show_path = display_path


def public_text(text: Any) -> str:
    """``text`` with absolute paths of this machine removed (project root -> relative, home -> ``~``).

    Applied to every message the dashboard shows, so a hosted page never reveals the user name
    or the directory layout (exception messages carry absolute paths for the server log).
    """
    out = str(text)
    root = str(project_root())
    out = out.replace(root + os.sep, "").replace(root, ".")
    home = str(Path.home())
    return out.replace(home, "~") if len(home) > 1 else out


def fmt_number(value: Any, template: str) -> str:
    """``template.format(value)`` for finite numbers, ``n/a`` otherwise (NaN, None, text)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return template.format(number) if math.isfinite(number) else "n/a"


def fmt_time(ts: Any) -> str:
    """``Thu 04 Sep, 14:00`` (local time); empty for NaT / None."""
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime("%a %d %b, %H:%M")


def hour_label(ts: Any, index: int) -> str:
    """Scrubber label of target hour ``index`` (lead time counted from the issue hour)."""
    return f"{pd.Timestamp(ts):%a %d %b %H:%M} · lead {int(index) + 1} h"


def threshold_options(model_threshold: float) -> list[float]:
    """Log-spaced alert thresholds (1-2-5 steps from 1e-6, then 0.1 steps) plus the model's own threshold."""
    grid = [m * 10.0 ** e for e in range(-6, -1) for m in (1.0, 2.0, 5.0)]
    grid += [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
    model = float(model_threshold)
    if not (math.isfinite(model) and 0.0 < model < 1.0):
        model = 0.5
    rounded = {round(v, 12) for v in grid}
    return sorted(rounded | {model})


def threshold_label(value: float, model_threshold: float) -> str:
    """Slider label of an alert threshold; the model's own threshold is marked ``(model)``."""
    text = format_prob(value)
    return f"{text} (model)" if math.isclose(value, model_threshold, rel_tol=1e-9, abs_tol=0.0) else text


# --------------------------------------------------------------------------- KPI maths


@dataclass(frozen=True)
class Notice:
    """A banner shown under the header (fallbacks, missing model, stale model)."""

    level: str        # info | warning | error | success
    message: str


def history_notices(metadata: Mapping[str, Any]) -> list[Notice]:
    """A warning when the model had less rain history than it needs (the gap was treated as dry)."""
    padded = int(metadata.get("padded_history_h") or 0)
    if padded <= 0:
        return []
    return [Notice("warning", f"**Short rain history** — only {metadata.get('history_hours')} h of rain precede the "
                              f"first predicted hour, but the model needs {metadata.get('history_needed_h')} h "
                              f"(rain look-back + warm-up). The missing {padded} h were treated as dry, so the first "
                              "hours' probabilities may be too low; raise `inference.history_hours` or replay a "
                              "later start.")]


def _member_range_text(members: int, spread: tuple[int, int] | None) -> str:
    """Sentence on the junctions at risk in each rain-field realisation (empty without members)."""
    if spread is None:
        return ""
    return (f" Taken one at a time, the {members} rain-field realisations put {spread[0]}–{spread[1]} "
            "junctions at risk; the count above uses the ensemble-mean probability, which is what the map shows.")


def build_kpis(result: PredictionResult, *, hours: int, peak_mode: bool, hour_index: int,
               metrics: Mapping[str, Any] | None, predictor_kind: str,
               dataset_cfg: Mapping[str, Any] | None = None, exact_field: bool = True,
               areal: Mapping[str, Any] | None = None) -> list[Kpi]:
    """The five headline KPIs; the alert threshold is ``result.threshold``.

    ``exact_field`` (a replay of the training record) picks the skill tile "given the junction rain
    field"; otherwise the tile shows ``areal`` (:func:`metrics_view.areal_skill_summary`), the skill
    given only corridor-average rain.
    """
    h = max(1, min(int(hours), result.n_hours))
    summary = result.summary(h)
    index = int(np.clip(hour_index, 0, h - 1))
    shown = result.horizon_max(h) if peak_mode else result.prob[index]
    at_risk = int((shown >= result.threshold).sum())
    where = f"at any hour of the next {h} h" if peak_mode else f"at {fmt_time(result.timestamps[index])}"
    label = "Junctions at risk" if peak_mode else f"At risk at {result.timestamps[index]:%H:%M}"
    top = float(shown.max()) if shown.size else 0.0
    top_tier = ml.tier_names([top], result.risk_tiers)[0]
    peak = pd.Timestamp(summary["peak_time"]) if summary["peak_time"] else None
    lead = None if peak is None else int(result.timestamps.get_loc(peak)) + 1
    extremes = result.at_risk_range(h, None if peak_mode else index)
    spread = _member_range_text(result.n_members, extremes)
    share = f"{at_risk / max(result.n_nodes, 1):.0%} of junctions"
    delta = share if extremes is None else f"{share} · {extremes[0]}–{extremes[1]} across fields"
    return [
        Kpi(label, f"{at_risk} / {result.n_nodes}", delta,
            f"Junctions whose flood probability reaches the alert threshold ({format_prob(result.threshold)}) "
            f"{where}.{spread}"),
        Kpi("Max flood probability", format_prob(top), top_tier, f"Highest junction probability {where}."),
        Kpi("Peak hour", peak.strftime("%a %H:%M") if peak is not None else "n/a",
            None if lead is None else f"lead {lead} h",
            "Hour with the most junctions at risk (or the highest probability when none reach the threshold)."),
        Kpi("Rain over horizon", f"{summary['rain_total_mm']:.1f} mm", f"peak {summary['peak_rain_mm_h']:.1f} mm/h",
            f"Areal rainfall over the next {h} h (hour-ending totals)."),
        mv.pr_auc_kpi(metrics, predictor_kind, dataset_cfg, exact_field=exact_field, areal=areal),
    ]


# --------------------------------------------------------------------------- junction selection

TOPK_IDS_KEY = "_nf_topk_ids"
_LAST_CLICK_KEY, _LAST_ROWS_KEY = "_nf_last_click", "_nf_last_rows"


def _table_rows(event: Any) -> tuple[int, ...]:
    """Selected row positions from ``st.dataframe``'s selection state (dict or attribute style)."""
    selection = event.get("selection") if isinstance(event, Mapping) else getattr(event, "selection", None)
    rows = selection.get("rows") if isinstance(selection, Mapping) else getattr(selection, "rows", None)
    try:
        return tuple(int(r) for r in rows or ())
    except (TypeError, ValueError):
        return ()


def resolve_junction(state: MutableMapping[str, Any], order: Sequence[Any], *, map_key: str, table_key: str,
                     junction_key: str) -> Any:
    """The junction to inspect: a *new* map click or table-row pick wins, else the previous choice,
    else the riskiest (``order[0]``). Writes the choice to ``state[junction_key]`` (before the
    selectbox with that key is drawn) and remembers the last click / rows so a stale selection
    is not re-applied on every rerun."""
    if not order:
        raise ValueError("resolve_junction needs at least one junction")
    lookup = {str(node): node for node in order}
    clicked = ml.selected_node_id(state.get(map_key))
    if clicked != state.get(_LAST_CLICK_KEY):
        state[_LAST_CLICK_KEY] = clicked
        if clicked in lookup:
            state[junction_key] = lookup[clicked]
    rows, ids = _table_rows(state.get(table_key)), list(state.get(TOPK_IDS_KEY) or [])
    if rows != state.get(_LAST_ROWS_KEY):
        state[_LAST_ROWS_KEY] = rows
        if rows and 0 <= rows[0] < len(ids) and str(ids[rows[0]]) in lookup:
            state[junction_key] = lookup[str(ids[rows[0]])]
    if state.get(junction_key) not in set(order):
        state[junction_key] = order[0]
    return state[junction_key]


# --------------------------------------------------------------------------- frames


def rain_categories(kind: str, source: str, is_forecast: np.ndarray, is_target: np.ndarray) -> list[str]:
    """Hyetograph category per hour (observed vs forecast vs what-if)."""
    if kind == "forecast":
        return ["Forecast" if f else "Recent (model analysis)" for f in is_forecast]
    if kind == "historical":
        return ["Synthetic record" if source == "synthetic" else "Observed record"] * len(is_target)
    if kind == "design_storm":
        return ["Design storm" if t else "Dry lead-in" for t in is_target]
    return ["Rain"] * len(is_target)


def rain_frame(scenario: Scenario) -> pd.DataFrame:
    """Hourly areal rain of the whole scenario with local wall-clock ``time`` and a category."""
    frame = scenario.to_frame()
    return pd.DataFrame({
        "time": frame.index.tz_localize(None),
        "rain_mm": frame["precipitation_mm"].to_numpy(dtype=np.float64),
        "category": rain_categories(scenario.kind, scenario.source, frame["is_forecast"].to_numpy(),
                                    frame["is_target"].to_numpy()),
        "is_target": frame["is_target"].to_numpy(),
    })


def junction_timeline(result: PredictionResult, node_index: int, hours: int) -> pd.DataFrame:
    """Hourly probability (± MC std), junction rain and simulated depth of one junction."""
    h = max(1, min(int(hours), result.n_hours))
    if not 0 <= int(node_index) < result.n_nodes:
        raise IndexError(f"node_index {node_index} outside [0, {result.n_nodes})")
    prob = result.prob[:h, node_index].astype(np.float64)
    std = (np.nan_to_num(result.prob_std[:h, node_index].astype(np.float64), nan=0.0)
           if result.prob_std is not None else np.zeros(h))
    depth = (result.depth_physics[:h, node_index].astype(np.float64) if result.depth_physics is not None
             else np.full(h, np.nan))
    return pd.DataFrame({
        "time": result.timestamps[:h].tz_localize(None), "prob": prob,
        "lo": np.clip(prob - std, 0.0, 1.0), "hi": np.clip(prob + std, 0.0, 1.0),
        "rain_mm_h": result.node_rain[:h, node_index].astype(np.float64), "depth_m": depth,
    })


def tier_counts(prob: Any, tiers: Sequence[tuple[str, float]]) -> pd.DataFrame:
    """Junctions per risk tier (``tier``, ``junctions``, ``order``, hex ``color``), lowest tier first."""
    names = ml.tier_names(prob, tiers)
    palette = ml.tier_palette(tiers)
    rows = [{"tier": name, "junctions": int((names == name).sum()), "order": i,
             "color": "#%02x%02x%02x" % palette[name]} for i, (name, _) in enumerate(ml.validate_tiers(tiers))]
    return pd.DataFrame(rows)


def horizon_table(result: PredictionResult) -> pd.DataFrame:
    """Junctions at risk, max probability and rain for each configured horizon the result covers."""
    rows = []
    for h in result.horizons_h:
        if h > result.n_hours:
            continue
        top = result.horizon_max(h)
        rows.append({"Horizon": f"{h} h", "At risk": int((top >= result.threshold).sum()),
                     "Max p": format_prob(float(top.max()) if top.size else 0.0),
                     "Rain (mm)": round(result.summary(h)["rain_total_mm"], 1)})
    return pd.DataFrame(rows, columns=["Horizon", "At risk", "Max p", "Rain (mm)"])


# --------------------------------------------------------------------------- charts


def _time_axis() -> alt.Axis:
    return alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=6, title=None)


def _naive(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _marker_layer(marker: Any, color: str = "#d62728", shift: pd.Timedelta | None = None) -> alt.Chart | None:
    if marker is None or pd.isna(marker):
        return None
    ts = _naive(marker) + (shift if shift is not None else pd.Timedelta(0))
    return alt.Chart(pd.DataFrame({"time": [ts]})).mark_rule(color=color, strokeWidth=2).encode(x="time:T")


def target_band(target_span: tuple[Any, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``[first, last + 1 h)``: the x-extent of the bars of the target hours ``first .. last``.

    Hourly bars use Vega-Lite's ``yearmonthdatehours`` time unit, which draws the bar of
    timestamp ``t`` over ``[t, t + 1 h)``; the shading must cover exactly those bars.
    """
    start, end = (_naive(t) for t in target_span)
    return start, end + pd.Timedelta(hours=1)


def hyetograph_chart(frame: pd.DataFrame, *, marker: Any = None, target_span: tuple[Any, Any] | None = None,
                     height: int = 230) -> alt.LayerChart:
    """Hourly areal rain bars (coloured observed / forecast / what-if) with the target window and hour marker."""
    categories = list(dict.fromkeys(frame["category"])) or ["Rain"]
    scale = alt.Scale(domain=categories, range=[RAIN_COLORS.get(c, "#4C78A8") for c in categories])
    bars = alt.Chart(frame).mark_bar(opacity=0.9).encode(
        x=alt.X("yearmonthdatehours(time):T", axis=_time_axis()),
        y=alt.Y("rain_mm:Q", title="Areal rain (mm/h)"),
        color=alt.Color("category:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("time:T", title="Hour ending", format="%a %d %b %H:%M"),
                 alt.Tooltip("rain_mm:Q", title="Rain (mm)", format=".2f"), alt.Tooltip("category:N", title="Series")])
    layers: list[alt.Chart] = []
    if target_span is not None and not any(pd.isna(t) for t in target_span):
        start, end = target_band(target_span)
        span = pd.DataFrame({"start": [start], "end": [end]})
        layers.append(alt.Chart(span).mark_rect(color="#7f7f7f", opacity=0.10).encode(x="start:T", x2="end:T"))
    layers.append(bars)
    rule = _marker_layer(marker, shift=pd.Timedelta(minutes=30))  # the centre of the marked hour's bar
    if rule is not None:
        layers.append(rule)
    return alt.layer(*layers).properties(height=height)


def timeline_chart(frame: pd.DataFrame, *, threshold: float, marker: Any = None, height: int = 250
                   ) -> alt.LayerChart:
    """One junction: probability line (± MC std band, alert threshold) over junction rain bars."""
    top = float(np.nanmax(frame[["prob", "hi"]].to_numpy())) if len(frame) else 0.0
    fmt = ".0%" if max(top, threshold) >= 0.01 else ".1e"
    x = alt.X("time:T", axis=_time_axis())
    rain = alt.Chart(frame).mark_bar(color="#9ECAE9", opacity=0.55).encode(
        x=alt.X("yearmonthdatehours(time):T", axis=_time_axis()),
        y=alt.Y("rain_mm_h:Q", title="Junction rain (mm/h)", axis=alt.Axis(orient="right")))
    prob_layers: list[alt.Chart] = []
    if frame["hi"].sub(frame["lo"]).fillna(0).gt(0).any():
        prob_layers.append(alt.Chart(frame).mark_area(color="#d62728", opacity=0.18).encode(x=x, y="lo:Q", y2="hi:Q"))
    prob_layers.append(alt.Chart(frame).mark_line(
        color="#d62728", strokeWidth=2, point=alt.OverlayMarkDef(color="#d62728", filled=True, size=22)).encode(
        x=x, y=alt.Y("prob:Q", title="Flood probability", axis=alt.Axis(format=fmt)),
        tooltip=[alt.Tooltip("time:T", title="Hour ending", format="%a %d %b %H:%M"),
                 alt.Tooltip("prob:Q", title="p", format=".3g"), alt.Tooltip("rain_mm_h:Q", title="Rain", format=".2f"),
                 alt.Tooltip("depth_m:Q", title="Sim. depth (m)", format=".3f")]))
    prob_layers.append(alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color="#555", strokeDash=[5, 4]).encode(y="y:Q"))
    rule = _marker_layer(marker, "#333")
    if rule is not None:
        prob_layers.append(rule)
    return alt.layer(rain, alt.layer(*prob_layers)).resolve_scale(y="independent").properties(height=height)


def tier_chart(counts: pd.DataFrame, height: int = 150) -> alt.Chart:
    """Horizontal bars of :func:`tier_counts` in tier colours."""
    scale = alt.Scale(domain=list(counts["tier"]), range=list(counts["color"]))
    return alt.Chart(counts).mark_bar().encode(
        y=alt.Y("tier:N", sort=list(counts["tier"]), title=None),
        x=alt.X("junctions:Q", title="Junctions"),
        color=alt.Color("tier:N", scale=scale, legend=None),
        tooltip=["tier:N", "junctions:Q"]).properties(height=height)


def reliability_chart(frame: pd.DataFrame, height: int = 240) -> alt.LayerChart:
    """Observed flood frequency vs mean predicted probability, with the perfect-calibration diagonal."""
    diagonal = alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})).mark_line(
        color="#999", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    points = alt.Chart(frame).mark_line(point=True, color="#4C78A8").encode(
        x=alt.X("mean_predicted:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_freq:Q", title="Observed flood frequency", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("mean_predicted:Q", format=".3g"), alt.Tooltip("observed_freq:Q", format=".3g"),
                 alt.Tooltip("count:Q", title="Junction-hours")])
    return alt.layer(diagonal, points).properties(height=height)


def history_chart(history: pd.DataFrame, height: int = 240) -> alt.Chart:
    """Validation PR-AUC / ROC-AUC / F2 per epoch (``training_history.csv``)."""
    columns = [c for c in ("pr_auc", "roc_auc", "f2") if c in history.columns]
    long = history.melt(id_vars=["epoch"], value_vars=columns, var_name="metric", value_name="value").dropna()
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)), y=alt.Y("value:Q", title="Validation score"),
        color=alt.Color("metric:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=["epoch:Q", "metric:N", alt.Tooltip("value:Q", format=".4g")]).properties(height=height)


# --------------------------------------------------------------------------- model & provenance tables


def read_json_file(path: str | Path) -> dict[str, Any] | None:
    """A JSON object from ``path``; ``None`` when missing, unreadable or not an object (WARNING)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable JSON %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        LOGGER.warning("Ignoring %s: expected a JSON object, got %s", path, type(data).__name__)
        return None
    return data


def read_csv_file(path: str | Path) -> pd.DataFrame | None:
    """A CSV as a DataFrame; ``None`` when missing, empty or unreadable (WARNING)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable CSV %s: %s", path, exc)
        return None


def provenance_rows(graph: Mapping[str, Any], weather_meta: Mapping[str, Any] | None, scenario: Scenario,
                    label_source: str | None, n_reports: int | None,
                    rain_field: str | None = None) -> list[tuple[str, str]]:
    """Where every input of the current view comes from (``rain_field``: the run's junction-rain note)."""
    meta = weather_meta or {}
    record = "n/a"
    if meta:
        sources = ", ".join(WEATHER_SOURCES.get(k, k) for k in (meta.get("sources") or {})) or "unknown"
        span = f"{str(meta.get('start') or '?')[:10]} → {str(meta.get('end') or '?')[:10]}"
        record = f"{sources}; {span} ({meta.get('rows', '?')} h)"
    reports = "none" if not n_reports else f"{n_reports} observed reports on file"
    label = (label_source or "simulated")
    return [
        ("Road graph", f"{GRAPH_SOURCES.get(graph.get('source'), graph.get('source', 'unknown'))} — "
                       f"{graph.get('nodes', '?')} junctions, {graph.get('edges', '?')} directed edges"),
        ("Elevation", ELEVATION_SOURCES.get(graph.get("elevation_source"), str(graph.get("elevation_source")))),
        ("Drains", DRAIN_SOURCES.get(graph.get("drain_source"), str(graph.get("drain_source")))),
        ("Weather record", record),
        ("This scenario", f"{WEATHER_SOURCES.get(scenario.source, scenario.source)} — {scenario.description}"),
        ("Junction rain", _sentence(rain_field or scenario.rain_field_note)),
        ("Flood labels", f"{label} (hydrology simulator = the model's teacher; observed reports: {reports})"),
    ]


def _sentence(text: str) -> str:
    text = str(text).strip()
    return text[:1].upper() + text[1:] if text else text


# --------------------------------------------------------------------------- rendering


def inject_css() -> None:
    """Small layout tweaks (tighter top padding, legend swatches)."""
    st.markdown(CSS, unsafe_allow_html=True)


def render_notices(notices: Iterable[Notice]) -> None:
    """Draw each notice with the Streamlit alert of its level (machine paths removed)."""
    for notice in notices:
        draw = {"info": st.info, "warning": st.warning, "error": st.error, "success": st.success}.get(
            notice.level, st.info)
        draw(public_text(notice.message))


def render_kpis(kpis: Sequence[Kpi]) -> None:
    """A row of bordered ``st.metric`` tiles."""
    for column, kpi in zip(st.columns(len(kpis)), kpis):
        with column:
            st.metric(kpi.label, kpi.value, kpi.delta, help=kpi.help, border=True, delta_color=kpi.delta_color,
                      delta_arrow=kpi.delta_arrow)


def render_legend(scale: ml.ColorScale, note: str = "") -> None:
    """Colour swatches of the map's scale plus a one-line explanation."""
    items = "".join(f'<span><span class="nf-swatch" style="background: rgb{rgb}"></span>{label}</span>'
                    for label, rgb in scale.legend())
    st.markdown(f'<div class="nf-legend">{items}</div>', unsafe_allow_html=True)
    st.caption(f"{scale.describe()} {note}".strip())


def render_table_rows(rows: Sequence[tuple[str, str]]) -> None:
    """A two-column (item, value) table."""
    st.dataframe(pd.DataFrame(rows, columns=["Item", "Value"]), hide_index=True, width="stretch")


def render_about(metrics: Mapping[str, Any] | None, describe: Mapping[str, Any] | None,
                 history: pd.DataFrame | None, backtest: Mapping[str, Any] | None, metrics_path: str | Path,
                 current_hash: str | None = None, dataset_cfg: Mapping[str, Any] | None = None,
                 areal_skill: Mapping[str, Any] | None = None) -> None:
    """The "About the model" expander body (``describe`` is None when no trained GNN is loaded;
    ``areal_skill`` is ``areal_skill.json``)."""
    if describe is None:
        st.info("No trained GNN is loaded, so there are no model metrics to show: the dashboard runs the physics "
                "baseline (the hydrology simulator that generates the training labels). Train the model with "
                "`python src/training/train.py`; this panel then shows its held-out scores.")
        return
    if not metrics:
        st.info(f"No training metrics found at {show_path(metrics_path)}. Train the model with "
                "`python src/training/train.py` to populate this panel.")
        return
    note = mv.stale_model_note(metrics, current_hash, describe.get("dataset_config_hash"))
    if note:
        st.warning(note)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Model evaluation — {mv.headline_wording(metrics, dataset_cfg)}**")
        st.dataframe(mv.comparison_table(metrics, dataset_cfg), hide_index=True, width="stretch")
        st.caption(f"{mv.GIVEN_RAIN_NOTE} Baselines see the same per-junction-hour inputs without the road "
                   "graph or memory; their thresholds were chosen on validation. "
                   f"{mv.calibration_caption(metrics, describe, dataset_cfg)} A random ranking's PR-AUC is the "
                   f"flood rate ({fmt_value(metrics.get('pos_rate'))}).")
        render_areal_skill(areal_skill, describe)
        reliability = mv.reliability_table(metrics)
        if len(reliability):
            split = mv.split_short(metrics, mv.evaluation_split(metrics), dataset_cfg)
            st.markdown(f"**Reliability ({split}, calibrated probabilities)**")
            st.altair_chart(reliability_chart(reliability), width="stretch")
    with right:
        st.markdown("**Model card**")
        render_table_rows(mv.model_card(metrics, describe, dataset_cfg))
        if history is not None and {"epoch", "pr_auc"} <= set(history.columns) and len(history):
            st.markdown("**Training history** (validation, per epoch)")
            st.altair_chart(history_chart(history), width="stretch")
    _render_backtest(backtest, describe)


def render_areal_skill(report: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> None:
    """The "skill when only corridor-average rain is known" table (``areal_skill.json``) or why it is missing."""
    table = mv.areal_skill_table(report)
    note = mv.areal_skill_note(report, describe)
    if not len(table):
        st.caption(f"**Skill when only corridor-average rain is known:** {note}")
        return
    st.markdown(f"**{mv.areal_skill_title(report)}**")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"{mv.AREAL_CAPTION} {note or ''}".strip())


def _render_backtest(backtest: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> None:
    table = mv.backtest_table(backtest)
    if not len(table):
        return
    period = f"{str(backtest.get('start', ''))[:10]} → {str(backtest.get('end', ''))[:10]}"
    scored = ("the physics baseline" if ((backtest.get("predictor") or {}).get("kind") == "physics")
              else "the model")
    st.markdown(f"**Forecast backtest** ({period}: {scored} driven by archived Open-Meteo forecasts, scored "
                "against labels simulated from the observed rain)")
    st.dataframe(table, width="stretch")
    note = mv.backtest_note(backtest, describe)
    caption = ("0 h ahead = the observed rain with the exact label field (emulation of the teacher); its physics "
               f"column is the label generator itself ({mv.TEACHER_LABEL}). *Perfect areal forecast (ensemble)* = the "
               "same observed corridor-average rain with independent junction rain fields: the ceiling any rain "
               "forecast can reach. The 24 h / 48 h rows add the archived forecasts' rain error; their physics runs "
               "are forecast-driven baselines. Window columns count a junction as warned when it floods at some hour "
               "of the 6 h / 24 h block.")
    st.caption(f"{caption} {note or ''}".strip())
