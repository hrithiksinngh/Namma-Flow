"""Namma-Flow — interactive 3D flood-risk dashboard for Bengaluru's Bellandur / ORR corridor.

Run from the project root::

    streamlit run app/app.py

The sidebar picks a scenario (live Open-Meteo forecast, historical replay or what-if design
storm), the predictor (trained spatio-temporal GNN or the physics baseline), the horizon and
hour, the alert threshold, Monte-Carlo-dropout uncertainty and the map style. The main panel
shows headline KPIs, a 3D pydeck map of junction flood probabilities, the rainfall hyetograph,
the riskiest junctions with a per-junction timeline, downloads, the model card and the data
provenance. Every failure mode (no graph, no model, no weather record, forecast API down,
offline) degrades to a clear message or a documented fallback instead of a traceback.

Configuration: ``config/config.yaml`` (or ``$NAMMA_FLOW_CONFIG``), section ``app``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _entry in (str(ROOT), str(APP_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import banners  # noqa: E402
import components as ui  # noqa: E402
import loaders as ld  # noqa: E402
import map_layers as ml  # noqa: E402
import metrics_view as mv  # noqa: E402
from src.inference import predictor as pr  # noqa: E402
from src.inference import scenarios as sc  # noqa: E402
from src.inference.field_ensemble import field_plan  # noqa: E402
from src.inference.results import PredictionResult  # noqa: E402
from src.utils.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH, ENV_CONFIG, ENV_OFFLINE, ConfigError, deep_merge, is_offline, resolve_path,
)
from src.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("app")
DEBUG_ENV = "NAMMA_FLOW_DEBUG"

MAP_KEY, TABLE_KEY, JUNCTION_KEY = "nf_map", "nf_topk", "nf_junction"
AREAL_SKILL_FILE = "areal_skill.json"
CUSTOM = "__custom__"
SETUP_COMMANDS = """python src/data_pipeline/01_extract_network.py      # OSM road graph (+ elevation & drains)
python src/data_pipeline/02_elevation_engine.py     # optional: refresh elevation / drains
python src/data_pipeline/03_weather_ingestion.py    # 2018-2024 Open-Meteo rainfall record
python src/data_pipeline/04_dataset_builder.py      # training windows + simulated flood labels
python src/training/train.py                        # train the spatio-temporal GNN"""


class AppStop(RuntimeError):
    """A handled condition that ends the run with a message instead of the dashboard."""

    def __init__(self, message: str, level: str = "error") -> None:
        super().__init__(message)
        self.level = level


@dataclass(frozen=True)
class AppContext:
    """Per-run configuration: the loaded config, its cache key and the validated settings."""

    cfg: dict
    cfg_key: str
    settings: ui.AppSettings
    inference: sc.InferenceSettings
    offline: bool
    config_path: Path


@dataclass(frozen=True)
class ScenarioChoice:
    """What the sidebar asked for (resolved into a :class:`Scenario` by ``_resolve_scenario``)."""

    mode: str
    start: str | None = None
    total_mm: float | None = None
    duration_h: int | None = None
    offset_h: int | None = None
    rain_scale: float | None = None


@dataclass(frozen=True)
class TimeChoice:
    """Horizon (effective hours), peak-over-horizon toggle and the scrubbed hour."""

    hours: int
    peak: bool
    hour_index: int


@dataclass(frozen=True, eq=False)
class Display:
    """What the main panel shows: the alert-adjusted result and the per-junction values on the map."""

    view: PredictionResult
    hours: int
    index: int
    table: pd.DataFrame
    ranking: pd.DataFrame
    shown: np.ndarray
    std: np.ndarray | None
    when: str
    marker: pd.Timestamp


def _display(result: PredictionResult, timing: TimeChoice, alert: float) -> Display:
    view = dataclasses.replace(result, threshold=alert)
    if view.n_hours == 0 or view.n_nodes == 0:
        raise AppStop("The prediction is empty (no target hours or no junctions).", level="info")
    hours, index = min(timing.hours, view.n_hours), min(timing.hour_index, view.n_hours - 1)
    table = view.node_table(hours)
    peak_time = view.summary(hours)["peak_time"]
    if timing.peak:
        shown, std = table["max_prob"].to_numpy(), table["prob_std"].to_numpy()
        when, marker = f"peak over next {hours} h", pd.Timestamp(peak_time) if peak_time else view.timestamps[0]
    else:
        shown = view.prob[index].astype(np.float64)
        std = None if view.prob_std is None else view.prob_std[index].astype(np.float64)
        when, marker = f"at {view.timestamps[index]:%a %H:%M}", view.timestamps[index]
    return Display(view, hours, index, table, view.top_k(view.n_nodes, hours), shown, std, when, marker)


# --------------------------------------------------------------------------- small utilities


def _config_path() -> Path:
    path = Path(os.environ.get(ENV_CONFIG) or DEFAULT_CONFIG_PATH)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _debug() -> bool:
    """Tracebacks are shown in the page only when ``$NAMMA_FLOW_DEBUG`` is set (they stay in the server log)."""
    return os.environ.get(DEBUG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- context & setup


def _context() -> AppContext:
    path = _config_path()
    try:
        cfg = ld.cached_config(str(path), ld.stamp(path), os.environ.get(ENV_OFFLINE, ""))
        settings = ui.AppSettings.from_config(cfg)
        inference = sc.InferenceSettings.from_config(cfg)
    except ConfigError as exc:
        raise AppStop(f"**Invalid configuration** ({ui.show_path(path)}): {exc}") from exc
    return AppContext(cfg, ld.digest(cfg), settings, inference, is_offline(cfg), path)


def _configure_threads(ctx: AppContext) -> None:
    """Apply ``inference.num_threads`` (torch's intra-op setting is per thread and Streamlit runs every
    rerun in a fresh script thread, so this is done on each run; it is a cheap call)."""
    if ctx.inference.num_threads:
        import torch

        torch.set_num_threads(int(ctx.inference.num_threads))


def _graph_stamp(ctx: AppContext) -> tuple:
    return ld.stamp(resolve_path(ctx.cfg, "graph_file"))


def _checkpoint_stamp(ctx: AppContext) -> tuple:
    return ld.stamp(resolve_path(ctx.cfg, "checkpoint_dir") / ctx.inference.checkpoint_name)


def _weather_stamp(ctx: AppContext) -> tuple:
    return ld.stamp(resolve_path(ctx.cfg, "weather_file"))


def _setup_page(ctx: AppContext, reason: str) -> None:
    """Shown when there is no usable road graph: pipeline instructions + an offline demo builder."""
    st.title("Namma-Flow")
    st.error(ui.public_text(f"**No usable road graph yet.** {reason}"))
    st.markdown("Build the pipeline from the project root (each stage caches its output and works offline "
                "with synthetic fallbacks):")
    st.code(SETUP_COMMANDS, language="bash")
    path = resolve_path(ctx.cfg, "graph_file")
    st.markdown("Or build the graph **offline** right away: it is replayed from the osmnx response cache "
                f"(`{ui.show_path(resolve_path(ctx.cfg, 'osm_cache_dir'))}`, the real OpenStreetMap graph) if one "
                "exists, else a **synthetic street grid** is generated (elevation from a local DEM if one exists, else "
                f"a synthetic valley). It is written to `{ui.show_path(path)}`; replace it later with "
                "`python src/data_pipeline/01_extract_network.py --force`. An existing OpenStreetMap graph is never "
                "replaced by the synthetic grid.")
    if st.button("Build the graph offline", type="primary", key="nf_build_demo", icon=":material/construction:"):
        with st.spinner("Building the road graph offline (osmnx cache, else a synthetic grid)…"):
            try:
                from src.data_pipeline.network import extract_network

                extract_network(deep_merge(ctx.cfg, {"project": {"offline": True}}), force=True,
                                allow_synthetic=banners.may_replace_graph(path))
            except Exception as exc:  # noqa: BLE001 - shown to the user; the stage raises many types
                LOGGER.exception("Offline graph build failed")
                st.error(ui.public_text(f"Could not build the demo graph: {type(exc).__name__}: {exc}"))
                return
        st.cache_resource.clear()
        st.rerun()


def _load_physics(ctx: AppContext) -> pr.PhysicsPredictor | None:
    try:
        return ld.physics_predictor(ctx.cfg_key, _graph_stamp(ctx), ctx.cfg)
    except pr.GraphNotReady as exc:
        _setup_page(ctx, str(exc))
        return None


# --------------------------------------------------------------------------- sidebar: scenario


def _forecast_controls(ctx: AppContext) -> ScenarioChoice:
    st.caption(f"Open-Meteo hourly forecast at the corridor centre (cached for {ctx.settings.forecast_ttl_s // 60} "
               "min). If it is unavailable the most recent notable storm is replayed instead.")
    if ctx.offline:
        st.caption(":orange-badge[Offline] the live forecast cannot be fetched.")
    if st.button("Refresh forecast", key="nf_refresh", icon=":material/refresh:"):
        ld.forecast.clear()
    return ScenarioChoice("forecast")


def _custom_start(span: tuple[pd.Timestamp, pd.Timestamp]) -> str:
    first, last = span
    default = max(first, last - pd.Timedelta(days=2)).date()
    day = st.date_input("Start date", value=default, min_value=first.date(), max_value=last.date(), key="nf_day")
    day = day[0] if isinstance(day, (tuple, list)) and day else day
    day = default if day is None else day
    hour = st.slider("Start hour (local)", 0, 23, 12, key="nf_start_hour")
    start = pd.Timestamp(day).tz_localize(first.tz) + pd.Timedelta(hours=int(hour))
    return min(max(start, first), last).isoformat()


def _historical_controls(ctx: AppContext) -> ScenarioChoice:
    stamp = _weather_stamp(ctx)
    blocks = ld.record_blocks(ctx.cfg_key, stamp, ctx.cfg)
    if not blocks:
        raise AppStop("**No weather record to replay.** Build it with "
                      "`python src/data_pipeline/03_weather_ingestion.py`, or choose *What-if design storm*.")
    events = ld.events(ctx.cfg_key, stamp, ctx.settings.notable_events, ctx.cfg)
    labels = {pd.Timestamp(row.start).isoformat(): f"#{row.rank} · {row.label}" for row in events.itertuples()}
    labels[CUSTOM] = "Pick a date and hour…"
    pick = st.selectbox("Replay (wettest 24 h periods of the record)", list(labels), format_func=labels.get,
                        key="nf_event")
    start = _custom_start((blocks[0][0], blocks[-1][1])) if pick == CUSTOM else pick
    spans = ", ".join(f"{lo:%Y-%m-%d} → {hi:%Y-%m-%d}" for lo, hi in blocks)
    gaps = "" if len(blocks) == 1 else " Hours between the stored blocks are missing and cannot be replayed."
    st.caption(f"Record: {spans} (hourly areal rain).{gaps}")
    return ScenarioChoice("historical", start=start)


def _storm_size(ctx: AppContext, horizon: int) -> tuple[str, float, int]:
    presets = {p.name: p for p in ctx.inference.design_storms}
    options = [*presets, CUSTOM]
    pick = st.selectbox("Storm", options, format_func=lambda o: "Custom storm…" if o == CUSTOM else o, key="nf_storm")
    if pick != CUSTOM:
        return pick, presets[pick].total_mm, min(presets[pick].duration_h, horizon)
    most = float(ctx.settings.custom_storm_max_mm)
    total = st.slider("Total rainfall (mm, rain-gauge)", 5.0, most, min(60.0, most), 5.0, key="nf_total")
    longest = min(ctx.settings.custom_storm_max_h, horizon)
    duration = st.slider("Duration (h)", 1, longest, min(3, longest), key="nf_duration") if longest > 1 else 1
    return "Custom storm", float(total), int(duration)


def _design_controls(ctx: AppContext) -> ScenarioChoice:
    horizon = ctx.inference.max_horizon_h
    _, total, duration = _storm_size(ctx, horizon)
    latest = horizon - duration
    offset = min(ctx.inference.design_storm_offset_h, latest)
    if latest > 0:
        offset = st.slider("Storm starts after (h)", 0, latest, offset, key=f"nf_offset_{latest}")
    with st.expander("Advanced"):
        configured = float(ctx.inference.design_storm_rain_scale)
        scale = st.number_input(
            "Gauge → areal factor", min_value=0.001, max_value=max(2.0, configured), value=configured,
            step=0.01, format="%.3f", key="nf_rain_scale",
            help="Design totals are point (rain-gauge) amounts; the model and hydrology were calibrated on ERA5 areal "
                 "rain, which smooths convective storms several-fold. 1.0 feeds the storm as given.")
    st.caption(f"{total:g} mm over {duration} h ≈ {total * scale:.1f} mm areal-equivalent, starting "
               f"{offset} h after the next full hour.")
    return ScenarioChoice("design", total_mm=float(total), duration_h=int(duration), offset_h=int(offset),
                          rain_scale=float(scale))


def _scenario_controls(ctx: AppContext) -> ScenarioChoice:
    st.header("Scenario")
    modes = list(ui.MODE_LABELS)
    mode = st.radio("Mode", modes, index=modes.index(ctx.settings.default_mode), format_func=ui.MODE_LABELS.get,
                    key="nf_mode")
    if mode == "forecast":
        return _forecast_controls(ctx)
    if mode == "historical":
        return _historical_controls(ctx)
    return _design_controls(ctx)


def _latest_event_replay(ctx: AppContext, reason: str, hours: int) -> tuple[sc.Scenario, list[ui.Notice]]:
    stamp = _weather_stamp(ctx)
    events = ld.events(ctx.cfg_key, stamp, max(ctx.settings.notable_events, 50), ctx.cfg)
    if events.empty:
        raise AppStop(f"**Live forecast unavailable** ({reason}) and there is no weather record to replay. Build the "
                      "record with `python src/data_pipeline/03_weather_ingestion.py`, or choose *What-if design "
                      "storm*.", level="warning")
    latest = events.sort_values("start").iloc[-1]
    scenario = ld.historical(ctx.cfg_key, stamp, pd.Timestamp(latest["start"]).isoformat(), hours, ctx.cfg)
    notice = ui.Notice("warning", f"**Live forecast unavailable** — {reason}. Showing the most recent notable "
                                  f"storm in the record instead: {latest['label']}.")
    return scenario, [notice]


def _resolve_scenario(ctx: AppContext, choice: ScenarioChoice) -> tuple[sc.Scenario, list[ui.Notice]]:
    hours = ctx.inference.max_horizon_h
    try:
        if choice.mode == "forecast":
            bucket = int(time.time() // ctx.settings.forecast_ttl_s)
            quick = deep_merge(ctx.cfg, {"weather": {"timeout_s": ctx.settings.forecast_timeout_s,
                                                     "max_retries": ctx.settings.forecast_max_retries}})
            scenario, reason = ld.forecast(ctx.cfg_key, bucket, hours, quick)
            return (scenario, []) if scenario is not None else _latest_event_replay(ctx, str(reason), hours)
        if choice.mode == "historical":
            return ld.historical(ctx.cfg_key, _weather_stamp(ctx), str(choice.start), hours, ctx.cfg), []
        now = pd.Timestamp.now(tz=str(ctx.cfg["project"].get("timezone", "Asia/Kolkata"))).floor("h").isoformat()
        return ld.design(ctx.cfg_key, choice.total_mm, choice.duration_h, choice.offset_h, choice.rain_scale, now,
                       hours, ctx.cfg), []
    except (sc.ScenarioError, ValueError) as exc:
        raise AppStop(f"**Could not build the scenario:** {exc}") from exc


# --------------------------------------------------------------------------- sidebar: model, time, map


def _predictor_controls(physics: Any, gnn: Any, reason: str | None,
                        hint: str | None = None) -> tuple[Any, list[ui.Notice]]:
    st.header("Predictor")
    options = ["gnn", "physics"]
    if gnn is None:
        st.radio("Model", options, index=1, format_func=ui.PREDICTOR_LABELS.get, disabled=True,
                 key="nf_predictor_unavailable")
        fix = f" To use the GNN: {hint}, then use *Clear cached data* in the sidebar." if hint else ""
        message = (f"**Showing the physics baseline** — no usable trained model: {reason or 'unknown reason'}. The "
                   f"baseline is the hydrology simulator that generates the training labels.{fix}")
        return physics, [ui.Notice("warning", message)]
    choice = st.radio("Model", options, format_func=ui.PREDICTOR_LABELS.get, key="nf_predictor",
                      help="The physics baseline is the hydrology simulator that generated the GNN's training labels.")
    return (gnn if choice == "gnn" else physics), []


def _time_controls(ctx: AppContext, scenario: sc.Scenario, scenario_id: str) -> TimeChoice:
    st.header("Time")
    horizons = list(ctx.inference.horizons_h)
    picked = st.segmented_control("Horizon", horizons, default=horizons[-1], format_func=lambda h: f"{h} h",
                                  key="nf_horizon")
    requested = int(picked) if picked is not None else horizons[-1]
    hours = max(1, min(requested, scenario.n_target_hours))
    if hours < requested:
        st.caption(f"Only {hours} h of this scenario are available.")
    peak = st.toggle("Peak over horizon", value=True, key="nf_peak",
                     help="Show each junction's highest probability over the horizon instead of one hour.")
    stamps = scenario.target_timestamps[:hours]
    index = 0
    if hours > 1:
        index = st.select_slider("Hour", list(range(hours)), value=0, disabled=peak,
                                 format_func=lambda i: ui.hour_label(stamps[i], i),
                                 key=f"nf_hour_{scenario_id}_{hours}")
    return TimeChoice(hours, bool(peak), int(index))


def _alert_control(predictor: Any, members: int = 1) -> float:
    # Ensemble-averaged probabilities (forecast / design storm) have their own validation-chosen threshold.
    model = float(predictor.threshold_for_members(members)) if hasattr(predictor, "threshold_for_members") \
        else float(predictor.threshold)
    options = ui.threshold_options(model)
    value = min(options, key=lambda v: abs(v - model))
    picked = st.select_slider("Alert threshold", options, value=value,
                              format_func=lambda v: ui.threshold_label(v, model),
                              key=f"nf_alert_{predictor.kind}_{model:.6g}",
                              help="A junction is 'at risk' when its flood probability reaches this value. The default "
                                   "is the model's threshold, chosen on its calibrated validation probabilities.")
    return float(picked)


def _uncertainty_controls(ctx: AppContext, predictor: Any) -> int:
    is_gnn = predictor.kind == "gnn"
    on = st.toggle("MC-dropout uncertainty", value=False, disabled=not is_gnn, key="nf_mc",
                   help="Repeat the GNN with dropout active and report mean ± std (the physics baseline is "
                        "deterministic). Forecasts and what-if storms always average over "
                        f"{ctx.inference.field_members} random junction rain fields (± then also includes that "
                        "spread); replays use the exact training field.")
    cap = max(1, min(ctx.inference.max_mc_samples, ctx.settings.max_mc_samples_ui))
    samples = cap
    if cap > 2:
        default = int(np.clip(ctx.inference.mc_dropout_samples, 2, cap))
        samples = st.slider("MC samples", 2, cap, default, disabled=not (on and is_gnn), key="nf_mc_samples")
    return int(samples) if on and is_gnn else 0


def _map_controls(ctx: AppContext) -> tuple[ml.MapOptions, str]:
    st.header("Map")
    three_d = st.toggle("3D columns", value=True, key="nf_3d")
    color = st.radio("Colour scale", ml.COLOR_SCALES, index=ml.COLOR_SCALES.index(ctx.settings.color_scale),
                     format_func=ui.COLOR_SCALE_LABELS.get, horizontal=True, key="nf_color",
                     help="Auto uses risk tiers when some junction reaches the Moderate tier, otherwise a scale "
                          "relative to the highest probability / alert threshold.")
    style = st.selectbox("Basemap", ml.MAP_STYLES, index=ml.MAP_STYLES.index(ctx.settings.map_style),
                         format_func=str.title, key="nf_basemap")
    roads = st.toggle("Road segments", value=True, key="nf_roads")
    drains = st.toggle("Drains & lakes", value=True, key="nf_drains")
    options = ml.MapOptions(three_d=bool(three_d), show_roads=bool(roads), show_drains=bool(drains), map_style=style,
                            column_radius_m=ctx.settings.column_radius_m, elevation_scale=ctx.settings.elevation_scale,
                            pitch_deg=ctx.settings.pitch_deg)
    return options, str(color)


def _sidebar_footer(ctx: AppContext) -> None:
    st.divider()
    if ctx.offline:
        st.markdown(":orange-badge[:material/cloud_off: Offline mode] no network calls are made.")
    st.caption(f"Config: `{ui.show_path(ctx.config_path)}`")
    if st.button("Clear cached data", key="nf_clear", icon=":material/delete_sweep:",
                 help="Reload the graph, model, weather record and forecast from disk / the API."):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# --------------------------------------------------------------------------- main panel


def _weather_meta(ctx: AppContext) -> dict | None:
    meta_path = resolve_path(ctx.cfg, "weather_file").with_suffix(".meta.json")
    return ld.json_file(str(meta_path), ld.stamp(meta_path))


def _header(ctx: AppContext, scenario: sc.Scenario, predictor: Any, mc: int) -> None:
    st.title("Namma-Flow")
    st.caption(banners.header_caption(predictor.provenance, predictor.kind))
    mode = "design" if scenario.kind == "design_storm" else scenario.kind
    badges = [f":blue-badge[{ui.PREDICTOR_LABELS.get(predictor.kind, predictor.kind)}]",
              f":violet-badge[{ui.MODE_LABELS.get(mode, scenario.kind)}]"]
    members = field_plan(ctx.cfg, ctx.inference, scenario.exact_field).members
    if members > 1:
        badges.append(f":gray-badge[{members} rain fields]")
    if mc:
        badges.append(f":green-badge[MC dropout × {mc}]")
    if ctx.offline:
        badges.append(":orange-badge[Offline]")
    st.markdown(" ".join(badges) + f" &nbsp; **{scenario.name}** — {scenario.description}")


def _current_dataset_hash(ctx: AppContext) -> str | None:
    try:
        from src.data_pipeline.dataset import dataset_config_hash

        return dataset_config_hash(ctx.cfg)
    except Exception as exc:  # noqa: BLE001 - diagnostics only; never block the dashboard
        LOGGER.debug("Could not compute the dataset config hash: %s", exc)
        return None


def _model_metrics(ctx: AppContext, gnn: Any) -> dict | None:
    """Metrics of the LOADED model: its checkpoint's own metrics when they differ from metrics.json
    (e.g. a stale report), else metrics.json. Nothing when no GNN is loaded (R4-02)."""
    if gnn is None:
        return None
    path = resolve_path(ctx.cfg, "reports_dir") / "metrics.json"
    from_file = ld.json_file(str(path), ld.stamp(path))
    from_ckpt = dict(gnn.metrics)
    if from_ckpt and (not from_file or from_file.get("created_utc") != from_ckpt.get("created_utc")):
        return from_ckpt
    return from_file or from_ckpt or None


def _water(ctx: AppContext, provenance: dict) -> ml.WaterLayerData | None:
    """OSM drains & lakes when the graph used them (minus the water bodies the pipeline excludes, parsed
    and validated by the drains module itself), else the synthetic drain line the pipeline assumed."""
    if provenance.get("drain_source") == "osm":
        path = resolve_path(ctx.cfg, "waterways_file")
        from src.data_pipeline import drains

        try:
            excluded = drains._exclude_values(drains._section(ctx.cfg))
        except ConfigError as exc:
            raise AppStop(f"**Invalid configuration** ({ui.show_path(ctx.config_path)}): {exc}") from exc
        return ld.waterways(str(path), ld.stamp(path), tuple(sorted(excluded)))
    region = ctx.cfg.get("region") or {}
    try:
        return ml.synthetic_drain_line(region.get("drain_fallback_lon", 77.675), region.get("bbox"))
    except (ValueError, TypeError):
        return None


def _render_map(ctx: AppContext, shown: Display, scale: ml.ColorScale, options: ml.MapOptions, selected: Any,
                predictor: Any) -> None:
    graph, view = predictor.graph, shown.view
    junctions = ml.junction_frame(shown.table, shown.shown, scale, std=shown.std, when=shown.when)
    segments = ml.undirected_segments(graph.edge_index, graph.num_nodes)
    roads = ml.road_frame(view.lon, view.lat, segments, shown.shown, scale)
    water = _water(ctx, predictor.provenance)
    deck = ml.build_deck(junctions, options, roads=roads, water=water, selected_node=selected)
    st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key=MAP_KEY,
                    height=ctx.settings.map_height_px)
    note = "" if water is None else f"Blue: {water.note}."
    ui.render_legend(scale, f"Click a junction to inspect it. {note}")


def _render_rain_and_tiers(scenario: sc.Scenario, view: PredictionResult, shown: np.ndarray, hours: int,
                           marker: Any) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Rainfall")
        span = (view.timestamps[0], view.timestamps[hours - 1])
        st.altair_chart(ui.hyetograph_chart(ui.rain_frame(scenario), marker=marker, target_span=span), width="stretch")
        note = str(view.metadata.get("rain_field") or scenario.rain_field_note)
        st.caption("Shaded: predicted hours in the selected horizon; red line: the hour shown on the map. "
                   f"Rain field: {note}.")
    with right:
        st.subheader("Risk distribution")
        st.altair_chart(ui.tier_chart(ui.tier_counts(shown, view.risk_tiers)), width="stretch")
        st.dataframe(ui.horizon_table(view), hide_index=True, width="stretch")


def _topk_table(view: PredictionResult, ranking: pd.DataFrame, k: int) -> None:
    st.subheader(f"Top {min(k, len(ranking))} riskiest junctions")
    top = ranking.head(k)
    st.session_state[ui.TOPK_IDS_KEY] = list(top["node_id"])
    display = pd.DataFrame({
        "Rank": top["rank"], "Junction": top["node_id"].astype(str), "Max p": top["max_prob"],
        "± std": top["prob_std"], "Fields at risk": top["field_share_at_risk"], "Tier": top["tier"],
        "Peak": [ui.fmt_time(t) for t in top["peak_time"]],
        "Rel. elev. (m)": top["relative_elevation"], "To drain (m)": top["dist_to_drain_m"],
        "Sim. depth (m)": top["max_depth_m"]})
    optional = ("± std", "Fields at risk", "Sim. depth (m)")
    display = display.drop(columns=[c for c in optional if display[c].isna().all()])
    p_format = "percent" if float(top["max_prob"].max() if len(top) else 0.0) >= 0.001 else "scientific"
    config = {"Max p": st.column_config.NumberColumn(format=p_format),
              "± std": st.column_config.NumberColumn(format=p_format),
              "Fields at risk": st.column_config.NumberColumn(
                  format="percent", help="Share of the rain-field realisations in which the junction reaches the "
                                         "alert threshold"),
              "Rel. elev. (m)": st.column_config.NumberColumn(format="%+.2f"),
              "To drain (m)": st.column_config.NumberColumn(format="%.0f"),
              "Sim. depth (m)": st.column_config.NumberColumn(format="%.2f")}
    st.dataframe(display, hide_index=True, width="stretch", column_config=config, on_select="rerun",
                 selection_mode="single-row", key=TABLE_KEY)
    spread = (" ± = spread across the rain-field realisations (and MC-dropout passes)."
              if view.prob_std is not None else "")
    st.caption(f"Ranked by peak probability over the horizon (alert threshold {ui.format_prob(view.threshold)})."
               f"{spread} Select a row to show its timeline.")


def _junction_panel(view: PredictionResult, ranking: pd.DataFrame, hours: int, marker: Any,
                    provenance: dict) -> None:
    st.subheader("Junction timeline")
    rows = {node: row for node, row in zip(ranking["node_id"], ranking.itertuples(index=False))}
    labels = {node: f"#{row.rank} · {node} · p {ui.format_prob(row.max_prob)} ({row.tier})"
              for node, row in rows.items()}
    node = st.selectbox("Junction", list(rows), format_func=labels.get, key=JUNCTION_KEY)
    row = rows[node]
    frame = ui.junction_timeline(view, view.node_ids.index(node), hours)
    st.altair_chart(ui.timeline_chart(frame, threshold=view.threshold, marker=marker), width="stretch")
    spread = "" if pd.isna(row.prob_std) else f" ± {ui.format_prob(row.prob_std)}"
    depth = "" if pd.isna(row.max_depth_m) else f" · simulated peak depth {row.max_depth_m:.2f} m"
    if not pd.isna(row.field_share_at_risk):
        depth += f" · at risk in {row.field_share_at_risk:.0%} of {view.n_members} rain fields"
    st.markdown(f"**Peak p {ui.format_prob(row.max_prob)}{spread}** ({row.tier}) at {ui.fmt_time(row.peak_time)}"
                f"{depth}  \nElevation {ui.fmt_number(row.elevation, '{:.1f} m')} · relative "
                f"{ui.fmt_number(row.relative_elevation, '{:+.2f} m')} · "
                f"{ui.fmt_number(row.dist_to_drain_m, '{:,.0f} m')} to the nearest drain · "
                f"({row.lat:.5f}, {row.lon:.5f})")
    if str(provenance.get("source", "")).startswith("osm"):
        st.link_button("Open in OpenStreetMap", f"https://www.openstreetmap.org/node/{node}", icon=":material/map:")


def _downloads(scenario: sc.Scenario, view: PredictionResult, prediction_id: str, hours: int) -> None:
    base = f"namma_flow_{scenario.kind}_{view.timestamps[0]:%Y%m%d_%H%M}_{hours}h"
    table = view.node_table(hours)
    table["peak_time"] = [None if pd.isna(t) else t.isoformat() for t in table["peak_time"]]
    cols = st.columns(3)
    cols[0].download_button("Download GeoJSON", ld.geojson_bytes(prediction_id, hours, view.threshold, view),
                            file_name=f"{base}.geojson", mime="application/geo+json", icon=":material/download:")
    cols[1].download_button("Download junction CSV", table.to_csv(index=False).encode("utf-8"),
                            file_name=f"{base}.csv", mime="text/csv", icon=":material/table:")
    cols[2].download_button("Download rainfall CSV", scenario.to_frame().to_csv().encode("utf-8"),
                            file_name=f"{base}_rain.csv", mime="text/csv", icon=":material/rainy:")


def _areal_report(ctx: AppContext) -> dict | None:
    path = resolve_path(ctx.cfg, "reports_dir") / AREAL_SKILL_FILE
    return ld.json_file(str(path), ld.stamp(path))


def _about(ctx: AppContext, gnn: Any, metrics: dict | None, scenario: sc.Scenario, predictor: Any,
           view: PredictionResult) -> None:
    reports = resolve_path(ctx.cfg, "reports_dir")
    with st.expander("About the model", icon=":material/model_training:"):
        history_path, backtest_path = reports / "training_history.csv", reports / "backtest.json"
        ui.render_about(metrics, gnn.describe() if gnn is not None else None,
                        ld.csv_file(str(history_path), ld.stamp(history_path)),
                        ld.json_file(str(backtest_path), ld.stamp(backtest_path)), reports / "metrics.json",
                        _current_dataset_hash(ctx), ctx.cfg.get("dataset"), _areal_report(ctx))
    with st.expander("Data provenance", icon=":material/source:"):
        reports_file = resolve_path(ctx.cfg, "flood_reports_file")
        observed = ld.csv_file(str(reports_file), ld.stamp(reports_file))
        graph = {**predictor.provenance, "nodes": predictor.graph.num_nodes, "edges": predictor.graph.num_edges}
        label_source = ((metrics or {}).get("dataset") or {}).get("label_source") or (ctx.cfg.get("labels") or {}).get(
            "source")
        ui.render_table_rows(ui.provenance_rows(graph, _weather_meta(ctx), scenario, label_source,
                                                None if observed is None else len(observed),
                                                view.metadata.get("rain_field")))


def _dashboard(ctx: AppContext, scenario: sc.Scenario, predictor: Any, gnn: Any, shown: Display, metrics: dict | None,
               options: ml.MapOptions, color: str, prediction_id: str) -> None:
    view = shown.view
    scale = ml.resolve_color_scale(color, float(np.max(shown.shown)), view.threshold, view.risk_tiers)
    selected = ui.resolve_junction(st.session_state, list(shown.ranking["node_id"]), map_key=MAP_KEY,
                                   table_key=TABLE_KEY, junction_key=JUNCTION_KEY)
    areal = None
    if gnn is not None and not scenario.exact_field:
        areal = mv.areal_skill_summary(_areal_report(ctx), metrics, gnn.describe())
    ui.render_kpis(ui.build_kpis(view, hours=shown.hours, peak_mode=shown.when.startswith("peak"),
                                 hour_index=shown.index, metrics=metrics, predictor_kind=predictor.kind,
                                 dataset_cfg=ctx.cfg.get("dataset"), exact_field=scenario.exact_field, areal=areal))
    _render_map(ctx, shown, scale, options, selected, predictor)
    _render_rain_and_tiers(scenario, view, shown.shown, shown.hours, shown.marker)
    left, right = st.columns(2)
    with left:
        _topk_table(view, shown.ranking, ctx.settings.top_k)
    with right:
        _junction_panel(view, shown.ranking, shown.hours, shown.marker, predictor.provenance)
    _downloads(scenario, view, prediction_id, shown.hours)
    _about(ctx, gnn, metrics, scenario, predictor, view)


def _run() -> None:
    ctx = _context()
    _configure_threads(ctx)
    physics = _load_physics(ctx)
    if physics is None:
        return
    gnn, reason, hint = ld.gnn_predictor(ctx.cfg_key, _graph_stamp(ctx), _checkpoint_stamp(ctx), ctx.cfg)
    top = st.container()
    with st.sidebar:
        choice = _scenario_controls(ctx)
    scenario, notices = _resolve_scenario(ctx, choice)
    scenario_id = ld.scenario_id(scenario)
    with st.sidebar:
        predictor, predictor_notices = _predictor_controls(physics, gnn, reason, hint)
        timing = _time_controls(ctx, scenario, scenario_id)
        st.header("Alerts & uncertainty")
        members = 1 if scenario.exact_field else int(ctx.inference.field_members)
        alert = _alert_control(predictor, members)
        mc = _uncertainty_controls(ctx, predictor)
        options, color = _map_controls(ctx)
        _sidebar_footer(ctx)
    metrics = _model_metrics(ctx, gnn)
    if predictor.kind == "gnn":
        stale = mv.stale_model_note(metrics, _current_dataset_hash(ctx), gnn.describe().get("dataset_config_hash"))
        if stale:
            predictor_notices.append(ui.Notice("info", stale))
    synthetic = banners.synthetic_notice(banners.synthetic_inputs(predictor.provenance, scenario))
    with top:
        _header(ctx, scenario, predictor, mc)
        ui.render_notices([*([synthetic] if synthetic else []), *notices, *predictor_notices,
                           *(ui.Notice("info", n) for n in scenario.notes)])
    stamps = (_graph_stamp(ctx), _checkpoint_stamp(ctx)) if predictor.kind == "gnn" else (_graph_stamp(ctx),)
    predictor_id = f"{predictor.kind}|{ctx.cfg_key}|{stamps}"
    try:
        result = ld.predict(predictor_id, scenario_id, mc, predictor, scenario)
    except (pr.PredictionError, pr.GraphNotReady, ValueError, RuntimeError) as exc:
        raise AppStop(f"**Prediction failed:** {exc}") from exc
    with top:
        ui.render_notices(ui.history_notices(result.metadata))
    _dashboard(ctx, scenario, predictor, gnn, _display(result, timing, alert), metrics, options, color,
               f"{predictor_id}|{scenario_id}|{mc}")


def main() -> None:
    """Page entry point: every handled condition ends in a message; a traceback is shown only with
    ``NAMMA_FLOW_DEBUG=1`` (it is always written to the server log)."""
    st.set_page_config(page_title="Namma-Flow · Bengaluru flood risk", page_icon=":material/flood:", layout="wide",
                       initial_sidebar_state="expanded")
    ui.inject_css()
    try:
        _run()
    except AppStop as stop:
        draw = {"error": st.error, "warning": st.warning, "info": st.info}.get(stop.level, st.error)
        draw(ui.public_text(stop))
    except Exception as exc:  # noqa: BLE001 - last line of defence: a readable message, details in the log
        LOGGER.exception("Unexpected dashboard error")
        st.error(ui.public_text(f"**Something went wrong while building the dashboard:** {type(exc).__name__}: "
                                f"{exc}. Try *Clear cached data* in the sidebar; details are in the server log."))
        if _debug():  # never expose tracebacks (paths, code) on a hosted page unless explicitly asked to
            with st.expander("Technical details"):
                st.code(traceback.format_exc(), language="text")


main()
