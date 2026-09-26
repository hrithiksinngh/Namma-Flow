"""Calibration diagnostics for the hydrology label simulator.

Usage (from the project root)::

    python -m src.hydrology.calibrate [--config PATH] [--offline] [--set KEY=VALUE ...]
                                      [--json PATH] [--strict] [--no-uniform]

Runs the full label chain exactly as the dataset builder does — road graph
(``paths.graph_file``) -> hourly areal rain (weather cache / archive / synthetic) -> junction
rain field (:func:`downscale_rainfall`) -> simulator — and prints the statistics the
``hydrology`` / ``rainfall_field`` defaults were tuned against. It is a read-only diagnostic:
a missing graph or weather record is built in memory (stage 01 / 03 logic with
``persist=False``) and never written to ``paths.graph_file`` / ``paths.weather_file``, so an
offline run on a fresh clone cannot leave a synthetic grid or record behind (F5-08):

(a) flooded node-hour rate over May-Nov in [0.5 %, 3 %];
(b) no flooded junction in hours whose trailing 6 h areal rain is < 4 mm;
(c) the Sept 4-5 2022 Outer Ring Road event floods 10-50 % of junctions at its peak;
(d) floods concentrate at low relative elevation / high flow accumulation / sinks (correlations);
(e) >= 15 distinct rain events per year on average produce some flooding;
(f) degeneracy: a rank-1 (hour state x node susceptibility) fit explains < 60 % of the label
    variance over flood-season hours;
(g) spatial rain matters: the Jaccard index of the season flood labels with the rain field
    disabled (areal rain broadcast to every junction) vs enabled is < 0.75;
(h) in hours with any flooding, the median share of junctions flooded is < 25 %;
plus the rain-field target (rf): median within-hour coefficient of variation of junction rain
over wet hours (areal >= 1 mm/h) in [0.3, 0.7] and p95 of the max/mean multiplier >= 2.
A separable "basin rain x node frequency" lookup PR-AUC on the last year is reported too.

``--set`` overrides a ``hydrology`` key (``--set spill_depth_m=0.5``) or any dotted config key
(``--set rainfall_field.n_cells=4``) for what-if calibration runs; values are parsed as YAML.
``--strict`` makes the exit code 1 when a criterion fails; ``--no-uniform`` skips the second
(uniform-rain) simulation, so (g) is reported as n/a.

Calibration record (2026-09-23, fix round; real osm_bbox graph, 1 035 junctions, SRTM elevation,
real OSM drains; Open-Meteo/ERA5 2018-2024, 61 368 h; ~10 s and 1.5 GB per evaluation)::

    criterion                     before (basin switch)   after (defaults)   target
    (a) May-Nov flooded node-h    0.774 %                 0.591 %            0.5-3 %
    (b) floods, 6 h areal < 4 mm  0                       0                  0
    (c) Sept 4-5 2022 peak        61.9 %                  42.4 %             10-50 %
    (d) Spearman relev/accum/sink -0.44 / +0.60 / +0.75   -0.45 / +0.60 / +0.75  <0 / >0 / >0
    (e) flooding events per year  23.9                    41.1               >= 15
    (f) rank-1 R2 (May-Nov)       0.806                   0.502              < 0.6
    (g) Jaccard vs uniform rain   0.953                   0.464              < 0.75
    (h) median share, flood hours 32.2 %                  8.3 %              < 25 %
    (rf) rain CV median / p95 max 0.083 / 1.47            0.464 / 3.93       0.3-0.7 / >= 2
    lookup PR-AUC (diagnostic)    0.929                   0.555              (lower = less separable)

Ablations of the defaults: ``surcharge_local_weight: 0`` (legacy basin-wide loading, keeping
the new rain field and gate) gives f 0.564 / g 0.555 / h 12.1 %; no tailwater gate breaks (b)
(3 389 node-hours); a constant stratiform share gives f 0.562 / g 0.556; uniform rain gives
f 0.747. Every criterion still passes with any single hydrology key scaled by 0.8 or 1.2
except (b), which picks up 1-69 node-hours for ponding_fraction x1.2, spill_depth_m x1.2 and
tailwater_threshold_mm x0.8: (b) is a hard zero, and the tailwater gate is deliberately set
just below the 4 mm / 6 h criterion.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from src.data_pipeline.graph_io import GraphArrays, graph_to_arrays
from src.hydrology.label_diagnostics import (
    flooded_share_stats,
    label_jaccard,
    rain_field_heterogeneity,
    rank1_r2,
    separable_lookup_pr_auc,
)
from src.hydrology.simulator import HydrologyParams, SimulationError, UrbanDrainageSimulator
from src.utils.config import ConfigError, deep_merge, get_section, load_config
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

SEASON_MONTHS = (5, 6, 7, 8, 9, 10, 11)
TARGET_RATE = (0.005, 0.03)
DRY_WINDOW_H = 6
DRY_RAIN_MM = 4.0
SEPT_2022 = ("2022-09-04 00:00", "2022-09-05 23:00")
PEAK_FRACTION_RANGE = (0.10, 0.50)
MIN_EVENTS_PER_YEAR = 15.0
MAX_RANK1_R2 = 0.60
MAX_UNIFORM_JACCARD = 0.75
MAX_MEDIAN_FLOODED_SHARE = 0.25
RAIN_CV_RANGE = (0.3, 0.7)
MIN_RAIN_PEAK_RATIO_P95 = 2.0
EVENT_TAIL_H = 6
HOURS_PER_YEAR = 365.25 * 24
STATIC_FEATURES = ("relative_elevation", "flow_accumulation", "is_sink", "elevation", "dist_to_drain_m")

CHECK_NAMES = (
    "a_season_rate_in_range",
    "b_no_floods_below_4mm_6h",
    "c_sept2022_peak_10_to_50pct",
    "d_floods_cluster_low_accumulating_sinks",
    "e_ge_15_flood_events_per_year",
    "f_rank1_r2_below_0_6",
    "g_uniform_rain_jaccard_below_0_75",
    "h_median_flooded_share_below_25pct",
    "rf_rain_field_heterogeneous",
)


# --------------------------------------------------------------------------- statistics


@dataclass(frozen=True)
class CalibrationStats:
    """Headline calibration statistics (criteria a-h + rain field) plus provenance of the run."""

    n_nodes: int
    n_hours: int
    period: tuple[str, str]
    season_rate: float
    season_node_hours: int
    dry_flood_node_hours: int
    dry_flood_hours: int
    sept2022_peak_fraction: float | None
    sept2022_peak_time: str | None
    sept2022_rain_mm: float | None
    correlations: dict[str, dict[str, float | None]]
    group_rates: dict[str, float | None]
    n_events: int
    events_with_flooding: int
    events_per_year_with_flooding: float
    events_with_flooding_by_year: dict[int, int]
    n_years: float
    rank1_r2: float | None = None
    uniform_jaccard: float | None = None
    flooded_share: dict[str, float | int | None] = field(default_factory=dict)
    rain_field: dict[str, float | int | None] = field(default_factory=dict)
    lookup_pr_auc: dict[str, float | int | None] = field(default_factory=dict)
    runtime_s: dict[str, float] = field(default_factory=dict)
    peak_depth_m: float | None = None
    mass_balance_error: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def checks(self) -> dict[str, bool | None]:
        """Pass/fail of each criterion (``None`` = not applicable / not computed for this run)."""
        corr = self.correlations
        relev = (corr.get("relative_elevation") or {}).get("spearman")
        accum = (corr.get("flow_accumulation") or {}).get("spearman")
        sink = (corr.get("is_sink") or {}).get("spearman")
        clustered = None if None in (relev, accum, sink) else bool(relev < 0 and accum > 0 and sink > 0)
        peak = self.sept2022_peak_fraction
        share = self.flooded_share.get("median")
        values = (
            bool(TARGET_RATE[0] <= self.season_rate <= TARGET_RATE[1]),
            bool(self.dry_flood_node_hours == 0),
            None if peak is None else bool(PEAK_FRACTION_RANGE[0] <= peak <= PEAK_FRACTION_RANGE[1]),
            clustered,
            bool(self.events_per_year_with_flooding >= MIN_EVENTS_PER_YEAR),
            None if self.rank1_r2 is None else bool(self.rank1_r2 < MAX_RANK1_R2),
            None if self.uniform_jaccard is None else bool(self.uniform_jaccard < MAX_UNIFORM_JACCARD),
            None if share is None else bool(share < MAX_MEDIAN_FLOODED_SHARE),
            _rain_field_ok(self.rain_field),
        )
        return dict(zip(CHECK_NAMES, values))

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["events_with_flooding_by_year"] = {str(k): v for k, v in self.events_with_flooding_by_year.items()}
        out["checks"] = self.checks()
        return out

    def format_lines(self) -> list[str]:
        """Human-readable report, one criterion per line (see the module docstring)."""
        mark = {True: "PASS", False: "FAIL", None: "n/a "}
        c = {name: mark[value] for name, value in self.checks().items()}
        lines = [
            f"record                : {self.period[0]} -> {self.period[1]} "
            f"({self.n_hours} h, {self.n_nodes} junctions)",
            f"[{c['a_season_rate_in_range']}] (a) flooded node-hours May-Nov : "
            f"{100 * self.season_rate:.3f} % (target {100 * TARGET_RATE[0]:.1f}-{100 * TARGET_RATE[1]:.0f} %)",
            f"[{c['b_no_floods_below_4mm_6h']}] (b) floods with 6 h areal rain < {DRY_RAIN_MM:g} mm : "
            f"{self.dry_flood_node_hours} node-hours in {self.dry_flood_hours} hours (target 0)",
            f"[{c['c_sept2022_peak_10_to_50pct']}] (c) Sept 4-5 2022 peak : {self._peak_text()} "
            f"(target {100 * PEAK_FRACTION_RANGE[0]:.0f}-{100 * PEAK_FRACTION_RANGE[1]:.0f} %)",
            f"[{c['d_floods_cluster_low_accumulating_sinks']}] (d) node flood frequency vs static features "
            "(Spearman / Pearson):",
        ]
        for name, values in self.correlations.items():
            lines.append(f"        {name:<20}: {_fmt(values.get('spearman'))} / {_fmt(values.get('pearson'))}")
        lines.append("        flood rate by group : " + ", ".join(
            f"{k}={_fmt_pct(v)}" for k, v in self.group_rates.items()))
        lines.append(
            f"[{c['e_ge_15_flood_events_per_year']}] (e) rain events producing floods : "
            f"{self.events_with_flooding}/{self.n_events} = {self.events_per_year_with_flooding:.1f} per year "
            f"(target >= {MIN_EVENTS_PER_YEAR:.0f}); by year: "
            + ", ".join(f"{y}: {n}" for y, n in self.events_with_flooding_by_year.items())
        )
        lines.extend(self._degeneracy_lines(c))
        lines.extend(self._footer_lines())
        return lines

    def _peak_text(self) -> str:
        if self.sept2022_peak_fraction is None:
            return "n/a (record does not cover Sept 2022)"
        return (f"{100 * self.sept2022_peak_fraction:.1f} % of junctions at {self.sept2022_peak_time} "
                f"({self.sept2022_rain_mm:.1f} mm areal rain)")

    def _degeneracy_lines(self, c: Mapping[str, str]) -> list[str]:
        share, rf, lookup = self.flooded_share, self.rain_field, self.lookup_pr_auc
        jaccard = "n/a (not computed)" if self.uniform_jaccard is None else f"{self.uniform_jaccard:.3f}"
        return [
            f"[{c['f_rank1_r2_below_0_6']}] (f) rank-1 (hour x node) R2 over May-Nov : "
            f"{_fmt_num(self.rank1_r2)} (target < {MAX_RANK1_R2})",
            f"[{c['g_uniform_rain_jaccard_below_0_75']}] (g) Jaccard(labels, labels with uniform rain) : "
            f"{jaccard} (target < {MAX_UNIFORM_JACCARD})",
            f"[{c['h_median_flooded_share_below_25pct']}] (h) share of junctions flooded in flood hours : "
            f"median {_fmt_pct(share.get('median'))}, p90 {_fmt_pct(share.get('p90'))}, "
            f"max {_fmt_pct(share.get('max'))} over {share.get('n_flood_hours', 0)} h (target median < "
            f"{100 * MAX_MEDIAN_FLOODED_SHARE:.0f} %)",
            f"[{c['rf_rain_field_heterogeneous']}] (rf) junction-rain CV in wet hours : median "
            f"{_fmt_num(rf.get('cv_median'))} (target {RAIN_CV_RANGE[0]}-{RAIN_CV_RANGE[1]}), max/mean p95 "
            f"{_fmt_num(rf.get('max_over_mean_p95'))} (target >= {MIN_RAIN_PEAK_RATIO_P95:g}) over "
            f"{rf.get('n_wet_hours', 0)} h",
            f"separable lookup PR-AUC (basin 5 h rain x node frequency, eval {lookup.get('eval_year', 'n/a')}) : "
            f"{_fmt_num(lookup.get('pr_auc'))} (diagnostic)",
        ]

    def _footer_lines(self) -> list[str]:
        lines = []
        if self.peak_depth_m is not None:
            lines.append(f"peak depth            : {self.peak_depth_m:.2f} m")
        if self.mass_balance_error is not None:
            lines.append(f"mass-balance error    : {self.mass_balance_error:.2e} (relative)")
        if self.runtime_s:
            lines.append("runtime               : " + ", ".join(f"{k} {v:.1f} s" for k, v in self.runtime_s.items()))
        if self.provenance:
            lines.append("provenance            : " + ", ".join(f"{k}={v}" for k, v in self.provenance.items()))
        return lines


def _rain_field_ok(rain_field: Mapping[str, Any]) -> bool | None:
    cv, ratio = rain_field.get("cv_median"), rain_field.get("max_over_mean_p95")
    if cv is None or ratio is None:
        return None
    return bool(RAIN_CV_RANGE[0] <= cv <= RAIN_CV_RANGE[1] and ratio >= MIN_RAIN_PEAK_RATIO_P95)


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:+.3f}"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.3f} %"


def _correlation(freq: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(values)
    x, y = values[finite], freq[finite]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"spearman": None, "pearson": None}
    return {
        "spearman": float(sps.spearmanr(x, y).statistic),
        "pearson": float(sps.pearsonr(x, y).statistic),
    }


def _group_rates(freq: np.ndarray, static: Mapping[str, np.ndarray]) -> dict[str, float | None]:
    def mean(mask: np.ndarray) -> float | None:
        return float(freq[mask].mean()) if mask.any() else None

    out: dict[str, float | None] = {}
    if "is_sink" in static:
        sink = np.asarray(static["is_sink"], dtype=float) > 0.5
        out.update(sinks=mean(sink), non_sinks=mean(~sink))
    if "relative_elevation" in static:
        rel = np.asarray(static["relative_elevation"], dtype=float)
        low = rel <= np.nanpercentile(rel, 25)
        out.update(low_rel_elev_q1=mean(low), other_rel_elev=mean(~low))
    return out


def _flooding_events(
    hour_flooded: np.ndarray, areal: np.ndarray, timestamps: pd.DatetimeIndex, max_dry_gap_h: int, tz: str
) -> tuple[int, int, dict[int, int]]:
    """``(n_events, n_events_with_flooding, per-year counts)`` using the rain-field event definition."""
    from src.data_pipeline.rain_field import find_rain_events

    events = find_rain_events(areal, timestamps, max_dry_gap_h=max_dry_gap_h, tz=tz)
    csum = np.concatenate([[0], np.cumsum(hour_flooded.astype(np.int64))])
    starts, ends = events.start_idx, events.end_idx
    next_start = np.concatenate([starts[1:], [hour_flooded.size]])
    stops = np.minimum(ends + EVENT_TAIL_H + 1, next_start)
    flooded = (csum[stops] - csum[starts]) > 0
    years = timestamps[starts].year if starts.size else np.zeros(0, dtype=int)
    by_year = {int(y): int(flooded[years == y].sum()) for y in sorted(set(timestamps.year))}
    return int(starts.size), int(flooded.sum()), by_year


def _sept2022_peak(
    per_hour: np.ndarray, areal: np.ndarray, stamps: pd.DatetimeIndex, n_nodes: int, tz: str
) -> tuple[float | None, str | None, float | None]:
    """``(peak flooded fraction, peak hour, areal rain mm)`` of the Sept 4-5 2022 event (``None`` if absent)."""
    sept = (stamps >= pd.Timestamp(SEPT_2022[0], tz=tz)) & (stamps <= pd.Timestamp(SEPT_2022[1], tz=tz))
    if not sept.any() or n_nodes == 0:
        return None, None, None
    frac = per_hour[sept] / n_nodes
    peak_time = f"{stamps[sept][int(frac.argmax())]:%Y-%m-%d %H:%M}"
    return float(frac.max()), peak_time, float(areal[sept].sum())


def _localize(timestamps: Any, tz: str) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(timestamps)
    return stamps.tz_localize(tz) if stamps.tz is None else stamps.tz_convert(tz)


def compute_statistics(
    labels: np.ndarray,
    areal: np.ndarray,
    timestamps: pd.DatetimeIndex,
    static: Mapping[str, np.ndarray],
    *,
    max_dry_gap_h: int = 2,
    tz: str = "Asia/Kolkata",
    uniform_labels: np.ndarray | None = None,
) -> CalibrationStats:
    """Criteria (a)-(h) for flood labels ``[T, N]`` driven by the areal series ``[T]``.

    ``uniform_labels`` (same shape) are the labels simulated with the rain field disabled;
    without them criterion (g) is ``None``. Rain-field statistics are added by the caller.
    """
    flooded = np.asarray(labels).astype(bool, copy=False)
    areal = np.nan_to_num(np.asarray(areal, dtype=np.float64), nan=0.0)
    stamps = _localize(timestamps, tz)
    if flooded.ndim != 2 or flooded.shape[0] != areal.size or areal.size != len(stamps):
        raise ValueError(f"labels {flooded.shape}, areal {areal.shape} and timestamps ({len(stamps)}) disagree")
    n_hours, n_nodes = flooded.shape
    season = np.isin(stamps.month, SEASON_MONTHS)
    csum = np.concatenate([[0.0], np.cumsum(areal)])
    idx = np.arange(1, n_hours + 1)
    dry = (csum[idx] - csum[np.maximum(idx - DRY_WINDOW_H, 0)]) < DRY_RAIN_MM
    per_hour = flooded.sum(axis=1)
    peak_fraction, peak_time, sept_rain = _sept2022_peak(per_hour, areal, stamps, n_nodes, tz)
    freq = flooded[season].mean(axis=0) if season.any() else np.zeros(n_nodes)
    usable = {k: np.asarray(v, dtype=float) for k, v in static.items() if k in STATIC_FEATURES}
    n_events, n_flooding, by_year = _flooding_events(per_hour > 0, areal, stamps, max_dry_gap_h, tz)
    n_years = n_hours / HOURS_PER_YEAR
    jaccard = None
    if uniform_labels is not None:
        uniform = np.asarray(uniform_labels).astype(bool, copy=False)
        if uniform.shape != flooded.shape:
            raise ValueError(f"uniform_labels {uniform.shape} must match labels {flooded.shape}")
        jaccard = label_jaccard(flooded[season], uniform[season])
    return CalibrationStats(
        n_nodes=int(n_nodes),
        n_hours=int(n_hours),
        period=(f"{stamps[0]:%Y-%m-%d}", f"{stamps[-1]:%Y-%m-%d}") if n_hours else ("", ""),
        season_rate=float(per_hour[season].sum() / (season.sum() * n_nodes)) if season.any() and n_nodes else 0.0,
        season_node_hours=int(per_hour[season].sum()),
        dry_flood_node_hours=int(per_hour[dry].sum()),
        dry_flood_hours=int((per_hour[dry] > 0).sum()),
        sept2022_peak_fraction=peak_fraction,
        sept2022_peak_time=peak_time,
        sept2022_rain_mm=sept_rain,
        correlations={k: _correlation(freq, v) for k, v in usable.items()},
        group_rates=_group_rates(freq, usable),
        n_events=n_events,
        events_with_flooding=n_flooding,
        events_per_year_with_flooding=float(n_flooding / n_years) if n_years > 0 else 0.0,
        events_with_flooding_by_year=by_year,
        n_years=float(n_years),
        rank1_r2=rank1_r2(flooded[season]) if season.any() else None,
        uniform_jaccard=jaccard,
        flooded_share=dataclasses.asdict(flooded_share_stats(flooded)),
        lookup_pr_auc=separable_lookup_pr_auc(flooded, areal, stamps),
    )


# --------------------------------------------------------------------------- full run


def _load_graph_arrays(cfg: Mapping[str, Any]) -> tuple[GraphArrays, dict[str, Any]]:
    """The road graph read-only: the cached graph (re-enriched in memory if needed), else built in memory."""
    from src.data_pipeline.network import extract_network  # lazy: heavy imports (osmnx)

    graph = extract_network(cfg, persist=False)
    provenance = {key: graph.graph.get(key) for key in ("source", "elevation_source", "drain_source")}
    if provenance["source"] == "synthetic_grid":
        LOGGER.warning("Calibrating on the synthetic fallback street grid (no OpenStreetMap graph at "
                       "paths.graph_file); the statistics do not describe the real corridor")
    return graph_to_arrays(graph), provenance


def _load_areal(cfg: Mapping[str, Any]) -> tuple[np.ndarray, pd.DatetimeIndex, str]:
    """The areal weather record read-only (the cache is never written by a diagnostics run)."""
    from src.data_pipeline.weather import areal_series, load_or_fetch_weather

    frame = load_or_fetch_weather(cfg, persist=False)
    areal, stamps = areal_series(frame)
    sources = ",".join(sorted(map(str, frame["source"].unique()))) if "source" in frame.columns else "unknown"
    return areal, stamps, sources


def _simulate_flooded(simulator: UrbanDrainageSimulator, rain: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``(flooded [T, N], peak depth, relative mass-balance error)``; raises on a non-finite budget."""
    result = simulator.run(rain)
    error = float(result.mass_balance.relative_error)
    if not (result.mass_balance.is_finite and math.isfinite(error)):
        raise SimulationError(f"hydrology mass balance is not finite ({error}); check the parameters")
    peak = float(result.depth_m.max()) if result.depth_m.size else 0.0
    return result.flooded, peak, error


def _uniform_labels(simulator: UrbanDrainageSimulator, areal: np.ndarray, n_nodes: int) -> np.ndarray:
    """Labels simulated with the rain field disabled (areal rain broadcast to every junction).

    Identical to ``downscale_rainfall`` with ``rainfall_field.enabled: false`` (non-finite and
    negative areal values count as 0), but as a zero-copy broadcast view to bound memory.
    """
    clean = np.nan_to_num(np.asarray(areal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    column = np.maximum(clean, 0.0).astype(np.float32)[:, None]
    return _simulate_flooded(simulator, np.broadcast_to(column, (column.shape[0], n_nodes)))[0]


def simulate_and_score(
    arrays: GraphArrays,
    rain: np.ndarray,
    areal: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cfg: Mapping[str, Any],
    *,
    compare_uniform: bool = True,
) -> CalibrationStats:
    """Simulate labels for prepared junction rain and compute the calibration statistics.

    With ``compare_uniform`` a second simulation with the areal rain broadcast to every
    junction provides criterion (g). Raises :class:`SimulationError` on a non-finite budget.
    """
    params = HydrologyParams.from_config(cfg)
    started = time.perf_counter()
    simulator = UrbanDrainageSimulator(arrays, params)
    flooded, peak, error = _simulate_flooded(simulator, rain)
    elapsed = time.perf_counter() - started
    runtime = {"simulation": elapsed}
    uniform = None
    if compare_uniform:
        uniform = _uniform_labels(simulator, areal, arrays.num_nodes)
        runtime["uniform_simulation"] = time.perf_counter() - started - elapsed
    tz = str((cfg.get("project") or {}).get("timezone", "Asia/Kolkata"))
    gap = int(get_section(cfg, "rainfall_field", {"max_dry_gap_h": 2})["max_dry_gap_h"])
    static = {k: arrays.node_attrs[k] for k in STATIC_FEATURES if k in arrays.node_attrs}
    stats = compute_statistics(flooded, areal, timestamps, static, max_dry_gap_h=gap, tz=tz, uniform_labels=uniform)
    return dataclasses.replace(
        stats,
        rain_field=dataclasses.asdict(rain_field_heterogeneity(rain, areal)),
        runtime_s=runtime,
        peak_depth_m=peak,
        mass_balance_error=error,
        params=params.to_dict(),
    )


def run_calibration(
    cfg: Mapping[str, Any], overrides: Mapping[str, Any] | None = None, *, compare_uniform: bool = True
) -> CalibrationStats:
    """Graph -> weather -> rain field -> simulator -> :class:`CalibrationStats` (``overrides`` deep-merged)."""
    from src.data_pipeline.rain_field import RainFieldParams, downscale_rainfall

    cfg = deep_merge(cfg, overrides or {})
    HydrologyParams.from_config(cfg)  # fail fast on bad overrides before the expensive steps
    RainFieldParams.from_config(cfg)
    t0 = time.perf_counter()
    arrays, provenance = _load_graph_arrays(cfg)
    areal, stamps, weather_source = _load_areal(cfg)
    t1 = time.perf_counter()
    rain = downscale_rainfall(areal, stamps, arrays.lon, arrays.lat, cfg)
    t2 = time.perf_counter()
    stats = simulate_and_score(arrays, rain, areal, stamps, cfg, compare_uniform=compare_uniform)
    runtime = {"load": t1 - t0, "rain_field": t2 - t1, **stats.runtime_s}
    return dataclasses.replace(stats, runtime_s=runtime, provenance={**provenance, "weather_source": weather_source})


# --------------------------------------------------------------------------- CLI


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["key=value", "section.key=value"]`` -> nested config overrides (bare keys -> ``hydrology``)."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        path = key.split(".") if "." in key else ["hydrology", key]
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"--set {key}: cannot parse value {raw!r}: {exc}") from exc
        nested: Any = value
        for part in reversed(path):
            nested = {part: nested}
        overrides = deep_merge(overrides, nested)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Namma-Flow hydrology calibration diagnostics")
    parser.add_argument("--config", default=None,
                        help="config YAML (default: $NAMMA_FLOW_CONFIG or config/config.yaml)")
    parser.add_argument("--offline", action="store_true", help="never touch the network (caches / synthetic only)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a hydrology key (or dotted config key); repeatable")
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the statistics as JSON")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 when a criterion fails")
    parser.add_argument("--no-uniform", action="store_true",
                        help="skip the uniform-rain simulation (criterion (g) becomes n/a; ~2x faster)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, {"project": {"offline": True}} if args.offline else None)
        stats = run_calibration(cfg, parse_overrides(args.set), compare_uniform=not args.no_uniform)
        if args.json:
            atomic_write_text(Path(args.json), json.dumps(stats.to_dict(), indent=2, default=str))
    except (ConfigError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 1
    print("Namma-Flow hydrology calibration")
    for line in stats.format_lines():
        print(f"  {line}")
    failed = [name for name, ok in stats.checks().items() if ok is False]
    if failed:
        print(f"  failed criteria: {', '.join(failed)}")
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
