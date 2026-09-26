"""Map layers for the Namma-Flow dashboard: colour scales, per-junction / per-road frames,
drains & lakes, the camera and the pydeck ``Deck``.

Everything here is a pure function of numpy / pandas inputs (no Streamlit), so it is unit
tested directly and reused by :mod:`app` for every rerun.

Colour semantics
----------------
* ``tiers`` — a junction takes the colour of its risk tier (``inference.risk_tiers``: Low,
  Moderate, High, Severe → green, yellow, orange, red); column height = probability x 100.
* ``relative`` — a continuous green → yellow → orange → red ramp over ``p / vmax`` with
  ``vmax = max(highest probability shown, alert threshold)``. A sharply calibrated rare-event
  model can put every junction in the lowest tier (its alert threshold may be far below 0.25);
  the relative scale keeps the spatial pattern visible without calling anything "Severe", and
  on a dry day (every p far below the alert threshold) the map stays green.
* ``auto`` — ``tiers`` when some junction reaches the second tier, otherwise ``relative``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pydeck as pdk

from src.utils.geo import validate_bbox
from src.utils.logger import get_logger

LOGGER = get_logger("app.map_layers")

RGB = tuple[int, int, int]
Tiers = tuple[tuple[str, float], ...]

RAMP_STOPS: tuple[tuple[float, RGB], ...] = (
    (0.0, (46, 158, 90)),     # green
    (0.5, (242, 196, 48)),    # yellow
    (0.75, (240, 120, 32)),   # orange
    (1.0, (206, 32, 41)),     # red
)
DEFAULT_TIERS: Tiers = (("Low", 0.0), ("Moderate", 0.25), ("High", 0.5), ("Severe", 0.75))
COLOR_SCALES = ("auto", "tiers", "relative")
MAP_STYLES = ("light", "dark", "road")

JUNCTION_LAYER_ID = "junctions"
SELECTED_LAYER_ID = "selected-junction"
ROAD_LAYER_ID = "roads"
WATER_LINE_LAYER_ID = "water-lines"
WATER_BODY_LAYER_ID = "water-bodies"

MAX_HEIGHT = 100.0                  # column height units at p = 1 (x elevation_scale = metres)
JUNCTION_ALPHA = 230
ROAD_ALPHA = 170
WATER_LINE_RGBA = (37, 116, 196, 220)
WATER_FILL_RGBA = (86, 160, 222, 80)
SELECTED_RGBA = (0, 190, 255, 255)
MAX_WATER_FEATURES = 5000
TOOLTIP_STYLE = {"backgroundColor": "rgba(20, 24, 32, 0.92)", "color": "#f5f7fa", "fontSize": "12px",
                 "padding": "8px 10px", "borderRadius": "6px", "maxWidth": "340px", "whiteSpace": "pre-line"}
# Tooltips are plain text (one field per line): Streamlit escapes values substituted into an HTML
# template, so HTML markup inside a value would be shown literally. Text mode is also injection-safe.
TOOLTIP = {"text": "{tooltip}", "style": TOOLTIP_STYLE}
JUNCTION_COLUMNS = ("node_id", "lon", "lat", "prob", "height", "color", "tier", "tooltip")
ROAD_COLUMNS = ("source", "target", "prob", "color", "tooltip")
# Only these columns are sent to the browser (pydeck serialises layer data as indented JSON on every rerun).
JUNCTION_LAYER_COLUMNS = ("node_id", "lon", "lat", "height", "color", "tooltip")
ROAD_LAYER_COLUMNS = ("source", "target", "color", "tooltip")
COORD_DECIMALS = 6                  # ~0.1 m
HEIGHT_DECIMALS = 2


# --------------------------------------------------------------------------- probabilities & tiers


def format_probability(value: Any) -> str:
    """Human-readable probability: ``82%``, ``4.3%``, ``3.6e-05``; NaN / None → ``n/a``."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "0%"
    if abs(p) >= 0.1:
        return f"{p:.0%}"
    if abs(p) >= 0.001:
        return f"{p:.1%}"
    return f"{p:.1e}"


def clean_probabilities(values: Any) -> np.ndarray:
    """``float64 [N]`` copy clipped to [0, 1]; NaN / inf / non-numeric entries become 0."""
    try:
        array = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"probabilities must be numeric: {exc}") from exc
    array[~np.isfinite(array)] = 0.0
    return np.clip(array, 0.0, 1.0)


def validate_tiers(tiers: Iterable[Sequence[Any]] | None) -> Tiers:
    """``((name, lower_bound), ...)`` sorted ascending; the first bound must be 0 and bounds < 1."""
    if tiers is None:
        return DEFAULT_TIERS
    try:
        parsed = tuple(sorted(((str(name), float(bound)) for name, bound in tiers), key=lambda t: t[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"risk tiers must be (name, lower bound) pairs, got {tiers!r}") from exc
    bounds = [bound for _, bound in parsed]
    if not parsed or bounds[0] != 0.0:
        raise ValueError(f"risk tiers must start at 0.0, got {tiers!r}")
    if len(set(bounds)) != len(bounds) or bounds[-1] >= 1.0 or not all(math.isfinite(b) for b in bounds):
        raise ValueError(f"risk tier bounds must be distinct, finite and < 1.0, got {tiers!r}")
    return parsed


def ramp_rgb(fraction: Any) -> np.ndarray:
    """Piecewise-linear green → yellow → orange → red colour (``uint8 [N, 3]``) for fractions in [0, 1]."""
    f = clean_probabilities(fraction)
    xs = [stop for stop, _ in RAMP_STOPS]
    channels = [np.interp(f, xs, [rgb[c] for _, rgb in RAMP_STOPS]) for c in range(3)]
    return np.rint(np.stack(channels, axis=1)).astype(np.uint8)


def tier_palette(tiers: Iterable[Sequence[Any]] | None = None) -> dict[str, RGB]:
    """Tier name → colour, spread evenly along the ramp (lowest tier green, highest red)."""
    parsed = validate_tiers(tiers)
    positions = np.linspace(0.0, 1.0, len(parsed)) if len(parsed) > 1 else np.zeros(1)
    colours = ramp_rgb(positions)
    return {name: tuple(int(c) for c in colours[i]) for i, (name, _) in enumerate(parsed)}


def tier_index(prob: Any, tiers: Iterable[Sequence[Any]] | None = None) -> np.ndarray:
    """Index of each probability's tier (lower bounds inclusive)."""
    parsed = validate_tiers(tiers)
    bounds = np.array([bound for _, bound in parsed])
    return np.clip(np.searchsorted(bounds, clean_probabilities(prob), side="right") - 1, 0, len(bounds) - 1)


def tier_names(prob: Any, tiers: Iterable[Sequence[Any]] | None = None) -> np.ndarray:
    """Tier name per probability (``str [N]``)."""
    parsed = validate_tiers(tiers)
    names = np.array([name for name, _ in parsed], dtype=object)
    return names[tier_index(prob, parsed)].astype(str)


@dataclass(frozen=True)
class ColorScale:
    """How probabilities map to colours and column heights (see the module docstring)."""

    mode: str
    tiers: Tiers = DEFAULT_TIERS
    vmax: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("tiers", "relative"):
            raise ValueError(f"ColorScale.mode must be 'tiers' or 'relative', got {self.mode!r}")
        object.__setattr__(self, "tiers", validate_tiers(self.tiers))
        vmax = float(self.vmax)
        if not (math.isfinite(vmax) and 0.0 < vmax <= 1.0):
            raise ValueError(f"ColorScale.vmax must lie in (0, 1], got {self.vmax!r}")
        object.__setattr__(self, "vmax", vmax)

    def fraction(self, prob: Any) -> np.ndarray:
        """Position on the scale in [0, 1] (the probability itself for tiers, ``p / vmax`` when relative)."""
        p = clean_probabilities(prob)
        return p if self.mode == "tiers" else np.clip(p / self.vmax, 0.0, 1.0)

    def rgb(self, prob: Any) -> np.ndarray:
        """``uint8 [N, 3]`` colours."""
        if self.mode == "relative":
            return ramp_rgb(self.fraction(prob))
        palette = tier_palette(self.tiers)
        table = np.array([palette[name] for name, _ in self.tiers], dtype=np.uint8)
        return table[tier_index(prob, self.tiers)]

    def rgba(self, prob: Any, alpha: int = JUNCTION_ALPHA) -> np.ndarray:
        """``uint8 [N, 4]`` colours with a constant alpha (clipped to 0..255)."""
        rgb = self.rgb(prob)
        return np.concatenate([rgb, np.full((len(rgb), 1), int(np.clip(alpha, 0, 255)), dtype=np.uint8)], axis=1)

    def heights(self, prob: Any) -> np.ndarray:
        """Column heights in [0, MAX_HEIGHT] (multiplied by the layer's elevation scale)."""
        return self.fraction(prob) * MAX_HEIGHT

    def legend(self) -> list[tuple[str, RGB]]:
        """``[(label, colour), ...]`` from lowest to highest."""
        if self.mode == "tiers":
            palette = tier_palette(self.tiers)
            entries = []
            for i, (name, lo) in enumerate(self.tiers):
                hi = self.tiers[i + 1][1] if i + 1 < len(self.tiers) else None
                low = format_probability(lo)
                span = f"≥ {low}" if hi is None else f"{low}–{format_probability(hi)}"
                entries.append((f"{name} ({span})", palette[name]))
            return entries
        stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        colours = ramp_rgb(stops)
        return [(f"p = {'0' if s == 0 else format_probability(s * self.vmax)}", tuple(int(c) for c in colours[i]))
                for i, s in enumerate(stops)]

    def describe(self) -> str:
        """One sentence for the legend caption."""
        if self.mode == "tiers":
            return "Colour = risk tier; column height ∝ flood probability (100 % = tallest)."
        return (f"Colour and height relative to p = {format_probability(self.vmax)} (the larger of the highest "
                "probability shown and the alert threshold).")


def resolve_color_scale(mode: str, max_prob: Any, threshold: Any, tiers: Iterable[Sequence[Any]] | None = None
                        ) -> ColorScale:
    """Pick the colour scale for a view (``mode`` in :data:`COLOR_SCALES`)."""
    if mode not in COLOR_SCALES:
        raise ValueError(f"colour scale must be one of {COLOR_SCALES}, got {mode!r}")
    parsed = validate_tiers(tiers)
    top = float(clean_probabilities([max_prob])[0])
    if mode == "auto":
        mode = "tiers" if len(parsed) > 1 and top >= parsed[1][1] else "relative"
    if mode == "tiers":
        return ColorScale("tiers", parsed)
    alert = float(clean_probabilities([threshold])[0])
    vmax = max(top, alert)
    return ColorScale("relative", parsed, vmax if vmax > 0.0 else 1.0)


# --------------------------------------------------------------------------- junctions


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _time_text(value: Any) -> str:
    if value is None or (not isinstance(value, pd.Timestamp) and pd.isna(value)):
        return ""
    try:
        return pd.Timestamp(value).strftime("%a %d %b %H:%M")
    except (TypeError, ValueError):
        return ""


def _junction_tooltip(row: Mapping[str, Any], when: str) -> str:
    """Plain-text tooltip, one fact per line (missing facts are left out)."""
    std = _number(row.get("std"))
    spread = f" ± {format_probability(std)}" if math.isfinite(std) else ""
    lines = [f"Junction {row['node_id']}",
             f"Flood probability ({when}): {format_probability(row['prob'])}{spread}",
             f"Risk tier: {row['tier']}"]
    peak = _time_text(row.get("peak_time"))
    if peak:
        lines.append(f"Peak: {peak} (p = {format_probability(row.get('max_prob'))})")
    elevation, relative = _number(row.get("elevation")), _number(row.get("relative_elevation"))
    drain, depth = _number(row.get("dist_to_drain_m")), _number(row.get("max_depth_m"))
    rain = _number(row.get("peak_rain_mm_h"))
    if math.isfinite(elevation):
        lines.append(f"Elevation: {elevation:.1f} m")
    if math.isfinite(relative):
        lines.append(f"Relative elevation: {relative:+.1f} m" + (" (local depression)" if relative < 0 else ""))
    if math.isfinite(drain):
        lines.append(f"Distance to drain: {drain:,.0f} m")
    if math.isfinite(depth):
        lines.append(f"Simulated peak depth: {depth:.2f} m")
    if math.isfinite(rain):
        lines.append(f"Peak junction rain: {rain:.1f} mm/h")
    share = _number(row.get("field_share_at_risk"))
    if math.isfinite(share):
        lines.append(f"At risk in {share:.0%} of the rain-field realisations (horizon)")
    return "\n".join(lines)


def junction_frame(table: pd.DataFrame, prob: Any, scale: ColorScale, *, std: Any = None,
                   when: str = "peak over horizon", alpha: int = JUNCTION_ALPHA) -> pd.DataFrame:
    """Per-junction layer data: position, displayed probability, height, colour, tier and tooltip.

    ``table`` is a :meth:`PredictionResult.node_table` (``node_id``, ``lon``, ``lat`` required;
    static attributes, ``peak_time``, ``max_prob``, ``max_depth_m`` … used when present).
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"table must be a DataFrame, got {type(table).__name__}")
    missing = [c for c in ("node_id", "lon", "lat") if c not in table.columns]
    if missing:
        raise ValueError(f"junction table is missing columns {missing}")
    p = clean_probabilities(prob)
    if p.size != len(table):
        raise ValueError(f"got {p.size} probabilities for {len(table)} junctions")
    spread = np.full(len(table), np.nan) if std is None else np.array(std, dtype=np.float64).reshape(-1)
    if spread.size != len(table):
        raise ValueError(f"got {spread.size} std values for {len(table)} junctions")
    base = table.reset_index(drop=True)
    frame = pd.DataFrame({
        "node_id": base["node_id"].astype(str),
        "lon": np.round(base["lon"].to_numpy(dtype=np.float64), COORD_DECIMALS),
        "lat": np.round(base["lat"].to_numpy(dtype=np.float64), COORD_DECIMALS),
        "prob": p,
        "height": np.round(scale.heights(p), HEIGHT_DECIMALS),
        "color": scale.rgba(p, alpha).tolist(),
        "tier": tier_names(p, scale.tiers),
    })
    details = base.drop(columns=["node_id", "lon", "lat"], errors="ignore").assign(
        node_id=frame["node_id"], prob=p, std=spread, tier=frame["tier"])
    frame["tooltip"] = [_junction_tooltip(row, when) for row in details.to_dict(orient="records")]
    return frame[list(JUNCTION_COLUMNS)]


# --------------------------------------------------------------------------- roads


def undirected_segments(edge_index: Any, num_nodes: int) -> np.ndarray:
    """Unique undirected street segments ``int64 [S, 2]`` (``i < j``; self-loops dropped)."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edges.shape}")
    if edges.min() < 0 or edges.max() >= int(num_nodes):
        raise ValueError(f"edge_index refers to junctions outside [0, {num_nodes})")
    lo, hi = np.minimum(edges[0], edges[1]), np.maximum(edges[0], edges[1])
    keep = lo != hi
    if not keep.any():
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)


def road_frame(lon: Any, lat: Any, segments: np.ndarray, prob: Any, scale: ColorScale,
               alpha: int = ROAD_ALPHA) -> pd.DataFrame:
    """Road-segment layer data coloured by the mean probability of the two end junctions."""
    lon, lat, p = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64), clean_probabilities(prob)
    if not (lon.shape == lat.shape == p.shape):
        raise ValueError(f"lon {lon.shape}, lat {lat.shape} and prob {p.shape} must align")
    pairs = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in ROAD_COLUMNS})
    if pairs.min() < 0 or pairs.max() >= lon.size:
        raise ValueError("road segments refer to unknown junctions")
    mean = (p[pairs[:, 0]] + p[pairs[:, 1]]) / 2.0
    lon, lat = np.round(lon, COORD_DECIMALS), np.round(lat, COORD_DECIMALS)
    return pd.DataFrame({
        "source": np.stack([lon[pairs[:, 0]], lat[pairs[:, 0]]], axis=1).tolist(),
        "target": np.stack([lon[pairs[:, 1]], lat[pairs[:, 1]]], axis=1).tolist(),
        "prob": mean,
        "color": scale.rgba(mean, alpha).tolist(),
        "tooltip": [f"Road segment\nMean end-junction probability: {format_probability(v)}" for v in mean],
    })


# --------------------------------------------------------------------------- drains & lakes


@dataclass(frozen=True)
class WaterFeature:
    """One drain / stream (``line``: a tuple of (lon, lat)) or water body (``polygon``: rings)."""

    kind: str
    label: str
    coords: tuple


@dataclass(frozen=True)
class WaterLayerData:
    """Water features for the map and where they came from (``osm`` or ``synthetic_line``)."""

    features: tuple[WaterFeature, ...]
    source: str
    note: str = ""

    @property
    def n_lines(self) -> int:
        """Number of drain / stream lines."""
        return sum(1 for f in self.features if f.kind == "line")

    @property
    def n_polygons(self) -> int:
        """Number of water bodies."""
        return sum(1 for f in self.features if f.kind == "polygon")


def _clean_ring(points: Any, minimum: int) -> tuple[tuple[float, float], ...] | None:
    """Valid (lon, lat) pairs of ``points``; ``None`` when fewer than ``minimum`` survive."""
    if not isinstance(points, (list, tuple)):
        return None
    clean = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        lon, lat = _number(point[0]), _number(point[1])
        if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
            clean.append((lon, lat))
    return tuple(clean) if len(clean) >= minimum else None


def _geometry_features(geometry: Any, label: str, depth: int = 0) -> list[WaterFeature]:
    if not isinstance(geometry, Mapping) or depth > 3:
        return []
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "GeometryCollection":
        return [f for g in geometry.get("geometries") or [] for f in _geometry_features(g, label, depth + 1)]
    lines = [coords] if kind == "LineString" else list(coords or []) if kind == "MultiLineString" else []
    polygons = [coords] if kind == "Polygon" else list(coords or []) if kind == "MultiPolygon" else []
    features = [WaterFeature("line", label, ring) for ring in (_clean_ring(c, 2) for c in lines) if ring]
    for polygon in polygons:
        rings = [r for r in (_clean_ring(ring, 3) for ring in (polygon or [])) if r] if isinstance(
            polygon, (list, tuple)) else []
        if rings:
            features.append(WaterFeature("polygon", label, tuple(rings)))
    return features


def _feature_label(properties: Any) -> str:
    props = properties if isinstance(properties, Mapping) else {}
    kind = next((str(props[k]) for k in ("waterway", "water", "natural") if props.get(k)), "water")
    name = props.get("name")
    kind_text = kind.replace("_", " ").title()
    return f"{name} ({kind_text})" if name else kind_text


def excluded_water_flags(water_tags: Sequence[Any], exclude_water: Iterable[str]) -> np.ndarray:
    """Which features the model's distance-to-drain ignored, by the pipeline's OWN predicate.

    ``water_tags`` are the features' OSM ``water=*`` values (lists are ``;``-joined like the
    pipeline's tag cleaning, so ``wastewater;pond`` matches ``wastewater``); the decision is
    :func:`src.data_pipeline.drains._drop_excluded_water`, so the map can never disagree with the
    model's features.
    """
    from src.data_pipeline import drains

    excluded = frozenset(str(v).strip().lower() for v in exclude_water)
    frame = pd.DataFrame({"water": [drains._stringify(v) for v in water_tags]}, index=np.arange(len(water_tags)))
    if not excluded or not len(frame):
        return np.zeros(len(frame), dtype=bool)
    kept = drains._drop_excluded_water(frame, excluded)
    return ~np.isin(np.arange(len(frame)), np.asarray(kept.index))


def _water_tag(item: Any) -> Any:
    props = item.get("properties") if isinstance(item, Mapping) else None
    return props.get("water") if isinstance(props, Mapping) else None


def parse_waterways(collection: Any, source: str = "osm", exclude_water: Sequence[str] = ()) -> WaterLayerData:
    """Water features of a GeoJSON ``FeatureCollection`` (points / invalid features are skipped).

    Features whose OSM ``water=*`` tag is excluded by the pipeline's ``drains.exclude_water_values``
    (treatment tanks, pools; see :func:`excluded_water_flags`) are left out, so the map shows
    exactly the drains and water bodies the model's distance-to-drain feature used.
    """
    if not isinstance(collection, Mapping) or collection.get("type") != "FeatureCollection":
        raise ValueError("waterways must be a GeoJSON FeatureCollection")
    raw = collection.get("features")
    if not isinstance(raw, list):
        raise ValueError("waterways 'features' must be a list")
    drop = excluded_water_flags([_water_tag(item) for item in raw], exclude_water)
    features: list[WaterFeature] = []
    skipped = excluded = 0
    for item, dropped in zip(raw, drop):
        if dropped:
            excluded += 1
            continue
        found = _geometry_features(item.get("geometry"), _feature_label(item.get("properties"))) if isinstance(
            item, Mapping) else []
        skipped += 0 if found else 1
        features.extend(found)
    if skipped:
        LOGGER.debug("waterways: skipped %d features without line/polygon geometry", skipped)
    if len(features) > MAX_WATER_FEATURES:
        LOGGER.warning("waterways: showing the first %d of %d features", MAX_WATER_FEATURES, len(features))
        features = features[:MAX_WATER_FEATURES]
    note = f"{len(features)} OpenStreetMap drains & water bodies"
    if excluded:
        note += f" ({excluded} treatment tanks / pools excluded, as in the model)"
    return WaterLayerData(tuple(features), source, note)


def load_waterways(path: str | Path, exclude_water: Sequence[str] = ()) -> WaterLayerData | None:
    """Read the waterways GeoJSON cache; ``None`` when it is missing or unreadable (WARNING)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return parse_waterways(json.loads(path.read_text(encoding="utf-8")), exclude_water=exclude_water)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        LOGGER.warning("Ignoring unreadable waterways file %s: %s", path, exc)
        return None


def synthetic_drain_line(drain_lon: Any, bbox: Sequence[float]) -> WaterLayerData:
    """The north-south drain line the pipeline assumes when OSM waterways are unavailable."""
    west, south, east, north = validate_bbox(bbox)
    lon = _number(drain_lon)
    if not math.isfinite(lon):
        raise ValueError(f"drain longitude must be a finite number, got {drain_lon!r}")
    label = f"Synthetic drain line (lon {lon:.4f}; OSM waterways unavailable)"
    feature = WaterFeature("line", label, ((lon, south), (lon, north)))
    note = "synthetic drain line" + ("" if west <= lon <= east else " (outside the corridor)")
    return WaterLayerData((feature,), "synthetic_line", note)


def water_frames(data: WaterLayerData | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(lines, polygons)`` layer data with ``path`` / ``polygon`` and ``tooltip`` columns."""
    features = data.features if data is not None else ()
    lines = [{"path": [list(p) for p in f.coords], "tooltip": f.label} for f in features if f.kind == "line"]
    polygons = [{"polygon": [[list(p) for p in ring] for ring in f.coords], "tooltip": f.label}
                for f in features if f.kind == "polygon"]
    return (pd.DataFrame(lines, columns=["path", "tooltip"]), pd.DataFrame(polygons, columns=["polygon", "tooltip"]))


# --------------------------------------------------------------------------- camera & deck


def zoom_for_extent(width_deg: float, height_deg: float, lat_deg: float,
                    viewport_px: tuple[int, int] = (900, 560), padding: float = 0.85) -> float:
    """Web-mercator zoom that fits a lon/lat extent into the viewport (clipped to [3, 17])."""
    vw, vh = viewport_px
    zooms = []
    if width_deg > 0:
        zooms.append(math.log2(360.0 * vw * padding / (512.0 * width_deg)))
    if height_deg > 0:
        stretch = 1.0 / max(math.cos(math.radians(lat_deg)), 1e-6)
        zooms.append(math.log2(360.0 * vh * padding / (512.0 * height_deg * stretch)))
    return float(np.clip(min(zooms), 3.0, 17.0)) if zooms else 15.0


def view_state(lon: Any, lat: Any, *, pitch: float = 45.0, bearing: float = 0.0,
               viewport_px: tuple[int, int] = (900, 560)) -> pdk.ViewState:
    """Camera centred on the junctions' extent."""
    lon, lat = np.asarray(lon, dtype=np.float64).reshape(-1), np.asarray(lat, dtype=np.float64).reshape(-1)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        raise ValueError("view_state needs at least one finite junction coordinate")
    lon, lat = lon[ok], lat[ok]
    centre_lon, centre_lat = (lon.min() + lon.max()) / 2.0, (lat.min() + lat.max()) / 2.0
    zoom = zoom_for_extent(float(np.ptp(lon)), float(np.ptp(lat)), centre_lat, viewport_px)
    return pdk.ViewState(longitude=float(centre_lon), latitude=float(centre_lat), zoom=round(zoom, 2),
                         pitch=float(pitch), bearing=float(bearing))


class CompactDeck(pdk.Deck):
    """A pydeck ``Deck`` whose JSON has no indentation.

    Streamlit ships ``Deck.to_json()`` to the browser on every rerun and pydeck pretty-prints it
    (``indent=2``); for ~1 000 junctions + ~1 200 road segments the compact form is less than half the size.
    """

    def to_json(self) -> str:
        """The pydeck spec as compact JSON (same content as ``pdk.Deck.to_json``)."""
        return json.dumps(json.loads(super().to_json()), separators=(",", ":"))


@dataclass(frozen=True)
class MapOptions:
    """User-selectable map settings (validated)."""

    three_d: bool = True
    show_roads: bool = True
    show_drains: bool = True
    map_style: str = "light"
    column_radius_m: float = 22.0
    elevation_scale: float = 6.0
    pitch_deg: float = 45.0
    bearing_deg: float = -15.0

    def __post_init__(self) -> None:
        if self.map_style not in MAP_STYLES:
            raise ValueError(f"map_style must be one of {MAP_STYLES}, got {self.map_style!r}")
        for name, lo, hi in (("column_radius_m", 1.0, 500.0), ("elevation_scale", 0.01, 1000.0),
                             ("pitch_deg", 0.0, 85.0), ("bearing_deg", -360.0, 360.0)):
            value = _number(getattr(self, name))
            if not lo <= value <= hi:
                raise ValueError(f"{name} must lie in [{lo}, {hi}], got {getattr(self, name)!r}")

    @property
    def pitch(self) -> float:
        """Camera tilt: ``pitch_deg`` in 3D mode, 0 (top-down) in 2D."""
        return self.pitch_deg if self.three_d else 0.0


def junction_layer(frame: pd.DataFrame, options: MapOptions) -> pdk.Layer:
    """Extruded columns (3D) or flat discs (2D) at the junctions."""
    frame = frame[[c for c in JUNCTION_LAYER_COLUMNS if c in frame.columns]]
    if options.three_d:
        return pdk.Layer("ColumnLayer", data=frame, id=JUNCTION_LAYER_ID, get_position=["lon", "lat"],
                         get_elevation="height", elevation_scale=options.elevation_scale,
                         radius=options.column_radius_m, disk_resolution=12, extruded=True,
                         get_fill_color="color", pickable=True, auto_highlight=True)
    return pdk.Layer("ScatterplotLayer", data=frame, id=JUNCTION_LAYER_ID, get_position=["lon", "lat"],
                     get_radius=options.column_radius_m, radius_min_pixels=3, get_fill_color="color",
                     stroked=True, get_line_color=[255, 255, 255, 160], line_width_min_pixels=0.5,
                     pickable=True, auto_highlight=True)


def build_deck(junctions: pd.DataFrame, options: MapOptions, *, roads: pd.DataFrame | None = None,
               water: WaterLayerData | None = None, selected_node: Any = None,
               view: pdk.ViewState | None = None) -> pdk.Deck:
    """The dashboard map: water bodies, drains, roads, junction columns and the selection ring."""
    layers: list[pdk.Layer] = []
    if options.show_drains and water is not None and water.features:
        lines, polygons = water_frames(water)
        if len(polygons):
            layers.append(pdk.Layer("PolygonLayer", data=polygons, id=WATER_BODY_LAYER_ID, get_polygon="polygon",
                                    get_fill_color=list(WATER_FILL_RGBA), get_line_color=list(WATER_LINE_RGBA),
                                    line_width_min_pixels=1, stroked=True, filled=True, pickable=True))
        if len(lines):
            layers.append(pdk.Layer("PathLayer", data=lines, id=WATER_LINE_LAYER_ID, get_path="path",
                                    get_color=list(WATER_LINE_RGBA), get_width=6, width_min_pixels=2,
                                    pickable=True))
    if options.show_roads and roads is not None and len(roads):
        roads = roads[[c for c in ROAD_LAYER_COLUMNS if c in roads.columns]]
        layers.append(pdk.Layer("LineLayer", data=roads, id=ROAD_LAYER_ID, get_source_position="source",
                                get_target_position="target", get_color="color", get_width=3, pickable=True))
    layers.append(junction_layer(junctions, options))
    if selected_node is not None:
        ring = junctions[junctions["node_id"] == str(selected_node)]
        if len(ring):
            layers.append(pdk.Layer("ScatterplotLayer", data=ring[["lon", "lat"]], id=SELECTED_LAYER_ID,
                                    get_position=["lon", "lat"], get_radius=options.column_radius_m * 2.2,
                                    radius_min_pixels=9, filled=False, stroked=True,
                                    get_line_color=list(SELECTED_RGBA), line_width_min_pixels=3, pickable=False))
    camera = view or view_state(junctions["lon"], junctions["lat"], pitch=options.pitch, bearing=options.bearing_deg)
    return CompactDeck(layers=layers, initial_view_state=camera, map_provider="carto", map_style=options.map_style,
                       tooltip=TOOLTIP)


def selected_node_id(event: Any, layer_id: str = JUNCTION_LAYER_ID) -> str | None:
    """Node id (as a string) of the junction clicked on the map, from ``st.pydeck_chart``'s selection state."""
    if event is None:
        return None
    selection = event.get("selection") if isinstance(event, Mapping) else getattr(event, "selection", None)
    if selection is None:
        return None
    objects = selection.get("objects") if isinstance(selection, Mapping) else getattr(selection, "objects", None)
    if not isinstance(objects, Mapping):
        return None
    items = objects.get(layer_id) or []
    first = items[0] if isinstance(items, (list, tuple)) and items else None
    node = first.get("node_id") if isinstance(first, Mapping) else None
    return None if node is None else str(node)
