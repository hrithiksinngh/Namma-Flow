"""The immutable result of a prediction run and its exports (tables, GeoJSON, CSV).

:class:`PredictionResult` is re-exported by :mod:`src.inference.predictor` (the public import
path). Probabilities are per junction and target hour; everything the app and CLI show
(horizon maxima, risk tiers, top-K junctions, KPIs) is derived here so every consumer uses
the same definitions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.config import project_root
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

DEFAULT_TIERS: tuple[tuple[str, float], ...] = (("Low", 0.0), ("Moderate", 0.25), ("High", 0.5), ("Severe", 0.75))
STATIC_COLUMNS = ("elevation", "dist_to_drain_m", "relative_elevation")
NODE_TABLE_COLUMNS = (
    "node_id", "lon", "lat", *STATIC_COLUMNS, "max_prob", "prob_std", "peak_time", "peak_rain_mm_h", "tier",
    "above_threshold", "rain_total_mm", "max_depth_m", "field_share_at_risk",
)


def _readonly(value: Any, dtype: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PredictionResult.{name} must be numeric: {exc}") from exc
    if array.shape != shape:
        raise ValueError(f"PredictionResult.{name} must have shape {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


def json_safe(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars to JSON types (NaN / inf / NaT → None)."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (Path, datetime)):
        return str(value) if isinstance(value, Path) else value.isoformat()
    return value


def display_path(path: str | Path | None) -> str | None:
    """``path`` relative to the project root when it lies inside it, else only its file name.

    Used wherever a path reaches a UI or an exported file, so neither leaks the user name or the
    directory layout of the machine that ran the model.
    """
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root().resolve()).as_posix()
    except (ValueError, OSError):
        return candidate.name or str(path)


@dataclass(frozen=True, eq=False)
class PredictionResult:
    """Probabilities ``prob [T, N]`` for the target hours of one scenario (T may be 0).

    ``prob_std`` (spread across rain-field members and MC-dropout passes) and ``depth_physics``
    (simulated depth, m) are optional ``[T, N]`` arrays; ``node_rain [T, N]`` is the (member-mean)
    junction rain (mm/h); ``static`` holds one row per junction (``node_id`` + static attributes).
    ``threshold`` is the alert probability, ``risk_tiers`` ``((name, lower_bound), ...)`` ascending
    from 0. ``member_prob [K, T, N]`` optionally keeps each rain-field member's probabilities
    (``prob`` is their mean), for the share of realisations in which a junction is at risk. All
    arrays are read-only copies.
    """

    timestamps: pd.DatetimeIndex
    prob: np.ndarray
    prob_std: np.ndarray | None
    node_rain: np.ndarray
    depth_physics: np.ndarray | None
    node_ids: tuple
    lon: np.ndarray
    lat: np.ndarray
    static: pd.DataFrame
    threshold: float
    horizons_h: tuple[int, ...]
    scenario_name: str
    risk_tiers: tuple[tuple[str, float], ...] = DEFAULT_TIERS
    predictor: str = "gnn"
    areal_mm: np.ndarray | None = None
    is_forecast: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    member_prob: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamps, pd.DatetimeIndex):
            raise ValueError("PredictionResult.timestamps must be a DatetimeIndex")
        n_hours, n_nodes = len(self.timestamps), len(self.node_ids)
        grid = (n_hours, n_nodes)
        prob = _readonly(self.prob, np.float32, "prob", grid)
        if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
            raise ValueError("PredictionResult.prob must be finite probabilities in [0, 1]")
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        set_("prob", prob)
        set_("node_rain", _readonly(self.node_rain, np.float32, "node_rain", grid))
        for name in ("prob_std", "depth_physics"):
            value = getattr(self, name)
            set_(name, None if value is None else _readonly(value, np.float32, name, grid))
        set_("areal_mm", None if self.areal_mm is None else _readonly(self.areal_mm, np.float64, "areal_mm",
                                                                      (n_hours,)))
        set_("is_forecast", None if self.is_forecast is None else _readonly(self.is_forecast, bool, "is_forecast",
                                                                            (n_hours,)))
        set_("lon", _readonly(self.lon, np.float64, "lon", (n_nodes,)))
        set_("lat", _readonly(self.lat, np.float64, "lat", (n_nodes,)))
        set_("node_ids", tuple(self.node_ids))
        if not isinstance(self.static, pd.DataFrame) or len(self.static) != n_nodes:
            raise ValueError(f"PredictionResult.static must be a DataFrame with {n_nodes} rows")
        set_("static", self.static.reset_index(drop=True).copy())
        threshold = float(self.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"PredictionResult.threshold must lie in [0, 1], got {self.threshold!r}")
        set_("threshold", threshold)
        horizons = tuple(sorted({int(h) for h in self.horizons_h}))
        if not horizons or horizons[0] < 1:
            raise ValueError(f"PredictionResult.horizons_h must be positive hours, got {self.horizons_h!r}")
        set_("horizons_h", horizons)
        tiers = tuple(sorted(((str(n), float(b)) for n, b in self.risk_tiers), key=lambda t: t[1]))
        if not tiers or tiers[0][1] != 0.0:
            raise ValueError(f"PredictionResult.risk_tiers must start at 0.0, got {self.risk_tiers!r}")
        set_("risk_tiers", tiers)
        set_("metadata", dict(self.metadata))
        if self.member_prob is not None:
            members = np.asarray(self.member_prob)
            if members.ndim != 3 or members.shape[0] < 1:
                raise ValueError(f"PredictionResult.member_prob must be [K, {n_hours}, {n_nodes}], got {members.shape}")
            member_prob = _readonly(members, np.float32, "member_prob", (members.shape[0], *grid))
            if not np.isfinite(member_prob).all() or (member_prob < 0).any() or (member_prob > 1).any():
                raise ValueError("PredictionResult.member_prob must be finite probabilities in [0, 1]")
            set_("member_prob", member_prob)

    # ------------------------------------------------------------------ basics

    @property
    def n_hours(self) -> int:
        return len(self.timestamps)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def _hours(self, hours: int) -> int:
        if isinstance(hours, bool) or not isinstance(hours, (int, np.integer)) or hours < 1:
            raise ValueError(f"hours must be a positive integer, got {hours!r}")
        return min(int(hours), self.n_hours)

    def horizon_max(self, hours: int) -> np.ndarray:
        """Max probability per junction over the first ``hours`` target hours (``[N]``; zeros if T = 0)."""
        h = self._hours(hours)
        return self.prob[:h].max(axis=0) if h else np.zeros(self.n_nodes, dtype=np.float32)

    def horizon_peak_index(self, hours: int) -> np.ndarray:
        """Index of each junction's peak-probability hour within the horizon (``-1`` if T = 0)."""
        h = self._hours(hours)
        return self.prob[:h].argmax(axis=0) if h else np.full(self.n_nodes, -1, dtype=np.int64)

    @property
    def n_members(self) -> int:
        """Rain-field members kept in :attr:`member_prob` (0 when not kept)."""
        return 0 if self.member_prob is None else int(self.member_prob.shape[0])

    def member_values(self, hours: int, hour_index: int | None = None) -> np.ndarray | None:
        """Per-member probabilities ``[K, N]``: the horizon maximum, or hour ``hour_index``; None without members."""
        if self.member_prob is None:
            return None
        h = self._hours(hours)
        if not h:
            return np.zeros((self.n_members, self.n_nodes), dtype=np.float32)
        if hour_index is None:
            return self.member_prob[:, :h].max(axis=1)
        return self.member_prob[:, int(np.clip(hour_index, 0, self.n_hours - 1))]

    def members_at_risk(self, hours: int, hour_index: int | None = None) -> np.ndarray | None:
        """Per junction, how many rain-field members reach the alert threshold (None without members)."""
        values = self.member_values(hours, hour_index)
        return None if values is None else (values >= self.threshold).sum(axis=0)

    def at_risk_range(self, hours: int, hour_index: int | None = None) -> tuple[int, int] | None:
        """``(min, max)`` junctions at risk across the rain-field members (None without members)."""
        values = self.member_values(hours, hour_index)
        if values is None:
            return None
        counts = (values >= self.threshold).sum(axis=1)
        return int(counts.min()), int(counts.max())

    def risk_tier(self, prob: Any) -> np.ndarray:
        """Tier name per probability (lower bounds inclusive); NaN counts as 0 (WARNING)."""
        values = np.array(prob, dtype=np.float64, copy=True)
        bad = ~np.isfinite(values)
        if bad.any():
            LOGGER.warning("risk_tier: %d non-finite probabilities treated as 0", int(bad.sum()))
            values[bad] = 0.0
        bounds = np.array([bound for _, bound in self.risk_tiers])
        names = np.array([name for name, _ in self.risk_tiers], dtype=object)
        index = np.clip(np.searchsorted(bounds, np.clip(values, 0.0, 1.0), side="right") - 1, 0, len(bounds) - 1)
        return names[index].astype(str)

    # ------------------------------------------------------------------ tables

    def node_table(self, hours: int = 48) -> pd.DataFrame:
        """One row per junction over the first ``hours`` target hours (columns :data:`NODE_TABLE_COLUMNS`)."""
        h = self._hours(hours)
        peak = self.horizon_peak_index(hours)
        max_prob = self.horizon_max(hours)
        cols = np.arange(self.n_nodes)
        table = pd.DataFrame({"node_id": list(self.node_ids), "lon": self.lon, "lat": self.lat})
        for name in STATIC_COLUMNS:
            table[name] = self.static[name].to_numpy(dtype=np.float64) if name in self.static else np.nan
        table["max_prob"] = max_prob.astype(np.float64)
        table["prob_std"] = (self.prob_std[peak, cols].astype(np.float64)
                             if self.prob_std is not None and h else np.nan)
        table["peak_time"] = self.timestamps[peak] if h else pd.NaT
        table["peak_rain_mm_h"] = self.node_rain[:h].max(axis=0).astype(np.float64) if h else 0.0
        table["tier"] = self.risk_tier(max_prob)
        table["above_threshold"] = max_prob >= self.threshold
        table["rain_total_mm"] = self.node_rain[:h].sum(axis=0).astype(np.float64) if h else 0.0
        table["max_depth_m"] = (self.depth_physics[:h].max(axis=0).astype(np.float64)
                                if self.depth_physics is not None and h else np.nan)
        at_risk = self.members_at_risk(hours)
        table["field_share_at_risk"] = np.nan if at_risk is None else at_risk / float(self.n_members)
        return table[list(NODE_TABLE_COLUMNS)]

    def top_k(self, k: int = 15, hours: int = 48) -> pd.DataFrame:
        """The ``k`` riskiest junctions (max probability, then simulated depth, then peak rain), ranked from 1."""
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        table = self.node_table(hours).sort_values(["max_prob", "max_depth_m", "peak_rain_mm_h"], ascending=False,
                                                   kind="stable", na_position="last")
        top = table.head(int(k)).reset_index(drop=True)
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        return top

    def summary(self, hours: int = 48) -> dict[str, Any]:
        """Headline KPIs over the first ``hours`` target hours plus per-horizon counts."""
        h = self._hours(hours)
        max_prob = self.horizon_max(hours)
        hourly_at_risk = (self.prob[:h] >= self.threshold).sum(axis=1) if h else np.zeros(0, dtype=np.int64)
        peak_hour = None
        if h:
            peak_hour = int(hourly_at_risk.argmax()) if hourly_at_risk.max() > 0 else int(self.prob[:h].max(1).argmax())
        rain = self.areal_mm[:h] if self.areal_mm is not None else self.node_rain[:h].mean(axis=1)
        tiers = pd.Series(self.risk_tier(max_prob)).value_counts()
        spread = self.at_risk_range(hours)
        return {
            "scenario": self.scenario_name, "predictor": self.predictor, "hours": h, "n_nodes": self.n_nodes,
            "start": self.timestamps[0].isoformat() if h else None,
            "end": self.timestamps[h - 1].isoformat() if h else None,
            "threshold": self.threshold, "junctions_at_risk": int((max_prob >= self.threshold).sum()),
            "max_prob": float(max_prob.max()) if self.n_nodes else 0.0,
            "peak_time": self.timestamps[peak_hour].isoformat() if peak_hour is not None else None,
            "peak_junctions_at_risk": int(hourly_at_risk.max()) if h else 0,
            "rain_total_mm": float(np.sum(rain)), "peak_rain_mm_h": float(np.max(rain)) if h else 0.0,
            "tier_counts": {name: int(tiers.get(name, 0)) for name, _ in self.risk_tiers},
            "field_members": self.n_members or int(self.metadata.get("field_members") or 1),
            "junctions_at_risk_range": None if spread is None else list(spread),
            "horizons": {int(hz): {"junctions_at_risk": int((self.horizon_max(hz) >= self.threshold).sum()),
                                   "max_prob": float(self.horizon_max(hz).max()) if self.n_nodes else 0.0}
                         for hz in self.horizons_h if hz <= self.n_hours},
        }

    # ------------------------------------------------------------------ exports

    def to_geojson(self, hours: int = 48, include_timeline: bool = False) -> dict[str, Any]:
        """GeoJSON ``FeatureCollection`` of junction points with the node-table properties.

        ``include_timeline`` adds each junction's hourly probabilities (``prob_timeline``).
        A top-level ``metadata`` member carries the scenario, predictor and :meth:`summary`.
        """
        table = self.node_table(hours)
        h = self._hours(hours)
        features = []
        for i, row in enumerate(table.to_dict(orient="records")):
            properties = json_safe(row)
            if include_timeline:
                properties["prob_timeline"] = [round(float(p), 4) for p in self.prob[:h, i]]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(self.lon[i]), float(self.lat[i])]},
                "properties": properties,
            })
        metadata = {
            "scenario": self.scenario_name, "predictor": self.predictor, "threshold": self.threshold,
            "horizon_hours": h, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timeline_start": self.timestamps[0].isoformat() if h else None,
            "summary": self.summary(hours), "run": self.metadata,
        }
        return {"type": "FeatureCollection", "features": features, "metadata": json_safe(metadata)}

    def write_geojson(self, path: str | Path, hours: int = 48, include_timeline: bool = False) -> Path:
        """Write :meth:`to_geojson` atomically; returns the path."""
        payload = json.dumps(self.to_geojson(hours, include_timeline), separators=(",", ":"), allow_nan=False)
        return atomic_write_text(Path(path), payload)

    def to_csv(self, path: str | Path, hours: int = 48) -> Path:
        """Write :meth:`node_table` (ISO ``peak_time``) atomically; returns the path."""
        table = self.node_table(hours)
        table["peak_time"] = [None if pd.isna(t) else t.isoformat() for t in table["peak_time"]]
        return atomic_write_text(Path(path), table.to_csv(index=False))
