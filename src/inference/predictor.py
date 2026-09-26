"""Junction flood-probability predictors and their result object.

* :class:`FloodPredictor` — the trained spatio-temporal GNN (``best.pt``). Inference mirrors
  training exactly: junction rain from :func:`~src.data_pipeline.rain_field.downscale_rainfall`
  (on :attr:`Scenario.field_timestamps`), features from
  :func:`~src.data_pipeline.features.build_node_features` with the checkpoint's scaler, and the
  target hours tiled by ``seq_len`` windows whose scored steps (``>= warmup_steps``) cover them —
  each window starts ``warmup_steps`` hours before its first target hour, like a training
  window, so the GRU never runs far past ``seq_len``. Windows are batched as disjoint graph
  copies. Probabilities are ``apply_calibration(logit, calibration)`` with the checkpoint's
  calibration (Platt ``sigmoid(slope * logit + intercept)`` in format v2, ``sigmoid(logit / T)``
  in v1) — the training package's single implementation.
* :class:`PhysicsPredictor` — the hydrology simulator (the label "teacher") as a baseline and
  as the fallback when no trained model exists: ``p = sigmoid((depth - threshold) / softness)``.

Rain-field ensemble (contract X5): the labels were simulated from ONE stochastic junction rain
field, which only a replay of the training record knows (``Scenario.exact_field``). Every other
scenario is predicted with ``K = inference.field_members`` independent field realisations
(:mod:`src.inference.field_ensemble`): the GNN runs ``max(K, mc_samples)`` passes with MC dropout
(else ``K``), pass ``i`` on member ``i % K``, batched like windows; ``prob`` is the mean over the
passes and ``prob_std`` their spread (rain-field members and MC-dropout passes). The physics
baseline averages probability and depth over the members.

Only finalized checkpoints are served (``finalized_utc``, set when training publishes
``best.pt``); an in-progress / interrupted per-epoch file raises :class:`ModelNotReady`. A
checkpoint whose ``graph_attributes_sha256`` differs from the current graph's static
attributes raises :class:`CheckpointMismatch`. Checkpoints load only with the safe
``weights_only`` unpickler — a file that needs the full unpickler is refused, never executed.

Both return a :class:`PredictionResult` (per-junction tables, risk tiers, GeoJSON / CSV).
:func:`load_predictor` picks the GNN and falls back to physics when the model is not ready.
"""

from __future__ import annotations

import math

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.data_pipeline.features import build_node_features
from src.data_pipeline.graph_io import GraphArrays, GraphFormatError, graph_to_arrays, load_graph
from src.hydrology.simulator import HydrologyParams, UrbanDrainageSimulator, physics_flood_probability
from src.inference.checkpoint_loading import (  # noqa: F401 - load_checkpoint_file is part of this module's API
    EDGE_FEATURES,
    REQUIRED_CHECKPOINT_KEYS,
    SUPPORTED_FORMAT_VERSIONS,
    build_checkpoint_model,
    check_attributes,
    check_finalized,
    check_graph_fit,
    checkpoint_features,
    checkpoint_serving_calibration,
    checkpoint_window,
    load_checkpoint_file,
    warn_config_drift,
)
from src.inference.errors import (  # noqa: F401 - re-exported (historical import path)
    FINALIZE_HINT,
    GRAPH_HINT,
    RETRAIN_HINT,
    TRAIN_HINT,
    CheckpointMismatch,
    GraphNotReady,
    ModelNotFinalized,
    ModelNotReady,
    PredictionError,
    remediation,
)
from src.inference.field_ensemble import field_plan, member_rain, passes_for, rain_field_note
from src.inference.results import DEFAULT_TIERS, STATIC_COLUMNS, PredictionResult, display_path
from src.inference.scenarios import InferenceSettings, Scenario
from src.models.stgcn import enable_mc_dropout
from src.training.calibration import apply_calibration
from src.utils.config import resolve_path
from src.utils.logger import get_logger
from src.utils.runtime import resolve_device

LOGGER = get_logger(__name__)

__all__ = [
    "CheckpointMismatch", "FloodPredictor", "GraphNotReady", "ModelNotFinalized", "ModelNotReady", "PhysicsPredictor",
    "PredictionError", "PredictionResult", "load_checkpoint_file", "load_graph_arrays", "load_predictor",
    "remediation",
]

# --------------------------------------------------------------------------- shared loaders


def load_graph_arrays(cfg: Mapping[str, Any]) -> tuple[GraphArrays, dict[str, str]]:
    """Load ``paths.graph_file`` as :class:`GraphArrays` plus its provenance graph attributes."""
    path = resolve_path(cfg, "graph_file")
    try:
        G = load_graph(path)
        arrays = graph_to_arrays(G)
    except FileNotFoundError as exc:
        raise GraphNotReady(f"Road graph {path} not found; {GRAPH_HINT}") from exc
    except (GraphFormatError, OSError, ValueError) as exc:
        raise GraphNotReady(f"Road graph {path} is unreadable ({exc}); {GRAPH_HINT}") from exc
    provenance = {k: str(G.graph.get(k, "unknown")) for k in ("source", "elevation_source", "drain_source")}
    return arrays, provenance


def _static_frame(graph: GraphArrays) -> pd.DataFrame:
    """Per-junction static attributes (NaN where the graph lacks one)."""
    data: dict[str, Any] = {"node_id": list(graph.node_ids)}
    for name in (*STATIC_COLUMNS, "flow_accumulation", "is_sink"):
        values = graph.node_attrs.get(name)
        if values is not None:
            data[name] = np.asarray(values, dtype=np.float64)
        elif name in STATIC_COLUMNS:
            data[name] = np.full(graph.num_nodes, np.nan)
    frame = pd.DataFrame(data)
    if "is_sink" in frame:
        frame["is_sink"] = frame["is_sink"].astype(bool)
    return frame


def _positive_samples(value: Any, cap: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"mc_samples must be a non-negative integer, got {value!r}")
    if value > cap:
        raise ValueError(f"mc_samples={value} exceeds inference.max_mc_samples={cap}")
    return int(value)


def _optional_threshold(value: Any, name: str) -> float | None:
    """A stored probability threshold in (0, 1), or None when absent / invalid (WARNING if invalid)."""
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float("nan")
    if not (math.isfinite(threshold) and 0.0 < threshold < 1.0):
        LOGGER.warning("Checkpoint %s %r is invalid; ignoring it", name, value)
        return None
    return threshold


@dataclass(frozen=True, eq=False)
class _Probabilities:
    """What a predictor computes for the target hours: mean probability, spread, depth, passes."""

    prob: np.ndarray
    std: np.ndarray | None
    depth: np.ndarray | None
    passes: int
    mc_samples: int = 0
    member_prob: np.ndarray | None = None


def _mean_and_spread(total: np.ndarray, total_sq: np.ndarray | None, count: int
                     ) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean (clipped to [0, 1]) and population std of ``count`` accumulated draws."""
    mean = total / count
    if total_sq is None:
        return mean.astype(np.float32), None
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
    return np.clip(mean, 0.0, 1.0).astype(np.float32), std.astype(np.float32)


# --------------------------------------------------------------------------- base predictor


class _BasePredictor:
    """Graph handling, input validation and result assembly shared by both predictors."""

    kind = "base"
    label = "base"

    def __init__(self, cfg: Mapping[str, Any], graph: GraphArrays, provenance: Mapping[str, str] | None = None):
        if not isinstance(graph, GraphArrays):
            raise TypeError(f"graph must be GraphArrays, got {type(graph).__name__}")
        self.cfg = cfg
        self.graph = graph
        self.provenance = dict(provenance or {})
        self.settings = InferenceSettings.from_config(cfg)
        self.static = _static_frame(graph)
        self._simulator: UrbanDrainageSimulator | None = None
        self._lock = threading.RLock()  # the app shares one predictor across sessions (MC toggles train mode)

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    @property
    def threshold(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def threshold_for_members(self, members: int) -> float:
        """Alert threshold for probabilities averaged over ``members`` rain-field realisations.

        One member (a replay's exact training field) uses :attr:`threshold`; subclasses with a
        separately chosen threshold for ensemble-averaged probabilities override this.
        """
        return self.threshold

    @property
    def metrics(self) -> dict[str, Any]:
        return {}

    @property
    def history_needed(self) -> int:
        """Hours of rain needed before the first target hour (0: the simulator spins up from dry)."""
        return 0

    def predict(self, scenario: Scenario, mc_samples: int = 0, *, field_members: int | None = None,
                field_seed_offset: int = 0, include_physics: bool = False,
                keep_members: bool = False) -> PredictionResult:
        """Predict every target hour of ``scenario`` (junction rain from the scenario's areal series).

        A scenario with ``exact_field`` (a replay of the training record) uses the training rain
        field; any other scenario averages over ``field_members`` (default
        ``inference.field_members``) stochastic rain-field realisations whose seeds start at
        ``field_seed_offset`` (see :mod:`src.inference.field_ensemble`). ``keep_members`` also
        stores each member's probabilities (:attr:`PredictionResult.member_prob`; only when there is
        more than one member).
        """
        if not isinstance(scenario, Scenario):
            raise TypeError(f"scenario must be a Scenario, got {type(scenario).__name__}")
        plan = field_plan(self.cfg, self.settings, scenario.exact_field, field_members, field_seed_offset)
        rain = member_rain(self.cfg, plan, scenario.areal_mm, scenario.field_timestamps, self.graph.lon,
                           self.graph.lat)
        extra = {"rain_field": rain_field_note(plan, design_storm=bool(scenario.field_shift_h)),
                 "rain_field_shift_h": scenario.field_shift_h, "scenario_notes": list(scenario.notes),
                 **plan.metadata()}
        return self.predict_rain(
            rain, scenario.timestamps, scenario.target_start, mc_samples=mc_samples, scenario_name=scenario.name,
            areal_mm=scenario.areal_mm, is_forecast=scenario.is_forecast, include_physics=include_physics,
            metadata=extra, keep_members=keep_members,
        )

    def predict_rain(self, node_rain: np.ndarray, timestamps: pd.DatetimeIndex, target_start: int, *,
                     mc_samples: int = 0, scenario_name: str = "custom", areal_mm: np.ndarray | None = None,
                     is_forecast: np.ndarray | None = None, include_physics: bool = False,
                     metadata: Mapping[str, Any] | None = None, keep_members: bool = False) -> PredictionResult:
        """Predict from junction rain ``[T, N]`` — or ``[K, T, N]`` for ``K`` rain-field members —
        directly (rows aligned with ``timestamps``).

        Probabilities are the mean over every pass (one per member, or ``max(K, mc_samples)``
        with MC dropout, pass ``i`` using member ``i % K``); ``prob_std`` is the spread over the
        passes when there is more than one; ``node_rain`` is the member-mean junction rain.
        ``metadata`` is merged into :attr:`PredictionResult.metadata`, which also records
        ``padded_history_h``: hours of missing history the model treated as dry (0 when the
        scenario had at least :attr:`history_needed` hours before ``target_start``).
        """
        rain = self._validate_rain(node_rain, timestamps, target_start)
        samples = _positive_samples(mc_samples, self.settings.max_mc_samples)
        started = time.perf_counter()
        keep = bool(keep_members) and rain.shape[0] > 1   # one member: its probabilities are ``prob``
        with self._lock:
            out = self._predict_probabilities(rain, int(target_start), samples, include_physics, keep)
        elapsed = time.perf_counter() - started
        t0 = int(target_start)
        members = rain.shape[0]
        meta = {"predictor": self.kind, "label": self.label, "mc_samples": out.mc_samples, "passes": out.passes,
                "threshold_kind": self._threshold_kind(members), "field_members": members, "field_seed_offset": 0, "elapsed_s": round(elapsed, 3), "history_hours": t0,
                "history_needed_h": self.history_needed, "padded_history_h": max(0, self.history_needed - t0),
                **self._metadata(), **dict(metadata or {})}
        LOGGER.info("%s: predicted %d h x %d junctions in %.2f s (%d rain-field member(s), %d pass(es), MC samples: "
                    "%d)", self.label, rain.shape[1] - t0, self.num_nodes, elapsed, members, out.passes,
                    out.mc_samples)
        return PredictionResult(
            timestamps=timestamps[t0:], prob=out.prob, prob_std=out.std, node_rain=rain[:, t0:].mean(axis=0),
            depth_physics=out.depth, node_ids=self.graph.node_ids, lon=self.graph.lon, lat=self.graph.lat,
            static=self.static, threshold=self.threshold_for_members(members), horizons_h=self.settings.horizons_h,
            scenario_name=str(scenario_name), risk_tiers=self.settings.risk_tiers or DEFAULT_TIERS,
            predictor=self.kind, areal_mm=None if areal_mm is None else np.asarray(areal_mm, dtype=np.float64)[t0:],
            is_forecast=None if is_forecast is None else np.asarray(is_forecast, dtype=bool)[t0:], metadata=meta,
            member_prob=out.member_prob,
        )

    # ------------------------------------------------------------------ internals

    def _metadata(self) -> dict[str, Any]:
        return {"graph": dict(self.provenance)}

    def _threshold_kind(self, members: int) -> str:
        return "areal_ensemble" if self.threshold_for_members(members) != self.threshold else "exact_field"

    def _predict_probabilities(self, rain: np.ndarray, t0: int, samples: int, include_physics: bool,
                               keep_members: bool) -> "_Probabilities":
        raise NotImplementedError  # pragma: no cover

    def _validate_rain(self, node_rain: Any, timestamps: Any, target_start: Any) -> np.ndarray:
        """``float32 [K, T, N]`` copy of ``node_rain`` (``[T, N]`` is one member)."""
        try:
            rain = np.array(node_rain, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"node rain must be numeric: {exc}") from exc
        if rain.ndim == 2:
            rain = rain[None]
        if rain.ndim != 3 or rain.shape[2] != self.num_nodes or rain.shape[0] < 1:
            raise PredictionError(f"node rain must be [T, {self.num_nodes}] or [members, T, {self.num_nodes}] "
                                  f"(graph junctions), got {np.shape(node_rain)}")
        if rain.shape[1] == 0:
            raise PredictionError("node rain has no hours (T = 0)")
        if not np.isfinite(rain).all() or (rain < 0).any():
            raise PredictionError("node rain must be finite and >= 0 mm/h")
        if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) != rain.shape[1]:
            raise PredictionError(f"timestamps must be a DatetimeIndex of length {rain.shape[1]}")
        if isinstance(target_start, bool) or not isinstance(target_start, (int, np.integer)) \
                or not 0 <= int(target_start) < rain.shape[1]:
            raise PredictionError(f"target_start must index the {rain.shape[1]} hours, got {target_start!r}")
        return rain

    def _mean_depth(self, rain: np.ndarray, t0: int) -> np.ndarray:
        """Member-mean simulated depth ``[T - t0, N]`` (m) of ``rain [K, T, N]``."""
        total = np.zeros((rain.shape[1] - t0, rain.shape[2]), dtype=np.float64)
        for member in rain:
            total += self._physics_depth(member)[t0:]
        return (total / rain.shape[0]).astype(np.float32)

    def _physics_depth(self, rain: np.ndarray) -> np.ndarray:
        """Simulated depth ``[T, N]`` (m) of the full rain series (history spins the state up)."""
        if self._simulator is None:
            try:
                self._simulator = UrbanDrainageSimulator(self.graph, HydrologyParams.from_config(self.cfg))
            except ValueError as exc:
                raise GraphNotReady(f"The road graph cannot drive the hydrology simulator: {exc}") from exc
        return self._simulator.run(rain).depth_m


# --------------------------------------------------------------------------- physics baseline


class PhysicsPredictor(_BasePredictor):
    """Hydrology-simulator baseline with the same ``predict`` API (deterministic; no MC dropout)."""

    kind = "physics"
    label = "Physics baseline"

    def __init__(self, cfg: Mapping[str, Any], graph: GraphArrays, provenance: Mapping[str, str] | None = None):
        super().__init__(cfg, graph, provenance)
        self.params = HydrologyParams.from_config(cfg)

    @classmethod
    def from_graph(cls, cfg: Mapping[str, Any]) -> "PhysicsPredictor":
        """Load ``paths.graph_file`` (raises :class:`GraphNotReady` if missing / unusable)."""
        arrays, provenance = load_graph_arrays(cfg)
        predictor = cls(cfg, arrays, provenance)
        predictor._physics_depth(np.zeros((1, arrays.num_nodes), dtype=np.float32))  # validates the graph early
        return predictor

    @property
    def threshold(self) -> float:
        return 0.5  # p = 0.5 exactly at hydrology.flood_depth_threshold_m

    def _metadata(self) -> dict[str, Any]:
        return {**super()._metadata(), "flood_depth_threshold_m": self.params.flood_depth_threshold_m,
                "softness_m": self.settings.physics_softness_m}

    def _predict_probabilities(self, rain: np.ndarray, t0: int, samples: int, include_physics: bool,
                               keep_members: bool) -> _Probabilities:
        """Member-mean flood probability and depth; the spread across rain-field members as ``std``."""
        if samples:
            LOGGER.warning("The physics baseline is deterministic; ignoring mc_samples=%d", samples)
        members = rain.shape[0]
        shape = (rain.shape[1] - t0, rain.shape[2])
        total, depth_total = np.zeros(shape), np.zeros(shape)
        total_sq = np.zeros(shape) if members > 1 else None
        kept = []
        for member in rain:
            depth = self._physics_depth(member)[t0:]
            prob = physics_flood_probability(depth, self.params.flood_depth_threshold_m,
                                             self.settings.physics_softness_m).astype(np.float64)
            total += prob
            depth_total += depth
            if total_sq is not None:
                total_sq += prob * prob
            if keep_members:
                kept.append(prob.astype(np.float32))
        mean, std = _mean_and_spread(total, total_sq, members)
        return _Probabilities(mean, std, (depth_total / members).astype(np.float32), members,
                              member_prob=np.stack(kept) if keep_members else None)


# --------------------------------------------------------------------------- trained GNN


CARD_KEYS = ("format_version", "epoch", "best_score", "created_utc", "finalized_utc", "config_hash",
             "dataset_config_hash", "graph_attributes_sha256")


class FloodPredictor(_BasePredictor):
    """Trained GNN predictor; build it with :meth:`from_artifacts`."""

    kind = "gnn"
    label = "Namma-Flow GNN"

    def __init__(self, cfg: Mapping[str, Any], graph: GraphArrays, checkpoint: Mapping[str, Any], *,
                 device: Any = "cpu", checkpoint_path: str | Path | None = None,
                 provenance: Mapping[str, str] | None = None) -> None:
        super().__init__(cfg, graph, provenance)
        self.device = resolve_device(device)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        check_finalized(checkpoint, self.checkpoint_path.name if self.checkpoint_path else "given")
        self.scaler, self.rolling_windows_h, self.static_feature_names = checkpoint_features(checkpoint)
        self.seq_len, self.warmup_steps, self.lookback_hours = checkpoint_window(checkpoint)
        self.feature_names = [str(n) for n in checkpoint["feature_names"]]
        check_graph_fit(checkpoint, graph, self.static_feature_names)
        check_attributes(checkpoint, graph, self.static_feature_names)
        self.calibration, self.temperature, self._threshold = checkpoint_serving_calibration(checkpoint)
        self._areal_threshold = _optional_threshold(checkpoint.get("areal_threshold"), "areal_threshold")
        self.model = build_checkpoint_model(checkpoint, len(self.feature_names), self.device)
        self.architecture = str(self.model.get_config()["architecture"])
        self._metrics = dict(checkpoint.get("metrics") or {})
        self._info = {k: checkpoint.get(k) for k in CARD_KEYS}
        if self.settings.history_hours < self.history_needed:
            LOGGER.warning("inference.history_hours=%d is shorter than the %d h of rain history this model needs "
                           "(lookback %d + warm-up %d); design storms and replays will be padded with dry hours",
                           self.settings.history_hours, self.history_needed, self.lookback_hours, self.warmup_steps)
        self.static_raw = graph.node_matrix(self.static_feature_names)
        edge_raw = graph.edge_matrix(EDGE_FEATURES) if graph.num_edges else np.zeros((0, 2), dtype=np.float32)
        self.edge_index = torch.as_tensor(np.asarray(graph.edge_index, dtype=np.int64).reshape(2, -1))
        self.edge_attr = torch.from_numpy(self.scaler.transform_edges(edge_raw[:, 0], edge_raw[:, 1]))
        self._batched: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        warn_config_drift(checkpoint, cfg)

    @classmethod
    def from_artifacts(cls, cfg: Mapping[str, Any], checkpoint_path: str | Path | None = None,
                       device: Any = "cpu") -> "FloodPredictor":
        """Load the road graph and ``best.pt``; raises :class:`ModelNotReady` / :class:`CheckpointMismatch`
        (model unusable) or :class:`GraphNotReady` (graph missing)."""
        settings = InferenceSettings.from_config(cfg)
        default = resolve_path(cfg, "checkpoint_dir") / settings.checkpoint_name
        path = Path(checkpoint_path) if checkpoint_path else default
        checkpoint = load_checkpoint_file(path)
        arrays, provenance = load_graph_arrays(cfg)
        predictor = cls(cfg, arrays, checkpoint, device=device, checkpoint_path=path, provenance=provenance)
        cal = predictor.calibration
        LOGGER.info("Loaded %s checkpoint %s (format v%s, epoch %s, %s calibration slope %.4g intercept %+.4g, "
                    "threshold=%.4g) on %s", predictor.architecture, path, predictor._info.get("format_version") or 1,
                    predictor._info.get("epoch"), cal["method"], cal["slope"], cal["intercept"], predictor.threshold,
                    predictor.device)
        return predictor

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def areal_threshold(self) -> float | None:
        """Alert threshold chosen on validation for rain-field-ensemble probabilities (None: not stored)."""
        return self._areal_threshold

    def threshold_for_members(self, members: int) -> float:
        if int(members) > 1 and self._areal_threshold is not None:
            return self._areal_threshold
        return self._threshold

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def history_needed(self) -> int:
        """Hours of rain needed before the first target hour (lookback + warm-up)."""
        return self.lookback_hours + self.warmup_steps

    def describe(self) -> dict[str, Any]:
        """Model card fields for the app / CLI (the checkpoint path relative to the project root)."""
        return {"architecture": self.architecture, "checkpoint": display_path(self.checkpoint_path),
                "calibration": dict(self.calibration), "temperature": self.temperature, "threshold": self.threshold,
                "areal_threshold": self.areal_threshold,
                "seq_len": self.seq_len, "warmup_steps": self.warmup_steps, "lookback_hours": self.lookback_hours,
                "feature_names": list(self.feature_names), "device": str(self.device), **self._info}

    def _metadata(self) -> dict[str, Any]:
        """Run metadata for exports: the checkpoint's file name and identity, never its absolute path."""
        return {**super()._metadata(), "architecture": self.architecture, "calibration": dict(self.calibration),
                "temperature": self.temperature,
                "checkpoint": self.checkpoint_path.name if self.checkpoint_path is not None else None,
                **{f"checkpoint_{k}": self._info.get(k) for k in ("epoch", "created_utc", "finalized_utc")}}

    # ------------------------------------------------------------------ inference

    def _predict_probabilities(self, rain: np.ndarray, t0: int, samples: int, include_physics: bool,
                               keep_members: bool) -> _Probabilities:
        """Mean over ``passes_for(K, samples)`` passes (pass ``i`` on member ``i % K``; dropout on only
        with MC samples) and their spread."""
        depth = self._mean_depth(rain, t0) if include_physics else None
        n_target = rain.shape[1] - t0
        if t0 < self.history_needed:
            LOGGER.warning("Only %d h of rain history before the first target hour; the model needs %d h "
                           "(lookback %d + warm-up %d): padding with dry hours", t0, self.history_needed,
                           self.lookback_hours, self.warmup_steps)
            pad = np.zeros((rain.shape[0], self.history_needed - t0, rain.shape[2]), dtype=np.float32)
            rain, t0 = np.concatenate([pad, rain], axis=1), self.history_needed
        passes = passes_for(rain.shape[0], samples)
        if samples == 0:
            mean, std, members = self._run_windows(rain, t0, n_target, passes, keep_members)
            return _Probabilities(mean, std, depth, passes, 0, members)
        seed = int((self.cfg.get("project") or {}).get("seed", 42))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            enable_mc_dropout(self.model)
            try:
                mean, std, members = self._run_windows(rain, t0, n_target, passes, keep_members)
            finally:
                self.model.eval()
        return _Probabilities(mean, std, depth, passes, samples, members)

    def _run_windows(self, rain: np.ndarray, t0: int, n_target: int, passes: int,
                     keep_members: bool = False) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """``(mean, std, member means)`` probability ``[n_target, N]`` over ``passes`` passes of the tiled windows.

        ``rain`` is ``[K, T, N]``; pass ``p`` runs member ``p % K``. Windows x passes are batched
        as disjoint graph copies (``inference.batch_windows`` per forward call) and accumulated
        (sum and sum of squares) rather than stored, so memory does not grow with the number of
        passes; ``std`` is None for a single pass. ``member means`` (``[K, n_target, N]``, each
        member averaged over its own passes) is returned only when ``keep_members``.
        """
        members, n_nodes = rain.shape[0], rain.shape[2]
        scored = self.seq_len - self.warmup_steps
        n_windows = -(-n_target // scored)
        needed = t0 + n_windows * scored
        if rain.shape[1] < needed:  # the model is causal: trailing dry padding never changes earlier hours
            pad = np.zeros((members, needed - rain.shape[1], n_nodes), dtype=np.float32)
            rain = np.concatenate([rain, pad], axis=1)
        starts = [t0 + k * scored - self.warmup_steps for k in range(n_windows)]
        items = [(k, p % members) for k in range(n_windows) for p in range(passes)]
        total = np.zeros((n_windows * scored, n_nodes), dtype=np.float64)
        total_sq = np.zeros_like(total) if passes > 1 else None
        by_member = np.zeros((members, *total.shape), dtype=np.float64) if keep_members else None
        cache: dict[tuple[int, int], np.ndarray] = {}

        def features(k: int, m: int) -> np.ndarray:
            if (k, m) not in cache:
                cache[(k, m)] = self._window_features(rain[m], starts[k])
            return cache[(k, m)]

        batch_size = self.settings.batch_windows
        for lo in range(0, len(items), batch_size):
            batch = items[lo: lo + batch_size]
            for key in [key for key in cache if key[0] < batch[0][0]]:
                del cache[key]
            probs = self._forward(np.stack([features(k, m) for k, m in batch]))[:, self.warmup_steps:]
            for (k, m), window_probs in zip(batch, probs.astype(np.float64)):
                rows = slice(k * scored, (k + 1) * scored)
                total[rows] += window_probs
                if total_sq is not None:
                    total_sq[rows] += window_probs * window_probs
                if by_member is not None:
                    by_member[m, rows] += window_probs
        mean, std = _mean_and_spread(total[:n_target], None if total_sq is None else total_sq[:n_target], passes)
        if by_member is None:
            return mean, std, None
        counts = np.bincount(np.arange(passes) % members, minlength=members).astype(np.float64)
        return mean, std, (by_member[:, :n_target] / counts[:, None, None]).astype(np.float32)

    def _window_features(self, rain: np.ndarray, start: int) -> np.ndarray:
        """Features ``[seq_len, N, F]`` of the window whose first hour is ``start`` (as in training)."""
        chunk = rain[start - self.lookback_hours: start + self.seq_len]
        return build_node_features(self.static_raw, chunk, self.lookback_hours, self.rolling_windows_h, self.scaler)

    def _batched_graph(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch not in self._batched:
            n_edges = self.edge_index.shape[1]
            offsets = (torch.arange(batch, dtype=torch.long) * self.num_nodes).repeat_interleave(n_edges)
            index = (self.edge_index.repeat(1, batch) + offsets).to(self.device)
            self._batched[batch] = (index, self.edge_attr.repeat(batch, 1).to(self.device))
        return self._batched[batch]

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """``x [B, seq_len, N, F]`` → calibrated probabilities ``[B, seq_len, N]``."""
        batch, steps, nodes, features = x.shape
        stacked = torch.from_numpy(np.ascontiguousarray(x.transpose(1, 0, 2, 3)).reshape(steps, batch * nodes,
                                                                                       features)).to(self.device)
        edge_index, edge_attr = self._batched_graph(batch)
        with torch.inference_mode():
            logits, _ = self.model.forward_sequence(stacked, edge_index, edge_attr, return_logits=True)
            probs = apply_calibration(logits, self.calibration)
        return probs.reshape(steps, batch, nodes).permute(1, 0, 2).float().cpu().numpy()


# --------------------------------------------------------------------------- selection


def load_predictor(cfg: Mapping[str, Any], *, prefer: str = "gnn", checkpoint_path: str | Path | None = None,
                   device: Any = "cpu") -> tuple[_BasePredictor, str | None]:
    """``(predictor, fallback_reason)``: the GNN, or the physics baseline when the model is not ready.

    ``prefer="physics"`` always returns the physics baseline. :class:`GraphNotReady` propagates
    (no predictor can run without the road graph).
    """
    choice = str(prefer).strip().lower()
    if choice not in ("gnn", "physics"):
        raise ValueError(f"prefer must be 'gnn' or 'physics', got {prefer!r}")
    if choice == "physics":
        return PhysicsPredictor.from_graph(cfg), None
    try:
        return FloodPredictor.from_artifacts(cfg, checkpoint_path=checkpoint_path, device=device), None
    except ModelNotReady as exc:
        LOGGER.warning("Falling back to the physics baseline: %s", exc)
        return PhysicsPredictor.from_graph(cfg), str(exc)
