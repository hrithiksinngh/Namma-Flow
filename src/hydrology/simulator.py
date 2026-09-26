"""Urban drainage simulator: the physics "teacher" that turns junction rainfall into flood labels.

Bengaluru has no public, junction-level flood record long enough to train a network on, so
Namma-Flow derives labels from a deliberately simple, fully vectorised hydrological model
that runs over the road graph itself (contract schema 2.4). It is a lumped *dual-drainage*
model: every junction ``i`` owns a small surface reservoir (the major system: road surface,
kerbs, depressions) fed by the road strip around it, connected to its neighbours by overland
flow and drained by street inlets into a minor system (storm drains) that can surcharge.

* **Catchment** ``A_i = catchment_width_m * sum(half length of incident street segments)``
  (at least 1 m^2); the pondable surface is ``ponding_fraction * A_i``.
* **Runoff** ``q_i = C_i(t) * max(r_i - infiltration_mm_h, 0) / 1000 * A_i`` (m^3/h) where the
  runoff coefficient rises linearly from ``runoff_coeff_dry`` to ``runoff_coeff_wet`` as the
  junction's rain over the previous ``antecedent_window_h`` hours approaches
  ``antecedent_saturation_mm`` (soils and open plots saturate during long spells).
* **Inlet drainage** removes up to ``A_i * cap_i * s_i(t) / 1000`` m^3/h with
  ``cap_i = far + (near - far) * exp(-dist_to_drain_m / drain_decay_m)`` mm/h: junctions next
  to a rajakaluve / lake have short, large connections to it; far ones rely on small roadside
  drains.
* **Local drain surcharge** ``s_i(t)``. The drain reach serving junction ``i`` carries the
  runoff of its whole contributing area — the junctions upstream of ``i`` in the same MFD
  graph, because roadside drains follow the streets and the terrain. Drains are sized in
  proportion to the area they serve, so a reach is overloaded when the *flow-weighted mean
  rain over its contributing area*, accumulated over the time of concentration
  (``surcharge_window_h``), exceeds what it was built for. The local excess
  ``E_i = clip((L_i - surcharge_threshold_mm) / surcharge_ramp_mm, 0, 1)`` uses the loading
  ``L_i(t) = w sum_j K_ij R_j(t) + (1 - w) mean_j R_j(t)`` where ``R_j`` is junction ``j``'s rain
  over the window, ``K`` the contributing-area operator of :mod:`src.hydrology.drainage` and
  ``w = surcharge_local_weight`` (1 = purely local; 0 reproduces the legacy basin-wide switch
  of fix-round finding R2-01 and is kept for ablations only).
* **Tailwater gate** ``G(t)``. Street inlets are throttled by *backwater*, which needs an
  elevated water level in the receiving trunk drains / rajakaluves / lake; that level responds
  to corridor-wide rain. ``G`` rises linearly from 0 to 1 as the basin-mean rain over the same
  window goes from ``tailwater_threshold_mm`` to ``tailwater_threshold_mm + tailwater_ramp_mm``.
  The drain capacity multiplier is ``s_i = 1 - (1 - surcharge_min_factor) * E_i * G``: the gate
  decides *when* backwater is possible (a single convective burst over a dry corridor drains
  freely into an empty trunk system), the local loading decides *where* it happens — a
  cloudburst over one sub-catchment surcharges the drains of that sub-catchment and of the
  junctions downstream of it, while drains a few hundred metres away keep working, so the same
  areal rain placed elsewhere floods different junctions.
* **Routing** over the undirected road adjacency (multiple-flow-direction): a junction releases
  ``min(1, outflow_rate_per_h * max(slope_i, min_routing_grade) / 0.01)`` of its remaining
  water per hour (linear reservoir; ``slope_i`` is the flow-weighted downhill slope), split
  among strictly-lower neighbours in proportion to ``(drop / length) ** routing_exponent``
  (weights normalised by the steepest slope first, so large exponents cannot underflow).
  Junctions without a lower neighbour (sinks) keep their water; runoff therefore accumulates
  downhill in the depressions of the street network.
* **Spill**: water deeper than ``spill_depth_m`` leaves the road network overland (into plots
  and open nallahs) and is booked as boundary outflow; ``null`` disables the cap.
* **Depth** ``= storage / (ponding_fraction * A_i)``; flooded ``= depth >= flood_depth_threshold_m``.

Each hour is applied in the order rain -> drains -> routing -> spill. The time loop runs over
hours only (numpy / scipy.sparse per step), so 61 000 hours x 1 000 junctions take seconds, and
the mass balance ``inflow = delta storage + drained + boundary outflow`` closes to rounding
(a non-finite budget raises :class:`SimulationError` instead of producing NaN "dry" labels).
Defaults were calibrated against the real Bellandur graph and the 2018-2024 Open-Meteo record
(see :mod:`src.hydrology.calibrate`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx
import numpy as np
from scipy.special import expit

from src.data_pipeline.graph_io import GraphArrays
from src.hydrology.drainage import DrainageNetwork, as_arrays, build_drainage_network
from src.hydrology.params import DEFAULTS, HydrologyParams, SimulationError
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_CHUNK_HOURS = 2048      # hours of rain converted to float64 at a time (bounds memory)

__all__ = [
    "DEFAULTS",
    "DrainageNetwork",
    "HydrologyParams",
    "MassBalance",
    "SimulationError",
    "SimulationResult",
    "UrbanDrainageSimulator",
    "build_drainage_network",
    "physics_flood_probability",
    "simulate_labels",
    "surcharge_factor",
    "tailwater_gate",
]


# --------------------------------------------------------------------------- dynamic forcing


def _window_excess(
    series: np.ndarray, window_h: int, threshold: float, ramp: float, history: np.ndarray | None
) -> np.ndarray:
    """``clip((trailing window sum - threshold) / ramp, 0, 1)`` along axis 0 of ``[T, K]``, history-aware."""
    lead = window_h - 1
    previous = np.zeros((lead, series.shape[1]), dtype=np.float64)
    if history is not None and lead:
        hist = np.asarray(history, dtype=np.float64)
        hist = hist[:, None] if hist.ndim == 1 else hist
        if hist.ndim != 2 or hist.shape[1] != series.shape[1]:
            raise ValueError(f"history shape {hist.shape} does not match loading {series.shape}")
        tail = hist[-lead:]
        previous[lead - tail.shape[0]:] = tail
    extended = np.concatenate([previous, series], axis=0)
    csum = np.concatenate([np.zeros((1, series.shape[1])), np.cumsum(extended, axis=0)], axis=0)
    rows = np.arange(series.shape[0])
    window = np.maximum(csum[rows + lead + 1] - csum[rows], 0.0)
    return np.clip((window - threshold) / ramp, 0.0, 1.0)


def _as_2d(values: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [T] or [T, N], got shape {np.shape(values)}")
    return arr, False


def surcharge_factor(
    loading_mm: np.ndarray, params: HydrologyParams, history_mm: np.ndarray | None = None
) -> np.ndarray:
    """Drain capacity multiplier from the trailing ``surcharge_window_h``-hour sum of a loading series.

    ``loading_mm`` is hourly rain loading, ``[T]`` (one basin value per hour) or ``[T, N]`` (per
    junction, e.g. contributing-area rain). ``history_mm`` holds the hours just before the
    block (``[H]`` / ``[H, N]``; missing hours count as dry), so a long record can be processed
    in chunks. Returns an array of the same shape as ``loading_mm`` in ``[surcharge_min_factor, 1]``
    (without the tailwater gate; see :func:`tailwater_gate`).
    """
    load, squeeze = _as_2d(loading_mm, "loading_mm")
    excess = _window_excess(load, params.surcharge_window_h, params.surcharge_threshold_mm,
                            params.surcharge_ramp_mm, history_mm)
    factor = 1.0 - (1.0 - params.surcharge_min_factor) * excess
    return factor[:, 0] if squeeze else factor


def tailwater_gate(
    basin_mm: np.ndarray, params: HydrologyParams, history_mm: np.ndarray | None = None
) -> np.ndarray:
    """Share ``[T]`` in [0, 1] of the local surcharge that can act, from the basin-mean rain loading.

    Inlets are throttled by backwater, which needs an elevated tailwater in the trunk drains /
    lake: the gate is 0 while the basin-mean rain over ``surcharge_window_h`` hours is below
    ``tailwater_threshold_mm`` and opens linearly over ``tailwater_ramp_mm``. It decides *when*
    backwater is possible; the local loading decides *where* it happens.
    """
    basin, _ = _as_2d(np.asarray(basin_mm, dtype=np.float64).reshape(-1), "basin_mm")
    gate = _window_excess(basin, params.surcharge_window_h, params.tailwater_threshold_mm,
                          params.tailwater_ramp_mm, history_mm)
    return gate[:, 0]


def _clean_block(block: np.ndarray) -> tuple[np.ndarray, int, int]:
    """float64 copy of a rain block with non-finite / negative values set to 0, plus their counts."""
    out = np.array(block, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    n_bad = int(bad.sum())
    if n_bad:
        out[bad] = 0.0
    negative = out < 0
    n_negative = int(negative.sum())
    if n_negative:
        out[negative] = 0.0
    return out, n_bad, n_negative


def _validate_rain(rain: Any, n_nodes: int) -> np.ndarray:
    try:
        arr = np.asarray(rain)
        if arr.dtype.kind not in "biuf":
            arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rain must be a numeric array [T, N]: {exc}") from exc
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"rain must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2 or arr.shape[1] != n_nodes:
        raise ValueError(f"rain must have shape [T, {n_nodes}] (hours x junctions), got {arr.shape}")
    return arr


def _validate_initial(initial: Any, n_nodes: int) -> np.ndarray:
    if initial is None:
        return np.zeros(n_nodes, dtype=np.float64)
    try:
        arr = np.array(initial, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"initial_storage_m3 must be numeric: {exc}") from exc
    if arr.shape != (n_nodes,):
        raise ValueError(f"initial_storage_m3 must have shape ({n_nodes},), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("initial_storage_m3 must be finite and non-negative")
    return arr


def _tail(history: np.ndarray, block: np.ndarray, keep: int) -> np.ndarray:
    """Last ``keep`` rows of ``history`` followed by ``block`` (a new array)."""
    if keep <= 0:
        return np.zeros((0, block.shape[1]), dtype=np.float64)
    return np.concatenate([history, block], axis=0)[-keep:].copy()


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class MassBalance:
    """Water budget of one run (m^3): ``inflow = (final - initial) + drained + boundary outflow``."""

    initial_storage_m3: float
    inflow_m3: float
    drained_m3: float
    boundary_outflow_m3: float
    final_storage_m3: float

    @property
    def residual_m3(self) -> float:
        change = self.final_storage_m3 - self.initial_storage_m3
        return self.inflow_m3 - change - self.drained_m3 - self.boundary_outflow_m3

    @property
    def relative_error(self) -> float:
        """``|residual| / max(inflow, initial storage)``; NaN when any budget term is not finite."""
        if not self.is_finite:
            return float("nan")
        scale = max(self.inflow_m3, self.initial_storage_m3)
        return abs(self.residual_m3) / scale if scale > 0 else 0.0

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.initial_storage_m3, self.inflow_m3, self.drained_m3,
                                             self.boundary_outflow_m3, self.final_storage_m3))


@dataclass(frozen=True)
class SimulationResult:
    """Simulator output: depth ``float32 [T, N]`` (m), flooded ``bool [T, N]``, end storage ``[N]`` (m^3)."""

    depth_m: np.ndarray
    flooded: np.ndarray
    final_storage_m3: np.ndarray
    mass_balance: MassBalance


@dataclass
class _Budget:
    """Mutable accumulator used inside one run (never exposed)."""

    inflow: float = 0.0
    drained: float = 0.0
    spilled: float = 0.0
    n_bad: int = 0
    n_negative: int = 0


@dataclass(frozen=True)
class _Carry:
    """Rain history carried across chunks (antecedent wetness, local and basin loading windows)."""

    antecedent: np.ndarray
    loading: np.ndarray
    basin: np.ndarray


# --------------------------------------------------------------------------- simulator


class UrbanDrainageSimulator:
    """Hour-by-hour storage routing over the road graph (see the module docstring for the physics)."""

    def __init__(
        self,
        graph: GraphArrays | nx.Graph,
        params: HydrologyParams,
        *,
        chunk_hours: int = DEFAULT_CHUNK_HOURS,
    ) -> None:
        if not isinstance(params, HydrologyParams):
            raise TypeError(f"params must be HydrologyParams, got {type(params).__name__}")
        if isinstance(chunk_hours, bool) or not isinstance(chunk_hours, int) or chunk_hours < 1:
            raise ValueError(f"chunk_hours must be a positive integer, got {chunk_hours!r}")
        self.graph = as_arrays(graph)
        self.params = params
        self.chunk_hours = chunk_hours
        self.network = build_drainage_network(self.graph, params)

    @property
    def num_nodes(self) -> int:
        return self.network.num_nodes

    def run(self, rain: np.ndarray, initial_storage_m3: np.ndarray | None = None) -> SimulationResult:
        """Simulate junction rain ``[T, N]`` (mm/h); NaN / negative rain counts as 0 (WARNING).

        Raises :class:`SimulationError` when the water budget or the depths are not finite.
        """
        rain_arr = _validate_rain(rain, self.num_nodes)
        storage = _validate_initial(initial_storage_m3, self.num_nodes)
        initial_total = float(storage.sum())
        n_hours = rain_arr.shape[0]
        depth = np.empty((n_hours, self.num_nodes), dtype=np.float32)
        budget = _Budget()
        lead = self.params.surcharge_window_h - 1
        carry = _Carry(np.zeros((self.params.antecedent_window_h, self.num_nodes)),
                       np.zeros((lead, self.num_nodes)), np.zeros((lead, 1)))
        for start in range(0, n_hours, self.chunk_hours):
            stop = min(start + self.chunk_hours, n_hours)
            block, n_bad, n_negative = _clean_block(rain_arr[start:stop])
            budget.n_bad += n_bad
            budget.n_negative += n_negative
            inflow = self._local_inflow(block, carry.antecedent)
            surcharge, carry = self._surcharge(block, carry)
            budget.inflow += float(inflow.sum())
            self._route(inflow, surcharge, storage, depth[start:stop], budget)
        self._warn_dirty(budget)
        balance = MassBalance(initial_total, budget.inflow, budget.drained, budget.spilled, float(storage.sum()))
        self._check_finite(balance, depth)
        flooded = depth >= np.float32(self.params.flood_depth_threshold_m)
        return SimulationResult(depth_m=depth, flooded=flooded, final_storage_m3=storage, mass_balance=balance)

    # ------------------------------------------------------------------ internals

    def _surcharge(self, block: np.ndarray, carry: _Carry) -> tuple[np.ndarray, _Carry]:
        """Drain capacity multiplier ``[c, N]`` of a block (local loading x tailwater gate) and the new carry."""
        p = self.params
        loading = self._surcharge_loading(block)
        basin = block.mean(axis=1, keepdims=True) if self.num_nodes else np.zeros((block.shape[0], 1))
        excess = _window_excess(loading, p.surcharge_window_h, p.surcharge_threshold_mm, p.surcharge_ramp_mm,
                                carry.loading)
        gate = tailwater_gate(basin, p, carry.basin)
        factor = 1.0 - (1.0 - p.surcharge_min_factor) * excess * gate[:, None]
        lead = p.surcharge_window_h - 1
        new_carry = _Carry(_tail(carry.antecedent, block, p.antecedent_window_h),
                           _tail(carry.loading, loading, lead), _tail(carry.basin, basin, lead))
        return factor, new_carry

    def _surcharge_loading(self, block: np.ndarray) -> np.ndarray:
        """Hourly rain loading of each drain reach ``[c, N]``: local (contributing-area) and tailwater (basin) terms."""
        weight = self.params.surcharge_local_weight
        loading = np.zeros_like(block)
        if weight > 0.0:
            loading += weight * np.asarray((self.network.contributing @ block.T).T)
        if weight < 1.0 and self.num_nodes:
            loading += (1.0 - weight) * block.mean(axis=1, keepdims=True)
        return loading

    def _local_inflow(self, block: np.ndarray, history: np.ndarray) -> np.ndarray:
        """Runoff volume per hour and junction ``[c, N]`` (m^3) with antecedent-dependent coefficients."""
        p = self.params
        window = history.shape[0]
        if window:
            extended = np.concatenate([history, block], axis=0)
            csum = np.concatenate([np.zeros((1, block.shape[1])), np.cumsum(extended, axis=0)], axis=0)
            rows = np.arange(block.shape[0])
            antecedent = np.maximum(csum[rows + window] - csum[rows], 0.0)
            wetness = np.minimum(antecedent / p.antecedent_saturation_mm, 1.0)
        else:
            wetness = np.zeros_like(block)
        coeff = p.runoff_coeff_dry + (p.runoff_coeff_wet - p.runoff_coeff_dry) * wetness
        excess = np.maximum(block - p.infiltration_mm_h, 0.0)
        return coeff * excess * (self.network.area_m2 / 1000.0)[None, :]

    def _route(
        self,
        inflow: np.ndarray,
        surcharge: np.ndarray,
        storage: np.ndarray,
        depth_out: np.ndarray,
        budget: _Budget,
    ) -> None:
        """Advance ``storage`` (in place, our own array) hour by hour and write depths into ``depth_out``.

        ``surcharge`` is the ``[c, N]`` drain capacity multiplier of every hour and junction.
        """
        net = self.network
        routes = net.routing.nnz > 0
        capacity = np.empty_like(storage)
        drained = np.empty_like(storage)
        moving = np.empty_like(storage)
        spill = np.empty_like(storage)
        inv_pond = 1.0 / net.pond_area_m2
        drained_total = np.zeros_like(storage)
        spilled_total = np.zeros_like(storage)
        for j in range(inflow.shape[0]):
            storage += inflow[j]
            np.multiply(net.drain_capacity_m3_h, surcharge[j], out=capacity)
            np.minimum(storage, capacity, out=drained)
            storage -= drained
            drained_total += drained
            if routes:
                np.multiply(net.outflow_fraction, storage, out=moving)
                storage -= moving
                storage += net.routing @ moving
            if net.spill_storage_m3 is not None:
                np.subtract(storage, net.spill_storage_m3, out=spill)
                np.maximum(spill, 0.0, out=spill)
                storage -= spill
                spilled_total += spill
            depth_out[j] = storage * inv_pond
        budget.drained += float(drained_total.sum())
        budget.spilled += float(spilled_total.sum())

    @staticmethod
    def _check_finite(balance: MassBalance, depth: np.ndarray) -> None:
        if not balance.is_finite or not math.isfinite(balance.relative_error):
            raise SimulationError(f"non-finite water budget {balance}; check the rain input and parameters")
        if depth.size and not np.isfinite(depth).all():
            raise SimulationError(f"{int((~np.isfinite(depth)).sum())} non-finite depths; check the parameters")

    @staticmethod
    def _warn_dirty(budget: _Budget) -> None:
        if budget.n_bad:
            LOGGER.warning("Rain input had %d non-finite (NaN/inf) values; treated as 0 mm", budget.n_bad)
        if budget.n_negative:
            LOGGER.warning("Rain input had %d negative values; treated as 0 mm", budget.n_negative)


# --------------------------------------------------------------------------- public helpers


def simulate_labels(
    graph: GraphArrays | nx.Graph, rain: np.ndarray, cfg: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Run the simulator with ``cfg['hydrology']`` and return ``(labels uint8 [T, N], depth float32 [T, N])``.

    Raises :class:`SimulationError` on a non-finite water budget (never returns NaN depths).
    """
    params = HydrologyParams.from_config(cfg)
    simulator = UrbanDrainageSimulator(graph, params)
    started = time.perf_counter()
    result = simulator.run(rain)
    labels = result.flooded.astype(np.uint8)
    elapsed = time.perf_counter() - started
    n_hours, n_nodes = labels.shape
    rate = float(labels.mean()) if labels.size else 0.0
    peak = float(result.depth_m.max()) if result.depth_m.size else 0.0
    LOGGER.info(
        "Simulated %d h x %d junctions in %.1f s: %.3f %% node-hours flooded, peak depth %.2f m, "
        "mass-balance error %.1e (surcharge local weight %.2f)",
        n_hours, n_nodes, elapsed, 100.0 * rate, peak, result.mass_balance.relative_error, params.surcharge_local_weight,
    )
    return labels, result.depth_m


def physics_flood_probability(depth_m: Any, threshold_m: float, softness_m: float = 0.05) -> np.ndarray:
    """Logistic flood probability ``sigmoid((depth - threshold) / softness)`` (float32, 0.5 at the threshold).

    Used by the app as the physics baseline; NaN depths count as dry (WARNING).
    """
    try:
        threshold = float(threshold_m)
        softness = float(softness_m)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold_m and softness_m must be numbers: {exc}") from exc
    if not math.isfinite(threshold):
        raise ValueError(f"threshold_m must be finite, got {threshold_m!r}")
    if not (math.isfinite(softness) and softness > 0):
        raise ValueError(f"softness_m must be finite and > 0, got {softness_m!r}")
    try:
        depth = np.array(depth_m, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"depth_m must be numeric: {exc}") from exc
    nan = np.isnan(depth)
    if nan.any():
        LOGGER.warning("physics_flood_probability: %d NaN depths treated as 0 m", int(nan.sum()))
        depth[nan] = 0.0
    return expit((depth - threshold) / softness).astype(np.float32)
