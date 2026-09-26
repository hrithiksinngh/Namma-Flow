"""Rain-field ensembles: junction rain when only corridor-average rain is known (contract X5).

Training labels come from the physics teacher driven by ONE stochastic junction rain field per
rain event (``rainfall_field.seed`` + the event's start hour), and the GNN is trained on that
exact field. Historical replays reproduce it (:attr:`Scenario.exact_field`), but a forecast, a
design storm or a custom series resolves only the areal rain, so the junction pattern is
unknown. The principled answer is Monte-Carlo marginalisation: predict with ``K`` independent
field realisations and average the probabilities.

Member ``k`` (``k = 0 .. K-1``) of a run with seed offset ``o`` downscales with
``rainfall_field.seed = base_seed + (o + k) * inference.field_seed_stride``; offset 0 / member 0
is the training field's seed (today's single-draw behaviour), and an offset >= 1 never reuses it
(the backtest scores its forecast rows with offset 1 so no member shares the label field).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_pipeline.rain_field import RainFieldParams, downscale_rainfall
from src.inference.settings import MAX_FIELD_MEMBERS, InferenceSettings, whole
from src.utils.config import deep_merge
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

EXACT_NOTE = ("exact training field: junction rain is the rain field the training labels were simulated from (a "
              "replay of the teacher's own input)")
UNIFORM_NOTE = "uniform junction rain: corridor-average rain at every junction (the stochastic rain field is disabled)"


@dataclass(frozen=True)
class FieldPlan:
    """Which rain-field realisations a prediction averages over.

    ``seeds[k]`` is the ``rainfall_field.seed`` of member ``k``; ``exact`` is True when the run
    reproduces the training field (one member, the base seed); ``uniform`` when the configured
    field is uniform, so every realisation is identical and one member suffices.
    """

    members: int
    offset: int
    seeds: tuple[int, ...]
    exact: bool
    uniform: bool = False

    def metadata(self) -> dict[str, Any]:
        """Run metadata (``field_members``, ``field_seed_offset``, ``field_seeds``, ``exact_field``)."""
        return {"field_members": self.members, "field_seed_offset": self.offset, "field_seeds": list(self.seeds),
                "exact_field": self.exact}


def _is_uniform(params: RainFieldParams) -> bool:
    return not params.enabled or params.n_cells == 0 or params.background_fraction >= 1.0


def member_seed(base_seed: int, offset: int, k: int, stride: int) -> int:
    """``rainfall_field.seed`` of member ``k`` at seed offset ``offset``."""
    return int(base_seed) + (int(offset) + int(k)) * int(stride)


def field_plan(cfg: Mapping[str, Any], settings: InferenceSettings, exact_field: bool,
               field_members: int | None = None, field_seed_offset: int = 0) -> FieldPlan:
    """The members of one run: 1 (the training field) when ``exact_field``, else ``field_members``
    (default ``inference.field_members``) seeds starting at ``field_seed_offset``."""
    params = RainFieldParams.from_config(cfg)
    offset = whole(field_seed_offset, "field_seed_offset", 0)
    if exact_field:
        if offset:
            LOGGER.debug("exact_field scenario: field_seed_offset=%d ignored (the training field is used)", offset)
        return FieldPlan(1, 0, (params.seed,), exact=True, uniform=_is_uniform(params))
    wanted = settings.field_members if field_members is None else whole(field_members, "field_members", 1)
    if wanted > MAX_FIELD_MEMBERS:
        raise ValueError(f"field_members={wanted} exceeds the maximum of {MAX_FIELD_MEMBERS}")
    if _is_uniform(params):
        LOGGER.debug("The rain field is uniform; one member represents every realisation")
        return FieldPlan(1, offset, (member_seed(params.seed, offset, 0, settings.field_seed_stride),), exact=False,
                         uniform=True)
    seeds = tuple(member_seed(params.seed, offset, k, settings.field_seed_stride) for k in range(wanted))
    return FieldPlan(wanted, offset, seeds, exact=False)


def member_rain(cfg: Mapping[str, Any], plan: FieldPlan, areal_mm: np.ndarray, timestamps: pd.DatetimeIndex,
                lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Junction rain ``float32 [K, T, N]`` of every member of ``plan`` (member ``k`` uses ``plan.seeds[k]``)."""
    members = [downscale_rainfall(areal_mm, timestamps, lon, lat, deep_merge(cfg, {"rainfall_field": {"seed": seed}}))
               for seed in plan.seeds]
    return np.stack(members).astype(np.float32, copy=False)


def rain_field_note(plan: FieldPlan, design_storm: bool = False) -> str:
    """What the junction rain of a run represents (captions, CLI notes, exported metadata)."""
    if plan.uniform:
        return UNIFORM_NOTE
    if plan.exact:
        return EXACT_NOTE
    axis = (" drawn on the canonical storm-anchored time axis shared by every design storm (clock-independent)"
            if design_storm else "")
    if plan.members == 1:
        return (f"junction rain from one stochastic rain-field realisation{axis} (only corridor-average rain is known, "
                "so the junction pattern is uncertain)")
    return (f"junction rain averaged over {plan.members} stochastic rain-field realisations{axis} (only "
            "corridor-average rain is known: probabilities are the ensemble mean, ± is the spread across "
            "realisations)")


def passes_for(members: int, mc_samples: int) -> int:
    """GNN passes of a run: ``max(members, mc_samples)`` with MC dropout, else one per member."""
    return max(int(members), int(mc_samples)) if mc_samples > 0 else int(members)
