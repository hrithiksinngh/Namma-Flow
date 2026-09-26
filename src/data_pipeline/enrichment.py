"""Least-destructive re-enrichment of a road graph that never silently degrades real inputs.

Shared by stage 01 (a cached graph that lacks attributes or was enriched with other
settings), stage 04 (:func:`src.data_pipeline.dataset._prepare_graph`) and in-memory graphs
of diagnostics runs:

* when only DERIVED attributes are missing (``relative_elevation``, ``flow_accumulation``,
  ``is_sink``, edge ``grade``), the stored elevations are finite and the enrichment settings
  did not change, :func:`recompute_derived` recomputes them from the stored elevations - no
  DEM, waterway cache or network access, so real SRTM / OSM values are kept as they are;
* otherwise the caller runs the full elevation & drainage chain and :func:`check_downgrade`
  refuses the result when it would replace real elevations (local DEM / SRTM / Open-Meteo) or
  real OSM drains with a synthetic fallback (e.g. offline without the DEM cache), unless
  ``allow_synthetic`` is set - the R1-03 guard of stage 02, applied on every enrich path.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import networkx as nx

from src.data_pipeline.elevation_settings import ElevationSettings, synthetic_downgrades
from src.data_pipeline.graph_io import graph_to_arrays
from src.data_pipeline.terrain import flow_accumulation, relative_elevation
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DERIVED_NODE_ATTRS = ("relative_elevation", "flow_accumulation", "is_sink")
DERIVED_EDGE_ATTRS = ("grade",)
DERIVED_ATTRS = frozenset(DERIVED_NODE_ATTRS + DERIVED_EDGE_ATTRS)


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def can_recompute_derived(G: nx.DiGraph, missing: Sequence[str], drift: str | None) -> bool:
    """True when every missing attribute is derived from elevations that ``G`` already holds."""
    if drift or not missing or not set(missing) <= DERIVED_ATTRS or G.number_of_nodes() == 0:
        return False
    return all(_finite(data.get("elevation")) for _, data in G.nodes(data=True))


def recompute_derived(G: nx.DiGraph, cfg: Mapping[str, Any]) -> nx.DiGraph:
    """NEW graph with TPI, edge grades, flow accumulation and sinks recomputed from the stored elevations.

    Elevations, drain distances and every provenance graph attribute (``elevation_source``,
    ``drain_source``, ``enrichment_config_hash``, ...) are kept unchanged.
    """
    from src.data_pipeline.elevation import compute_edge_grades  # lazy: pulls in the raster stack

    settings = ElevationSettings.from_config(cfg)
    H = compute_edge_grades(G, cfg)
    arrays = graph_to_arrays(H)
    tpi = relative_elevation(arrays.lon, arrays.lat, arrays.node_attrs["elevation"], settings.tpi_radius_m)
    accumulation, sinks = flow_accumulation(H)
    for i, node in enumerate(arrays.node_ids):
        H.nodes[node].update(relative_elevation=float(tpi[i]), flow_accumulation=float(accumulation[i]),
                             is_sink=bool(sinks[i]))
    H.graph["derived_recomputed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOGGER.info("Recomputed relative elevation, grades, flow accumulation and sinks of %d junctions from the "
                "stored elevations (elevation source %s kept)", H.number_of_nodes(),
                H.graph.get("elevation_source", "unknown"))
    return H


def check_downgrade(
    before: nx.DiGraph | None,
    after: nx.DiGraph,
    *,
    allow_synthetic: bool,
    target: str,
    error: type[Exception],
    hint: str,
) -> list[str]:
    """Raise ``error`` when ``after`` replaces real elevation / drain inputs of ``before`` with synthetic ones.

    Returns the lost provenance (empty when nothing was lost). With ``allow_synthetic`` the
    downgrade is accepted with a WARNING.
    """
    lost = [] if before is None else synthetic_downgrades(before, after)
    if lost and not allow_synthetic:
        raise error(f"Refusing to replace {target}: the new enrichment would degrade real inputs to synthetic "
                    f"fallbacks ({'; '.join(lost)}; see the WARNINGs above). Nothing was saved. {hint}")
    if lost:
        LOGGER.warning("Replacing %s with synthetic fallbacks (%s) because allow_synthetic is set", target,
                       "; ".join(lost))
    return lost
