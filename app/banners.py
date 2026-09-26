"""Page-level wording of the dashboard that must stay true to the loaded data (pure helpers).

* :func:`header_caption` — the subtitle names the predictor and the graph actually loaded (the
  OpenStreetMap road graph, or a synthetic demo grid);
* :func:`synthetic_inputs` / :func:`synthetic_notice` — a warning banner whenever the street grid,
  the elevation or the rain record is synthetic demo data, so the numbers are not mistaken for
  Bengaluru results;
* :func:`may_replace_graph` — the setup page's offline build may fall back to the synthetic grid
  only when that cannot replace a real OpenStreetMap graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from components import Notice
from src.inference.scenarios import Scenario
from src.utils.logger import get_logger

LOGGER = get_logger("app.banners")
OSM_GRAPH_SOURCES = ("osm_place", "osm_bbox")

SYNTHETIC_NOTICE = ("**Synthetic demo data** — {what} synthetic, so the numbers have no physical meaning. Build the "
                    "real inputs with the pipeline (see the README) for Bengaluru results.")
GRAPH_PHRASES = {"osm_place": "the OpenStreetMap road graph", "osm_bbox": "the OpenStreetMap road graph",
                 "synthetic_grid": "a synthetic demo street grid (not Bengaluru's roads)"}
MODEL_PHRASES = {"gnn": "a spatio-temporal graph neural network",
                 "physics": "the hydrology simulator (physics baseline)"}


def header_caption(provenance: Mapping[str, Any], predictor_kind: str) -> str:
    """The page subtitle, true to the loaded graph and predictor."""
    graph = GRAPH_PHRASES.get(str(provenance.get("source")), "the road graph")
    model = MODEL_PHRASES.get(predictor_kind, "a flood model")
    return (f"Junction-level flood probabilities 12–48 h ahead for Bengaluru's Bellandur / Outer Ring Road corridor "
            f"— {model} over {graph}.")


def synthetic_inputs(provenance: Mapping[str, Any], scenario: Scenario) -> list[str]:
    """Which inputs of this view are synthetic demo data (street grid, elevation, replayed rain record)."""
    found = []
    if provenance.get("source") == "synthetic_grid":
        found.append("the street grid")
    if provenance.get("elevation_source") == "synthetic":
        found.append("the elevation")
    if scenario.source == "synthetic" or "includes synthetic hours" in scenario.description:
        found.append("the rain record")
    return found


def synthetic_notice(found: list[str]) -> Notice | None:
    """A warning banner when the dashboard runs on synthetic demo inputs."""
    if not found:
        return None
    what = found[0] if len(found) == 1 else ", ".join(found[:-1]) + " and " + found[-1]
    verb = "is" if len(found) == 1 else "are"
    return Notice("warning", SYNTHETIC_NOTICE.format(what=f"{what[:1].upper()}{what[1:]} {verb}"))


def may_replace_graph(path: Path) -> bool:
    """Whether the offline build may fall back to the synthetic grid: only when no graph file exists or
    the existing one is itself synthetic (a readable OSM graph is never silently replaced; an
    unreadable file is not protected by stage 01 either way)."""
    path = Path(path)
    if not path.is_file():
        return True
    try:
        from src.data_pipeline.graph_io import load_graph

        return str(load_graph(path).graph.get("source", "")) not in OSM_GRAPH_SOURCES
    except Exception as exc:  # noqa: BLE001 - unreadable: stage 01 rebuilds it anyway
        LOGGER.info("Existing graph %s is unreadable (%s); the offline build may replace it", path, exc)
        return True
