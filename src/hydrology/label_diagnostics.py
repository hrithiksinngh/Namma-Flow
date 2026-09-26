"""Degeneracy and rain-field diagnostics for simulated flood labels (calibration criteria f-h).

A label set that is "one basin-wide rain switch x a fixed per-junction map" can be predicted
without any graph, memory or junction-level rain, so it cannot test the project's hypothesis
(localised rainfall intensity + runoff accumulating downhill along the streets). The
functions here quantify how far the labels are from that degenerate structure:

* :func:`rank1_r2` — share of label variance explained by the best rank-1 outer product
  ``hour_state[t] x node_susceptibility[i]`` (the separable model). A basin switch times a fixed
  map scores ~1; labels whose flooded set depends on where the rain falls score much lower.
* :func:`label_jaccard` — overlap of two label sets (here: stochastic rain field vs the same
  areal rain broadcast uniformly). A value near 1 means the rain field is irrelevant.
* :func:`flooded_share_stats` — how many junctions flood in an hour that has any flooding
  (a basin switch floods a large, fixed share at once).
* :func:`rain_field_heterogeneity` — within-hour coefficient of variation of junction rain and
  the peak/mean multiplier over wet hours (does the field carry convective structure at all?).

Every function is pure numpy, never mutates its inputs and handles empty inputs by
returning ``None`` for undefined statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WET_AREAL_MM_H = 1.0          # "wet hour" for the rain-field statistics (areal rain >= 1 mm/h)
_CHUNK_ROWS = 4096            # rows converted to float64 at a time (bounds memory)


def _as_bool_2d(labels: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [hours, junctions] array, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def rank1_r2(labels: np.ndarray) -> float | None:
    """Variance share of binary labels ``[T, N]`` explained by the best rank-1 outer product.

    ``R2 = 1 - ||Y - s1 u1 v1^T||^2 / ||Y - mean(Y)||^2`` with ``(s1, u1, v1)`` the leading
    singular triple of the (uncentred) label matrix — the least-squares "hour state x node
    susceptibility" fit. All-dry hours contribute nothing to ``Y^T Y`` and are fitted exactly by
    ``u1 = 0``, so the SVD runs on the hours with any flooding only (cheap), while the total
    sum of squares uses every hour. ``None`` when the labels are constant.
    """
    flooded = _as_bool_2d(labels, "labels")
    n_elements = flooded.size
    if n_elements == 0:
        return None
    wet_rows = np.flatnonzero(flooded.any(axis=1))
    n_pos = float(flooded[wet_rows].sum()) if wet_rows.size else 0.0
    total_ss = n_pos - n_pos * n_pos / n_elements   # sum((y - mean)^2) for 0/1 data
    if total_ss <= 0.0:
        return None
    block = flooded[wet_rows].astype(np.float64)
    gram = block.T @ block if block.shape[0] > block.shape[1] else block @ block.T
    top = float(np.linalg.eigvalsh(gram)[-1])      # largest squared singular value
    residual = max(n_pos - top, 0.0)
    return float(1.0 - residual / total_ss)


def label_jaccard(first: np.ndarray, second: np.ndarray) -> float | None:
    """Jaccard index ``|A & B| / |A | B|`` of two flood-label sets of the same shape (``None`` if both empty)."""
    a = _as_bool_2d(first, "first")
    b = _as_bool_2d(second, "second")
    if a.shape != b.shape:
        raise ValueError(f"label sets differ in shape: {a.shape} vs {b.shape}")
    both = either = 0
    for lo in range(0, a.shape[0], _CHUNK_ROWS):
        sl = slice(lo, lo + _CHUNK_ROWS)
        both += int(np.count_nonzero(a[sl] & b[sl]))
        either += int(np.count_nonzero(a[sl] | b[sl]))
    return None if either == 0 else float(both / either)


@dataclass(frozen=True)
class FloodedShareStats:
    """Share of junctions flooded in hours with any flooding (``None`` when no hour floods)."""

    n_flood_hours: int
    median: float | None
    p90: float | None
    max: float | None


def flooded_share_stats(labels: np.ndarray) -> FloodedShareStats:
    """Median / p90 / max share of junctions flooded over the hours in which any junction floods."""
    flooded = _as_bool_2d(labels, "labels")
    if flooded.shape[1] == 0:
        return FloodedShareStats(0, None, None, None)
    share = flooded.sum(axis=1) / flooded.shape[1]
    share = share[share > 0]
    if share.size == 0:
        return FloodedShareStats(0, None, None, None)
    return FloodedShareStats(
        n_flood_hours=int(share.size),
        median=float(np.median(share)),
        p90=float(np.percentile(share, 90)),
        max=float(share.max()),
    )


@dataclass(frozen=True)
class RainFieldStats:
    """Spatial structure of junction rain over wet hours (areal >= ``wet_threshold_mm_h``)."""

    n_wet_hours: int
    cv_median: float | None
    cv_p90: float | None
    max_over_mean_median: float | None
    max_over_mean_p95: float | None
    wet_threshold_mm_h: float


def rain_field_heterogeneity(
    rain: np.ndarray, areal: np.ndarray, wet_threshold_mm_h: float = WET_AREAL_MM_H
) -> RainFieldStats:
    """Within-hour coefficient of variation and peak/mean multiplier of junction rain ``[T, N]``."""
    field = np.asarray(rain)
    areal_arr = np.nan_to_num(np.asarray(areal, dtype=np.float64).reshape(-1), nan=0.0)
    if field.ndim != 2 or field.shape[0] != areal_arr.size:
        raise ValueError(f"rain {field.shape} and areal ({areal_arr.size}) disagree")
    wet = np.flatnonzero(areal_arr >= wet_threshold_mm_h)
    if wet.size == 0 or field.shape[1] == 0:
        return RainFieldStats(0, None, None, None, None, float(wet_threshold_mm_h))
    cv_parts, ratio_parts = [], []
    for lo in range(0, wet.size, _CHUNK_ROWS):
        block = field[wet[lo : lo + _CHUNK_ROWS]].astype(np.float64)
        mean = block.mean(axis=1)
        ok = mean > 0
        cv_parts.append(block[ok].std(axis=1) / mean[ok])
        ratio_parts.append(block[ok].max(axis=1) / mean[ok])
    cv = np.concatenate(cv_parts)
    ratio = np.concatenate(ratio_parts)
    if cv.size == 0:
        return RainFieldStats(0, None, None, None, None, float(wet_threshold_mm_h))
    return RainFieldStats(
        n_wet_hours=int(cv.size),
        cv_median=float(np.median(cv)),
        cv_p90=float(np.percentile(cv, 90)),
        max_over_mean_median=float(np.median(ratio)),
        max_over_mean_p95=float(np.percentile(ratio, 95)),
        wet_threshold_mm_h=float(wet_threshold_mm_h),
    )


LOOKUP_WINDOW_H = 5
LOOKUP_BINS_MM = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
LOOKUP_SEASON_MONTHS = (5, 6, 7, 8, 9, 10, 11)


def separable_lookup_pr_auc(labels: np.ndarray, areal: np.ndarray, timestamps: pd.DatetimeIndex) -> dict:
    """PR-AUC of a graph-free, rain-field-free separable predictor on the last year of the record.

    ``score[t, i] = P(flooded | bin of trailing 5 h areal rain at t) x frequency of node i``,
    both fitted on the flood-season hours of every other year and evaluated on the last year's
    flood season. It is the reviewer's "basin switch x node map" baseline: high values mean the
    labels can be predicted without the road graph, memory or junction-level rain. Returns
    ``{"eval_year", "pr_auc", "n_pos"}`` (``None`` values when undefined).
    """
    from sklearn.metrics import average_precision_score  # lazy: only the calibration CLI needs sklearn

    flooded = _as_bool_2d(labels, "labels")
    areal_arr = np.nan_to_num(np.asarray(areal, dtype=np.float64).reshape(-1), nan=0.0)
    years = np.asarray(timestamps.year)
    empty = {"eval_year": None, "pr_auc": None, "n_pos": 0}
    if flooded.shape[0] != areal_arr.size or np.unique(years).size < 2 or flooded.shape[1] == 0:
        return empty
    season = np.isin(np.asarray(timestamps.month), LOOKUP_SEASON_MONTHS)
    eval_year = int(years.max())
    test, train = season & (years == eval_year), season & (years != eval_year)
    csum = np.concatenate([[0.0], np.cumsum(areal_arr)])
    idx = np.arange(1, areal_arr.size + 1)
    basin = csum[idx] - csum[np.maximum(idx - LOOKUP_WINDOW_H, 0)]
    bins = np.digitize(basin, LOOKUP_BINS_MM)
    hour_frac = flooded.mean(axis=1)
    table = np.zeros(len(LOOKUP_BINS_MM) + 1)
    for b in range(table.size):
        sel = train & (bins == b)
        table[b] = hour_frac[sel].mean() if sel.any() else 0.0
    node_freq = flooded[train].mean(axis=0) if train.any() else np.zeros(flooded.shape[1])
    y_true = flooded[test].reshape(-1)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == y_true.size:
        return {**empty, "eval_year": eval_year, "n_pos": n_pos}
    score = (table[bins[test]][:, None] * (node_freq[None, :] + 1e-6)).reshape(-1)
    return {"eval_year": eval_year, "pr_auc": float(average_precision_score(y_true, score)), "n_pos": n_pos}
