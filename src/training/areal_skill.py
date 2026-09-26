"""Areal-only skill (X6): how good is the model when only corridor-average rain is known?

Why (F1-01). The training labels come from the physics teacher driven by ONE stochastic
junction rain field per event, and the GNN is trained and evaluated with that exact field as
input. The headline test PR-AUC therefore measures *emulation of the teacher given its own
junction rain field*. A forecast, a design storm or a custom scenario only knows the
corridor-average (areal) rain; the junction field is unknown, so part of the label pattern is
unpredictable by construction. This module measures that on a dataset split (test by
default) with the published model. Every row holds the :func:`evaluate_predictions` metrics
at the checkpoint's alert threshold on the split's scored node-steps and labels:

* ``exact_field`` - the teacher's own junction field (the payload rain): the headline number;
* ``field_ensemble`` - mean calibrated probability over K independent field realisations of
  the same areal rain (seeds ``base + (1 + k) * stride``, never the label seed): the Monte
  Carlo marginalisation the forecast / design-storm modes serve (X5), i.e. what the model reaches
  given a PERFECT areal forecast (it improves with K and saturates around 32 members);
* ``single_other_field`` - one independent realisation (member 0);
* ``uniform_areal`` - the areal rain (node mean of the payload rain) broadcast to every junction;
* ``physics_other_field`` - the teacher itself re-simulated with member 0's field,
  ``p = physics_flood_probability(depth)`` at threshold 0.5, against its own labels: a
  single-field reference (even the teacher cannot reproduce its labels without the label field);
* ``physics_field_ensemble`` - the teacher averaged over the same K fields: a Monte-Carlo estimate
  of the best any predictor can do with only areal rain. It is noisy and biased LOW for small K
  (test 2024: 0.63 at K=8, 0.69 at K=32, 0.70 at K=64), so compare it with the GNN ensemble at the
  same K.
  It runs from 30 days before the split's first hour (dry start), which reproduces the
  full-record simulation exactly; the same run with the label field re-checks that every time
  (``teacher_label_mismatches``).

Every field is generated like stage 04 does - over the FULL weather record - and a check
reproduces the label field with the label seed; if it does not match the payload rain (the
weather record or the rain-field settings changed since the build) the report is
``unavailable``. Output: ``reports_dir/areal_skill.json`` (see :func:`compute_areal_skill`).
The trainer's finalisation computes it after the test evaluation and publishes it with
``best.pt``; the CLI recomputes it for the published model::

    python -m src.training.areal_skill [--config PATH] [--offline] [--members N]
                                       [--split test|validation] [--threads N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.data_pipeline.dataset import FloodSequenceDataset, make_loader
from src.data_pipeline.sequence_dataset import load_payload
from src.training import areal_fields as fields
from src.training.areal_fields import ArealSkillError
from src.training.calibration import apply_calibration, checkpoint_calibration
from src.training.checkpoint import (
    CheckpointError,
    checkpoint_paths,
    is_finalized,
    load_checkpoint,
    model_from_checkpoint,
    to_builtin,
    validate_checkpoint,
)
from src.training.evaluation import evaluate
from src.training.metrics import best_threshold, evaluate_predictions, ranking_metrics
from src.training.reports import portable_path
from src.training.run_state import fmt, now_utc
from src.training.splits import test_dataset_path, window_years
from src.utils.config import ConfigError, get_section, load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.runtime import atomic_write_text

LOGGER = get_logger(__name__)

__all__ = ["REPORT_NAME", "ROWS", "SPLITS", "ArealSkillError", "compute_areal_skill", "main", "summary_of",
           "unavailable_report", "write_areal_skill"]

REPORT_NAME = "areal_skill.json"
ROWS = ("exact_field", "field_ensemble", "single_other_field", "uniform_areal", "physics_other_field",
        "physics_field_ensemble")
# Extra row when the checkpoint carries an areal serving threshold: the field ensemble scored at the
# threshold the predictor uses for ensemble-averaged (forecast / design-storm) probabilities.
AREAL_THRESHOLD_ROW = "field_ensemble_at_areal_threshold"
SPLITS = ("test", "validation")
REPRODUCTION_TOLERANCE_MM_H = 1e-4
PHYSICS_THRESHOLD = 0.5
EVAL_BATCH = 1  # windows per forward pass: as fast as 4 on CPU, ~4x less peak memory (measured on the real graph)
NOTE = ("exact_field feeds the model the teacher's own stochastic junction rain field (the field that produced the "
        "labels), so it measures emulation of the teacher - the headline metric. field_ensemble (mean over "
        "independent field realisations), single_other_field and uniform_areal use only the corridor-average rain "
        "of the same hours, which is all a forecast, design storm or custom scenario can supply: even a perfect areal "
        "forecast is limited to these rows. physics_other_field re-runs the teacher itself with one other field "
        "(threshold p = 0.5): a single-field reference. physics_field_ensemble averages the teacher over the same "
        "fields: a Monte-Carlo estimate of the best possible skill with areal inputs (biased low for few fields).")


# --------------------------------------------------------------------------- inputs


def _payload_path(cfg: Mapping[str, Any], split: str) -> Path:
    path = test_dataset_path(cfg) if split == "test" else resolve_path(cfg, "val_dataset")
    if path is None or not Path(path).exists():
        raise ArealSkillError(f"No {split} dataset ({portable_path(path) if path else 'not configured'}); build it "
                              "with: python src/data_pipeline/04_dataset_builder.py")
    return Path(path)


def _load_split(cfg: Mapping[str, Any], split: str, payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is not None:
        return payload
    try:
        return load_payload(_payload_path(cfg, split))
    except (OSError, ValueError) as exc:  # DatasetError is a ValueError
        raise ArealSkillError(f"The {split} dataset is unreadable ({exc})") from exc


def _load_model_checkpoint(cfg: Mapping[str, Any], checkpoint: Any) -> dict[str, Any]:
    source = checkpoint_paths(cfg).best if checkpoint is None else checkpoint
    try:
        ckpt = load_checkpoint(source) if isinstance(source, (str, Path)) else validate_checkpoint(source)
    except (FileNotFoundError, CheckpointError) as exc:
        raise ArealSkillError(f"No usable published model ({exc})") from exc
    if not is_finalized(ckpt):
        raise ArealSkillError("The checkpoint is not a finalized (calibrated) model; train one first: "
                              "python src/training/train.py")
    return ckpt


def _check_fit(ckpt: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    """Raise when the model cannot score the split; return identity warnings (different build)."""
    keys = ("feature_names", "graph_signature", "scaler", "seq_len", "warmup_steps", "lookback_hours",
            "rolling_windows_h")
    differ = [key for key in keys if to_builtin(ckpt.get(key)) != to_builtin(payload.get(key))]
    if differ:
        raise ArealSkillError(f"The model was trained on other inputs than this dataset ({', '.join(differ)} "
                              "differ); retrain it or rebuild the datasets")
    notes = [f"{key} {ckpt.get(ckey)!r} (model) vs {payload.get(key)!r} (dataset)"
             for ckey, key in (("dataset_config_hash", "config_hash"), ("weather_fingerprint", "weather_fingerprint"))
             if ckpt.get(ckey) is not None and ckpt.get(ckey) != payload.get(key)]
    if notes:
        LOGGER.warning("Areal skill: the model was trained on another dataset build (%s); the report scores the "
                       "current dataset", "; ".join(notes))
    return notes


def _checkpoint_block(ckpt: Mapping[str, Any] | None) -> dict[str, Any]:
    ckpt = ckpt or {}
    return {key: ckpt.get(key) for key in ("run_id", "epoch", "created_utc", "finalized_utc", "published_id")}


# --------------------------------------------------------------------------- scoring


def _gnn_probs(model: torch.nn.Module, payload: Mapping[str, Any], rain: np.ndarray | None,
               calibration: Mapping[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels and calibrated probabilities on the split's scored node-steps with ``rain`` as input."""
    dataset = FloodSequenceDataset(payload if rain is None else {**payload, "rain": rain})
    y, logits, _ = evaluate(model, make_loader(dataset, batch_size, shuffle=False), "cpu")
    if not np.isfinite(logits).all():
        raise ArealSkillError(f"The model produced {int((~np.isfinite(logits)).sum())} non-finite logits")
    return y, apply_calibration(np.asarray(logits, dtype=np.float64), calibration)


def _row(y: np.ndarray, p: np.ndarray, threshold: float, options: Mapping[str, Any]) -> dict[str, Any]:
    metrics = evaluate_predictions(y, p, threshold, **options)
    return to_builtin({**metrics, "threshold": float(threshold), "mean_probability": float(np.mean(p))})


def _physics_row(cfg: Mapping[str, Any], payload: Mapping[str, Any], windows: tuple[np.ndarray, np.ndarray],
                 inputs: fields.WeatherInputs, span: slice, options: Mapping[str, Any]) -> dict[str, Any]:
    """The teacher re-simulated with another field, scored against its own labels (a reference ceiling).

    ``windows`` = (label field, other field) over ``span``; the label-field run checks that the
    re-simulated teacher reproduces the dataset labels (``teacher_label_mismatches``).
    """
    try:
        from src.hydrology.simulator import physics_flood_probability

        arrays, local = fields.physics_graph(cfg, payload), inputs.rows - span.start
        starts, seq, warm = (fields.as_numpy(payload["window_starts"], np.int64), int(payload["seq_len"]),
                             int(payload["warmup_steps"]))
        stored = fields.as_numpy(payload["labels"], np.uint8)
        mismatches = int((fields.teacher_run(arrays, cfg, windows[0], local)[0] != stored).sum())
        depth = fields.teacher_run(arrays, cfg, windows[1], local)[1]
        labels = fields.scored_values(stored, starts, seq, warm)
        prob = physics_flood_probability(fields.scored_values(depth, starts, seq, warm),
                                         float(payload["flood_threshold_m"]), fields.physics_softness(cfg))
    except (ImportError, ArealSkillError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
        LOGGER.warning("Areal skill: the physics reference row is unavailable (%s: %s)", type(exc).__name__, exc)
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    if mismatches:
        LOGGER.warning("Areal skill: with the label field the re-simulated teacher differs from the dataset labels on "
                       "%d of %d node-hours (hydrology settings changed since the build?); the physics row compares "
                       "another teacher", mismatches, stored.size)
    return {**_row(labels, prob, PHYSICS_THRESHOLD, options), "status": "ok", "spinup_h": fields.PHYSICS_SPINUP_H,
            "teacher_label_mismatches": mismatches}


def _physics_member_prob(cfg: Mapping[str, Any], payload: Mapping[str, Any], arrays: Any, field_window: np.ndarray,
                         inputs: fields.WeatherInputs, span: slice) -> tuple[np.ndarray, np.ndarray]:
    """Scored labels and the teacher's flood probability when it is driven by ``field_window``."""
    from src.hydrology.simulator import physics_flood_probability

    local = inputs.rows - span.start
    starts, seq, warm = (fields.as_numpy(payload["window_starts"], np.int64), int(payload["seq_len"]),
                         int(payload["warmup_steps"]))
    labels = fields.scored_values(fields.as_numpy(payload["labels"], np.uint8), starts, seq, warm)
    depth = fields.teacher_run(arrays, cfg, field_window, local)[1]
    prob = physics_flood_probability(fields.scored_values(depth, starts, seq, warm),
                                     float(payload["flood_threshold_m"]), fields.physics_softness(cfg))
    return labels, np.asarray(prob, dtype=np.float64)


class _PhysicsEnsemble:
    """Running mean of the teacher's probability over independent rain fields.

    With only corridor-average rain known, the probability that a junction floods is the share of
    possible rain fields in which the teacher floods it; averaging the teacher over K independent
    fields estimates exactly that, so this row is (up to Monte-Carlo error) the best any predictor
    can do with areal inputs - the ceiling the GNN's field ensemble is compared with.
    """

    def __init__(self, cfg: Mapping[str, Any], payload: Mapping[str, Any], inputs: fields.WeatherInputs,
                 span: slice) -> None:
        self.cfg, self.payload, self.inputs, self.span = cfg, payload, inputs, span
        self.total: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.members = 0
        self.error: str | None = None
        self.arrays = None
        try:
            self.arrays = fields.physics_graph(cfg, payload)
        except (ImportError, ArealSkillError, ValueError, RuntimeError, KeyError, OSError) as exc:
            LOGGER.warning("Areal skill: the physics ensemble row is unavailable (%s: %s)", type(exc).__name__, exc)
            self.error = f"{type(exc).__name__}: {exc}"

    def add(self, field_window: np.ndarray) -> None:
        if self.error is not None:
            return
        try:
            labels, prob = _physics_member_prob(self.cfg, self.payload, self.arrays, field_window, self.inputs,
                                                self.span)
        except (ImportError, ArealSkillError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
            LOGGER.warning("Areal skill: the physics ensemble row is unavailable (%s: %s)", type(exc).__name__, exc)
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self.labels = labels if self.labels is None else self.labels
        self.total = prob if self.total is None else self.total + prob
        self.members += 1

    def row(self, options: Mapping[str, Any]) -> dict[str, Any]:
        if self.error is not None or self.total is None or self.labels is None:
            return {"status": "unavailable", "reason": self.error or "no rain-field members"}
        return {**_row(self.labels, self.total / self.members, PHYSICS_THRESHOLD, options), "status": "ok",
                "members": self.members}


def _teacher_reproduced(row: Mapping[str, Any]) -> bool | None:
    mismatches = row.get("teacher_label_mismatches")
    return None if mismatches is None else mismatches == 0


def _uniform_rain(payload: Mapping[str, Any]) -> np.ndarray:
    rain = fields.as_numpy(payload["rain"], np.float32)
    return np.repeat(rain.mean(axis=1, keepdims=True), rain.shape[1], axis=1).astype(np.float32)


def _reproduction_check(cfg: Mapping[str, Any], payload: Mapping[str, Any], inputs: fields.WeatherInputs,
                        span: slice) -> tuple[float, np.ndarray]:
    """Max |label field regenerated here - payload rain| (mm/h) and the label field over ``span``.

    Raises when the field is not reproduced (the dataset is stale).
    """
    full = fields.junction_field(inputs, cfg, fields.label_seed(cfg))
    regenerated, window = full[inputs.rows], np.array(full[span])
    del full
    diff = float(np.max(np.abs(regenerated - fields.as_numpy(payload["rain"], np.float32)), initial=0.0))
    if diff > REPRODUCTION_TOLERANCE_MM_H:
        raise ArealSkillError(f"The label rain field cannot be reproduced from the current weather record and "
                              f"rainfall_field settings (max difference {diff:.3g} mm/h): the dataset is stale; "
                              "rebuild it with: python src/data_pipeline/04_dataset_builder.py")
    return diff, window


def _ensemble(cfg: Mapping[str, Any], payload: Mapping[str, Any], inputs: fields.WeatherInputs, model: Any,
              ckpt: Mapping[str, Any], members: int, options: Mapping[str, Any], batch_size: int,
              label_window: tuple[np.ndarray, slice] | None
              ) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Rows from independent rain-field realisations, plus the physics reference on member 0.

    The physics simulation runs first, before any model evaluation, so the process's peak
    memory is the larger of the two rather than their sum. ``label_window`` None skips the
    physics row. Returns (rows, per-member detail, labels of the scored node-steps, mean
    member probability per node-step).
    """
    calibration, threshold = checkpoint_calibration(ckpt), float(ckpt["threshold"])
    rows: dict[str, Any] = {}
    total, detail, y = None, [], None
    physics = None if label_window is None else _PhysicsEnsemble(cfg, payload, inputs, label_window[1])
    for k in range(members):
        seed = fields.field_member_seed(cfg, k)
        full = fields.junction_field(inputs, cfg, seed)
        rain = full[inputs.rows]
        other = np.array(full[label_window[1]]) if label_window is not None else None
        del full
        if other is not None:
            if k == 0:
                rows["physics_other_field"] = _physics_row(cfg, payload, (label_window[0], other), inputs,
                                                           label_window[1], options)
            physics.add(other)
            del other
        y_k, p_k = _gnn_probs(model, payload, rain, calibration, batch_size)
        if y is None:
            y, total = y_k, np.zeros(y_k.size, dtype=np.float64)
            rows["single_other_field"] = _row(y, p_k, threshold, options)
        elif not np.array_equal(y_k, y):
            raise ArealSkillError("internal error: the labels changed between rain-field members")
        total += p_k
        detail.append({"member": k, "seed": seed, "pr_auc": ranking_metrics(y, p_k)["pr_auc"]})
        LOGGER.info("Areal skill: rain-field member %d/%d (seed %d) PR-AUC %s", k + 1, members, seed,
                    fmt(detail[-1]["pr_auc"]))
    mean_p = total / members
    rows["field_ensemble"] = _row(y, mean_p, threshold, options)
    if physics is not None:
        rows["physics_field_ensemble"] = physics.row(options)
    return rows, detail, y, mean_p


def _areal_rows(model: Any, payload: Mapping[str, Any], ckpt: Mapping[str, Any], y: np.ndarray,
                options: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    """The exact-field (teacher's own field) and uniform-areal rows."""
    calibration, threshold = checkpoint_calibration(ckpt), float(ckpt["threshold"])
    rows = {}
    for name, rain in (("exact_field", None), ("uniform_areal", _uniform_rain(payload))):
        y_row, p_row = _gnn_probs(model, payload, rain, calibration, batch_size)
        if not np.array_equal(y_row, y):
            raise ArealSkillError("internal error: the labels changed between rain inputs")
        rows[name] = _row(y, p_row, threshold, options)
    return rows


# --------------------------------------------------------------------------- public API


def _valid_threshold(value: Any) -> float | None:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return None
    return threshold if 0.0 < threshold < 1.0 else None


def areal_serving_threshold(cfg: Mapping[str, Any], checkpoint: Mapping[str, Any], payload: Mapping[str, Any], *,
                            members: int = 8, metric: str = "f2", batch_size: int = 1) -> dict[str, Any]:
    """Alert threshold for ensemble-averaged probabilities, chosen on ``payload`` (the VALIDATION split).

    The checkpoint's ``threshold`` is optimal for probabilities driven by the exact junction rain
    field. Forecast and design-storm predictions average over ``members`` independent fields, which
    spreads probability mass (a junction that floods in 3 of 8 fields gets p = 0.375 instead of 0 or
    1), so the F-score-optimal cut differs. This scores the field ensemble on validation exactly as
    :func:`compute_areal_skill` does and returns ``{"threshold", "score", "metric", "split", "members",
    "mean_probability", "pos_rate"}``. Raises :class:`ArealSkillError` / ``ValueError`` on bad input.
    """
    members, _ = _validate_request(members, "validation")
    ckpt = dict(checkpoint)
    inputs = fields.load_weather_inputs(cfg, payload)
    model = model_from_checkpoint(ckpt)
    _, _, y, mean_p = _ensemble(cfg, payload, inputs, model, ckpt, members, _options(cfg), batch_size, None)
    threshold, score = best_threshold(y, mean_p, metric=metric)
    LOGGER.info("Areal serving threshold (validation, %d rain-field members): %.4f (%s %.4f; exact-field threshold "
                "%.4f)", members, threshold, metric, score, float(ckpt["threshold"]))
    return to_builtin({"threshold": float(threshold), "score": float(score), "metric": str(metric),
                       "split": str(payload.get("split") or "validation"), "members": members,
                       "mean_probability": float(np.mean(mean_p)), "pos_rate": float(np.mean(y)) if y.size else None})


def unavailable_report(reason: str, *, split: str = "test", members: int = 8,
                       checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The report written when the evaluation cannot run (same schema, empty ``rows``)."""
    return to_builtin({"status": "unavailable", "reason": str(reason), "split": split, "years": None,
                       "members": int(members), "threshold": (checkpoint or {}).get("threshold"),
                       "checkpoint": _checkpoint_block(checkpoint), "rows": {}, "note": NOTE,
                       "created_utc": now_utc()})


def _options(cfg: Mapping[str, Any]) -> dict[str, Any]:
    t = get_section(cfg, "training", {"reliability_bins": 10, "threshold_metric": "f2"})
    return {"n_bins": int(t["reliability_bins"]), "threshold_metric": str(t["threshold_metric"]).strip().lower()}


def _validate_request(members: Any, split: Any) -> tuple[int, str]:
    if isinstance(members, bool) or not isinstance(members, (int, np.integer)) or members < 1:
        raise ValueError(f"members must be an integer >= 1, got {members!r}")
    name = str(split).strip().lower()
    if name not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    return int(members), name


def _compute(cfg: Mapping[str, Any], ckpt: dict[str, Any], payload: Mapping[str, Any], members: int, split: str,
             batch_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    notes = _check_fit(ckpt, payload)
    inputs = fields.load_weather_inputs(cfg, payload)
    span = fields.physics_span(inputs)
    diff, label_window = _reproduction_check(cfg, payload, inputs, span)
    model, options = model_from_checkpoint(ckpt), _options(cfg)
    rows, detail, y, mean_p = _ensemble(cfg, payload, inputs, model, ckpt, members, options, batch_size,
                                        (label_window, span))
    del label_window
    rows = {**rows, **_areal_rows(model, payload, ckpt, y, options, batch_size)}
    areal_threshold = _valid_threshold(ckpt.get("areal_threshold"))
    ordered = {name: rows[name] for name in ROWS}
    if areal_threshold is not None:
        ordered[AREAL_THRESHOLD_ROW] = _row(y, mean_p, areal_threshold, options)
    return to_builtin({
        "status": "ok", "split": split, "years": window_years(FloodSequenceDataset(payload)), "members": members,
        "threshold": float(ckpt["threshold"]), "checkpoint": _checkpoint_block(ckpt),
        "rows": ordered, "areal_threshold": areal_threshold,
        "areal_threshold_info": ckpt.get("areal_threshold_info"),
        "note": NOTE, "created_utc": now_utc(), "n": int(y.size), "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if y.size else None, "label_seed": fields.label_seed(cfg),
        "field_seed_offset": fields.FIELD_SEED_OFFSET, "field_seeds": [d["seed"] for d in detail],
        "member_pr_auc": [d["pr_auc"] for d in detail],
        "checks": {"label_field_max_abs_diff_mm_h": diff, "label_field_reproduced": True,
                   "weather_fingerprint_matches": inputs.fingerprint == payload.get("weather_fingerprint"),
                   "teacher_labels_reproduced": _teacher_reproduced(rows["physics_other_field"]),
                   "model_dataset_build_notes": notes},
        "runtime_s": round(time.perf_counter() - started, 2),
    })


def compute_areal_skill(cfg: Mapping[str, Any], checkpoint: Any = None, *, members: int = 8, split: str = "test",
                        payload: Mapping[str, Any] | None = None, batch_size: int | None = None) -> dict[str, Any]:
    """Skill of the model on ``split`` when only the areal rain is known (see the module docstring).

    ``checkpoint``: None (``paths.checkpoint_dir/best.pt``), a path or a loaded (finalized)
    checkpoint dict. ``payload``: the split's dataset payload when already in memory (else
    it is read from the configured path). ``batch_size``: windows per forward pass (default
    :data:`EVAL_BATCH`; results do not depend on it). Returns the report dict
    ``{"status": "ok" | "unavailable", "reason"?, "split", "years", "members", "threshold",
    "checkpoint": {"run_id", "epoch", "created_utc", "finalized_utc", "published_id"},
    "rows": {name: metrics}, "note", "created_utc", ...}``. Missing inputs (dataset, model,
    weather record, a stale dataset) give ``status: "unavailable"`` with a WARNING; invalid
    arguments raise ``ValueError``.
    """
    members, split = _validate_request(members, split)
    batch = EVAL_BATCH if batch_size is None else int(batch_size)
    if batch < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    ckpt: dict[str, Any] | None = None
    try:
        ckpt = _load_model_checkpoint(cfg, checkpoint)
        report = _compute(cfg, ckpt, _load_split(cfg, split, payload), members, split, batch)
    except ArealSkillError as exc:
        LOGGER.warning("Areal-only skill is unavailable: %s", exc)
        return unavailable_report(str(exc), split=split, members=members, checkpoint=ckpt)
    ens, exact = report["rows"]["field_ensemble"]["pr_auc"], report["rows"]["exact_field"]["pr_auc"]
    LOGGER.info("Areal-only skill (%s, %d members): PR-AUC exact field %s -> field ensemble %s (uniform %s) in %.1f s",
                split, members, fmt(exact), fmt(ens), fmt(report["rows"]["uniform_areal"]["pr_auc"]),
                report["runtime_s"])
    return report


def summary_of(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compact block stored as ``metrics["areal_skill"]``."""
    rows = report.get("rows") or {}

    def pr(name: str) -> Any:
        return (rows.get(name) or {}).get("pr_auc")

    at_areal = rows.get(AREAL_THRESHOLD_ROW) or {}
    summary = {"status": report.get("status"), "split": report.get("split"), "members": report.get("members"),
               "exact_field_pr_auc": pr("exact_field"), "field_ensemble_pr_auc": pr("field_ensemble"),
               "uniform_areal_pr_auc": pr("uniform_areal"), "areal_threshold": report.get("areal_threshold"),
               "physics_field_ensemble_pr_auc": pr("physics_field_ensemble"),
               "field_ensemble_f2_at_areal_threshold": at_areal.get("f2")}
    return {**summary, "reason": report["reason"]} if report.get("reason") else summary


def report_text(report: Mapping[str, Any]) -> str:
    """Strict JSON text of a report (non-finite numbers become null)."""
    return json.dumps(to_builtin(report), indent=2, allow_nan=False)


def write_areal_skill(cfg: Mapping[str, Any], report: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Atomically write ``report`` to ``path`` (default ``reports_dir/areal_skill.json``)."""
    target = Path(path) if path is not None else resolve_path(cfg, "reports_dir") / REPORT_NAME
    atomic_write_text(target, report_text(report))
    return target


# --------------------------------------------------------------------------- CLI


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("expected a number >= 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Namma-Flow: skill of the published model when only "
                                                 "corridor-average rain is known (writes areal_skill.json)")
    parser.add_argument("--config", default=None, help="config YAML (default: $NAMMA_FLOW_CONFIG or "
                                                       "config/config.yaml)")
    parser.add_argument("--offline", action="store_true", help="never touch the network (the weather record is "
                                                               "always read from the cache)")
    parser.add_argument("--members", type=_positive, default=None,
                        help="independent rain-field realisations (default: training.areal_skill.members)")
    parser.add_argument("--split", choices=SPLITS, default="test", help="dataset split to score (default: test)")
    parser.add_argument("--threads", type=_positive, default=None, help="torch CPU threads")
    return parser.parse_args(argv)


def format_report(report: Mapping[str, Any]) -> str:
    """Human-readable table of the rows."""
    years = ", ".join(str(y) for y in report.get("years") or []) or "?"
    lines = [f"Areal-only skill - {report['split']} split (years {years}), {report['members']} rain-field members, "
             f"alert threshold {report['threshold']:.4f} (physics row: p > 0.5)",
             f"  {'row':<24}{'PR-AUC':>9}{'ROC-AUC':>9}{'F2':>9}{'Recall':>9}{'Precision':>11}"]
    names = [*ROWS, *([AREAL_THRESHOLD_ROW] if AREAL_THRESHOLD_ROW in (report.get("rows") or {}) else [])]
    for name in names:
        row = report["rows"].get(name) or {}
        label = (f"field_ensemble @ {float(report['areal_threshold']):.3f}"
                 if name == AREAL_THRESHOLD_ROW and report.get("areal_threshold") is not None else name)
        cells = [row.get(k) for k in ("pr_auc", "roc_auc", "f2", "recall", "precision")]
        text = "".join(f"{('n/a' if v is None else f'{v:.4f}'):>{w}}" for v, w in zip(cells, (9, 9, 9, 9, 11)))
        lines.append(f"  {label:<24}{text}" if row.get("status", "ok") == "ok" else
                     f"  {label:<24}unavailable: {row.get('reason')}")
    lines.append(f"  runtime {report.get('runtime_s')} s; label field reproduced "
                 f"(max diff {report['checks']['label_field_max_abs_diff_mm_h']:.2g} mm/h)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; 0 when the report was written, 1 when it is unavailable or the inputs are invalid."""
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, overrides={"project": {"offline": True}} if args.offline else None)
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        members = args.members or int(get_section(cfg, "training", {"areal_skill": {"members": 8}})
                                      .get("areal_skill", {}).get("members", 8))
        report = compute_areal_skill(cfg, None, members=members, split=args.split)
        if report["status"] != "ok":
            print(f"ERROR: areal-only skill unavailable: {report['reason']} (nothing written)", file=sys.stderr)
            return 1
        path = write_areal_skill(cfg, report)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    print(f"  written to {portable_path(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
