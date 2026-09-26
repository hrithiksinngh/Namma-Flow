"""How the dashboard reads and words the model's evaluation (``metrics.json`` / checkpoint metrics).

Pure helpers without Streamlit side effects, unit tested directly:

* the headline split: metrics format v2 reports its top-level metrics on the held-out TEST
  year(s) (``evaluation_split: "test"``) with model selection, calibration and the alert
  threshold tuned on the VALIDATION year(s); older files only have validation metrics, which
  are then worded as optimistic (the same hours chose the epoch, calibration and threshold);
* the strongest graph-free baseline (``strongest_baseline`` / ``baselines`` — gradient-boosted
  trees and logistic regression on the same per-junction-hour inputs) for the KPI delta;
* the GNN-vs-baselines table on both splits, the forecast-backtest table (the lead-0 physics
  run is the label generator: "teacher (= labels)"; the "perfect areal forecast (ensemble)"
  ceiling row; warning-window columns) and the model card;
* the areal-only skill (``areal_skill.json``, written by ``python -m src.training.areal_skill``):
  what the model reaches when only corridor-average rain is known.

The headline scores are "given the junction rain field": the model is fed the exact synthetic
junction rain field the teacher simulated the labels from, so they measure how well it emulates
the teacher. Forecasts and what-if storms know only corridor-average rain; their realistic
ceiling is the rain-field-ensemble score of ``areal_skill.json`` (and the backtest's ceiling row),
and the backtest adds rain-forecast error on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.inference.backtest import CEILING_KEY, TEACHER_LABEL, is_teacher, predictor_is_teacher

SPLITS = ("test", "validation")
GNN_LABEL = "Namma-Flow GNN"
BASELINES: dict[str, tuple[str, str]] = {          # name -> (short label, table label)
    "hist_gbdt": ("GBDT", "Gradient-boosted trees (no graph)"),
    "logreg": ("logistic", "Logistic regression (no graph)"),
}
METRIC_COLUMNS = ["PR-AUC", "ROC-AUC", "F2", "Precision", "Recall", "CSI", "Brier", "ECE", "Threshold"]
_METRIC_KEYS = ("pr_auc", "roc_auc", "f2", "precision", "recall", "csi", "brier", "ece", "threshold")
GIVEN_RAIN_NOTE = ("All scores are junction-hours against simulated labels **given the junction rain field**: "
                   "the model is fed the exact synthetic junction rain the labels were simulated from (observed "
                   "reanalysis rain downscaled by the teacher's own random field), so they measure how well it "
                   "emulates the teacher, not what a forecast can reach; the skill with only corridor-average rain "
                   "and the lead-time skill are below.")
AREAL_ROWS: dict[str, str] = {        # areal_skill.json row -> table label
    "exact_field": "Exact junction field (emulates the teacher; the headline)",
    "field_ensemble": "Rain-field ensemble (what forecasts / what-if storms use)",
    "field_ensemble_at_areal_threshold": "… the same ensemble, at the areal serving threshold",
    "single_other_field": "One other rain-field realisation",
    "uniform_areal": "Uniform corridor-average rain at every junction",
    "physics_other_field": "Physics teacher with one other field (single-field reference)",
    "physics_field_ensemble": "Physics teacher averaged over the same fields (estimate of the areal-only ceiling)",
}
AREAL_CAPTION = ("The labels were simulated from one random junction rain field that no forecast can know. "
                 "**Exact field** = the model is given that very field (it emulates the teacher). **Ensemble / "
                 "uniform** = the model is given only the corridor-average rain, as in forecast and what-if modes: "
                 "no rain forecast, however accurate, gives more. The physics rows are the teacher itself, driven by "
                 "one other field or averaged over the same fields (an estimate of the best possible skill with "
                 "corridor-average rain, which rises with the number of fields), scored against its own labels.")


def fmt_value(value: Any, digits: int = 3) -> str:
    """Metric value with ``digits`` significant figures (scientific below 0.01); ``n/a`` for None/NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number != 0 and abs(number) < 0.01:
        return f"{number:.{max(digits - 1, 1)}e}"
    return f"{number:.{digits}g}"


@dataclass(frozen=True)
class Kpi:
    """One ``st.metric`` tile."""

    label: str
    value: str
    delta: str | None = None
    help: str | None = None
    delta_color: str = "off"
    delta_arrow: str = "off"


# --------------------------------------------------------------------------- splits & wording


def evaluation_split(metrics: Mapping[str, Any] | None) -> str:
    """``"test"`` or ``"validation"``: the split the top-level (headline) metrics were computed on."""
    split = str((metrics or {}).get("evaluation_split") or "validation").strip().lower()
    return split if split in SPLITS else "validation"


def _years(value: Any) -> list[int]:
    """Sorted unique years from a list (or a single year); anything unparseable is skipped."""
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    years = set()
    for item in value:
        try:
            years.add(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(years)


def split_years(metrics: Mapping[str, Any] | None, split: str, dataset_cfg: Mapping[str, Any] | None = None
                ) -> list[int]:
    """Calendar years of ``split`` ("test" / "validation").

    Read from the metrics' dataset section (``dataset.years.<split>`` as training records it, or
    ``dataset.<test|val>_years``). The config's ``dataset`` section is a fallback only for metrics
    in the split-aware format (``evaluation_split`` present): older files predate the current
    split configuration, so their years are not guessed.
    """
    m = metrics or {}
    dataset = m.get("dataset") if isinstance(m.get("dataset"), Mapping) else {}
    by_split = dataset.get("years") if isinstance(dataset.get("years"), Mapping) else {}
    flat_key = "test_years" if split == "test" else "val_years"
    candidates = [by_split.get(split), dataset.get(flat_key)]
    if "evaluation_split" in m and isinstance(dataset_cfg, Mapping):
        candidates.append(dataset_cfg.get(flat_key))
    for candidate in candidates:
        if _years(candidate):
            return _years(candidate)
    return []


def split_label(split: str, years: Sequence[int]) -> str:
    """``test year 2024`` / ``validation years 2022, 2023`` / ``test years`` (unknown)."""
    name = "test" if split == "test" else "validation"
    if not years:
        return f"{name} years"
    return f"{name} year{'s' if len(years) > 1 else ''} {', '.join(str(y) for y in years)}"


def split_short(metrics: Mapping[str, Any] | None, split: str, dataset_cfg: Mapping[str, Any] | None = None) -> str:
    """Compact split tag for KPI labels and table cells: ``test 2024`` / ``val 2022``."""
    years = split_years(metrics, split, dataset_cfg)
    name = "test" if split == "test" else "val"
    return f"{name} {', '.join(str(y) for y in years)}" if years else name


def headline_wording(metrics: Mapping[str, Any] | None, dataset_cfg: Mapping[str, Any] | None = None) -> str:
    """Where the headline metrics come from, e.g. ``held-out test year 2024; model selection and
    calibration used validation year 2022``."""
    val = split_label("validation", split_years(metrics, "validation", dataset_cfg))
    if evaluation_split(metrics) == "test":
        test = split_label("test", split_years(metrics, "test", dataset_cfg))
        return f"held-out {test}; model selection and calibration used {val}"
    return (f"{val}, which also chose the epoch, the calibration and the alert threshold "
            "(no untouched test split — optimistic)")


# --------------------------------------------------------------------------- metrics per split / model


def _usable(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping) and value and value.get("status", "ok") == "ok":
        return value
    return None


def split_metrics(metrics: Mapping[str, Any] | None, split: str) -> Mapping[str, Any] | None:
    """The GNN's metrics on ``split`` (``metrics[split]``, or the headline when it is that split)."""
    m = metrics or {}
    stored = _usable(m.get(split))
    if stored is not None:
        return stored
    return m if evaluation_split(m) == split and m.get("pr_auc") is not None else None


def baseline_metrics(metrics: Mapping[str, Any] | None, name: str, split: str) -> Mapping[str, Any] | None:
    """Metrics of baseline ``name`` on ``split`` (``baselines[name][split]``; legacy ``baseline_logreg``)."""
    m = metrics or {}
    baselines = m.get("baselines")
    if isinstance(baselines, Mapping) and isinstance(baselines.get(name), Mapping):
        entry = baselines[name]
        return _usable(entry.get(split)) if entry.get("status", "ok") == "ok" else None
    if name == "logreg" and split == evaluation_split(m):
        return _usable(m.get("baseline_logreg"))
    return None


def baseline_names(metrics: Mapping[str, Any] | None) -> list[str]:
    """Known baselines first (strongest family first), then any other names in ``baselines``."""
    extra = (metrics or {}).get("baselines")
    others = [str(k) for k in extra if str(k) not in BASELINES] if isinstance(extra, Mapping) else []
    return [*BASELINES, *others]


def strongest_baseline(metrics: Mapping[str, Any] | None) -> tuple[str, float] | None:
    """``(name, PR-AUC)`` of the strongest baseline on the headline split."""
    m = metrics or {}
    stored = m.get("strongest_baseline")
    if isinstance(stored, Mapping) and stored.get("name") and _finite(stored.get("pr_auc")):
        return str(stored["name"]), float(stored["pr_auc"])
    split = evaluation_split(m)
    found = [(name, float(b["pr_auc"])) for name in baseline_names(m)
             if (b := baseline_metrics(m, name, split)) is not None and _finite(b.get("pr_auc"))]
    return max(found, key=lambda item: item[1]) if found else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def baseline_label(name: str, short: bool = False) -> str:
    labels = BASELINES.get(name, (name, f"{name} (no graph)"))
    return labels[0] if short else labels[1]


def _checkpoint_matches(report: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> bool:
    """Whether ``areal_skill.json`` was computed for the loaded checkpoint (same created / finalized stamps)."""
    ckpt = (report or {}).get("checkpoint")
    if not isinstance(ckpt, Mapping) or not describe:
        return False
    keys = [k for k in ("created_utc", "finalized_utc") if ckpt.get(k) and describe.get(k)]
    return bool(keys) and all(str(ckpt[k]) == str(describe[k]) for k in keys)


def _areal_pr_auc(report: Mapping[str, Any] | None, row: str) -> float | None:
    rows = (report or {}).get("rows")
    value = (rows.get(row) or {}).get("pr_auc") if isinstance(rows, Mapping) and isinstance(rows.get(row), Mapping) \
        else None
    return float(value) if _finite(value) else None


def _same_split(source: Mapping[str, Any], metrics: Mapping[str, Any] | None) -> bool:
    """Whether an areal-skill report / summary scored the headline metrics' split (the KPI's split tag).

    ``python -m src.training.areal_skill --split validation`` writes the same file, so a validation
    report must not appear under the headline's "test" tag. A source without a split (legacy) or
    no metrics to compare with is accepted.
    """
    split = str(source.get("split") or "").strip().lower()
    return not split or not metrics or split == evaluation_split(metrics)


def areal_skill_summary(report: Mapping[str, Any] | None, metrics: Mapping[str, Any] | None,
                        describe: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """PR-AUC with only corridor-average rain known, for the loaded model; None when unknown.

    From ``areal_skill.json`` when it is ``ok``, matches the loaded checkpoint and scored the
    headline split, else from the compact ``metrics["areal_skill"]`` summary training stores
    with the model's own metrics (same split rule).
    Keys: ``field_ensemble_pr_auc``, ``uniform_areal_pr_auc``, ``exact_field_pr_auc``, ``members``, ``source``.
    """
    if (isinstance(report, Mapping) and report.get("status") == "ok" and _checkpoint_matches(report, describe)
            and _same_split(report, metrics)):
        ensemble = _areal_pr_auc(report, "field_ensemble")
        if ensemble is not None:
            return {"field_ensemble_pr_auc": ensemble, "uniform_areal_pr_auc": _areal_pr_auc(report, "uniform_areal"),
                    "exact_field_pr_auc": _areal_pr_auc(report, "exact_field"), "members": report.get("members"),
                    "physics_field_ensemble_pr_auc": _areal_pr_auc(report, "physics_field_ensemble"),
                    "source": "areal_skill.json"}
    stored = (metrics or {}).get("areal_skill")
    if isinstance(stored, Mapping) and _finite(stored.get("field_ensemble_pr_auc")) and _same_split(stored, metrics):
        return {"field_ensemble_pr_auc": float(stored["field_ensemble_pr_auc"]),
                "uniform_areal_pr_auc": float(stored["uniform_areal_pr_auc"])
                if _finite(stored.get("uniform_areal_pr_auc")) else None,
                "exact_field_pr_auc": float(stored["exact_field_pr_auc"])
                if _finite(stored.get("exact_field_pr_auc")) else None,
                "members": stored.get("members"), "source": "metrics",
                "physics_field_ensemble_pr_auc": float(stored["physics_field_ensemble_pr_auc"])
                if _finite(stored.get("physics_field_ensemble_pr_auc")) else None}
    return None


def pr_auc_kpi(metrics: Mapping[str, Any] | None, predictor_kind: str,
               dataset_cfg: Mapping[str, Any] | None = None, *, exact_field: bool = True,
               areal: Mapping[str, Any] | None = None) -> Kpi:
    """The model-skill KPI of the current mode.

    Replays (``exact_field``) show the headline PR-AUC "given the junction rain field" with the
    delta vs the strongest graph-free baseline (comparable: the baselines see the same field).
    Forecast / what-if modes know only corridor-average rain, so they show the rain-field-ensemble
    PR-AUC of :func:`areal_skill_summary` ("given areal rain", no delta); without it, the
    headline with an explicit caveat.
    """
    if predictor_kind != "gnn":
        return _physics_kpi(exact_field, areal)
    m = metrics or {}
    value = m.get("pr_auc")
    if not _finite(value):
        return Kpi("Model PR-AUC", "n/a", help="No evaluation metrics found (artifacts/reports/metrics.json).")
    split = evaluation_split(m)
    tag = split_short(m, split, dataset_cfg)
    rate = m.get("pos_rate")
    base = f" A random ranking scores the flood rate ({fmt_value(rate)})." if rate is not None else ""
    if not exact_field:
        return _areal_kpi(float(value), tag, headline_wording(m, dataset_cfg), areal, base)
    strongest = strongest_baseline(m)
    delta = None if strongest is None else f"{float(value) - strongest[1]:+.2g} vs {baseline_label(strongest[0], True)}"
    versus = "" if strongest is None else (f" Delta: against the strongest graph-free baseline on the same inputs, "
                                           f"{baseline_label(strongest[0])} (PR-AUC {fmt_value(strongest[1])}).")
    help_text = (f"Average precision over junction-hours of the {headline_wording(m, dataset_cfg)}, given the junction "
                 "rain field: this replay feeds the model the exact synthetic junction rain the labels were simulated "
                 f"from, so the score measures emulation of the teacher.{versus}{base}")
    return Kpi(f"PR-AUC given junction rain · {tag}", fmt_value(value), delta, help_text,
               "normal" if delta else "off", "auto" if delta else "off")


def _physics_kpi(exact_field: bool, areal: Mapping[str, Any] | None) -> Kpi:
    """Skill tile of the physics baseline: the label generator in replays, a real predictor otherwise."""
    if exact_field:
        return Kpi("Model PR-AUC", "teacher", help="On a replay the physics simulator runs on the exact field that "
                   "generated the training labels, so it has no independent score (it would be 1.0 against its own "
                   "labels).")
    value = (areal or {}).get("physics_field_ensemble_pr_auc")
    members = (areal or {}).get("members")
    if not _finite(value):
        return Kpi("Physics PR-AUC given areal rain", "n/a", help="Averaged over rain fields the simulator is no longer "
                   "the label generator; its score with only corridor-average rain is in areal_skill.json "
                   "(python -m src.training.areal_skill).")
    return Kpi("Physics PR-AUC given areal rain", fmt_value(value), help=(
        f"The simulator averaged over {members or 'several'} rain fields, scored against the test-year labels. With "
        "only corridor-average rain it no longer knows the label field, so this is a genuine score (an estimate of "
        "the best possible with areal rain)."))


def _areal_kpi(headline: float, tag: str, wording: str, areal: Mapping[str, Any] | None, base: str) -> Kpi:
    """Forecast / what-if skill tile: the rain-field-ensemble PR-AUC, or the headline with a caveat."""
    if areal and _finite(areal.get("field_ensemble_pr_auc")):
        members = areal.get("members")
        uniform = areal.get("uniform_areal_pr_auc")
        extra = f" Broadcasting the areal rain uniformly scores {fmt_value(uniform)}." if _finite(uniform) else ""
        help_text = (f"Average precision over junction-hours of the {wording}, given only the corridor-average rain "
                     f"(as in this mode): the model averaged over {members or 'K'} random junction rain fields. With "
                     f"the exact junction field it reaches {fmt_value(headline)} (emulation of the teacher); no "
                     f"forecast can know that field.{extra} Rain-forecast error at 24-48 h lead lowers skill further "
                     f"(forecast backtest, About the model).{base}")
        return Kpi(f"PR-AUC given areal rain · {tag}", fmt_value(areal["field_ensemble_pr_auc"]), None, help_text)
    help_text = (f"Average precision over junction-hours of the {wording}, given the EXACT junction rain field the "
                 "labels were simulated from. This mode knows only corridor-average rain, so its real skill is much "
                 "lower; the areal-only score is not available (run python -m "
                 f"src.training.areal_skill).{base}")
    return Kpi(f"PR-AUC given junction rain · {tag}", fmt_value(headline), "not this mode's skill", help_text)


# --------------------------------------------------------------------------- tables


def _row(metrics: Mapping[str, Any]) -> list[str]:
    return [fmt_value(metrics.get(key)) for key in _METRIC_KEYS]


def comparison_table(metrics: Mapping[str, Any] | None, dataset_cfg: Mapping[str, Any] | None = None
                     ) -> pd.DataFrame:
    """GNN vs graph-free baselines on the test split (if any) and the validation split (formatted)."""
    columns = ["Model", "Split", *METRIC_COLUMNS]
    if not metrics:
        return pd.DataFrame(columns=columns)
    rows = []
    for split in SPLITS:
        tag = split_short(metrics, split, dataset_cfg)
        gnn = split_metrics(metrics, split)
        if gnn is not None:
            rows.append([GNN_LABEL, tag, *_row(gnn)])
        for name in baseline_names(metrics):
            baseline = baseline_metrics(metrics, name, split)
            if baseline is not None:
                rows.append([baseline_label(name), tag, *_row(baseline)])
    return pd.DataFrame(rows, columns=columns)


def reliability_table(metrics: Mapping[str, Any] | None) -> pd.DataFrame:
    """Non-empty reliability bins (mean predicted vs observed frequency) of the headline split."""
    curve = (metrics or {}).get("reliability") or {}
    keys = ("bin_centers", "mean_predicted", "observed_freq", "counts")
    if not isinstance(curve, Mapping) or not all(isinstance(curve.get(k), list) for k in keys):
        return pd.DataFrame(columns=["bin_center", "mean_predicted", "observed_freq", "count"])
    frame = pd.DataFrame({"bin_center": curve["bin_centers"], "mean_predicted": curve["mean_predicted"],
                          "observed_freq": curve["observed_freq"], "count": curve["counts"]})
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    return frame[frame["count"] > 0].reset_index(drop=True)


def areal_skill_table(report: Mapping[str, Any] | None) -> pd.DataFrame:
    """``areal_skill.json`` rows (formatted metrics); empty unless ``status == ok``."""
    columns = ["Junction rain given to the model", *METRIC_COLUMNS]
    rows = (report or {}).get("rows")
    if not report or report.get("status") != "ok" or not isinstance(rows, Mapping):
        return pd.DataFrame(columns=columns)
    names = [*[n for n in AREAL_ROWS if n in rows], *[str(n) for n in rows if n not in AREAL_ROWS]]
    members = report.get("members")
    out = []
    for name in names:
        metrics = rows.get(name)
        if not isinstance(metrics, Mapping):
            continue
        label = AREAL_ROWS.get(name, name)
        if name == "field_ensemble" and members:
            label = f"{label}, {members} fields"
        if metrics.get("status", "ok") != "ok":
            label = f"{label} — unavailable ({metrics.get('reason') or 'not computed'})"
        out.append([label, *_row(metrics)])
    return pd.DataFrame(out, columns=columns)


def areal_skill_title(report: Mapping[str, Any] | None) -> str:
    """``Skill when only corridor-average rain is known (test 2024)``."""
    r = report or {}
    split = "test" if str(r.get("split") or "test").lower() == "test" else "val"
    years = _years(r.get("years"))
    tag = f"{split} {', '.join(str(y) for y in years)}" if years else split
    return f"Skill when only corridor-average rain is known ({tag})"


def areal_skill_note(report: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> str | None:
    """Why the areal-skill table is missing or may not describe the loaded model (None when it does)."""
    if not report:
        return ("Not computed yet: run `python -m src.training.areal_skill` (training also writes it) to see what the "
                "model reaches with only corridor-average rain.")
    if report.get("status") != "ok":
        return f"Areal-only skill unavailable: {report.get('reason') or 'unknown reason'}."
    if describe and not _checkpoint_matches(report, describe):
        return ("These scores were computed for a different checkpoint than the one loaded now; re-run "
                "`python -m src.training.areal_skill`.")
    return None


def _backtest_row_name(key: str, lead: Mapping[str, Any]) -> str:
    if key == CEILING_KEY:
        return "Perfect areal forecast (ensemble)"
    return f"{lead.get('lead_hours', key)} h ahead"


def _field_label(lead: Mapping[str, Any]) -> str:
    field = lead.get("field")
    if not isinstance(field, Mapping):
        return "exact (label field)" if lead.get("lead_hours") == 0 else "one field"
    return "exact (label field)" if field.get("exact") else f"ensemble of {field.get('members', '?')}"


def _cell(metrics: Mapping[str, Any] | None, key: str, teacher: bool) -> str:
    """A backtest score cell; the label generator's own scores are shown as the teacher."""
    return TEACHER_LABEL if teacher else fmt_value((metrics or {}).get(key))


def backtest_table(report: Mapping[str, Any] | None) -> pd.DataFrame:
    """Forecast-skill backtest per row (``backtest.json``); empty unless ``status == ok``.

    Rows: 0 h ahead (observed rain, exact label field), the "perfect areal forecast (ensemble)"
    ceiling, and the 24 h / 48 h forecasts (rain-field ensembles). The lead-0 physics run is the
    label generator on the same rain, shown as "teacher (= labels)"; a physics-predictor report's
    own lead-0 scores are shown the same way. Warning-window columns score "the junction floods at
    some hour of the 6 h / 24 h block".
    """
    if not report or report.get("status") != "ok" or not isinstance(report.get("leads"), Mapping):
        return pd.DataFrame()
    rows = {}
    for name, lead in report["leads"].items():
        if not isinstance(lead, Mapping):
            continue
        rain, jh = lead.get("rain") or {}, lead.get("junction_hours") or {}
        physics = ((lead.get("physics_baseline") or {}).get("junction_hours") or {})
        teacher = predictor_is_teacher(lead)
        block6, block24 = lead.get("junction_block_6h") or {}, lead.get("junction_block_24h") or {}
        rows[_backtest_row_name(str(name), lead)] = {
            "Rain field": _field_label(lead), "Rain correlation": fmt_value(rain.get("correlation")),
            "PR-AUC": _cell(jh, "pr_auc", teacher), "ROC-AUC": _cell(jh, "roc_auc", teacher),
            "Recall": _cell(jh, "recall", teacher), "Precision": _cell(jh, "precision", teacher),
            "6 h window PR-AUC": _cell(block6, "pr_auc", teacher),
            "24 h window PR-AUC": _cell(block24, "pr_auc", teacher),
            "24 h window recall": _cell(block24, "recall", teacher),
            "24 h window precision": _cell(block24, "precision", teacher),
            "Physics PR-AUC": TEACHER_LABEL if is_teacher(lead) else fmt_value(physics.get("pr_auc"))}
    return pd.DataFrame.from_dict(rows, orient="index")


def backtest_note(report: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> str | None:
    """Which predictor / checkpoint the backtest scored, flagged when it is not the GNN loaded now."""
    run = (report or {}).get("predictor") or {}
    if not isinstance(run, Mapping) or not run:
        return None
    if run.get("kind") == "physics":
        return ("This backtest scored the **physics baseline**, not the loaded GNN (its 0 h row is the label generator "
                "itself); run `python -m src.inference.predict --backtest START END` for the GNN.")
    exact = run.get("threshold")
    served = sorted({round(float(e["threshold"]), 6) for e in ((report or {}).get("leads") or {}).values()
                     if isinstance(e, Mapping) and _finite(e.get("threshold"))} - ({round(float(exact), 6)}
                                                                                   if _finite(exact) else set()))
    thresholds = f"alert threshold {fmt_value(exact)}" + (
        f" (0 h row) / {', '.join(fmt_value(t) for t in served)} (rain-field-ensemble rows)" if served else "")
    text = f"Backtest predictor: {run.get('label', run.get('kind', '?'))}, epoch {run.get('epoch', '?')}, {thresholds}."
    current = (describe or {}).get("created_utc")
    if current and run.get("created_utc") and run.get("created_utc") != current:
        text += (" It was run with a different checkpoint than the one loaded now; re-run it with "
                 "`python -m src.inference.predict --backtest START END`.")
    return text


def stale_model_note(metrics: Mapping[str, Any] | None, current_hash: str | None = None,
                     trained_hash: str | None = None) -> str | None:
    """A note when the model's training data came from a different data configuration than now.

    ``trained_hash`` (the loaded checkpoint's ``dataset_config_hash``) wins over the metrics'
    dataset section; it is compared with ``current_hash`` (the current config's hash) when both
    are known, else the ``stale`` flag training recorded is used. Retraining alone would reuse the
    stale datasets, so the note says to rebuild them first.
    """
    dataset = (metrics or {}).get("dataset")
    dataset = dataset if isinstance(dataset, Mapping) else {}
    trained = trained_hash or dataset.get("dataset_config_hash")
    current = current_hash or dataset.get("current_dataset_config_hash")
    stale = (trained != current) if (trained and current_hash) else bool(dataset.get("stale"))
    if not stale:
        return None
    return (f"The model was trained on an older data configuration (dataset hash {trained} vs current {current}: "
            "rain field / hydrology / feature settings changed). Rebuild the datasets "
            "(python src/data_pipeline/04_dataset_builder.py --force) and then retrain "
            "(python src/training/train.py) for predictions that match the current pipeline.")


def _calibration(metrics: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    for source in (describe or {}, metrics or {}):
        value = source.get("calibration")
        if isinstance(value, Mapping) and _finite(value.get("slope")) and _finite(value.get("intercept")):
            return value
    return None


def calibration_label(metrics: Mapping[str, Any] | None, describe: Mapping[str, Any] | None) -> str:
    """Model-card text of the probability calibration."""
    cal = _calibration(metrics, describe)
    if cal is not None:
        method, slope, intercept = str(cal.get("method", "platt")), float(cal["slope"]), float(cal["intercept"])
        if method == "platt":
            return f"Platt: σ({slope:.3g} · logit {'+' if intercept >= 0 else '−'} {abs(intercept):.3g})"
        if method == "temperature":
            return f"Temperature: σ(logit / {1.0 / slope:.3g})"
        if slope == 1.0 and intercept == 0.0:
            return "None (σ(logit))"
    temperature = (describe or {}).get("temperature", (metrics or {}).get("temperature"))
    if _finite(temperature) and float(temperature) != 1.0:
        return f"Temperature: σ(logit / {float(temperature):.3g})"
    return "None (σ(logit))"


def calibration_caption(metrics: Mapping[str, Any] | None, describe: Mapping[str, Any] | None,
                        dataset_cfg: Mapping[str, Any] | None = None) -> str:
    """One sentence on how the probabilities (and the alert threshold) were calibrated."""
    val = split_label("validation", split_years(metrics, "validation", dataset_cfg))
    metric = str((metrics or {}).get("threshold_metric") or "F2").upper()
    label = calibration_label(metrics, describe)
    if label.startswith("Platt"):
        return (f"Probabilities are Platt-calibrated on the {val} ({label[7:]}); the alert threshold maximises "
                f"{metric} on those calibrated validation probabilities.")
    if label.startswith("Temperature"):
        return f"Probabilities are temperature-calibrated on the {val} ({label[13:]})."
    return "Probabilities are not calibrated (σ(logit)); treat them as scores, not frequencies."


def _card_thresholds(describe: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    """Both validation-chosen alert thresholds: exact field (replays) and rain-field ensemble (forecasts)."""
    metric = str(metrics.get("threshold_metric") or "f2").upper()
    exact = fmt_value(describe.get("threshold", metrics.get("threshold")))
    areal = describe.get("areal_threshold")
    if not _finite(areal):
        return f"{exact} (max {metric} on validation)"
    return f"{exact} exact field (replays) / {fmt_value(areal)} rain-field ensemble (forecasts, what-if); max {metric} on validation"


def model_card(metrics: Mapping[str, Any] | None, describe: Mapping[str, Any] | None,
               dataset_cfg: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """``(item, value)`` rows describing the loaded checkpoint (``FloodPredictor.describe()`` + metrics)."""
    m, d = metrics or {}, describe or {}
    model, training = m.get("model") or {}, m.get("training") or {}
    split = evaluation_split(m)
    return [
        ("Architecture", str(d.get("architecture") or model.get("architecture") or "n/a")),
        ("Parameters", f"{model['n_parameters']:,}" if isinstance(model.get("n_parameters"), int) else "n/a"),
        ("Epochs (best / run)", f"{m.get('best_epoch', 'n/a')} / {m.get('epochs_run', 'n/a')}"),
        ("Calibration", calibration_label(m, d)),
        ("Alert threshold", _card_thresholds(d, m)),
        ("Window", f"{d.get('seq_len', 'n/a')} h ({d.get('warmup_steps', 'n/a')} h warm-up, "
                   f"{d.get('lookback_hours', 'n/a')} h rain look-back)"),
        (f"ECE ({split_short(m, split, dataset_cfg)})", fmt_value(m.get("ece"))),
        ("Mean epoch time", f"{training['mean_epoch_time_s']:.0f} s" if isinstance(
            training.get("mean_epoch_time_s"), (int, float)) else "n/a"),
        ("Checkpoint", f"{d.get('checkpoint') or 'n/a'} (format v{d.get('format_version') or 1})"),
        ("Trained (UTC)", str(d.get("finalized_utc") or m.get("created_utc") or d.get("created_utc") or "n/a")),
    ]
