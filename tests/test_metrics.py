"""Tests for evaluation metrics (``src.training.metrics``) and probability calibration -
Platt and temperature scaling (``src.training.calibration``).

Reference values are either hand-computed confusion matrices or scikit-learn's own
implementations, so the fast vectorised code paths are checked against a trusted oracle.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

from src.training.calibration import (
    PLATT_INTERCEPT_BOUNDS,
    PLATT_SLOPE_BOUNDS,
    TEMPERATURE_BOUNDS,
    apply_calibration,
    apply_temperature,
    calibrated_logits,
    checkpoint_calibration,
    fit_calibration,
    fit_platt,
    fit_temperature,
    identity_calibration,
    temperature_of,
    validate_calibration,
)
from src.training.metrics import (
    DEFAULT_THRESHOLD_GRID,
    THRESHOLD_METRICS,
    best_threshold,
    binary_metrics,
    evaluate_predictions,
    fbeta_from_counts,
    ranking_metrics,
    reliability_curve,
)

pytestmark = pytest.mark.unit


def _toy():
    """8 samples: 3 positives; at threshold 0.5 -> tp=2, fp=1, fn=1, tn=4."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.4, 0.1, 0.05, 0.3])
    return y, p


def _imbalanced(n: int = 20_000, pos_rate: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(np.uint8)
    score = rng.normal(0, 1, n) + 2.0 * y
    return y, 1.0 / (1.0 + np.exp(-(score - 3.0)))


# --------------------------------------------------------------------------- binary metrics


def test_binary_metrics_hand_computed_confusion_matrix():
    y, p = _toy()
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["f2"] == pytest.approx(fbeta_score(y, p >= 0.5, beta=2))
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["csi"] == pytest.approx(2 / 4)
    assert m["pos_rate"] == pytest.approx(3 / 8)
    assert m["predicted_pos_rate"] == pytest.approx(3 / 8)
    assert m["n"] == 8 and m["threshold"] == 0.5


def test_binary_metrics_threshold_is_inclusive():
    m = binary_metrics([1, 0], [0.5, 0.49], 0.5)
    assert (m["tp"], m["fp"]) == (1, 0)


def test_binary_metrics_zero_division_is_zero_not_nan():
    m = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.9)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f2"] == 0.0 and m["csi"] == 0.0
    assert m["accuracy"] == 1.0


def test_binary_metrics_accepts_torch_bool_and_2d_inputs():
    y = torch.tensor([[True, False], [False, True]])
    p = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)


def test_binary_metrics_empty_input():
    m = binary_metrics([], [], 0.5)
    assert m["n"] == 0 and m["tp"] == 0 and m["accuracy"] == 0.0 and m["pos_rate"] == 0.0


def test_binary_metrics_results_are_json_builtin_types():
    y, p = _toy()
    m = binary_metrics(y, p, 0.5)
    assert all(isinstance(v, (int, float)) and not isinstance(v, np.generic) for v in m.values())
    json.dumps(m)


@pytest.mark.parametrize(
    "y, p, match",
    [
        ([1, 0], [0.5], "same number"),
        ([1, 2], [0.5, 0.5], "0 or 1"),
        ([1, np.nan], [0.5, 0.5], "0 or 1"),
        ([1, 0], [0.5, np.nan], "finite"),
        ([1, 0], [1.5, 0.2], r"\[0, 1\]"),
        ([1, 0], [-0.1, 0.2], r"\[0, 1\]"),
    ],
)
def test_input_validation_errors(y, p, match):
    with pytest.raises(ValueError, match=match):
        binary_metrics(y, p, 0.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "0.5", None, True])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        binary_metrics([1, 0], [0.4, 0.6], threshold)


def test_fbeta_from_counts_matches_definition():
    assert fbeta_from_counts(2, 1, 1, beta=1.0) == pytest.approx(2 / 3)
    assert fbeta_from_counts(0, 0, 0, beta=2.0) == 0.0
    with pytest.raises(ValueError):
        fbeta_from_counts(1, 1, 1, beta=0.0)


# --------------------------------------------------------------------------- ranking metrics


def test_ranking_metrics_match_sklearn():
    y, p = _imbalanced()
    m = ranking_metrics(y, p)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["n"] == y.size and m["n_pos"] == int(y.sum())
    assert m["log_loss"] > 0


@pytest.mark.parametrize("label", [0, 1])
def test_ranking_metrics_single_class_gives_none(label):
    m = ranking_metrics(np.full(10, label), np.linspace(0, 1, 10))
    assert m["pr_auc"] is None and m["roc_auc"] is None
    assert m["brier"] is not None


def test_ranking_metrics_empty():
    m = ranking_metrics([], [])
    assert m == {"n": 0, "n_pos": 0, "pr_auc": None, "roc_auc": None, "brier": None, "log_loss": None}


def test_ranking_metrics_perfect_ranking():
    m = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)


def test_log_loss_is_finite_for_confident_mistakes():
    m = ranking_metrics([1, 0], [0.0, 1.0])
    assert math.isfinite(m["log_loss"])


# --------------------------------------------------------------------------- best threshold


def test_best_threshold_finds_grid_optimum_like_brute_force():
    y, p = _imbalanced(5000)
    grid = np.linspace(0.01, 0.99, 99)
    thr, score = best_threshold(y, p, metric="f2", grid=grid)
    brute = [fbeta_score(y, p >= t, beta=2, zero_division=0) for t in grid]
    assert score == pytest.approx(max(brute))
    assert fbeta_score(y, p >= thr, beta=2) == pytest.approx(score)


@pytest.mark.parametrize("metric", THRESHOLD_METRICS)
def test_best_threshold_all_metrics_return_valid_pairs(metric):
    y, p = _imbalanced(3000)
    thr, score = best_threshold(y, p, metric=metric)
    assert 0.0 < thr < 1.0 and math.isfinite(score)


def test_best_threshold_separable_data_scores_one():
    thr, score = best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8], metric="f1")
    assert score == pytest.approx(1.0)
    assert 0.2 < thr <= 0.7


def test_best_threshold_prefers_middle_of_a_tied_plateau():
    # Every grid threshold in (0.2, 0.7] gives F1 = 1, the chosen one lies well inside.
    thr, _ = best_threshold([0, 1], [0.2, 0.7], metric="f1", grid=np.linspace(0.05, 0.95, 19))
    assert 0.35 <= thr <= 0.55


@pytest.mark.parametrize("label", [0, 1])
def test_best_threshold_single_class(label):
    assert best_threshold(np.full(5, label), np.linspace(0, 1, 5)) == (0.5, 0.0)


def test_best_threshold_rejects_unknown_metric_and_bad_grid():
    with pytest.raises(ValueError, match="metric"):
        best_threshold([0, 1], [0.2, 0.8], metric="accuracy_at_top")
    with pytest.raises(ValueError, match="grid"):
        best_threshold([0, 1], [0.2, 0.8], grid=[])
    with pytest.raises(ValueError, match="grid"):
        best_threshold([0, 1], [0.2, 0.8], grid=[0.5, 1.5])


def test_best_threshold_metric_is_case_insensitive():
    assert best_threshold([0, 1], [0.2, 0.8], metric="F2")[1] == pytest.approx(1.0)


def test_default_candidates_adapt_to_tiny_calibrated_probabilities():
    # Calibrated rare-event probabilities can all lie far below the fixed grid (observed on the real
    # datasets: temperature scaling put every flood near 1e-5). The optimum must still be found.
    rng = np.random.default_rng(5)
    y = (rng.random(50_000) < 0.001).astype(np.uint8)
    p = np.where(y == 1, rng.uniform(2e-6, 9e-6, y.size), rng.uniform(0.0, 1.9e-6, y.size))
    thr, score = best_threshold(y, p, metric="f2")
    assert score > 0.9 and thr < 1e-5
    grid_only, _ = best_threshold(y, p, metric="f2", grid=DEFAULT_THRESHOLD_GRID)
    assert best_threshold(y, p, metric="f2", grid=DEFAULT_THRESHOLD_GRID)[1] == 0.0 and grid_only >= 1e-4
    assert evaluate_predictions(y, p, thr)["best_threshold"]["score"] == pytest.approx(score)


def test_default_candidates_reach_the_exact_optimum():
    y, p = _imbalanced(5000)
    _, score = best_threshold(y, p, metric="f1")
    exact = max(fbeta_score(y, p >= t, beta=1) for t in np.unique(p[y == 1]))
    assert score == pytest.approx(exact)


def test_default_grid_covers_rare_event_probabilities():
    assert DEFAULT_THRESHOLD_GRID.min() < 1e-3 and DEFAULT_THRESHOLD_GRID.max() >= 0.99
    assert np.all(np.diff(DEFAULT_THRESHOLD_GRID) > 0)


def test_best_threshold_is_fast_on_millions_of_samples():
    y, p = _imbalanced(2_000_000, pos_rate=0.001)
    start = time.perf_counter()
    best_threshold(y, p)
    assert time.perf_counter() - start < 5.0


# --------------------------------------------------------------------------- reliability


def test_reliability_curve_hand_computed():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    curve = reliability_curve(y, p, n_bins=2)
    assert curve["counts"] == [3, 2]
    assert curve["bin_centers"] == [0.25, 0.75]
    assert curve["mean_predicted"][0] == pytest.approx((0.05 + 0.15 + 0.15) / 3)
    assert curve["observed_freq"] == [pytest.approx(1 / 3), pytest.approx(1.0)]
    expected_ece = (3 * abs(1 / 3 - 0.35 / 3) + 2 * abs(1.0 - 0.975)) / 5
    assert curve["ece"] == pytest.approx(expected_ece)
    assert curve["mce"] == pytest.approx(abs(1 / 3 - 0.35 / 3))


def test_reliability_curve_empty_bins_are_none_and_json_safe():
    curve = reliability_curve([0, 1], [0.01, 0.02], n_bins=10)
    assert curve["counts"][0] == 2 and curve["counts"][5] == 0
    assert curve["observed_freq"][5] is None and curve["mean_predicted"][5] is None
    json.dumps(curve)


def test_reliability_curve_perfectly_calibrated_data_has_small_ece():
    rng = np.random.default_rng(1)
    p = rng.random(200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert reliability_curve(y, p)["ece"] < 0.01


def test_reliability_curve_empty_and_bad_bins():
    assert reliability_curve([], [])["ece"] is None
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve([0, 1], [0.1, 0.9], n_bins=0)


# --------------------------------------------------------------------------- evaluate_predictions


def test_evaluate_predictions_merges_everything():
    y, p = _imbalanced(4000)
    out = evaluate_predictions(y, p, 0.3)
    for key in ("precision", "recall", "f1", "f2", "tp", "pr_auc", "roc_auc", "brier", "ece"):
        assert key in out
    assert out["reliability"]["ece"] == out["ece"]
    assert out["best_threshold"]["metric"] == "f2"
    assert out["threshold"] == 0.3
    json.dumps(out)


def test_evaluate_predictions_single_class():
    out = evaluate_predictions(np.zeros(20), np.linspace(0, 0.5, 20), 0.5)
    assert out["pr_auc"] is None and out["best_threshold"]["threshold"] == 0.5


# --------------------------------------------------------------------------- temperature scaling


def _calibration_data(true_t: float, n: int = 50_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(-1.0, 3.0, n)
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits / true_t))).astype(np.float32)
    return torch.tensor(logits, dtype=torch.float32), torch.tensor(labels)


@pytest.mark.parametrize("true_t", [0.5, 1.0, 2.5])
def test_fit_temperature_recovers_the_generating_temperature(true_t):
    logits, labels = _calibration_data(true_t)
    assert fit_temperature(logits, labels) == pytest.approx(true_t, rel=0.08)


def test_fit_temperature_improves_nll():
    logits, labels = _calibration_data(3.0)
    t = fit_temperature(logits, labels)
    nll = lambda temp: torch.nn.functional.binary_cross_entropy(apply_temperature(logits, temp), labels)  # noqa: E731
    assert nll(t) < nll(1.0)


def test_fit_temperature_accepts_numpy_and_2d():
    logits, labels = _calibration_data(2.0, n=4000)
    t = fit_temperature(logits.numpy().reshape(40, 100), labels.numpy().reshape(40, 100))
    assert 1.5 < t < 2.7


@pytest.mark.parametrize(
    "logits, labels",
    [
        ([], []),
        ([1.0, 2.0, -1.0], [1.0, 1.0, 1.0]),
        ([1.0, 2.0, -1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ([float("nan"), float("inf")], [0.0, 1.0]),
    ],
)
def test_fit_temperature_degenerate_inputs_return_one(logits, labels):
    assert fit_temperature(torch.tensor(logits), torch.tensor(labels)) == 1.0


def test_fit_temperature_drops_non_finite_logits():
    logits, labels = _calibration_data(2.0, n=5000)
    logits[:10] = float("nan")
    assert 1.6 < fit_temperature(logits, labels) < 2.5


def test_fit_temperature_is_bounded_for_separable_data():
    logits = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert fit_temperature(logits, labels) == pytest.approx(TEMPERATURE_BOUNDS[0])


def test_fit_temperature_is_bounded_for_random_labels():
    rng = np.random.default_rng(3)
    logits = torch.tensor(rng.normal(0, 20, 2000))
    labels = torch.tensor((rng.random(2000) < 0.5).astype(np.float32))
    t = fit_temperature(logits, labels)
    assert TEMPERATURE_BOUNDS[0] <= t <= TEMPERATURE_BOUNDS[1]
    assert t > 5.0


def test_fit_temperature_validates_inputs():
    with pytest.raises(ValueError, match="same number"):
        fit_temperature(torch.zeros(3), torch.zeros(4))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_temperature(torch.zeros(3), torch.tensor([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="max_iter"):
        fit_temperature(torch.zeros(3), torch.zeros(3), max_iter=0)


def test_apply_temperature():
    logits = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.allclose(apply_temperature(logits, 2.0), torch.sigmoid(logits / 2.0))
    assert torch.allclose(apply_temperature(np.array([0.0]), 1.0), torch.tensor([0.5], dtype=torch.float64))
    for bad in (0.0, -1.0, float("inf"), float("nan"), "2"):
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature(logits, bad)


# --------------------------------------------------------------------------- Platt scaling (R2-02)


def _shifted_data(slope: float, intercept: float, n: int = 200_000, seed: int = 0):
    """Logits of an over-confident, upward-shifted model: truth is sigmoid(slope * z + intercept)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, n)
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-(slope * logits + intercept)))).astype(np.float64)
    return logits, labels


@pytest.mark.parametrize("slope, intercept", [(0.8, -4.0), (1.5, -2.0), (0.3, 0.5)])
def test_fit_platt_recovers_the_generating_map(slope, intercept):
    logits, labels = _shifted_data(slope, intercept)
    fitted = fit_platt(logits, labels)
    assert fitted["method"] == "platt"
    assert fitted["slope"] == pytest.approx(slope, rel=0.05)
    assert fitted["intercept"] == pytest.approx(intercept, abs=0.08)


def test_platt_matches_the_base_rate_where_temperature_cannot():
    """The class-rebalanced model's constant logit offset needs an intercept (review R2-02)."""
    logits, labels = _shifted_data(0.8, -4.0)
    platt = apply_calibration(logits, fit_platt(logits, labels))
    temperature = apply_temperature(logits, fit_temperature(logits, labels)).numpy()
    base_rate = labels.mean()
    assert platt.mean() == pytest.approx(base_rate, rel=0.1)
    assert abs(temperature.mean() - base_rate) > 0.5 * base_rate  # temperature alone stays far off
    nll = lambda p: -np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))  # noqa: E731
    assert nll(platt) < nll(temperature)


def test_platt_keeps_the_ranking():
    logits, labels = _shifted_data(0.8, -4.0, n=20_000)
    calibrated = apply_calibration(logits, fit_platt(logits, labels))
    assert average_precision_score(labels, calibrated) == pytest.approx(average_precision_score(labels, logits))


@pytest.mark.parametrize(
    "logits, labels",
    [
        ([], []),
        ([1.0, 2.0, -1.0], [1.0, 1.0, 1.0]),
        ([1.0, 2.0, -1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ([float("nan"), float("inf")], [0.0, 1.0]),
    ],
)
def test_fit_platt_degenerate_inputs_give_the_identity(logits, labels):
    assert fit_platt(np.array(logits), np.array(labels)) == {"method": "platt", "slope": 1.0, "intercept": 0.0}


def test_fit_platt_is_bounded_and_refits_the_intercept():
    logits = np.array([-1.0, -0.5, 0.5, 1.0] * 50)
    labels = np.array([0.0, 0.0, 1.0, 1.0] * 50)
    fitted = fit_platt(logits, labels)  # separable: the optimum slope is unbounded
    assert fitted["slope"] == pytest.approx(PLATT_SLOPE_BOUNDS[1])
    assert PLATT_INTERCEPT_BOUNDS[0] <= fitted["intercept"] <= PLATT_INTERCEPT_BOUNDS[1]
    assert abs(fitted["intercept"]) < 1.0  # re-fitted around the symmetric decision boundary
    rare = fit_platt(np.zeros(1000) + np.r_[1e-3, np.zeros(999)], np.r_[1.0, np.zeros(999)])
    assert PLATT_INTERCEPT_BOUNDS[0] <= rare["intercept"] <= PLATT_INTERCEPT_BOUNDS[1]


def _platt_nll(z: np.ndarray, y: np.ndarray, slope, intercept) -> np.ndarray:
    scores = np.multiply.outer(np.asarray(slope, dtype=np.float64), z) + np.asarray(intercept)[..., None]
    return np.mean(np.logaddexp(0.0, scores) - y * scores, axis=-1)


def _box_case(name: str, n: int = 20_000, seed: int = 3):
    """F2-01 reviewer cases whose unconstrained optimum has |intercept| > 20."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(np.float64)
    z = {"under_dispersed": rng.normal(-1.5, 0.08, n) + 0.12 * y, "shift_up": rng.normal(25.0, 1.0, n) + y,
         "shift_down": rng.normal(-30.0, 1.0, n) + 3.0 * y}[name]
    return z, y


@pytest.mark.parametrize("name", ["under_dispersed", "shift_up", "shift_down"])
def test_fit_platt_returns_the_box_constrained_optimum(name):
    """F2-01: clipping the intercept of an unconstrained fit left the slope wrong (mean p 0.0008 vs 0.03) or fell
    back to the identity; the joint L-BFGS-B fit must be the NLL minimum inside the bounds."""
    z, y = _box_case(name)
    fitted = fit_platt(z, y)
    slope, intercept = fitted["slope"], fitted["intercept"]
    assert PLATT_SLOPE_BOUNDS[0] <= slope <= PLATT_SLOPE_BOUNDS[1]
    assert PLATT_INTERCEPT_BOUNDS[0] <= intercept <= PLATT_INTERCEPT_BOUNDS[1]
    assert abs(intercept) == pytest.approx(20.0)  # the intercept bound is active in all three cases
    nll = float(_platt_nll(z, y, slope, intercept))
    slopes, intercepts = np.meshgrid(np.linspace(*PLATT_SLOPE_BOUNDS, 61), np.linspace(*PLATT_INTERCEPT_BOUNDS, 61))
    grid = _platt_nll(z, y, slopes.ravel(), intercepts.ravel())
    assert nll <= grid.min() + 1e-9
    # KKT: the free slope has zero gradient; the active intercept bound blocks a descent direction
    scores = slope * z + intercept
    residual = 1.0 / (1.0 + np.exp(-scores)) - y
    grad_slope, grad_intercept = np.mean(residual * z), np.mean(residual)
    assert abs(grad_slope) < 1e-6
    assert grad_intercept * np.sign(intercept) <= 1e-9
    # slope strictly inside its bounds: the fitted probabilities track the base rate
    assert PLATT_SLOPE_BOUNDS[0] < slope < PLATT_SLOPE_BOUNDS[1]
    mean_p = float(np.mean(1.0 / (1.0 + np.exp(-scores))))
    assert mean_p == pytest.approx(y.mean(), rel=0.1)
    assert nll < float(_platt_nll(z, y, 1.0, 0.0))  # never the uncalibrated identity for these logits


def test_fit_platt_matches_scipy_from_another_start():
    from scipy.optimize import minimize

    z, y = _box_case("under_dispersed", n=50_000, seed=7)
    fitted = fit_platt(z, y)
    reference = minimize(lambda x: float(_platt_nll(z, y, x[0], x[1])), np.array([10.0, 5.0]), method="L-BFGS-B",
                         bounds=[PLATT_SLOPE_BOUNDS, PLATT_INTERCEPT_BOUNDS], options={"ftol": 1e-15})
    assert float(_platt_nll(z, y, fitted["slope"], fitted["intercept"])) <= reference.fun + 1e-6


def test_fit_platt_falls_back_to_the_best_safety_net_fit(monkeypatch, caplog):
    """F2-01: when the optimiser fails, the identity / intercept-only / near-constant fits are compared."""
    import logging

    from src.training import calibration

    def broken(*args, **kwargs):
        raise ValueError("solver exploded")

    monkeypatch.setattr(calibration, "minimize", broken)
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    try:
        z, y = _box_case("shift_down")
        fitted = fit_platt(z, y)
    finally:
        logger.removeHandler(caplog.handler)
    # every optimiser call failed: the near-constant logit(base rate) fit beats the identity (p ~ 1e-13)
    assert fitted["slope"] == pytest.approx(PLATT_SLOPE_BOUNDS[0])
    assert 1.0 / (1.0 + np.exp(-fitted["intercept"])) == pytest.approx(y.mean(), rel=0.2)
    text = caplog.text
    assert "did not converge" in text and "near-constant fit beats" in text


def test_fit_platt_drops_non_finite_logits_and_validates_inputs():
    logits, labels = _shifted_data(0.8, -4.0, n=20_000)
    logits[:10] = np.nan
    assert fit_platt(logits, labels)["slope"] == pytest.approx(0.8, rel=0.15)
    with pytest.raises(ValueError, match="same number"):
        fit_platt(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="max_iter"):
        fit_platt(np.zeros(3), np.zeros(3), max_iter=0)


def test_fit_calibration_methods():
    logits, labels = _shifted_data(0.5, 0.0, n=20_000)
    assert fit_calibration(logits, labels, "platt")["method"] == "platt"
    temperature = fit_calibration(logits, labels, "temperature")
    assert temperature["method"] == "temperature" and temperature["intercept"] == 0.0
    assert temperature["slope"] == pytest.approx(1.0 / fit_temperature(logits, labels))
    assert fit_calibration(logits, labels, "NONE") == identity_calibration("none")
    with pytest.raises(ValueError, match="Unknown calibration"):
        fit_calibration(logits, labels, "isotonic")


def test_apply_calibration_numpy_torch_and_dtypes():
    calibration = {"method": "platt", "slope": 0.5, "intercept": -1.0}
    z = np.array([-4.0, 0.0, 2.0, 50.0, -800.0])
    expected = 1.0 / (1.0 + np.exp(-(0.5 * z - 1.0)))
    out = apply_calibration(z, calibration)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    assert np.isfinite(out).all() and out[-1] == pytest.approx(0.0, abs=1e-12)  # no overflow warning
    assert apply_calibration(z.astype(np.float32), calibration).dtype == np.float32
    assert apply_calibration([0, 2], calibration).dtype == np.float64
    tensor = apply_calibration(torch.tensor(z, dtype=torch.float32), calibration)
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
    np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-5, atol=1e-7)
    assert apply_calibration(torch.tensor([1, 2]), calibration).dtype == torch.float32
    half = apply_calibration(torch.tensor([0.0], dtype=torch.float16), calibration)
    assert half.dtype == torch.float16
    np.testing.assert_allclose(calibrated_logits(z, calibration), 0.5 * z - 1.0)
    assert torch.allclose(calibrated_logits(torch.tensor([2.0]), calibration), torch.tensor([0.0]))


def test_apply_calibration_accepts_v1_forms():
    z = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(apply_calibration(z, 2.0), 1.0 / (1.0 + np.exp(-z / 2.0)))  # a bare temperature
    np.testing.assert_allclose(apply_calibration(z, {"method": "temperature", "temperature": 2.0}),
                               1.0 / (1.0 + np.exp(-z / 2.0)))
    np.testing.assert_allclose(apply_calibration(z, None), 1.0 / (1.0 + np.exp(-z)))


@pytest.mark.parametrize(
    "calibration, match",
    [
        ({"method": "platt", "slope": 0.0, "intercept": 0.0}, "slope"),
        ({"method": "platt", "slope": float("nan"), "intercept": 0.0}, "slope"),
        ({"method": "platt", "slope": 1.0, "intercept": float("inf")}, "intercept"),
        ({"method": "platt", "slope": True, "intercept": 0.0}, "slope"),
        ({"method": "isotonic", "slope": 1.0, "intercept": 0.0}, "method"),
        ({"method": "temperature", "temperature": -1.0}, "temperature"),
        ("platt", "dict"),
        (0.0, "temperature"),
    ],
)
def test_apply_calibration_rejects_unusable_calibrations(calibration, match):
    with pytest.raises(ValueError, match=match):
        apply_calibration(np.zeros(2), calibration)


def test_validate_calibration_strict_bounds_and_helpers():
    assert validate_calibration({"slope": np.float32(2.0), "intercept": torch.tensor(-1.0)}) == {
        "method": "platt", "slope": 2.0, "intercept": -1.0}
    assert validate_calibration({"method": "platt", "slope": 50.0, "intercept": 0.0})["slope"] == 50.0
    with pytest.raises(ValueError, match="outside"):
        validate_calibration({"method": "platt", "slope": 50.0, "intercept": 0.0}, strict=True)
    assert temperature_of({"method": "platt", "slope": 4.0, "intercept": 1.0}) == 0.25
    v2 = {"format_version": 2, "calibration": {"method": "platt", "slope": 0.5, "intercept": -3.0}, "temperature": 2.0}
    assert checkpoint_calibration(v2) == {"method": "platt", "slope": 0.5, "intercept": -3.0}
    assert checkpoint_calibration({"temperature": 4.0}) == {"method": "temperature", "slope": 0.25, "intercept": 0.0}
    assert checkpoint_calibration({}) == {"method": "temperature", "slope": 1.0, "intercept": 0.0}


def test_fit_platt_is_fast_on_millions_of_points():
    logits, labels = _shifted_data(0.8, -4.0, n=2_000_000, seed=1)
    start = time.perf_counter()
    fit_platt(logits.astype(np.float32), labels)
    assert time.perf_counter() - start < 10.0
