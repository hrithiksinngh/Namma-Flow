"""Post-hoc probability calibration: Platt scaling (default) and temperature scaling.

A focal-loss model trained with an up-weighted positive class (``loss.alpha``) and
oversampled flood windows ranks junctions well, but every logit is shifted by a roughly
constant offset, so its raw probabilities over-forecast. A pure temperature
(``sigmoid(z / T)``) rescales but cannot remove an offset; Platt scaling fits both::

    p = sigmoid(slope * logit + intercept)

by maximum likelihood on held-out (validation) logits. With an intercept the fitted
probabilities reproduce the base rate of the fitting data (``mean(p) == mean(y)`` at the
optimum). ``slope > 0`` keeps the map monotone, so PR-AUC / ROC-AUC are unchanged.

Every calibration is stored as one dict (checkpoint key ``calibration``, format v2)::

    {"method": "platt" | "temperature" | "none", "slope": a, "intercept": b}

and applied with :func:`apply_calibration` (torch tensors or numpy arrays) - the one
implementation training, evaluation and inference share. :func:`checkpoint_calibration`
reads it from a v2 checkpoint, or derives it from ``temperature`` for a v1 checkpoint.

The Platt NLL is jointly convex in ``(slope, intercept)``, so :func:`fit_platt` solves the
box-constrained problem directly (float64, scipy L-BFGS-B with an analytic gradient, slope in
:data:`PLATT_SLOPE_BOUNDS`, intercept in :data:`PLATT_INTERCEPT_BOUNDS`): the result is the
best calibration inside the box, including when a bound is active (then the free parameter
is optimal given the bound - clipping one parameter of an unconstrained fit is not). The box
optimum is compared with the identity ``(1, 0)``, the intercept-only fit ``(1, b*)`` and the
near-constant ``(slope_min, logit(base rate))`` as a safety net, and the lowest NLL wins
(WARNING when a bound is active or a fallback wins). Degenerate inputs (no finite logits, a
single class, constant logits) give slope 1, intercept 0 with a WARNING.
:func:`fit_temperature` / :func:`apply_temperature` are kept for the temperature-only method
and v1 checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit, logit
from torch import Tensor

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

__all__ = [
    "CALIBRATION_METHODS", "PLATT_INTERCEPT_BOUNDS", "PLATT_SLOPE_BOUNDS", "TEMPERATURE_BOUNDS", "apply_calibration",
    "apply_temperature", "calibrated_logits", "checkpoint_calibration", "fit_calibration", "fit_platt",
    "fit_temperature", "identity_calibration", "temperature_of", "validate_calibration",
]

TEMPERATURE_BOUNDS: tuple[float, float] = (0.05, 20.0)
PLATT_SLOPE_BOUNDS: tuple[float, float] = (0.05, 20.0)
PLATT_INTERCEPT_BOUNDS: tuple[float, float] = (-20.0, 20.0)
CALIBRATION_METHODS: tuple[str, ...] = ("platt", "temperature", "none")
_NEUTRAL = 1.0
_NLL_TOLERANCE = 1e-12


# --------------------------------------------------------------------------- input handling


def _as_tensor(values: Any, name: str) -> Tensor:
    if isinstance(values, Tensor):
        return values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    try:
        return torch.as_tensor(np.asarray(values, dtype=np.float64)).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric array-like: {exc}") from exc


def _check_temperature(temperature: Any) -> float:
    ok = isinstance(temperature, (int, float, np.integer, np.floating)) and not isinstance(temperature, bool)
    if isinstance(temperature, Tensor) and temperature.numel() == 1:
        ok, temperature = True, float(temperature.item())
    if not ok or not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    return float(temperature)


def _prepare(logits: Any, labels: Any, caller: str) -> tuple[Tensor, Tensor]:
    z, y = _as_tensor(logits, "logits"), _as_tensor(labels, "labels")
    if z.numel() != y.numel():
        raise ValueError(f"logits and labels must have the same number of elements, got {z.numel()} and {y.numel()}")
    if y.numel() and (not bool(torch.isfinite(y).all()) or float(y.min()) < 0.0 or float(y.max()) > 1.0):
        raise ValueError("labels must be finite values in [0, 1]")
    finite = torch.isfinite(z)
    if not bool(finite.all()):
        LOGGER.warning("%s: ignoring %d non-finite logits", caller, int((~finite).sum()))
        z, y = z[finite], y[finite]
    return z, y


def _degenerate_reason(z: Tensor, y: Tensor) -> str | None:
    if z.numel() == 0:
        return "no finite logits"
    if float(y.min()) == float(y.max()):
        return "labels contain a single class"
    if float(z.abs().max()) < 1e-12:
        return "all logits are zero"
    return None


def _check_max_iter(max_iter: Any) -> int:
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    return int(max_iter)


def _nll(z: Tensor, y: Tensor, slope: float, intercept: float = 0.0) -> float:
    return float(F.binary_cross_entropy_with_logits(z * slope + intercept, y))


def _lbfgs(params: list[Tensor], loss_fn: Callable[[], Tensor], max_iter: int) -> bool:
    """Minimise ``loss_fn`` over ``params`` in place; False when L-BFGS fails numerically."""
    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError as exc:  # pragma: no cover - LBFGS numerical failure is not reproducible on demand
        LOGGER.warning("Calibration: L-BFGS failed (%s)", exc)
        return False
    return all(bool(torch.isfinite(p.detach()).all()) for p in params)


# --------------------------------------------------------------------------- calibration dicts


def identity_calibration(method: str = "none") -> dict[str, Any]:
    """The neutral calibration ``p = sigmoid(logit)``."""
    return {"method": method, "slope": 1.0, "intercept": 0.0}


def _number(calibration: Mapping[str, Any], key: str) -> float:
    value = calibration.get(key)
    if isinstance(value, Tensor) and value.numel() == 1:
        value = value.item()
    ok = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    if not ok or not math.isfinite(float(value)):
        raise ValueError(f"calibration '{key}' must be a finite number, got {value!r}")
    return float(value)


def validate_calibration(calibration: Any, *, strict: bool = False) -> dict[str, Any]:
    """Normalised ``{"method", "slope", "intercept"}`` (floats); raises ``ValueError`` when unusable.

    A dict with only ``temperature`` (method ``temperature``) gets ``slope = 1 / T``.
    ``strict`` also enforces :data:`PLATT_SLOPE_BOUNDS` / :data:`PLATT_INTERCEPT_BOUNDS`
    (what the trainer writes); otherwise any ``slope > 0`` is accepted.
    """
    if not isinstance(calibration, Mapping):
        raise ValueError(f"calibration must be a dict with slope and intercept, got {type(calibration).__name__}")
    method = str(calibration.get("method", "platt")).strip().lower()
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"calibration method must be one of {list(CALIBRATION_METHODS)}, got {method!r}")
    if "slope" not in calibration and "temperature" in calibration:
        slope, intercept = 1.0 / _check_temperature(calibration["temperature"]), 0.0
    else:
        slope, intercept = _number(calibration, "slope"), _number(calibration, "intercept")
    if slope <= 0.0:
        raise ValueError(f"calibration slope must be > 0 (a monotone map), got {slope!r}")
    if strict:
        (s_low, s_high), (i_low, i_high) = PLATT_SLOPE_BOUNDS, PLATT_INTERCEPT_BOUNDS
        if not (s_low - 1e-9 <= slope <= s_high + 1e-9 and i_low - 1e-9 <= intercept <= i_high + 1e-9):
            raise ValueError(f"calibration slope {slope:.4g} / intercept {intercept:.4g} lie outside "
                             f"{PLATT_SLOPE_BOUNDS} / {PLATT_INTERCEPT_BOUNDS}")
    return {"method": method, "slope": slope, "intercept": intercept}


def temperature_of(calibration: Mapping[str, Any]) -> float:
    """The informational temperature ``1 / slope`` of a calibration dict."""
    return 1.0 / validate_calibration(calibration)["slope"]


def checkpoint_calibration(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Calibration of a training checkpoint: its ``calibration`` dict (format v2) or, for a
    v1 checkpoint without one, ``{"method": "temperature", "slope": 1 / temperature, ...}``."""
    stored = checkpoint.get("calibration")
    if stored is not None:
        return validate_calibration(stored)
    temperature = _check_temperature(checkpoint.get("temperature", _NEUTRAL))
    return {"method": "temperature", "slope": 1.0 / temperature, "intercept": 0.0}


def _resolve(calibration: Any) -> tuple[float, float]:
    if calibration is None:
        LOGGER.warning("apply_calibration: no calibration given; returning uncalibrated sigmoid(logit)")
        return 1.0, 0.0
    if isinstance(calibration, (int, float, np.integer, np.floating, Tensor)) and not isinstance(calibration, bool):
        return 1.0 / _check_temperature(calibration), 0.0  # a bare number is a temperature (v1)
    normalised = validate_calibration(calibration)
    return normalised["slope"], normalised["intercept"]


def calibrated_logits(logits: Any, calibration: Any) -> Any:
    """``slope * logits + intercept`` (same container type as ``logits``; float)."""
    slope, intercept = _resolve(calibration)
    if isinstance(logits, Tensor):
        values = logits if logits.is_floating_point() else logits.to(torch.float32)
        return values * slope + intercept
    values = np.asarray(logits)
    values = values if values.dtype.kind == "f" else values.astype(np.float64)
    return (values * slope + intercept).astype(values.dtype, copy=False)


def apply_calibration(logits: Any, calibration: Any) -> Any:
    """Calibrated probabilities ``sigmoid(slope * logits + intercept)``.

    ``logits``: a torch tensor (returns a tensor on the same device) or anything numpy
    accepts (returns an ndarray). Floating inputs keep their dtype (half precision is
    computed in float32/float64); integer inputs give float32 tensors / float64 arrays. ``calibration``: a dict
    as written by the trainer (``method``/``slope``/``intercept``; ``{"temperature": T}``
    also works), a bare number (a v1 temperature) or ``None`` (identity, with a WARNING).
    Raises ``ValueError`` for an unusable calibration.
    """
    slope, intercept = _resolve(calibration)
    if isinstance(logits, Tensor):
        values = logits if logits.is_floating_point() else logits.to(torch.float32)
        if values.dtype in (torch.float16, torch.bfloat16):
            return torch.sigmoid(values.float() * slope + intercept).to(values.dtype)
        return torch.sigmoid(values * slope + intercept)
    try:
        values = np.asarray(logits)
        scaled = values.astype(np.float64) * slope + intercept
    except (TypeError, ValueError) as exc:
        raise ValueError(f"logits must be numeric array-like: {exc}") from exc
    return expit(scaled).astype(values.dtype if values.dtype.kind == "f" else np.float64, copy=False)


# --------------------------------------------------------------------------- fitting


_BOUND_TOLERANCE = 1e-9


def _platt_objective(z: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Mean NLL of ``sigmoid(a * z + b)`` and its gradient in ``(a, b)`` (float64, overflow-safe)."""

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        scores = params[0] * z + params[1]
        residual = expit(scores) - y
        value = float(np.mean(np.logaddexp(0.0, scores) - y * scores))
        return value, np.array([np.mean(residual * z), np.mean(residual)])

    return objective


def _box_minimum(objective: Callable[[np.ndarray], tuple[float, np.ndarray]], x0: np.ndarray,
                 bounds: list[tuple[float, float]], max_iter: int) -> np.ndarray | None:
    """L-BFGS-B minimum of ``objective`` inside ``bounds``; None when the solver fails numerically."""
    start = np.clip(np.asarray(x0, dtype=np.float64), [b[0] for b in bounds], [b[1] for b in bounds])
    try:
        result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12})
    except (ValueError, FloatingPointError) as exc:  # non-finite objective or solver failure
        LOGGER.warning("fit_platt: L-BFGS-B failed (%s)", exc)
        return None
    x = np.asarray(result.x, dtype=np.float64)
    return x if np.isfinite(x).all() else None


def _intercept_only(z: np.ndarray, y: np.ndarray, max_iter: int) -> np.ndarray:
    """``(1, b*)``: the best intercept at slope 1 (inside the intercept bounds)."""
    shifted = _platt_objective(z, y)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = shifted(np.array([1.0, params[0]]))
        return value, grad[1:]

    best = _box_minimum(objective, np.zeros(1), [PLATT_INTERCEPT_BOUNDS], max_iter)
    return np.array([1.0, 0.0 if best is None else float(best[0])])


def _platt_candidates(z: np.ndarray, y: np.ndarray, max_iter: int) -> dict[str, np.ndarray]:
    """The box optimum and the safety-net fits it is compared with."""
    base_rate = min(max(float(y.mean()), 1e-12), 1.0 - 1e-12)
    candidates = {"identity": np.array([1.0, 0.0]), "intercept-only": _intercept_only(z, y, max_iter),
                  "near-constant": np.array([PLATT_SLOPE_BOUNDS[0],
                                             float(np.clip(logit(base_rate), *PLATT_INTERCEPT_BOUNDS))])}
    box = _box_minimum(_platt_objective(z, y), np.array([1.0, 0.0]), [PLATT_SLOPE_BOUNDS, PLATT_INTERCEPT_BOUNDS],
                       max_iter)
    if box is None:
        LOGGER.warning("fit_platt: the box-constrained optimisation did not converge to finite values")
    else:
        candidates = {"box optimum": box, **candidates}
    return candidates


def _active_bounds(slope: float, intercept: float) -> list[str]:
    names = []
    for name, value, (low, high) in (("slope", slope, PLATT_SLOPE_BOUNDS), ("intercept", intercept,
                                                                            PLATT_INTERCEPT_BOUNDS)):
        if value <= low + _BOUND_TOLERANCE or value >= high - _BOUND_TOLERANCE:
            names.append(f"{name} {value:.4g} at its bound {(low, high)}")
    return names


def fit_platt(logits: Any, labels: Any, max_iter: int = 200) -> dict[str, Any]:
    """Platt scaling ``sigmoid(slope * logits + intercept)`` minimising the NLL on ``labels``.

    Accepts tensors or array-likes of any shape (flattened); ``labels`` may be soft in
    ``[0, 1]``. Returns ``{"method": "platt", "slope", "intercept"}``: the joint NLL minimum
    inside :data:`PLATT_SLOPE_BOUNDS` x :data:`PLATT_INTERCEPT_BOUNDS` (never worse than the
    identity, the intercept-only or the near-constant fit); degenerate inputs give slope 1,
    intercept 0.
    """
    max_iter = _check_max_iter(max_iter)
    z_t, y_t = _prepare(logits, labels, "fit_platt")
    reason = _degenerate_reason(z_t, y_t)
    if reason is not None:
        LOGGER.warning("fit_platt: %s; using slope 1, intercept 0 (uncalibrated)", reason)
        return identity_calibration("platt")
    z, y = z_t.numpy(), y_t.numpy()
    objective = _platt_objective(z, y)
    scored = {name: (objective(x)[0], x) for name, x in _platt_candidates(z, y, max_iter).items()}
    best = min(scored, key=lambda name: scored[name][0] - (_NLL_TOLERANCE if name == "box optimum" else 0.0))
    nll, (slope, intercept) = scored[best][0], (float(v) for v in scored[best][1])
    if best == "identity":
        LOGGER.warning("fit_platt: no fit improves the NLL over the identity; using slope 1, intercept 0")
        return identity_calibration("platt")
    if best != "box optimum":
        LOGGER.warning("fit_platt: the %s fit beats the box-constrained optimiser; using it", best)
    active = _active_bounds(slope, intercept)
    if active:
        LOGGER.warning("fit_platt: the best calibration inside the bounds has %s (near-separable or strongly "
                       "shifted logits); the other parameter is optimal given it", "; ".join(active))
    LOGGER.info("Platt scaling: slope %.4f, intercept %.4f (NLL %.5f -> %.5f on %d points; mean p %.4g vs base "
                "rate %.4g)", slope, intercept, scored["identity"][0], nll, z.size,
                float(expit(slope * z + intercept).mean()), float(y.mean()))
    return {"method": "platt", "slope": slope, "intercept": intercept}


def fit_temperature(logits: Any, labels: Any, max_iter: int = 200) -> float:
    """Temperature ``T`` minimising the NLL of ``sigmoid(logits / T)`` against ``labels``.

    Accepts tensors or array-likes of any shape (flattened). ``labels`` may be soft in
    ``[0, 1]``. Returns a float in :data:`TEMPERATURE_BOUNDS`; degenerate inputs → 1.0.
    """
    max_iter = _check_max_iter(max_iter)
    z, y = _prepare(logits, labels, "fit_temperature")
    reason = _degenerate_reason(z, y)
    if reason is not None:
        LOGGER.warning("fit_temperature: %s; using temperature 1.0", reason)
        return _NEUTRAL
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    if not _lbfgs([log_t], lambda: F.binary_cross_entropy_with_logits(z * torch.exp(-log_t), y), max_iter):
        LOGGER.warning("fit_temperature: the optimisation failed; using temperature 1.0")
        return _NEUTRAL
    raw = float(torch.exp(log_t.detach()).item())
    low, high = TEMPERATURE_BOUNDS
    temperature = min(max(raw, low), high) if math.isfinite(raw) else _NEUTRAL
    if temperature != raw:
        LOGGER.warning("fit_temperature: optimum %.4g lies outside %s; clipped to %.4g", raw, TEMPERATURE_BOUNDS,
                       temperature)
    if _nll(z, y, 1.0 / temperature) > _nll(z, y, 1.0) + _NLL_TOLERANCE:
        LOGGER.warning("fit_temperature: fitted T=%.4g does not improve the NLL over T=1; using 1.0", temperature)
        return _NEUTRAL
    LOGGER.info("Temperature scaling: T=%.4f (NLL %.5f -> %.5f)", temperature, _nll(z, y, 1.0),
                _nll(z, y, 1.0 / temperature))
    return float(temperature)


def fit_calibration(logits: Any, labels: Any, method: str = "platt", max_iter: int = 200) -> dict[str, Any]:
    """Fit a calibration dict with ``method`` in :data:`CALIBRATION_METHODS`."""
    name = str(method).strip().lower()
    if name == "platt":
        return fit_platt(logits, labels, max_iter=max_iter)
    if name == "temperature":
        return {"method": "temperature", "slope": 1.0 / fit_temperature(logits, labels, max_iter=max_iter),
                "intercept": 0.0}
    if name == "none":
        return identity_calibration("none")
    raise ValueError(f"Unknown calibration method {method!r}; expected one of {list(CALIBRATION_METHODS)}")


def apply_temperature(logits: Any, temperature: float) -> Tensor:
    """Calibrated probabilities ``sigmoid(logits / temperature)`` (tensor; same shape and float dtype)."""
    temp = _check_temperature(temperature)
    values = logits if isinstance(logits, Tensor) else torch.as_tensor(np.asarray(logits, dtype=np.float64))
    if not values.is_floating_point():
        values = values.to(torch.float32)
    return torch.sigmoid(values / temp)
