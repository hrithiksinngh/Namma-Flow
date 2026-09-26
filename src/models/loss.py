"""Class-imbalance-aware binary losses for junction flood classification.

Flooded junction-hours are rare (well under 5 % of node-hours), so an unweighted
cross-entropy lets a model score well by predicting "dry" everywhere. Two remedies
are provided, both with the same call signature::

    loss = criterion(inputs, targets, mask=None)

* :class:`FocalLoss` — binary focal loss (Lin et al., 2017, "Focal Loss for Dense
  Object Detection"): ``alpha_t * (1 - p_t) ** gamma * CE``. ``alpha`` weights the
  positive (flooded) class, ``gamma`` down-weights easy, confidently-correct examples.
  The spec's defaults are ``gamma=2.0, alpha=0.85``.
* :class:`WeightedBCELoss` — binary cross-entropy with a positive-class weight.

Both accept logits (default, numerically stable via ``binary_cross_entropy_with_logits``)
or probabilities (``from_logits=False``; clamped to ``[eps, 1 - eps]``), soft targets in
``[0, 1]``, and an optional boolean ``mask`` selecting which elements count (e.g. the
non-warm-up timesteps of a training window). NaN/inf *inputs* are propagated, not
raised, so the trainer can detect and skip a non-finite step; invalid *targets* raise.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.config import deep_merge
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

REDUCTIONS: tuple[str, ...] = ("none", "mean", "sum")
LOSS_NAMES: tuple[str, ...] = ("focal", "weighted_bce")
_LOSS_ALIASES: dict[str, str] = {
    "focal": "focal",
    "focal_loss": "focal",
    "weighted_bce": "weighted_bce",
    "wbce": "weighted_bce",
}
LOSS_DEFAULTS: dict[str, Any] = {
    "name": "focal",
    "gamma": 2.0,
    "alpha": 0.85,
    "pos_weight": None,
    "reduction": "mean",
    "from_logits": True,
    "eps": 1e-6,
}
POS_WEIGHT_BOUNDS: tuple[float, float] = (1.0, 1000.0)
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


# --------------------------------------------------------------------------- validation helpers


def _real_number(name: str, value: Any) -> float:
    """Return ``value`` as a finite float or raise ``ValueError`` (bools and strings rejected)."""
    if isinstance(value, Tensor) and value.numel() == 1:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r} ({type(value).__name__})")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _check_reduction(reduction: Any) -> str:
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    return str(reduction)


def _check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _check_eps(eps: Any) -> float:
    eps = _real_number("eps", eps)
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must lie in (0, 0.5), got {eps}")
    return eps


def _resolve_mask(mask: Any, shape: torch.Size, device: torch.device) -> Tensor | None:
    """Convert ``mask`` into a boolean tensor of exactly ``shape``.

    A mask whose shape equals the *leading* dimensions of the inputs (e.g. a per-step
    mask ``[T]`` for inputs ``[T, N]``) is aligned on those leading dimensions;
    otherwise standard (trailing) broadcasting applies.
    """
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype != torch.bool:
        if mask.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"mask must be a boolean (or 0/1 integer) tensor, got dtype {mask.dtype}; use mask.bool()")
        if mask.numel() and bool(((mask != 0) & (mask != 1)).any()):
            raise ValueError("integer mask must contain only 0/1 values")
        mask = mask != 0
    if 0 < mask.dim() < len(shape) and tuple(mask.shape) == tuple(shape[: mask.dim()]):
        mask = mask.reshape(tuple(mask.shape) + (1,) * (len(shape) - mask.dim()))
    try:
        return mask.expand(shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask of shape {tuple(mask.shape)} cannot be broadcast to inputs of shape {tuple(shape)}"
        ) from exc


def _check_value_range(values: Tensor, what: str, hint: str = "") -> None:
    """Raise if ``values`` has entries outside [0, 1] (NaN counts as outside)."""
    if values.numel() == 0:
        return
    low, high = torch.aminmax(values.detach())
    if not (bool(low >= 0.0) and bool(high <= 1.0)):
        raise ValueError(
            f"{what} must lie in [0, 1] (and be finite); got range [{low.item():.4g}, {high.item():.4g}]{hint}"
        )


# --------------------------------------------------------------------------- base class


class _MaskedBinaryLoss(nn.Module):
    """Shared input validation, masking and reduction for elementwise binary losses."""

    def __init__(self, reduction: str = "mean", from_logits: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.reduction = _check_reduction(reduction)
        self.from_logits = _check_bool("from_logits", from_logits)
        self.eps = _check_eps(eps)

    def _elementwise(self, inputs: Tensor, targets: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, inputs: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        """Loss of ``inputs`` (logits or probabilities) against ``targets`` in [0, 1].

        ``mask`` (bool, broadcastable; see :func:`_resolve_mask`) selects the elements
        that count. With ``reduction="none"`` the result has the input shape and
        unselected elements are 0; ``"mean"`` averages over selected elements only and an
        empty selection returns a zero that is still connected to the autograd graph.
        """
        if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
            raise TypeError(
                f"inputs and targets must be torch tensors, got {type(inputs).__name__} and {type(targets).__name__}"
            )
        if not inputs.is_floating_point():
            raise TypeError(f"inputs must be floating point (logits or probabilities), got dtype {inputs.dtype}")
        if inputs.shape != targets.shape:
            raise ValueError(
                f"inputs {tuple(inputs.shape)} and targets {tuple(targets.shape)} must have identical shapes "
                "(implicit broadcasting, e.g. [N, 1] vs [N], would silently compare every pair)"
            )
        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        selected = _resolve_mask(mask, inputs.shape, inputs.device)
        if selected is None:
            values_in, values_tg = inputs, targets
        else:
            values_in, values_tg = inputs[selected], targets[selected]

        _check_value_range(values_tg, "targets")
        if not self.from_logits:
            _check_value_range(values_in, "probability inputs", "; did you pass logits? Use from_logits=True")
        losses = self._elementwise(values_in, values_tg)
        return self._reduce(losses, inputs, selected)

    def _reduce(self, losses: Tensor, inputs: Tensor, selected: Tensor | None) -> Tensor:
        if self.reduction == "none":
            if selected is None:
                return losses
            return torch.zeros_like(inputs).masked_scatter(selected, losses)
        total = losses.sum()
        if self.reduction == "sum":
            return total
        return total / max(losses.numel(), 1)

    def _clamped_probs(self, probs: Tensor) -> Tensor:
        return probs.clamp(self.eps, 1.0 - self.eps)


# --------------------------------------------------------------------------- losses


class FocalLoss(_MaskedBinaryLoss):
    """Binary focal loss ``alpha_t * (1 - p_t) ** gamma * CE(p, y)``.

    Args:
        gamma: focusing parameter (>= 0); 0 recovers (alpha-weighted) cross-entropy.
        alpha: weight of the positive class in [0, 1] (negatives get ``1 - alpha``);
            ``None`` disables class weighting.
        reduction: ``"none" | "mean" | "sum"``.
        from_logits: inputs are logits (stable) rather than probabilities.
        eps: probability clamp used when ``from_logits=False``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.85,
        reduction: str = "mean",
        from_logits: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(reduction=reduction, from_logits=from_logits, eps=eps)
        gamma = _real_number("gamma", gamma)
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None:
            alpha = _real_number("alpha", alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha must lie in [0, 1] (or be None), got {alpha}")
        self.gamma = gamma
        self.alpha = alpha

    def _elementwise(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.from_logits:
            ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            # 1 - p_t without cancellation: sigmoid(-x) = 1 - sigmoid(x) computed directly.
            one_minus_pt = targets * torch.sigmoid(-inputs) + (1.0 - targets) * torch.sigmoid(inputs)
        else:
            probs = self._clamped_probs(inputs)
            ce = -(targets * torch.log(probs) + (1.0 - targets) * torch.log1p(-probs))
            one_minus_pt = targets * (1.0 - probs) + (1.0 - targets) * probs
        loss = ce
        if self.gamma != 0.0:
            # Clamping keeps d/du u**gamma finite at u == 0 for 0 < gamma < 1.
            tiny = torch.finfo(one_minus_pt.dtype).tiny
            loss = loss * one_minus_pt.clamp_min(tiny).pow(self.gamma)
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets))
        return loss

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, alpha={self.alpha}, reduction={self.reduction!r}, "
            f"from_logits={self.from_logits}"
        )


class WeightedBCELoss(_MaskedBinaryLoss):
    """Binary cross-entropy with a positive-class weight (``pos_weight=None`` → 1)."""

    pos_weight: Tensor

    def __init__(
        self,
        pos_weight: float | Tensor | None = None,
        reduction: str = "mean",
        from_logits: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(reduction=reduction, from_logits=from_logits, eps=eps)
        weight = 1.0 if pos_weight is None else _real_number("pos_weight", pos_weight)
        if weight <= 0.0:
            raise ValueError(f"pos_weight must be > 0, got {weight}")
        # A buffer follows .to(device/dtype); it is not part of any checkpoint.
        self.register_buffer("pos_weight", torch.tensor(weight), persistent=False)

    def _elementwise(self, inputs: Tensor, targets: Tensor) -> Tensor:
        weight = self.pos_weight.to(device=inputs.device, dtype=inputs.dtype)
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=weight, reduction="none")
        probs = self._clamped_probs(inputs)
        return -(weight * targets * torch.log(probs) + (1.0 - targets) * torch.log1p(-probs))

    def extra_repr(self) -> str:
        return (
            f"pos_weight={self.pos_weight.item():.4g}, reduction={self.reduction!r}, "
            f"from_logits={self.from_logits}"
        )


# --------------------------------------------------------------------------- factory


def pos_weight_from_rate(pos_rate: float | None) -> float:
    """Positive-class weight ``(1 - pos_rate) / pos_rate`` clipped to :data:`POS_WEIGHT_BOUNDS`.

    ``None`` (unknown class balance) falls back to 1.0 with a WARNING; rates of exactly
    0 or 1 clip to the bounds with a WARNING; anything outside [0, 1] raises.
    """
    low, high = POS_WEIGHT_BOUNDS
    if pos_rate is None:
        LOGGER.warning("pos_weight: training pos_rate unknown; using pos_weight=1.0 (unweighted BCE)")
        return 1.0
    try:
        rate = _real_number("pos_rate", pos_rate)
    except ValueError as exc:
        raise ValueError(f"pos_rate must be a finite number in [0, 1]: {exc}") from exc
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"pos_rate must lie in [0, 1], got {rate}")
    if rate == 0.0:
        LOGGER.warning("pos_weight: training split has no positives (pos_rate=0); clipping pos_weight to %.0f", high)
        return high
    raw = (1.0 - rate) / rate
    if rate == 1.0:
        LOGGER.warning("pos_weight: training split is all positives (pos_rate=1); clipping pos_weight to %.0f", low)
    return float(min(max(raw, low), high))


def build_loss(loss_cfg: Mapping[str, Any] | None, pos_rate: float | None = None) -> nn.Module:
    """Build the loss named by ``loss_cfg["name"]`` (the ``loss`` config section).

    ``focal`` uses ``gamma``/``alpha``; ``weighted_bce`` uses ``pos_weight`` or, when it
    is null, derives it from the training positive rate via :func:`pos_weight_from_rate`.
    Unknown names raise ``ValueError``.
    """
    if loss_cfg is None:
        loss_cfg = {}
    if not isinstance(loss_cfg, Mapping):
        raise ValueError(f"loss config must be a mapping, got {type(loss_cfg).__name__}")
    cfg = deep_merge(LOSS_DEFAULTS, loss_cfg)
    raw_name = str(cfg["name"]).strip().lower().replace("-", "_")
    name = _LOSS_ALIASES.get(raw_name)
    if name is None:
        raise ValueError(f"Unknown loss {cfg['name']!r}; expected one of {LOSS_NAMES}")
    common = {"reduction": cfg["reduction"], "from_logits": cfg["from_logits"], "eps": cfg["eps"]}
    if name == "focal":
        loss: nn.Module = FocalLoss(gamma=cfg["gamma"], alpha=cfg["alpha"], **common)
    else:
        pos_weight = cfg["pos_weight"]
        if pos_weight is None:
            pos_weight = pos_weight_from_rate(pos_rate)
        loss = WeightedBCELoss(pos_weight=pos_weight, **common)
    LOGGER.info("Loss: %s(%s)", type(loss).__name__, loss.extra_repr())
    return loss
