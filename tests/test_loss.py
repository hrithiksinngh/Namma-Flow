"""Tests for ``src.models.loss`` (focal loss, weighted BCE, loss factory)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.loss import (
    LOSS_DEFAULTS,
    POS_WEIGHT_BOUNDS,
    FocalLoss,
    WeightedBCELoss,
    build_loss,
    pos_weight_from_rate,
)

pytestmark = pytest.mark.unit


def _np_focal(logits: np.ndarray, targets: np.ndarray, gamma: float, alpha: float | None) -> np.ndarray:
    """Independent float64 reference of the binary focal loss (Lin et al., 2017)."""
    p = 1.0 / (1.0 + np.exp(-logits))
    p_t = np.where(targets == 1, p, 1 - p)
    ce = -np.log(p_t)
    loss = (1 - p_t) ** gamma * ce
    if alpha is not None:
        loss = np.where(targets == 1, alpha, 1 - alpha) * loss
    return loss


@pytest.fixture
def log_records(caplog):
    """Capture WARNING records of the project logger (it does not propagate to root)."""
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(6, 40, generator=gen, dtype=torch.float64) * 3
    targets = (torch.rand(6, 40, generator=gen) < 0.2).double()
    return logits, targets


# --------------------------------------------------------------------------- focal: values


def test_focal_gamma0_alpha_half_equals_half_bce(batch):
    logits, targets = batch
    loss = FocalLoss(gamma=0.0, alpha=0.5)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(loss, 0.5 * bce, atol=1e-12)


def test_focal_gamma0_alpha_half_soft_labels_equals_half_bce():
    logits = torch.tensor([-2.0, 0.3, 1.7, 4.0], dtype=torch.float64)
    targets = torch.tensor([0.1, 0.5, 0.9, 0.0], dtype=torch.float64)
    loss = FocalLoss(gamma=0.0, alpha=0.5, reduction="none")(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(loss, 0.5 * bce, atol=1e-12)


def test_focal_hand_computed_reference_value():
    # logit 0, positive: p = 0.5, CE = ln 2, modulating factor (1 - 0.5)^2 = 0.25, alpha = 0.85.
    loss = FocalLoss(gamma=2.0, alpha=0.85)(torch.zeros(1), torch.ones(1))
    assert loss.item() == pytest.approx(0.85 * 0.25 * math.log(2.0), rel=1e-6)


@pytest.mark.parametrize("gamma,alpha", [(2.0, 0.85), (0.5, 0.25), (3.0, None), (1.0, 0.0), (2.0, 1.0)])
def test_focal_matches_numpy_reference(batch, gamma, alpha):
    logits, targets = batch
    loss = FocalLoss(gamma=gamma, alpha=alpha, reduction="none")(logits, targets)
    expected = _np_focal(logits.numpy(), targets.numpy(), gamma, alpha)
    np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-10, atol=1e-12)


def test_focal_matches_torchvision_if_available(batch):
    ops = pytest.importorskip("torchvision.ops")
    logits, targets = batch
    ours = FocalLoss(gamma=2.0, alpha=0.85, reduction="sum")(logits, targets)
    theirs = ops.sigmoid_focal_loss(logits, targets, alpha=0.85, gamma=2.0, reduction="sum")
    assert torch.allclose(ours, theirs, rtol=1e-8)


def test_focal_downweights_easy_examples():
    targets = torch.ones(2)
    loss = FocalLoss(gamma=2.0, alpha=0.5, reduction="none")(torch.tensor([4.0, -4.0]), targets)
    bce = F.binary_cross_entropy_with_logits(torch.tensor([4.0, -4.0]), targets, reduction="none")
    ratio = loss / bce
    assert ratio[0] < 1e-3 < ratio[1]  # easy example suppressed, hard example kept


def test_focal_alpha_weights_positives():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    loss = FocalLoss(gamma=2.0, alpha=0.85, reduction="none")(logits, targets)
    assert loss[0] / loss[1] == pytest.approx(0.85 / 0.15, rel=1e-6)


# --------------------------------------------------------------------------- focal: reductions


def test_reductions_are_consistent(batch):
    logits, targets = batch
    none = FocalLoss(reduction="none")(logits, targets)
    assert none.shape == logits.shape
    assert torch.allclose(FocalLoss(reduction="sum")(logits, targets), none.sum())
    assert torch.allclose(FocalLoss(reduction="mean")(logits, targets), none.mean())


# --------------------------------------------------------------------------- focal: logits vs probs


@pytest.mark.parametrize("gamma,alpha", [(2.0, 0.85), (0.0, 0.5), (1.5, None)])
def test_logits_and_probs_agree(batch, gamma, alpha):
    logits, targets = batch
    logits = logits.clamp(-8, 8)  # keep probabilities away from the eps clamp
    from_logits = FocalLoss(gamma=gamma, alpha=alpha, reduction="none")(logits, targets)
    from_probs = FocalLoss(gamma=gamma, alpha=alpha, reduction="none", from_logits=False)(
        torch.sigmoid(logits), targets
    )
    assert torch.allclose(from_logits, from_probs, rtol=1e-6, atol=1e-9)


def test_probs_are_clamped_to_eps():
    loss = FocalLoss(gamma=0.0, alpha=None, from_logits=False, eps=1e-6)(torch.zeros(1), torch.ones(1))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(-math.log(1e-6), rel=1e-4)


def test_probs_outside_unit_interval_rejected():
    with pytest.raises(ValueError, match="from_logits=True"):
        FocalLoss(from_logits=False)(torch.tensor([-0.5, 2.0]), torch.tensor([0.0, 1.0]))


# --------------------------------------------------------------------------- focal: robustness


@pytest.mark.parametrize("gamma", [2.0, 0.5, 0.0])
@pytest.mark.parametrize("magnitude", [100.0, 1e4])
def test_extreme_logits_finite_loss_and_grad(gamma, magnitude):
    logits = torch.tensor([magnitude, -magnitude, magnitude, -magnitude], requires_grad=True)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = FocalLoss(gamma=gamma, alpha=0.85)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    # Confidently wrong predictions dominate the loss.
    per = FocalLoss(gamma=gamma, alpha=0.85, reduction="none")(logits.detach(), targets)
    assert per[1] > per[0] and per[2] > per[3]


def test_all_negative_batch(batch):
    logits, _ = batch
    logits = logits.clone().requires_grad_(True)
    targets = torch.zeros_like(logits)
    loss = FocalLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad >= 0).all()  # pushing every logit down reduces the loss
    confident = FocalLoss()(torch.full((10,), -12.0), torch.zeros(10))
    assert confident.item() < 1e-6


def test_boolean_and_integer_targets_accepted(batch):
    logits, targets = batch
    reference = FocalLoss()(logits, targets)
    assert torch.allclose(FocalLoss()(logits, targets.bool()), reference)
    assert torch.allclose(FocalLoss()(logits, targets.to(torch.uint8)), reference)


@pytest.mark.parametrize(
    "bad", [torch.tensor([0.0, 2.0]), torch.tensor([-1.0, 0.0]), torch.tensor([float("nan"), 0.0])]
)
def test_targets_out_of_range_rejected(bad):
    with pytest.raises(ValueError, match="targets"):
        FocalLoss()(torch.zeros(2), bad)


@pytest.mark.parametrize("shapes", [((5, 1), (5,)), ((4,), (5,)), ((2, 3), (3, 2))])
def test_shape_mismatch_raises(shapes):
    a, b = shapes
    with pytest.raises(ValueError, match="identical shapes"):
        FocalLoss()(torch.zeros(a), torch.zeros(b))


def test_non_tensor_and_integer_inputs_rejected():
    with pytest.raises(TypeError):
        FocalLoss()([0.0, 1.0], torch.zeros(2))
    with pytest.raises(TypeError, match="floating"):
        FocalLoss()(torch.zeros(2, dtype=torch.long), torch.zeros(2))


def test_nan_logits_propagate_for_trainer_to_detect():
    loss = FocalLoss()(torch.tensor([float("nan"), 0.0]), torch.tensor([1.0, 0.0]))
    assert torch.isnan(loss)


def test_empty_input_gives_zero():
    logits = torch.zeros(0, requires_grad=True)
    loss = FocalLoss()(logits, torch.zeros(0))
    assert loss.item() == 0.0
    loss.backward()


# --------------------------------------------------------------------------- masking


def test_mask_selects_elements(batch):
    logits, targets = batch
    mask = torch.rand(logits.shape, generator=torch.Generator().manual_seed(1)) < 0.5
    masked = FocalLoss()(logits, targets, mask=mask)
    expected = FocalLoss()(logits[mask], targets[mask])
    assert torch.allclose(masked, expected)
    summed = FocalLoss(reduction="sum")(logits, targets, mask=mask)
    assert torch.allclose(summed, FocalLoss(reduction="sum")(logits[mask], targets[mask]))


def test_mask_reduction_none_keeps_shape_and_zeroes_unselected(batch):
    logits, targets = batch
    mask = torch.zeros(logits.shape, dtype=torch.bool)
    mask[1:3] = True
    out = FocalLoss(reduction="none")(logits, targets, mask=mask)
    assert out.shape == logits.shape
    assert (out[~mask] == 0).all()
    assert torch.allclose(out[mask], FocalLoss(reduction="none")(logits, targets)[mask])


def test_per_timestep_mask_aligns_on_leading_dims(batch):
    logits, targets = batch  # [T=6, N=40]
    step_mask = torch.tensor([False, False, True, True, True, True])
    loss = FocalLoss()(logits, targets, mask=step_mask)
    assert torch.allclose(loss, FocalLoss()(logits[2:], targets[2:]))


def test_trailing_broadcast_mask(batch):
    logits, targets = batch
    node_mask = torch.zeros(40, dtype=torch.bool)
    node_mask[:10] = True
    loss = FocalLoss()(logits, targets, mask=node_mask)
    assert torch.allclose(loss, FocalLoss()(logits[:, :10], targets[:, :10]))


def test_integer_mask_accepted(batch):
    logits, targets = batch
    step_mask = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.uint8)
    assert torch.allclose(FocalLoss()(logits, targets, mask=step_mask), FocalLoss()(logits[2:], targets[2:]))


def test_empty_mask_returns_zero_with_grad(batch):
    logits, targets = batch
    logits = logits.clone().requires_grad_(True)
    loss = FocalLoss()(logits, targets, mask=torch.zeros(logits.shape, dtype=torch.bool))
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_masked_out_nan_does_not_poison_loss_or_grad():
    logits = torch.tensor([float("nan"), 1.0, float("inf"), -2.0], requires_grad=True)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mask = torch.tensor([False, True, False, True])
    loss = FocalLoss()(logits, targets, mask=mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "mask,error",
    [
        (torch.ones(7, dtype=torch.bool), "broadcast"),
        (torch.ones(6, 40, 2, dtype=torch.bool), "broadcast"),
        (torch.ones(6, 40), "boolean"),
        (torch.full((6, 40), 2, dtype=torch.long), "0/1"),
    ],
)
def test_invalid_masks_rejected(batch, mask, error):
    logits, targets = batch
    with pytest.raises(ValueError, match=error):
        FocalLoss()(logits, targets, mask=mask)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"gamma": -0.1}, "gamma"),
        ({"gamma": float("nan")}, "gamma"),
        ({"gamma": float("inf")}, "gamma"),
        ({"gamma": True}, "gamma"),
        ({"gamma": "2"}, "gamma"),
        ({"alpha": -0.01}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"reduction": "avg"}, "reduction"),
        ({"eps": 0.0}, "eps"),
        ({"eps": 0.6}, "eps"),
        ({"from_logits": "yes"}, "from_logits"),
    ],
)
def test_focal_parameter_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FocalLoss(**kwargs)


def test_focal_repr_mentions_parameters():
    text = repr(FocalLoss(gamma=2.0, alpha=0.85))
    assert "gamma=2.0" in text and "alpha=0.85" in text


# --------------------------------------------------------------------------- weighted BCE


def test_weighted_bce_default_is_plain_bce(batch):
    logits, targets = batch
    assert torch.allclose(WeightedBCELoss()(logits, targets), F.binary_cross_entropy_with_logits(logits, targets))


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_weighted_bce_matches_torch_pos_weight(batch, reduction):
    logits, targets = batch
    ours = WeightedBCELoss(pos_weight=7.5, reduction=reduction)(logits, targets)
    theirs = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(7.5, dtype=torch.float64), reduction=reduction
    )
    assert torch.allclose(ours, theirs)


def test_weighted_bce_logits_and_probs_agree(batch):
    logits, targets = batch
    logits = logits.clamp(-8, 8)
    a = WeightedBCELoss(pos_weight=4.0, reduction="none")(logits, targets)
    b = WeightedBCELoss(pos_weight=4.0, reduction="none", from_logits=False)(torch.sigmoid(logits), targets)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)


def test_weighted_bce_mask_and_extremes():
    logits = torch.tensor([100.0, -100.0, float("nan")], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0])
    loss = WeightedBCELoss(pos_weight=10.0)(logits, targets, mask=torch.tensor([True, True, False]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert loss.item() == pytest.approx((100.0 + 10.0 * 100.0) / 2, rel=1e-5)


def test_weighted_bce_pos_weight_follows_device_and_dtype(batch):
    logits, targets = batch  # float64
    module = WeightedBCELoss(pos_weight=3.0).to(torch.float64)
    assert torch.isfinite(module(logits, targets))
    assert "pos_weight" not in module.state_dict()


@pytest.mark.parametrize("pos_weight", [0.0, -1.0, float("nan"), float("inf"), True, "3"])
def test_weighted_bce_invalid_pos_weight(pos_weight):
    with pytest.raises(ValueError, match="pos_weight"):
        WeightedBCELoss(pos_weight=pos_weight)


def test_weighted_bce_accepts_tensor_pos_weight():
    module = WeightedBCELoss(pos_weight=torch.tensor(2.0))
    assert module.pos_weight.item() == 2.0


# --------------------------------------------------------------------------- factory


def test_build_loss_from_project_config(cfg):
    loss = build_loss(cfg["loss"], pos_rate=0.01)
    assert isinstance(loss, FocalLoss)
    assert loss.gamma == 2.0 and loss.alpha == 0.85
    assert loss.reduction == "mean" and loss.from_logits


def test_build_loss_defaults_when_missing():
    assert isinstance(build_loss(None), FocalLoss)
    assert isinstance(build_loss({}), FocalLoss)
    assert LOSS_DEFAULTS["name"] == "focal"


def test_build_loss_names_are_normalised():
    assert isinstance(build_loss({"name": " Focal "}), FocalLoss)
    assert isinstance(build_loss({"name": "Weighted-BCE", "pos_weight": 2.0}), WeightedBCELoss)


def test_build_loss_explicit_pos_weight():
    loss = build_loss({"name": "weighted_bce", "pos_weight": 12.0}, pos_rate=0.5)
    assert loss.pos_weight.item() == pytest.approx(12.0)


def test_build_loss_derives_pos_weight_from_rate():
    loss = build_loss({"name": "weighted_bce", "pos_weight": None}, pos_rate=0.01)
    assert loss.pos_weight.item() == pytest.approx(99.0)


@pytest.mark.parametrize("rate,expected", [(1e-6, POS_WEIGHT_BOUNDS[1]), (0.9, POS_WEIGHT_BOUNDS[0]), (0.2, 4.0)])
def test_pos_weight_from_rate_clipping(rate, expected):
    assert pos_weight_from_rate(rate) == pytest.approx(expected)


def test_pos_weight_from_rate_degenerate_rates_warn(log_records):
    assert pos_weight_from_rate(0.0) == POS_WEIGHT_BOUNDS[1]
    assert pos_weight_from_rate(1.0) == POS_WEIGHT_BOUNDS[0]
    assert pos_weight_from_rate(None) == 1.0
    warnings = [r for r in log_records.records if r.levelno == logging.WARNING]
    assert sum("pos_weight" in r.getMessage() for r in warnings) == 3


@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan"), "0.1", True])
def test_pos_weight_from_rate_invalid(rate):
    with pytest.raises(ValueError, match="pos_rate"):
        pos_weight_from_rate(rate)


def test_build_loss_unknown_name():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss({"name": "hinge"})


def test_build_loss_rejects_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        build_loss(["focal"])


def test_build_loss_propagates_parameter_validation():
    with pytest.raises(ValueError, match="alpha"):
        build_loss({"name": "focal", "alpha": 2.0})


def test_build_loss_reduction_and_from_logits_options():
    loss = build_loss({"name": "focal", "reduction": "sum", "from_logits": False})
    assert loss.reduction == "sum" and not loss.from_logits
