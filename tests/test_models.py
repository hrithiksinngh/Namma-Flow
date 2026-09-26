"""Tests for ``src.models.stgcn`` (spec GATv2-GRU model, A3T-GCN alternative, factory)."""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from src.data_pipeline.graph_io import graph_to_arrays
from src.models.stgcn import (
    ARCHITECTURES,
    MODEL_DEFAULTS,
    A3TGCNFlood,
    SpatioTemporalFloodGNN,
    build_model,
    count_parameters,
    enable_mc_dropout,
)
from tests.conftest import make_grid_graph

F_IN, E_DIM, HID = 8, 2, 16
MODEL_CLASSES = (SpatioTemporalFloodGNN, A3TGCNFlood)


# --------------------------------------------------------------------------- helpers & fixtures


class _SpecModel(nn.Module):
    """Verbatim copy of the model in namma_flow_project_context.md section 5.3."""

    def __init__(self, node_in_dim=8, edge_dim=2, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim, edge_dim=edge_dim, heads=1, concat=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, hidden_state=None):
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        if hidden_state is None:
            hidden_state = torch.zeros_like(h)
        h_temporal = self.gru(h, hidden_state)
        out = F.relu(self.fc1(h_temporal))
        probs = torch.sigmoid(self.fc2(self.dropout(out)))
        return probs, h_temporal


def _random_graph(num_nodes: int = 12, chords: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional ring plus random chords, with random edge features."""
    gen = torch.Generator().manual_seed(seed)
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    if chords:
        src = torch.cat([src, torch.randint(0, num_nodes, (chords,), generator=gen)])
        dst = torch.cat([dst, torch.randint(0, num_nodes, (chords,), generator=gen)])
    keep = src != dst
    src, dst = src[keep], dst[keep]
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    edge_attr = torch.randn(edge_index.size(1), E_DIM, generator=gen)
    return edge_index, edge_attr


def _x_seq(steps: int, num_nodes: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(steps, num_nodes, F_IN, generator=torch.Generator().manual_seed(seed))


def _make(cls, seed: int = 0, **kwargs) -> nn.Module:
    torch.manual_seed(seed)
    params = {"node_in_dim": F_IN, "edge_dim": E_DIM, "hidden_dim": HID, "dropout": 0.2}
    params.update(kwargs)
    return cls(**params)


def _naive_loop(model, x_seq, edge_index, edge_attr, hidden=None):
    """Reference: one single-step ``forward_logits`` call per timestep."""
    outs = []
    for t in range(x_seq.size(0)):
        logits, hidden = model.forward_logits(x_seq[t], edge_index, edge_attr, hidden)
        outs.append(logits.squeeze(-1))
    return torch.stack(outs), hidden


def _block_graph(edge_index, edge_attr, num_nodes, copies):
    """Disjoint union of ``copies`` graphs, exactly like the dataset collate function."""
    ei = torch.cat([edge_index + b * num_nodes for b in range(copies)], dim=1)
    return ei, edge_attr.repeat(copies, 1)


@pytest.fixture
def graph():
    return _random_graph()


@pytest.fixture
def log_records(caplog):
    logger = logging.getLogger("namma_flow")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="namma_flow")
    yield caplog
    logger.removeHandler(caplog.handler)


# --------------------------------------------------------------------------- spec fidelity


@pytest.mark.unit
def test_spec_model_identical_parameters_and_outputs(graph):
    edge_index, edge_attr = graph
    torch.manual_seed(3)
    spec = _SpecModel(F_IN, E_DIM, HID).eval()
    torch.manual_seed(3)
    ours = SpatioTemporalFloodGNN(F_IN, E_DIM, HID).eval()
    spec_state, our_state = spec.state_dict(), ours.state_dict()
    assert list(spec_state) == list(our_state)
    for key in spec_state:
        assert torch.equal(spec_state[key], our_state[key]), key
    x = _x_seq(1, 12)[0]
    h0 = torch.randn(12, HID)
    p_spec, h_spec = spec(x, edge_index, edge_attr, h0)
    p_ours, h_ours = ours(x, edge_index, edge_attr, h0)
    assert torch.equal(p_spec, p_ours) and torch.equal(h_spec, h_ours)


@pytest.mark.unit
def test_spec_state_dict_loads_strictly(graph):
    spec = _SpecModel(F_IN, E_DIM, HID)
    ours = SpatioTemporalFloodGNN(F_IN, E_DIM, HID)
    ours.load_state_dict(spec.state_dict(), strict=True)


@pytest.mark.unit
def test_architecture_layout():
    model = SpatioTemporalFloodGNN(node_in_dim=8, edge_dim=2, hidden_dim=64, heads=2)
    assert isinstance(model.conv1, GATv2Conv) and model.conv1.heads == 2 and model.conv1.concat
    assert model.conv2.heads == 1 and not model.conv2.concat and model.conv2.in_channels == 128
    assert isinstance(model.gru, nn.GRUCell) and model.gru.hidden_size == 64
    assert model.fc1.out_features == 32 and model.fc2.out_features == 1


# --------------------------------------------------------------------------- shapes and semantics


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_forward_shapes_and_probability_range(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls).eval()
    x = _x_seq(1, 12)[0]
    probs, hidden = model(x, edge_index, edge_attr)
    assert probs.shape == (12, 1) and hidden.shape == (12, HID)
    assert ((probs >= 0) & (probs <= 1)).all()
    logits, hidden2 = model.forward_logits(x, edge_index, edge_attr)
    assert torch.allclose(torch.sigmoid(logits), probs) and torch.allclose(hidden, hidden2)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_forward_sequence_shapes_and_probs(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls).eval()
    x_seq = _x_seq(5, 12)
    logits, h_last = model.forward_sequence(x_seq, edge_index, edge_attr)
    probs, _ = model.forward_sequence(x_seq, edge_index, edge_attr, return_logits=False)
    assert logits.shape == (5, 12) and h_last.shape == (12, HID)
    assert torch.allclose(torch.sigmoid(logits), probs)


@pytest.mark.unit
@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 64])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_gatv2_sequence_matches_naive_loop(graph, chunk, checkpointing):
    edge_index, edge_attr = graph
    model = _make(SpatioTemporalFloodGNN, spatial_chunk=chunk, gradient_checkpointing=checkpointing).eval()
    x_seq = _x_seq(5, 12)
    hidden = torch.randn(12, HID)
    with torch.enable_grad():  # checkpointing only engages when autograd is recording
        out, h_last = model.forward_sequence(x_seq, edge_index, edge_attr, hidden)
    ref, h_ref = _naive_loop(model, x_seq, edge_index, edge_attr, hidden)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h_last, h_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_gatv2_sequence_matches_naive_loop_train_mode_without_dropout(graph):
    edge_index, edge_attr = graph
    model = _make(SpatioTemporalFloodGNN, dropout=0.0, spatial_chunk=4).train()
    x_seq = _x_seq(7, 12)
    out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    ref, _ = _naive_loop(model, x_seq, edge_index, edge_attr)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_gatv2_sequence_matches_naive_loop_on_real_grid_graph():
    arrays = graph_to_arrays(make_grid_graph(5, 7))
    edge_index = torch.as_tensor(arrays.edge_index)
    edge_attr = torch.as_tensor(arrays.edge_matrix(["length", "grade"]))
    edge_attr = (edge_attr - edge_attr.mean(0)) / edge_attr.std(0).clamp_min(1e-6)
    model = _make(SpatioTemporalFloodGNN, spatial_chunk=6).eval()
    x_seq = _x_seq(9, arrays.num_nodes)
    out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    ref, _ = _naive_loop(model, x_seq, edge_index, edge_attr)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_batched_windows_equal_separate_windows(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls, spatial_chunk=3).eval()
    windows = [_x_seq(6, 12, seed=s) for s in range(3)]
    ei_b, ea_b = _block_graph(edge_index, edge_attr, 12, 3)
    batched, _ = model.forward_sequence(torch.cat(windows, dim=1), ei_b, ea_b)
    for b, window in enumerate(windows):
        single, _ = model.forward_sequence(window, edge_index, edge_attr)
        assert torch.allclose(batched[:, b * 12 : (b + 1) * 12], single, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_outputs_are_causal(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls, spatial_chunk=4).eval()
    x_seq = _x_seq(6, 12)
    perturbed = x_seq.clone()
    perturbed[3:] += 5.0
    out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    out_p, _ = model.forward_sequence(perturbed, edge_index, edge_attr)
    assert torch.allclose(out[:3], out_p[:3], atol=1e-6)
    assert not torch.allclose(out[3:], out_p[3:])


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_single_timestep_matches_forward(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls).eval()
    x_seq = _x_seq(1, 12)
    out, h = model.forward_sequence(x_seq, edge_index, edge_attr)
    logits, h_ref = model.forward_logits(x_seq[0], edge_index, edge_attr)
    assert torch.allclose(out[0], logits.squeeze(-1), atol=1e-6) and torch.allclose(h, h_ref, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_zero_timesteps_returns_empty(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls).eval()
    out, h = model.forward_sequence(torch.zeros(0, 12, F_IN), edge_index, edge_attr)
    assert out.shape == (0, 12) and h.shape == (12, HID) and torch.count_nonzero(h) == 0
    hidden = torch.randn(12, HID)
    _, h2 = model.forward_sequence(torch.zeros(0, 12, F_IN), edge_index, edge_attr, hidden)
    assert torch.equal(h2, hidden)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_graph_without_edges(cls):
    model = _make(cls, spatial_chunk=3).eval()
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.zeros(0, E_DIM)
    x_seq = _x_seq(4, 5)
    out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    assert out.shape == (4, 5) and torch.isfinite(out).all()
    if cls is SpatioTemporalFloodGNN:
        ref, _ = _naive_loop(model, x_seq, edge_index, edge_attr)
    else:  # sequence-level attention: compare with the dense reference instead
        with torch.no_grad():
            ref, _ = _a3t_reference(model, x_seq, edge_index, edge_attr)
    assert torch.allclose(out, ref, atol=1e-5)
    # Without edges each junction is independent: permuting nodes permutes outputs.
    perm = torch.tensor([4, 2, 0, 1, 3])
    out_perm, _ = model.forward_sequence(x_seq[:, perm], edge_index, edge_attr)
    assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
@pytest.mark.parametrize("with_self_loop", [False, True])
def test_single_node_graph(cls, with_self_loop):
    model = _make(cls).eval()
    if with_self_loop:
        edge_index, edge_attr = torch.zeros(2, 1, dtype=torch.long), torch.ones(1, E_DIM)
    else:
        edge_index, edge_attr = torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, E_DIM)
    out, h = model.forward_sequence(_x_seq(3, 1), edge_index, edge_attr)
    assert out.shape == (3, 1) and h.shape == (1, HID) and torch.isfinite(out).all()


@pytest.mark.unit
def test_input_self_loops_handled_consistently(graph):
    edge_index, edge_attr = graph
    loops = torch.arange(12).repeat(2, 1)
    ei = torch.cat([edge_index, loops], dim=1)
    ea = torch.cat([edge_attr, torch.randn(12, E_DIM)], dim=0)
    model = _make(SpatioTemporalFloodGNN, spatial_chunk=4).eval()
    out, _ = model.forward_sequence(_x_seq(5, 12), ei, ea)
    ref, _ = _naive_loop(model, _x_seq(5, 12), ei, ea)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_float64_inputs_and_int32_edges_are_converted(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls).eval()
    ref, _ = model.forward_sequence(_x_seq(3, 12), edge_index, edge_attr)
    out, _ = model.forward_sequence(_x_seq(3, 12).double(), edge_index.int(), edge_attr.double())
    assert out.dtype == torch.float32 and torch.allclose(out, ref, atol=1e-6)


@pytest.mark.unit
def test_edge_dim_zero_model_needs_no_edge_attr(graph):
    edge_index, _ = graph
    for cls in MODEL_CLASSES:
        model = _make(cls, edge_dim=0).eval()
        out, _ = model.forward_sequence(_x_seq(3, 12), edge_index, None)
        assert out.shape == (3, 12)
        for bad in (torch.ones(edge_index.size(1), 2), torch.ones(edge_index.size(1)), [1.0]):
            with pytest.raises(ValueError, match="edge_dim=0"):
                model.forward_sequence(_x_seq(3, 12), edge_index, bad)
        out_empty, _ = model.forward_sequence(_x_seq(3, 12), edge_index, torch.zeros(edge_index.size(1), 0))
        assert torch.equal(out_empty, out)


@pytest.mark.unit
def test_edge_dim_one_accepts_flat_edge_attr(graph):
    edge_index, edge_attr = graph
    model = _make(SpatioTemporalFloodGNN, edge_dim=1).eval()
    a, _ = model.forward_sequence(_x_seq(2, 12), edge_index, edge_attr[:, 0])
    b, _ = model.forward_sequence(_x_seq(2, 12), edge_index, edge_attr[:, :1])
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- A3T-GCN reference


def _a3t_reference(model: A3TGCNFlood, x_seq, edge_index, edge_attr):
    """Dense re-implementation: T-GCN recurrence + causal additive attention (eval mode)."""
    steps, num_nodes, _ = x_seq.shape
    src, dst = edge_index
    weights = torch.sigmoid(model.edge_gate(edge_attr)).squeeze(-1)
    adj = torch.zeros(num_nodes, num_nodes).index_put((dst, src), weights, accumulate=True)
    adj = adj + torch.eye(num_nodes)
    d_inv_sqrt = adj.sum(1).pow(-0.5)
    a_hat = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    hid = model.hidden_dim
    h = torch.zeros(num_nodes, hid)
    states = []
    for t in range(steps):
        gx = model.input_proj(a_hat @ x_seq[t])
        z, r = torch.sigmoid(gx[:, : 2 * hid] + model.hidden_gates(a_hat @ h)).chunk(2, dim=-1)
        cand = torch.tanh(gx[:, 2 * hid :] + model.hidden_cand(a_hat @ (r * h)))
        h = z * h + (1 - z) * cand
        states.append(h)
    stacked = torch.stack(states)
    scores = model.att_score(torch.tanh(model.att_proj(stacked))).squeeze(-1)  # [T(s), N]
    causal = torch.tril(torch.ones(steps, steps, dtype=torch.bool))  # [t, s]
    masked = scores.unsqueeze(0).expand(steps, steps, num_nodes).masked_fill(~causal[..., None], float("-inf"))
    attn = torch.softmax(masked, dim=1)
    context = torch.einsum("tsn,snh->tnh", attn, stacked)
    head_in = torch.cat([stacked, context], dim=-1)
    return model.fc2(F.relu(model.fc1(head_in))).squeeze(-1), h


@pytest.mark.unit
def test_a3t_matches_dense_reference(graph):
    edge_index, edge_attr = graph
    model = _make(A3TGCNFlood, spatial_chunk=2).eval()
    x_seq = _x_seq(7, 12)
    out, h = model.forward_sequence(x_seq, edge_index, edge_attr)
    with torch.no_grad():
        ref, h_ref = _a3t_reference(model, x_seq, edge_index, edge_attr)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h, h_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_a3t_attention_is_stable_for_large_scores(graph):
    edge_index, edge_attr = graph
    model = _make(A3TGCNFlood).eval()
    with torch.no_grad():
        model.att_score.weight.mul_(1e3)  # scores of order +-1e4 would overflow a naive softmax
    out, _ = model.forward_sequence(_x_seq(6, 12), edge_index, edge_attr)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- training behaviour


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradients_reach_every_parameter(cls, graph, checkpointing):
    edge_index, edge_attr = graph
    model = _make(cls, spatial_chunk=2, gradient_checkpointing=checkpointing).train()
    out, _ = model.forward_sequence(_x_seq(5, 12), edge_index, edge_attr)
    target = (torch.rand(out.shape, generator=torch.Generator().manual_seed(0)) < 0.3).float()
    F.binary_cross_entropy_with_logits(out, target).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


@pytest.mark.unit
def test_checkpointing_reproduces_gradients_with_dropout(graph):
    edge_index, edge_attr = graph
    x_seq = _x_seq(6, 12)

    def grads(checkpointing: bool) -> dict[str, torch.Tensor]:
        model = _make(SpatioTemporalFloodGNN, seed=11, dropout=0.4, spatial_chunk=4,
                      gradient_checkpointing=checkpointing).train()
        torch.manual_seed(99)
        out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
        out.pow(2).mean().backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()}

    plain, recomputed = grads(False), grads(True)
    for name in plain:
        assert torch.allclose(plain[name], recomputed[name], atol=1e-6, rtol=1e-5), name


@pytest.mark.unit
def test_a3t_checkpointing_matches_plain_forward_and_gradients(graph):
    edge_index, edge_attr = graph
    x_seq = _x_seq(6, 12)
    results = {}
    for checkpointing in (False, True):
        model = _make(A3TGCNFlood, seed=4, dropout=0.3, gradient_checkpointing=checkpointing).train()
        torch.manual_seed(21)
        out, h_last = model.forward_sequence(x_seq, edge_index, edge_attr)
        out.pow(2).mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        results[checkpointing] = (out.detach(), h_last.detach(), grads)
    (out_a, h_a, g_a), (out_b, h_b, g_b) = results[False], results[True]
    assert torch.allclose(out_a, out_b, atol=1e-6) and torch.allclose(h_a, h_b, atol=1e-6)
    for name in g_a:
        assert torch.allclose(g_a[name], g_b[name], atol=1e-6, rtol=1e-5), name


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_deterministic_under_seed(cls, graph):
    edge_index, edge_attr = graph
    outputs = []
    for _ in range(2):
        model = _make(cls, seed=123).train()
        torch.manual_seed(7)
        outputs.append(model.forward_sequence(_x_seq(4, 12), edge_index, edge_attr)[0])
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_dropout_inactive_in_eval_active_in_train(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls, dropout=0.5)
    x_seq = _x_seq(4, 12)
    model.eval()
    a, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    b, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    assert torch.equal(a, b)
    model.train()
    c, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    d, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    assert not torch.equal(c, d)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_state_dict_roundtrip(cls, graph, tmp_path):
    edge_index, edge_attr = graph
    model = _make(cls, seed=5).eval()
    path = tmp_path / "model.pt"
    torch.save({"state": model.state_dict(), "config": model.get_config()}, path)
    payload = torch.load(path, weights_only=True)
    restored = build_model(payload["config"])
    restored.load_state_dict(payload["state"])
    restored.eval()
    x_seq = _x_seq(4, 12)
    assert torch.equal(
        model.forward_sequence(x_seq, edge_index, edge_attr)[0],
        restored.forward_sequence(x_seq, edge_index, edge_attr)[0],
    )


# --------------------------------------------------------------------------- input validation


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_sequence_input_validation(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls)
    x_seq = _x_seq(3, 12)
    cases = [
        ((x_seq[0], edge_index, edge_attr), ValueError, "3-D"),
        ((torch.zeros(3, 12, F_IN + 1), edge_index, edge_attr), ValueError, "node_in_dim"),
        ((torch.zeros(3, 0, F_IN), edge_index[:, :0], edge_attr[:0]), ValueError, "no nodes"),
        ((x_seq.long(), edge_index, edge_attr), TypeError, "floating"),
        ((x_seq.masked_fill(x_seq > 1.5, float("nan")), edge_index, edge_attr), ValueError, "NaN"),
        ((x_seq, edge_index, edge_attr[:-1]), ValueError, "rows"),
        ((x_seq, edge_index, edge_attr[:, :1]), ValueError, "edge_dim"),
        ((x_seq, edge_index, None), ValueError, "edge_attr is required"),
        ((x_seq, edge_index, edge_attr.masked_fill(edge_attr > 1, float("inf"))), ValueError, "NaN"),
        ((x_seq, edge_index[0], edge_attr), ValueError, r"\[2, E\]"),
        ((x_seq, edge_index.float(), edge_attr), TypeError, "integer"),
        ((x_seq, edge_index.clamp_max(11) + 1, edge_attr), ValueError, "out of range"),
        ((x_seq, edge_index - 1, edge_attr), ValueError, "out of range"),
        (("not a tensor", edge_index, edge_attr), TypeError, "tensor"),
        ((x_seq, edge_index.tolist(), edge_attr), TypeError, "edge_index must be a torch tensor"),
        ((x_seq, edge_index, edge_attr.tolist()), TypeError, "edge_attr must be a torch tensor"),
        ((x_seq, edge_index, edge_attr.long()), TypeError, "floating"),
    ]
    for args, error, match in cases:
        with pytest.raises(error, match=match):
            model.forward_sequence(*args)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_hidden_state_validation(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls)
    bad_states = (
        torch.zeros(12, HID + 1),
        torch.zeros(11, HID),
        torch.zeros(12, HID, 1),
        torch.zeros(12, HID, dtype=torch.long),
        [[0.0] * HID] * 12,
    )
    for bad in bad_states:
        with pytest.raises(ValueError, match="hidden_state"):
            model.forward_sequence(_x_seq(2, 12), edge_index, edge_attr, bad)
    with pytest.raises(ValueError, match="hidden_state"):
        model.forward(_x_seq(1, 12)[0], edge_index, edge_attr, torch.full((12, HID), float("nan")))


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_step_input_validation(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls)
    with pytest.raises(ValueError, match="2-D"):
        model.forward(_x_seq(2, 12), edge_index, edge_attr)
    with pytest.raises(ValueError, match="node_in_dim"):
        model.forward_logits(torch.zeros(12, 3), edge_index, edge_attr)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"node_in_dim": 0}, "node_in_dim"),
        ({"hidden_dim": -4}, "hidden_dim"),
        ({"hidden_dim": 16.5}, "hidden_dim"),
        ({"hidden_dim": True}, "hidden_dim"),
        ({"edge_dim": -1}, "edge_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": "0.2"}, "dropout"),
        ({"spatial_chunk": 0}, "spatial_chunk"),
        ({"max_chunk_edges": 0}, "max_chunk_edges"),
        ({"gradient_checkpointing": "sometimes"}, "gradient_checkpointing"),
    ],
)
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_constructor_validation(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make(cls, **kwargs)


@pytest.mark.unit
def test_heads_validation():
    with pytest.raises(ValueError, match="heads"):
        _make(SpatioTemporalFloodGNN, heads=0)


# --------------------------------------------------------------------------- chunking policy


@pytest.mark.unit
def test_effective_chunk_respects_edge_budget():
    model = _make(SpatioTemporalFloodGNN, spatial_chunk=8, max_chunk_edges=1000)
    assert model.effective_chunk(steps=24, num_nodes=10, num_edges=40) == 8  # 50 edges/step → cap 20
    assert model.effective_chunk(steps=24, num_nodes=100, num_edges=300) == 2  # 400 edges/step
    assert model.effective_chunk(steps=24, num_nodes=900, num_edges=900) == 1  # never below 1
    assert model.effective_chunk(steps=3, num_nodes=10, num_edges=40) == 3  # never above T


@pytest.mark.unit
def test_auto_checkpointing_policy():
    model = _make(SpatioTemporalFloodGNN, gradient_checkpointing="auto")
    big, small = (24, 16_000, 70_400), (24, 1_000, 4_400)
    with torch.enable_grad():
        assert model.uses_checkpointing(*big)
        assert not model.uses_checkpointing(*small)
    with torch.no_grad():
        assert not model.uses_checkpointing(*big)
    assert _make(SpatioTemporalFloodGNN, gradient_checkpointing=True).uses_checkpointing(*small)
    assert not _make(SpatioTemporalFloodGNN, gradient_checkpointing=False).uses_checkpointing(*big)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value,expected", [("auto", "auto"), ("AUTO", "auto"), (None, "auto"), ("true", True), ("off", False), (True, True)]
)
def test_checkpointing_value_normalisation(value, expected):
    assert _make(SpatioTemporalFloodGNN, gradient_checkpointing=value).gradient_checkpointing == expected


# --------------------------------------------------------------------------- factory & utilities


@pytest.mark.unit
def test_build_model_from_project_config(cfg):
    model = build_model(cfg["model"])
    assert isinstance(model, SpatioTemporalFloodGNN)
    assert model.node_in_dim == cfg["model"]["node_in_dim"] == 10
    assert model.hidden_dim == cfg["model"]["hidden_dim"]
    assert model.spatial_chunk == cfg["model"].get("spatial_chunk", MODEL_DEFAULTS["spatial_chunk"])


@pytest.mark.unit
def test_build_model_accepts_full_config(cfg):
    assert isinstance(build_model(cfg), SpatioTemporalFloodGNN)


@pytest.mark.unit
def test_build_model_defaults_and_architectures():
    assert isinstance(build_model(None), SpatioTemporalFloodGNN)
    assert isinstance(build_model({}), SpatioTemporalFloodGNN)
    assert isinstance(build_model({"architecture": "a3t_gcn", "hidden_dim": 8}), A3TGCNFlood)
    assert isinstance(build_model({"architecture": "A3T-GCN"}), A3TGCNFlood)
    assert set(ARCHITECTURES) == {"gatv2_gru", "a3t_gcn"}


@pytest.mark.unit
def test_build_model_ignores_training_hyperparameters():
    model = build_model({"hidden_dim": 8, "learning_rate": 0.1, "batch_size": 4, "epochs": 2})
    assert model.hidden_dim == 8


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cfg,match",
    [
        ({"architecture": "transformer"}, "Unknown model architecture"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"heads": -1}, "heads"),
        ({"dropout": 2}, "dropout"),
        ({"node_in_dim": "eight"}, "node_in_dim"),
        (["gatv2_gru"], "mapping"),
    ],
)
def test_build_model_rejects_bad_config(model_cfg, match):
    with pytest.raises(ValueError, match=match):
        build_model(model_cfg)


@pytest.mark.unit
@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_get_config_round_trip(arch):
    model = build_model({"architecture": arch, "hidden_dim": 12, "spatial_chunk": 3})
    config = model.get_config()
    assert config["architecture"] == arch
    clone = build_model(config)
    assert type(clone) is type(model) and clone.get_config() == config
    assert count_parameters(clone) == count_parameters(model)


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_count_parameters(cls):
    model = _make(cls)
    total = sum(p.numel() for p in model.parameters())
    assert count_parameters(model) == total
    model.fc2.weight.requires_grad_(False)
    assert count_parameters(model) == total - model.fc2.weight.numel()
    assert count_parameters(model, trainable_only=False) == total


@pytest.mark.unit
def test_spec_sized_parameter_count():
    model = build_model({"architecture": "gatv2_gru", "node_in_dim": 8, "edge_dim": 2, "hidden_dim": 64, "heads": 2})
    assert count_parameters(model) == count_parameters(_SpecModel())


@pytest.mark.unit
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_enable_mc_dropout(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls, dropout=0.5).train()
    assert enable_mc_dropout(model) is None
    assert not model.training
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert dropouts and all(m.training for m in dropouts)
    others = [m for m in model.modules() if not isinstance(m, nn.Dropout)]
    assert not any(m.training for m in others)
    x_seq = _x_seq(3, 12)
    with torch.no_grad():
        a, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
        b, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    assert not torch.equal(a, b)


@pytest.mark.unit
def test_enable_mc_dropout_warns_without_active_dropout(log_records):
    enable_mc_dropout(_make(SpatioTemporalFloodGNN, dropout=0.0))
    assert any("MC dropout" in r.getMessage() and r.levelno == logging.WARNING for r in log_records.records)


# --------------------------------------------------------------------------- scale & devices


@pytest.mark.integration
@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_real_size_graph_inference_is_fast(arch):
    num_nodes, undirected = 1000, 2200
    gen = torch.Generator().manual_seed(0)
    src = torch.randint(0, num_nodes, (undirected,), generator=gen)
    dst = (src + torch.randint(1, 30, (undirected,), generator=gen)) % num_nodes
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    edge_attr = torch.randn(edge_index.size(1), 2, generator=gen)
    model = build_model({"architecture": arch}).eval()
    x_seq = torch.randn(24, num_nodes, 8, generator=gen)
    start = time.perf_counter()
    with torch.no_grad():
        out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    assert out.shape == (24, num_nodes) and torch.isfinite(out).all()
    assert time.perf_counter() - start < 10.0


@pytest.mark.integration
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS not available")
@pytest.mark.parametrize("cls", MODEL_CLASSES)
def test_mps_matches_cpu(cls, graph):
    edge_index, edge_attr = graph
    model = _make(cls, spatial_chunk=3).eval()
    x_seq = _x_seq(4, 12)
    with torch.no_grad():
        cpu_out, _ = model.forward_sequence(x_seq, edge_index, edge_attr)
    device = torch.device("mps")
    with torch.no_grad():
        mps_out, _ = model.to(device).forward_sequence(x_seq, edge_index, edge_attr)  # inputs moved automatically
    assert mps_out.device.type == "mps"
    np.testing.assert_allclose(mps_out.cpu().numpy(), cpu_out.numpy(), atol=1e-4, rtol=1e-4)
