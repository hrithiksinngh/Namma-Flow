"""Spatio-temporal graph neural networks for junction-level flood prediction.

Two architectures share one interface::

    probs,  h = model(x, edge_index, edge_attr, hidden_state=None)            # one hour, [N, F]
    logits, h = model.forward_logits(x, edge_index, edge_attr, hidden_state)  # one hour, logits
    out, h_last = model.forward_sequence(x_seq, edge_index, edge_attr)        # [T, N, F] -> [T, N]

* :class:`SpatioTemporalFloodGNN` (``"gatv2_gru"``) — the exact architecture of the
  project spec: per hour, two GATv2 attention layers over the road graph (edge features
  = normalised segment length and grade) produce a spatial embedding; a ``GRUCell``
  carries the hydrologic state (antecedent water) from hour to hour; an MLP head emits
  the flood logit. Parameter names and initialisation order match the spec verbatim, so
  a spec-trained ``state_dict`` loads strictly.
* :class:`A3TGCNFlood` (``"a3t_gcn"``) — an A3T-GCN style alternative: a T-GCN cell (a
  GRU whose gates are graph convolutions with learned, attribute-dependent edge weights)
  followed by causal additive attention over all hidden states seen so far in the
  sequence.

Performance of ``SpatioTemporalFloodGNN.forward_sequence``: the GATv2 stack does not
depend on the recurrent state, so it runs for ``spatial_chunk`` hours at once on a
time-stacked graph (``c`` disjoint copies of the graph, one per hour), capped so a single
call never stacks more than ``max_chunk_edges`` edges; only the cheap ``GRUCell`` loops
over time. Back-propagation through a 24-hour window must otherwise keep every GATv2
activation alive (~6.6 GB for 16 windows of a 1 000-junction graph), so with
``gradient_checkpointing="auto"`` large batches keep only the hidden state at chunk
boundaries and recompute each chunk (GATv2 + GRU + head) during the backward pass
(bounded memory; measured faster on this CPU too, as it avoids page-faulting gigabytes).
Outputs are numerically identical (allclose) to the naive per-hour loop.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv

from src.utils.config import deep_merge
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

ARCHITECTURES: tuple[str, ...] = ("gatv2_gru", "a3t_gcn")
_ARCHITECTURE_ALIASES: dict[str, str] = {
    "gatv2_gru": "gatv2_gru",
    "gatv2": "gatv2_gru",
    "stgcn": "gatv2_gru",
    "spec": "gatv2_gru",
    "a3t_gcn": "a3t_gcn",
    "a3tgcn": "a3t_gcn",
}
HEAD_HIDDEN_DIM = 32  # width of fc1 in the spec head
# Defaults chosen by benchmark (8-core Apple-silicon CPU, 4 threads, N=1000, E=4400, T=24):
# GATv2 calls are fastest with ~65k-175k stacked edges; beyond that memory grows with no gain.
DEFAULT_SPATIAL_CHUNK = 12  # max hours per stacked GATv2 call (1 = per-hour loop)
DEFAULT_MAX_CHUNK_EDGES = 180_000  # cap on stacked edges (incl. self-loops) per GATv2 call
# "auto" checkpointing engages once T x (E + N) edge-steps would be kept for backward.
# Measured peak RSS without it: ~0.4 GB + ~3.6 KB per edge-step, so 600k keeps training
# under ~2.5 GB (e.g. 24 h windows, >= 5 windows of a 1000-junction / 4400-edge graph).
AUTO_CHECKPOINT_EDGE_STEPS = 600_000

MODEL_DEFAULTS: dict[str, Any] = {
    "architecture": "gatv2_gru",
    "node_in_dim": 8,
    "edge_dim": 2,
    "hidden_dim": 64,
    "heads": 2,
    "dropout": 0.2,
    "spatial_chunk": DEFAULT_SPATIAL_CHUNK,
    "max_chunk_edges": DEFAULT_MAX_CHUNK_EDGES,
    "gradient_checkpointing": "auto",
}
_TRUE_STRINGS = {"true", "yes", "on", "1"}
_FALSE_STRINGS = {"false", "no", "off", "0"}
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)


# --------------------------------------------------------------------------- hyper-parameter validation


def _int_param(name: str, value: Any, minimum: int) -> int:
    """Integer hyper-parameter >= ``minimum`` (integral floats accepted, bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _dropout_param(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) < 1.0:
        raise ValueError(f"dropout must be a number in [0, 1), got {value!r}")
    return float(value)


def _checkpoint_param(value: Any) -> bool | str:
    """Normalise ``gradient_checkpointing`` to ``True``, ``False`` or ``"auto"`` (None → auto)."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "auto"
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    raise ValueError(f"gradient_checkpointing must be true, false or 'auto', got {value!r}")


# --------------------------------------------------------------------------- input validation


def _require_finite(tensor: Tensor, what: str) -> None:
    if tensor.numel() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{what} contains NaN or inf values")


def _check_node_features(x: Any, *, sequence: bool, in_dim: int, ref: Tensor) -> Tensor:
    shape_text = "[T, N, F] (3-D)" if sequence else "[N, F] (2-D)"
    name = "x_seq" if sequence else "x"
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch tensor of shape {shape_text}, got {type(x).__name__}")
    if x.dim() != (3 if sequence else 2):
        raise ValueError(f"{name} must have shape {shape_text}; got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be floating point, got dtype {x.dtype}")
    if x.size(-1) != in_dim:
        raise ValueError(f"{name} has {x.size(-1)} features but the model was built with node_in_dim={in_dim}")
    if x.size(-2) == 0:
        raise ValueError(f"{name} describes a graph with no nodes (N=0)")
    x = x.to(device=ref.device, dtype=ref.dtype)
    _require_finite(x, name)
    return x


def _check_edge_index(edge_index: Any, num_nodes: int, ref: Tensor) -> Tensor:
    if not isinstance(edge_index, Tensor):
        raise TypeError(f"edge_index must be a torch tensor of shape [2, E], got {type(edge_index).__name__}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}")
    if edge_index.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"edge_index must be an integer tensor, got dtype {edge_index.dtype}")
    edge_index = edge_index.to(device=ref.device, dtype=torch.long)
    if edge_index.numel():
        low, high = int(edge_index.min()), int(edge_index.max())
        if low < 0 or high >= num_nodes:
            raise ValueError(
                f"edge_index node ids out of range: found [{low}, {high}] but the graph has N={num_nodes} nodes"
            )
    return edge_index


def _check_edge_attr(edge_attr: Any, num_edges: int, edge_dim: int, ref: Tensor) -> Tensor | None:
    if edge_dim == 0:
        if edge_attr is not None and (not isinstance(edge_attr, Tensor) or edge_attr.numel() > 0):
            shape = tuple(edge_attr.shape) if isinstance(edge_attr, Tensor) else type(edge_attr).__name__
            raise ValueError(
                f"model was built with edge_dim=0 but edge_attr {shape} was given; "
                "rebuild the model with the right edge_dim or pass edge_attr=None"
            )
        return None
    if edge_attr is None:
        raise ValueError(f"edge_attr is required (model was built with edge_dim={edge_dim})")
    if not isinstance(edge_attr, Tensor):
        raise TypeError(f"edge_attr must be a torch tensor, got {type(edge_attr).__name__}")
    if edge_attr.dim() == 1 and edge_dim == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    if edge_attr.dim() != 2 or edge_attr.size(0) != num_edges:
        raise ValueError(
            f"edge_attr must be [E, edge_dim] with one row per edge: got {tuple(edge_attr.shape)} rows "
            f"for E={num_edges} edges"
        )
    if edge_attr.size(1) != edge_dim:
        raise ValueError(f"edge_attr has {edge_attr.size(1)} columns but the model expects edge_dim={edge_dim}")
    if not edge_attr.is_floating_point():
        raise TypeError(f"edge_attr must be floating point, got dtype {edge_attr.dtype}")
    edge_attr = edge_attr.to(device=ref.device, dtype=ref.dtype)
    _require_finite(edge_attr, "edge_attr")
    return edge_attr


def _check_hidden(hidden: Any, num_nodes: int, hidden_dim: int, ref: Tensor) -> Tensor | None:
    if hidden is None:
        return None
    if not isinstance(hidden, Tensor) or not hidden.is_floating_point():
        raise ValueError("hidden_state must be a floating-point tensor of shape [N, hidden_dim] or None")
    if tuple(hidden.shape) != (num_nodes, hidden_dim):
        raise ValueError(f"hidden_state must have shape ({num_nodes}, {hidden_dim}); got {tuple(hidden.shape)}")
    hidden = hidden.to(device=ref.device, dtype=ref.dtype)
    _require_finite(hidden, "hidden_state")
    return hidden


# --------------------------------------------------------------------------- graph helpers


def _replicate_graph(
    edge_index: Tensor, edge_attr: Tensor | None, num_nodes: int, copies: int
) -> tuple[Tensor, Tensor | None]:
    """Disjoint union of ``copies`` graphs, time-major (copy k uses node ids k*N .. k*N+N-1)."""
    if copies == 1:
        return edge_index, edge_attr
    offsets = torch.arange(copies, device=edge_index.device, dtype=edge_index.dtype) * num_nodes
    stacked = (edge_index.unsqueeze(1) + offsets.view(1, copies, 1)).reshape(2, -1)
    return stacked, (edge_attr.repeat(copies, 1) if edge_attr is not None else None)


def _gcn_norm(edge_index: Tensor, weight: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Symmetric GCN normalisation with unit self-loops: ``D^-1/2 (A + I) D^-1/2``.

    Input self-loops are dropped (like GATv2) and replaced by weight-1 loops; the degree
    is the weighted in-degree at the target node. Returns ``(src, dst, norm)``.
    """
    keep = edge_index[0] != edge_index[1]
    loops = torch.arange(num_nodes, device=edge_index.device)
    src = torch.cat([edge_index[0, keep], loops])
    dst = torch.cat([edge_index[1, keep], loops])
    weight = torch.cat([weight[keep], weight.new_ones(num_nodes)])
    degree = weight.new_zeros(num_nodes).index_add(0, dst, weight)
    inv_sqrt = degree.rsqrt()  # degree >= 1 thanks to the self-loop
    return src, dst, inv_sqrt[src] * weight * inv_sqrt[dst]


def _propagate(x: Tensor, src: Tensor, dst: Tensor, norm: Tensor) -> Tensor:
    """Weighted neighbourhood sum ``out[dst] += norm * x[src]`` for node features ``x [N, D]``.

    Gather + ``index_add`` (measured ~10x faster than ``torch.sparse.mm`` on CPU, and
    differentiable with respect to both ``x`` and the edge weights).
    """
    messages = x.index_select(0, src) * norm.unsqueeze(-1)
    return x.new_zeros(x.shape).index_add(0, dst, messages)


# --------------------------------------------------------------------------- shared base


class _FloodSequenceModel(nn.Module):
    """Validation, chunking policy and configuration shared by both architectures."""

    architecture: ClassVar[str] = ""

    def __init__(
        self,
        node_in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float,
        spatial_chunk: int,
        max_chunk_edges: int,
        gradient_checkpointing: bool | str,
    ) -> None:
        super().__init__()
        self.node_in_dim = _int_param("node_in_dim", node_in_dim, 1)
        self.edge_dim = _int_param("edge_dim", edge_dim, 0)
        self.hidden_dim = _int_param("hidden_dim", hidden_dim, 1)
        self.dropout_rate = _dropout_param(dropout)
        self.spatial_chunk = _int_param("spatial_chunk", spatial_chunk, 1)
        self.max_chunk_edges = _int_param("max_chunk_edges", max_chunk_edges, 1)
        self.gradient_checkpointing = _checkpoint_param(gradient_checkpointing)

    # ------------------------------------------------------------------ public API

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """One hour: ``(probs [N, 1], hidden [N, hidden_dim])``."""
        logits, hidden = self.forward_logits(x, edge_index, edge_attr, hidden_state)
        return torch.sigmoid(logits), hidden

    def forward_logits(  # pragma: no cover - implemented by subclasses
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward_sequence(
        self,
        x_seq: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        hidden_state: Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Whole window: ``x_seq [T, N, F]`` → ``(logits or probs [T, N], h_last [N, hidden_dim])``.

        ``T == 0`` returns an empty ``[0, N]`` output and the initial state (zeros if None).
        """
        x_seq, edge_index, edge_attr, hidden = self._prepare(x_seq, edge_index, edge_attr, hidden_state, True)
        steps, num_nodes, _ = x_seq.shape
        if hidden is None:
            hidden = x_seq.new_zeros(num_nodes, self.hidden_dim)
        if steps == 0:
            return x_seq.new_zeros(0, num_nodes), hidden
        logits, h_last = self._sequence_logits(x_seq, edge_index, edge_attr, hidden)
        return (logits if return_logits else torch.sigmoid(logits)), h_last

    def effective_chunk(self, steps: int, num_nodes: int, num_edges: int) -> int:
        """Hours per stacked spatial call: ``min(spatial_chunk, T, max_chunk_edges // (E + N))``, >= 1."""
        per_step = max(num_edges + num_nodes, 1)
        cap = self.max_chunk_edges // per_step
        return max(1, min(self.spatial_chunk, steps, cap))

    def uses_checkpointing(self, steps: int, num_nodes: int, num_edges: int) -> bool:
        """Whether a sequence of this size is gradient-checkpointed (activations recomputed in backward)."""
        if not torch.is_grad_enabled():
            return False
        if self.gradient_checkpointing == "auto":
            return steps * (num_edges + num_nodes) >= AUTO_CHECKPOINT_EDGE_STEPS
        return bool(self.gradient_checkpointing)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments (plus ``architecture``); ``build_model(get_config())`` rebuilds the model."""
        return {
            "architecture": self.architecture,
            "node_in_dim": self.node_in_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "spatial_chunk": self.spatial_chunk,
            "max_chunk_edges": self.max_chunk_edges,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    # ------------------------------------------------------------------ internals

    def _sequence_logits(  # pragma: no cover - implemented by subclasses
        self, x_seq: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _prepare(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        hidden_state: Tensor | None,
        sequence: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Validate shapes/dtypes/values and move every input to the model's device and dtype."""
        ref = next(self.parameters())
        x = _check_node_features(x, sequence=sequence, in_dim=self.node_in_dim, ref=ref)
        num_nodes = x.size(-2)
        edge_index = _check_edge_index(edge_index, num_nodes, ref)
        edge_attr = _check_edge_attr(edge_attr, edge_index.size(1), self.edge_dim, ref)
        hidden = _check_hidden(hidden_state, num_nodes, self.hidden_dim, ref)
        return x, edge_index, edge_attr, hidden


# --------------------------------------------------------------------------- spec architecture


class SpatioTemporalFloodGNN(_FloodSequenceModel):
    """Spec model: GATv2(F→H, heads, concat) → ELU → dropout → GATv2(H·heads→H) → ELU → GRUCell → MLP.

    Args:
        node_in_dim: node feature count = static features + precipitation + rolling sums (spec default 8;
            the project config uses 10: 5 static + 1 + 4).
        edge_dim: edge feature count (2 = length, grade); 0 disables edge features.
        hidden_dim: GATv2 / GRU width.
        dropout: dropout after the first GATv2 layer and inside the head.
        heads: attention heads of the first GATv2 layer (concatenated).
        spatial_chunk, max_chunk_edges, gradient_checkpointing: ``forward_sequence``
            performance knobs (see module docstring); they do not change results.
    """

    architecture: ClassVar[str] = "gatv2_gru"

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        heads: int = 2,
        *,
        spatial_chunk: int = DEFAULT_SPATIAL_CHUNK,
        max_chunk_edges: int = DEFAULT_MAX_CHUNK_EDGES,
        gradient_checkpointing: bool | str = "auto",
    ) -> None:
        super().__init__(
            node_in_dim, edge_dim, hidden_dim, dropout, spatial_chunk, max_chunk_edges, gradient_checkpointing
        )
        self.heads = _int_param("heads", heads, 1)
        gat_edge_dim = self.edge_dim or None
        # Registration order matches the spec so equal seeds give identical initial weights.
        self.conv1 = GATv2Conv(self.node_in_dim, self.hidden_dim, edge_dim=gat_edge_dim, heads=self.heads, concat=True)
        self.conv2 = GATv2Conv(
            self.hidden_dim * self.heads, self.hidden_dim, edge_dim=gat_edge_dim, heads=1, concat=False
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, HEAD_HIDDEN_DIM)
        self.fc2 = nn.Linear(HEAD_HIDDEN_DIM, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "heads": self.heads}

    def forward_logits(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """One hour, exactly as the spec's ``forward`` but returning logits ``[N, 1]``."""
        x, edge_index, edge_attr, hidden = self._prepare(x, edge_index, edge_attr, hidden_state, False)
        spatial = self._spatial(x, edge_index, edge_attr)
        if hidden is None:
            hidden = torch.zeros_like(spatial)
        h_temporal = self.gru(spatial, hidden)
        return self._head(h_temporal), h_temporal

    def _spatial(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None) -> Tensor:
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = self.dropout(h)
        return F.elu(self.conv2(h, edge_index, edge_attr))

    def _head(self, h: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(h))))

    def _sequence_logits(
        self, x_seq: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Process ``chunk`` hours at a time; with checkpointing only chunk-boundary states are kept."""
        steps, num_nodes, _ = x_seq.shape
        num_edges = edge_index.size(1)
        chunk = self.effective_chunk(steps, num_nodes, num_edges)
        recompute = self.uses_checkpointing(steps, num_nodes, num_edges)
        LOGGER.debug(
            "forward_sequence: T=%d N=%d E=%d chunk=%d checkpointing=%s", steps, num_nodes, num_edges, chunk, recompute
        )
        stacked_graphs: dict[int, tuple[Tensor, Tensor | None]] = {}
        logits: list[Tensor] = []
        for start in range(0, steps, chunk):
            size = min(chunk, steps - start)
            if size not in stacked_graphs:
                stacked_graphs[size] = _replicate_graph(edge_index, edge_attr, num_nodes, size)
            args = (x_seq[start : start + size], hidden, *stacked_graphs[size])
            if recompute:
                chunk_logits, hidden = checkpoint(self._chunk_forward, *args, use_reentrant=False)
            else:
                chunk_logits, hidden = self._chunk_forward(*args)
            logits.append(chunk_logits)
        return (logits[0] if len(logits) == 1 else torch.cat(logits, dim=0)), hidden

    def _chunk_forward(
        self, x_chunk: Tensor, hidden: Tensor, stacked_index: Tensor, stacked_attr: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """``c`` hours: one GATv2 call on the time-stacked graph, then the GRU loop and head."""
        size, num_nodes, in_dim = x_chunk.shape
        flat = x_chunk.reshape(size * num_nodes, in_dim)
        spatial = self._spatial(flat, stacked_index, stacked_attr).view(size, num_nodes, -1)
        states = []
        for t in range(size):
            hidden = self.gru(spatial[t], hidden)
            states.append(hidden)
        return self._head(torch.stack(states)).squeeze(-1), hidden


# --------------------------------------------------------------------------- A3T-GCN alternative


class A3TGCNFlood(_FloodSequenceModel):
    """T-GCN cell with learned edge weights + causal additive temporal attention (A3T-GCN).

    Per hour ``t`` with ``Â = D^-1/2 (W + I) D^-1/2`` (``W`` = sigmoid of a linear map of
    the edge features, so e.g. steep downhill segments can carry more weight)::

        [z, r] = σ(Â x_t W_x^{zr} + Â h W_h^{zr} + b)      c = tanh(Â x_t W_x^c + Â (r ⊙ h) W_h^c + b)
        h_t = z ⊙ h + (1 − z) ⊙ c
        e_t = vᵀ tanh(W_a h_t + b_a)          ctx_t = Σ_{s ≤ t} softmax_s(e_s) h_s   (causal)
        logit_t = fc2(dropout(relu(fc1(dropout([h_t, ctx_t])))))

    The attention covers the hidden states produced within one ``forward_sequence`` call
    (online, max-stabilised softmax — exact and O(T)). A single-step ``forward`` therefore
    attends only to its own state; use ``forward_sequence`` for windows.
    The recurrence cannot be batched over time (every graph convolution sees ``h``), so
    it runs hour by hour and ``spatial_chunk`` / ``max_chunk_edges`` are accepted only for
    interface parity. With checkpointing (same ``"auto"`` policy as the spec model) each
    hour keeps just its carried state for backward and is recomputed there (a 16-window
    batch: 4.5 GB → < 1.5 GB peak, and faster).
    """

    architecture: ClassVar[str] = "a3t_gcn"

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        *,
        spatial_chunk: int = DEFAULT_SPATIAL_CHUNK,
        max_chunk_edges: int = DEFAULT_MAX_CHUNK_EDGES,
        gradient_checkpointing: bool | str = "auto",
    ) -> None:
        super().__init__(
            node_in_dim, edge_dim, hidden_dim, dropout, spatial_chunk, max_chunk_edges, gradient_checkpointing
        )
        width = self.hidden_dim
        self.edge_gate = nn.Linear(self.edge_dim, 1) if self.edge_dim > 0 else None
        self.input_proj = nn.Linear(self.node_in_dim, 3 * width)
        self.hidden_gates = nn.Linear(width, 2 * width, bias=False)
        self.hidden_cand = nn.Linear(width, width, bias=False)
        self.att_proj = nn.Linear(width, width)
        self.att_score = nn.Linear(width, 1, bias=False)
        self.fc1 = nn.Linear(2 * width, HEAD_HIDDEN_DIM)
        self.fc2 = nn.Linear(HEAD_HIDDEN_DIM, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward_logits(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """One hour (attention over this hour's state only): ``(logits [N, 1], h [N, H])``."""
        x, edge_index, edge_attr, hidden = self._prepare(x, edge_index, edge_attr, hidden_state, False)
        if hidden is None:
            hidden = x.new_zeros(x.size(0), self.hidden_dim)
        logits, h_last = self._sequence_logits(x.unsqueeze(0), edge_index, edge_attr, hidden)
        return logits[0].unsqueeze(-1), h_last

    def _sequence_logits(
        self, x_seq: Tensor, edge_index: Tensor, edge_attr: Tensor | None, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        steps, num_nodes, _ = x_seq.shape
        weight = (
            torch.sigmoid(self.edge_gate(edge_attr)).squeeze(-1)
            if self.edge_gate is not None
            else x_seq.new_ones(edge_index.size(1))
        )
        src, dst, norm = _gcn_norm(edge_index, weight, num_nodes)
        recompute = self.uses_checkpointing(steps, num_nodes, edge_index.size(1))
        run_max = x_seq.new_full((num_nodes, 1), float("-inf"))
        numer = x_seq.new_zeros(num_nodes, self.hidden_dim)
        denom = x_seq.new_zeros(num_nodes, 1)
        logits = []
        for t in range(steps):
            carry = (x_seq[t], hidden, run_max, numer, denom, src, dst, norm)
            if recompute:  # keep only the carried state per hour; recompute the hour in backward
                logit, hidden, run_max, numer, denom = checkpoint(self._step, *carry, use_reentrant=False)
            else:
                logit, hidden, run_max, numer, denom = self._step(*carry)
            logits.append(logit)
        return torch.stack(logits), hidden

    def _step(
        self,
        x: Tensor,
        hidden: Tensor,
        run_max: Tensor,
        numer: Tensor,
        denom: Tensor,
        src: Tensor,
        dst: Tensor,
        norm: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One hour: T-GCN update, online (max-stabilised) causal softmax, head → ``(logit [N], state...)``."""
        hidden = self._cell(self.input_proj(_propagate(x, src, dst, norm)), hidden, src, dst, norm)
        score = self.att_score(torch.tanh(self.att_proj(hidden)))
        new_max = torch.maximum(run_max, score.detach())  # stabiliser only: result is invariant to it
        decay, weight = torch.exp(run_max - new_max), torch.exp(score - new_max)
        numer = numer * decay + weight * hidden
        denom = denom * decay + weight
        head_in = self.dropout(torch.cat([hidden, numer / denom], dim=-1))
        logit = self.fc2(self.dropout(F.relu(self.fc1(head_in)))).squeeze(-1)
        return logit, hidden, new_max, numer, denom

    def _cell(self, input_gates: Tensor, hidden: Tensor, src: Tensor, dst: Tensor, norm: Tensor) -> Tensor:
        width = self.hidden_dim
        gates = torch.sigmoid(input_gates[:, : 2 * width] + self.hidden_gates(_propagate(hidden, src, dst, norm)))
        update, reset = gates.chunk(2, dim=-1)
        candidate = torch.tanh(
            input_gates[:, 2 * width :] + self.hidden_cand(_propagate(reset * hidden, src, dst, norm))
        )
        return update * hidden + (1.0 - update) * candidate


# --------------------------------------------------------------------------- factory & utilities


def _normalise_architecture(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in _ARCHITECTURE_ALIASES:
        raise ValueError(f"Unknown model architecture {value!r}; expected one of {ARCHITECTURES}")
    return _ARCHITECTURE_ALIASES[key]


def build_model(model_cfg: Mapping[str, Any] | None) -> nn.Module:
    """Build the model described by the ``model`` config section (missing keys → defaults).

    A full project config (with a ``model`` section) is accepted too. Training-only keys
    (learning_rate, batch_size, …) are ignored, as is ``heads`` for ``a3t_gcn``. Invalid
    dimensions or an unknown ``architecture`` raise ``ValueError``.
    """
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, Mapping):
        raise ValueError(f"model config must be a mapping, got {type(model_cfg).__name__}")
    if "architecture" not in model_cfg and isinstance(model_cfg.get("model"), Mapping):
        model_cfg = model_cfg["model"]
    cfg = deep_merge(MODEL_DEFAULTS, model_cfg)
    architecture = _normalise_architecture(cfg["architecture"])
    common = {
        key: cfg[key]
        for key in (
            "node_in_dim",
            "edge_dim",
            "hidden_dim",
            "dropout",
            "spatial_chunk",
            "max_chunk_edges",
            "gradient_checkpointing",
        )
    }
    if architecture == "gatv2_gru":
        model: nn.Module = SpatioTemporalFloodGNN(heads=cfg["heads"], **common)
    else:
        model = A3TGCNFlood(**common)
    LOGGER.info("Built %s model with %d trainable parameters", architecture, count_parameters(model))
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable, by default) scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put ``model`` in eval mode but keep its dropout layers stochastic (Monte-Carlo dropout)."""
    model.eval()
    active = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
            active += int(module.p > 0)
    if active == 0:
        LOGGER.warning("enable_mc_dropout: no dropout layer with p > 0; MC dropout samples will be identical")
