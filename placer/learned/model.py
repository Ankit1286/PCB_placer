"""GNN warm-start model: bipartite component<->net message passing.

Predicts each component's absolute position directly from graph structure
(component sizes, pin counts, net connectivity) -- no force-directed init or
other spatial prior involved. An earlier version predicted a residual on top
of `baseline.py::_force_directed_init`'s output, but profiling showed that
init dominating training-step time (95%+ at |V|=300, and it gets worse:
~3.1s of a ~3.2s step -- the GNN's own forward/backward pass is 5-16ms
regardless of board size). Dropping it removes both that bottleneck and the
residual's implicit displacement cap, at the cost of the GNN no longer
having a "reasonable starting point" to fall back on -- gradient descent has
to do all the spreading, the same way analytic global placement methods
(e.g. DREAMPlace-style) typically initialize with no spatial prior either.

Rotation and layer are deliberately not modeled here -- they're assigned
afterward by the existing, tested `_greedy_layer_assignment`/`_snap_rotations`
heuristics (see `placer/learned/placer.py`), which keeps this model's
training loss purely continuous and avoids the differentiability headache of
discrete choices.

Message passing is hand-rolled with `index_add_`-based scatter-mean rather
than via `torch_geometric`: boards top out at |V|=1000 components / ~2000
nets, small enough that plain tensor ops are simple and sufficient, and it
avoids a new dependency beyond the `torch` already in requirements.txt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from placer.board import Board

COMP_FEAT_DIM = 4  # width, height, log(pin_count), log1p(degree) (all normalized/log-scaled)
NET_FEAT_DIM = 2  # log(weight), arity


@dataclass
class GraphTensors:
    """Tensor form of one board's component<->net bipartite graph, for GNN input/loss."""

    comp_feats: torch.Tensor  # (num_components, COMP_FEAT_DIM)
    net_feats: torch.Tensor  # (num_nets, NET_FEAT_DIM)
    edge_comp_idx: torch.Tensor  # (num_edges,) long, index into comp_feats/component_ids
    edge_net_idx: torch.Tensor  # (num_edges,) long, index into net_feats
    net_weights: torch.Tensor  # (num_nets,) raw omega_e, for the wirelength loss term
    widths: torch.Tensor  # (num_components,) raw component width, for the overlap loss term
    heights: torch.Tensor  # (num_components,) raw component height
    component_ids: list[str]  # component_ids[i] is the id of the component at row i


def board_to_tensors(board: Board) -> GraphTensors:
    """Build graph tensors from a board's fixed properties -- components need not be placed yet.

    Unlike an earlier version, this reads only size/pin-count/connectivity (never x/y), so it works
    directly on a freshly-`generate_board`'d, still-unplaced board -- no initializer required first.
    """
    components = board.components
    id_to_idx = {c.component_id: i for i, c in enumerate(components)}

    degree: dict[str, int] = {c.component_id: 0 for c in components}
    for net in board.nets.values():
        for cid, _ in net.pin_refs:
            degree[cid] += 1

    comp_rows, widths, heights = [], [], []
    for c in components:
        comp_rows.append(
            [
                c.width / board.width,
                c.height / board.height,
                float(np.log(len(c.pins))),
                float(np.log1p(degree[c.component_id])),
            ]
        )
        widths.append(c.width)
        heights.append(c.height)

    net_rows, net_weights, edge_comp, edge_net = [], [], [], []
    for net_idx, net in enumerate(board.nets.values()):
        net_rows.append([float(np.log(net.weight)), float(len(net.pin_refs))])
        net_weights.append(net.weight)
        seen: set[int] = set()
        for cid, _ in net.pin_refs:
            ci = id_to_idx[cid]
            if ci in seen:
                continue
            seen.add(ci)
            edge_comp.append(ci)
            edge_net.append(net_idx)

    return GraphTensors(
        comp_feats=torch.tensor(comp_rows, dtype=torch.float32),
        net_feats=torch.tensor(net_rows, dtype=torch.float32) if net_rows else torch.zeros((0, NET_FEAT_DIM)),
        edge_comp_idx=torch.tensor(edge_comp, dtype=torch.long),
        edge_net_idx=torch.tensor(edge_net, dtype=torch.long),
        net_weights=torch.tensor(net_weights, dtype=torch.float32),
        widths=torch.tensor(widths, dtype=torch.float32),
        heights=torch.tensor(heights, dtype=torch.float32),
        component_ids=[c.component_id for c in components],
    )


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, num_targets: int) -> torch.Tensor:
    """Mean-aggregate `values` (rows) into `num_targets` buckets given by `index`, via index_add_.

    A target bucket with no incoming edges gets an all-zero embedding (nothing to aggregate).
    """
    dim = values.shape[1]
    summed = torch.zeros((num_targets, dim), dtype=values.dtype, device=values.device)
    summed.index_add_(0, index, values)
    counts = torch.zeros((num_targets, 1), dtype=values.dtype, device=values.device)
    counts.index_add_(0, index, torch.ones((index.shape[0], 1), dtype=values.dtype, device=values.device))
    return summed / counts.clamp(min=1.0)


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class PlacementGNN(nn.Module):
    """Bipartite component<->net GNN predicting each component's absolute position.

    `num_layers` rounds of message passing let information travel component -> its nets -> other
    components sharing those nets -> ... (a few hops), so the model can learn board-wide netlist
    structure a purely-local method can't (e.g. clustering components that are only indirectly
    connected through a shared net). There's no spatial input at all -- position is derived purely
    from size/pin-count/connectivity features, so an untrained model places everything near the
    board center (where `sigmoid` sits at zero input) rather than near any particular heuristic.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.comp_encoder = _mlp(COMP_FEAT_DIM, hidden_dim, hidden_dim)
        self.net_encoder = _mlp(NET_FEAT_DIM, hidden_dim, hidden_dim)
        self.comp_updates = nn.ModuleList([_mlp(hidden_dim * 2, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.net_updates = nn.ModuleList([_mlp(hidden_dim * 2, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.position_head = _mlp(hidden_dim, 2, hidden_dim)

    def forward(self, g: GraphTensors) -> torch.Tensor:
        """Return predicted normalized positions (num_components, 2), each coordinate in [0, 1]."""
        num_comp = g.comp_feats.shape[0]
        num_net = g.net_feats.shape[0]

        comp_emb = self.comp_encoder(g.comp_feats)
        net_emb = self.net_encoder(g.net_feats) if num_net > 0 else torch.zeros((0, self.hidden_dim))

        if g.edge_comp_idx.numel() > 0:
            for layer in range(self.num_layers):
                comp_to_net = _scatter_mean(comp_emb[g.edge_comp_idx], g.edge_net_idx, num_net)
                net_emb = net_emb + self.net_updates[layer](torch.cat([net_emb, comp_to_net], dim=1))

                net_to_comp = _scatter_mean(net_emb[g.edge_net_idx], g.edge_comp_idx, num_comp)
                comp_emb = comp_emb + self.comp_updates[layer](torch.cat([comp_emb, net_to_comp], dim=1))

        return torch.sigmoid(self.position_head(comp_emb))
