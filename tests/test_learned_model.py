"""Tests for placer.learned: the graph-tensor conversion, GNN building blocks, and place() pipeline.

`model.py`'s `PlacementGNN` is no longer part of the active `place()` pipeline (see placer.py's
module docstring for the pivot to per-board analytic optimization) but is kept in the repo, so its
own shape/forward-pass tests stay here as unit tests of that standalone code, not of place() itself.
"""

import numpy as np
import torch

from placer.feasibility import check_feasibility
from placer.generator import GeneratorConfig, generate_board
from placer.learned.loss import ProxyLossNormalizer, proxy_cost
from placer.learned.model import PlacementGNN, board_to_tensors
from placer.learned.placer import _optimize_positions, place


def _small_board(seed=0, num_components=50):
    return generate_board(GeneratorConfig(num_components=num_components, seed=seed))


def test_board_to_tensors_shapes():
    board = _small_board()
    graph = board_to_tensors(board)

    n = len(board.components)
    m = len(board.nets)
    assert graph.comp_feats.shape == (n, 4)
    assert graph.net_feats.shape == (m, 2)
    assert graph.net_weights.shape == (m,)
    assert graph.widths.shape == (n,)
    assert graph.heights.shape == (n,)
    assert graph.edge_comp_idx.shape == graph.edge_net_idx.shape
    assert torch.all(graph.edge_comp_idx < n)
    assert torch.all(graph.edge_net_idx < m)


def test_forward_pass_produces_finite_bounded_positions():
    board = _small_board()
    graph = board_to_tensors(board)

    model = PlacementGNN(hidden_dim=16, num_layers=2)
    with torch.no_grad():
        pred = model(graph)

    assert pred.shape == (len(board.components), 2)
    assert torch.isfinite(pred).all()
    assert torch.all(pred >= 0.0) and torch.all(pred <= 1.0)


def test_analytic_optimization_reduces_proxy_loss():
    """The new per-board optimizer (placer.py's pivot away from the GNN) should actually move
    positions toward lower proxy cost, not just run without crashing."""
    board = _small_board(seed=3, num_components=50)
    rng = np.random.default_rng(3)
    graph = board_to_tensors(board)

    init = torch.tensor(
        np.stack([rng.uniform(0, board.width, len(graph.component_ids)), rng.uniform(0, board.height, len(graph.component_ids))], axis=1),
        dtype=torch.float32,
    )
    normalizer = ProxyLossNormalizer(warmup_steps=100)
    with torch.no_grad():
        first_loss = proxy_cost(
            init, graph.widths, graph.heights, graph.edge_comp_idx, graph.edge_net_idx, graph.net_weights,
            num_nets=graph.net_feats.shape[0], board_width=board.width, board_height=board.height, normalizer=normalizer,
        ).item()

    rng2 = np.random.default_rng(3)
    final_pos = _optimize_positions(board, graph, rng2, time_budget=5.0)
    final = torch.tensor(final_pos, dtype=torch.float32)
    with torch.no_grad():
        last_loss = proxy_cost(
            final, graph.widths, graph.heights, graph.edge_comp_idx, graph.edge_net_idx, graph.net_weights,
            num_nets=graph.net_feats.shape[0], board_width=board.width, board_height=board.height, normalizer=normalizer,
        ).item()

    assert last_loss < first_loss
    assert not np.allclose(final_pos, init.numpy()), "positions should have actually moved"


def test_place_produces_a_feasible_board():
    """place() must return a feasible board even with a tiny time budget -- feasibility comes from
    legalize()/SA structurally, not from how well-optimized the analytic placement phase was."""
    board = _small_board(seed=2, num_components=50)
    placed = place(board, seed=2, time_budget=3.0)
    report = check_feasibility(placed)
    assert report.is_feasible, report.summary()
