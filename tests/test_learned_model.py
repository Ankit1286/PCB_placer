"""Tests for placer.learned: the GNN warm-start model, proxy loss, and place() pipeline.

Uses tiny boards and few steps -- this suite checks wiring/gradients, not placement quality (that's
scripts/benchmark.py's job, once a real checkpoint has been trained).
"""

import torch

from placer.feasibility import check_feasibility
from placer.generator import GeneratorConfig, generate_board
from placer.learned.loss import ProxyLossNormalizer, proxy_cost
from placer.learned.model import PlacementGNN, board_to_tensors
from placer.learned.placer import place


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


def test_training_step_reduces_proxy_loss():
    board = _small_board(seed=1, num_components=50)
    graph = board_to_tensors(board)
    board_dims = torch.tensor([board.width, board.height], dtype=torch.float32)

    model = PlacementGNN(hidden_dim=16, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    normalizer = ProxyLossNormalizer()

    def _current_loss():
        positions = model(graph) * board_dims
        return proxy_cost(
            positions,
            graph.widths,
            graph.heights,
            graph.edge_comp_idx,
            graph.edge_net_idx,
            graph.net_weights,
            num_nets=graph.net_feats.shape[0],
            board_width=board.width,
            board_height=board.height,
            normalizer=normalizer,
        )

    first_loss = _current_loss().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = _current_loss()
        loss.backward()
        optimizer.step()
    last_loss = _current_loss().item()

    assert last_loss < first_loss


def test_place_produces_a_feasible_board():
    """place() must return a feasible board even with an untrained (randomly-initialized) model --
    feasibility comes from legalize()/SA, not from placement quality, so no checkpoint is needed."""
    board = _small_board(seed=2, num_components=50)
    placed = place(board, seed=2, checkpoint_path="__no_such_checkpoint__.pt", time_budget=3.0)
    report = check_feasibility(placed)
    assert report.is_feasible, report.summary()
