"""Verify IncrementalCost's running total always agrees with a from-scratch compute_cost().

This is the most important correctness test in the project: simulated
annealing runs entirely off IncrementalCost's running total for speed, so if
it silently drifted from the true cost function, every downstream result
(baseline numbers, the learned placer's benchmark comparison) would be
wrong without any error being raised.
"""

import numpy as np
import pytest

from placer.board import VALID_ROTATIONS
from placer.cost import OVERLAP_WEIGHT, compute_cost
from placer.generator import GeneratorConfig, generate_board
from placer.incremental_cost import IncrementalCost


def _random_placement(board, rng):
    for c in board.components:
        c.rotation = int(rng.choice(VALID_ROTATIONS))
        w, h = c.footprint()
        c.x = float(rng.uniform(0, max(board.width - w, 0.001)))
        c.y = float(rng.uniform(0, max(board.height - h, 0.001)))
        c.layer = "top" if rng.random() < 0.5 else "bottom"


def test_incremental_matches_full_recompute_after_moves():
    rng = np.random.default_rng(7)
    board = generate_board(GeneratorConfig(num_components=50, seed=7))
    _random_placement(board, rng)

    inc = IncrementalCost(board)
    full = compute_cost(board)
    assert inc.total == pytest.approx(full.total, rel=1e-6)
    assert inc.wirelength_total == pytest.approx(full.wirelength, rel=1e-6)
    assert inc.congestion_total == pytest.approx(full.congestion, rel=1e-6)

    for _ in range(200):
        c = board.components[rng.integers(0, len(board.components))]
        old_bbox = c.bbox()
        old_layer = c.layer
        c.rotation = int(rng.choice(VALID_ROTATIONS))
        w, h = c.footprint()
        c.x = float(rng.uniform(0, max(board.width - w, 0.001)))
        c.y = float(rng.uniform(0, max(board.height - h, 0.001)))
        c.layer = "top" if rng.random() < 0.5 else "bottom"
        inc.component_moved(c, old_bbox, old_layer)

    full_after = compute_cost(board)
    assert inc.total == pytest.approx(full_after.total, rel=1e-6)
    assert inc.wirelength_total == pytest.approx(full_after.wirelength, rel=1e-6)
    assert OVERLAP_WEIGHT * inc.overlap_total == pytest.approx(full_after.overlap, rel=1e-6)
    assert inc.congestion_total == pytest.approx(full_after.congestion, rel=1e-6)
