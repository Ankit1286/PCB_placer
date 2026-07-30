"""Tests for placer.baseline.

Uses small boards and a short SA run (not the full 50,000 iterations --
that's for scripts/benchmark.py, not the test suite) so this stays fast.
"""

import numpy as np
import pytest

from placer.baseline import (
    _force_directed_init,
    _greedy_layer_assignment,
    _snap_rotations,
    place_baseline,
    simulated_annealing,
)
from placer.cost import compute_cost
from placer.feasibility import check_feasibility
from placer.generator import GeneratorConfig, generate_board
from placer.legalize import legalize


def _small_board(seed=0, num_components=50):
    return generate_board(GeneratorConfig(num_components=num_components, seed=seed))


def test_force_directed_init_produces_valid_finite_positions():
    board = _small_board()
    rng = np.random.default_rng(0)
    _force_directed_init(board, rng, iterations=50)
    for c in board.components:
        assert c.is_placed
        assert np.isfinite(c.x) and np.isfinite(c.y)


def test_greedy_layer_assignment_is_balanced_not_collapsed():
    board = _small_board()
    rng = np.random.default_rng(0)
    _force_directed_init(board, rng, iterations=50)
    _greedy_layer_assignment(board)
    top = sum(1 for c in board.components if c.layer == "top")
    bottom = sum(1 for c in board.components if c.layer == "bottom")
    assert top > 0 and bottom > 0
    # Shouldn't be wildly lopsided given the balance term.
    assert min(top, bottom) / len(board.components) > 0.25


def test_snap_rotations_does_not_increase_overlap():
    board = _small_board()
    rng = np.random.default_rng(0)
    _force_directed_init(board, rng, iterations=50)
    _greedy_layer_assignment(board)
    from placer.cost import total_overlap_cost

    before = total_overlap_cost(board.components)
    _snap_rotations(board, rng)
    after = total_overlap_cost(board.components)
    assert after <= before + 1e-6


def test_legalize_after_init_leaves_no_keepout_violations():
    board = _small_board()
    rng = np.random.default_rng(0)
    _force_directed_init(board, rng, iterations=50)
    _greedy_layer_assignment(board)
    _snap_rotations(board, rng)
    legalize(board)
    report = check_feasibility(board)
    assert report.in_keepout == []


def test_short_sa_run_reduces_cost_and_stays_feasible():
    board = _small_board()
    rng = np.random.default_rng(0)
    _force_directed_init(board, rng, iterations=50)
    _greedy_layer_assignment(board)
    _snap_rotations(board, rng)
    legalize(board)

    cost_before = compute_cost(board).total
    cost_state, history = simulated_annealing(board, rng, iterations=300, record_history=True)
    cost_after_incremental = cost_state.total
    cost_after_full = compute_cost(board).total

    assert cost_after_incremental == pytest.approx(cost_after_full, rel=1e-6)
    assert cost_after_incremental <= cost_before  # SA should never make the best-seen-so-far worse on net

    report = check_feasibility(board)
    assert report.in_keepout == []
    assert report.bad_rotation == []
    assert report.out_of_bounds == []
    assert len(history.costs) == 300


def test_sa_never_leaves_a_component_centered_in_a_keepout():
    """Regression test: SA perturbations must not silently violate the keepout hard constraint.

    C(p) has no keepout term, so this can only be true if perturbations
    actively push components back out (see _apply_random_perturbation) --
    without that, a jitter move landing in a keepout could be accepted since
    cost wouldn't object.
    """
    board = _small_board(seed=3, num_components=80)
    rng = np.random.default_rng(3)
    _force_directed_init(board, rng, iterations=50)
    _greedy_layer_assignment(board)
    _snap_rotations(board, rng)
    legalize(board)
    simulated_annealing(board, rng, iterations=1000)
    report = check_feasibility(board)
    assert report.in_keepout == []


def test_place_baseline_end_to_end_is_fully_feasible():
    """Regression test: a full place_baseline() run must be feasible, including zero overlap.

    A prior version left 3 same-layer pairs overlapping after a full
    50,000-iteration run on a 60-component board: overlap is weighted 100x
    in C(p), but that's a finite weight, not an infinite barrier, so very
    low end-of-schedule temperatures can freeze in a residual overlap that
    would need a net cost increase to escape. place_baseline now runs a
    final legalize() pass specifically to backstop this. Uses a reduced SA
    iteration count so this test stays fast while still exercising that
    final legalize step (which runs regardless of how many SA iterations
    preceded it).
    """
    board = _small_board(seed=0, num_components=60)
    board, cost_state, _ = place_baseline(board, seed=0, sa_iterations=2000)
    report = check_feasibility(board)
    assert report.is_feasible, report.summary()
    assert cost_state.total == pytest.approx(compute_cost(board).total, rel=1e-6)
