"""Day-1 validation spike: does a warm start + budgeted SA actually beat a from-scratch full SA run?

This is the load-bearing assumption behind the whole "learned placer" plan
(see DESIGN.md): since the baseline is exempt from the 60s time budget but
the learned placer isn't, the only way to win is to start SA from a much
better point so far fewer refinement iterations are needed. Before spending
Day 2 training a GNN on that premise, this script checks it with the
cheapest possible substitute for "a good warm start": a naive, net-unaware
shelf-packing initializer. Two outcomes matter:

  - If [naive init + budgeted SA] already comes *close to or beats* the full
    baseline, that's good news for the overall approach's feasibility, but
    it also raises the bar for the GNN -- it has to add real value over and
    above "any reasonable warm start," not just "any warm start at all."
  - If [naive init + budgeted SA] falls well short, that confirms a smart
    (learned) warm start is doing real work, not just any warm start.

Either way this is a quick, informative experiment before committing to
GNN training.
"""

import time

import numpy as np

from placer.baseline import place_baseline, simulated_annealing
from placer.cost import compute_cost
from placer.feasibility import check_feasibility
from placer.generator import GeneratorConfig, generate_board
from placer.legalize import legalize


def naive_shelf_pack(board, rng):
    """Deterministic, net-unaware shelf (row) packing -- a cheap stand-in for "any reasonable warm start"."""
    components = sorted(board.components, key=lambda c: -c.height)
    x_cursor, y_cursor, row_height = 0.0, 0.0, 0.0
    for i, c in enumerate(components):
        if x_cursor + c.width > board.width:
            x_cursor = 0.0
            y_cursor += row_height
            row_height = 0.0
        c.x, c.y, c.rotation = x_cursor, y_cursor, 0
        c.layer = "top" if i % 2 == 0 else "bottom"
        x_cursor += c.width
        row_height = max(row_height, c.height)
    legalize(board)


def run_spike(num_components: int, refine_seconds: float, seed: int = 0) -> None:
    print(f"\n=== |V|={num_components}, refinement budget={refine_seconds}s ===")

    board_a = generate_board(GeneratorConfig(num_components=num_components, seed=seed))
    t0 = time.perf_counter()
    _, baseline_cost_state, _ = place_baseline(board_a, seed=seed)
    baseline_time = time.perf_counter() - t0
    baseline_feasible = check_feasibility(board_a).is_feasible
    print(f"[full baseline]      cost={baseline_cost_state.total:,.1f}  time={baseline_time:.1f}s  feasible={baseline_feasible}")

    board_b = generate_board(GeneratorConfig(num_components=num_components, seed=seed))
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    naive_shelf_pack(board_b, rng)
    naive_init_cost = compute_cost(board_b).total
    naive_init_time = time.perf_counter() - t0
    print(f"[naive init only]    cost={naive_init_cost:,.1f}  time={naive_init_time:.1f}s")

    t0 = time.perf_counter()
    cost_state, _ = simulated_annealing(board_b, rng, iterations=1_000_000, max_seconds=refine_seconds)
    budgeted_time = time.perf_counter() - t0
    budgeted_feasible = check_feasibility(board_b).is_feasible
    print(f"[naive + budgeted SA] cost={cost_state.total:,.1f}  time={budgeted_time:.1f}s  feasible={budgeted_feasible}")

    improvement = (baseline_cost_state.total - cost_state.total) / baseline_cost_state.total * 100
    verdict = "BEATS baseline" if cost_state.total < baseline_cost_state.total else "does not beat baseline"
    print(f"-> naive+budgeted vs full baseline: {improvement:+.1f}% ({verdict})")


if __name__ == "__main__":
    run_spike(num_components=150, refine_seconds=30.0, seed=0)
    run_spike(num_components=200, refine_seconds=45.0, seed=1)
