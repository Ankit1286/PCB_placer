"""Learned placer inference: per-board analytic optimization + heuristic rotation/layer + budgeted SA.

`place(board)` is the required entry point (see scripts/benchmark.py, which already imports it),
matching `place_baseline`'s contract except budgeted to the spec's 60s wall-clock limit instead of
running 50,000 fixed SA iterations.

Late pivot from a trained-GNN warm start to DREAMPlace-style per-board analytic placement -- see
DESIGN.md's final section for the full reasoning. Short version: the GNN had to solve a strictly
harder problem than the task requires (generalize across an entire board-size distribution ahead of
time) when the actual requirement is just "place *this* board well in 60s, with a GPU available."
Per-board optimization reuses the exact same differentiable proxy (`loss.py`) but treats the
placement itself -- not any model's weights -- as the thing gradient descent improves, directly
against the one board being placed, for as many steps as the time budget allows. No training data, no
checkpoint, no generalization question at all.

Pipeline:

  1. Random starting positions -- deliberately *not* `_force_directed_init`: that function's 500
     iterations of O(V^2) physics cost ~3s at |V|=300 (already profiled, and why it was dropped from
     GNN training earlier in this project) but scales roughly quadratically -- ~36s at |V|=1000,
     which would eat most of the 60s budget before optimization even starts. Gradient descent below
     does the actual spreading-out work anyway (the same reason DREAMPlace-style methods typically
     use no spatial prior at all), so a cheap uniform-random start is sufficient.
  2. Per-board gradient descent against `loss.py::proxy_cost` (wirelength + overlap + congestion),
     wall-clock budgeted -- `positions` itself is the optimized tensor, via `torch.optim.Adam`.
  3. `_greedy_layer_assignment` + `_snap_rotations` (reused from baseline.py) -- rotation/layer still
     aren't part of the continuous optimization (same reason as when the GNN didn't model them
     either: keeps the objective purely continuous, avoids a discrete-choice differentiability
     problem), so they're assigned exactly as the baseline does, given the optimized positions.
  4. `legalize` -- clamp/keepout-push/overlap-resolve into a feasible starting point.
  5. `simulated_annealing`, wall-clock-budgeted via `max_seconds` -- the same refinement loop the
     baseline uses, just stopped early to respect the 60s inference budget.
  6. A final `legalize` backstop -- overlap's cost weight is finite, not infinite (DESIGN.md sec
     4d), so budgeted SA can end with residual overlap the same way a full baseline run can.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from placer.baseline import _greedy_layer_assignment, _snap_rotations, simulated_annealing
from placer.board import Board
from placer.legalize import legalize
from placer.learned.loss import ProxyLossNormalizer, proxy_cost
from placer.learned.model import board_to_tensors

TIME_BUDGET_SECONDS = 60.0

# Fraction of the total budget spent on per-board gradient descent (step 2) -- the rest covers
# layer/rotation assignment, legalize, the SA polish, and the size-scaled safety margin below (random
# init itself is negligible, unlike the _force_directed_init this replaced). Not rigorously tuned; a
# reasonable split given SA still earns its keep as a discrete-choice (layer/rotation) and
# residual-overlap cleanup pass, not just polish.
ANALYTIC_TIME_FRACTION = 0.55  # empirically: more steps than this made |V|=200 notably *worse*
# (-31% -> -66% at 0.75), not better -- likely the same kind of frozen-normalizer imbalance drift
# seen during amortized training, just within a single board's optimization instead of across many.
# Not swept properly given the time constraint; a real candidate for revisiting.
ANALYTIC_LR = 2.0  # positions are raw board-coordinate values (mm), not [0,1]-normalized -- needs a
# correspondingly larger step size than the old model-weight training (lr=1e-3); picked by a quick
# empirical check that positions actually move a meaningful distance per step, not tuned further.
ANALYTIC_WARMUP_STEPS = 100  # shorter than training's 300: this optimizer only ever sees one board,
# for a few thousand steps total, not thousands of different boards over tens of thousands of steps.
ANALYTIC_MAX_STEPS = 2_000_000  # safety cap independent of the wall-clock check; the wall-clock check
# should be what actually stops the loop -- this just guards against an infinite loop if it doesn't

# Slack reserved for the final legalize() backstop, which runs *after* the budgeted SA loop and
# isn't itself time-bounded. A flat margin isn't enough: the first full benchmark run showed every
# |V|=1000 board finishing at 61-62s despite a 6.0s margin, while |V|=200 stayed under budget --
# legalize's cost (overlap-resolution passes, relocate-stuck fallback) grows with board size, so the
# margin needs to scale with it too, not just cover the smallest boards tested.
BASE_SAFETY_MARGIN_SECONDS = 5.0
PER_COMPONENT_SAFETY_MARGIN_SECONDS = 0.02  # e.g. +4s at |V|=200, +20s at |V|=1000

# baseline.py's default (500) is fine when unbudgeted, but calibration isn't counted against
# simulated_annealing's own max_seconds timer (it runs before that timer starts), so every sample
# here silently eats into this function's safety margin instead of the SA loop itself. Under a hard
# 60s budget -- especially at |V|=1000, where SA is already starved for touches-per-component -- a
# smaller calibration sample buys back real annealing time at the cost of a less precisely-picked T0.
CALIBRATION_SAMPLES = 100


def _optimize_positions(board: Board, graph, rng: np.random.Generator, time_budget: float) -> np.ndarray:
    """Per-board gradient descent against the differentiable proxy cost, starting from cheap random
    positions. Returns (num_components, 2) real board-coordinate positions -- `graph.component_ids[i]`
    is the id of the component at row i.

    This is the DREAMPlace-style pivot: `positions` is a plain tensor, not a model's output --
    gradient descent optimizes the placement itself, directly against this one board, rather than a
    reusable model evaluated once. No checkpoint, no training data, nothing to generalize.

    Runs on CUDA automatically if available (this is the whole point of the pivot -- a GPU sitting
    idle during the 60s budget was the original motivation) with a silent CPU fallback otherwise, so
    the same code runs correctly (just slower) in this dev environment, which has no GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init = np.stack([rng.uniform(0, board.width, len(graph.component_ids)), rng.uniform(0, board.height, len(graph.component_ids))], axis=1)
    positions = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    widths = graph.widths.to(device)
    heights = graph.heights.to(device)
    edge_comp_idx = graph.edge_comp_idx.to(device)
    edge_net_idx = graph.edge_net_idx.to(device)
    net_weights = graph.net_weights.to(device)

    optimizer = torch.optim.Adam([positions], lr=ANALYTIC_LR)
    normalizer = ProxyLossNormalizer(warmup_steps=ANALYTIC_WARMUP_STEPS)
    num_nets = graph.net_feats.shape[0]
    max_x, max_y = board.width, board.height

    loop_start = time.perf_counter()
    for step in range(ANALYTIC_MAX_STEPS):
        # Checked every iteration, not every N: per-step cost scales with |V| (O(V^2) overlap,
        # O(num_nets*grid_size^2) congestion), so a fixed check interval that's safe at small |V|
        # can overshoot badly at large |V| -- exactly what happened at |V|=1000 (73s of a 60s budget)
        # before this was tightened. time.perf_counter() itself is cheap; checking every step is not
        # a meaningful cost next to the actual optimization step.
        if time.perf_counter() - loop_start > time_budget:
            break
        optimizer.zero_grad()
        loss = proxy_cost(
            positions,
            widths,
            heights,
            edge_comp_idx,
            edge_net_idx,
            net_weights,
            num_nets=num_nets,
            board_width=board.width,
            board_height=board.height,
            normalizer=normalizer,
        )
        loss.backward()
        optimizer.step()
        # Nothing bounded positions during descent (unlike the old GNN, whose sigmoid output was
        # structurally confined to [0,1]) -- a component drifting far off-board or into a tangled
        # overlap made legalize()'s cleanup (esp. the relocate-stuck fallback) expensive enough to
        # blow the 60s budget on its own. Clamping each step to a small margin outside the board
        # keeps gradient descent from ever wandering into that regime, without meaningfully
        # constraining it -- the real board area is where the optimum should be anyway.
        with torch.no_grad():
            positions[:, 0].clamp_(-0.05 * max_x, 1.05 * max_x)
            positions[:, 1].clamp_(-0.05 * max_y, 1.05 * max_y)

    return positions.detach().cpu().numpy()


def place(board: Board, seed: int = 0, time_budget: float = TIME_BUDGET_SECONDS) -> Board:
    """Place `board` via per-board analytic optimization + budgeted SA refinement, within `time_budget`s."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    graph = board_to_tensors(board)
    final_pos = _optimize_positions(board, graph, rng, time_budget=time_budget * ANALYTIC_TIME_FRACTION)
    for i, cid in enumerate(graph.component_ids):
        c = board.component_by_id(cid)
        c.x = float(final_pos[i, 0])
        c.y = float(final_pos[i, 1])
        c.rotation = 0  # placeholder; _snap_rotations below picks the real value
        c.layer = "top"  # placeholder; _greedy_layer_assignment below picks the real value

    _greedy_layer_assignment(board)
    _snap_rotations(board, rng)
    legalize(board, rng=rng)

    safety_margin = BASE_SAFETY_MARGIN_SECONDS + PER_COMPONENT_SAFETY_MARGIN_SECONDS * len(board.components)
    elapsed = time.perf_counter() - start
    remaining = max(time_budget - elapsed - safety_margin, 0.0)
    simulated_annealing(board, rng, iterations=1_000_000, max_seconds=remaining, calibration_samples=CALIBRATION_SAMPLES)

    legalize(board, rng=rng)
    return board
