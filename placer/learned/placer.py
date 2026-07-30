"""Learned placer inference: GNN warm-start + heuristic rotation/layer + budgeted SA refinement.

`place(board)` is the required entry point (see scripts/benchmark.py, which already imports it),
matching `place_baseline`'s contract except budgeted to the spec's 60s wall-clock limit instead of
running 50,000 fixed SA iterations. Pipeline:

  1. GNN forward pass -- predicts each component's absolute position directly from graph structure
     (see model.py; no force-directed init or other initializer involved -- profiling showed that
     step alone dominating training-step time at larger |V|, so it was dropped from both training
     and inference, which also frees up more of the 60s budget for SA below).
  2. `_greedy_layer_assignment` + `_snap_rotations` (reused from baseline.py) -- rotation/layer
     aren't modeled by the GNN (see model.py's docstring for why), so they're assigned exactly as
     the baseline does, given the GNN's positions. Both need a placed board first, so rotation/layer
     placeholders (0 / "top") are set right after the GNN's positions are written back.
  3. `legalize` -- clamp/keepout-push/overlap-resolve into a feasible starting point.
  4. `simulated_annealing`, wall-clock-budgeted via `max_seconds` -- the same refinement loop the
     baseline uses, just stopped early to respect the 60s inference budget.
  5. A final `legalize` backstop -- overlap's cost weight is finite, not infinite (DESIGN.md sec
     4d), so budgeted SA can end with residual overlap the same way a full baseline run can.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from placer.baseline import _greedy_layer_assignment, _snap_rotations, simulated_annealing
from placer.board import Board
from placer.legalize import legalize
from placer.learned.model import PlacementGNN, board_to_tensors

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "placement_gnn_best.pt"
TIME_BUDGET_SECONDS = 60.0

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


def _load_model(checkpoint_path: Path) -> PlacementGNN:
    """Load a trained GNN if a checkpoint exists; otherwise fall back to a randomly-initialized one.

    The random-weights fallback exists so `place()` stays runnable (feasibility guaranteed by
    legalize/SA regardless of placement quality) even before any training has happened -- useful
    for wiring/smoke tests, though real quality obviously requires an actual trained checkpoint.
    """
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model = PlacementGNN(hidden_dim=state.get("hidden_dim", 64))
        model.load_state_dict(state["model_state_dict"])
    else:
        model = PlacementGNN()
    model.eval()
    return model


def place(
    board: Board,
    seed: int = 0,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    time_budget: float = TIME_BUDGET_SECONDS,
) -> Board:
    """Place `board` using a GNN warm start + budgeted SA refinement, within `time_budget` seconds."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    model = _load_model(Path(checkpoint_path))
    graph = board_to_tensors(board)
    with torch.no_grad():
        pred_pos_norm = model(graph)
    pred_pos = pred_pos_norm.numpy()
    for i, cid in enumerate(graph.component_ids):
        c = board.component_by_id(cid)
        c.x = float(pred_pos[i, 0] * board.width)
        c.y = float(pred_pos[i, 1] * board.height)
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
