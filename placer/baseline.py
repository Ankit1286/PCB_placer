"""Baseline placer: force-directed initialization + simulated annealing refinement.

Two phases, per spec:

  (a) Force-directed init -- treat each net as a set of springs (constant =
      the net's weight omega_e) pulling together the components whose pins
      it touches; run 500 iterations of a position update; greedily assign
      layers to minimize cross-layer nets; snap each component's rotation to
      whichever of {0, 90, 180, 270} minimizes overlap with its neighbors;
      then legalize (clamp to bounds, resolve overlaps) to get a feasible
      starting point.

  (b) Simulated annealing refinement -- 50,000 perturbations of
      position/rotation/layer, exponential cooling from T0 to T0/200, T0
      calibrated so the *initial* acceptance rate is ~0.5.

The baseline is exempt from the 60s time budget, so this prioritizes
correctness/quality over wall-clock speed -- though it still uses
`IncrementalCost` (see incremental_cost.py) rather than recomputing the full
board cost every SA step, since even a time-unconstrained baseline needs to
finish in a practical amount of dev/grading time (50,000 full recomputes at
|V|=1000 would take on the order of a day; delta caching brings that down to
minutes).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from placer.board import Board, Component, VALID_ROTATIONS
from placer.cost import overlap_against_many
from placer.incremental_cost import IncrementalCost
from placer.legalize import clamp_to_bounds, legalize, push_out_of_keepouts

FORCE_DIRECTED_ITERATIONS = 500
SA_ITERATIONS = 50_000
COOLING_RATIO = 1.0 / 200.0  # T_final = T0 * COOLING_RATIO
CALIBRATION_SAMPLES = 500
JITTER_STD_FRACTION = 0.08  # position-jitter std, as a fraction of board width/height
REPULSION_STRENGTH_FRACTION = 0.02  # see _force_directed_init
FORCE_STEP_SIZE = 0.02


@dataclass
class SAHistory:
    """Cost trajectory recorded during simulated annealing, for diagnostics/plots."""

    costs: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    accepted: list[bool] = field(default_factory=list)


def _force_directed_init(board: Board, rng: np.random.Generator, iterations: int = FORCE_DIRECTED_ITERATIONS) -> None:
    """Settle component *centers* via net-spring attraction + pairwise repulsion, then write back positions.

    Rotation is fixed at 0 during this phase (footprints are unrotated for
    spacing purposes) -- rotation is a separate discrete decision made in
    `_snap_rotations` afterward, once positions have settled.
    """
    n = len(board.components)
    positions = np.zeros((n, 2))
    for i, c in enumerate(board.components):
        positions[i, 0] = rng.uniform(0, max(board.width - c.width, 0.001))
        positions[i, 1] = rng.uniform(0, max(board.height - c.height, 0.001))

    id_to_idx = {c.component_id: i for i, c in enumerate(board.components)}
    edge_weight: dict[tuple[int, int], float] = {}
    for net in board.nets.values():
        comp_indices = sorted({id_to_idx[cid] for cid, _ in net.pin_refs})
        for a in range(len(comp_indices)):
            for b in range(a + 1, len(comp_indices)):
                key = (comp_indices[a], comp_indices[b])
                edge_weight[key] = edge_weight.get(key, 0.0) + net.weight

    if edge_weight:
        edges = np.array(list(edge_weight.keys()), dtype=int)
        weights = np.array(list(edge_weight.values()))
        ei, ej = edges[:, 0], edges[:, 1]
    else:
        ei = ej = np.array([], dtype=int)
        weights = np.array([])

    # Repulsion keeps attraction-only springs from collapsing everything to a point.
    # Strength is scaled off board area per component so it's in the same rough
    # regime regardless of |V| or board size.
    repulsion_strength = REPULSION_STRENGTH_FRACTION * (board.width * board.height) / n

    for _ in range(iterations):
        force = np.zeros((n, 2))
        if len(ei) > 0:
            diff = positions[ej] - positions[ei]
            contrib = weights[:, None] * diff
            np.add.at(force, ei, contrib)
            np.add.at(force, ej, -contrib)

        delta = positions[:, None, :] - positions[None, :, :]
        dist2 = np.sum(delta**2, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist2 = np.maximum(dist2, 1e-6)
        force += repulsion_strength * np.sum(delta / dist2[:, :, None], axis=1)

        positions += FORCE_STEP_SIZE * force
        positions[:, 0] = np.clip(positions[:, 0], 0, board.width)
        positions[:, 1] = np.clip(positions[:, 1], 0, board.height)

    for i, c in enumerate(board.components):
        c.x, c.y = float(positions[i, 0]), float(positions[i, 1])
        c.rotation = 0
        c.layer = "top"


BALANCE_WEIGHT = 0.5  # see _greedy_layer_assignment


def _greedy_layer_assignment(board: Board) -> None:
    """Assign each component top/bottom to greedily minimize cross-layer nets, subject to staying balanced.

    Processes components in decreasing order of net-degree (most-constrained
    first) and assigns each to whichever layer already has more of its
    already-assigned neighbors -- a standard greedy heuristic for this kind
    of minimize-cut assignment (exact minimization is NP-hard).

    Pure net-affinity greedy has a degenerate global optimum here: putting
    *every* component on the same layer trivially drives cross-layer nets to
    zero. That defeats the purpose of having two layers at all -- overlap is
    penalized per-layer, so collapsing onto one layer roughly doubles the
    number of same-layer pairs at risk of overlapping. A small balance term
    (`BALANCE_WEIGHT`, penalizing whichever layer is already more populated)
    keeps the assignment from collapsing while still preferring to group
    net-connected components together when it doesn't fight balance.
    """
    neighbors: dict[str, list[str]] = defaultdict(list)
    for net in board.nets.values():
        comp_ids = sorted({cid for cid, _ in net.pin_refs})
        for a in range(len(comp_ids)):
            for b in range(a + 1, len(comp_ids)):
                neighbors[comp_ids[a]].append(comp_ids[b])
                neighbors[comp_ids[b]].append(comp_ids[a])

    order = sorted(board.components, key=lambda c: -len(neighbors[c.component_id]))
    assigned: dict[str, str] = {}
    top_used = bottom_used = 0
    for c in order:
        top_count = sum(1 for nb in neighbors[c.component_id] if assigned.get(nb) == "top")
        bottom_count = sum(1 for nb in neighbors[c.component_id] if assigned.get(nb) == "bottom")
        score_top = top_count - BALANCE_WEIGHT * top_used
        score_bottom = bottom_count - BALANCE_WEIGHT * bottom_used
        layer = "top" if score_top >= score_bottom else "bottom"
        if layer == "top":
            top_used += 1
        else:
            bottom_used += 1
        assigned[c.component_id] = layer
        c.layer = layer


def _snap_rotations(board: Board, rng: np.random.Generator) -> None:
    """Pick each component's rotation to (locally) minimize overlap with its same-layer neighbors.

    Processed in random order so no component is systematically favored/disadvantaged.
    """
    order = list(board.components)
    rng.shuffle(order)
    for c in order:
        orig_x, orig_y = c.x, c.y
        best_rotation, best_overlap = c.rotation, None
        others = [o.bbox() for o in board.components if o is not c and o.layer == c.layer]
        for rotation in VALID_ROTATIONS:
            # Reset to the pre-trial position before each candidate -- clamp_to_bounds mutates
            # x/y in place, and without resetting, one trial's clamp would silently carry into
            # the next (previously caused snap_rotations to sometimes *increase* total overlap).
            c.x, c.y = orig_x, orig_y
            c.rotation = rotation
            clamp_to_bounds(c, board)
            overlap = overlap_against_many(c.bbox(), others)
            if best_overlap is None or overlap < best_overlap:
                best_overlap, best_rotation = overlap, rotation
        c.x, c.y = orig_x, orig_y
        c.rotation = best_rotation
        clamp_to_bounds(c, board)


def _apply_random_perturbation(c: Component, board: Board, rng: np.random.Generator) -> None:
    """Mutate one component's position, rotation, or layer (one of the three, chosen uniformly).

    C(p) has no keepout term -- it's a pure hard constraint per spec, not
    something SA's cost-driven acceptance has any incentive to fix. So every
    perturbation that could move a component's center into a keepout zone
    (position jitter, or a rotation change that shifts the bbox/center) is
    followed by `push_out_of_keepouts`, keeping "never centered in a
    keepout" true by construction rather than by hoping the annealer cares.
    """
    choice = rng.integers(0, 3)
    if choice == 0:
        std_x = JITTER_STD_FRACTION * board.width
        std_y = JITTER_STD_FRACTION * board.height
        c.x += float(rng.normal(0, std_x))
        c.y += float(rng.normal(0, std_y))
        clamp_to_bounds(c, board)
        push_out_of_keepouts(c, board)
    elif choice == 1:
        options = [r for r in VALID_ROTATIONS if r != c.rotation]
        c.rotation = int(options[rng.integers(0, len(options))])
        clamp_to_bounds(c, board)  # footprint w/h may have swapped
        push_out_of_keepouts(c, board)
    else:
        c.layer = "bottom" if c.layer == "top" else "top"


def _calibrate_temperature(
    board: Board, cost_state: IncrementalCost, rng: np.random.Generator, samples: int = CALIBRATION_SAMPLES
) -> float:
    """Pick T0 so that the mean uphill (cost-increasing) perturbation has ~50% acceptance probability.

    Standard SA calibration: sample `samples` random perturbations from the
    current state, record the cost increase for the uphill ones, revert each
    immediately, then solve exp(-mean_uphill_delta / T0) = 0.5 for T0.
    """
    components = board.components
    uphill_deltas = []
    for _ in range(samples):
        c = components[rng.integers(0, len(components))]
        old_x, old_y, old_rotation, old_layer = c.x, c.y, c.rotation, c.layer
        old_bbox = c.bbox()
        before = cost_state.total

        _apply_random_perturbation(c, board, rng)
        cost_state.component_moved(c, old_bbox, old_layer)
        delta = cost_state.total - before
        if delta > 0:
            uphill_deltas.append(delta)

        reverted_bbox, reverted_layer = c.bbox(), c.layer
        c.x, c.y, c.rotation, c.layer = old_x, old_y, old_rotation, old_layer
        cost_state.component_moved(c, reverted_bbox, reverted_layer)

    mean_uphill = float(np.mean(uphill_deltas)) if uphill_deltas else 1.0
    return max(mean_uphill / np.log(2.0), 1e-9)


def simulated_annealing(
    board: Board,
    rng: np.random.Generator,
    iterations: int = SA_ITERATIONS,
    cooling_ratio: float = COOLING_RATIO,
    record_history: bool = False,
    max_seconds: float | None = None,
    cost_state: IncrementalCost | None = None,
    calibration_samples: int = CALIBRATION_SAMPLES,
) -> tuple[IncrementalCost, SAHistory | None]:
    """Run simulated annealing refinement on an already-legalized board, in place.

    Args:
        iterations: cooling is scheduled to reach `cooling_ratio` at this
            many steps; also the hard cap on steps if `max_seconds` is None.
        max_seconds: if set, stop early once this much wall-clock time has
            elapsed, even if `iterations` hasn't been reached -- this is
            what lets the learned placer's refinement pass share this exact
            function while respecting the 60s inference budget. The cooling
            schedule is still paced against `iterations`, so stopping early
            just means less cooling was achieved, not a different schedule.
        cost_state: reuse an existing `IncrementalCost` instead of building
            a fresh one (the learned placer already has one after legalizing
            its GNN-proposed placement).
        calibration_samples: perturbations used to pick T0 (see
            `_calibrate_temperature`). This cost isn't counted against
            `max_seconds` (calibration runs before this function's own timer
            starts), so it silently eats into the caller's time budget --
            irrelevant for the unbudgeted baseline, but real seconds taken
            from the learned placer's 60s window. Callers under a tight
            budget can pass a smaller value to trade calibration precision
            for more actual annealing time.

    Returns:
        The final `IncrementalCost` (so callers get the final cost for free
        without another full recompute) and, if requested, a per-iteration
        history for diagnostics/plots.
    """
    if cost_state is None:
        cost_state = IncrementalCost(board)
    T0 = _calibrate_temperature(board, cost_state, rng, samples=calibration_samples)
    T_final = T0 * cooling_ratio
    alpha = (T_final / T0) ** (1.0 / iterations)
    T = T0

    history = SAHistory() if record_history else None
    components = board.components

    start = time.perf_counter() if max_seconds is not None else None
    for step in range(iterations):
        if start is not None and (step % 50 == 0) and time.perf_counter() - start > max_seconds:
            break

        c = components[rng.integers(0, len(components))]
        old_x, old_y, old_rotation, old_layer = c.x, c.y, c.rotation, c.layer
        old_bbox = c.bbox()
        before = cost_state.total

        _apply_random_perturbation(c, board, rng)
        cost_state.component_moved(c, old_bbox, old_layer)
        delta = cost_state.total - before

        accept = delta <= 0 or rng.random() < np.exp(-delta / T)
        if not accept:
            reverted_bbox, reverted_layer = c.bbox(), c.layer
            c.x, c.y, c.rotation, c.layer = old_x, old_y, old_rotation, old_layer
            cost_state.component_moved(c, reverted_bbox, reverted_layer)

        if history is not None:
            history.costs.append(cost_state.total)
            history.temperatures.append(T)
            history.accepted.append(bool(accept))

        T *= alpha

    return cost_state, history


def place_baseline(
    board: Board, seed: int = 0, record_history: bool = False, sa_iterations: int = SA_ITERATIONS
) -> tuple[Board, IncrementalCost, SAHistory | None]:
    """Run the full baseline pipeline (force-directed init + SA refinement) on `board`, in place.

    Exempt from the 60s time budget per spec -- at |V|=1000 the SA phase
    alone takes several minutes.
    """
    rng = np.random.default_rng(seed)
    _force_directed_init(board, rng)
    _greedy_layer_assignment(board)
    _snap_rotations(board, rng)
    legalize(board, rng=rng)
    cost_state, history = simulated_annealing(board, rng, iterations=sa_iterations, record_history=record_history)

    # Feasibility backstop: overlap is weighted 100x in C(p), but that's still a *finite*
    # weight, not an infinite barrier. At very low end-of-schedule temperature, escaping a
    # residual overlap can require a net cost increase (overlap drops but wirelength rises
    # more) -- exactly the kind of move Metropolis acceptance makes statistically unreachable
    # once T is small. Observed in practice: a full 50k-iteration run left 3 same-layer pairs
    # still overlapping. A final legalize() pass resolves that mechanically, same reasoning as
    # why keepout-avoidance is enforced structurally rather than left to the cost function.
    legalize(board, rng=rng)
    cost_state = IncrementalCost(board)
    return board, cost_state, history
