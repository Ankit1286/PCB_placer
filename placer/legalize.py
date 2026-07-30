"""Shared placement legalization: clamp to board bounds, push out of keepouts, resolve overlaps.

Both the baseline's force-directed init (which produces a continuous,
likely-overlapping layout) and the learned placer's GNN forward pass
(same situation) need to turn a "roughly good but not necessarily legal"
continuous suggestion into something feasibility.py would accept, before
handing off to simulated-annealing refinement. This module is that shared
step.
"""

from __future__ import annotations

import numpy as np

from placer.board import Board, Component
from placer.cost import overlap_against_many


def clamp_to_bounds(component: Component, board: Board) -> None:
    """Clip a component's position so its bounding box lies entirely within the board."""
    w, h = component.footprint()
    component.x = float(np.clip(component.x, 0.0, max(board.width - w, 0.0)))
    component.y = float(np.clip(component.y, 0.0, max(board.height - h, 0.0)))


def push_out_of_keepouts(component: Component, board: Board, max_attempts: int = 8) -> None:
    """If component's center falls in a keepout zone, nudge it just outside the nearest *reachable* zone edge.

    Two things "nearest edge" alone gets wrong:

    1. If the keepout touches the board boundary on that side, pushing
       toward it and then clamping back into bounds just lands the
       component right back inside the zone -- so each candidate direction
       is checked for whether it fits on the board before being chosen.
    2. With two keepout zones close together, escaping *this* zone via its
       nearest edge can land the component's center inside a *different*
       zone -- observed in practice with two zones ~0.18mm apart, narrower
       than the stuck component, causing the old edge-distance-only logic
       to oscillate between them across attempts without ever escaping.
       So candidates are checked against *every* keepout, not just the one
       currently containing the center, and the closest fully-clear option
       is preferred.
    """
    for _ in range(max_attempts):
        cx, cy = component.center()
        zone = next((k for k in board.keepouts if k.contains_point(cx, cy)), None)
        if zone is None:
            return

        w, h = component.footprint()
        eps = 1e-6
        max_x, max_y = max(board.width - w, 0.0), max(board.height - h, 0.0)
        options = sorted(
            [
                (cx - zone.xmin, zone.xmin - w - eps, component.y),
                (zone.xmax - cx, zone.xmax + eps, component.y),
                (cy - zone.ymin, component.x, zone.ymin - h - eps),
                (zone.ymax - cy, component.x, zone.ymax + eps),
            ],
            key=lambda o: o[0],
        )

        in_bounds = [(nx, ny) for _, nx, ny in options if 0.0 <= nx <= max_x and 0.0 <= ny <= max_y]
        fully_clear = next(
            (
                (nx, ny)
                for nx, ny in in_bounds
                if not any(k.contains_point(nx + w / 2, ny + h / 2) for k in board.keepouts)
            ),
            None,
        )
        if fully_clear is not None:
            component.x, component.y = fully_clear
        elif in_bounds:
            # No single move clears every zone at once (adjacent/nested keepouts) -- take the
            # closest bounds-feasible option anyway; the next loop iteration (up to
            # max_attempts) will continue escaping whichever zone it lands in next.
            component.x, component.y = in_bounds[0]
        else:
            # No direction escapes within bounds at all (keepout spans the board on every
            # axis) -- take the closest anyway; clamping below will at least keep it on the board.
            _, new_x, new_y = options[0]
            component.x, component.y = new_x, new_y
        clamp_to_bounds(component, board)


def _resolve_overlaps_one_pass(components: list[Component], board: Board) -> bool:
    """Push every overlapping same-layer pair apart along their axis of least overlap. Returns True if any moved."""
    by_layer: dict[str, list[Component]] = {"top": [], "bottom": []}
    for c in components:
        by_layer[c.layer].append(c)

    any_moved = False
    for layer_components in by_layer.values():
        n = len(layer_components)
        if n < 2:
            continue
        boxes = np.array([c.bbox() for c in layer_components])
        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

        overlap_x = np.minimum(xmax[:, None], xmax[None, :]) - np.maximum(xmin[:, None], xmin[None, :])
        overlap_y = np.minimum(ymax[:, None], ymax[None, :]) - np.maximum(ymin[:, None], ymin[None, :])
        overlapping = (overlap_x > 1e-9) & (overlap_y > 1e-9)
        np.fill_diagonal(overlapping, False)

        pairs = np.argwhere(np.triu(overlapping, k=1))
        if len(pairs) == 0:
            continue
        any_moved = True

        push = np.zeros((n, 2))
        for i, j in pairs:
            ox, oy = overlap_x[i, j], overlap_y[i, j]
            # Minimum-translation-vector heuristic: separate along whichever axis needs less movement.
            # Prefer moving whichever of the two has enough room to absorb the *entire* separation by
            # itself -- that resolves the pair exactly in this one pass. Only fall back to splitting
            # the push across both (which, if either is boundary-locked, only halves the overlap and
            # needs further passes) when neither alone has enough room.
            if ox < oy:
                low, high = (i, j) if cx[i] <= cx[j] else (j, i)
                room_low, room_high = xmin[low], board.width - xmax[high]
                if room_low >= ox:
                    push[low, 0] -= ox
                elif room_high >= ox:
                    push[high, 0] += ox
                else:
                    push[low, 0] -= room_low
                    push[high, 0] += room_high
            else:
                low, high = (i, j) if cy[i] <= cy[j] else (j, i)
                room_low, room_high = ymin[low], board.height - ymax[high]
                if room_low >= oy:
                    push[low, 1] -= oy
                elif room_high >= oy:
                    push[high, 1] += oy
                else:
                    push[low, 1] -= room_low
                    push[high, 1] += room_high

        for c, (dx, dy) in zip(layer_components, push):
            if dx != 0.0 or dy != 0.0:
                c.x += dx
                c.y += dy
                clamp_to_bounds(c, board)
                push_out_of_keepouts(c, board)

    return any_moved


def _relocate_stuck_components(board: Board, rng: np.random.Generator, max_tries: int = 300) -> None:
    """Last-resort fallback: rejection-sample a fresh position for components still overlapping.

    The iterative pairwise push in `_resolve_overlaps_one_pass` can reach a
    fixed point without fully resolving a *tangled cluster* of several
    mutually-overlapping components (observed in practice after SA on a
    60-component board: 6 components stayed pairwise-overlapping through
    500 passes). Rather than chase a more elaborate multi-body push rule,
    this handles the residual directly and robustly: for each component
    still in an overlapping pair, try random positions until one has zero
    overlap against every other same-layer component. Guaranteed to
    terminate given the spec's 40-70% utilization bound leaves real free
    area, though it may cost more wirelength for the relocated components
    -- an acceptable one-time trade for guaranteeing feasibility.
    """
    from placer.feasibility import check_feasibility

    for _ in range(10):  # outer retries: relocating one component could newly overlap another
        report = check_feasibility(board)
        if not report.overlapping_pairs:
            return
        stuck_ids = {cid for pair in report.overlapping_pairs for cid in pair}
        for cid in stuck_ids:
            c = board.component_by_id(cid)
            w, h = c.footprint()
            max_x, max_y = max(board.width - w, 0.0), max(board.height - h, 0.0)
            others = [o.bbox() for o in board.components if o.component_id != cid and o.layer == c.layer]
            for _ in range(max_tries):
                trial_x = rng.uniform(0, max_x) if max_x > 0 else 0.0
                trial_y = rng.uniform(0, max_y) if max_y > 0 else 0.0
                bbox = (trial_x, trial_y, trial_x + w, trial_y + h)
                trial_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                in_a_keepout = any(k.contains_point(*trial_center) for k in board.keepouts)
                if not in_a_keepout and overlap_against_many(bbox, others) < 1e-9:
                    c.x, c.y = trial_x, trial_y
                    break


def legalize(board: Board, max_passes: int = 40, rng: np.random.Generator | None = None) -> None:
    """Clamp every placed component to bounds/keepouts, then iteratively resolve same-layer overlaps.

    Falls back to `_relocate_stuck_components` for any residual overlap
    still present after `max_passes` (a tangled multi-component cluster the
    pairwise push couldn't fully untangle) -- see that function's docstring.
    """
    for c in board.components:
        if c.is_placed:
            clamp_to_bounds(c, board)
            push_out_of_keepouts(c, board)

    placed = [c for c in board.components if c.is_placed]
    resolved = False
    for _ in range(max_passes):
        if not _resolve_overlaps_one_pass(placed, board):
            resolved = True
            break

    if not resolved:
        _relocate_stuck_components(board, rng if rng is not None else np.random.default_rng())
