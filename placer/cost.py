"""The placement cost function C(p) and its three components.

    C(p) = sum_e w_e * RSMT(e)  +  100 * sum_(u,v) max(0, overlap(u, v))  +  C_cong(p)

Where RSMT(e) is a net's rectilinear Steiner minimum tree length (the
shortest way to wire together a set of pins using only horizontal/vertical
segments, optionally branching through extra "Steiner" junction points),
plus a fixed penalty per layer crossing (each crossing needs a via, a hole
connecting the two copper layers).

Two genuinely non-obvious engineering decisions live in this file (see
DESIGN.md for the full reasoning):

1. RSMT is NP-hard in general. For nets with <=4 pins we compute it exactly
   using Hanan's theorem (below). For larger nets we fall back to a
   rectilinear minimum *spanning* tree (RMST), a well-known heuristic with a
   proven worst-case approximation ratio of 1.5x optimal (Hwang 1976).
2. Layer-crossing count and congestion rasterization are both computed
   against the RMST-over-real-pins topology (not whatever extra Steiner
   points the exact <=4 solver finds), because vias/congestion are physical
   phenomena tied to real pins, not to geometric convenience points. See
   DESIGN.md for why this is a reasonable simplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from placer.board import Board, Component

NU_VIA = 5.0  # mm penalty per layer crossing
OVERLAP_WEIGHT = 100.0
GRID_SIZE = 32  # congestion grid is GRID_SIZE x GRID_SIZE per layer
CONGESTION_FREE_CAPACITY = 10  # cells can hold this many crossings for free


def rect_overlap_area(bbox_a: tuple[float, float, float, float], bbox_b: tuple[float, float, float, float]) -> float:
    """Intersection area of two axis-aligned bounding boxes (xmin, ymin, xmax, ymax). 0 if they don't overlap."""
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def _rectilinear_mst(points: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Minimum spanning tree over `points` using Manhattan (L1) distance.

    Manhattan distance between any two points equals the length of a real
    rectilinear (horizontal/vertical-only) monotone path between them, so
    this MST is always a physically routable rectilinear tree — it's the
    RMST heuristic mentioned in the spec for nets with >4 pins.

    Implemented as a hand-rolled dense Prim's algorithm rather than
    `scipy.sparse.csgraph.minimum_spanning_tree`: nets top out at 8 pins (and
    the exact-RSMT solver below calls this on <=10-point sets tens of
    thousands of times per full cost evaluation), and scipy's sparse-graph
    validation/construction overhead per call dominates at that scale —
    profiling showed it responsible for the bulk of a 7.7s full-board
    evaluation at |V|=1000. Dense Prim's on a handful of points has none of
    that overhead.

    Returns:
        (total_length, edges) where edges are (i, j) index pairs into `points`.

    Note: nets top out at 8 pins and the exact-RSMT solver below adds at
    most 2 more candidate points, so every call here has n<=~10. This is
    plain Python (lists of tuples), not numpy, deliberately — numpy's
    per-call overhead exceeds its vectorization benefit at this size, and
    this function is called ~100x per net during the Steiner-point search,
    then that whole search runs again for every SA perturbation.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n <= 1:
        return 0.0, []

    in_tree = [False] * n
    in_tree[0] = True
    min_edge = [abs(pts[0][0] - x) + abs(pts[0][1] - y) for x, y in pts]
    min_edge[0] = float("inf")
    parent = [0] * n

    edges = []
    total = 0.0
    for _ in range(n - 1):
        j, best = -1, float("inf")
        for k in range(n):
            if not in_tree[k] and min_edge[k] < best:
                best, j = min_edge[k], k
        total += best
        edges.append((parent[j], j))
        in_tree[j] = True

        pjx, pjy = pts[j]
        for k in range(n):
            if not in_tree[k]:
                d = abs(pjx - pts[k][0]) + abs(pjy - pts[k][1])
                if d < min_edge[k]:
                    min_edge[k] = d
                    parent[k] = j

    return total, edges


def _exact_rsmt_length(points: np.ndarray) -> float:
    """Exact rectilinear Steiner minimum tree length for n<=4 terminals.

    Uses Hanan's theorem (Hanan, 1966): an optimal rectilinear Steiner tree
    for any point set always has a realization using only points on the
    "Hanan grid" — the grid formed by drawing a horizontal and vertical line
    through every terminal. For n<=4 terminals this grid has at most 16
    points, and an optimal tree needs at most n-2 of the non-terminal grid
    points as extra Steiner (branching) junctions. We brute-force every
    subset of up to (n-2) candidate Steiner points, take the rectilinear MST
    of terminals+subset for each, and keep the shortest — small enough
    (<=66 subsets of size 2 from <=12 candidates) to be instant.
    """
    n = points.shape[0]
    if n <= 2:
        return float(np.abs(points[0] - points[1]).sum()) if n == 2 else 0.0

    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    grid = np.array([[x, y] for x in xs for y in ys])
    terminal_set = {tuple(p) for p in points.tolist()}
    candidates = np.array([p for p in grid.tolist() if tuple(p) not in terminal_set])

    best, _ = _rectilinear_mst(points)
    max_extra = max(0, n - 2)
    if len(candidates) > 0:
        for k in range(1, max_extra + 1):
            for combo in combinations(range(len(candidates)), k):
                pts = np.vstack([points, candidates[list(combo)]])
                length, _ = _rectilinear_mst(pts)
                if length < best:
                    best = length
    return float(best)


def rsmt_and_crossings(points: np.ndarray, layers: list[str]) -> tuple[float, int]:
    """Compute a net's (geometric wire length, number of layer crossings).

    Args:
        points: (n, 2) array of board-space pin coordinates.
        layers: length-n list of "top"/"bottom", parallel to `points`.

    Returns:
        length: exact RSMT for n<=4 pins, RMST heuristic (<=1.5x optimal) otherwise.
        crossings: number of RMST-topology edges whose two endpoints are on different layers.
    """
    n = points.shape[0]
    if n < 2:
        return 0.0, 0

    base_length, base_edges = _rectilinear_mst(points)
    length = _exact_rsmt_length(points) if n <= 4 else base_length
    crossings = sum(1 for i, j in base_edges if layers[i] != layers[j])
    return length, crossings


def _cells_along_row(x1: float, x2: float, row: int, cell_w: float) -> list[tuple[int, int]]:
    """Grid cells covered by a horizontal segment at a fixed row, from x1 to x2."""
    c1 = int(min(x1 // cell_w, GRID_SIZE - 1))
    c2 = int(min(x2 // cell_w, GRID_SIZE - 1))
    lo, hi = sorted((c1, c2))
    return [(row, col) for col in range(max(lo, 0), min(hi, GRID_SIZE - 1) + 1)]


def _cells_along_col(y1: float, y2: float, col: int, cell_h: float) -> list[tuple[int, int]]:
    """Grid cells covered by a vertical segment at a fixed column, from y1 to y2."""
    r1 = int(min(y1 // cell_h, GRID_SIZE - 1))
    r2 = int(min(y2 // cell_h, GRID_SIZE - 1))
    lo, hi = sorted((r1, r2))
    return [(row, col) for row in range(max(lo, 0), min(hi, GRID_SIZE - 1) + 1)]


def _net_congestion_cells(
    points: np.ndarray, layers: list[str], edges: list[tuple[int, int]], cell_w: float, cell_h: float
) -> dict[str, list[tuple[int, int]]]:
    """Rasterize a net's RMST edges onto the per-layer grid as an L-shaped (horizontal-then-vertical) path each.

    When an edge's two endpoints are on different layers, the horizontal leg
    is attributed to the source pin's layer and the vertical leg to the
    destination pin's layer (modeling "route on one layer up to the via,
    continue on the other after it") — a documented simplifying assumption,
    since the spec does not define a router.
    """
    cells: dict[str, list[tuple[int, int]]] = {"top": [], "bottom": []}
    for i, j in edges:
        x1, y1 = points[i]
        x2, y2 = points[j]
        row1 = int(min(y1 // cell_h, GRID_SIZE - 1))
        col2 = int(min(x2 // cell_w, GRID_SIZE - 1))
        cells[layers[i]].extend(_cells_along_row(x1, x2, row1, cell_w))
        cells[layers[j]].extend(_cells_along_col(y1, y2, col2, cell_h))
    return cells


def overlap_against_many(bbox: tuple[float, float, float, float], other_boxes: list[tuple]) -> float:
    """Sum of `bbox`'s overlap area against each box in `other_boxes` (vectorized, all same layer)."""
    if not other_boxes:
        return 0.0
    arr = np.array(other_boxes)
    ix = np.maximum(0.0, np.minimum(arr[:, 2], bbox[2]) - np.maximum(arr[:, 0], bbox[0]))
    iy = np.maximum(0.0, np.minimum(arr[:, 3], bbox[3]) - np.maximum(arr[:, 1], bbox[1]))
    return float(np.sum(ix * iy))


def total_overlap_cost(components: list[Component]) -> float:
    """Sum of pairwise bounding-box overlap area, per layer only, over all placed components.

    Vectorized with numpy broadcasting: O(n^2) time/memory per layer, which
    is trivial up to |V|=1000 (a 1000x1000 float array is ~8MB).
    """
    by_layer: dict[str, list[tuple[float, float, float, float]]] = {"top": [], "bottom": []}
    for c in components:
        if c.is_placed:
            by_layer[c.layer].append(c.bbox())

    total = 0.0
    for boxes in by_layer.values():
        n = len(boxes)
        if n < 2:
            continue
        arr = np.array(boxes)
        xmin, ymin, xmax, ymax = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        ix = np.maximum(0.0, np.minimum(xmax[:, None], xmax[None, :]) - np.maximum(xmin[:, None], xmin[None, :]))
        iy = np.maximum(0.0, np.minimum(ymax[:, None], ymax[None, :]) - np.maximum(ymin[:, None], ymin[None, :]))
        overlap = ix * iy
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        total += float(overlap[mask].sum())
    return total


@dataclass
class CostBreakdown:
    """The three additive terms of C(p), plus their sum, for logging/debugging."""

    wirelength: float
    overlap: float
    congestion: float

    @property
    def total(self) -> float:
        """Total placement cost C(p) = wirelength + overlap + congestion."""
        return self.wirelength + self.overlap + self.congestion


def compute_cost(board: Board) -> CostBreakdown:
    """Compute the full placement cost C(p) for a (fully or partially) placed board.

    Unplaced components are skipped entirely (their nets/overlaps are
    ignored) — callers are expected to only call this on fully placed boards
    except during incremental/partial construction.
    """
    cell_w = board.width / GRID_SIZE
    cell_h = board.height / GRID_SIZE
    demand = {"top": np.zeros((GRID_SIZE, GRID_SIZE)), "bottom": np.zeros((GRID_SIZE, GRID_SIZE))}

    wirelength_cost = 0.0
    for net in board.nets.values():
        components = [board.component_by_id(cid) for cid, _ in net.pin_refs]
        if not all(c.is_placed for c in components):
            continue
        pin_data = [c.pin_board_position(c.pins[pidx]) for c, (_, pidx) in zip(components, net.pin_refs)]
        points = np.array([[x, y] for x, y, _ in pin_data])
        layers = [l for _, _, l in pin_data]
        length, crossings = rsmt_and_crossings(points, layers)
        wirelength_cost += net.weight * (length + crossings * NU_VIA)

        _, edges = _rectilinear_mst(points)
        cell_hits = _net_congestion_cells(points, layers, edges, cell_w, cell_h)
        for layer, cells in cell_hits.items():
            for r, c in cells:
                demand[layer][r, c] += 1

    overlap_cost = OVERLAP_WEIGHT * total_overlap_cost(board.components)

    congestion_cost = 0.0
    for layer_demand in demand.values():
        congestion_cost += float(np.sum(np.maximum(0.0, layer_demand - CONGESTION_FREE_CAPACITY) ** 2))

    return CostBreakdown(wirelength=wirelength_cost, overlap=overlap_cost, congestion=congestion_cost)
