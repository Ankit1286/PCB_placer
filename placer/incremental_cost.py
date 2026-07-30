"""Incremental (delta) cost tracking for simulated annealing.

`compute_cost` in cost.py recomputes the whole board from scratch (~2.3s at
|V|=1000). Simulated annealing needs 50,000 perturbations, each touching one
component — recomputing the whole board every time would take on the order
of a day. `IncrementalCost` instead keeps running totals and, after any
single component is mutated, only recomputes the handful of nets that
component belongs to (its wirelength + congestion contribution) and that
component's own overlap contribution against the rest of its layer (O(V),
not O(V^2), since every other pairwise overlap is unaffected by this move).

This is the same machinery the learned placer's budgeted refinement pass
reuses at inference time, where speed matters even more (60s hard limit).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from placer.board import Board, Component, Net
from placer.cost import (
    CONGESTION_FREE_CAPACITY,
    GRID_SIZE,
    NU_VIA,
    OVERLAP_WEIGHT,
    _net_congestion_cells,
    _rectilinear_mst,
    overlap_against_many,
    rsmt_and_crossings,
)


def _cell_cost(demand: float) -> float:
    """Congestion cost contribution of a single grid cell at a given demand level."""
    return max(0.0, demand - CONGESTION_FREE_CAPACITY) ** 2


class IncrementalCost:
    """Maintains C(p) for a fully-placed board and updates it cheaply after single-component moves."""

    def __init__(self, board: Board):
        self.board = board
        self.cell_w = board.width / GRID_SIZE
        self.cell_h = board.height / GRID_SIZE
        self.demand = {"top": np.zeros((GRID_SIZE, GRID_SIZE)), "bottom": np.zeros((GRID_SIZE, GRID_SIZE))}
        self._net_cost: dict[str, float] = {}
        self._net_cells: dict[str, dict[str, list[tuple[int, int]]]] = {}
        self.component_nets: dict[str, list[str]] = defaultdict(list)

        self.wirelength_total = 0.0
        self.congestion_total = 0.0
        self.overlap_total = 0.0

        for net in board.nets.values():
            for cid, _ in net.pin_refs:
                if net.name not in self.component_nets[cid]:
                    self.component_nets[cid].append(net.name)
            self._recompute_net(net)

        self._recompute_overlap_from_scratch()

    def _net_geometry(self, net: Net) -> tuple[np.ndarray, list[str]]:
        pin_data = [
            self.board.component_by_id(cid).pin_board_position(self.board.component_by_id(cid).pins[pidx])
            for cid, pidx in net.pin_refs
        ]
        points = np.array([[x, y] for x, y, _ in pin_data])
        layers = [l for _, _, l in pin_data]
        return points, layers

    def _add_cells(self, cells: dict[str, list[tuple[int, int]]]) -> None:
        for layer, cell_list in cells.items():
            grid = self.demand[layer]
            for r, c in cell_list:
                old_d = grid[r, c]
                new_d = old_d + 1
                self.congestion_total += _cell_cost(new_d) - _cell_cost(old_d)
                grid[r, c] = new_d

    def _remove_cells(self, cells: dict[str, list[tuple[int, int]]]) -> None:
        for layer, cell_list in cells.items():
            grid = self.demand[layer]
            for r, c in cell_list:
                old_d = grid[r, c]
                new_d = old_d - 1
                self.congestion_total += _cell_cost(new_d) - _cell_cost(old_d)
                grid[r, c] = new_d

    def _recompute_net(self, net: Net) -> None:
        """Fully recompute one net's wirelength+congestion contribution and update running totals."""
        if net.name in self._net_cost:
            self.wirelength_total -= self._net_cost[net.name]
            self._remove_cells(self._net_cells[net.name])

        points, layers = self._net_geometry(net)
        length, crossings = rsmt_and_crossings(points, layers)
        cost = net.weight * (length + crossings * NU_VIA)
        _, edges = _rectilinear_mst(points)
        cells = _net_congestion_cells(points, layers, edges, self.cell_w, self.cell_h)

        self._add_cells(cells)
        self._net_cost[net.name] = cost
        self._net_cells[net.name] = cells
        self.wirelength_total += cost

    def _recompute_overlap_from_scratch(self) -> None:
        from placer.cost import total_overlap_cost

        self.overlap_total = total_overlap_cost(self.board.components)

    def component_moved(
        self,
        component: Component,
        old_bbox: tuple[float, float, float, float],
        old_layer: str,
    ) -> None:
        """Call after mutating `component`'s x/y/rotation/layer to bring all running totals up to date.

        `old_bbox`/`old_layer` must be captured (via `component.bbox()` /
        `component.layer`) *before* the mutation was applied.
        """
        new_bbox = component.bbox()
        new_layer = component.layer

        others_old_layer = [
            c.bbox() for c in self.board.components if c.component_id != component.component_id and c.layer == old_layer
        ]
        others_new_layer = (
            others_old_layer
            if new_layer == old_layer
            else [c.bbox() for c in self.board.components if c.component_id != component.component_id and c.layer == new_layer]
        )

        old_contrib = overlap_against_many(old_bbox, others_old_layer)
        new_contrib = overlap_against_many(new_bbox, others_new_layer)
        self.overlap_total += new_contrib - old_contrib

        for net_name in self.component_nets[component.component_id]:
            self._recompute_net(self.board.nets[net_name])

    @property
    def total(self) -> float:
        """Total placement cost C(p), kept in sync incrementally after each `component_moved` call."""
        return self.wirelength_total + OVERLAP_WEIGHT * self.overlap_total + self.congestion_total
