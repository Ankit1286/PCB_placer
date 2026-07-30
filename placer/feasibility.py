"""Feasibility checks: is a placed Board actually legal per the spec?

Feasibility (per spec) means, for every component:
  1. Its bounding box lies entirely within the board.
  2. Its center is not inside any keepout zone.
  3. It has zero same-layer overlap with any other component.
  4. Its rotation is one of {0, 90, 180, 270}.

This module is deliberately independent of cost.py: cost measures "how
good," feasibility measures "is this even legal." Both baseline.py and the
learned placer call into here to check their own outputs, and benchmark.py
uses it to verify the final answer before trusting a cost number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placer.board import Board, Component, VALID_ROTATIONS
from placer.cost import rect_overlap_area


@dataclass
class FeasibilityReport:
    """Result of checking a board's placement for legality."""

    is_feasible: bool
    out_of_bounds: list[str] = field(default_factory=list)
    in_keepout: list[str] = field(default_factory=list)
    overlapping_pairs: list[tuple[str, str]] = field(default_factory=list)
    bad_rotation: list[str] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a short human-readable summary of any violations found."""
        if self.is_feasible:
            return "feasible"
        parts = []
        if self.unplaced:
            parts.append(f"{len(self.unplaced)} unplaced")
        if self.out_of_bounds:
            parts.append(f"{len(self.out_of_bounds)} out-of-bounds")
        if self.in_keepout:
            parts.append(f"{len(self.in_keepout)} center-in-keepout")
        if self.overlapping_pairs:
            parts.append(f"{len(self.overlapping_pairs)} overlapping pairs")
        if self.bad_rotation:
            parts.append(f"{len(self.bad_rotation)} bad rotation")
        return "infeasible: " + ", ".join(parts)


def check_feasibility(board: Board) -> FeasibilityReport:
    """Check every feasibility rule for a (supposedly) fully-placed board.

    Returns:
        A FeasibilityReport with `is_feasible=True` iff every component is
        placed, in-bounds, rotation-valid, not centered in a keepout, and
        has zero same-layer overlap with every other component.
    """
    unplaced = [c.component_id for c in board.components if not c.is_placed]
    out_of_bounds = []
    in_keepout = []
    bad_rotation = []

    placed = [c for c in board.components if c.is_placed]

    for c in placed:
        if c.rotation not in VALID_ROTATIONS:
            bad_rotation.append(c.component_id)
            continue  # bbox()/center() below assume a valid rotation

        xmin, ymin, xmax, ymax = c.bbox()
        if xmin < -1e-9 or ymin < -1e-9 or xmax > board.width + 1e-9 or ymax > board.height + 1e-9:
            out_of_bounds.append(c.component_id)

        cx, cy = c.center()
        for kz in board.keepouts:
            if kz.contains_point(cx, cy):
                in_keepout.append(c.component_id)
                break

    overlapping_pairs = []
    valid_placed = [c for c in placed if c.rotation in VALID_ROTATIONS]
    for i in range(len(valid_placed)):
        for j in range(i + 1, len(valid_placed)):
            a, b = valid_placed[i], valid_placed[j]
            if a.layer != b.layer:
                continue
            if rect_overlap_area(a.bbox(), b.bbox()) > 1e-9:
                overlapping_pairs.append((a.component_id, b.component_id))

    is_feasible = not (unplaced or out_of_bounds or in_keepout or overlapping_pairs or bad_rotation)
    return FeasibilityReport(
        is_feasible=is_feasible,
        out_of_bounds=out_of_bounds,
        in_keepout=in_keepout,
        overlapping_pairs=overlapping_pairs,
        bad_rotation=bad_rotation,
        unplaced=unplaced,
    )
