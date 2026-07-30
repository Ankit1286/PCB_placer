"""Tests for placer.cost, including a hand-verified Steiner-tree example.

The 3-point example below is a classic textbook case where the optimal
rectilinear Steiner tree is strictly shorter than the rectilinear minimum
*spanning* tree (the >4-pin heuristic) — it's the case that justifies why
we bother with the exact Hanan-grid solver at all for small nets.
"""

import numpy as np
import pytest

from placer.cost import (
    _exact_rsmt_length,
    _rectilinear_mst,
    rect_overlap_area,
    rsmt_and_crossings,
    total_overlap_cost,
)
from placer.board import Component


def test_rect_overlap_area_no_overlap():
    assert rect_overlap_area((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_rect_overlap_area_partial_overlap():
    # Two 2x2 boxes overlapping in a 1x2 strip.
    assert rect_overlap_area((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(2.0)


def test_two_pin_rsmt_is_manhattan_distance():
    points = np.array([[0.0, 0.0], [3.0, 4.0]])
    length, edges = _rectilinear_mst(points)
    assert length == pytest.approx(7.0)
    assert edges == [(0, 1)]


def test_three_pin_steiner_tree_beats_spanning_tree():
    # (0,0), (2,0), (1,2): optimal RSMT uses Steiner point (1,0) -> length 4.
    # The naive rectilinear MST (no Steiner point) can only reach length 5.
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    mst_length, _ = _rectilinear_mst(points)
    exact_length = _exact_rsmt_length(points)
    assert mst_length == pytest.approx(5.0)
    assert exact_length == pytest.approx(4.0)
    # Sanity check on the documented 1.5x worst-case approximation ratio.
    assert mst_length <= 1.5 * exact_length + 1e-9


def test_rsmt_and_crossings_counts_layer_changes():
    points = np.array([[0.0, 0.0], [3.0, 0.0]])
    length, crossings = rsmt_and_crossings(points, ["top", "top"])
    assert length == pytest.approx(3.0)
    assert crossings == 0

    length, crossings = rsmt_and_crossings(points, ["top", "bottom"])
    assert crossings == 1


def _placed(x, y, w, h, layer="top"):
    c = Component(component_id=f"c{x}{y}", width=w, height=h, pins=[])
    c.x, c.y, c.rotation, c.layer = x, y, 0, layer
    return c


def test_total_overlap_cost_same_layer_pair():
    a = _placed(0, 0, 2, 2, layer="top")
    b = _placed(1, 0, 2, 2, layer="top")
    assert total_overlap_cost([a, b]) == pytest.approx(2.0)  # 1x2 overlap strip


def test_total_overlap_cost_ignores_different_layers():
    a = _placed(0, 0, 2, 2, layer="top")
    b = _placed(1, 0, 2, 2, layer="bottom")
    assert total_overlap_cost([a, b]) == pytest.approx(0.0)
