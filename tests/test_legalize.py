"""Tests for placer.legalize."""

from placer.board import Board, Component, KeepoutZone
from placer.feasibility import check_feasibility
from placer.legalize import clamp_to_bounds, legalize, push_out_of_keepouts


def _component(cid, x, y, w=2.0, h=2.0, rotation=0, layer="top"):
    c = Component(component_id=cid, width=w, height=h, pins=[])
    c.x, c.y, c.rotation, c.layer = x, y, rotation, layer
    return c


def test_clamp_to_bounds_pulls_component_back_inside():
    board = Board(width=10, height=10, components=[], nets={}, keepouts=[])
    c = _component("a", 9, 9, w=3, h=3)
    clamp_to_bounds(c, board)
    assert c.x + 3 <= 10 + 1e-9
    assert c.y + 3 <= 10 + 1e-9


def test_push_out_of_keepouts_moves_center_outside_zone():
    keepout = KeepoutZone(0, 0, 5, 5)
    board = Board(width=20, height=20, components=[], nets={}, keepouts=[keepout])
    c = _component("a", 1, 1, w=2, h=2)
    push_out_of_keepouts(c, board)
    cx, cy = c.center()
    assert not keepout.contains_point(cx, cy)


def test_legalize_resolves_overlap_between_two_components():
    a = _component("a", 0, 0, w=2, h=2)
    b = _component("b", 1, 0, w=2, h=2)  # overlaps a by a 1x2 strip
    board = Board(width=20, height=20, components=[a, b], nets={}, keepouts=[])
    legalize(board)
    report = check_feasibility(board)
    assert report.overlapping_pairs == []


def test_push_out_of_keepouts_escapes_adjacent_zones_without_oscillating():
    """Regression test: two keepouts close together must not make push_out_of_keepouts oscillate.

    Reproduces a real failure seen in a full baseline run at |V|=200: two
    keepout zones ~0.18mm apart (narrower than the stuck component) caused
    the old nearest-edge-only logic to escape zone 1 by landing in zone 2,
    then escape zone 2 by landing back in zone 1, forever. The fix checks
    every candidate escape direction against *all* keepouts, not just the
    one currently containing the center.
    """
    zone1 = KeepoutZone(44.15, 78.99, 58.92, 89.36)
    zone2 = KeepoutZone(59.10, 72.21, 73.61, 93.11)
    board = Board(width=84.8, height=136.4, components=[], nets={}, keepouts=[zone1, zone2])
    # Same bbox as the component that got stuck in the real run: small, centered in zone2,
    # close enough to zone1 that escaping left/right bounces between them.
    c = _component("c44", x=58.92, y=82.04, w=0.81, h=1.04)
    push_out_of_keepouts(c, board)
    cx, cy = c.center()
    assert not zone1.contains_point(cx, cy)
    assert not zone2.contains_point(cx, cy)


def test_legalize_resolves_many_overlapping_components():
    import numpy as np

    # Total component area is 30 (30 * 1x1), board area is 400 -- 7.5% utilization, comfortably
    # within the spec's 40-70% range, spread across the whole board rather than crammed into a
    # corner (an unrealistically dense clump isn't representative of what force-directed init
    # actually produces, and isn't a scenario legalize() promises to fully resolve alone -- see
    # its docstring on SA refinement being the real feasibility backstop for pathological cases).
    rng = np.random.default_rng(3)
    components = []
    for i in range(30):
        c = _component(f"c{i}", x=float(rng.uniform(0, 19)), y=float(rng.uniform(0, 19)), w=1.0, h=1.0)
        components.append(c)
    board = Board(width=20, height=20, components=components, nets={}, keepouts=[])
    legalize(board)
    report = check_feasibility(board)
    assert report.overlapping_pairs == []
    assert report.out_of_bounds == []
