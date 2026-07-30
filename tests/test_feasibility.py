"""Tests for placer.feasibility.check_feasibility."""

from placer.board import Board, Component, KeepoutZone
from placer.feasibility import check_feasibility


def _component(cid, x, y, w=2.0, h=2.0, rotation=0, layer="top"):
    c = Component(component_id=cid, width=w, height=h, pins=[])
    c.x, c.y, c.rotation, c.layer = x, y, rotation, layer
    return c


def _board(components, keepouts=None, width=20.0, height=20.0):
    return Board(width=width, height=height, components=components, nets={}, keepouts=keepouts or [])


def test_fully_valid_placement_is_feasible():
    b = _board([_component("a", 0, 0), _component("b", 10, 10)])
    report = check_feasibility(b)
    assert report.is_feasible
    assert report.summary() == "feasible"


def test_unplaced_component_is_infeasible():
    c = Component(component_id="a", width=1.0, height=1.0, pins=[])
    b = _board([c])
    report = check_feasibility(b)
    assert not report.is_feasible
    assert "a" in report.unplaced


def test_out_of_bounds_detected():
    b = _board([_component("a", 19, 19, w=5, h=5)], width=20, height=20)
    report = check_feasibility(b)
    assert not report.is_feasible
    assert "a" in report.out_of_bounds


def test_negative_position_out_of_bounds():
    b = _board([_component("a", -1, 0)])
    report = check_feasibility(b)
    assert "a" in report.out_of_bounds


def test_keepout_violation_detected():
    keepout = KeepoutZone(0, 0, 5, 5)
    b = _board([_component("a", 0, 0, w=2, h=2)], keepouts=[keepout])
    report = check_feasibility(b)
    assert not report.is_feasible
    assert "a" in report.in_keepout


def test_same_layer_overlap_detected():
    b = _board([_component("a", 0, 0, layer="top"), _component("b", 1, 0, layer="top")])
    report = check_feasibility(b)
    assert not report.is_feasible
    assert ("a", "b") in report.overlapping_pairs


def test_different_layer_overlap_is_allowed():
    b = _board([_component("a", 0, 0, layer="top"), _component("b", 1, 0, layer="bottom")])
    report = check_feasibility(b)
    assert report.is_feasible


def test_bad_rotation_detected():
    b = _board([_component("a", 0, 0, rotation=45)])
    report = check_feasibility(b)
    assert not report.is_feasible
    assert "a" in report.bad_rotation
