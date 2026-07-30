"""Tests for the coordinate transform in placer.board.Component.pin_board_position."""

from placer.board import Component, Pin


def make_component(x=0.0, y=0.0, rotation=0, layer="top"):
    c = Component(
        component_id="C0",
        width=4.0,
        height=2.0,
        pins=[Pin(local_x=1.0, local_y=0.0, layer="top")],
    )
    c.x, c.y, c.rotation, c.layer = x, y, rotation, layer
    return c


def test_footprint_swaps_dims_on_90_and_270():
    c = make_component(rotation=0)
    assert c.footprint() == (4.0, 2.0)
    c.rotation = 90
    assert c.footprint() == (2.0, 4.0)
    c.rotation = 180
    assert c.footprint() == (4.0, 2.0)
    c.rotation = 270
    assert c.footprint() == (2.0, 4.0)


def test_pin_position_no_rotation_no_offset():
    c = make_component(x=0.0, y=0.0, rotation=0, layer="top")
    bx, by, layer = c.pin_board_position(c.pins[0])
    assert (bx, by, layer) == (1.0, 0.0, "top")


def test_pin_position_translates_with_component_position():
    c = make_component(x=10.0, y=5.0, rotation=0, layer="top")
    bx, by, layer = c.pin_board_position(c.pins[0])
    assert (bx, by) == (11.0, 5.0)


def test_pin_position_rotation_90():
    # local (1, 0) on a 4x2 component rotated 90deg should map to (0, 4-1) = (0, 3) pre-translation.
    c = make_component(x=0.0, y=0.0, rotation=90, layer="top")
    bx, by, _ = c.pin_board_position(c.pins[0])
    assert (bx, by) == (0.0, 3.0)


def test_pin_position_rotation_180():
    # local (1, 0) on a 4x2 rotated 180 -> (4-1, 2-0) = (3, 2).
    c = make_component(x=0.0, y=0.0, rotation=180, layer="top")
    bx, by, _ = c.pin_board_position(c.pins[0])
    assert (bx, by) == (3.0, 2.0)


def test_pin_position_bottom_layer_mirrors_x_and_flips_pin_layer():
    # Bottom-mounted component: local x=1 on width=4 mirrors to 4-1=3; pin's own "top" flips to "bottom".
    c = make_component(x=0.0, y=0.0, rotation=0, layer="bottom")
    bx, by, layer = c.pin_board_position(c.pins[0])
    assert (bx, by) == (3.0, 0.0)
    assert layer == "bottom"


def test_is_placed_false_until_all_fields_set():
    c = Component(component_id="C1", width=1.0, height=1.0, pins=[])
    assert not c.is_placed
    c.x, c.y, c.rotation, c.layer = 0.0, 0.0, 0, "top"
    assert c.is_placed
