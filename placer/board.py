"""Data model for a PCB (printed circuit board) and its components.

A `Board` is the whole circuit board: its rectangular size, a list of
`Component`s (the physical parts, e.g. resistors/chips) sitting on it, the
`Net`s that specify which pins must be electrically connected, and the
`KeepoutZone`s (regions nothing may be centered in, e.g. reserved for a
mounting screw). `generate_board()` (see generator.py) builds one of these
with every component's position/rotation/layer left unset ("unplaced"); a
placer's job is to fill those in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Rotations are always one of these four multiples of 90 degrees.
VALID_ROTATIONS = (0, 90, 180, 270)
LAYERS = ("top", "bottom")


@dataclass
class Pin:
    """A single connection point on a component, in the component's own local frame.

    Local frame means (local_x, local_y) are measured from the component's
    bottom-left corner *before* any rotation/position is applied — see
    `Component.pin_board_position` for the transform into board coordinates.
    """

    local_x: float
    local_y: float
    layer: str  # "top" or "bottom", relative to the component itself

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"Pin layer must be one of {LAYERS}, got {self.layer!r}")


@dataclass
class Component:
    """A rectangular part with pins. Width/height are fixed; position/rotation/layer are set by a placer."""

    component_id: str
    width: float
    height: float
    pins: list[Pin]

    # Mutable placement state. `None` means "not yet placed".
    x: float | None = None
    y: float | None = None
    rotation: int | None = None  # degrees, one of VALID_ROTATIONS
    layer: str | None = None  # "top" or "bottom"

    @property
    def is_placed(self) -> bool:
        """True once position, rotation, and layer have all been assigned."""
        return self.x is not None and self.y is not None and self.rotation is not None and self.layer is not None

    def footprint(self) -> tuple[float, float]:
        """Return (width, height) of this component's axis-aligned bounding box after rotation.

        Rotation is always a multiple of 90 degrees, so a rotated rectangle's
        bounding box is just the original box with width/height swapped for
        90/270 degrees (no partial-angle trigonometry needed).
        """
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height

    def bbox(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) of this component's placed bounding box."""
        if not self.is_placed:
            raise ValueError(f"Component {self.component_id} is not placed yet")
        w, h = self.footprint()
        return self.x, self.y, self.x + w, self.y + h

    def center(self) -> tuple[float, float]:
        """Return the (x, y) board-space center of this component's bounding box."""
        xmin, ymin, xmax, ymax = self.bbox()
        return (xmin + xmax) / 2, (ymin + ymax) / 2

    def pin_board_position(self, pin: Pin) -> tuple[float, float, str]:
        """Map a pin's local-frame coordinates to board-space (x, y) and effective layer.

        Applies, in order: mirroring (if the component is flipped to the
        bottom layer, the board is flipped left-right, like flipping a
        physical part over), rotation about the component's own bottom-left
        corner, then translation by the component's placed position. A
        pin's own layer is relative to the component, so it too gets
        mirrored (top<->bottom) when the component is on the bottom.

        Returns:
            (board_x, board_y, effective_layer)
        """
        if not self.is_placed:
            raise ValueError(f"Component {self.component_id} is not placed yet")

        lx, ly = pin.local_x, pin.local_y
        w, h = self.width, self.height

        # Mirror left-right if the component is flipped to the bottom (like
        # looking at the part from the other side of the board).
        if self.layer == "bottom":
            lx = w - lx
            effective_pin_layer = "top" if pin.layer == "bottom" else "bottom"
        else:
            effective_pin_layer = pin.layer

        # Rotate (lx, ly) about the origin by self.rotation, then account for
        # the fact that rotating swaps which corner is "bottom-left".
        rot = self.rotation
        if rot == 0:
            rx, ry = lx, ly
        elif rot == 90:
            rx, ry = ly, w - lx
        elif rot == 180:
            rx, ry = w - lx, h - ly
        elif rot == 270:
            rx, ry = h - ly, lx
        else:
            raise ValueError(f"Invalid rotation {rot}")

        return self.x + rx, self.y + ry, effective_pin_layer


@dataclass
class Net:
    """A required electrical connection between 2-8 pins across one or more components."""

    name: str
    pin_refs: list[tuple[str, int]]  # (component_id, pin_index into that component's .pins list)
    weight: float  # criticality weight, omega_e in the cost function


@dataclass
class KeepoutZone:
    """A rectangular region where no component's center may fall."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains_point(self, x: float, y: float) -> bool:
        """Return True if (x, y) lies within this keepout rectangle."""
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class Board:
    """A full PCB: dimensions, components, netlist, and keepout zones."""

    width: float
    height: float
    components: list[Component]
    nets: dict[str, Net]
    keepouts: list[KeepoutZone]
    _index: dict[str, Component] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Cache id -> Component. Safe because the component *list* is fixed after
        # generation; only their placement fields (x/y/rotation/layer) mutate.
        # Without this, component_by_id was a measured hot-path bottleneck: an
        # O(V) linear scan called ~100x per SA perturbation during profiling.
        self._index = {c.component_id: c for c in self.components}

    def component_by_id(self, component_id: str) -> Component:
        """Look up a component by its id in O(1)."""
        try:
            return self._index[component_id]
        except KeyError:
            raise KeyError(f"No component with id {component_id!r}") from None

    def is_fully_placed(self) -> bool:
        """True if every component on the board has been placed."""
        return all(c.is_placed for c in self.components)

    def net_pin_positions(self, net: Net) -> list[tuple[float, float, str]]:
        """Return board-space (x, y, layer) for every pin referenced by a net."""
        positions = []
        for component_id, pin_index in net.pin_refs:
            component = self.component_by_id(component_id)
            pin = component.pins[pin_index]
            positions.append(component.pin_board_position(pin))
        return positions

    def copy_unplaced(self) -> "Board":
        """Return a deep-enough copy of this board with all placement state cleared.

        Useful for running multiple placers against the exact same board
        layout (same components/pins/nets/keepouts) without one placer's
        mutations leaking into another's.
        """
        new_components = [
            Component(
                component_id=c.component_id,
                width=c.width,
                height=c.height,
                pins=[Pin(p.local_x, p.local_y, p.layer) for p in c.pins],
            )
            for c in self.components
        ]
        new_nets = {
            name: Net(name=n.name, pin_refs=list(n.pin_refs), weight=n.weight)
            for name, n in self.nets.items()
        }
        new_keepouts = [KeepoutZone(k.xmin, k.ymin, k.xmax, k.ymax) for k in self.keepouts]
        return Board(
            width=self.width,
            height=self.height,
            components=new_components,
            nets=new_nets,
            keepouts=new_keepouts,
        )
