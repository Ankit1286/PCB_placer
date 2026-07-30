"""Random unplaced board generation: `generate_board(config) -> Board`.

Produces a Board with every component's fixed properties (size, pins) and
the netlist set, but placement state (position/rotation/layer) left unset —
that's the placer's job.
A few points in the spec are genuinely ambiguous (how exactly nets should be
formed, how pins are distributed along edges). Where we had to make a call,
it's documented in a comment at the decision point and in DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from placer.board import Board, Component, KeepoutZone, Net, Pin

MIN_COMPONENT_DIM = 0.5  # mm
MAX_COMPONENT_DIM = 20.0  # mm
MIN_PINS = 2
MAX_PINS = 128
MIN_NET_ARITY = 2
MAX_NET_ARITY = 8
MIN_NET_WEIGHT = 0.1
MAX_NET_WEIGHT = 10.0


@dataclass
class GeneratorConfig:
    """Parameters for `generate_board`.

    Args:
        num_components: |V|, number of components on the board (spec range [50, 1000]).
        target_utilization: fraction of board area that should be covered by component area (spec: 0.4-0.7).
        seed: RNG seed, for reproducibility — the same seed always yields the same board.
        num_keepouts: how many rectangular keepout zones to place (spec: 2-5).
    """

    num_components: int
    target_utilization: float = 0.55
    seed: int = 0
    num_keepouts: int | None = None  # None -> sampled uniformly in [2, 5]


def _log_uniform(rng: np.random.Generator, low: float, high: float, size=None):
    """Sample from a log-uniform distribution on [low, high] (equal probability per order of magnitude)."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def _make_pins(rng: np.random.Generator, width: float, height: float) -> list[Pin]:
    """Distribute 2-128 pins along one or two edges of a component's footprint.

    Decision: pin count is drawn log-uniform in [MIN_PINS, MAX_PINS] (mirrors
    the log-uniform sizing elsewhere in the spec, so most components have a
    handful of pins and a few "connector-like" components have many). Pins
    are placed along either one randomly chosen edge or two opposite edges
    ("one or both sides" per spec), evenly spaced with small jitter, and
    alternate top/bottom layer.
    """
    num_pins = int(round(_log_uniform(rng, MIN_PINS, MAX_PINS)))
    num_pins = max(MIN_PINS, min(MAX_PINS, num_pins))

    # Edges: 0=bottom, 1=right, 2=top, 3=left.
    if rng.random() < 0.5:
        edges = [rng.integers(0, 4)]
    else:
        first = rng.integers(0, 4)
        edges = [first, (first + 2) % 4]  # opposite edge

    pins: list[Pin] = []
    per_edge = int(np.ceil(num_pins / len(edges)))
    remaining = num_pins
    for edge in edges:
        n_this_edge = min(per_edge, remaining)
        if n_this_edge <= 0:
            break
        # Evenly spaced positions along the edge, inset slightly from the corners.
        t = np.linspace(0.1, 0.9, n_this_edge) if n_this_edge > 1 else np.array([0.5])
        for frac in t:
            if edge == 0:  # bottom
                lx, ly = frac * width, 0.0
            elif edge == 1:  # right
                lx, ly = width, frac * height
            elif edge == 2:  # top
                lx, ly = frac * width, height
            else:  # left
                lx, ly = 0.0, frac * height
            layer = "top" if rng.random() < 0.5 else "bottom"
            pins.append(Pin(local_x=float(lx), local_y=float(ly), layer=layer))
        remaining -= n_this_edge

    return pins[:num_pins] if len(pins) > num_pins else pins


def _make_components(rng: np.random.Generator, num_components: int) -> list[Component]:
    """Create `num_components` components with log-uniform width/height and randomly placed pins."""
    components = []
    for i in range(num_components):
        w = float(_log_uniform(rng, MIN_COMPONENT_DIM, MAX_COMPONENT_DIM))
        h = float(_log_uniform(rng, MIN_COMPONENT_DIM, MAX_COMPONENT_DIM))
        pins = _make_pins(rng, w, h)
        components.append(Component(component_id=f"C{i}", width=w, height=h, pins=pins))
    return components


def _make_board_dims(rng: np.random.Generator, components: list[Component], target_utilization: float) -> tuple[float, float]:
    """Pick board (width, height) so total component area / board area ~= target_utilization.

    Aspect ratio is drawn randomly within a modest range so boards aren't
    all perfect squares.
    """
    total_area = sum(c.width * c.height for c in components)
    board_area = total_area / target_utilization
    aspect = float(_log_uniform(rng, 0.5, 2.0))  # width/height ratio
    height = float(np.sqrt(board_area / aspect))
    width = float(board_area / height)
    return width, height


def _make_keepouts(rng: np.random.Generator, width: float, height: float, num_keepouts: int) -> list[KeepoutZone]:
    """Place `num_keepouts` random rectangular keepout zones, each small relative to the board.

    Each keepout is capped at 5% of board area so a handful of them can't
    plausibly make the board infeasible to pack.
    """
    keepouts = []
    max_kw = width * 0.2
    max_kh = height * 0.2
    for _ in range(num_keepouts):
        kw = rng.uniform(width * 0.02, max_kw)
        kh = rng.uniform(height * 0.02, max_kh)
        kx = rng.uniform(0, max(width - kw, 0.0))
        ky = rng.uniform(0, max(height - kh, 0.0))
        keepouts.append(KeepoutZone(kx, ky, kx + kw, ky + kh))
    return keepouts


def _make_nets(rng: np.random.Generator, components: list[Component]) -> dict[str, Net]:
    """Generate a netlist with |E| ~= 2|V| via preferential attachment on pin count.

    Decision (documented ambiguity): the spec says "higher pin count => more
    nets" but doesn't fully specify net formation. We interpret this as
    classic preferential attachment: pick pins with probability proportional
    to their parent component's total pin count (so high-pin-count
    "connector-like" parts end up in more nets), each pin used by at most one
    net (a physical pad can only belong to one net), net arity in [2, 8].
    """
    num_nets = 2 * len(components)

    # Flat list of (component_index, pin_index) for every pin, plus a weight
    # per pin equal to its parent's pin count (this is what "preferential
    # attachment on pin count" operationalizes: components with more pins
    # contribute more, individually-equal-weight pins to the selection pool).
    all_pins = []
    weights = []
    for ci, c in enumerate(components):
        n_pins = len(c.pins)
        for pi in range(n_pins):
            all_pins.append((ci, pi))
            weights.append(n_pins)
    weights = np.array(weights, dtype=float)

    available = np.ones(len(all_pins), dtype=bool)
    nets: dict[str, Net] = {}
    net_idx = 0
    attempts = 0
    max_attempts = num_nets * 20

    while net_idx < num_nets and attempts < max_attempts:
        attempts += 1
        arity = int(rng.integers(MIN_NET_ARITY, MAX_NET_ARITY + 1))
        avail_idx = np.nonzero(available)[0]
        if len(avail_idx) < arity:
            break
        w = weights[avail_idx]
        p = w / w.sum()
        chosen = rng.choice(avail_idx, size=arity, replace=False, p=p)
        available[chosen] = False
        pin_refs = [(components[all_pins[idx][0]].component_id, all_pins[idx][1]) for idx in chosen]
        weight = float(_log_uniform(rng, MIN_NET_WEIGHT, MAX_NET_WEIGHT))
        name = f"N{net_idx}"
        nets[name] = Net(name=name, pin_refs=pin_refs, weight=weight)
        net_idx += 1

    return nets


def generate_board(config: GeneratorConfig) -> Board:
    """Generate a random unplaced board from a `GeneratorConfig`.

    Returns:
        A `Board` with components, pins, netlist, dimensions, and keepouts
        all set, but no component placed (position/rotation/layer are None).
    """
    if not (50 <= config.num_components <= 1000):
        raise ValueError("num_components must be in [50, 1000] per spec")

    rng = np.random.default_rng(config.seed)

    components = _make_components(rng, config.num_components)
    width, height = _make_board_dims(rng, components, config.target_utilization)
    num_keepouts = config.num_keepouts if config.num_keepouts is not None else int(rng.integers(2, 6))
    keepouts = _make_keepouts(rng, width, height, num_keepouts)
    nets = _make_nets(rng, components)

    return Board(width=width, height=height, components=components, nets=nets, keepouts=keepouts)
