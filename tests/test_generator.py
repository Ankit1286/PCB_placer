"""Tests for placer.generator.generate_board."""

import pytest

from placer.generator import GeneratorConfig, generate_board


def test_same_seed_is_reproducible():
    cfg = GeneratorConfig(num_components=60, seed=42)
    b1 = generate_board(cfg)
    b2 = generate_board(cfg)
    assert b1.width == b2.width
    assert b1.height == b2.height
    assert [c.width for c in b1.components] == [c.width for c in b2.components]
    assert set(b1.nets.keys()) == set(b2.nets.keys())


def test_different_seed_gives_different_board():
    b1 = generate_board(GeneratorConfig(num_components=60, seed=1))
    b2 = generate_board(GeneratorConfig(num_components=60, seed=2))
    assert [c.width for c in b1.components] != [c.width for c in b2.components]


def test_component_count_matches_config():
    b = generate_board(GeneratorConfig(num_components=100, seed=0))
    assert len(b.components) == 100


def test_num_components_out_of_range_rejected():
    with pytest.raises(ValueError):
        generate_board(GeneratorConfig(num_components=10, seed=0))
    with pytest.raises(ValueError):
        generate_board(GeneratorConfig(num_components=2000, seed=0))


def test_net_count_is_roughly_2v():
    b = generate_board(GeneratorConfig(num_components=200, seed=0))
    # "|E| ~= 2|V|" -- allow slack since pin exhaustion can cut generation short.
    assert 0.5 * 2 * 200 <= len(b.nets) <= 2 * 200


def test_utilization_within_spec_bounds():
    b = generate_board(GeneratorConfig(num_components=150, seed=0, target_utilization=0.55))
    total_component_area = sum(c.width * c.height for c in b.components)
    utilization = total_component_area / (b.width * b.height)
    assert 0.4 <= utilization <= 0.7


def test_keepout_count_within_spec_bounds():
    b = generate_board(GeneratorConfig(num_components=60, seed=0))
    assert 2 <= len(b.keepouts) <= 5


def test_all_components_unplaced():
    b = generate_board(GeneratorConfig(num_components=60, seed=0))
    assert all(not c.is_placed for c in b.components)


def test_pin_counts_within_spec_bounds():
    b = generate_board(GeneratorConfig(num_components=60, seed=0))
    for c in b.components:
        assert 2 <= len(c.pins) <= 128


def test_net_arity_within_spec_bounds():
    b = generate_board(GeneratorConfig(num_components=60, seed=0))
    for net in b.nets.values():
        assert 2 <= len(net.pin_refs) <= 8
