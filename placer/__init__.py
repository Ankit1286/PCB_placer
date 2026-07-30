"""Quilter PCB placer: the two required entry points.

    from placer import generate_board, place

    board = generate_board(GeneratorConfig(num_components=200, seed=0))
    placed_board = place(board)
"""

from placer.generator import GeneratorConfig, generate_board
from placer.learned.placer import place

__all__ = ["generate_board", "GeneratorConfig", "place"]
