import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client import parse_state
from tablut import *


# def test_function():
#     state = _read_state("custom_states/test.json")
#     next_states = state.next_moves()
#     assert next_states is not None
#     assert len(next_states) > 0


def test_capture():
    def validate_state(path: str, actives, inactives):
        board = _read_state(path).board

        for row, col in inactives:
            board = copy.deepcopy(board)
            board.solve_captures(row, col)
            assert board == board.previous

        for row, col in actives:
            board = copy.deepcopy(board)
            board.solve_captures(row, col)
            assert board != board.previous

    validate_state(
        "custom_states/captures/44-capture-hor.json", [(4, 5), (4, 3)], [(4, 4), (8, 8)]
    )
    validate_state(
        "custom_states/captures/44-capture-ver.json", [(5, 4), (3, 4)], [(4, 4), (8, 8)]
    )


def _read_state(path: str):
    with open(path, "r") as file:
        initial_state_string = file.read()
        game_state, _ = parse_state(initial_state_string, Player.BLACK)
        return game_state
