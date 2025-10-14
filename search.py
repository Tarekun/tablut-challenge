import random
import statistics
from typing import Callable
from tablut import GameState, Player, MAX_PAWN_MOVES, WHITE_PIECES, BLACK_PIECES
from utils import rescale


def alpha_beta(
    heuristic: Callable[[GameState], float],
    stop_criterion: Callable[[GameState, int], bool],
    move_sequence: Callable[[GameState], list[GameState]],
):
    """
    Minmax search algorithm with alpha/beta pruning.

    Parameters
    ----------
    heuristic : GameState -> [-1, 1]
        State estimation function that evaluates a GameState and returns a
        float between -1 and 1, where positive values favor the maximizing player
        and negative values favor the minimizing player.
    stop_criterion : [GameState, int] -> bool
        Boolean function that determines whether to stop search at a node.
        Takes current GameState and current search depth as input and returns
        True if search should stop (leaf node or depth limit reached).
    move_sequence : GameState -> list[GameState]
        Policy function that generates and orders possible moves from a given
        GameState. Proper ordering is crucial for effective alpha-beta pruning.

    Returns
    -------
    GameState -> GameState
        Function that takes current GameState as input and returns the
        optimal next GameState according to the minmax search with alpha-beta pruning.
    """

    # TODO fix this list | None
    def max_value(state, alpha, beta, depth) -> tuple[GameState, float]:
        if stop_criterion(state, depth):
            # probabile va messa una vera valutazione qua, specie se finise il gioco
            return (state, heuristic(state))

        local_best = float("-inf")
        best_move = None
        print(" " * depth, end="")
        for move in move_sequence(state):
            (_, value) = min_value(move, alpha, beta, depth + 1)
            if value > local_best:
                local_best = value
                best_move = move
            if local_best >= beta:
                return (best_move, local_best)  # type: ignore
            alpha = max(alpha, local_best)

        return (best_move, local_best)  # type: ignore

    # TODO fix this list | None
    def min_value(state, alpha, beta, depth) -> tuple[GameState, float]:
        if stop_criterion(state, depth):
            # probabile va messa una vera valutazione qua, specie se finise il gioco
            return (state, heuristic(state))

        local_best = float("inf")
        best_move = None
        print(" " * depth, end="")
        for move in move_sequence(state):
            (_, value) = max_value(move, alpha, beta, depth + 1)
            if value < local_best:
                local_best = value
                best_move = move
            if local_best <= alpha:
                return (best_move, local_best)  # type: ignore
            beta = min(beta, local_best)

        return (best_move, local_best)  # type: ignore

    def search_algorithm(state: GameState) -> GameState:
        (best_move, _) = max_value(state, float("-inf"), float("inf"), 0)
        return best_move

    return search_algorithm


def monte_carlo_tree_search(
    heuristic: Callable[[GameState], float],
    stop_criterion: Callable[[GameState, int], bool],
    move_sequence: Callable[[GameState], list[GameState]],
):
    """
    Monte Carlo Tree Search (MCTS) algorithm.

    Parameters
    ----------
    TODO

    Returns
    -------
    GameState -> GameState
        Function that takes current GameState as input and returns the
        optimal next GameState according to the minmax search with alpha-beta pruning.
    """
    pass
