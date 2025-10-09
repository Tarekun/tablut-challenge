from typing import Dict, Any, Callable
from tablut import Board, Player, BOARD_LENGTH


def heuristic(state: Board) -> float:
    return 0.0


def max_depth_criterion(state: Board, depth: int) -> bool:
    return depth > 10


def generate_all_moves(state: Board, player: Player) -> list:
    moves = []
    for row in range(1, BOARD_LENGTH):
        for col in range(1, BOARD_LENGTH):
            if player.owns_pawn(state[row][col]):
                pawn_moves = state.pawn_moves(row, col)
                moves.extend(pawn_moves)

    return moves


def alpha_beta(
    heuristic: Callable[[Any], float],
    stop_criterion: Callable[[Any, int], bool],
    move_sequence: Callable[[Any, Player], list],
):
    def max_value(
        state, player: Player, alpha, beta, depth
    ) -> tuple[list | None, float]:  # TODO fix this list | None
        if stop_criterion(state, depth):
            # probabile va messa una vera valutazione qua, specie se finise il gioco
            return (state, heuristic(state))

        local_best = float("-inf")
        best_move = None
        for move in move_sequence(state, player):
            (_, value) = min_value(move, player.complement(), alpha, beta, depth + 1)
            if value > local_best:
                local_best = value
                best_move = move
            if local_best >= beta:
                return (best_move, local_best)
            alpha = max(alpha, local_best)

        return (best_move, local_best)

    def min_value(
        state, player: Player, alpha, beta, depth
    ) -> tuple[list | None, float]:
        if stop_criterion(state, depth):
            # probabile va messa una vera valutazione qua, specie se finise il gioco
            return (state, heuristic(state))

        local_best = float("inf")
        best_move = None
        for move in move_sequence(state, player):
            (_, value) = max_value(move, player.complement(), alpha, beta, depth + 1)
            if value < local_best:
                local_best = value
                best_move = move
            if local_best <= alpha:
                return (best_move, local_best)
            beta = min(beta, local_best)

        return (best_move, local_best)

    def search_algorithm(state, player: Player):
        (best_move, _) = max_value(state, player, float("-inf"), float("inf"), 0)
        return best_move

    return search_algorithm
