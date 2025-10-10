import random
import statistics
from typing import Any, Callable
from tablut import Board, Player, MAX_PAWN_MOVES, WHITE_PIECES, BLACK_PIECES
from utils import rescale


def heuristic(state: Board, playing_as: Player) -> float:
    black_piece_count = state.piece_count(Player.BLACK) / BLACK_PIECES
    black_move_count = state.moves_count_ratio(Player.BLACK)
    white_piece_count = state.piece_count_ratio(Player.WHITE) / WHITE_PIECES
    white_move_count = state.moves_count_ratio(Player.WHITE)
    king_moves = state.king_moves() / MAX_PAWN_MOVES

    if playing_as == Player.WHITE:
        white_piece_count = 1 - white_piece_count
        white_move_count = 1 - white_move_count
        king_moves = 1 - king_moves
    else:
        black_piece_count = 1 - black_piece_count
        black_move_count = 1 - black_move_count

    heuristics = [
        white_move_count,
        white_piece_count,
        king_moves,
        black_move_count,
        black_piece_count,
    ]
    heuristics = [rescale((0, 1), (-1, 1), h) for h in heuristics]

    return statistics.mean(heuristics)


def max_depth_criterion(state: Board, depth: int) -> bool:
    return depth > 10


def move_sequence(state: Board, player: Player) -> list:
    moves = state.generate_all_moves(player)
    random.shuffle(moves)
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
