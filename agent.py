import random
import statistics
from typing import Callable
from tablut import GameState, Player, MAX_PAWN_MOVES, WHITE_PIECES, BLACK_PIECES
from utils import rescale


def heuristic(state: GameState) -> float:  # [-1, 1]
    board = state.board
    black_piece_count = board.piece_count(Player.BLACK) / BLACK_PIECES
    black_move_count = board.moves_count_ratio(Player.BLACK)
    white_piece_count = board.piece_count_ratio(Player.WHITE) / WHITE_PIECES
    white_move_count = board.moves_count_ratio(Player.WHITE)
    king_moves = board.king_moves() / MAX_PAWN_MOVES

    if state.playing_as == Player.BLACK:
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


def max_depth_criterion(state: GameState, depth: int) -> bool:
    return depth > 2 or state.is_end_state()


def move_sequence(state: GameState) -> list:
    moves = state.next_moves()
    print(f"generated {len(moves)} moves")
    random.shuffle(moves)
    # moves.sort(key=lambda move: heuristic(move), reverse=True)
    return moves[:10]


def alpha_beta(
    heuristic: Callable[[GameState], float],
    stop_criterion: Callable[[GameState, int], bool],
    move_sequence: Callable[[GameState], list[GameState]],
):
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
