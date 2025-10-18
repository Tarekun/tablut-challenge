import random
import statistics
import torch
import numpy as np
from typing import Callable
from search import alpha_beta
from network.model import TablutNet
from tablut import GameState, Player, BLACK_PIECES, WHITE_PIECES, MAX_PAWN_MOVES
from utils import rescale


# This model contains various functions that return a search strategy (ie a function with type
# GameState -> GameState) that can be used as an argument to `client.play_game` to play one
# game with that playstyle. Useful to sample different algorithms when selfplaying

# Most of the components of search algorithms (stopping criterions, action policies and state
# heuristics) are defined to configurable and reusable so one can plug n play different combinations
# of them with the same base search algorithm


################### SEARCH STRATEGIES
def alpha_beta_basic(
    max_depth: int, branching: int
) -> Callable[[GameState], GameState]:
    """Basic Minimax algorithm with alpha/beta cuts. Uses a handfull of averaged
    handpicked heuristic, picks a `branching` number of random actions from the available
    ones and searches up to a `max_depth`"""

    return alpha_beta(
        _handpicked_heuristics,
        _max_depth_criterion(max_depth),
        _random_fixed_actions(branching),
    )


def alpha_beta_value_model(
    model: TablutNet, max_depth: int
) -> Callable[[GameState], GameState]:
    """Minimax algorithm with alpha/beta cuts that uses `model` as a state estimator
    when reaching a node at `max_depth`"""

    return alpha_beta(
        _network_value_heuristic(model),
        _max_depth_criterion(max_depth),
        _random_fixed_actions(10),
    )


def alpha_beta_policy_model(
    model: TablutNet, top_p: float, max_depth: int
) -> Callable[[GameState], GameState]:
    """Minimax algorithm with alpha/beta cuts that uses hand picked averaged heuristics,
    a `max_depth` stopping cretirion, and a top P policy algorithm with the probability
    distributions provided by `model`.
    `model` takes all available actions and returns a probability distribution over them,
    we pick the most probable action to search until the cumulative probability reaches top_p
    """

    return alpha_beta(
        _handpicked_heuristics,
        _max_depth_criterion(max_depth),
        _network_top_p_policy(model, top_p),
    )


def alpha_beta_full_model(model: TablutNet, top_p: float, max_depth: int):
    """Minimax alpha/beta search fully guided by the network `model`.
    It selects actions with cumulative probability of `top_p` according to `model`
    and evaluates states with `model`, up to `max_depth` in the search tree"""
    return alpha_beta(
        _network_value_heuristic(model),
        _max_depth_criterion(max_depth),
        _network_top_p_policy(model, top_p),
    )


def model_value_maximization_search(
    model: TablutNet,
) -> Callable[[GameState], GameState]:
    def maximize_heuristic(state: GameState) -> GameState:
        best_value = float("-inf")
        best_move = None
        moves: list[GameState] = state.next_moves()

        with torch.no_grad():
            (values, probs) = model(moves)  # shape: (N,2)

        # find best move index
        best_idx = int(torch.argmax(values).item())
        best_value = values[best_idx].item()

        #print(f"best value found was: {best_value}")
        return moves[best_idx]

    return maximize_heuristic


################### SEARCH STRATEGIES


################### REUSABLE HEURISTICS
def _handpicked_heuristics(state: GameState) -> float:  # [-1, 1]
    board = state.board
    king_escapes = board.king_escapes()
    king_surr = board.king_surr()

    if king_escapes >= 2:
        return 1 if state.turn_player == Player.WHITE else -1
    if king_surr == 4:
        return 1 if state.turn_player == Player.BLACK else -1
    
    black_piece_count = board.piece_count(Player.BLACK) / BLACK_PIECES
    black_move_count = board.moves_count_ratio(Player.BLACK)
    white_piece_count = board.piece_count_ratio(Player.WHITE) / WHITE_PIECES
    white_move_count = board.moves_count_ratio(Player.WHITE)
    king_moves = board.king_moves() / MAX_PAWN_MOVES
    king_escapes /= 4
    king_surr /= 4

    if state.playing_as == Player.BLACK:
        white_piece_count = 1 - white_piece_count
        white_move_count = 1 - white_move_count
        king_moves = 1 - king_moves
        king_escapes = 1 - king_escapes
    else:
        black_piece_count = 1 - black_piece_count
        black_move_count = 1 - black_move_count
        king_surr = 1 - king_surr

    heuristics = np.array([
        white_move_count,
        white_piece_count,
        king_moves,
        black_move_count,
        black_piece_count,
        king_escapes,
        king_surr
    ])

    weights = np.array([0, 0, 0, 0, 0, 5, 0])

    heuristics = np.array([rescale((0, 1), (-1, 1), h) for h in heuristics])

    return float(np.average(heuristics, weights=weights))


def _network_value_heuristic(model: TablutNet):
    def implementation(state: GameState) -> float:
        with torch.no_grad():
            value = model(state)  # shape: (N,2)
            #print(f"computed value is: {value}")
            return value

    return implementation


################### REUSABLE HEURISTICS


################### REUSABLE POLICIES
def _random_fixed_actions(num: int):
    def implementation(state: GameState):
        moves = state.next_moves()
        #print(f"generated {len(moves)} moves")
        random.shuffle(moves)
        # moves.sort(key=lambda move: heuristic(move), reverse=True)
        return moves[:num]

    return implementation


def _network_top_p_policy(model: TablutNet, top_p: float):
    def implementation(state: GameState):
        moves: list[GameState] = state.next_moves()
        with torch.no_grad():
            _, probs = model(moves)

        # sort by prob (descending)
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_probs = probs[sorted_indices]
        sorted_moves = [moves[i] for i in sorted_indices.tolist()]

        # cumulative probability filtering
        cumulative = torch.cumsum(sorted_probs, dim=0)
        cutoff_mask = cumulative <= top_p
        if not torch.any(cutoff_mask):
            # fallback to at least one move
            cutoff_mask[0] = True

        filtered_moves = [
            m for m, keep in zip(sorted_moves, cutoff_mask) if keep.item()
        ]
        return filtered_moves

    return implementation


################### REUSABLE POLICIES


################### REUSABLE STOPPING CRITERIONS
def _max_depth_criterion(max_depth: int):
    def implementation(state: GameState, depth: int) -> bool:
        return depth > max_depth or state.is_end_state()

    return implementation


################### REUSABLE STOPPING CRITERIONS
