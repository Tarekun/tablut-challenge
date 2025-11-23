import random
import statistics
import torch
import numpy as np
from typing import Callable
from search import alpha_beta, monte_carlo_tree_search
from network.model import TablutNet
from tablut import GameState, Player, BLACK_PIECES, WHITE_PIECES, MAX_PAWN_MOVES
from utils import rescale
from typing import Union


# This file contains various functions that return a search strategy (ie a function with type
# GameState -> GameState) that can be used as an argument to `client.play_game` to play one
# game with that playstyle. Useful to sample different algorithms when selfplaying

# Most of the components of search algorithms (stopping criterions, action policies and state
# heuristics) are defined to configurable and reusable so one can plug n play different combinations
# of them with the same base search algorithm


################### SEARCH STRATEGIES
def mcts_shallow_model(
    model: TablutNet,
    time_limit_s: float = 55,
) -> Callable[[GameState], GameState]:
    return monte_carlo_tree_search(
        model.policy, _network_value_heuristic(model), time_limit_s
    )


def mcts_fixed_model(
    model: TablutNet,
    max_depth: int,
    time_limit_s: float = 55,
) -> Callable[[GameState], GameState]:
    return monte_carlo_tree_search(
        model.policy, _model_rollout(model, max_depth), time_limit_s
    )


def mcts_deep_model(
    model: TablutNet,
    time_limit_s: float = 55,
) -> Callable[[GameState], GameState]:
    return monte_carlo_tree_search(
        model.policy, _model_rollout(model, float("inf")), time_limit_s
    )


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
    model: TablutNet, max_depth: int, branching: int
) -> Callable[[GameState], GameState]:
    """Minimax algorithm with alpha/beta cuts that uses `model` as a state estimator
    when reaching a node at `max_depth`"""

    return alpha_beta(
        _network_value_heuristic(model),
        _max_depth_criterion(max_depth),
        _random_fixed_actions(branching),
    )


def alpha_beta_policy_model(
    model: TablutNet, max_depth: int, branching: int
) -> Callable[[GameState], GameState]:
    """Minimax algorithm with alpha/beta cuts that uses hand picked averaged heuristics,
    a `max_depth` stopping cretirion, and a top P policy algorithm with the probability
    distributions provided by `model`.
    `model` takes all available actions and returns a probability distribution over them,
    we pick the `branching` most probable actions
    """

    return alpha_beta(
        _handpicked_heuristics,
        _max_depth_criterion(max_depth),
        _network_top_n_policy(model, branching),
    )


def alpha_beta_full_model(model: TablutNet, max_depth: int, branching: int):
    """Minimax alpha/beta search fully guided by the network `model`.
    It selects actions with cumulative probability of `top_p` according to `model`
    and evaluates states with `model`, up to `max_depth` in the search tree"""
    return alpha_beta(
        _network_value_heuristic(model),
        _max_depth_criterion(max_depth),
        _network_top_n_policy(model, branching),
    )


def model_value_maximization(
    model: TablutNet,
) -> Callable[[GameState], GameState]:
    """Model heuristic maximization search that given a state returns the action
    that maximizes the heuristic value"""

    def maximize_heuristic(state: GameState) -> GameState:
        moves = state.next_moves()
        with torch.no_grad():
            values = model.value(moves)  # shape: (N,2)

        # find best move index
        best_idx = int(torch.argmax(values).item())

        return moves[best_idx]

    return maximize_heuristic


def model_greedy_sampling(model: TablutNet):
    """Greedy sampling model search that given a state returns the action with the
    highest probability according to the policy head of the network"""

    def implementation(state: GameState):
        moves = state.next_moves()
        with torch.no_grad():
            probs = model.policy(moves)  # shape: (N,2)

        # find best move index
        best_idx = int(torch.argmax(probs).item())

        return moves[best_idx]

    return implementation


################### SEARCH STRATEGIES


################### REUSABLE HEURISTICS
def _handpicked_heuristics(state: GameState) -> float:  # [-1, 1]
    board = state.board
    king_escapes = board.king_escapes()
    king_surr = board.king_surr()

    # win/lose state available => return the outcome
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

    heuristics = np.array(
        [
            white_move_count,
            white_piece_count,
            king_moves,
            black_move_count,
            black_piece_count,
            king_escapes,
            king_surr,
        ]
    )
    # king metrics 3 times important as other metrics
    weights = np.array([1, 1, 1, 1, 1, 3, 3])
    heuristics = np.array([rescale((0, 1), (-1, 1), h) for h in heuristics])

    return float(np.average(heuristics, weights=weights))


def _network_value_heuristic(model: TablutNet):
    def implementation(state: GameState) -> float:
        with torch.no_grad():
            value = model.value(state)
            return value

    return implementation


################### REUSABLE HEURISTICS


################### REUSABLE POLICIES
def _random_fixed_actions(num: int):
    def implementation(state: GameState):
        moves = state.next_moves()
        random.shuffle(moves)
        # moves.sort(key=lambda move: heuristic(move), reverse=True)
        return moves[:num]

    return implementation


def _heuristic_fixed_actions(num: int, heuristic):
    def implementation(state: GameState):
        moves = state.next_moves()
        moves.sort(key=lambda move: heuristic(move), reverse=True)
        return moves[:num]

    return implementation


def _network_top_p_policy(model: TablutNet, top_p: float):
    def implementation(state: GameState):
        moves: list[GameState] = state.next_moves()
        with torch.no_grad():
            probs = model.policy(moves)

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


def _network_top_n_policy(model: TablutNet, top_n: int):
    def implementation(state: GameState):
        moves: list[GameState] = state.next_moves()
        if len(moves) <= top_n:
            return moves

        with torch.no_grad():
            probs = model.policy(moves)

        top_indices = torch.topk(probs, top_n, sorted=True).indices
        top_moves = [moves[i] for i in top_indices.tolist()]
        return top_moves

    return implementation


################### REUSABLE POLICIES


################### REUSABLE STOPPING CRITERIONS
def _max_depth_criterion(max_depth: int):
    def implementation(state: GameState, depth: int) -> bool:
        return depth > max_depth or state.is_end_state()

    return implementation


################### REUSABLE STOPPING CRITERIONS


################### REUSABLE ROLLOUT FUNCTIONS
def _model_rollout(
    model: TablutNet, max_depth: Union[int, float]
) -> Callable[[GameState], float]:
    def implementation(root_state: GameState) -> float:
        state = root_state
        depth: int = 0

        with torch.no_grad():
            while depth < max_depth and not state.is_end_state():
                search_space = state.next_moves()
                probs = model.policy(search_space)
                move_idx = torch.multinomial(probs, num_samples=1).item()

                state = search_space[int(move_idx)]
                depth += 1

            if state.is_end_state():
                if state.winner() == root_state.turn_player:
                    return 1
                elif state.winner() == "DRAW":
                    return 0
                else:
                    return -1
            else:
                return model.value(state)

    return implementation


def _random_fixed_depth_rollout(heuristic, max_depth: Union[int, float]):
    def implementation(root_state: GameState) -> float:
        state = root_state
        depth: int = 0

        while depth < max_depth and not state.is_end_state():
            search_space = state.next_moves()
            move_idx = random.choice(len(search_space))

            state = search_space[move_idx]
            depth += 1

        if state.is_end_state():
            if state.winner() == root_state.turn_player:
                return 1
            elif state.winner() == "DRAW":
                return 0
            else:
                return -1
        else:
            return heuristic(state)

    return implementation


################### REUSABLE ROLLOUT FUNCTIONS
