from functools import reduce
import math
import random
import time
from typing import Callable
from tablut import GameState


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
        start_time = time.perf_counter()
        (best_move, _) = max_value(state, float("-inf"), float("inf"), 0)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"MIN/MAX search took {duration:.3f} seconds")
        return best_move

    return search_algorithm


def monte_carlo_tree_search(
    probability: Callable[[list[GameState]], list[float]],
    rollout_to_value: Callable[[GameState], float],
    time_limit_s: float = 55,
):
    """
    Monte Carlo Tree Search (MCTS) algorithm.

    Parameters
    ----------
    probability : list[GameState] -> list[float]
        Function that given a list of GameStates to explore returns a probability
        distribution over those states
    rollout_to_value : GameState -> float
        Rollout strategy function for leaf nodes in the search process
    time_limit_s : float
        Time limit in seconds to iterate the search

    Returns
    -------
    GameState -> GameState
        Function that takes current GameState as input and returns the
        optimal next GameState according to MCTS
    """

    class MCTSNode:
        def __init__(self, state: GameState, parent=None):
            self.state: GameState = state
            self.parent: MCTSNode | None = parent
            self._children: list[MCTSNode] | None = None
            self.visits = 0
            self.total_score = 0

        def is_terminal(self):
            """Restituisce True se la partita è finita."""
            return self.state.is_end_state()

        def is_fully_expanded(self):
            return self.visits == 0

        @property
        def children(self):
            if self._children is None:
                self._children = [
                    MCTSNode(child, self) for child in self.state.next_moves()
                ]

            return self._children

        def best_child(self, exp_const: float):
            child_states = [
                child.state
                for child in self.children
                if self.is_fully_expanded() or child.visits == 0
            ]

            probs = probability(child_states)

            def puct_score(node: "MCTSNode", prior: float):
                total_visits = sum(child.visits for child in self.children)

                avg_score = node.total_score / node.visits if node.visits != 0 else 0
                u_score = (
                    exp_const * prior * math.sqrt(total_visits) / (node.visits + 1)
                )
                return avg_score + u_score

            best = max(
                zip(self.children, probs),
                key=lambda pair: (
                    puct_score(pair[0], pair[1])
                    if self.is_fully_expanded()
                    else pair[1]
                ),
            )
            return best[0]

        def backpropagate(self, result: float):
            """Backpropagates the visit and the rollout result"""
            self.visits += 1
            self.total_score += result
            if self.parent:
                self.parent.backpropagate(result)

    def search_algoritm(root_state: GameState) -> GameState:
        print("MCTS called")
        root = MCTSNode(root_state)
        player = root_state.turn_player
        exp_const = 2.5

        start_time = time.time()
        end_time = start_time + time_limit_s
        print(f"start time {start_time} end time {end_time}")
        iterations = 0

        while time.time() < end_time:
            node = root
            # descend down the tree
            while not node.state.is_end_state() and not node.visits == 0:
                node = node.best_child(exp_const)

            if node.state.is_end_state():
                value = 1 if node.state.winner() == player else -1
            else:
                value = rollout_to_value(node.state)

            node.backpropagate(value)
            # exponential annealing of the exploration constant
            exp_const = 0.95 * exp_const
            iterations += 1

        print(f"Completed {iterations} iterations in {time_limit_s} seconds")
        # pick the most visited child
        return max(root.children, key=lambda c: c.visits).state

    return search_algoritm
