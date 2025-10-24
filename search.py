import math
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
    heuristic: Callable[[GameState], float],  # heuristic
    probability: Callable[[list[GameState]], list[float]],  # policy
):
    """
    Monte Carlo Tree Search (MCTS) algorithm.

    Parameters
    ----------
    """

    class MCTSNode:
        def __init__(self, state: GameState, parent=None):
            self.state: GameState = state
            self.parent: MCTSNode | None = parent
            self.children: list[MCTSNode] = []
            self.visits = 0
            self.wins = 0
            self.untried_states = self.state.next_moves()

        def is_terminal(self):
            """Restituisce True se la partita è finita."""
            return self.state.is_end_state()

        def is_fully_expanded(self):
            return len(self.untried_states) == 0 and len(self.children) > 0

        def expand(self):
            """Crea un figlio prendendo uno stato non ancora esplorato."""
            new_state = self.untried_states.pop()
            child = MCTSNode(new_state, parent=self)
            self.children.append(child)
            return child

        def best_child(self, c=1.4):
            """Scelta del figlio migliore secondo UCT."""
            # Prendiamo gli stati dei figli
            child_states = [child.state for child in self.children]

            # Otteniamo le probabilità corrispondenti
            probs = probability(child_states)

            # Ora combiniamo le probabilità nella formula UCT modificata
            best = max(
                zip(self.children, probs),
                key=lambda pair: (pair[0].wins / (pair[0].visits + 1e-6))
                + c
                * pair[1]
                * math.sqrt(math.log(self.visits + 1) / (pair[0].visits + 1e-6)),
            )
            return best[0]

        def rollout(self, max_depth=None):
            """Simula il valore dello stato tramite euristica"""
            return heuristic(self.state)

            # while not sim_state.is_end_state():
            #     moves = sim_state.next_moves()
            #     if not moves:
            #         print(sim_state.board)
            #         print("not moves")
            #         break
            #     sim_state = random.choice(moves)

            # if sim_state.is_end_state():
            #     # usa 1/0/0.5 basato sul risultato reale
            #     return sim_state.get_result(self.state.playing_as)

        def backpropagate(self, result):
            """Aggiorna le statistiche lungo la catena dei padri."""
            self.visits += 1
            self.wins += result
            if self.parent:
                self.parent.backpropagate(result)

    def search_algoritm(root_state, iterations=100):
        root = MCTSNode(root_state)

        for _ in range(iterations):
            node = root

            # Selection: scendi lungo l’albero
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion: aggiungi un figlio
            if not node.is_terminal():
                node = node.expand()

            # Simulation: gioca random fino alla fine
            result = node.rollout()
            # Backpropagation: aggiorna i punteggi
            node.backpropagate(result)

        # Alla fine scegli il figlio con più visite (non più alto UCB)
        return max(root.children, key=lambda c: c.visits).state

    return search_algoritm
    """
    Returns
    -------
    GameState -> GameState
        Function that takes current GameState as input and returns the
        optimal next GameState according to the minmax search with alpha-beta pruning.
    """
