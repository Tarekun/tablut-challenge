from typing import Dict, Any, Callable
from tablut import Turn, Player
from client import encode_action


def pick_move(internal_state: Dict[str, Any]):
    pass


def play_turn(turn: Turn, player: Player):
    print("play turn being called")
    # move_dict = pick_move(current_state)
    # move_json = encode_action(move_dict)

    # print(f"Sending move: {move_json}")
    # _write_string_to_stream(client_socket, move_json)


def alpha_beta(heuristic: Callable, stop_criterion: Callable, move_sequence: Callable):
    def max_value(
        board, player: Player, alpha, beta, depth
    ) -> tuple[list | None, float]:  # TODO fix this list | None
        if stop_criterion(board, depth):
            # probabile va messa una vera valutazione qua, specie se finise il gioco
            return (board, heuristic(board))

        local_best = float("-inf")
        best_move = None
        for move in move_sequence(board, player):
            (_, value) = min_value(move, player.complement(), alpha, beta, depth + 1)
            if value > local_best:
                local_best = value
                best_move = move
            if local_best >= beta:
                return (best_move, local_best)
            alpha = max(alpha, local_best)

        return (best_move, local_best)

    def min_value(
        board, player: Player, alpha, beta, depth
    ) -> tuple[list | None, float]:
        if stop_criterion(board, depth):
            # probabile va messa una vera valutazione qua, specie se finise il gioco
            return (board, heuristic(board))

        local_best = float("inf")
        best_move = None
        for move in move_sequence(board, player):
            (_, value) = max_value(move, player.complement(), alpha, beta, depth + 1)
            if value < local_best:
                local_best = value
                best_move = move
            if local_best <= alpha:
                return (best_move, local_best)
            beta = min(beta, local_best)

        return (best_move, local_best)

    def search_algorithm(board, player: Player):
        (best_move, _) = max_value(board, player, float("-inf"), float("inf"), 0)
        return best_move

    return search_algorithm
