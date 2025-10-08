from typing import Dict, Any
from tablut import Turn, Player
from client import encode_action


def pick_move(internal_state: Dict[str, Any]) -> Dict[str, str]:
    pass


def play_turn(turn: Turn, player: Player):
    print("play turn being called")
    # move_dict = pick_move(current_state)
    # move_json = encode_action(move_dict)

    # print(f"Sending move: {move_json}")
    # _write_string_to_stream(client_socket, move_json)
