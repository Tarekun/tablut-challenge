import socket
import json
import struct
import time
from typing import Callable, Dict
from tablut import Board, GameState, Player, Turn
from agent import alpha_beta, heuristic, move_sequence, max_depth_criterion


WHITE_PORT = 5800
BLACK_PORT = 5801


def play_turn(
    client_socket: socket.socket,
    playing_as: Player,
    search_algorithm: Callable[[GameState], GameState],
) -> bool:
    """Plays one turn of the game. Returns True if the game is over, False otherwise"""
    state_json = _read_string_from_stream(client_socket)
    board, turn = parse_state(state_json)

    if turn.plays(playing_as):
        print(f"It's our turn ({playing_as}). Calculating move...")
        game_state = GameState(board, playing_as, playing_as)
        print(f"current game state:\n{game_state}")
        print("RUNNINGG SEARCH")
        move = search_algorithm(game_state)
        action = board.action_to(move.board)
        print(f"sending action {action}")
        _write_string_to_stream(client_socket, json.dumps(action))
        return False

    elif turn.game_finished():
        print(f"Game End Detected. Result: {turn}")
        return True
    else:
        print(f"It's opponent's turn. Waiting for next state...")
        return False


def play_game(player: Player, name: str, ip: str):
    """Implements the client's connection and turn loop."""

    print(f"Connecting as player {player}...")
    port = WHITE_PORT if player.is_white() else BLACK_PORT
    search = alpha_beta(heuristic, max_depth_criterion, move_sequence)

    try:
        client_socket = initialize_connection(name, ip, port)
        while True:
            # try:
            is_over = play_turn(client_socket, player, search)
            if is_over:
                break

            # time.sleep(0.1)  # Small pause to prevent rapid looping/spam
        # except socket.timeout:
        #     print(
        #         f"Timeout Error: Connection or read operation took longer than 60 seconds."
        #     )
        #     break
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")
        #     break

    finally:
        if "client_socket" in locals():
            client_socket.close()
        print("Client script terminated.")


########## HELPER FUNCTIONS
def parse_state(json_string: str) -> tuple[Board, Turn]:
    """Parses the JSON string of the game state provided by the server"""

    state = json.loads(json_string)
    # print(f"parsed state is {state}")
    turn = None
    for member in Turn:
        if member.value == state["turn"]:
            turn = member
    if turn is None:
        raise ValueError(
            f"Received game state returned a `turn` value which couldn't be matched: {state}"
        )

    return Board(state["board"]), turn


def initialize_connection(player_name: str, ip: str, port: int):
    # 1. Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.settimeout(TIMEOUT_SECONDS)
    client_socket.connect((ip, port))
    print(f"Connected to {ip}:{port}")

    # 2. Send Player Name (Handshake)
    print(f"Sending player name: '{player_name}'")
    name_json = json.dumps(player_name)
    _write_string_to_stream(client_socket, name_json)

    return client_socket


# --- COMMUNICATION UTILITIES (Assuming 4-byte length prefix) ---
# TODO: review these functions

STRING_FORMAT = "utf-8"


def _read_n_bytes(sock: socket.socket, n: int) -> bytes:
    """Reads exactly `n` bytes from the socket"""

    data = b""
    print(f"read n bytes with {sock} and {n}")
    while len(data) < n:
        print("ciclo")
        chunk = sock.recv(n - len(data))
        if not chunk:
            # connection closed by peer
            raise ConnectionResetError("Connection closed by peer while reading")
        data += chunk
    return data


def _read_string_from_stream(sock: socket.socket) -> str:
    # read 4 bytes to get the length of the message
    raw_len = _read_n_bytes(sock, 4)
    print(struct.unpack(">I", raw_len))
    length = struct.unpack(">I", raw_len)[0]
    #   try this in case of problems
    #   raw_len = _read_n_bytes(sock, 2)  # java writeUTF with 2-byte length prefix
    #   length = struct.unpack(">H", raw_len)[0]
    if length == 0:
        return ""
    # read rest of the message
    payload = _read_n_bytes(sock, length)
    return payload.decode(STRING_FORMAT)


def _write_string_to_stream(sock: socket.socket, data: str):
    """Writes a length-prefixed string to a socket."""
    payload = data.encode(STRING_FORMAT)
    length = len(payload)

    # 1. Create the 4-byte length prefix (big-endian)
    raw_length = struct.pack(">I", length)
    # 2. Send the prefix and the payload
    sock.sendall(raw_length + payload)
