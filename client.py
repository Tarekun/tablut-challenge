import socket
import json
import struct
import time
from typing import Dict
from tablut import Board, Player, Turn
from agent import alpha_beta, heuristic, move_sequence, max_depth_criterion


WHITE_PORT = 5800
BLACK_PORT = 5801


def play_turn(turn: Turn, board: Board, player: Player, search_algorithm) -> bool:
    """Plays one turn of the game. Returns True if the game is over, False otherwise"""

    if turn.plays(player):
        print(f"It's our turn ({player}). Calculating move...")
        move = search_algorithm(board, player)
        # TODO: encode the move and send it to the server
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
            print("Waiting for game state from server...")
            try:
                state_json = _read_string_from_stream(client_socket)
                board, server_turn = parse_state(state_json)
                is_over = play_turn(server_turn, board, player, search)
                if is_over:
                    break

                # time.sleep(0.1)  # Small pause to prevent rapid looping/spam
            except socket.timeout:
                print(
                    f"Timeout Error: Connection or read operation took longer than 60 seconds."
                )
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    finally:
        if "client_socket" in locals():
            client_socket.close()
        print("Client script terminated.")


########## HELPER FUNCTIONS
def parse_state(json_string: str) -> tuple[Board, Turn]:
    """Parses the JSON string of the game state provided by the server"""

    state = json.loads(json_string)
    print(f"parsed state is {state}")
    turn = None
    for member in Turn:
        if member.value == state["turn"]:
            turn = member
    if turn is None:
        raise ValueError(
            f"Received game state returned a `turn` value which couldn't be matched: {state}"
        )

    return Board(state["board"]), turn


def encode_action(move_dict: Dict[str, str]) -> str:
    """
    ASSUMED FUNCTION: Maps an internal action representation back to the JSON string.
    """
    return json.dumps(move_dict)


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


def read_n_bytes(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            # connection closed by peer
            raise ConnectionResetError("Connection closed by peer while reading")
        data += chunk
    return data


def _read_string_from_stream(sock: socket.socket) -> str:
    # Read 4-byte big-endian length prefix (same as Java DataOutputStream.writeInt)
    raw_len = read_n_bytes(sock, 4)
    length = struct.unpack(">I", raw_len)[0]  # big-endian unsigned int
    if length == 0:
        return ""
    payload = read_n_bytes(sock, length)
    return payload.decode("utf-8")
    # raw_length = sock.recv(4)
    # if not raw_length:
    #     raise ConnectionResetError("Connection closed by server.")
    # if len(raw_length) < 4:
    #     raise EOFError("Incomplete length prefix received.")

    # # Convert 4 bytes (big-endian) to an integer
    # length = struct.unpack(">I", raw_length)[0]

    # # 2. Read the full payload
    # data = b""
    # while len(data) < length:
    #     chunk = sock.recv(length - len(data))
    #     if not chunk:
    #         raise ConnectionResetError("Connection closed while reading payload.")
    #     data += chunk

    # return data.decode("utf-8")


def _write_string_to_stream(sock: socket.socket, data: str):
    """Writes a length-prefixed string to a socket."""
    payload = data.encode("utf-8")
    length = len(payload)

    # 1. Create the 4-byte length prefix (big-endian)
    raw_length = struct.pack(">I", length)

    # 2. Send the prefix and the payload
    sock.sendall(raw_length + payload)
