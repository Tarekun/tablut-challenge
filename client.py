import socket
import json
import struct
import time
from typing import Dict, Any
from tablut import Player, Turn


WHITE_PORT = 5800
BLACK_PORT = 5801


def parse_state(json_string: str) -> Dict[str, Any]:
    """
    ASSUMED FUNCTION: Maps the received JSON string to an internal state representation.
    For this example, we return the parsed dictionary directly.
    """
    return json.loads(json_string)


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


def play_game(player: Player, name: str, ip: str):
    """Implements the client's connection and turn loop."""

    print(f"Connecting as player {player}...")
    port = WHITE_PORT if player.is_white() else BLACK_PORT

    try:
        client_socket = initialize_connection(name, ip, port)
        while True:
            print("Waiting for game state from server...")
            try:
                state_json = _read_string_from_stream(client_socket)
                current_state = parse_state(state_json)
                print(current_state)
                board = current_state["board"]
                server_turn: Turn = current_state["turn"]

                if server_turn.plays(player):
                    print(f"It's our turn ({player}). Calculating move...")
                elif server_turn.game_finished():
                    print(f"Game End Detected. Result: {server_turn}")
                    break
                else:
                    print(f"It's opponent's turn. Waiting for next state...")

                time.sleep(0.1)  # Small pause to prevent rapid looping/spam

            except (ConnectionResetError, EOFError) as e:
                print(f"Server disconnected. Game over. Error: {e}")
                break
            except socket.timeout:
                print(
                    f"Timeout Error: Connection or read operation took longer than 60 seconds."
                )
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    finally:
        if "client_socket" in locals():
            client_socket.close()
        print("Client script terminated.")


# --- COMMUNICATION UTILITIES (Assuming 4-byte length prefix) ---


def _read_string_from_stream(sock: socket.socket) -> str:
    """Reads a length-prefixed string from a socket."""
    # 1. Read the 4-byte length prefix
    raw_length = sock.recv(4)
    if not raw_length:
        raise ConnectionResetError("Connection closed by server.")
    if len(raw_length) < 4:
        raise EOFError("Incomplete length prefix received.")

    # Convert 4 bytes (big-endian) to an integer
    length = struct.unpack(">I", raw_length)[0]

    # 2. Read the full payload
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionResetError("Connection closed while reading payload.")
        data += chunk

    return data.decode("utf-8")


def _write_string_to_stream(sock: socket.socket, data: str):
    """Writes a length-prefixed string to a socket."""
    payload = data.encode("utf-8")
    length = len(payload)

    # 1. Create the 4-byte length prefix (big-endian)
    raw_length = struct.pack(">I", length)

    # 2. Send the prefix and the payload
    sock.sendall(raw_length + payload)
