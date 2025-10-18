import socket
import json
import struct
from typing import Callable
from tablut import Board, GameState, Player, Turn


WHITE_PORT = 5800
BLACK_PORT = 5801


def play_turn(
    client_socket: socket.socket,
    playing_as: Player,
    search_algorithm: Callable[[GameState], GameState],
) -> tuple[int | None, GameState | None]:
    """Plays one turn of the game. Returns a tuple containig a int | None in the first element
    valued with the game's outcome iff this was the last turn, the GameState analized if it was
    our turn"""
    state_json = _read_string_from_stream(client_socket)
    game_state, turn = parse_state(state_json, playing_as)
    print(f"current game state:\n{game_state}")

    if turn.plays(playing_as):
        print(f"It's our turn ({playing_as}). Calculating move...")
        move = search_algorithm(game_state)
        action = game_state.board.action_to(move.board)
        print(f"New State:\n{move}")
        _write_string_to_stream(client_socket, json.dumps(action))
        return (None, game_state)

    elif turn.wins(playing_as):
        print(f"Endgame, {playing_as} won! Yippie")
        return (1, game_state)
    elif turn.wins(playing_as.complement()):
        print(f"Endgame, {playing_as} lost! Damn...")
        return (-1, game_state)
    elif turn == Turn.DRAW:
        print(f"Endgame, it's a draw")
        return (0, game_state)

    else:
        print(f"It's opponent's turn. Waiting for next state...")
        return (None, None)


def play_game(
    player: Player,
    name: str,
    ip: str,
    search_algorithm: Callable[[GameState], GameState],
    track: bool = False,
) -> tuple[int, list[GameState]]:
    """Implements the client's connection and gameplay loop."""

    port = WHITE_PORT if player.is_white() else BLACK_PORT
    tracked_states = []
    client_socket = None

    try:
        client_socket = initialize_connection(name, ip, port)
        while True:
            (outcome, analyzed_state) = play_turn(
                client_socket, player, search_algorithm
            )
            if track and analyzed_state is not None:
                tracked_states.append(analyzed_state)
            if outcome is not None:
                break

    finally:
        if client_socket is not None:
            client_socket.close()

    return (outcome, tracked_states)


########## HELPER FUNCTIONS
def parse_state(
    json_string: str, playing_as: Player | None = None
) -> tuple[GameState, Turn]:
    """Parses the JSON string of the game state provided by the server. playing_as is optional
    and if left unspecified will be set to the turn player"""

    state = json.loads(json_string)
    turn = None
    for member in Turn:
        if member.value == state["turn"]:
            turn = member
    if turn is None:
        raise ValueError(
            f"Received game state returned a `turn` value which couldn't be matched: {state['turn']}"
        )

    # TODO: review this
    turn_player = Player.WHITE if turn.plays(Player.WHITE) else Player.BLACK
    playing_as = playing_as if playing_as is not None else turn_player
    return (GameState(Board(state["board"]), playing_as, turn), turn)


def initialize_connection(player_name: str, ip: str, port: int):
    print(f"Connecting to {ip}:{port}")
    # 1. Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.settimeout(TIMEOUT_SECONDS)
    client_socket.connect((ip, port))

    # 2. Send Player Name (Handshake)
    name_json = json.dumps(player_name)
    _write_string_to_stream(client_socket, name_json)

    return client_socket


STRING_FORMAT = "utf-8"


def _read_n_bytes(sock: socket.socket, n: int) -> bytes:
    """Reads exactly `n` bytes from the socket"""

    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            # connection closed by peer
            raise ConnectionResetError("Connection closed by peer while reading")
        data += chunk
    return data


def _read_string_from_stream(sock: socket.socket) -> str:
    # read 4 bytes to get the length of the message
    raw_len = _read_n_bytes(sock, 4)
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
