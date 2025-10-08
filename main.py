from client import play_game
from tablut import Player


SERVER_IP = "127.0.0.1"
PLAYER_NAME = "MyPythonBot"
TIMEOUT_SECONDS = 60

play_game(Player.WHITE, PLAYER_NAME, SERVER_IP)
