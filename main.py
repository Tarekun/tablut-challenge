import sys
from profiles import alpha_beta_basic
from client import play_game
from tablut import Player


if len(sys.argv) != 3:
    print("Usage: python main.py <player> <server ip>")
    sys.exit(1)

player_input = sys.argv[1].lower()
server_ip = sys.argv[2]

if player_input == "white":
    player = Player.WHITE
elif player_input == "black":
    player = Player.BLACK
else:
    print("Invalid player. Must be either 'white' or 'black'.")
    sys.exit(1)

search = alpha_beta_basic(3, 10)
play_game(player, "MyPythonBot", server_ip, search)
