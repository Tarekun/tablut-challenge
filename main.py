import sys
import torch
from profiles import mcts_fixed_model
from client import play_game
from network.model import TablutNet
from tablut import Player


if len(sys.argv) != 4:
    print("Usage: python main.py <player> <timeout> <server ip>")
    sys.exit(1)

player_input = sys.argv[1].lower()
timeout = int(sys.argv[2])
server_ip = sys.argv[3]

if player_input == "white":
    player = Player.WHITE
elif player_input == "black":
    player = Player.BLACK
else:
    raise ValueError(
        f"Invalid player. Must be either 'white' or 'black', not {player_input}."
    )

# search = alpha_beta_basic(4, 30)
device = torch.device("cpu")
model = TablutNet().to(device)
checkpoint_path = "checkpoints/final_checkpoint.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
search = mcts_fixed_model(model, 20, timeout - 5)
play_game(player, "Tulbat", server_ip, search)
