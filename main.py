import sys

import torch
from network.model import TablutNet
from profiles import mcts_fixed_model
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
    raise ValueError(
        f"Invalid player. Must be either 'white' or 'black', not {player_input}."
    )

device = torch.device("cpu")
print(f"running on {device}")
model = TablutNet().to(device)
path = "checkpoints/final_checkpoint.pth"
if path:
    print(f"Loading Checkpoint {path}")
    checkpoint = torch.load(path)
    print(f"Loaded checkpoint from {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

search = mcts_fixed_model(model)
play_game(player, "MyPythonBot", server_ip, search)