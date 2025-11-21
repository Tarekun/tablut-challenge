import sys
import torch
from network.model import TablutNet
from profiles import alpha_beta_basic, mcts_deep_model
from client import play_game
from tablut import Player
from torch.optim import Adam
from utils import get_latest_checkpoint


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
path, last_iter = get_latest_checkpoint("checkpoints", prefix = "tablut_model_checkpoint_iter_", ext = ".pth")  
if path:
    print(f"Loading Checkpoint {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

search = mcts_deep_model(model, 55)
play_game(player, "MyPythonBot", server_ip, search)
