import os
from client import parse_state
from tablut import GameState, Player
from network.training import run_self_play_games, train
from network.model import TablutNet
import torch.utils.benchmark as benchmark
import torch
from torch.optim import Adam
from torch.nn import MSELoss
import time


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = TablutNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    path = "checkpoints/tablut_model_checkpoint_iter.pth"
    if os.path.isfile(path):
        print(f"Loading Checkpoint")
        checkpoint = torch.load("path")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    loss_fn = MSELoss()

    train(
        model,
        optimizer,
        loss_fn,
        iterations=2,
        games=2,
        train_steps=10,
        batch_size=50
    )
    print("Model and trainer initialized.")
    print(
        "You need to implement the 'run_self_play_games' method with your game logic."
    )

    # net = TablutNet()
    # run_self_play_games(net)
