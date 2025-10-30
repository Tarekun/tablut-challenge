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
    loss_fn = MSELoss()

    train(
        model,
        optimizer,
        loss_fn,
        iterations=2,
        games=3,
        train_steps=5,
        batch_size=32,
    )
    print("Model and trainer initialized.")
    print(
        "You need to implement the 'run_self_play_games' method with your game logic."
    )

    # net = TablutNet()
    # run_self_play_games(net)
