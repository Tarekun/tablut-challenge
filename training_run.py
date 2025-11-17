from utils import flag_manager, get_latest_checkpoint
from network.training import train
from network.model import TablutNet
import torch.utils.benchmark as benchmark
import torch
from torch.optim import Adam, AdamW
from torch.nn import MSELoss


if __name__ == "__main__":
    res = flag_manager().res
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"running on {device}")
    model = TablutNet(res=res).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    path = get_latest_checkpoint(
        "checkpoints", prefix="tablut_model_checkpoint_iter_", ext=".pth"
    )
    if path:
        print(f"Loading Checkpoint {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["state_dict"])
    loss_fn = MSELoss()

    train(
        model,
        optimizer,
        loss_fn,
        iterations=20,
        games=2,
        train_steps=200,
        batch_size=50,
    )
    print("Model and trainer initialized.")
    print(
        "You need to implement the 'run_self_play_games' method with your game logic."
    )

    # net = TablutNet()
    # run_self_play_games(net)
