import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import subprocess
import time
from client import play_game
from tablut import Player
from tablut import BOARD_LENGTH, GameState
from profiles import model_value_maximization_search
from network.model import TablutNet


# TODO: also collect the picked action
def run_self_play_game(model, num_games=1) -> list:
    """Runs `num_games` number of games in a sequence collecting the states being played
    and the picked action at each state, together with the game result (±1) from the
    perspective of the turn player"""

    game_history = []
    search = model_value_maximization_search(model)

    for _ in range(num_games):
        player = Player.WHITE

        print("Starting server...", end=" ")
        server = subprocess.Popen(
            ["ant", "server", "WHITE", "localhost"],
            cwd="C:\\Users\\danie\\codice\\uni\\TablutCompetition\\Tablut",
            stdout=open("server.log", "w"),
            start_new_session=True,  # detach completely
            shell=True,
        )
        time.sleep(1)
        print("Done")

        print("Starting opponent...", end=" ")
        opponent = subprocess.Popen(
            ["python", "main.py", player.complement().value, "localhost"],
            cwd="C:\\Users\\danie\\codice\\uni\\tablut-challenge",
            stdout=open("opponent.log", "w"),
            start_new_session=True,  # detach completely
            shell=True,
        )
        time.sleep(1)
        print("Done")

        outcome, game_states = play_game(
            player, "Trainee", "localhost", search, track=True
        )
        for state in game_states:
            print(f"processing:\n{state}\n")
            outcome = outcome if state.turn_player == player else -1 * outcome
            game_history.append((state, outcome))

        for state, outcome in game_history:
            print(
                f"THIS FOLLOWING STATE PLAYED BY WHITE GOT OUTCOME {outcome}\n{state}",
                end="\n\n\n",
            )

    return game_history


def train_step(
    model: TablutNet, experience_buffer: list, optimizer, loss_fn, batch_size=32
):
    """
    Performs one training step using a batch of experiences sampled randomly from the buffer.
    """

    if len(experience_buffer) < batch_size:
        raise ValueError(
            f"Eperience buffer contains only {len(experience_buffer)} samples which is less than the required {batch_size} batch size."
        )

    # Sample a batch of experiences randomly
    batch = random.sample(experience_buffer, batch_size)
    states, target_values = zip(*batch)

    # Convert to tensors
    # Assume states are tuples (board, turn_ind, player_ind)
    board_batch = torch.stack([s[0] for s in states])
    turn_batch = torch.stack([s[1] for s in states])
    player_batch = torch.stack([s[2] for s in states])
    target_batch = torch.tensor(target_values, dtype=torch.float32)

    # Forward pass
    # TODO: da rivedere
    predicted_values = model(board_batch, turn_batch, player_batch)
    loss = loss_fn(predicted_values, target_batch)
    # Backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    # Optional: Gradient clipping to prevent exploding gradients
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def train(
    model: TablutNet,
    optimizer,
    loss_fn,
    iterations=10000,
    games: int = 10,
    train_steps=100,
    batch_size: int = 32,
):
    """
    Main training loop. This runs `games` number of games each iteration per `iterations` times, collecting
    moves in a experience buffer. Each iteration it run `train_steps` training steps where it samples randomly
    `batch_size` actions from the experience buffer and performs gradient optimization on those samples
    """
    for iteration in range(iterations):
        # print(f"Starting Iteration {iteration + 1}/{iterations}")
        # print(f"\tRunning {games} self-play games...")
        experience_buffer = run_self_play_game(model, num_games=games)

        print(
            f"\tOptimizing model for {train_steps} steps with batch size of {batch_size}..."
        )
        total_loss = 0
        for step in range(train_steps):
            loss = train_step(model, experience_buffer, optimizer, loss_fn, batch_size)
            if loss is not None:
                total_loss += loss
        avg_loss = total_loss / train_steps if train_steps > 0 else 0
        print(f"\tIteration {iteration + 1} completed. Average Loss: {avg_loss:.6f}")

        # Save model checkpoint periodically
        if (iteration + 1) % 100 == 0:
            torch.save(
                model.state_dict(),
                f"tablut_model_checkpoint_iter_{iteration + 1}.pth",
            )
            print(f"  Model checkpoint saved at iteration {iteration + 1}")


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TablutNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    train(
        model,
        optimizer,
        loss_fn,
        iterations=1000,
        games=10,
        train_steps=100,
        batch_size=32,
    )
    print("Model and trainer initialized.")
    print("You need to implement the 'run_self_play_game' method with your game logic.")
