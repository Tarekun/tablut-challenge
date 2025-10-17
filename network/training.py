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
from profiles import *
from network.model import TablutNet
import threading


def train(
    model: TablutNet,
    optimizer,
    loss_fn,
    iterations: int,
    games: int = 1,
    train_steps: int = 1,
    batch_size: int = 32,
):
    """
    Main training loop. This runs `games` number of games each iteration per `iterations` times, collecting
    moves in a experience buffer. Each iteration it run `train_steps` training steps where it samples randomly
    `batch_size` actions from the experience buffer and performs gradient optimization on those samples
    """
    for iteration in range(iterations):
        print(f"Starting Iteration {iteration + 1}/{iterations}")
        print(f"\tRunning {games} self-play games...")
        experience_buffer = run_self_play_games(model, num_games=games)

        print(
            f"\tOptimizing model for {train_steps} steps with batch size of {batch_size}..."
        )
        total_loss = 0
        for _ in range(train_steps):
            loss = train_step(model, experience_buffer, optimizer, loss_fn, batch_size)
            if loss is not None:
                total_loss += loss
        avg_loss = total_loss / train_steps if train_steps > 0 else 0
        print(f"\tIteration {iteration + 1} completed. Average Loss: {avg_loss:.6f}")

        # Save model checkpoint periodically
        # if (iteration + 1) % 100 == 0:
        #     torch.save(
        #         model.state_dict(),
        #         f"tablut_model_checkpoint_iter_{iteration + 1}.pth",
        #     )
        #     print(f"  Model checkpoint saved at iteration {iteration + 1}")


def train_step(
    model: TablutNet,
    experience_buffer: list[tuple[GameState, int]],
    optimizer,
    loss_fn,
    batch_size: int,
):
    """
    Performs one training step using a batch of experiences sampled randomly from the buffer.
    """

    if len(experience_buffer) < batch_size:
        raise ValueError(
            f"Eperience buffer contains only {len(experience_buffer)} samples which is less than the required {batch_size} batch size."
        )

    # sample a random batch of experiences
    batch = random.sample(experience_buffer, batch_size)
    states = [item[0] for item in batch]
    outcomes = [item[1] for item in batch]
    tensor_outcomes = torch.tensor(outcomes, dtype=torch.float32).to(
        next(model.parameters()).device
    )

    # forward pass
    values, probs = model(states)
    loss = loss_fn(values, tensor_outcomes)
    print(outcomes)
    print(values)

    # backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    # Optional: Gradient clipping to prevent exploding gradients
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# TODO: also collect the picked action
def run_self_play_games(model, num_games=1) -> list[tuple[GameState, int]]:
    """Runs `num_games` number of games in a sequence collecting the states being played
    and the picked action at each state, together with the game result (±1) from the
    perspective of the turn player"""

    def simulate_one_game(game_history: list):
        player = random.choice([Player.WHITE, Player.BLACK])

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
        opp_results = {}
        opp_thread = threading.Thread(
            target=_opponent_gameloop_wrapper,
            args=(
                player.complement(),
                "opponent",
                "localhost",
                _random_opponent_search(model),
                opp_results,
            ),
        )
        opp_thread.start()
        time.sleep(1)
        print("Done")

        outcome, game_states = play_game(
            player, "Trainee", "localhost", search, track=True
        )
        game_history.extend(_prepare_game_state_data(outcome, game_states, player))
        opp_thread.join()
        opp_outcome = opp_results["outcome"]
        opp_game_states = opp_results["game_states"]
        game_history.extend(
            _prepare_game_state_data(opp_outcome, opp_game_states, player.complement())
        )

        return {
            "trainee_player": player.value,
            "trainee_strategy": None,
            "opp_player": player.complement().value,
            "opp_strategy": None,
        }

    game_history = []
    total_duration = 0
    search = model_value_maximization(model)

    for _ in range(num_games):
        start_time = time.time()
        simulate_one_game(game_history)
        end_time = time.time()
        duration = end_time - start_time
        total_duration += duration
        print(f"Completed game simulation in {duration:.2f} seconds")

    avg_duration = total_duration / num_games
    print(
        f"All {num_games} games completedwith an average of {avg_duration} seconds per game. Collected a total of {len(game_history)} experiences"
    )

    return game_history


#################### HELPER FUNCTIONS
def _random_opponent_search(model: TablutNet) -> Callable[[GameState], GameState]:
    default_depth = 4
    default_branching = 12
    searches = [
        # alpha_beta_basic(default_depth, default_branching),
        # alpha_beta_value_model(model, default_depth, default_branching),
        # alpha_beta_policy_model(model, 0.5, default_depth),
        # alpha_beta_full_model(model, 0.5, default_depth),
        model_value_maximization(model),
        # model_greedy_sampling(model),
    ]
    return random.choice(searches)


def _prepare_game_state_data(
    outcome: int, game_states: list[GameState], playing_as: Player
) -> list[tuple[GameState, int]]:
    game_history = []
    for state in game_states:
        print(f"processing:\n{state}\n")
        outcome = outcome if state.turn_player == playing_as else -1 * outcome
        game_history.append((state, outcome))

    return game_history


def _opponent_gameloop_wrapper(player, opponent_name, ip, search_func, results):
    outcome, game_states = play_game(player, opponent_name, ip, search_func, track=True)
    results["outcome"] = outcome
    results["game_states"] = game_states
