import torch
import torch.nn.functional as F
import random
import subprocess
import time
from client import play_game
from tablut import Player
from tablut import GameState
from profiles import *
from network.model import TablutNet
from network.training_db import persist_self_play_run
import threading
import datetime


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
    experience_buffer: list[tuple[GameState, GameState, int]],
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
    picked_moves = [item[1] for item in batch]
    outcomes = [item[2] for item in batch]
    tensor_outcomes = torch.tensor(outcomes, dtype=torch.float32).to(
        next(model.parameters()).device
    )
    policy_inputs = []
    target_indices = []
    for i in range(len(batch)):
        target_move = picked_moves[i]
        search_space = batch[i][0].next_moves()
        # distribution = torch.zeros(len(search_space))
        for j in range(len(search_space)):
            if search_space[j] == target_move:
                target_indices.append(j)
                break
        policy_inputs.append(search_space)

    # value forward pass
    values = model.value(states)
    value_loss = loss_fn(values, tensor_outcomes)
    # policy forward pass
    logits_per_group = model.train_policy(policy_inputs)
    policy_losses = []
    for logits, target_idx in zip(logits_per_group, target_indices):
        print(len(logits_per_group))
        log_probs = F.log_softmax(logits, dim=0)
        policy_losses.append(-log_probs[target_idx])
    policy_loss = torch.stack(policy_losses).mean()

    # backward pass and update weights
    total_loss = value_loss + 1.0 * policy_loss
    optimizer.zero_grad()
    total_loss.backward()
    # Optional: Gradient clipping to prevent exploding gradients
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


def run_self_play_games(model, num_games=1) -> list[tuple[GameState, GameState, int]]:
    """Runs `num_games` number of games in a sequence collecting the states being played
    and the picked action at each state, together with the game result (±1) from the
    perspective of the turn player"""

    game_history: list[tuple[GameState, GameState, int]] = []
    analytics = []
    total_duration = 0

    # TODO: run games in parallel
    for _ in range(num_games):
        start_time = time.time()
        analytics_record, experiences = _simulate_one_game(model)
        end_time = time.time()

        analytics.append(analytics_record)
        game_history.extend(experiences)
        duration = end_time - start_time
        total_duration += duration
        print(f"Completed game simulation in {duration:.2f} seconds")

    avg_duration = total_duration / num_games
    print(
        f"All {num_games} games completedwith an average of {avg_duration} seconds per game. Collected a total of {len(game_history)} experiences"
    )

    persist_self_play_run(game_history, analytics)
    return game_history


#################### HELPER FUNCTIONS
def _simulate_one_game(model: TablutNet):
    player = random.choice([Player.WHITE, Player.BLACK])
    player_search_name, player_search = _random_search_profile(model)
    opp_search_name, opp_search = _random_search_profile(model)
    experiences: list[tuple[GameState, GameState, int]] = []
    start_time = datetime.datetime.now()

    print("Starting server...", end=" ")
    server = subprocess.Popen(
        ["ant", "server", "WHITE", "localhost"],
        cwd="/home/danieletarek.iaisy/codice/personal/TablutCompetition/Tablut",
        # cwd="C:\\Users\\danie\\codice\\uni\\TablutCompetition\\Tablut",
        stdout=open("server.log", "w"),
        start_new_session=True,  # detach completely
        shell=True,
    )
    print("Done")
    time.sleep(5)

    print("Starting opponent...", end=" ")
    opp_results = {}
    opp_thread = threading.Thread(
        target=_opponent_gameloop_wrapper,
        args=(
            player.complement(),
            "opponent",
            "localhost",
            opp_search,
            opp_results,
        ),
    )
    opp_thread.start()
    time.sleep(1)
    print("Done")

    outcome, game_turns = play_game(
        player, "trainee", "localhost", player_search, track=True
    )
    end_time = datetime.datetime.now()
    experiences.extend(_prepare_game_state_data(outcome, game_turns, player))
    opp_thread.join()
    opp_outcome = opp_results["outcome"]
    opp_game_turns = opp_results["game_turns"]
    experiences.extend(
        _prepare_game_state_data(opp_outcome, opp_game_turns, player.complement())
    )

    analytics = {
        "trainee_player": player.value,
        "trainee_strategy": player_search_name,
        "trainee_outcome": outcome,
        "opp_player": player.complement().value,
        "opp_strategy": opp_search_name,
        "opp_outcome": opp_outcome,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": (end_time - start_time).total_seconds(),
    }
    return analytics, experiences


def _random_search_profile(
    model: TablutNet,
) -> tuple[str, Callable[[GameState], GameState]]:
    default_depth = 4
    default_branching = 9
    default_top_p = 0.15
    searches = [
        # ("alpha_beta_basic", alpha_beta_basic(default_depth, default_branching)),
        # (
        #     "alpha_beta_value_model",
        #     alpha_beta_value_model(model, default_depth, default_branching),
        # ),
        # (
        #     "alpha_beta_policy_model",
        #     alpha_beta_policy_model(model, default_depth, default_branching),
        # ),
        (
            "alpha_beta_full_model",
            alpha_beta_full_model(model, default_depth, default_branching),
        ),
        # ("model_value_maximization", model_value_maximization(model)),
        # ("model_greedy_sampling", model_greedy_sampling(model)),
    ]
    return random.choice(searches)


def _prepare_game_state_data(
    outcome: int, game_turns: list[tuple[GameState, GameState]], playing_as: Player
) -> list[tuple[GameState, GameState, int]]:
    game_history: list[tuple[GameState, GameState, int]] = []
    for state, move in game_turns:
        outcome = outcome if state.turn_player == playing_as else -1 * outcome
        game_history.append((state, move, outcome))

    return game_history


def _opponent_gameloop_wrapper(player, opponent_name, ip, search_func, results):
    outcome, game_turns = play_game(player, opponent_name, ip, search_func, track=True)
    results["outcome"] = outcome
    results["game_turns"] = game_turns
