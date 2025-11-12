import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import subprocess
import threading
import time
import torch
import torch.nn.functional as F
from client import play_game, parse_state
from tablut import Board, GameState, Player, Tile
from profiles import *
from network.model import TablutNet
from network.training_db import persist_self_play_run, latest_experiences
import datetime


def train(
    model: TablutNet,
    optimizer,
    loss_fn,
    iterations: int = 3,
    games: int = 5,
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

        # self playing the game
        print(f"\tRunning {games} self-play games...")
        run_self_play_games(model, num_games=games)

        # training on past experiences
        print(
            f"\tOptimizing model for {train_steps} steps with batch size of {batch_size}..."
        )
        total_loss = 0
        for _ in range(train_steps):
            loss = train_step(model, optimizer, loss_fn, batch_size)
            if loss is not None:
                total_loss += loss
        avg_loss = total_loss / train_steps if train_steps > 0 else 0

        print(f"\tIteration {iteration + 1} completed. Average Loss: {avg_loss:.6f}")
        # save model checkpoint periodically
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(
            model.state_dict(),
            f"checkpoints/tablut_model_checkpoint_iter_{timestamp}.pth",
        )
        print(f"  Model checkpoint saved at iteration {timestamp}")


def train_step(
    model: TablutNet,
    optimizer,
    loss_fn,
    batch_size: int,
):
    """
    Performs one training step using a batch of experiences sampled randomly from the buffer.
    """

    def value_loss(batch: list[tuple[GameState, GameState, int]]):
        states = [item[0] for item in batch]
        outcomes = [item[2] for item in batch]
        tensor_outcomes = torch.tensor(outcomes, dtype=torch.float32).to(
            next(model.parameters()).device
        )
        values = model.value(states)
        return loss_fn(values, tensor_outcomes)

    def policy_loss(batch: list[tuple[GameState, GameState, int]]):
        picked_moves = [item[1] for item in batch]
        policy_groups: list[list[GameState]] = []
        target_indices = []

        for i in range(len(batch)):
            target_move = picked_moves[i]
            search_space: list[GameState] = batch[i][0].next_moves()
            found = False
            for j in range(len(search_space)):
                if search_space[j] == target_move:
                    target_indices.append(j)
                    found = True
                    break
            policy_groups.append(search_space)
            if not found:
                print(batch[i][0])
                print(batch[i][0].board.board, end="\n\n\n")
                print(target_move)
                print(target_move.board.board, end="\n\n\n")
                exit(1)
        if len(policy_groups) != len(target_indices):
            raise Exception(
                "Some picked actions didnt match anyone in the search space apparently:"
                + f"len(policy_groups)={len(policy_groups)}\tlen(target_indices)={len(target_indices)}"
            )

        flat_candidates = [s for group in policy_groups for s in group]
        group_sizes = [len(g) for g in policy_groups]
        logits = model.train_policy(flat_candidates)
        logits_flat = logits.view(-1)

        policy_losses = []
        start = 0
        for i in range(len(target_indices)):
            group_logits = logits_flat[start : start + group_sizes[i]]
            log_probs = F.log_softmax(group_logits, dim=0)
            policy_losses.append(-log_probs[target_indices[i]])
            start = start + group_sizes[i]

        return torch.stack(policy_losses).mean()

    experience_buffer: list[tuple[GameState, GameState, int]] = latest_experiences()
    if len(experience_buffer) < batch_size:
        raise ValueError(
            f"Eperience buffer contains only {len(experience_buffer)} samples which is less than the required {batch_size} batch size."
        )
    # sample a random batch of experiences
    batch = random.sample(experience_buffer, batch_size)
    v_loss = value_loss(batch)
    p_loss = policy_loss(batch)
    total_loss = v_loss + 0.8 * p_loss

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
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    game_history: list[tuple[GameState, GameState, int]] = []
    analytics = []
    total_duration = 0
    for game_num in range(num_games):
        analytics_record, experiences = _simulate_one_game(model)
        analytics.append(analytics_record)
        game_history.extend(experiences)

        duration = analytics_record["duration_s"]
        total_duration += duration
        print(f"Completed game {game_num} simulation in {duration:.2f} seconds")

    # with ThreadPoolExecutor() as executor:
    #     future_to_game = {
    #         executor.submit(_simulate_one_game, model): i for i in range(num_games)
    #     }

    #     for future in as_completed(future_to_game):
    #         game_num = future_to_game[future]
    #         try:
    #             analytics_record, experiences = future.result()
    #             analytics.append(analytics_record)
    #             game_history.extend(experiences)

    #             duration = analytics_record["duration_s"]
    #             total_duration += duration
    #             print(f"Completed game {game_num} simulation in {duration:.2f} seconds")

    #         except Exception as exc:
    #             print(f"Game {game_num} generated an exception: {exc}")

    avg_duration = total_duration / num_games
    print(
        f"All {num_games} games completed with an average of {avg_duration} seconds per game. Collected {len(game_history)} experiences"
    )

    persist_self_play_run(game_history, analytics)
    return game_history


#################### HELPER FUNCTIONS
def self_contained_game_loop(
    player: Player,
    player_search: Callable[[GameState], GameState],
    opp_search: Callable[[GameState], GameState],
) -> tuple[list[tuple[GameState, GameState, int]], int]:
    with open("custom_states/initialState.json", "r") as file:
        initial_state_string = file.read()
    (game_state, _) = parse_state(initial_state_string, player)
    experience_buffer = []
    turn = 1

    print(f"Running self play game with player {player}")
    while not game_state.is_end_state():
        print(f"\n{game_state}")
        turn_search = player_search if player == game_state.turn_player else opp_search
        try:
            move = turn_search(game_state)
        except KeyboardInterrupt:
            while True:
                print("enter move")
                s = input()
                if s == "q":
                    move = turn_search(game_state)
                    break
                # trin & parse coordinates
                args = s.replace(" ", "").split(",")  # 0,1,0,5
                row_from, col_from = int(args[0]), int(args[1])
                row_to, col_to = int(args[2]), int(args[3])
                # if valid update the state
                if game_state.board.valid_move(row_from, col_from, row_to, col_to):
                    new_board = copy.deepcopy(game_state.board.board)
                    new_board[row_to][col_to] = new_board[row_from][col_from]
                    new_board[row_from][col_from] = Tile.EMPTY.value
                    new_board = Board(new_board)
                    new_board.solve_captures(row_to, col_to)

                    move = GameState(
                        new_board,
                        game_state.playing_as,
                        game_state.turn.complement(),
                    )
                    break
                else:
                    print(
                        "input a valid move in format `row_from, col_from, row_to, col_to`"
                    )
        experience_buffer.append((game_state, move))

        game_state = move  # note: this assumes move solved captured pawn
        turn += 1

    print(f"finished game on state\n{game_state}")
    if game_state.winner() == player:
        outcome = 1
    elif game_state.winner() == player.complement():
        outcome = -1
    else:
        outcome = 0

    return _prepare_game_state_data(outcome, experience_buffer, player), outcome


def server_game_loop(
    player: Player,
    player_search: Callable[[GameState], GameState],
    opp_search: Callable[[GameState], GameState],
) -> tuple[list[tuple[GameState, GameState, int]], int]:
    experiences: list[tuple[GameState, GameState, int]] = []

    print("Starting server...", end=" ")
    server = subprocess.Popen(
        ["ant", "server", "WHITE", "localhost"],
        cwd="C:\\Users\\danie\\codice\\uni\\TablutCompetition\\Tablut",
        # cwd="/home/danieletarek.iaisy/codice/personal/TablutCompetition/Tablut",
        stdout=open("server.log", "w"),
        start_new_session=True,  # detach completely
        shell=True,
    )
    time.sleep(1)
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
    experiences.extend(_prepare_game_state_data(outcome, game_turns, player))
    opp_thread.join()
    opp_outcome = opp_results["outcome"]
    opp_game_turns = opp_results["game_turns"]
    experiences.extend(
        _prepare_game_state_data(opp_outcome, opp_game_turns, player.complement())
    )

    return experiences, outcome


def _simulate_one_game(model: TablutNet):
    player = random.choice([Player.WHITE, Player.BLACK])
    # player_search_name, player_search = _random_search_profile(model)
    player_search_name, player_search = _random_search_profile(model)
    # player_search_name, player_search = ("mcts_deep_model", mcts_deep_model(model, 100))
    opp_search_name, opp_search = _random_search_profile(model)
    print(f"Selected opponent search: {opp_search_name}")

    start_time = datetime.datetime.now()
    experiences, outcome = self_contained_game_loop(player, player_search, opp_search)
    # experiences, outcome = server_game_loop(player, player_search, opp_search)
    end_time = datetime.datetime.now()

    analytics = {
        "trainee_player": player.value,
        "trainee_strategy": player_search_name,
        "trainee_outcome": outcome,
        "opp_player": player.complement().value,
        "opp_strategy": opp_search_name,
        "opp_outcome": -1 * outcome,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": (end_time - start_time).total_seconds(),
    }
    return analytics, experiences


def _random_search_profile(
    model: TablutNet,
) -> tuple[str, Callable[[GameState], GameState]]:
    default_depth = 5
    default_branching = 10

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
        # (
        #     "alpha_beta_full_model",
        #     alpha_beta_full_model(model, default_depth, default_branching),
        # ),
        # ("model_value_maximization", model_value_maximization(model)),
        # ("model_greedy_sampling", model_greedy_sampling(model)),
        ("mcts_fixed_model", mcts_fixed_model(model, 20, 90)),
        ("mcts_deep_model", mcts_deep_model(model, 90)),
        # ("mcts_shallow_model", mcts_shallow_model(model, 90)),
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
