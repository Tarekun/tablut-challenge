from datetime import datetime
import glob
import numpy as np
from tablut import GameState, Board
from client import create_dict_state
import os
import json


def append_to_json_file(file_path, new_data):
    # read already existing data
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        if not isinstance(existing_data, list):
            raise ValueError(f"JSON file {file_path} does not contain an array")
    else:
        existing_data = []

    if isinstance(new_data, list):
        existing_data.extend(new_data)
    else:
        existing_data.append(new_data)

    # write back with new updates
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, default=str)


def create_timestamped_json_file(base_filename, data):
    """
    Create a new JSON file with timestamp in the filename
    """
    # Create directory if it doesn't exist

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.json"

    # Write data to the new file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def latest_experiences() -> list[tuple[GameState, GameState, int]]:
    files = glob.glob("trainruns/experiences_*.json")
    files.sort(reverse=True)
    latest_files = files[:7]
    # latest_files.append("trainruns/handcraftedExperiences.json")

    # fetch some experiences from the db
    all_experiences: list[tuple[GameState, GameState, int]] = []
    for file in latest_files:
        with open(file, "r") as f:
            data = json.load(f)
            for exp in data:
                all_experiences.append(
                    (
                        create_dict_state(exp["state"])[0],
                        create_dict_state(exp["move"])[0],
                        exp["outcome"],
                    )
                )
    # augment experiences by symmetries
    augmented_experiences = []
    for state, move, outcome in all_experiences:
        augmented_experiences.extend(transform_state(state, move, outcome))

    return augmented_experiences


def transform_state(
    state: GameState, move: GameState, outcome: int
) -> list[tuple[GameState, GameState, int]]:
    pairs = []
    seen = set()
    # rotate and flip horizontally 4 times
    for k in range(4):
        rot_state = np.rot90(state.board.board, k=k)
        rot_next = np.rot90(move.board.board, k=k)

        for s, m in [
            (rot_state, rot_next),
            (np.fliplr(rot_state), np.fliplr(rot_next)),
        ]:
            key = s.tobytes() + m.tobytes()
            if key not in seen:
                seen.add(key)
                symmetric_state = GameState(
                    Board(s.tolist()), state.playing_as, state.turn
                )
                symmetric_move = GameState(
                    Board(m.tolist()), move.playing_as, move.turn
                )
                pairs.append((symmetric_state, symmetric_move, outcome))

    return pairs


def persist_self_play_run(
    experiences: list[tuple[GameState, GameState, int]], analytics: list[dict]
):
    mapped_experiences = []
    for state, move, outcome in experiences:
        state_dict = {"board": state.board.board, "turn": state.turn.value}
        move_dict = {"board": move.board.board, "turn": move.turn.value}
        experience_dict = {"state": state_dict, "move": move_dict, "outcome": outcome}
        mapped_experiences.append(experience_dict)

    create_timestamped_json_file("trainruns/experiences", mapped_experiences)
    append_to_json_file("trainruns/analytics.json", analytics)
