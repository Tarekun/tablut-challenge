from datetime import datetime
import glob

import numpy as np
from tablut import GameState
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
        json.dump(data, f, indent=2)


def latest_experiences() -> list[tuple[GameState, GameState, int]]:
    files = glob.glob("trainruns/experiences_*.json")
    files.sort(reverse=True)
    latest_files = files[:5]
    # latest_files.append("trainruns/handcraftedExperiences.json")

    all_experiences: list[tuple[GameState, GameState, int]] = []
    for file in latest_files:
        with open(file, "r", encoding='utf-8') as f:
            data = json.load(f)
            for exp in data:
                all_experiences.append(
                    (
                        create_dict_state(exp["state"])[0],
                        create_dict_state(exp["move"])[0],
                        exp["outcome"],
                    )
                )

    return all_experiences


def persist_self_play_run(
    experiences: list[tuple[GameState, GameState, int]], analytics: list[dict]
):
    counter = 0
    mapped_experiences = []
    for state, move, outcome in experiences:
        state_board = state.board.board
        move_board = move.board.board
        if isinstance(state_board, np.ndarray):
            state_board = state_board.tolist()
        if isinstance(move_board, np.ndarray):
            move_board = move_board.tolist()
        state_dict = {"board": state_board, "turn": state.turn.value}
        move_dict = {"board": move_board, "turn": move.turn.value}
        experience_dict = {"state": state_dict, "move": move_dict, "outcome": outcome}
        mapped_experiences.append(experience_dict)

    create_timestamped_json_file("trainruns/experiences", mapped_experiences)
    append_to_json_file("trainruns/analytics.json", analytics)
