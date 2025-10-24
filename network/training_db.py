from tablut import GameState
from client import parse_state
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


def persist_self_play_run(
    experiences: list[tuple[GameState, GameState, int]], analytics: list[dict]
):
    mapped_experiences = []
    for state, move, outcome in experiences:
        state_dict = {"board": state.board.board, "turn": state.turn.value}
        move_dict = {"board": move.board.board, "turn": move.turn.value}
        experience_dict = {"state": state_dict, "move": move_dict, "outcome": outcome}
        mapped_experiences.append(experience_dict)

    append_to_json_file("experiences.json", mapped_experiences)
    append_to_json_file("analytics.json", analytics)
