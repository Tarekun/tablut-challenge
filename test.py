import json
import numpy as np

def transform_state(board):
    pairs = []
    seen = set()
    # Rotate k times and flip horizontally to augment data
    for k in range(4):
        rot_state = np.rot90(board, k=k)

        # Check if we've already seen this transformation
        for s in [rot_state, np.fliplr(rot_state)]:
            key = s.tobytes()
            if key not in seen:
                seen.add(key)
                pairs.append((s))

    return pairs

matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, "B", 0, "W", 0, 0, 0],
    [0, 0, "B", 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, "W", 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, "K", 0, 0],
    [0, 0, 0, 0, 0, "W", 0, "W", 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, "W", 0]
]

matrix = [
    ["0", "0", "0", "0", "B", "0", "0", "0", "0"],
    ["0", "B", "0", "B", "0", "W", "0", "0", "0"],
    ["0", "0", "B", "0", "0", "0", "0", "W", "0"],
    ["B", "B", "0", "0", "0", "W", "0", "0", "0"],
    ["0", "0", "0", "B", "0", "0", "0", "0", "0"],
    ["0", "0", "0", "0", "0", "0", "K", "0", "0"],
    ["0", "0", "B", "0", "0", "W", "0", "W", "0"],
    ["0", "B", "0", "0", "B", "0", "0", "0", "0"],
    ["0", "0", "0", "B", "0", "0", "0", "W", "0"]
]

BOARD = 9
white = 0
black = 0
king = 0

for row in range(BOARD):
    for col in range(BOARD):
        if matrix[row][col] == "B":
            black += 1
        if matrix[row][col] == "W":
            white += 1
        if matrix[row][col] == "K":
            king += 1

print("Black:", black, "White:", white, "King:", king)

augmented_states = transform_state(np.array(matrix))

print(f"Generated {len(augmented_states)} augmented states:")
for i, state in enumerate(augmented_states):
    print(f"\n🟢 Augmented State #{i+1}")
    for row in state:
        print(" ".join(row))

board_dict = []

for i, state in enumerate(augmented_states):
    board_dict.append({"board": state.tolist(), "turn": "W", "outcome": 1})

with open("trainruns/handcrafted.json", "w") as file:
    json.dump(board_dict, file, indent=4)