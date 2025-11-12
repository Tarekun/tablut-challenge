import numpy as np
import json

import random

def transform_state(board, next_board):
    pairs = []
    seen = set()
    # Rotate k times and flip horizontally to augment data
    for k in range(4):
        rot_state = np.rot90(board, k=k)
        rot_next = np.rot90(next_board, k=k)

        # Check if we've already seen this transformation
        for s, n in [
            (rot_state, rot_next),
            # TODO da rivedere
            (np.fliplr(rot_state), np.fliplr(rot_next)),
        ]:
            key = s.tobytes() + n.tobytes()
            if key not in seen:
                seen.add(key)
                pairs.append((s, n))

    return pairs


# # Create good black state
# BOARD = 9
# zero = np.zeros([9, 9], dtype=object)
# BLACK = 12
# WHITE = 8
# KING = 1
# black = 0
# white = 0
# king = 0

# king_pos = random.randint(0, BOARD * BOARD - 1)  # place king randomly

# for row in range(BOARD):
#     for col in range(BOARD):
#         if king < KING:
#             if king_pos == row * BOARD + col:
#                 zero[row][col] = "K"
#                 king += 1
#                 continue
#         if white < WHITE:
#             prob_white = random.randint(0, 25)
#             if prob_white == 0:
#                 zero[row][col] = "W"
#                 white += 1
#                 continue  
#         if black < BLACK:
#             prob_black = random.randint(0, 10)
#             if prob_black == 0:
#                 zero[row][col] = "B"
#                 black += 1
#                 continue
          

# print(zero)

# c_w = 0
# c_b = 0
# c_k = 0

# for row in range(BOARD):
#     for col in range(BOARD):
#         if zero[row][col] == "B":
#             c_b += 1
#         if zero[row][col] == "W":
#             c_w += 1
#         if zero[row][col] == "K":
#             c_k += 1
# print("Black:", c_b, "White:", c_w, "King:", c_k)
import random

BOARD = 9

# Coordinate speciali
THRONE = (4, 4)
CAMPS = {
    (3,0), (4,0), (5,0), (4,1),
    (0,3), (0,4), (0,5), (1,4),
    (3,8), (4,8), (5,8), (4,7),
    (8,3), (8,4), (8,5), (7,4)
}
ESCAPES = {
    (0,0), (0,1), (0,2), (0,6), (0,7), (0,8),
    (1,0), (1,8),
    (2,0), (2,8),
    (6,0), (6,8),
    (7,0), (7,8),
    (8,0), (8,1), (8,2), (8,6), (8,7), (8,8)
}

def generate_white_favorable_state():
    board = [["." for _ in range(BOARD)] for _ in range(BOARD)]

    # Posizioni valide vicino ai bordi ma NON accampamenti
    edge_positions = [
        (0,2), (0,6),
        (2,0), (6,0),
        (2,8), (6,8),
        (8,2), (8,6)
    ]

    # Posiziona il Re
    kx, ky = random.choice(edge_positions)
    board[kx][ky] = "K"

    # Posiziona 8–10 bianchi vicino al Re (evitando trono e accampamenti)
    white_count = random.randint(8, 10)
    while white_count > 0:
        x, y = random.randint(0, 8), random.randint(0, 8)
        if (
            board[x][y] == "."
            and (x, y) not in CAMPS
            and (x, y) != THRONE
            and abs(x - kx) <= 3
            and abs(y - ky) <= 3
        ):
            board[x][y] = "W"
            white_count -= 1

    # Posiziona 5–7 neri, preferibilmente negli accampamenti
    black_count = random.randint(5, 7)
    camp_positions = list(CAMPS)
    random.shuffle(camp_positions)
    for pos in camp_positions:
        if black_count <= 0:
            break
        x, y = pos
        board[x][y] = "B"
        black_count -= 1

    # Se restano neri da aggiungere, li mettiamo lontani dal Re
    while black_count > 0:
        x, y = random.randint(0,8), random.randint(0,8)
        if (
            board[x][y] == "."
            and (x, y) not in CAMPS
            and (x, y) != THRONE
            and (abs(x - kx) > 3 or abs(y - ky) > 3)
        ):
            board[x][y] = "B"
            black_count -= 1

    if (kx, ky) in ESCAPES:
        d_board = {"board": board, "turn": "B"}
    else:
        d_board = {"board": board, "turn": "W"}
    return d_board


# Genera e mostra 3 stati
for i in range(3):
    state = generate_white_favorable_state()
    print(f"\n🟢 Stato favorevole per i Bianchi #{i+1}")
    for row in state:
        print(" ".join(row))

with open("trainruns/handcrafted_states.json", "w") as file:
    handcrafted_states = []
    for i in range(10):
        state = generate_white_favorable_state()
        handcrafted_states.append(state)

    json.dump(handcrafted_states, file, indent=4)