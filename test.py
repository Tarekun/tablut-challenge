from network.model import TablutNet
from network.training import train
from profiles import alpha_beta_basic
from tablut import *
import torch


model = TablutNet()
optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss()
train(model, optim, loss_fn, 1, 1, 1, 2)


# from client import parse_state

# with open("initialState.json", "r") as file:
#     initial_state_string = file.read()
# player = Player.BLACK
# (game_state, _) = parse_state(initial_state_string, player)
# search = alpha_beta_basic(2, 3)

# print(game_state)
# game_state = search(game_state)
# print(game_state)
# print("ora le mosse")
# for nextt in game_state.next_moves():
#     print(nextt)
