import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tablut import BOARD_LENGTH, GameState


# TODO: should move here the to_tensor function probably


class TablutNet(nn.Module):
    """Tablut value/policy network. It is implemented as a CNN that feeds its
    featmaps to 2 different linear layer heads: one computes a value in [-1,1]
    estimating if the turn player is going to win from this position, the other
    outputs a probability distribution over the input states.

    Turn player and who we are playing as is fed to the network as 2-dimensional
    1-hot-encoded vectors.

    `forward` method supports both single state evaluation (policy is P=1) and
    multistate evaluation + probability distribution"""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        conv_filters = [in_channels, 32, 64, 128]
        kernel_size = 3
        padding = 1
        # flattened feature size after convs (since padding keeps 9x9)
        conv_output_size = conv_filters[-1] * BOARD_LENGTH * BOARD_LENGTH

        # CNN section
        self.conv1 = nn.Conv2d(
            conv_filters[0], conv_filters[1], kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            conv_filters[1], conv_filters[2], kernel_size, padding=padding
        )
        self.conv3 = nn.Conv2d(
            conv_filters[2], conv_filters[3], kernel_size, padding=padding
        )

        # FCL head
        extra_features = 2 * 2  # 2 features 1-hot-encoded to a 2 dimensional vector
        hidden_size = 750
        # hidden_size = conv_output_size // 8
        self.fc = nn.Linear(conv_output_size + extra_features, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, 1)

        # --- activation functions for the last layer ---
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    @staticmethod
    def batch_to_tensor(
        states: list[GameState],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch a list of GameStates into tensors suitable for CNN input."""
        boards = []
        playing_as = []
        turn_players = []

        for s in states:
            b, p, t = s.to_tensor()
            boards.append(b)
            playing_as.append(p)
            turn_players.append(t)

        board_batch = torch.stack(boards)  # (N, C, H, W)
        playing_as_batch = torch.cat(playing_as)  # (N, 2)
        turn_players_batch = torch.cat(turn_players)  # (N, 2)

        return board_batch, playing_as_batch, turn_players_batch

    def forward(self, game_state: GameState | list[GameState]):
        if isinstance(game_state, list):
            board, playing_as, turn_player = TablutNet.batch_to_tensor(game_state)
        else:
            board, playing_as, turn_player = game_state.to_tensor()
            # add batch dimension for consistency
            board = board.unsqueeze(0)

        x = board.to(next(self.parameters()).device)

        # --- CNN Feature Extraction ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # concatenate all scalar info
        extra_features = torch.cat([turn_player, playing_as], dim=1).to(x.device)
        x = torch.cat((x, extra_features), dim=1)

        # --- Fully Connected Head ---
        x = F.relu(self.fc(x))

        value = self.tanh(self.value_head(x)).squeeze(-1)
        if not isinstance(game_state, list):
            return value
            # return value.squeeze(0)

        policy_logits = self.policy_head(x).squeeze(-1)
        probs = self.softmax(policy_logits)

        return value, probs
