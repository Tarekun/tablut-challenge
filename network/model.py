import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tablut import BOARD_LENGTH, GameState, Board, Player, Tile


def embed_player(player: Player) -> Tensor:
    """Returns a (1,2) 1-hot-encoded tensor of the player"""
    if player == Player.WHITE:
        return torch.tensor([[1, 0]], dtype=torch.float32)
    else:
        return torch.tensor([[0, 1]], dtype=torch.float32)


def embed_board(board: Board) -> Tensor:
    """Returns a (3,9,9) tensor representing the board state. First channel is for the King's position,
    second channel encodes other white pawns and third channel is for black pawns"""

    encoded = np.zeros((3, 9, 9), dtype=np.float32)
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            tile = board[i][j]
            if tile == Tile.KING.value:
                encoded[0, i, j] = 1.0
            elif tile == Tile.WHITE.value:
                encoded[1, i, j] = 1.0
            elif tile == Tile.BLACK.value:
                encoded[2, i, j] = 1.0
            else:
                # skip empty tiles
                pass

    return torch.tensor(encoded)


def embed_game_state(
    state: GameState,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns a tuple containing game state elements as tensor. First element is the board,
    second is the 1-hot-encoding of we're playing as and third is the turn player"""
    return (
        embed_board(state.board),
        embed_player(state.playing_as),
        embed_player(state.turn_player),
    )


def embed_batch_states(
    states: list[GameState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch a list of GameStates into tensors suitable for CNN input."""
    boards = []
    playing_as = []
    turn_players = []

    for s in states:
        b, p, t = embed_game_state(s)
        boards.append(b)
        playing_as.append(p)
        turn_players.append(t)

    board_batch = torch.stack(boards)  # (N, C, H, W)
    playing_as_batch = torch.cat(playing_as)  # (N, 2)
    turn_players_batch = torch.cat(turn_players)  # (N, 2)

    return board_batch, playing_as_batch, turn_players_batch


class TablutNet(nn.Module):
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
        self.bn1 = nn.BatchNorm2d(conv_filters[1])
        self.conv2 = nn.Conv2d(
            conv_filters[1], conv_filters[2], kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(conv_filters[2])
        self.conv3 = nn.Conv2d(
            conv_filters[2], conv_filters[3], kernel_size, padding=padding
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(conv_filters[1], conv_filters[3], kernel_size=1),
            nn.BatchNorm2d(conv_filters[3])
        )
        self.bn3 = nn.BatchNorm2d(conv_filters[3])


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

    def _feature_extraction(self, game_state: GameState | list[GameState]):
        """Embed single or batch of states → flattened features."""
        if isinstance(game_state, list):
            board, playing_as, turn_player = embed_batch_states(game_state)
        else:
            board, playing_as, turn_player = embed_game_state(game_state)
            board = board.unsqueeze(0)  # add batch dimension for consistency

        x = board.to(next(self.parameters()).device)
        x = F.relu(self.bn1(self.conv1(x)))
        residual = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # print(f"Residual: {self.downsample(residual).shape}")
        # print(f"X: {x.shape}")
        
        x += self.downsample(residual)
        x = F.relu(x)
        x = x.view(x.size(0), -1)

        extra_features = torch.cat([turn_player, playing_as], dim=1).to(x.device)
        x = torch.cat((x, extra_features), dim=1)
        x = F.relu(self.fc(x))
        return x

    def policy(self, states: list[GameState]):
        """
        Given candidate next states, returns a probability distribution
        over them (sums to 1).
        """
        logits = self.train_policy(states)
        probs = self.softmax(logits)
        return probs

    def value(self, state: GameState | list[GameState]):
        """
        Computes scalar value(s) ∈ [-1, 1] for given state(s).
        - Single GameState → single scalar tensor
        - List[GameState] → tensor of shape (N,)
        """
        x = self._feature_extraction(state)
        value = self.tanh(self.value_head(x)).squeeze(-1)
        return value

    def train_policy(self, states: list[GameState]):
        """
        Given candidate next states, returns raw logits (no softmax).
        Used during training to compute per-group log-softmax and NLL loss.
        """
        x = self._feature_extraction(states)
        logits = self.policy_head(x).squeeze(-1)
        return logits

    def forward(self, game_state: GameState | list[GameState]):
        pass
