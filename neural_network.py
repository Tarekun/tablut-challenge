import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tablut import BOARD_LENGTH, GameState


class TablutHeuristicNet(nn.Module):
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
            board, playing_as, turn_player = TablutHeuristicNet.batch_to_tensor(
                game_state
            )
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


class TablutTrainer:
    """
    Handles self-play, experience collection, and training of the neural network.
    """

    def __init__(self, model, learning_rate=1e-4, buffer_size=100000):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        self.experience_buffer = deque(
            maxlen=buffer_size
        )  # Stores (state, target_value)

    def train_step(self, batch_size=32):
        """
        Performs one training step using a batch of experiences sampled randomly from the buffer.
        """

        if len(self.experience_buffer) < batch_size:
            raise ValueError(
                f"Eperience buffer contains only {len(self.experience_buffer)} samples which is less than the required {batch_size} batch size."
            )

        # Sample a batch of experiences randomly
        batch = random.sample(self.experience_buffer, batch_size)
        states, target_values = zip(*batch)

        # Convert to tensors
        # Assume states are tuples (board, turn_ind, player_ind)
        board_batch = torch.stack([s[0] for s in states])
        turn_batch = torch.stack([s[1] for s in states])
        player_batch = torch.stack([s[2] for s in states])
        target_batch = torch.tensor(target_values, dtype=torch.float32)

        # Forward pass
        # TODO: da rivedere
        predicted_values = self.model(board_batch, turn_batch, player_batch)
        loss = self.mse_loss(predicted_values, target_batch)
        # Backward pass and update weights
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        iterations=10000,
        games: int = 10,
        train_steps=100,
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
            run_self_play_game(self.model, num_games=games)

            print(
                f"\tOptimizing model for {train_steps} steps with batch size of {batch_size}..."
            )
            total_loss = 0
            for step in range(train_steps):
                loss = self.train_step(batch_size)
                if loss is not None:
                    total_loss += loss
            avg_loss = total_loss / train_steps if train_steps > 0 else 0
            print(
                f"\tIteration {iteration + 1} completed. Average Loss: {avg_loss:.6f}"
            )

            # Save model checkpoint periodically
            if (iteration + 1) % 100 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"tablut_model_checkpoint_iter_{iteration + 1}.pth",
                )
                print(f"  Model checkpoint saved at iteration {iteration + 1}")


def heuristic_maximization_search(model: TablutHeuristicNet):
    def maximize_heuristic(state: GameState) -> GameState:
        best_value = float("-inf")
        best_move = None
        moves: list[GameState] = state.next_moves()
        print(f"processing {len(moves)} moves")

        with torch.no_grad():
            print(f"inputting {len(moves)} moves into the model")
            (values, probs) = model(moves)  # shape: (N,2)
            print(f"got back {len(values)} heuristic values {values}")
            print(f"got back {len(probs)} probabilities over available moves {probs}")

        # find best move index
        best_idx = int(torch.argmax(values).item())
        best_value = values[best_idx].item()

        print(f"best value found was: {best_value}")
        return moves[best_idx]

    return maximize_heuristic


def run_self_play_game(model, num_games=1):
    import subprocess
    import time
    from client import play_game
    from tablut import Player

    game_history = []
    search = heuristic_maximization_search(model)

    for _ in range(num_games):
        player = Player.WHITE

        print("Starting server...", end=" ")
        server = subprocess.Popen(
            ["ant", "server", "WHITE", "localhost"],
            cwd="C:\\Users\\danie\\codice\\uni\\TablutCompetition\\Tablut",
            stdout=open("server.log", "w"),
            start_new_session=True,  # detach completely
            shell=True,
        )
        time.sleep(1)
        print("Done")

        print("Starting opponent...", end=" ")
        opponent = subprocess.Popen(
            ["python", "main.py", player.complement().value, "localhost"],
            cwd="C:\\Users\\danie\\codice\\uni\\tablut-challenge",
            stdout=open("opponent.log", "w"),
            start_new_session=True,  # detach completely
            shell=True,
        )
        time.sleep(1)
        print("Done")

        outcome, game_states = play_game(
            player, "Trainee", "localhost", search, track=True
        )
        for state in game_states:
            print(f"processing:\n{state}\n")
            outcome = outcome if state.turn_player == player else -1 * outcome
            game_history.append((state, outcome))

        for state, outcome in game_history:
            print(
                f"THIS FOLLOWING STATE PLAYED BY WHITE GOT OUTCOME {outcome}\n{state}",
                end="\n\n\n",
            )

        return game_history


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TablutHeuristicNet().to(device)

    # Initialize the trainer
    trainer = TablutTrainer(model)

    # Start the training loop
    # Note: The run_self_play_game method needs a full game implementation to work.
    # trainer.train(num_iterations=1000) # Example call
    print("Model and trainer initialized.")
    print("You need to implement the 'run_self_play_game' method with your game logic.")
