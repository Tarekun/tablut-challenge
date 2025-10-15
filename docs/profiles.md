# profiles.py

This file contains various agent profiles, an instance of a search algorithm implemented as a function `GameState` -> `GameState` that given the current game state returns the next one selected.
It implements various search profiles from naive alpha/beta search with hand picked heuristics, fix depth and width search, to max value/greedy sampling neural network search, to search augmented with a value and policy model.

## API Reference of the supported search profiles

### `alpha_beta_basic(max_depth: int, branching: int) -> Callable[[GameState], GameState]`

Basic Minimax algorithm with alpha/beta cuts. Uses a handfull of averaged
handpicked heuristic, picks a `branching` number of random actions from the available
ones and searches up to a `max_depth`

### `alpha_beta_value_model(model: TablutNet, max_depth: int) -> Callable[[GameState], GameState]`

Minimax algorithm with alpha/beta cuts that uses `model` as a state estimator
when reaching a node at `max_depth`

### `alpha_beta_policy_model(model: TablutNet, top_p: float, max_depth: int) -> Callable[[GameState], GameState]`

Minimax algorithm with alpha/beta cuts that uses hand picked averaged heuristics,
a `max_depth` stopping cretirion, and a top P policy algorithm with the probability
distributions provided by `model`.
`model` takes all available actions and returns a probability distribution over them,
we pick the most probable action to search until the cumulative probability reaches top_p

### `alpha_beta_full_model(model: TablutNet, top_p: float, max_depth: int) -> Callable[[GameState], GameState]`

Minimax alpha/beta search fully guided by the network `model`.
It selects actions with cumulative probability of `top_p` according to `model`
and evaluates states with `model`, up to `max_depth` in the search tree

### `model_value_maximization_search(model: TablutNet) -> Callable[[GameState], GameState]`

Given `model` that outputs a value for each state given as input
selects the action that maximizes this value

## API Reference of reusable building blocks

### `_handpicked_heuristics(state: GameState) -> float`

Hand-picked heuristic function that combines multiple board metrics into a single evaluation score between -1 and 1.

### `_network_value_heuristic(model: TablutNet) -> Callable[[GameState], float]`

Creates a heuristic function that uses a neural network to evaluate game states.

### `_random_fixed_actions(num: int) -> Callable[[GameState], list[GameState]]`

Creates an action selection policy that randomly selects a fixed number of actions from available moves.

### `_network_top_p_policy(model: TablutNet, top_p: float) -> Callable[[GameState], list[GameState]]`

Creates an action selection policy that uses a neural network's probability distribution to select actions with cumulative probability up to `top_p`.

### `_max_depth_criterion(max_depth: int) -> Callable[[GameState, int], bool]`

Creates a stopping criterion that terminates search when maximum depth is reached or game is over.
