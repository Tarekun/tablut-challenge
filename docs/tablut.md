# tablut.py

This file contains data structures that model the game of tablut in our python code. Main classes defined here are `GameState`, `Board`, and `Player`, although `Turn` and `Tile` are also included it's unlikely that the rest of the code will need to use those.

`Player` is an enum that can be either `WHITE` or `BLACK` and also implements a couple of uesful methods such as `is_white`/`is_black` and `complement` to get the other player in recursive research.

`Board` implements the board state, represented as a matrix (`list[list]`) of strings containing values of the enumeration `Tile`. It supports functions to check the state at specific tiles, like `is_empty`, `is_camp`, `is_escape`, functions to modify the the board state, like `pawn_moves` and `generate_all_moves`, and metrics to feed into heuristics, like `piece_count_ratio` or `move_count`.

`GameState` models the full game state from the POV of our agent tracking the current board, our agent player, the turn player, and the current turn number. This is the main type to be used in search algorithm as it supports generating the available moves with `next_moves` to be processed and allows to know if the state is a final one with `is_end_state`

## API Reference

### GameState

- `__init__(board: Board, playing_as: Player, turn_player: Player, turn=0)`
  - Initializes a GameState with the given board, player information and turn number
- `__str__() -> str`
  - Returns a string representation of the game state
- `to_tensor() -> tuple[Tensor, Tensor, Tensor]`
  - Returns a tuple containing game state elements as tensor. First element is the board,
    second is the 1-hot-encoding of we're playing as and third is the turn player
- `board: Board`
  - Returns the board of this game state
- `playing_as: Player`
  - Returns the player that we are playing as in this game state
- `turn_player: Player`
  - Returns the player whose turn it currently is
- `turn: int`
  - Returns the current turn number
- `next_moves() -> list[GameState]`
  - Generates all possible next game states from this state
- `is_end_state() -> bool`
  - Returns True if this game state represents an end state (game over), False otherwise

### Board

- `to_tensor() -> Tensor`
  - Returns a (3,9,9) tensor representing the board state. First channel is for the King's position,
    second channel encodes other white pawns and third channel is for black pawns
- `action_to(other) -> dict`
  - Returns the action (from, to) that transforms this board into the other board
- `at(row: int, col: int) -> str`
  - Returns the tile at [row,col]
- `is_empty(row: int, col: int) -> bool`
  - Returns True if the tile at [row,col] is empty, False otherwise
- `is_camp(row: int, col: int) -> bool`
  - Returns True if the tile at [row,col] is a camp position, False otherwise
- `is_escape(row: int, col: int) -> bool`
  - Returns True if the tile at [row,col] is an escape position, False otherwise
- `solve_captures()`
  - Updates the current board if captures can be made, by removing from the board
    the captured pawn
- `pawn_moves(row: int, col: int) -> list[Board]`
  - Generates all boards where the pawn at [row,col] can move to
- `generate_all_moves(player: Player) -> list[Board]`
  - Generates all possible moves for the given player
- `piece_count(player: Player) -> int`
  - Returns the number of pieces owned by the given player on the board
- `piece_count_ratio(player: Player) -> float`
  - Returns the ratio of pieces owned by the given player to the total possible pieces
- `moves_count(player: Player) -> int`
  - Returns the number of possible moves for the given player
- `moves_count_ratio(player: Player) -> float`
  - Returns the ratio of possible moves for the given player to the maximum possible moves
- `king_moves() -> int`
  - Returns the number of possible moves for the king piece

### Player

#### Variants

- `WHITE = "WHITE"`
- `BLACK = "BLACK"`

#### Methods

- `to_tensor() -> Tensor`
  - Returns a (1,2) 1-hot-encoded tensor of the player
- `is_white() -> bool`
  - Returns True if player is WHITE, False otherwise
- `is_black() -> bool`
  - Returns True if player is BLACK, False otherwise
- `complement() -> Player`
  - Returns the complementary player (WHITE <-> BLACK)
- `owns_pawn(tile: str) -> bool`
  - Returns True if the player can move the pawn at the given tile, False otherwise

### Tile

- `EMPTY = "EMPTY"`
- `BLACK = "BLACK"`
- `WHITE = "WHITE"`
- `KING = "KING"`

### Turn

#### Variants

- `WHITE = "WHITE"`
- `BLACK = "BLACK"`
- `WHITE_WINS = "WHITEWIN"`
- `BLACK_WINS = "BLACKWIN"`
- `DRAW = "DRAW"`

#### Methods

- `plays(player: Player) -> bool`
  - Returns True if the given player is playing this turn, False otherwise
- `wins(player: Player) -> bool`
  - Returns True if the given player has won this turn, False otherwise
- `game_finished() -> bool`
  - Returns True if the game has finished (a player has won or it's a draw), False otherwise
