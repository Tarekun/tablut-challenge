from enum import Enum
import copy
import numpy as np
from torch import float32, Tensor, tensor


BOARD_LENGTH = 9
WHITE_PIECES = 9
BLACK_PIECES = 16
MAX_PAWN_MOVES = 16
WHITE_BLACK_RATIO = WHITE_PIECES / BLACK_PIECES
cols_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]


def string_coordinates(row: int, col: int) -> str:
    """Converts tile coordinates from 0-index (row,col) to server internal letter/digit system"""
    return f"{cols_letters[col]}{row+1}"


class Player(Enum):
    WHITE = "WHITE"
    BLACK = "BLACK"

    def to_tensor(self) -> Tensor:  # (1, 2), dtype=torch.float32
        """Returns a (1,2) 1-hot-encoded tensor of the player"""
        if self == Player.WHITE:
            return tensor([[1, 0]], dtype=float32)
        else:
            return tensor([[0, 1]], dtype=float32)

    def is_white(self) -> bool:
        """Returns True if player is WHITE, False otherwise"""
        return self.value == Player.WHITE.value

    def is_black(self) -> bool:
        """Returns True if player is BLACK, False otherwise"""
        return self.value == Player.BLACK.value

    def complement(self):
        """Returns the complementary player (WHITE <-> BLACK)"""
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK

    def owns_pawn(self, tile: str):
        """Returns True if the player can move the pawn at the given tile, False otherwise"""
        if self == Player.BLACK:
            return tile == Tile.BLACK.value
        elif self == Player.WHITE:
            return tile == Tile.WHITE.value or tile == Tile.KING.value
        else:
            return False


class Turn(Enum):
    WHITE = "WHITE"
    BLACK = "BLACK"
    WHITE_WINS = "WHITEWIN"
    BLACK_WINS = "BLACKWIN"
    DRAW = "DRAW"

    def plays(self, player: Player) -> bool:
        """Returns True if the given player is playing this turn, False otherwise"""
        return self.value == player.value

    def wins(self, player: Player) -> bool:
        """Returns True if the given player has won this turn, False otherwise"""
        return (self == Turn.WHITE_WINS and player == Player.WHITE) or (
            self == Turn.BLACK_WINS and player == Player.BLACK
        )

    def game_finished(self):
        """Returns True if the game has finished (a player has won or it's a draw), False otherwise"""
        return self == Turn.WHITE_WINS or self == Turn.BLACK_WINS or self == Turn.DRAW


class Tile(Enum):
    EMPTY = "EMPTY"
    BLACK = "BLACK"
    WHITE = "WHITE"
    KING = "KING"
    # THRONE = "THRONE"


class Board:
    def __init__(self, board: list[list[str]]):
        self.board: list[list[str]] = board

    def __str__(self) -> str:
        string = ""
        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
                string += (
                    "░"
                    if self.board[row][col] == Tile.EMPTY.value
                    else self.board[row][col][0:1]
                )
            string += "\n"
        return string

    def to_tensor(self) -> Tensor:
        """Returns a (3,9,9) tensor representing the board state. First channel is for the King's position,
        second channel encodes other white pawns and third channel is for black pawns"""
        encoded = np.zeros((3, 9, 9), dtype=np.float32)
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                tile = self[i][j]
                if tile == Tile.KING.value:
                    encoded[0, i, j] = 1.0
                elif tile == Tile.WHITE.value:
                    encoded[1, i, j] = 1.0
                elif tile == Tile.BLACK.value:
                    encoded[2, i, j] = 1.0
                else:
                    # skip empty tiles
                    pass

        return tensor(encoded)

    def action_to(self, other) -> dict:  # type: ignore
        """Returns the action (from, to) that transforms this board into the other board"""
        differences = []
        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
                if self[row][col] != other[row][col]:
                    differences.append({"row": row, "col": col})

        if len(differences) != 2:
            raise ValueError(
                f"This board and argument board have {len(differences)} different tiles so no action can take from one to another"
            )

        # take first and second different tiles
        first = differences[0]
        second = differences[1]

        if other[first["row"]][first["col"]] == Tile.EMPTY.value:
            return {
                "from": string_coordinates(first["row"], first["col"]),
                "to": string_coordinates(second["row"], second["col"]),
                "turn": self[second["row"]][second["col"]],
            }
        else:
            return {
                "from": string_coordinates(second["row"], second["col"]),
                "to": string_coordinates(first["row"], first["col"]),
                "turn": self[first["row"]][first["col"]],
            }

    def at(self, row: int, col: int):
        """Returns the tile at [row,col]"""
        return self.board[row][col]

    def __getitem__(self, row: int) -> list[str]:
        """Enables board[row] syntax, returns the row"""
        return self.board[row]

    def is_empty(self, row: int, col: int) -> bool:
        """Returns True if the tile at [row,col] is empty, False otherwise"""
        try:
            return self[row][col] == Tile.EMPTY.value
        # TODO: should i just raise and foggetabadit?
        except IndexError:
            return False

    def is_camp(self, row: int, col: int) -> bool:
        """Returns True if the tile at [row,col] is a camp position, False otherwise"""
        return any(
            [
                # horizontal camps
                (row == 0 and col in (3, 4, 5)),
                (row == 1 and col == 4),
                (row == 7 and col == 4),
                (row == 8 and col in (3, 4, 5)),
                # vertical camps
                (col == 0 and row in (3, 4, 5)),
                (col == 1 and row == 4),
                (col == 7 and row == 4),
                (col == 8 and row in (3, 4, 5)),
            ]
        )

    def is_escape(self, row: int, col: int) -> bool:
        """Returns True if the tile at [row,col] is an escape position, False otherwise"""
        return any(
            [
                # horizontal escapes
                (row == 0 and col in (1, 2, 6, 7)),
                (row == 8 and col in (1, 2, 6, 7)),
                # vertical escapes
                (col == 0 and row in (1, 2, 6, 7)),
                (col == 8 and row in (1, 2, 6, 7)),
            ]
        )

    def solve_captures(self):
        """Updates the current board if captures can be made, by removing from the board
        the captured pawn"""
        # TODO: implement if needed
        pass

    def pawn_moves(self, row: int, col: int) -> list:
        """Generates all boards where the pawn at [row,col] can move to"""
        up = (1, 0)
        down = (-1, 0)
        left = (0, -1)
        right = (0, 1)
        moves: list[Board] = []
        pawn = self.at(row, col)
        if pawn == Tile.EMPTY.value:
            raise ValueError(f"Tile at [{row},{col}] is empty and no pawn can be moved")

        for direction in [up, down, left, right]:
            row_change, col_change = direction
            for step in range(1, BOARD_LENGTH):
                moved_row = row + (step * row_change)
                moved_col = col + (step * col_change)
                # TODO: refactor to a simpler valid_move?
                # TODO: pawns in the middle of the camp cant move now
                if (
                    self.is_empty(moved_row, moved_col)
                    and not self.is_camp(moved_row, moved_col)
                    and 0 <= moved_row < BOARD_LENGTH
                    and 0 <= moved_col < BOARD_LENGTH
                ):
                    new_board = copy.deepcopy(self.board)
                    new_board[row][col] = Tile.EMPTY.value
                    new_board[moved_row][moved_col] = pawn
                    new_board_class = Board(new_board)
                    # TODO: is this really needed or does the server handle it?
                    # new_board_class.solve_captures()
                    moves.append(new_board_class)

                else:
                    # cont move there so the path is blocked, change direction
                    break

        return moves

    def generate_all_moves(self, player: Player) -> list:
        """Generates all possible moves for the given player"""
        moves = []
        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
                if player.owns_pawn(self[row][col]):
                    pawn_moves = self.pawn_moves(row, col)
                    moves.extend(pawn_moves)

        return moves

    def piece_count(self, player: Player) -> int:
        """Returns the number of pieces owned by the given player on the board"""
        total = 0
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if player.owns_pawn(self[i][j]):
                    total += 1

        return total

    def piece_count_ratio(self, player: Player) -> float:
        """Returns the ratio of pieces owned by the given player to the total possible pieces"""
        if player.is_white():
            return self.piece_count(player) / WHITE_PIECES
        else:
            return self.piece_count(player) / BLACK_PIECES

    def moves_count(self, player: Player) -> int:
        """Returns the number of possible moves for the given player"""
        return len(self.generate_all_moves(player))

    def moves_count_ratio(self, player: Player) -> float:
        """Returns the ratio of possible moves for the given player to the maximum possible moves"""
        max_moves = (
            MAX_PAWN_MOVES * BLACK_PIECES
            if player.is_black()
            else MAX_PAWN_MOVES * WHITE_PIECES
        )
        return self.moves_count(player) / max_moves

    def king_moves(self) -> int:
        """Returns the number of possible moves for the king piece"""
        row, col = None, None
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self[i][j] == Tile.KING.value:
                    row, col = i, j

        if row is None or col is None:
            return 0
        else:
            return len(self.pawn_moves(row, col))


class GameState:
    def __init__(self, board: Board, playing_as: Player, turn_player: Player, turn=0):
        """Initializes a GameState with the given board, player information and turn number"""
        self._board = board
        self._playing_as = playing_as
        self._turn_player = turn_player
        self._turn = turn

    def __str__(self) -> str:
        """Returns a string representation of the game state"""
        header = f"PLAYNG AS: {self._playing_as}\nTURN: {self._turn_player}\n"
        return f"{header}\n{self.board}"

    def to_tensor(self) -> tuple[Tensor, Tensor, Tensor]:
        """Returns a tuple containing game state elements as tensor. First element is the board,
        second is the 1-hot-encoding of we're playing as and third is the turn player"""
        return (
            self.board.to_tensor(),
            self.playing_as.to_tensor(),
            self.turn_player.to_tensor(),
        )

    @property
    def board(self) -> Board:
        return self._board

    @property
    def playing_as(self) -> Player:
        return self._playing_as

    @property
    def turn_player(self) -> Player:
        return self._turn_player

    @property
    def turn(self) -> int:
        return self._turn

    def next_moves(self):
        """Generates all possible next game states from this state"""
        moves = []
        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
                if self.turn_player.owns_pawn(self.board[row][col]):
                    pawn_moves = self.board.pawn_moves(row, col)
                    moves.extend(
                        [
                            GameState(
                                move,
                                self.playing_as,
                                self.turn_player.complement(),
                                turn=self.turn + 1,
                            )
                            for move in pawn_moves
                        ]
                    )

        return moves

    def is_end_state(self) -> bool:
        """Returns True if this game state represents an end state (game over), False otherwise"""
        # TODO: implement
        return False
