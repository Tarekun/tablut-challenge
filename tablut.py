from enum import Enum


BOARD_LENGTH = 9
WHITE_PIECES = 9
BLACK_PIECES = 16
MAX_PAWN_MOVES = 16
WHITE_BLACK_RATIO = WHITE_PIECES / BLACK_PIECES


class Player(Enum):
    WHITE = "WHITE"
    BLACK = "BLACK"

    def is_white(self) -> bool:
        return self.value == "W"

    def is_black(self) -> bool:
        return self.value == "B"

    def complement(self):
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
    WHITE_WINS = "WHITEWINS"
    BLACK_WINS = "BLACKWINS"
    DRAW = "DRAW"

    def plays(self, player: Player) -> bool:
        """Returns True if the given player is playing this turn, False otherwise"""
        print(self.value)
        print(player.value)
        return self.value == player.value

    def game_finished(self):
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
                if self.is_empty(moved_row, moved_col) and not self.is_camp(
                    moved_row, moved_col
                ):
                    new_board = self.board.copy()
                    new_board[row][col] = Tile.EMPTY.value
                    new_board[moved_row][moved_col] = pawn
                    new_board_class = Board(new_board)
                    # TODO: is this really needed or does the server handle it?
                    new_board_class.solve_captures()
                    moves.append(new_board_class)

                else:
                    # cont move there so the path is blocked, change direction
                    break

        return moves

    def generate_all_moves(self, player: Player) -> list:
        moves = []
        for row in range(1, BOARD_LENGTH):
            for col in range(1, BOARD_LENGTH):
                if player.owns_pawn(self[row][col]):
                    pawn_moves = self.pawn_moves(row, col)
                    moves.extend(pawn_moves)

        return moves

    def piece_count(self, player: Player) -> int:
        total = 0
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if player.owns_pawn(self[i][j]):
                    total += 1

        return total

    def piece_count_ratio(self, player: Player) -> float:
        if player.is_white():
            return self.piece_count(player) / WHITE_PIECES
        else:
            return self.piece_count(player) / BLACK_PIECES

    def moves_count(self, player: Player) -> int:
        return len(self.generate_all_moves(player))

    def moves_count_ratio(self, player: Player) -> float:
        max_moves = (
            MAX_PAWN_MOVES * BLACK_PIECES
            if player.is_black()
            else MAX_PAWN_MOVES * WHITE_PIECES
        )
        return self.moves_count(player) / max_moves

    def king_moves(self) -> int:
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
        self._board = board
        self._playing_as = playing_as
        self._turn_player = turn_player
        self._turn = turn

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
        moves = []
        for row in range(1, BOARD_LENGTH):
            for col in range(1, BOARD_LENGTH):
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
        return False
