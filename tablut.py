from enum import Enum
import copy
import numpy as np
from typing import Tuple


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

    def is_white(self) -> bool:
        return self.value == Player.WHITE.value

    def is_black(self) -> bool:
        return self.value == Player.BLACK.value

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
    WHITE_WINS = "WHITEWIN"
    BLACK_WINS = "BLACKWIN"
    DRAW = "DRAW"

    def plays(self, player: Player) -> bool:
        """Returns True if the given player is playing this turn, False otherwise"""
        return self.value == player.value

    def wins(self, player: Player) -> bool:
        return (self == Turn.WHITE_WINS and player == Player.WHITE) or (
            self == Turn.BLACK_WINS and player == Player.BLACK
        )

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
        self._previous: list[list[str]] | None = None

    @property
    def previous(self):
        if self._previous is not None:
            return self._previous
        else:
            return self.board

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

    def action_to(self, other) -> dict:  # type: ignore
        other = other.previous
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

    def camp_id(self, row: int, col: int) -> str | None:
        if (row == 0 and col in (3, 4, 5)) or (row == 1 and col == 4):
            return "top"
        if (row == 8 and col in (3, 4, 5)) or (row == 7 and col == 4):
            return "bottom"
        if (col == 0 and row in (3, 4, 5)) or (col == 1 and row == 4):
            return "left"
        if (col == 8 and row in (3, 4, 5)) or (col == 7 and row == 4):
            return "right"
        return None

    def is_throne(self, row: int, col: int) -> bool:
        return row == 4 and col == 4

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

    def valid_move(
        self, from_row: int, from_col: int, to_row: int, to_col: int
    ) -> bool:
        """Returns `True` iff the move from (from_row, from_col) to (to_row, to_col) is
        a valid one in the current board"""
        return (
            self.is_empty(to_row, to_col)
            and self.check_inside_board(to_row, to_col)
            and (
                # we stay inside the same camp
                self.camp_id(to_row, to_col) == self.camp_id(from_row, from_col)
                # OR we're targetting a non-camp tile
                or self.is_camp(to_row, to_col) is None
            )
        )

    def solve_captures(self, row, col):
        """Updates the current board if captures can be made, by removing from the board
        the captured pawn. `row` and `col` refer to the tile the last pawn moved to"""

        up = (1, 0)
        down = (-1, 0)
        left = (0, -1)
        right = (0, 1)
        pawn = self.at(row, col)

        def capture_available(pawn, enemy_pawn, ally_pawn):
            # se gioca white
            return (
                (pawn == Tile.WHITE.value or pawn == Tile.KING.value)
                and enemy_pawn == Tile.BLACK.value
                and (
                    ally_pawn == Tile.WHITE.value
                    or ally_pawn == Tile.KING.value
                    or self.is_throne(ally_row, ally_col)
                    or self.is_camp(ally_row, ally_col)
                )
            )

        for rd, cd in [up, down, left, right]:
            check_row = row + rd
            check_col = col + cd
            if self.check_inside_board(check_row, check_col):
                enemy_pawn = self.at(check_row, check_col)
                # PLAYING WHITE
                if pawn == Tile.WHITE.value or pawn == Tile.KING.value:
                    if enemy_pawn == Tile.BLACK.value:
                        ally_row = check_row + rd
                        ally_col = check_col + cd
                        if self.check_inside_board(ally_row, ally_col):
                            ally_pawn = self.at(ally_row, ally_col)

                            # TODO; make this logic consistent
                            if (
                                ally_pawn == Tile.WHITE.value
                                or ally_pawn == Tile.KING.value
                                or
                                # (new_board_class.is_camp(ally_row, ally_col) and ally_pawn == Tile.EMPTY.value) or
                                self.is_throne(ally_row, ally_col)
                            ):

                                self._previous = copy.deepcopy(self.board)
                                self.board[check_row][check_col] = Tile.EMPTY.value

                # PLAYING BLACK
                elif pawn == Tile.BLACK.value:
                    # ENEMY WHITE
                    if enemy_pawn == Tile.WHITE.value:
                        ally_row = check_row + rd
                        ally_col = check_col + cd
                        if self.check_inside_board(ally_row, ally_col):
                            ally_pawn = self.at(ally_row, ally_col)
                            if (
                                self.is_camp(ally_row, ally_col)
                                or ally_pawn == Tile.BLACK.value
                                or (
                                    ally_pawn == Tile.EMPTY.value
                                    and self.is_throne(ally_row, ally_col)
                                )
                            ):

                                self._previous = copy.deepcopy(self.board)
                                self.board[check_row][check_col] = Tile.EMPTY.value

                    # ENEMY KING
                    elif enemy_pawn == Tile.KING.value:
                        for row_step, col_step in [up, down, left, right]:
                            ally_row = check_row + row_step
                            ally_col = check_col + col_step
                            if (ally_row, ally_col) != (
                                row,
                                col,
                            ) and self.check_inside_board(ally_row, ally_col):
                                ally_pawn = self.at(ally_row, ally_col)
                                if (
                                    ally_pawn == Tile.EMPTY.value
                                    or ally_pawn == Tile.WHITE.value
                                ):
                                    break
                                # else:
                                #     caught_board[check_row][check_col] = Tile.EMPTY.value
        return

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

                if self.valid_move(row, col, moved_row, moved_col):
                    new_board = copy.deepcopy(self.board)
                    new_board[row][col] = Tile.EMPTY.value
                    new_board[moved_row][moved_col] = pawn

                    new_board_class = Board(new_board)
                    new_board_class.solve_captures(moved_row, moved_col)
                    moves.append(new_board_class)
                else:
                    # the path is blocked so cant move there, change direction
                    break

        return moves

    def check_inside_board(self, row, col):
        if 0 <= row < BOARD_LENGTH and 0 <= col < BOARD_LENGTH:
            return True
        else:
            return False

    def generate_all_moves(self, player: Player) -> list:
        moves = []
        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
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

    def king_escapes(self) -> int:

        up = (1, 0)
        down = (-1, 0)
        left = (0, -1)
        right = (0, 1)

        row, col = None, None
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self[i][j] == Tile.KING.value:
                    row, col = i, j
                    break
            if row is not None:
                break
        escapes = 0
        for dr, dc in [up, down, left, right]:
            r, c = row + dr, col + dc
            while self.check_inside_board(r, c):
                if self.is_escape(r, c):
                    if self.at(r, c) == Tile.EMPTY.value:
                        escapes += 1
                    break
                if (
                    self.at(r, c) != Tile.EMPTY.value
                    or self.is_camp(r, c)
                    or self.is_throne(r, c)
                ):
                    break
                r += dr
                c += dc

        return escapes

    def king_surr(self) -> int:
        up = (1, 0)
        down = (-1, 0)
        left = (0, -1)
        right = (0, 1)
        row, col = None, None
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self[i][j] == Tile.KING.value:
                    row, col = i, j
                    break
            if row is not None:
                break
        surr = 0

        for dr, dc in [up, down, left, right]:
            r, c = dr + row, dc + col
            if self.check_inside_board(r, c):
                if (
                    self.at(r, c) == Tile.BLACK.value
                    or self.is_camp(r, c)
                    or self.is_throne(r, c)
                ):
                    surr += 1
        return surr


class GameState:
    def __init__(self, board: Board, playing_as: Player, turn_player: Player, turn=0):
        self._board = board
        self._playing_as = playing_as
        self._turn_player = turn_player
        self._turn = turn

    def __str__(self) -> str:
        header = f"PLAYNG AS: {self._playing_as}\nTURN: {self._turn_player}\n"
        return f"{header}\n{self.board}"

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
        up = (1, 0)
        down = (-1, 0)
        left = (0, -1)
        right = (0, 1)

        board = Board(self._board.previous)

        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
                if board.at(row, col) == Tile.KING.value:
                    if board.is_escape(row, col):
                        return True
                    capture = 0
                    for rd, cd in [up, down, left, right]:
                        moved_row = row + rd
                        moved_col = col + cd
                        if (
                            board.at(moved_row, moved_col) == Tile.EMPTY.value
                            or board.at(moved_row, moved_col) == Tile.WHITE.value
                        ):
                            break
                        else:
                            capture += 1
                    if capture == 4:
                        return True

        return False
