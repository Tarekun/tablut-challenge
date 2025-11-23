from enum import Enum
import copy
import numpy as np
from typing import Tuple, Union


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

    def complement(self):
        if self == Turn.WHITE:
            return Turn.BLACK
        # TODO: should check this isnt a final state and raise?
        else:
            return Turn.WHITE


class Tile(Enum):
    EMPTY = "EMPTY"
    BLACK = "BLACK"
    WHITE = "WHITE"
    KING = "KING"
    # THRONE = "THRONE"

    @staticmethod
    def is_black(tile: str):
        return tile == Tile.BLACK.value

    @staticmethod
    def is_white(tile: str):
        return tile == Tile.WHITE.value or tile == Tile.KING.value


class Board:
    def __init__(self, board: list[list[str]]):
        self.board: list[list[str]] = board
        self._previous: Union[list[list[str]], None] = None

    @property
    def previous(self):
        if self._previous is not None:
            return self._previous
        else:
            return self.board

    def __eq__(self, other):
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self[i][j] != other[i][j]:
                    return False
        return True

    def __str__(self) -> str:
        string = ""
        for row in range(BOARD_LENGTH):
            for col in range(BOARD_LENGTH):
                value = ""
                if self.board[row][col] == Tile.EMPTY.value:
                    if self.is_camp(row, col):
                        value = "▫"
                    elif self.is_escape(row, col):
                        value = "█"
                    elif self.is_throne(row, col):
                        value = "▪"
                    else:
                        value = "░"
                elif self.board[row][col] == Tile.KING.value:
                    value = "♣"
                elif self.board[row][col] == Tile.WHITE.value:
                    value = "♠"
                elif self.board[row][col] == Tile.BLACK.value:
                    value = "♤"
                else:
                    value = self.board[row][col][0:1]
                string += value
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

    def at(self, row: int, col: int) -> str:
        """Returns the tile at [row,col]"""
        return self.board[row][col]

    def __getitem__(self, row: int) -> list[str]:
        """Enables board[row] syntax, returns the row"""
        return self.board[row]

    def king_position(self) -> tuple[Union[int, None], Union[int, None]]:
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if self[i][j] == Tile.KING.value:
                    return i, j

        return None, None

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

    def camp_id(self, row: int, col: int) -> Union[str, None]:
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
                or not self.is_camp(to_row, to_col)
            )
            and (
                # the move does NOT pass through (4,4)
                (from_row == 4 and from_col == 4)
                or (
                    not (
                        (
                            # vertical movement crossing 4,4
                            from_col == 4
                            and min(from_row, to_row) <= 4 <= max(from_row, to_row)
                        )
                        or (
                            # horizontal movement crossing 4,4
                            from_row == 4
                            and min(from_col, to_col) <= 4 <= max(from_col, to_col)
                        )
                    )
                )
            )
        )

    def solve_captures(self, row, col):
        """Updates the current board if captures can be made, by removing from the board
        the captured pawn. `row` and `col` refer to the tile the last pawn moved to"""

        def _generic_capture(
            moved_pawn: str,
            enemy_pawn: str,
            ally_row: int,
            ally_col: int,
            enemy_row: int,
            enemy_col: int,
        ) -> bool:
            ally_pawn: str = self[ally_row][ally_col]
            valid_ally = (
                (
                    (Tile.is_black(moved_pawn) and Tile.is_black(ally_pawn))
                    or (Tile.is_white(moved_pawn) and Tile.is_white(ally_pawn))
                )
                or (
                    self.is_throne(ally_row, ally_col)
                    and self.is_empty(ally_row, ally_col)
                )
                or (
                    self.is_camp(ally_row, ally_col)
                    and self.is_empty(ally_row, ally_col)
                )
            )
            valid_enemy = (
                (moved_pawn == Tile.WHITE.value or moved_pawn == Tile.KING.value)
                and enemy_pawn == Tile.BLACK.value
            ) or (
                moved_pawn == Tile.BLACK.value
                and (enemy_pawn == Tile.WHITE.value or enemy_pawn == Tile.KING.value)
            )
            return valid_ally and valid_enemy

        up = (1, 0)
        down = (-1, 0)
        left = (0, -1)
        right = (0, 1)
        pawn: str = self.at(row, col)
        for rd, cd in [up, down, left, right]:
            enemy_row = row + rd
            enemy_col = col + cd
            ally_row = enemy_row + rd
            ally_col = enemy_col + cd
            # proceed only if both are inside (captured is not on the side of the board)
            if self.check_inside_board(
                enemy_row, enemy_col
            ) and self.check_inside_board(ally_row, ally_col):
                enemy_pawn = self.at(enemy_row, enemy_col)

                # if king it needs to be cornered from all directions
                if pawn == Tile.BLACK.value and enemy_pawn == Tile.KING.value:
                    fully_cornered = True
                    for row_step, col_step in [up, down, left, right]:
                        ally_row = enemy_row + row_step
                        ally_col = enemy_col + col_step
                        if self.check_inside_board(
                            ally_row, ally_col
                        ) and not _generic_capture(
                            pawn, enemy_pawn, ally_row, ally_col, enemy_row, enemy_col
                        ):
                            fully_cornered = False

                    if fully_cornered:
                        self._previous = copy.deepcopy(self.board)
                        self.board[enemy_row][enemy_col] = Tile.EMPTY.value
                # otherwise run generic capture check
                elif _generic_capture(
                    pawn, enemy_pawn, ally_row, ally_col, enemy_row, enemy_col
                ):
                    self._previous = copy.deepcopy(self.board)
                    self.board[enemy_row][enemy_col] = Tile.EMPTY.value

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

        row, col = self.king_position()
        if row is None or col is None:
            # king gone, no escapes available
            return 0

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

        row, col = self.king_position()
        if row is None or col is None:
            # king gone, no escapes available
            return 0
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
    def __init__(
        self, board: Board, playing_as: Player, turn: Turn, previous=None, turn_num=0
    ):
        self._board = board
        self._playing_as = playing_as
        self._turn_player = Player.WHITE if turn.plays(Player.WHITE) else Player.BLACK
        self._turn = turn
        self._turn_num = turn_num
        self._previous = previous

    @classmethod
    def clone_state_from_board(
        cls, parent_state: "GameState", board: Board
    ) -> "GameState":
        return cls(
            board=board,
            playing_as=parent_state.playing_as,
            turn=parent_state.turn,
            turn_num=parent_state.turn_num,
            previous=parent_state.previous,
        )

    def __str__(self) -> str:
        header = f"TURN: {self._turn_player}"
        return f"{header}\n{self.board}"

    def __eq__(self, other):
        return self.turn_player == other.turn_player and self.board == other.board

    @property
    def previous(self):
        return self._previous

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
    def turn(self) -> Turn:
        return self._turn

    @property
    def turn_num(self) -> int:
        return self._turn_num

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
                                self.turn.complement(),
                                turn_num=self.turn_num + 1,
                                previous=self,
                            )
                            for move in pawn_moves
                        ]
                    )

        return moves

    def winner(self) -> Union[Player, None, str]:
        previous = self.previous
        for _ in range(3):
            if previous is None:
                break
            previous = previous.previous

        if previous and self.board.board == previous.board.board:
            return "DRAW"
        row, col = self.board.king_position()
        if row is None or col is None:
            # king was captured
            return Player.BLACK
        elif self.board.is_escape(row, col):
            # king escaped
            return Player.WHITE
        elif self.board.king_surr() == 4:
            # king is cornered
            return Player.BLACK
        elif self.next_moves() == []:
            # current player has no moves
            return self.turn_player.complement()

    def is_end_state(self) -> bool:
        return self.winner() is not None
