from enum import Enum


class Player(Enum):
    WHITE = "W"
    BLACK = "B"

    def is_white(self) -> bool:
        return self.value == "W"

    def is_black(self) -> bool:
        return self.value == "B"

    def complement(self):
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK


class Turn(Enum):
    WHITE = "W"
    BLACK = "B"
    WHITE_WINS = "WW"
    BLACK_WINS = "BW"
    DRAW = "D"

    def plays(self, player: Player) -> bool:
        """Returns True if the given player is playing this turn, False otherwise"""
        return self.value == player.value

    def game_finished(self):
        return self == Turn.WHITE_WINS or self == Turn.BLACK_WINS or self == Turn.DRAW
