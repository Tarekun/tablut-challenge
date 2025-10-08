from enum import Enum


class Player(Enum):
    WHITE = "W"
    BLACK = "B"

    def is_white(self):
        return self.value == "W"

    def is_black(self):
        return self.value == "B"


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
