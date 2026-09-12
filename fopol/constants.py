from enum import StrEnum


class Position(StrEnum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class Grain(StrEnum):
    """What one record counts as."""

    FIXTURE = "fixture"
    SEASON = "season"


# FPL scoring by position.
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3

# Goalkeepers and defenders lose a point per two goals conceded.
CONCEDE_PENALTY_POSITIONS = ("GK", "DEF")
