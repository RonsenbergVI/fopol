"""Concrete records: what a source returns and what a model consumes."""

from fopol.data.player import PlayerRef, PlayerSeason, PlayerStat
from fopol.data.prediction import (
    AppearancePrediction,
    FixturePrediction,
    InvolvementPrediction,
    PointsPrediction,
)
from fopol.data.result import Result, to_team_stats
from fopol.data.team import TeamRef, TeamSeason, TeamStat

__all__ = [
    "AppearancePrediction",
    "FixturePrediction",
    "InvolvementPrediction",
    "PointsPrediction",
    "PlayerRef",
    "PlayerSeason",
    "PlayerStat",
    "Result",
    "TeamRef",
    "TeamSeason",
    "TeamStat",
    "to_team_stats",
]
