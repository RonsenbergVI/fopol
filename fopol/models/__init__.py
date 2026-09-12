"""Bayesian models over FPL, one family per role.

Each family has an abstract base that pins the record types, and one or more
variants beneath it. Anything satisfying the base is interchangeable with any
other variant of the same family, which is what makes comparing them meaningful.
"""

from fopol.models.involvement import EmpiricalBayesInvolvement, InvolvementModel
from fopol.models.minutes import MinutesModel, SequentialMinutesModel
from fopol.models.points import PointsModel, SimulatedPointsModel
from fopol.models.team import DixonColesTeamModel, TeamModel

__all__ = [
    "DixonColesTeamModel",
    "EmpiricalBayesInvolvement",
    "InvolvementModel",
    "MinutesModel",
    "PointsModel",
    "SequentialMinutesModel",
    "SimulatedPointsModel",
    "TeamModel",
]
