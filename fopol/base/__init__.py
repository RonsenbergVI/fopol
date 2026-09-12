"""Foundations: abstract records, the typed collection, clients, and models.

Record types live in :mod:`fopol.data`, clients in :mod:`fopol.sources`, model
variants in :mod:`fopol.models`, and name resolution in :mod:`fopol.identity`.
"""

from fopol.base.data import Agg, Data, FixtureId, Grain, Position, Ref, Season
from fopol.base.dataset import Dataset
from fopol.base.model import (
    Inference,
    InvolvementModel,
    MinutesModel,
    Model,
    NotFittedError,
    PointsModel,
    TeamModel,
)
from fopol.base.source import Kind, Source, UnsupportedKind, cached_get, cached_get_many

__all__ = [
    "Agg",
    "Data",
    "Dataset",
    "FixtureId",
    "Grain",
    "Inference",
    "InvolvementModel",
    "Kind",
    "MinutesModel",
    "Model",
    "NotFittedError",
    "PointsModel",
    "Position",
    "Ref",
    "Season",
    "Source",
    "TeamModel",
    "UnsupportedKind",
    "cached_get",
    "cached_get_many",
]
