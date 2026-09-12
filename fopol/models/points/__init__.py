"""Points models. All satisfy :class:`PointsModel`; pit them freely."""

from fopol.base.model import PointsModel
from fopol.models.points.simulation import SimulatedPointsModel

__all__ = ["PointsModel", "SimulatedPointsModel"]
