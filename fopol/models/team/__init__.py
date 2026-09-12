"""Team-level goal models. All satisfy :class:`TeamModel`; pit them freely."""

from fopol.base.model import TeamModel
from fopol.models.team.dixon_coles import DixonColesTeamModel

__all__ = ["DixonColesTeamModel", "TeamModel"]
