"""Teams: the ref, the per-fixture stat, and the season it aggregates into."""

from __future__ import annotations

from typing import Annotated, ClassVar

import pandas as pd

from fopol.base.data import Agg, Data, FixtureId, Grain, Ref, Season
from fopol.identity import canonical_team_id

__all__ = ["TeamRef", "TeamSeason", "TeamStat"]


class TeamRef(Ref):
    """A club, canonical across seasons and sources."""

    @classmethod
    def from_name(cls, name: str) -> TeamRef:
        """Resolve any source's spelling: 'Spurs', 'Tottenham', 'Ipswich Town'."""
        return cls(id=canonical_team_id(name))


class TeamSeason(Data):
    """A team's season, derived from its per-fixture rows."""

    grain: ClassVar[Grain] = Grain.SEASON

    season: Annotated[Season, Agg.KEY]
    team: Annotated[TeamRef, Agg.KEY]
    games: Annotated[int, Agg.SUM]
    goals_for: Annotated[int, Agg.SUM]
    goals_against: Annotated[int, Agg.SUM]
    xg_for: Annotated[float | None, Agg.SUM] = None
    xg_against: Annotated[float | None, Agg.SUM] = None

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


class TeamStat(Data):
    """One team's involvement in one fixture. The long view, built for rollup."""

    grain: ClassVar[Grain] = Grain.FIXTURE
    aggregates_to: ClassVar[type[Data] | None] = TeamSeason

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    team: Annotated[TeamRef, Agg.KEY]
    opponent: Annotated[TeamRef, Agg.DROP]
    is_home: Annotated[bool, Agg.DROP]
    goals_for: Annotated[int, Agg.SUM]
    goals_against: Annotated[int, Agg.SUM]
    kickoff: Annotated[pd.Timestamp | None, Agg.DROP] = None
    gameweek: Annotated[int | None, Agg.DROP] = None
    xg_for: Annotated[float | None, Agg.SUM] = None
    xg_against: Annotated[float | None, Agg.SUM] = None
