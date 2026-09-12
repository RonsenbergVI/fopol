"""Result: one fixture's scoreline, both teams together."""

from __future__ import annotations

from typing import Annotated, ClassVar

import pandas as pd

from fopol.base.data import Agg, Data, FixtureId, Grain, Season
from fopol.base.dataset import Dataset
from fopol.data.team import TeamRef, TeamStat

__all__ = ["Result", "to_team_stats"]


class Result(Data):
    """One fixture's scoreline, both teams together.

    Deliberately joint rather than one row per team. The scoreline likelihood is
    joint: the Dixon-Coles correction needs both teams' goals in one object, and
    the correlation between them is the thing being modelled. Splitting a result
    into two independent rows discards exactly that.

    :class:`~fopol.data.team.TeamStat` is the long view, derived from this.
    ``gameweek`` is an FPL concept and optional: a results feed has no such thing.
    ``kickoff`` is optional too, because a fixture can exist before it is scheduled.
    """

    grain: ClassVar[Grain] = Grain.FIXTURE

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    home: Annotated[TeamRef, Agg.FIRST]
    away: Annotated[TeamRef, Agg.FIRST]
    kickoff: Annotated[pd.Timestamp | None, Agg.FIRST] = None
    gameweek: Annotated[int | None, Agg.FIRST] = None
    home_goals: Annotated[int | None, Agg.SUM] = None
    away_goals: Annotated[int | None, Agg.SUM] = None
    home_xg: Annotated[float | None, Agg.SUM] = None
    away_xg: Annotated[float | None, Agg.SUM] = None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    def to_team_stats(self) -> tuple[TeamStat, TeamStat]:
        """Split into the two per-team rows. Unplayed fixtures cannot be split."""
        home_goals, away_goals = self.home_goals, self.away_goals
        if home_goals is None or away_goals is None:
            raise ValueError(f"{self.fixture_id}: cannot split an unplayed fixture")
        return (
            TeamStat(
                season=self.season,
                fixture_id=self.fixture_id,
                gameweek=self.gameweek,
                kickoff=self.kickoff,
                team=self.home,
                opponent=self.away,
                is_home=True,
                goals_for=home_goals,
                goals_against=away_goals,
                xg_for=self.home_xg,
                xg_against=self.away_xg,
            ),
            TeamStat(
                season=self.season,
                fixture_id=self.fixture_id,
                gameweek=self.gameweek,
                kickoff=self.kickoff,
                team=self.away,
                opponent=self.home,
                is_home=False,
                goals_for=away_goals,
                goals_against=home_goals,
                xg_for=self.away_xg,
                xg_against=self.home_xg,
            ),
        )


def to_team_stats(results: Dataset[Result]) -> Dataset[TeamStat]:
    """The long, per-team view of a set of results. Skips unplayed fixtures."""
    rows = [row for r in results if r.played for row in r.to_team_stats()]
    return Dataset(rows, record_type=TeamStat, source=results.source)
