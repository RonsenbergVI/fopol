"""Players: the ref, the per-fixture stat, and the season it aggregates into."""

from __future__ import annotations

from typing import Annotated, ClassVar

import pandas as pd

from fopol.base.data import Agg, Data, FixtureId, Grain, Position, Ref, Season
from fopol.data.team import TeamRef
from fopol.identity import canonical_player_id

__all__ = ["PlayerRef", "PlayerSeason", "PlayerStat"]


class PlayerRef(Ref):
    """A player, canonical across seasons, clubs and sources."""

    @classmethod
    def from_name(cls, name: str) -> PlayerRef:
        return cls(id=canonical_player_id(name))


class PlayerSeason(Data):
    """A player's season, derived from appearances.

    Rates are properties, never stored fields. A per-90 figure is a ratio of two
    sums; storing it invites someone to average it, which is wrong whenever
    minutes vary between games -- that is, always.
    """

    grain: ClassVar[Grain] = Grain.SEASON

    season: Annotated[Season, Agg.KEY]
    player: Annotated[PlayerRef, Agg.KEY]
    player_name: Annotated[str, Agg.FIRST]
    team: Annotated[TeamRef, Agg.LAST]
    position: Annotated[Position, Agg.LAST]
    games: Annotated[int, Agg.SUM]
    minutes: Annotated[int, Agg.SUM]
    started: Annotated[int, Agg.SUM] = 0
    goals: Annotated[int, Agg.SUM] = 0
    assists: Annotated[int, Agg.SUM] = 0
    xg: Annotated[float | None, Agg.SUM] = None
    xa: Annotated[float | None, Agg.SUM] = None
    shots: Annotated[int | None, Agg.SUM] = None
    key_passes: Annotated[int | None, Agg.SUM] = None
    goals_conceded: Annotated[int, Agg.SUM] = 0
    clean_sheet: Annotated[int, Agg.SUM] = 0
    saves: Annotated[int, Agg.SUM] = 0
    defensive_contribution: Annotated[int, Agg.SUM] = 0
    bps: Annotated[int, Agg.SUM] = 0
    bonus: Annotated[int, Agg.SUM] = 0
    total_points: Annotated[int, Agg.SUM] = 0
    price: Annotated[float | None, Agg.LAST] = None
    ownership: Annotated[float | None, Agg.LAST] = None

    def per90(self, field: str) -> float:
        """Total over exposure -- the only correct way to form a rate."""
        value = getattr(self, field)
        if self.minutes <= 0 or value is None:
            return 0.0
        return float(value) / (self.minutes / 90.0)

    @property
    def xg90(self) -> float:
        return self.per90("xg")

    @property
    def xa90(self) -> float:
        return self.per90("xa")


class PlayerStat(Data):
    """One player's involvement in one fixture.

    Optional fields are genuinely optional: not every source carries xG, and a
    source that lacks it should say nothing rather than invent a zero.
    """

    grain: ClassVar[Grain] = Grain.FIXTURE
    aggregates_to: ClassVar[type[Data] | None] = PlayerSeason

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    player: Annotated[PlayerRef, Agg.KEY]
    player_name: Annotated[str, Agg.FIRST]
    team: Annotated[TeamRef, Agg.LAST]
    opponent: Annotated[TeamRef, Agg.DROP]
    position: Annotated[Position, Agg.LAST]
    is_home: Annotated[bool, Agg.DROP]
    kickoff: Annotated[pd.Timestamp | None, Agg.DROP] = None
    gameweek: Annotated[int | None, Agg.DROP] = None
    minutes: Annotated[int | None, Agg.SUM] = None
    """None until the fixture is played -- that is what a prediction row is."""
    started: Annotated[int, Agg.SUM] = 0
    goals: Annotated[int, Agg.SUM] = 0
    assists: Annotated[int, Agg.SUM] = 0
    xg: Annotated[float | None, Agg.SUM] = None
    xa: Annotated[float | None, Agg.SUM] = None
    shots: Annotated[int | None, Agg.SUM] = None
    key_passes: Annotated[int | None, Agg.SUM] = None
    goals_conceded: Annotated[int, Agg.SUM] = 0
    clean_sheet: Annotated[int, Agg.SUM] = 0
    saves: Annotated[int, Agg.SUM] = 0
    defensive_contribution: Annotated[int, Agg.SUM] = 0
    bps: Annotated[int, Agg.SUM] = 0
    bonus: Annotated[int, Agg.SUM] = 0
    yellow_cards: Annotated[int, Agg.SUM] = 0
    red_cards: Annotated[int, Agg.SUM] = 0
    own_goals: Annotated[int, Agg.SUM] = 0
    total_points: Annotated[int, Agg.SUM] = 0
    price: Annotated[float | None, Agg.LAST] = None
    ownership: Annotated[float | None, Agg.LAST] = None

    @property
    def played(self) -> bool:
        return self.minutes is not None

    @property
    def band(self) -> int | None:
        """FPL appearance band: 0 = none, 1 = 1-59 minutes, 2 = 60+."""
        if self.minutes is None:
            return None
        return 0 if self.minutes == 0 else (1 if self.minutes < 60 else 2)
