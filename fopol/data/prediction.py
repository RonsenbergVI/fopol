"""Prediction records: what a model says about a fixture it has not seen.

Every model in a family produces the same prediction record, which is what
makes two variants comparable -- score them on the same held-out records, read
the same fields. A prediction is a *summary* of a posterior predictive: a mean,
a spread, and the probabilities that decisions actually turn on. Raw draws are
available from the model when needed; they are not records.

Keys mirror the input record's keys so a prediction can always be joined back
to the fixture it describes.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from fopol.base.data import Agg, Data, FixtureId, Grain, Season
from fopol.data.player import PlayerRef
from fopol.data.team import TeamRef

__all__ = ["AppearancePrediction", "FixturePrediction", "InvolvementPrediction", "PointsPrediction"]


class FixturePrediction(Data):
    """A team model's view of one fixture."""

    grain: ClassVar[Grain] = Grain.FIXTURE

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    home: Annotated[TeamRef, Agg.FIRST]
    away: Annotated[TeamRef, Agg.FIRST]
    home_xg: Annotated[float, Agg.SUM]
    away_xg: Annotated[float, Agg.SUM]
    p_home: Annotated[float, Agg.MEAN]
    p_draw: Annotated[float, Agg.MEAN]
    p_away: Annotated[float, Agg.MEAN]
    p_home_clean_sheet: Annotated[float, Agg.MEAN]
    p_away_clean_sheet: Annotated[float, Agg.MEAN]


class AppearancePrediction(Data):
    """A minutes model's view of one player in one fixture."""

    grain: ClassVar[Grain] = Grain.FIXTURE

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    player: Annotated[PlayerRef, Agg.KEY]
    p_0: Annotated[float, Agg.MEAN]
    p_1_59: Annotated[float, Agg.MEAN]
    p_60: Annotated[float, Agg.MEAN]

    @property
    def expected_appearance_points(self) -> float:
        return self.p_1_59 + 2.0 * self.p_60


class PointsPrediction(Data):
    """A points model's view of one player in one fixture.

    ``mean`` is the one number decisions reduce to. ``sd`` and ``p_haul`` are
    why it is worth keeping a posterior: a differential's value is in its tail.
    """

    grain: ClassVar[Grain] = Grain.FIXTURE

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    player: Annotated[PlayerRef, Agg.KEY]
    mean: Annotated[float, Agg.SUM]
    sd: Annotated[float, Agg.MEAN]
    p_blank: Annotated[float, Agg.MEAN]
    p_haul: Annotated[float, Agg.MEAN]
    p_60: Annotated[float, Agg.MEAN]

    def value(self, ownership: float, kappa: float = 0.0) -> float:
        """Decision scalar: mean plus a risk premium on unowned upside."""
        return self.mean + kappa * (1.0 - ownership) * self.sd


class InvolvementPrediction(Data):
    """An involvement model's view of one player in one fixture: rates per 90."""

    grain: ClassVar[Grain] = Grain.FIXTURE

    season: Annotated[Season, Agg.KEY]
    fixture_id: Annotated[FixtureId, Agg.KEY]
    player: Annotated[PlayerRef, Agg.KEY]
    goals_p90: Annotated[float, Agg.MEAN]
    assists_p90: Annotated[float, Agg.MEAN]
