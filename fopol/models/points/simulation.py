"""Simulated points: compose team, minutes and involvement into a points posterior.

For each draw: sample a band from the minutes model, goals and assists from
the involvement rates scaled by the team model's expected goals for this
fixture and by time on the pitch, goals conceded from the opponent's rate,
then apply the FPL scoring rules. Rates are coherent across layers;
realisations are sampled independently, which gets every marginal right and
understates co-movement between teammates -- the spread of a squad stacked on
one club will read a little safer than it is.

Scoring implemented: appearance, goals, assists, clean sheets, goals conceded.
Not implemented: saves, bonus, defensive contribution, cards, penalties -- so
keepers and defenders are undervalued, visibly: a keeper's ``p_haul`` is zero.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, InvolvementModel, MinutesModel, PointsModel, TeamModel
from fopol.data.player import PlayerStat

__all__ = ["SimulatedPointsModel"]

GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
LEAGUE_AVERAGE_GOALS = 1.45


class SimulatedPointsModel(PointsModel):
    """Monte Carlo composition of three fitted sub-models.

    The sub-models are hyperparameters in the sklearn sense -- a meta-estimator
    -- and must be fitted before this model's ``fit``, which learns nothing
    itself. Swap any of them for another variant of the same family.
    """

    consumes: ClassVar[type[PlayerStat]] = PlayerStat

    def __init__(
        self,
        *,
        team: TeamModel,
        minutes: MinutesModel,
        involvement: InvolvementModel,
        n_draws: int = 4000,
        inference: Inference | None = None,
    ) -> None:
        self.team = team
        self.minutes = minutes
        self.involvement = involvement
        self.n_draws = n_draws
        self.inference = inference or Inference()

    def _fit(self, data: Dataset[PlayerStat]) -> None:
        for name in ("team", "minutes", "involvement"):
            getattr(self, name).check_is_fitted()
        self.posterior_: dict[str, np.ndarray] = {}
        self.fitted_on_ = len(data)

    # ------------------------------------------------------------ pieces

    def _team_rates(self, rows: list[PlayerStat]) -> tuple[np.ndarray, np.ndarray]:
        home = [r.team if r.is_home else r.opponent for r in rows]
        away = [r.opponent if r.is_home else r.team for r in rows]
        lam_h, lam_a = self.team.goal_rate_draws(home, away)
        is_home = np.array([r.is_home for r in rows])
        own = np.where(is_home, lam_h.mean(0), lam_a.mean(0))
        against = np.where(is_home, lam_a.mean(0), lam_h.mean(0))
        return own, against

    def p_60(self, data):
        return self.minutes.band_probs(self.minutes.features_for(data))[:, 2]

    def points_draws(self, data):
        rows = list(data)
        rng = np.random.default_rng(self.inference.seed)
        n, D = len(rows), self.n_draws

        band_p = self.minutes.band_probs(self.minutes.features_for(data))
        band = (rng.random((D, n))[:, :, None] > np.cumsum(band_p, 1)[None]).sum(2)
        played, full = band > 0, band == 2
        exposure = np.where(full, 1.0, np.where(played, 0.4, 0.0))

        team_for, team_against = self._team_rates(rows)
        g90, a90 = self.involvement.rates(data)
        scale = team_for / LEAGUE_AVERAGE_GOALS  # known flaw: rates already embed a club
        goals = rng.poisson(np.clip(g90 * scale, 0, None)[None] * exposure)
        assists = rng.poisson(np.clip(a90 * scale, 0, None)[None] * exposure)
        conceded = rng.poisson(np.broadcast_to(team_against[None], (D, n)))
        clean = full & (conceded == 0)

        pos = [r.position.value for r in rows]
        gv = np.array([GOAL_POINTS[p] for p in pos])
        cv = np.array([CLEAN_SHEET_POINTS[p] for p in pos])
        docked = np.array([p in ("GK", "DEF") for p in pos])
        return (
            np.where(full, 2, np.where(played, 1, 0))
            + goals * gv
            + assists * ASSIST_POINTS
            + clean * cv
            - np.where(full & docked[None], conceded // 2, 0)
        ).astype(np.float32)
