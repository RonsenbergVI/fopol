"""Share involvement: a player's rate is his share of his club's goals.

The empirical-Bayes variant estimates a player's goals per 90 from his own
history and then the points composition scales that by his club's expected
goals -- so a club's strength is counted twice, once inside the rate and once
outside it. This variant estimates what is actually stable about a player:
the **share** of his club's goals (and assists) that are his while he is on the
pitch. The rate handed to the composition is that share times the league
average, so multiplying by the club's expected goals gives share x club goals,
counted once.

The share is the thing you want when looking for players who carry a weak
side: a striker taking 40% of a bottom-half club's goals has a bigger share
than most players at a top club, and this is where that shows.

Set-piece duty enters as a prior. A first-choice penalty taker's prior share is
raised by roughly what penalties are worth (a tenth of a club's goals), and a
first-choice corner taker's assist share likewise; history overrides the prior
at the usual rate. That is how a player who has just inherited the penalties
is credited before he has taken one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, InvolvementModel
from fopol.constants import LEAGUE_AVERAGE_GOALS
from fopol.data.player import PlayerStat

__all__ = ["ShareInvolvement"]


class ShareInvolvement(InvolvementModel):
    """Shrunken share of club goals and assists, with set-piece duty as a prior.

    Args:
        prior_goals: Strength of the shrinkage toward the position share, in
            club goals witnessed. A player is pulled to his position's share
            until his club has scored about this many goals with him on the pitch.
        penalty_share: Added to the prior goal share of a first-choice penalty
            taker. Penalties are about a tenth of goals; a taker converts most.
        freekick_share: Added to the prior goal share of a first-choice
            direct free-kick taker.
        corner_share: Added to the prior assist share of a first-choice
            corner and indirect free-kick taker.
        inference: Only ``seed`` is used.
    """

    def __init__(
        self,
        *,
        prior_goals: float = 12.0,
        penalty_share: float = 0.08,
        freekick_share: float = 0.02,
        corner_share: float = 0.06,
        inference: Inference | None = None,
    ) -> None:
        self.prior_goals = prior_goals
        self.penalty_share = penalty_share
        self.freekick_share = freekick_share
        self.corner_share = corner_share
        self.inference = inference or Inference()

    def _fit(self, data: Dataset[PlayerStat]) -> None:
        played = [r for r in data if r.minutes]
        if not played:
            raise ValueError("no played appearances to fit on")

        goals_in: dict[tuple[str, str], float] = defaultdict(float)
        assists_in: dict[tuple[str, str], float] = defaultdict(float)
        for r in played:
            goals_in[(r.fixture_id, str(r.team))] += r.goals
            assists_in[(r.fixture_id, str(r.team))] += r.assists
        by_fixture_club = {key: (goals_in[key], assists_in[key]) for key in goals_in}

        totals: dict[str, dict] = {}
        by_pos: dict[str, dict] = {}
        latest_orders: dict[str, tuple[int | None, int | None, int | None]] = {}
        for r in played:
            g_club, a_club = by_fixture_club[(r.fixture_id, str(r.team))]
            exposure = (r.minutes or 0) / 90.0
            t = totals.setdefault(
                str(r.player), {"g": 0.0, "a": 0.0, "eg": 0.0, "ea": 0.0, "pos": []}
            )
            t["g"] += r.goals
            t["a"] += r.assists
            t["eg"] += g_club * exposure
            t["ea"] += a_club * exposure
            t["pos"].append(r.position.value)
            p = by_pos.setdefault(r.position.value, {"g": 0.0, "a": 0.0, "eg": 0.0, "ea": 0.0})
            p["g"] += r.goals
            p["a"] += r.assists
            p["eg"] += g_club * exposure
            p["ea"] += a_club * exposure
            latest_orders[str(r.player)] = (
                r.penalties_order,
                r.direct_freekicks_order,
                r.corners_order,
            )

        self.position_shares_ = {
            pos: (v["g"] / max(v["eg"], 1e-9), v["a"] / max(v["ea"], 1e-9))
            for pos, v in by_pos.items()
        }
        self.shares_ = {}
        self.exposure_ = {}
        for player, v in totals.items():
            pos = max(set(v["pos"]), key=v["pos"].count)
            prior_g, prior_a = self._prior(pos, latest_orders[player])
            k = self.prior_goals
            self.shares_[player] = (
                (v["g"] + prior_g * k) / (v["eg"] + k),
                (v["a"] + prior_a * k) / (v["ea"] + k),
            )
            self.exposure_[player] = v["eg"]
        self.posterior_: dict[str, np.ndarray] = {}

    def _prior(
        self, position: str, orders: tuple[int | None, int | None, int | None]
    ) -> tuple[float, float]:
        """Position share, lifted by the set-piece duties a player currently holds."""
        prior_g, prior_a = self.position_shares_.get(position, (0.0, 0.0))
        pens, freekicks, corners = orders
        if pens == 1:
            prior_g += self.penalty_share
        if freekicks == 1:
            prior_g += self.freekick_share
        if corners == 1:
            prior_a += self.corner_share
        return prior_g, prior_a

    def share(self, data: Dataset[PlayerStat]) -> tuple[np.ndarray, np.ndarray]:
        """``(goal share, assist share)`` per row: the fraction of club output that is the player's.

        A player absent from training gets his position's share, lifted by the
        set-piece duties the row says he holds.
        """
        self.check_is_fitted()
        g = np.empty(len(data))
        a = np.empty(len(data))
        for i, r in enumerate(data):
            known = self.shares_.get(str(r.player))
            if known is None:
                known = self._prior(
                    r.position.value,
                    (r.penalties_order, r.direct_freekicks_order, r.corners_order),
                )
            g[i], a[i] = known
        return g, a

    def rates(self, data: Dataset[PlayerStat]) -> tuple[np.ndarray, np.ndarray]:
        """Per-90 rates at an average club: share times the league-average goals."""
        g, a = self.share(data)
        return g * LEAGUE_AVERAGE_GOALS, a * LEAGUE_AVERAGE_GOALS
