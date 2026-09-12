"""Empirical-Bayes involvement: Gamma-Poisson shrinkage toward the position rate.

A player with few minutes is pulled toward his position's average rather than
credited with the rate implied by one lucky cameo. Rates are estimated per
player from his own history and are **not** conditioned on his club, which is
the known flaw: a striker who moves carries his old side's volume with him.
"""

from __future__ import annotations

import numpy as np

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, InvolvementModel
from fopol.data.player import PlayerStat

__all__ = ["EmpiricalBayesInvolvement"]


class EmpiricalBayesInvolvement(InvolvementModel):
    """Gamma-Poisson shrinkage of per-90 rates toward the position mean.

    Args:
        prior_games: Strength of the shrinkage prior, in 90-minute games.
            Higher pools harder toward the position mean.
        inference: Only ``seed`` is used.
    """

    def __init__(self, *, prior_games: float = 8.0, inference: Inference | None = None) -> None:
        self.prior_games = prior_games
        self.inference = inference or Inference()

    def _fit(self, data: Dataset[PlayerStat]) -> None:
        totals: dict[str, dict] = {}
        by_pos: dict[str, dict] = {}
        for r in data:
            if not r.played:
                continue
            t = totals.setdefault(str(r.player), {"m": 0, "g": 0, "a": 0, "pos": []})
            t["m"] += r.minutes
            t["g"] += r.goals
            t["a"] += r.assists
            t["pos"].append(r.position.value)
            p = by_pos.setdefault(r.position.value, {"m": 0, "g": 0, "a": 0})
            p["m"] += r.minutes
            p["g"] += r.goals
            p["a"] += r.assists
        if not totals:
            raise ValueError("no played appearances to fit on")

        self.position_rates_ = {
            pos: (v["g"] / max(v["m"] / 90.0, 1e-9), v["a"] / max(v["m"] / 90.0, 1e-9))
            for pos, v in by_pos.items()
        }
        k = self.prior_games
        self.rates_ = {}
        for player, v in totals.items():
            pos = max(set(v["pos"]), key=v["pos"].count)
            pg, pa = self.position_rates_[pos]
            games = v["m"] / 90.0
            self.rates_[player] = (
                (v["g"] + pg * k) / (games + k),
                (v["a"] + pa * k) / (games + k),
            )
        self.posterior_: dict[str, np.ndarray] = {}  # nothing sampled; the estimator is closed-form

    def rates(self, data):
        g = np.empty(len(data))
        a = np.empty(len(data))
        for i, r in enumerate(data):
            g[i], a[i] = self.rates_.get(
                str(r.player), self.position_rates_.get(r.position.value, (0.0, 0.0))
            )
        return g, a
