"""Involvement family: shrinkage behaves, unknown players fall back, score is a density."""

import pandas as pd
import pytest

from fopol.base.dataset import Dataset
from fopol.data.player import PlayerRef, PlayerStat
from fopol.data.team import TeamRef
from fopol.models.involvement import EmpiricalBayesInvolvement


def test_rates_reflect_output(involvement_history):
    """A prolific scorer's fitted rate sits above a blank one's."""
    m = EmpiricalBayesInvolvement().fit(involvement_history)
    g, a = m.rates(
        involvement_history.filter(
            lambda r: (
                r.player.id in ("prolific", "average", "playmaker") and r.fixture_id.endswith("-0")
            )
        )
    )
    assert g[0] > 0.7 and g[1] < 0.2 and a[2] > 0.7


def test_small_samples_are_shrunk(involvement_history):
    """One goal in ten minutes implies 9.0 per 90. It must not be believed."""
    m = EmpiricalBayesInvolvement().fit(involvement_history)
    (g,), _ = m.rates(involvement_history.filter(lambda r: r.player.id == "lucky-sub"))
    assert g < 9.0 / 5


def test_unknown_player_falls_back_to_position(involvement_history):
    """A player absent from training gets the position rate."""
    m = EmpiricalBayesInvolvement().fit(involvement_history)
    row = PlayerStat(
        season="2024-25",
        fixture_id="x",
        player=PlayerRef(id="nobody"),
        player_name="n",
        team=TeamRef(id="a"),
        opponent=TeamRef(id="b"),
        position="FWD",
        is_home=True,
        kickoff=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    (g,), _ = m.rates(Dataset([row], record_type=PlayerStat))
    assert g == pytest.approx(m.position_rates_["FWD"][0])


def test_stronger_prior_shrinks_harder(involvement_history):
    """Raising ``prior_games`` pulls rates closer to the position mean."""
    weak = EmpiricalBayesInvolvement(prior_games=1.0).fit(involvement_history)
    strong = EmpiricalBayesInvolvement(prior_games=50.0).fit(involvement_history)
    assert strong.rates_["prolific"][0] < weak.rates_["prolific"][0]


def test_score_is_finite_log_density(involvement_history):
    """``score`` is a finite mean log density on played rows."""
    m = EmpiricalBayesInvolvement().fit(involvement_history)
    s = m.score(involvement_history)
    assert s < 0 and s > -10
