"""Share involvement: rates are shares of club output, and set-piece duty is a prior."""

import numpy as np
import pytest

from fopol.base.dataset import Dataset
from fopol.constants import LEAGUE_AVERAGE_GOALS
from fopol.data.player import PlayerStat
from fopol.models.involvement import EmpiricalBayesInvolvement, ShareInvolvement


def test_share_is_fraction_of_club_goals(share_history):
    """The talisman of a low-scoring club takes most of its goals.

    His share says so where his raw per-90 rate would not.
    """
    m = ShareInvolvement(prior_goals=1.0).fit(share_history)
    rows = share_history.filter(lambda r: r.fixture_id.endswith("-0"))
    g, _ = m.share(rows)
    by = dict(zip([str(r.player) for r in rows], g, strict=True))
    assert by["talisman"] > 0.5
    assert by["talisman"] > by["big-club-striker"]


def test_rates_are_quoted_at_an_average_club(share_history):
    """``rates`` is share x league average.

    The composition then scales by the club, so the club is counted once.
    """
    m = ShareInvolvement().fit(share_history)
    rows = share_history.filter(lambda r: r.fixture_id.endswith("-0"))
    share_g, _ = m.share(rows)
    rate_g, _ = m.rates(rows)
    assert rate_g == pytest.approx(share_g * LEAGUE_AVERAGE_GOALS)


def test_penalty_taker_gets_a_higher_prior(share_history, share_row):
    """A first-choice penalty taker with no history is credited above his position's share."""
    m = ShareInvolvement(penalty_share=0.1).fit(share_history)
    taker = share_row("new-taker", "FWD", penalties_order=1)
    nobody = share_row("new-nobody", "FWD")
    g, _ = m.share(Dataset([taker, nobody], record_type=PlayerStat))
    assert g[0] == pytest.approx(g[1] + 0.1)


def test_history_overrides_the_set_piece_prior(share_history):
    """A taker with a long blank history is pulled down to what he actually does."""
    strong = ShareInvolvement(prior_goals=1.0).fit(share_history)
    weak = ShareInvolvement(prior_goals=100.0).fit(share_history)
    rows = share_history.filter(
        lambda r: r.player.id == "blank-taker" and r.fixture_id.endswith("-0")
    )
    (g_strong,), _ = strong.share(rows)
    (g_weak,), _ = weak.share(rows)
    assert g_strong < g_weak


def test_variants_score_on_the_same_contract(share_history):
    """Both involvement variants score the same rows, so they can be pitted against each other."""
    share = ShareInvolvement().fit(share_history).score(share_history)
    eb = EmpiricalBayesInvolvement().fit(share_history).score(share_history)
    assert np.isfinite(share) and np.isfinite(eb)
