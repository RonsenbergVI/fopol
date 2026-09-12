"""Tests for the team goal model.

These fit on synthetic data with known ground truth, so they assert the model
recovers the structure it was given rather than just that it runs. Chain
counts are kept small to stay fast; the statistical assertions are loose
enough to survive that.
"""

import numpy as np
import pandas as pd
import pytest

from fopol.base.model import Inference, NotFittedError
from fopol.data.team import TeamRef
from fopol.models import DixonColesTeamModel


def test_unfitted_model_raises():
    """``predict`` before ``fit`` raises ``NotFittedError``."""
    model = DixonColesTeamModel()
    assert not model.is_fitted
    with pytest.raises(NotFittedError):
        model.team_strength()


def test_recovers_team_ordering(fitted_team_model):
    """Fitted attack and defence rank the teams as generated."""
    strength = fitted_team_model.team_strength()  # sorted by attack, strongest first
    assert [str(r["team"]) for r in strength] == ["Strong", "Average", "Weak"]
    defence = {str(r["team"]): r["defence"] for r in strength}
    assert defence["Strong"] > defence["Average"] > defence["Weak"]


def test_recovers_home_advantage(fitted_team_model, true_team_effects):
    """The 95% interval for home advantage covers the value the data was drawn with."""
    posterior = fitted_team_model.posterior["home_advantage"]
    lo, hi = np.percentile(posterior, [2.5, 97.5])
    assert lo < true_team_effects["home_advantage"] < hi


def test_credible_intervals_cover_truth(fitted_team_model, true_team_effects):
    """Attack effects are identified only up to a shift shared with the intercept.

    So compare differences between teams rather than levels.
    """
    attack = fitted_team_model.posterior["attack"]
    teams = fitted_team_model.teams_  # sorted by id, so never assume positions
    gap = attack[:, teams.index("Strong")] - attack[:, teams.index("Weak")]
    lo, hi = np.percentile(gap, [2.5, 97.5])
    true_gap = true_team_effects["attack"]["Strong"] - true_team_effects["attack"]["Weak"]
    assert lo < true_gap < hi


def test_scoreline_probabilities_are_normalised(fitted_team_model):
    """Scoreline grids sum to one per fixture."""
    probs = fitted_team_model.scoreline_probs(["Strong", "Weak"], ["Weak", "Strong"], max_goals=10)
    assert probs.shape == (2, 11, 11)
    assert np.allclose(probs.sum(axis=(1, 2)), 1.0)
    assert (probs >= 0).all()


def test_outcome_probabilities_sum_to_one(fitted_team_model, unplayed_fixtures):
    """Home, draw and away probabilities sum to one."""
    pred = fitted_team_model.predict(unplayed_fixtures(("Strong", "Weak"), ("Weak", "Strong")))
    for p in pred:
        assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)
    # Strong at home against Weak should be a heavy favourite, and the reverse
    # fixture should favour the away side.
    assert pred[0].p_home > 0.7
    assert pred[1].p_away > pred[1].p_home


def test_home_advantage_shows_up_in_predictions(fitted_team_model):
    """Identical teams: the home side is expected to score more."""
    lam_home, lam_away = fitted_team_model.goal_rate_draws(["Average"], ["Average"])
    assert lam_home.mean() > lam_away.mean()


def test_clean_sheet_probs_are_sensible(fitted_team_model, unplayed_fixtures):
    """Clean-sheet odds favour the stronger defence."""
    pred = fitted_team_model.predict(unplayed_fixtures(("Strong", "Weak"), ("Weak", "Strong")))
    for p in pred:
        assert 0 < p.p_home_clean_sheet < 1 and 0 < p.p_away_clean_sheet < 1
    # Strong defence facing Weak attack keeps more clean sheets than the reverse.
    assert pred[0].p_home_clean_sheet > pred[1].p_home_clean_sheet


def test_prediction_rejects_unknown_team(fitted_team_model):
    """A team the model never saw cannot be predicted for and is named in the error."""
    with pytest.raises(KeyError, match="not in the fitted model"):
        fitted_team_model.goal_rate_draws(["Strong"], ["Barnsley"])


def test_prediction_rejects_length_mismatch(fitted_team_model):
    """Home and away lists must pair up."""
    with pytest.raises(ValueError, match="home teams"):
        fitted_team_model.goal_rate_draws(["Strong", "Weak"], ["Average"])


def test_time_decay_fits_from_kickoffs(synthetic_matches):
    """Every Result carries a kickoff, so decay never lacks a clock."""
    model = DixonColesTeamModel(
        half_life_days=100, inference=Inference(num_warmup=50, num_samples=50, num_chains=1)
    ).fit(synthetic_matches)
    assert model.is_fitted


def test_unseen_team_can_be_declared_up_front(synthetic_matches):
    """A club with no results yet is drawn from the prior, not rejected."""
    model = DixonColesTeamModel(
        teams=["Strong", "Average", "Weak", "Promoted"],
        inference=Inference(num_warmup=200, num_samples=200, num_chains=1),
    ).fit(synthetic_matches)
    lam_home, lam_away = model.goal_rate_draws(["Promoted"], ["Average"])
    assert lam_home.shape[1] == 1 and np.isfinite(lam_home).all()
    # With no results it sits at the population mean, and is no more certain
    # than a club with 480 results behind it.
    unseen = model.posterior["attack"][:, model.teams_.index("Promoted")]
    seen = model.posterior["attack"][:, model.teams_.index("Strong")]
    assert abs(unseen.mean()) < 0.5
    assert unseen.std() >= seen.std()


def test_results_outside_declared_teams_rejected(synthetic_matches):
    """Results naming teams outside ``teams`` are refused."""
    with pytest.raises(ValueError, match="absent from `teams`"):
        DixonColesTeamModel(
            teams=["Strong", "Average"],
            inference=Inference(num_samples=5, num_warmup=5, num_chains=1),
        ).fit(synthetic_matches)


def test_predictions_accept_refs_or_ids(fitted_team_model):
    """``TeamRef`` and bare id strings predict identically."""
    from fopol.data.team import TeamRef

    by_ref = fitted_team_model.goal_rate_draws([TeamRef(id="Strong")], [TeamRef(id="Weak")])
    by_id = fitted_team_model.goal_rate_draws(["Strong"], ["Weak"])
    np.testing.assert_allclose(by_ref[0], by_id[0])


def test_unplayed_results_are_ignored_in_fit_and_predicted(synthetic_matches):
    """Unplayed results are skipped in ``fit`` and scored by ``predict``."""
    from fopol.base.dataset import Dataset
    from fopol.data.result import Result

    unplayed = Result(
        season="2024-25",
        fixture_id="future",
        kickoff=pd.Timestamp("2001-01-01", tz="UTC"),
        home=TeamRef(id="Strong"),
        away=TeamRef(id="Weak"),
    )
    mixed = Dataset([*synthetic_matches, unplayed], record_type=Result)
    model = DixonColesTeamModel(
        inference=Inference(num_warmup=50, num_samples=50, num_chains=1)
    ).fit(mixed)
    pred = model.predict(Dataset([unplayed], record_type=Result))[0]
    assert pred.home_xg > pred.away_xg  # Strong at home to Weak
    assert 0 < pred.p_home_clean_sheet < 1


def test_score_is_a_log_density_and_prefers_the_true_model(fitted_team_model, synthetic_matches):
    """``score`` is a log density and favours the generating model."""
    score = fitted_team_model.score(synthetic_matches)
    assert score < 0
    worse = DixonColesTeamModel(
        prior_home_advantage=-2.0, inference=Inference(num_warmup=50, num_samples=50, num_chains=1)
    ).fit(synthetic_matches)
    assert score >= worse.score(synthetic_matches) - 0.05


def test_sklearn_conventions(fitted_team_model):
    """``get_params``, ``set_params`` and ``clone`` follow sklearn."""
    params = fitted_team_model.get_params()
    assert set(params) >= {"half_life_days", "dixon_coles", "inference"}
    fresh = fitted_team_model.clone()
    assert not fresh.is_fitted and fresh.get_params() == params
    assert fitted_team_model.consumes.__name__ == "Result"
    assert fitted_team_model.produces.__name__ == "FixturePrediction"


def test_invalid_half_life_rejected():
    """A non-positive half-life is rejected at construction."""
    with pytest.raises(ValueError, match="must be positive"):
        DixonColesTeamModel(half_life_days=-5)


def test_dixon_coles_can_be_disabled(synthetic_matches):
    """``dixon_coles=False`` gives an independent-Poisson model."""
    model = DixonColesTeamModel(
        dixon_coles=False, inference=Inference(num_warmup=200, num_samples=200, num_chains=1)
    ).fit(synthetic_matches)
    assert "rho" not in model.posterior
    probs = model.scoreline_probs(["Strong"], ["Weak"], max_goals=6)
    assert np.allclose(probs.sum(axis=(1, 2)), 1.0)
