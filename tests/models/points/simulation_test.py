"""Points family: composition through stub sub-models, so the arithmetic is checked alone."""

import numpy as np
import pytest

from fopol.base.dataset import Dataset
from fopol.base.model import NotFittedError
from fopol.data.player import PlayerStat


def test_requires_fitted_submodels(points_model, points_rows):
    """Fitting with an unfitted sub-model raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        points_model.fit(points_rows)


def test_predictions_are_records_with_sane_shape(fitted_points_model, points_rows):
    """One prediction per row, with probabilities in range."""
    pred = fitted_points_model.predict(points_rows)
    assert len(pred) == 3 and pred.record_type.__name__ == "PointsPrediction"
    by = {str(p.player): p for p in pred}
    assert by["striker"].mean > by["fringe"].mean
    assert by["striker"].p_haul > by["keeper"].p_haul
    assert 0 <= by["fringe"].p_blank <= 1


def test_keeper_has_no_haul_without_saves_or_bonus(fitted_points_model, points_rows):
    """Visible consequence of unmodelled scoring, kept as a reminder."""
    by = {str(p.player): p for p in fitted_points_model.predict(points_rows)}
    assert by["keeper"].p_haul == 0.0
    assert by["keeper"].mean <= 6.0


def test_score_uses_observed_points(fitted_points_model, points_rows):
    """A keeper without saves or bonus cannot haul in this composition."""
    played = Dataset(
        [r.model_copy(update={"minutes": 90, "total_points": 2}) for r in points_rows],
        record_type=PlayerStat,
    )
    s = fitted_points_model.score(played)
    assert np.isfinite(s) and s < 0


def test_value_scalar_rewards_unowned_upside(fitted_points_model, points_rows):
    """``score`` is finite and rewards rows that land near the mean."""
    by = {str(p.player): p for p in fitted_points_model.predict(points_rows)}
    p = by["striker"]
    assert p.value(ownership=0.5, kappa=0.0) == pytest.approx(p.mean)
    assert p.value(ownership=0.05, kappa=1.0) > p.value(ownership=0.60, kappa=1.0)
