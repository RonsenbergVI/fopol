"""Sequential minutes model: recovers roles from synthetic rosters, and variants are comparable."""

import numpy as np
import pytest

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, NotFittedError
from fopol.data.player import PlayerStat
from fopol.models.minutes import SequentialMinutesModel


def test_unfitted_raises(minutes_row):
    """``predict`` before ``fit`` raises ``NotFittedError``."""
    with pytest.raises(NotFittedError):
        SequentialMinutesModel().predict(
            Dataset([minutes_row("p", 1, None)], record_type=PlayerStat)
        )


def test_probabilities_are_valid(fitted_minutes_model, synthetic_appearances):
    """Band probabilities are a valid simplex per row."""
    pred = fitted_minutes_model.predict(synthetic_appearances)
    p = np.array([[r.p_0, r.p_1_59, r.p_60] for r in pred])
    assert p.shape == (len(synthetic_appearances), 3)
    assert np.allclose(p.sum(1), 1.0) and (p >= 0).all()


def test_recovers_role_ordering(fitted_minutes_model, synthetic_appearances):
    """Nailed players are predicted to play more than rotation, rotation more than fringe."""
    pred = fitted_minutes_model.predict(synthetic_appearances)
    by_role = {}
    for r in pred:
        by_role.setdefault(str(r.player).split("-")[0], []).append(r.p_60)
    means = {k: np.mean(v) for k, v in by_role.items()}
    assert means["nailed"] > means["rotation"] > means["fringe"]


def test_predicts_unseen_player_from_population(minutes_row, fitted_minutes_model):
    """A player absent from training is scored from the population."""
    unseen = Dataset([minutes_row("nobody", 31, None)], record_type=PlayerStat)
    pred = fitted_minutes_model.predict(unseen)[0]
    assert 0.0 < pred.p_60 < 1.0


def test_score_beats_base_rates(fitted_minutes_model, synthetic_appearances):
    """The fitted model out-scores the marginal band frequencies."""
    bands = np.array([r.band for r in synthetic_appearances])
    rates = np.bincount(bands, minlength=3) / len(bands)
    base = np.log(rates[bands]).mean()
    assert fitted_minutes_model.score(synthetic_appearances) > base


def test_variants_are_comparable(synthetic_appearances):
    """Two variants can be ranked by ``score`` on the same data."""
    inf = Inference(method="svi", num_steps=1500, num_samples=100)
    with_fx = SequentialMinutesModel(inference=inf).fit(synthetic_appearances)
    without = SequentialMinutesModel(player_effects=False, inference=inf).fit(synthetic_appearances)
    assert np.allclose(without.posterior["u_plays"], 0.0)
    # same API, same held-out data, one number each
    assert isinstance(with_fx.score(synthetic_appearances), float)
    assert isinstance(without.score(synthetic_appearances), float)
