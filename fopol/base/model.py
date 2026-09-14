"""The sklearn-shaped contract every fopol model satisfies, and each family's abstract base.

The shape is sklearn's because its conventions are worth having: a constructor
that only stores hyperparameters, ``fit`` that returns ``self``, fitted state
marked with a trailing underscore, ``get_params`` for reproducibility. What
changes is the types. Inputs are :class:`~fopol.base.dataset.Dataset`, not
``X, y``; a model declares which record it *consumes* and which it *produces*,
and the base holds it to that. Outputs are posteriors, not points, so
``predict`` returns records that carry uncertainty and ``score`` is a proper
scoring rule, not accuracy.

Below :class:`Model` sit the four family bases -- :class:`TeamModel`,
:class:`MinutesModel`, :class:`InvolvementModel`, :class:`PointsModel`. Each
pins the record types for its role and turns a variant's posterior into
records and into a score, so a variant implements only the posterior. Anything
satisfying a family base is interchangeable with any other variant of it, which
is what makes comparing them meaningful. Variants live in :mod:`fopol.models`.

Three conventions to keep to when subclassing.

**``__init__`` stores hyperparameters and nothing else.** No data, no fitting,
no derived state. Every argument becomes an attribute of the same name. This is
what makes ``get_params`` / ``set_params`` and ``clone`` work, and what lets a
backtest sweep a hyperparameter without knowing the model.

**Fitted state ends in an underscore.** ``posterior_``, ``teams_``. That is how
:meth:`Model.is_fitted` knows, how ``clone`` knows what to drop, and how a
reader tells hyperparameter from estimate at a glance.

**``predict`` takes the same record type as ``fit``, with outcomes blank.** An
unplayed :class:`~fopol.data.result.Result` has ``home_goals=None``; the model
fills in what it can say about it. No second input schema to maintain.
"""

from __future__ import annotations

import copy
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from fopol.constants import HAUL
from fopol.base.data import Data
from fopol.base.dataset import Dataset
from fopol.data.player import PlayerStat
from fopol.data.prediction import (
    AppearancePrediction,
    FixturePrediction,
    InvolvementPrediction,
    PointsPrediction,
)
from fopol.data.result import Result
from fopol.data.team import TeamRef

__all__ = [
    "Inference",
    "InvolvementModel",
    "MinutesModel",
    "Model",
    "NotFittedError",
    "PointsModel",
    "TeamModel",
]


class NotFittedError(RuntimeError):
    """The model has no posterior yet; call ``fit`` first."""


class Inference(BaseModel):
    """How the posterior is obtained. A hyperparameter like any other.

    Kept off the model class so that "which sampler, how many draws" is data
    that travels with the model's params, and so that a backtest can swap NUTS
    for SVI without touching model code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = "nuts"
    """``"nuts"`` for MCMC, ``"svi"`` for variational."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    num_steps: int = 8000
    """SVI only: optimisation steps."""
    seed: int = 0
    target_accept_prob: float = 0.9
    progress_bar: bool = False


class Model(ABC):
    """Base for every model.

    Subclasses declare :attr:`consumes` and :attr:`produces`, implement
    :meth:`_fit` and :meth:`_predict`, and store everything learned in
    attributes ending with an underscore.
    """

    consumes: ClassVar[type[Data]]
    """Record type ``fit`` and ``predict`` take."""

    produces: ClassVar[type[Data]]
    """Record type ``predict`` returns."""

    inference: Inference

    # -------------------------------------------------------- subclass hooks

    @abstractmethod
    def _fit(self, data: Dataset) -> None:
        """Learn from ``data``; set ``*_`` attributes; return nothing."""

    @abstractmethod
    def _predict(self, data: Dataset) -> Dataset:
        """Posterior predictive summaries for ``data``, as :attr:`produces` records."""

    def _log_score(self, data: Dataset) -> np.ndarray:
        """Per-record log predictive density of the observed outcome.

        Optional. Implement it and :meth:`score` becomes the backtest metric
        for free; leave it and :meth:`score` raises.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define a log score")

    # ------------------------------------------------------------ public API

    def fit(self, data: Dataset) -> Self:
        """Fit and return ``self``, so calls chain."""
        self._check_type(data, "fit")
        self._fit(data)
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}._fit set no fitted state; "
                "fitted attributes must end with an underscore"
            )
        return self

    def predict(self, data: Dataset) -> Dataset:
        """Posterior predictive summaries, one :attr:`produces` record per input record."""
        self.check_is_fitted()
        self._check_type(data, "predict")
        out = self._predict(data)
        if out.record_type is not self.produces:
            raise TypeError(
                f"{type(self).__name__}._predict returned {out.record_type.__name__}, "
                f"declared {self.produces.__name__}"
            )
        return out

    def fit_predict(self, data: Dataset) -> Dataset:
        return self.fit(data).predict(data)

    def score(self, data: Dataset) -> float:
        """Mean log predictive density on held-out records. Higher is better.

        A proper scoring rule, so it rewards calibration as well as accuracy --
        which is what a downstream optimiser needs. Negate it for a logloss.
        """
        self.check_is_fitted()
        self._check_type(data, "score")
        return float(np.mean(self._log_score(data)))

    @property
    def is_fitted(self) -> bool:
        return any(k.endswith("_") and not k.startswith("_") for k in vars(self))

    def check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call .fit(data) first")

    @property
    def posterior(self) -> Mapping[str, np.ndarray]:
        """Parameter draws, ``{site: (n_draws, ...)}``. Engine-agnostic."""
        self.check_is_fitted()
        posterior = getattr(self, "posterior_", None)
        if posterior is None:
            raise NotFittedError(f"{type(self).__name__} keeps no `posterior_`")
        return posterior

    @classmethod
    def _param_names(cls) -> tuple[str, ...]:
        signature = inspect.signature(cls.__init__)
        return tuple(
            p.name
            for p in signature.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        )

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters, exactly as passed to the constructor."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        unknown = set(params) - set(self._param_names())
        if unknown:
            raise ValueError(f"{type(self).__name__} has no parameters {sorted(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def clone(self) -> Self:
        """Same hyperparameters, no fitted state."""
        return type(self)(**copy.deepcopy(self.get_params()))

    def __repr__(self) -> str:  # pragma: no cover - display only
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({params})  # {state}"

    def _check_type(self, data: Dataset, where: str) -> None:
        if not isinstance(data, Dataset):
            raise TypeError(
                f"{type(self).__name__}.{where} takes a Dataset, got {type(data).__name__}"
            )
        if data.record_type is not self.consumes:
            raise TypeError(
                f"{type(self).__name__}.{where} consumes {self.consumes.__name__}, "
                f"got Dataset[{data.record_type.__name__}]"
            )


def _poisson_logpmf(k: np.ndarray, rate: np.ndarray) -> np.ndarray:
    lgamma = np.vectorize(math.lgamma)
    return k * np.log(rate) - rate - lgamma(k + 1.0)


class TeamModel(Model):
    """Abstract team goal model. Subclasses implement the posterior only."""

    consumes: ClassVar[type[Result]] = Result
    produces: ClassVar[type[FixturePrediction]] = FixturePrediction

    teams_: list[str]
    """Fitted team ids; the index space of every per-team parameter."""

    @abstractmethod
    def goal_rate_draws(
        self, home: Sequence[TeamRef | str], away: Sequence[TeamRef | str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior draws of expected goals, ``(n_draws, n_fixtures)`` per side."""

    def scoreline_probs(
        self, home: Sequence[TeamRef | str], away: Sequence[TeamRef | str], *, max_goals: int = 8
    ) -> np.ndarray:
        """``(n_fixtures, max_goals+1, max_goals+1)`` joint scoreline probabilities.

        Independent Poissons by default; a variant with a low-score correction
        overrides this.
        """
        lam_h, lam_a = self.goal_rate_draws(home, away)
        goals = np.arange(max_goals + 1, dtype=float)
        log_h = _poisson_logpmf(goals[None, None, :], lam_h[:, :, None])
        log_a = _poisson_logpmf(goals[None, None, :], lam_a[:, :, None])
        joint = np.exp(log_h[:, :, :, None] + log_a[:, :, None, :]).mean(axis=0)
        return joint / joint.sum(axis=(1, 2), keepdims=True)

    def _team_index(self, names: Sequence[TeamRef | str]) -> np.ndarray:
        ids = [str(n) for n in names]
        lookup = {t: i for i, t in enumerate(self.teams_)}
        missing = sorted(set(ids) - lookup.keys())
        if missing:
            raise KeyError(
                f"team(s) not in the fitted model: {missing}. "
                f"Known: {', '.join(sorted(self.teams_))}"
            )
        return np.array([lookup[i] for i in ids], dtype=int)

    def _predict(self, data: Dataset[Result]) -> Dataset[FixturePrediction]:
        fixtures = list(data)
        if not fixtures:
            return Dataset([], record_type=FixturePrediction, source=type(self).__name__)
        home = [r.home for r in fixtures]
        away = [r.away for r in fixtures]
        lam_h, lam_a = self.goal_rate_draws(home, away)
        probs = self.scoreline_probs(home, away)
        idx = np.arange(probs.shape[1])
        home_more = idx[:, None] > idx[None, :]
        level = idx[:, None] == idx[None, :]

        records = [
            FixturePrediction(
                season=r.season,
                fixture_id=r.fixture_id,
                home=r.home,
                away=r.away,
                home_xg=float(lam_h[:, i].mean()),
                away_xg=float(lam_a[:, i].mean()),
                p_home=float((probs[i] * home_more).sum()),
                p_draw=float((probs[i] * level).sum()),
                p_away=float((probs[i] * ~(home_more | level)).sum()),
                # P(clean sheet) = P(opponent scores 0) = E[exp(-lambda_opp)]
                p_home_clean_sheet=float(np.exp(-lam_a[:, i]).mean()),
                p_away_clean_sheet=float(np.exp(-lam_h[:, i]).mean()),
            )
            for i, r in enumerate(fixtures)
        ]
        return Dataset(records, record_type=FixturePrediction, source=type(self).__name__)

    def _log_score(self, data: Dataset[Result]) -> np.ndarray:
        """Log probability of the observed scoreline under the posterior predictive."""
        played = [r for r in data if r.played]
        if not played:
            raise ValueError("score needs played results")
        probs = self.scoreline_probs([r.home for r in played], [r.away for r in played])
        cap = probs.shape[1] - 1
        hg = np.minimum(np.array([r.home_goals for r in played], dtype=int), cap)
        ag = np.minimum(np.array([r.away_goals for r in played], dtype=int), cap)
        return np.log(np.clip(probs[np.arange(len(played)), hg, ag], 1e-12, None))


class MinutesModel(Model):
    """Abstract appearance model over the three FPL bands."""

    consumes: ClassVar[type[PlayerStat]] = PlayerStat
    produces: ClassVar[type[AppearancePrediction]] = AppearancePrediction

    @abstractmethod
    def features_for(self, data: Dataset[PlayerStat]) -> Any:
        """Whatever the variant needs to score ``data``.

        Built causally against the history it was fitted on; opaque to the base.
        """

    @abstractmethod
    def band_probs(self, features: Any) -> np.ndarray:
        """``(n_rows, 3)`` posterior-mean probabilities of the bands 0 / 1-59 / 60+."""

    @abstractmethod
    def observed_bands(self, features: Any) -> np.ndarray:
        """``(n_rows,)`` observed band per row of ``features``, for scoring."""

    def _predict(self, data: Dataset[PlayerStat]) -> Dataset[AppearancePrediction]:
        rows = list(data)
        if not rows:
            return Dataset([], record_type=AppearancePrediction, source=type(self).__name__)
        p = self.band_probs(self.features_for(data))
        records = [
            AppearancePrediction(
                season=r.season,
                fixture_id=r.fixture_id,
                player=r.player,
                p_0=float(p[i, 0]),
                p_1_59=float(p[i, 1]),
                p_60=float(p[i, 2]),
            )
            for i, r in enumerate(rows)
        ]
        return Dataset(records, record_type=AppearancePrediction, source=type(self).__name__)

    def _log_score(self, data: Dataset[PlayerStat]) -> np.ndarray:
        played = Dataset([r for r in data if r.played], record_type=PlayerStat)
        if not len(played):
            raise ValueError("score needs played appearances")
        f = self.features_for(played)
        p = self.band_probs(f)
        return np.log(np.clip(p[np.arange(len(played)), self.observed_bands(f)], 1e-12, None))


class InvolvementModel(Model):
    """Abstract per-90 involvement rates."""

    consumes: ClassVar[type[PlayerStat]] = PlayerStat
    produces: ClassVar[type[InvolvementPrediction]] = InvolvementPrediction

    @abstractmethod
    def rates(self, data: Dataset[PlayerStat]) -> tuple[np.ndarray, np.ndarray]:
        """``(goals_p90, assists_p90)`` per row, posterior means."""

    def _predict(self, data: Dataset[PlayerStat]) -> Dataset[InvolvementPrediction]:
        rows = list(data)
        if not rows:
            return Dataset([], record_type=InvolvementPrediction, source=type(self).__name__)
        g, a = self.rates(data)
        return Dataset(
            [
                InvolvementPrediction(
                    season=r.season,
                    fixture_id=r.fixture_id,
                    player=r.player,
                    goals_p90=float(g[i]),
                    assists_p90=float(a[i]),
                )
                for i, r in enumerate(rows)
            ],
            record_type=InvolvementPrediction,
            source=type(self).__name__,
        )

    def _log_score(self, data: Dataset[PlayerStat]) -> np.ndarray:
        """Poisson log-likelihood of observed goals and assists given the rates."""
        played = [r for r in data if r.played and r.minutes]
        if not played:
            raise ValueError("score needs played appearances with minutes")
        sub = Dataset(played, record_type=PlayerStat)
        g, a = self.rates(sub)
        exposure = np.array([r.minutes for r in played]) / 90.0
        lg = np.vectorize(math.lgamma)
        out = np.zeros(len(played))
        for rate, obs in ((g, [r.goals for r in played]), (a, [r.assists for r in played])):
            k = np.array(obs, dtype=float)
            lam = np.clip(rate * exposure, 1e-9, None)
            out += k * np.log(lam) - lam - lg(k + 1.0)
        return out


class PointsModel(Model):
    """Abstract points model."""

    consumes: ClassVar[type[PlayerStat]] = PlayerStat
    produces: ClassVar[type[PointsPrediction]] = PointsPrediction

    @abstractmethod
    def points_draws(self, data: Dataset[PlayerStat]) -> np.ndarray:
        """``(n_draws, n_rows)`` simulated FPL points."""

    @abstractmethod
    def p_60(self, data: Dataset[PlayerStat]) -> np.ndarray:
        """``(n_rows,)`` probability of playing 60+ minutes."""

    def _predict(self, data: Dataset[PlayerStat]) -> Dataset[PointsPrediction]:
        rows = list(data)
        if not rows:
            return Dataset([], record_type=PointsPrediction, source=type(self).__name__)
        d = self.points_draws(data)
        p60 = self.p_60(data)
        return Dataset(
            [
                PointsPrediction(
                    season=r.season,
                    fixture_id=r.fixture_id,
                    player=r.player,
                    mean=float(d[:, i].mean()),
                    sd=float(d[:, i].std()),
                    p_blank=float((d[:, i] <= 2).mean()),
                    p_haul=float((d[:, i] >= HAUL).mean()),
                    p_60=float(p60[i]),
                )
                for i, r in enumerate(rows)
            ],
            record_type=PointsPrediction,
            source=type(self).__name__,
        )

    def _log_score(self, data: Dataset[PlayerStat]) -> np.ndarray:
        """Log probability of the observed points under the simulated distribution.

        The pmf is empirical over draws with add-one smoothing on the observed
        support, so an outcome the simulation never produced is improbable, not
        impossible.
        """
        played = [r for r in data if r.played]
        if not played:
            raise ValueError("score needs played appearances")
        d = self.points_draws(Dataset(played, record_type=PlayerStat))
        lo, hi = int(min(d.min(), -5)), int(max(d.max(), 25))
        support = hi - lo + 1
        out = np.empty(len(played))
        for i, r in enumerate(played):
            hits = int((np.round(d[:, i]).astype(int) == r.total_points).sum())
            out[i] = np.log((hits + 1.0) / (d.shape[0] + support))
        return out
