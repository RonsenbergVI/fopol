"""Dixon-Coles team model: hierarchical bivariate Poisson with a low-score correction.

Each team gets latent attacking and defending strengths drawn jointly from a
population distribution with a learned correlation -- good sides tend to be
good at both -- so teams with little data are shrunk toward the league rather
than fit to noise. For home ``h`` and away ``a``::

    log lambda_home = intercept + home_advantage + attack[h] - defence[a]
    log lambda_away = intercept                  + attack[a] - defence[h]

with the Dixon-Coles ``tau`` correction inflating the four low-scoring
outcomes that independent Poissons underestimate. Optional exponential time
decay downweights older results.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from numpyro.infer import MCMC, NUTS

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, TeamModel, _poisson_logpmf
from fopol.data.result import Result
from fopol.data.team import TeamRef

__all__ = ["DixonColesTeamModel"]


def _dc_log_tau(hg, ag, lam_h, lam_a, rho):
    tau = jnp.where(
        (hg == 0) & (ag == 0),
        1.0 - lam_h * lam_a * rho,
        jnp.where(
            (hg == 0) & (ag == 1),
            1.0 + lam_h * rho,
            jnp.where(
                (hg == 1) & (ag == 0),
                1.0 + lam_a * rho,
                jnp.where((hg == 1) & (ag == 1), 1.0 - rho, 1.0),
            ),
        ),
    )
    return jnp.log(jnp.clip(tau, 1e-8, None))


class DixonColesTeamModel(TeamModel):
    """Hierarchical Dixon-Coles, fitted by NUTS.

    Args:
        half_life_days: Exponentially downweight older results with this
            half-life; ``None`` weights every result equally.
        dixon_coles: Apply the low-score correction. Off gives independent
            Poissons, a useful baseline.
        prior_home_advantage: Prior mean of the log home-advantage term.
        teams: Fixed team ordering, optionally a superset of the teams seen,
            so a club with no results yet is drawn from the population prior.
        promoted: Clubs new to the league this season. A club with no history
            would otherwise be drawn as league-average; promoted sides are not,
            so they share a shift on attack and defence with prior mean
            ``prior_promoted`` and scale 0.15, learned from whatever results
            they have. Ignored for clubs with a full season behind them.
        prior_promoted: Prior mean of that (attack, defence) shift on the log
            scale; the default is roughly what promoted sides have averaged
            against the league -- about 25% fewer goals scored, 20% more conceded.
        population_df: Degrees of freedom of a Student-t population
            distribution for club effects; ``None`` is Gaussian. Heavy tails let
            the best and worst clubs sit further from the pack than a Gaussian
            population allows -- Baio and Blangiardo's over-shrinkage problem --
            while the pack is pooled as before. Must exceed 2.
        inference: Sampler settings.
    """

    def __init__(
        self,
        *,
        half_life_days: float | None = None,
        dixon_coles: bool = True,
        prior_home_advantage: float = 0.25,
        teams: list[str] | None = None,
        promoted: list[str] | None = None,
        prior_promoted: tuple[float, float] = (-0.25, -0.2),
        population_df: float | None = None,
        inference: Inference | None = None,
    ) -> None:
        if half_life_days is not None and half_life_days <= 0:
            raise ValueError("half_life_days must be positive or None")
        self.half_life_days = half_life_days
        self.dixon_coles = dixon_coles
        self.prior_home_advantage = prior_home_advantage
        self.teams = teams
        self.promoted = promoted
        self.prior_promoted = prior_promoted
        if population_df is not None and population_df <= 2:
            raise ValueError("population_df must exceed 2 (finite variance) or be None")
        self.population_df = population_df
        self.inference = inference or Inference()

    # ------------------------------------------------------------- numpyro

    def _numpyro_model(
        self, home_idx, away_idx, home_goals, away_goals, n_teams, weights, promoted_mask
    ):
        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
        home_advantage = numpyro.sample(
            "home_advantage", dist.Normal(self.prior_home_advantage, 0.25)
        )
        sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(0.5).expand([2]).to_event(1)))
        chol_corr = jnp.asarray(numpyro.sample("chol_corr", dist.LKJCholesky(2, concentration=2.0)))
        scale_tril = sigma[:, None] * chol_corr

        with numpyro.plate("teams", n_teams):
            base = (
                dist.Normal(0.0, 1.0)
                if self.population_df is None
                else dist.StudentT(self.population_df, 0.0, 1.0)
            )
            z = jnp.asarray(numpyro.sample("z", base.expand([2]).to_event(1)))
        effects = z @ scale_tril.T
        if promoted_mask is not None:
            shift = jnp.asarray(
                numpyro.sample(
                    "promoted", dist.Normal(jnp.asarray(self.prior_promoted), 0.15).to_event(1)
                )
            )
            effects = effects + promoted_mask[:, None] * shift
        attack = jnp.asarray(numpyro.deterministic("attack", effects[:, 0]))
        defence = jnp.asarray(numpyro.deterministic("defence", effects[:, 1]))
        numpyro.deterministic("attack_defence_corr", chol_corr[1, 0])

        lam_h = jnp.exp(intercept + home_advantage + attack[home_idx] - defence[away_idx])
        lam_a = jnp.exp(intercept + attack[away_idx] - defence[home_idx])

        ll = (home_goals * jnp.log(lam_h) - lam_h - gammaln(home_goals + 1.0)) + (
            away_goals * jnp.log(lam_a) - lam_a - gammaln(away_goals + 1.0)
        )
        if self.dixon_coles:
            rho = numpyro.sample("rho", dist.Normal(0.0, 0.05))
            ll = ll + _dc_log_tau(home_goals, away_goals, lam_h, lam_a, rho)
        if weights is not None:
            ll = ll * weights
        numpyro.factor("match_likelihood", ll.sum())

    # ------------------------------------------------------------------ fit

    def _fit(self, data: Dataset[Result]) -> None:
        played = [r for r in data if r.played]
        if not played:
            raise ValueError("no played results to fit on")
        seen = {str(r.home) for r in played} | {str(r.away) for r in played}
        teams = sorted(seen) if self.teams is None else [str(t) for t in self.teams]
        lookup = {t: i for i, t in enumerate(teams)}
        unknown = sorted(seen - lookup.keys())
        if unknown:
            raise ValueError(f"results mention teams absent from `teams`: {unknown}")

        home_idx = np.array([lookup[str(r.home)] for r in played], dtype=np.int32)
        away_idx = np.array([lookup[str(r.away)] for r in played], dtype=np.int32)
        hg = np.array([r.home_goals for r in played], dtype=np.float32)
        ag = np.array([r.away_goals for r in played], dtype=np.float32)

        weights = None
        if self.half_life_days is not None:
            kickoffs = [r.kickoff for r in played if r.kickoff is not None]
            if len(kickoffs) != len(played):
                raise ValueError("half_life_days needs a kickoff on every played result")
            latest = max(kickoffs)
            days = np.array([(k - latest).total_seconds() / 86400.0 for k in kickoffs])
            weights = jnp.asarray(np.exp(math.log(2.0) / self.half_life_days * days))

        promoted_mask = None
        if self.promoted:
            unknown_promoted = sorted(set(map(str, self.promoted)) - lookup.keys())
            if unknown_promoted:
                raise ValueError(f"`promoted` names teams absent from `teams`: {unknown_promoted}")
            promoted_mask = jnp.asarray(
                np.array([t in set(map(str, self.promoted)) for t in teams], dtype=np.float32)
            )

        inf = self.inference
        mcmc = MCMC(
            NUTS(self._numpyro_model, target_accept_prob=inf.target_accept_prob),
            num_warmup=inf.num_warmup,
            num_samples=inf.num_samples,
            num_chains=inf.num_chains,
            progress_bar=inf.progress_bar,
        )
        mcmc.run(
            jax.random.PRNGKey(inf.seed),
            home_idx=jnp.asarray(home_idx),
            away_idx=jnp.asarray(away_idx),
            home_goals=jnp.asarray(hg),
            away_goals=jnp.asarray(ag),
            n_teams=len(teams),
            weights=weights,
            promoted_mask=promoted_mask,
        )
        self.mcmc_ = mcmc
        self.posterior_ = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self.teams_ = teams

    # ------------------------------------------------------------- predict

    def goal_rate_draws(self, home, away):
        s = self.posterior
        h, a = self._team_index(home), self._team_index(away)
        if len(h) != len(a):
            raise ValueError(f"got {len(h)} home teams but {len(a)} away teams")
        base = s["intercept"][:, None]
        adv = s["home_advantage"][:, None]
        lam_h = np.exp(base + adv + s["attack"][:, h] - s["defence"][:, a])
        lam_a = np.exp(base + s["attack"][:, a] - s["defence"][:, h])
        return lam_h, lam_a

    def scoreline_probs(self, home, away, *, max_goals: int = 8) -> np.ndarray:
        lam_h, lam_a = self.goal_rate_draws(home, away)
        goals = np.arange(max_goals + 1, dtype=float)
        log_h = _poisson_logpmf(goals[None, None, :], lam_h[:, :, None])
        log_a = _poisson_logpmf(goals[None, None, :], lam_a[:, :, None])
        joint = np.exp(log_h[:, :, :, None] + log_a[:, :, None, :])
        if self.dixon_coles and "rho" in self.posterior and max_goals >= 1:
            rho = self.posterior["rho"][:, None]
            tau = np.ones_like(joint)
            tau[:, :, 0, 0] = 1.0 - lam_h * lam_a * rho
            tau[:, :, 0, 1] = 1.0 + lam_h * rho
            tau[:, :, 1, 0] = 1.0 + lam_a * rho
            tau[:, :, 1, 1] = 1.0 - rho
            joint = joint * np.clip(tau, 1e-8, None)
        probs = joint.mean(axis=0)
        return probs / probs.sum(axis=(1, 2), keepdims=True)

    # ------------------------------------------------------------- reports

    def team_strength(self) -> list[dict]:
        """Posterior mean and spread of each team's attack and defence."""
        s = self.posterior
        rows = []
        for i, team in enumerate(self.teams_):
            rows.append(
                {
                    "team": TeamRef(id=team),
                    "attack": float(s["attack"][:, i].mean()),
                    "attack_sd": float(s["attack"][:, i].std()),
                    "defence": float(s["defence"][:, i].mean()),
                    "defence_sd": float(s["defence"][:, i].std()),
                }
            )
        return sorted(rows, key=lambda r: -r["attack"])
