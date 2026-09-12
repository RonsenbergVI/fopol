"""Sequential (continuation-ratio) appearance model.

The three bands are ordered, but the two thresholds are different mechanisms.
Whether a player features is a selection decision; whether he lasts an hour
once selected is a substitution decision. A proportional-odds ordered logit
forces one coefficient per feature across both, and the data refuses -- a
player's longer-run start rate loads negatively on selection and strongly
positively on lasting the hour. So each stage gets its own coefficients::

    P(plays)       = sigmoid(a1[pos] + u1[player] + x . b1)
    P(60+ | plays) = sigmoid(a2[pos] + u2[player] + x . b2)

Per-position intercepts recover the rotation structure -- keepers are hardest
to select and likeliest to finish -- without being told it. Player effects are
pooled within position and, measured on a season, help when refreshed in-season
and hurt when fitted once and left; ``player_effects=False`` is the ablation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, MinutesModel
from fopol.data.player import PlayerStat
from fopol.models.minutes.features import POSITIONS, Features, build_features

__all__ = ["SequentialMinutesModel"]

STAGES = ("plays", "sixty")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SequentialMinutesModel(MinutesModel):
    """Two-stage appearance model, SVI by default.

    Args:
        player_effects: Hierarchical per-player intercepts at each stage.
        prior_coef_scale: Normal prior scale on feature coefficients.
        inference: Defaults to SVI, which fits in seconds; ``"nuts"`` is exact and
            slow at ~1,500 players.
    """

    def __init__(
        self,
        *,
        player_effects: bool = True,
        prior_coef_scale: float = 1.0,
        inference: Inference | None = None,
    ) -> None:
        self.player_effects = player_effects
        self.prior_coef_scale = prior_coef_scale
        self.inference = inference or Inference(method="svi")

    # ------------------------------------------------------------- numpyro

    def _offsets(self, stage, n_players, player_position, n_positions):
        if not self.player_effects:
            return jnp.zeros(n_players)
        sigma = jnp.asarray(
            numpyro.sample(f"sigma_{stage}", dist.HalfNormal(1.5).expand([n_positions]).to_event(1))
        )
        with numpyro.plate(f"players_{stage}", n_players):
            z = jnp.asarray(numpyro.sample(f"z_{stage}", dist.Normal(0.0, 1.0)))
        return jnp.asarray(numpyro.deterministic(f"u_{stage}", sigma[player_position] * z))

    def _numpyro_model(
        self, x, player_idx, position_idx, player_position, n_players, n_positions, band=None
    ):
        n_features = x.shape[1]
        etas = []
        for stage in STAGES:
            a = jnp.asarray(
                numpyro.sample(
                    f"intercept_{stage}", dist.Normal(0.0, 3.0).expand([n_positions]).to_event(1)
                )
            )
            b = jnp.asarray(
                numpyro.sample(
                    f"beta_{stage}",
                    dist.Normal(0.0, self.prior_coef_scale).expand([n_features]).to_event(1),
                )
            )
            u = self._offsets(stage, n_players, player_position, n_positions)
            etas.append(a[position_idx] + u[player_idx] + x @ b)
        eta_plays, eta_sixty = etas

        with numpyro.plate("selection", x.shape[0]):
            numpyro.sample(
                "plays",
                dist.Bernoulli(logits=eta_plays),
                obs=None if band is None else jnp.asarray(band > 0),
            )
        if band is not None:
            band = jnp.asarray(band)
            idx = jnp.nonzero(band > 0, size=band.shape[0], fill_value=-1)[0]
            keep = idx >= 0
            sixty = (band[idx] == 2).astype(jnp.int32)
            with numpyro.plate("substitution", idx.shape[0]), numpyro.handlers.mask(mask=keep):
                numpyro.sample("sixty", dist.Bernoulli(logits=eta_sixty[idx]), obs=sixty)

    # ------------------------------------------------------------------ fit

    def _fit(self, data: Dataset[PlayerStat]) -> None:
        history = [r for r in data if r.played]
        if not history:
            raise ValueError("no played appearances to fit on")
        f = build_features(history)
        # each player's position: modal over his rows
        pos_by_player: dict[str, list[int]] = {}
        for r in history:
            pos_by_player.setdefault(str(r.player), []).append(POSITIONS.index(r.position.value))
        player_position = np.array(
            [max(set(v), key=v.count) for p in f.players for v in [pos_by_player[p]]],
            dtype=np.int32,
        )

        args = dict(
            x=jnp.asarray(f.x),
            player_idx=jnp.asarray(f.player_idx),
            position_idx=jnp.asarray(f.position_idx),
            player_position=jnp.asarray(player_position),
            n_players=len(f.players),
            n_positions=len(POSITIONS),
            band=jnp.asarray(f.band),
        )
        inf = self.inference
        key = jax.random.PRNGKey(inf.seed)
        if inf.method == "nuts":
            mcmc = MCMC(
                NUTS(self._numpyro_model, target_accept_prob=inf.target_accept_prob),
                num_warmup=inf.num_warmup,
                num_samples=inf.num_samples,
                num_chains=inf.num_chains,
                progress_bar=inf.progress_bar,
            )
            mcmc.run(key, **args)
            self.mcmc_ = mcmc
            samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        elif inf.method == "svi":
            guide = autoguide.AutoNormal(self._numpyro_model)
            svi = SVI(self._numpyro_model, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
            result = svi.run(key, inf.num_steps, progress_bar=inf.progress_bar, **args)
            draws = Predictive(guide, params=result.params, num_samples=inf.num_samples)(
                jax.random.PRNGKey(inf.seed + 1), **args
            )
            samples = {k: np.asarray(v) for k, v in draws.items()}
        else:
            raise ValueError(f"inference.method must be 'nuts' or 'svi', got {inf.method!r}")

        for stage in STAGES:  # AutoNormal returns latent sites only; rebuild offsets
            if f"u_{stage}" not in samples:
                samples[f"u_{stage}"] = (
                    samples[f"sigma_{stage}"][:, player_position] * samples[f"z_{stage}"]
                    if self.player_effects
                    else np.zeros((samples["beta_plays"].shape[0], len(f.players)))
                )
        self.posterior_ = samples
        self.history_ = history
        self.players_ = f.players

    # ------------------------------------------------------------- predict

    def features_for(self, data: Dataset[PlayerStat]) -> Features:
        """Causal features for ``data`` against the fitted history."""
        self.check_is_fitted()
        return build_features(self.history_, list(data), players=self.players_)

    def observed_bands(self, features: Features) -> np.ndarray:
        return features.band

    def _logits(self, features: Features, stage: str) -> np.ndarray:
        s = self.posterior
        u = np.concatenate([s[f"u_{stage}"], np.zeros((s[f"u_{stage}"].shape[0], 1))], axis=1)
        return (
            s[f"intercept_{stage}"][:, features.position_idx]
            + u[:, features.player_idx]
            + s[f"beta_{stage}"] @ features.x.T
        )

    def band_probs(self, features: Features) -> np.ndarray:
        p1 = _sigmoid(self._logits(features, "plays"))
        p2 = _sigmoid(self._logits(features, "sixty"))
        probs = np.stack([1.0 - p1, p1 * (1.0 - p2), p1 * p2], axis=-1).mean(axis=0)
        probs = np.clip(probs, 1e-9, 1.0)
        return probs / probs.sum(axis=-1, keepdims=True)
