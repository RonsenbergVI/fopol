# fopol

Bayesian models of Fantasy Premier League, built on NumPyro.

fopol is a library, not an app. It fetches per-game data from the sources that publish it, keeps that data as typed records, fits hierarchical models over it, and composes those models into a posterior over a player's FPL points rather than a point estimate. That last step is the argument the library makes: with a distribution in hand you can ask for the probability of a haul when choosing a captain, instead of settling for an expectation.

## How it fits together

- **Sources** are clients. Each knows one place data lives and how to translate its dialect into fopol records, at per-game granularity, with identity resolved at the boundary.
- **Data** are the typed records everything else is written against. A column declares how it aggregates in its own annotation, so a season view is derived rather than fetched.
- **Models** come in four families, one per role: team goals, appearance minutes, goal involvement, and the points model that composes the other three. Each family has an abstract base that pins the record types, and implementations that can be swapped freely.

## Installing

```bash
pip install fopol
```

The `diagnostics` extra adds `arviz` and `matplotlib` for posterior checks.

## Reading the models

The team model is a Dixon and Coles style bivariate Poisson with hierarchical team strengths. For a fixture between home team $h$ and away team $a$:

$$
\begin{aligned}
\log \lambda_{\text{home}} &= \mu + \gamma + \alpha_h - \delta_a \\
\log \lambda_{\text{away}} &= \mu + \alpha_a - \delta_h
\end{aligned}
$$

with $\mu$ the intercept, $\gamma$ home advantage, and $\alpha$, $\delta$ each team's attack and defence drawn jointly from a population distribution. Teams with little data are shrunk toward the league average, which is what you want for a newly promoted side in August. Every model page in the API reference states its likelihood the same way.
