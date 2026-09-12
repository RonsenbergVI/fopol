# fopol

Bayesian models of Fantasy Premier League, built on NumPyro.

fopol fetches per-game data from the places that publish it, keeps it as typed records, fits hierarchical models over those records, and composes the models into a posterior over a player's FPL points. That last part is the point: with a distribution rather than a number you can ask for the probability of a haul when choosing a captain, or price a differential by its tail instead of its mean.

## Installing

```bash
pip install fopol
```

Python 3.11 or newer. The `diagnostics` extra adds `arviz` and `matplotlib` for posterior checks:

```bash
pip install "fopol[diagnostics]"
```

## Quickstart

Every model has the sklearn shape: a constructor that stores hyperparameters, `fit` that returns `self`, `predict` that returns records, `score` that is a proper scoring rule. Inputs and outputs are `Dataset` objects of typed records, never frames.

```python
from fopol.models import (
    DixonColesTeamModel,
    EmpiricalBayesInvolvement,
    SequentialMinutesModel,
    SimulatedPointsModel,
)
from fopol.sources import FPLArchiveSource

# One client per place data lives. Fetches are cached on disk.
archive = FPLArchiveSource()
results = archive.load("results", ["2023-24", "2024-25"])
appearances = archive.load("appearances", ["2023-24", "2024-25"])

# Three layers, each a hierarchical model in its own right.
team = DixonColesTeamModel(half_life_days=250).fit(results)
minutes = SequentialMinutesModel().fit(appearances)
involvement = EmpiricalBayesInvolvement().fit(appearances)

# Composed by simulation into a distribution over points.
points = SimulatedPointsModel(team=team, minutes=minutes, involvement=involvement)
points.fit(appearances)

# Predict on the same record type, with the outcomes left blank.
for p in points.predict(upcoming):
    print(p.player, round(p.mean, 2), round(p.sd, 2), round(p.p_haul, 3))
```

`upcoming` is a `Dataset` of `PlayerStat` rows for the fixtures you want scored, with `minutes` and the other outcomes left as `None`. That is what a prediction row is; there is no second input schema.

Any fitted model scores held-out data with its own likelihood, so variants of the same family compare directly:

```python
baseline = DixonColesTeamModel(dixon_coles=False).fit(results)
print(team.score(held_out), baseline.score(held_out))  # mean log predictive density
```

## How it is put together

- **Sources** (`fopol.sources`) are clients: the FPL archive, the live FPL API, and Understat. Each knows one place data lives and how to translate its dialect into fopol records, at per-game granularity, with team and player identity resolved at the boundary.
- **Data** (`fopol.data`) are the records everything is written against: `Result`, `PlayerStat`, `TeamStat`, and the prediction records the models return. A column declares how it aggregates in its own annotation, so a season view is derived from per-game rows rather than fetched.
- **Models** (`fopol.models`) come in four families, one per role. Team goals: a Dixon and Coles bivariate Poisson with team strengths pooled across the league. Minutes: a two-stage model of whether a player features and whether he lasts the hour. Involvement: per-90 goal and assist rates shrunk toward the position. Points: the composition of the other three by simulation. Each family has an abstract base that pins the record types, and any variant of a family is interchangeable with any other.

## What it does not do yet

Scoring covers appearance, goals, assists, clean sheets and goals conceded. Saves, bonus, defensive contribution, cards and penalties are not modelled, so keepers and defenders are undervalued. The involvement layer estimates each player from his own history and is not conditioned on his club, so a striker who moves carries his old side's volume with him. Both are stated in the relevant module docstrings.

## Documentation

The full reference, with each model's likelihood written out, is at [fopol.readthedocs.io](https://fopol.readthedocs.io).

## Development

The environment is managed by `uv`:

```bash
uv sync --all-extras
uv run pytest -m "not integration"
uv run ruff check . && uv run ruff format --check . && uv run mypy fopol
uv run mkdocs build --strict
```

`AGENTS.md` describes how code is added here: the conventions, the test layout, and how releases are cut.

## License

MIT. See [LICENSE](LICENSE).
