# AGENTS.md

fopol is a Python library of Bayesian models over Fantasy Premier League, built on NumPyro, with sources that fetch per-game data, typed records the models consume, and model families that compose into a points posterior. The PuLP optimisers that turn a posterior into a squad were removed in the move to records and will come back on top of them; nothing in the tree should anticipate their shape. Architecture background lives in the module docstrings: `fopol/base/data.py` (why records, not frames), `fopol/base/source.py` (the three rules every source obeys), `fopol/base/model.py` (the sklearn-shaped contract and the four families), `fopol/models/points/simulation.py` (how the layers compose). This file is about *how to add code* here.

## Scope discipline

Do exactly what was asked — nothing more. If the request is a text or analysis task (e.g. "write a PR description"), produce only that; do not edit code, config, or files as a side effect. When the right move is unclear, or you're tempted to go beyond the literal request, stop and ask instead of doing something off-task.

## The rule: no glue code

New behavior must become an organic part of the codebase, expressed through the concepts and interfaces that already exist — not bridged to them.

Concretely, when an addition doesn't fit an existing contract:

- **Do not** introduce adapter/wrapper classes, module-level helper functions that convert between representations, or a parallel representation (a dict, a frame, a namedtuple) that stands in for the type the contract wants.
- **Do** reshape the addition so it genuinely satisfies the existing contract — usually by giving it the base type, the declared record type, or the annotation the contract reads, even when that ripples across many files. A wide mechanical refactor that conforms to the existing pattern is preferred over a small bolt-on that doesn't.
- Method sets follow the pattern of their peers. If existing types implement a contract a certain way, a new participant implements it the same way, with the same signature shape.

### Worked example (the canonical one)

Every record declares its own schema in its annotations, and that annotation is the whole contract:

```python
class Result(Data):
    grain: ClassVar[Grain] = Grain.FIXTURE

    season: Annotated[Season, Agg.KEY]
    home_goals: Annotated[int | None, Agg.SUM]
```

`Dataset.aggregate` reads `Agg` off the annotation to collapse per-game rows into a season; `Dataset.to_frame` / `from_frame` are the only places a frame appears, and they round-trip. When a new column is needed, the *wrong* answers are: a side table of column-to-aggregation mappings, an `aggregate(records, how={"xg": "sum"})` override, or a function that takes a `DataFrame` and adds the column there. All of those are glue — a second schema to keep in sync, or a frame leaking into a signature.

The right answer is the one in the tree: add the field to the record with its `Agg` in the annotation, teach the source that carries it to fill it, and let `aggregate` do the rest. If that means every source and every fixture touching that record changes, that cost is accepted; a kaleidoscope of special cases is not.

The same shape applies to a new source. It subclasses `Source`, declares which record type it returns for each `Kind`, emits canonical refs through `fopol.identity`, and fetches through `cached_get`. It does not return a frame, keep its own alias table, or wrap a raw client in a class the models then have to know about.

## Existing concepts — extend these, don't invent parallels

- `fopol.base.data` — `Data`, `Agg`, `Grain`, `Ref`: the typed records everything is written against. A column's `Agg` annotation is its schema. Grain lives in the type: a per-fixture record and a season record are different classes, never one class with nullable keys.
- `fopol.base.dataset` — `Dataset[T]`: a homogeneous, provenance-carrying collection and the *only* bridge to pandas. Frames are allowed inside a function, never in a signature; a processing step is always `Data -> frame -> work -> Data`.
- `fopol.identity` — canonical team and player ids, resolved once at the source. No analysis code writes an alias table.
- `fopol.base.source` — `Source`, `Kind`, `cached_get`: a client knows one place data lives and how to translate its dialect into records. Per-game granularity always; seasons are derived by `Dataset.aggregate`. Fetches are cached on disk so a load is reproducible.
- `fopol.data` — the concrete records (`Result`, `PlayerStat`, `TeamStat`, and their season forms).
- `fopol.sources` — the clients (`FPLArchiveSource`, `FPLApiSource`, `UnderstatSource`). Every one returns `Dataset`s of records; nothing frame-based survives here.
- `fopol.base.model` — `Model`, the sklearn-shaped contract (`fit` returns `self`, fitted state ends in `_`, `predict` returns records that carry uncertainty, `score` is a mean log predictive density), and the four family bases: `TeamModel`, `MinutesModel`, `InvolvementModel`, `PointsModel`. A family base declares what it consumes and produces and does the record plumbing; a variant implements the family's abstract draws and nothing else.
- `fopol.models` — the variants, one subpackage per family (`team/`, `minutes/`, `involvement/`, `points/`). A new variant subclasses its family base so it can be pitted against its peers through `score` on the same data; hierarchical structure and shrinkage are the point, so a new model pools where its peers pool.
- `fopol.optim` — not in the tree right now. When the PuLP formulations return they consume `PointsPrediction` records, and every constraint is asserted independently in tests; an optimiser that quietly returns an illegal squad is worse than one that fails.

The top-level package deliberately does not import its submodules, so a broken corner cannot break every other import. Keep it that way.

## Conventions

- Doc comments state the contract and the *why*, not the mechanics — a module docstring explains what the module refuses to do and what breaks if it did; a class docstring explains why it is a class rather than a frame or a function.
- **No code separators.** Never add a banner comment that rules a file into sections, like this:

  ```python
  # ------------------------------------------------------------ contract
  ```

  A file that needs signposts is a file that should be split; names and docstrings are what make code findable. Do not add one to a new file, and do not add one when editing an old file that already has some — leave the existing ones alone.
- Formatting, linting and type-checking are non-negotiable and run as pre-commit hooks: `ruff` (lint with `--fix`, then format) and `ty`, plus the file-hygiene hooks in `.pre-commit-config.yaml`. `uv run pre-commit install` once per clone; `uv run pre-commit run --all-files` is exactly what the `pull-request` workflow runs, so a commit that passes locally passes CI.
- Markdown prose is never hard-wrapped: one paragraph is one source line, however long — the same for a list item and for a blockquote line. Editors and renderers do the wrapping; hard breaks mid-paragraph make the file read as compacted and ragged, and turn every later edit into a rewrap diff. Code blocks, tables and the GitHub issue templates keep their own line structure. This applies to README.md, this file, and every other prose .md in the tree.

### Python

- Test files are named `*_test.py` (not `test_*.py`), and the suite mirrors the package one-to-one: a module maps to exactly one test file under the same relative path, and that file holds the module's unit tests and its integration tests together.

  ```text
  fopol/                → tests/

  base/dataset.py       → base/dataset_test.py
  models/points/simulation.py → models/points/simulation_test.py
  sources/understat.py  → sources/understat_test.py
  ```

  **A test file is named after the module under test, never after the scenario it exercises.** `shrinkage_test.py`, `constraints_test.py`, `backtest_test.py` are all wrong names for tests of `models/points/simulation.py` or `base/source.py`: each describes a concern, and a concern is a *section inside* the mirrored file, not a file of its own, and the test names are what make those sections findable.

  Integration tests live in the same mirrored file as the module's unit tests, after the unit tests, and every one carries `@pytest.mark.integration`. Here "integration" means it leaves the machine: the FPL API, the GitHub archive mirror, Understat. `-m "not integration"` is the unit run and touches no network; `-m integration` needs it and a warm cache is not an excuse to drop the marker.

  The point is that the tests for a given module are findable from its path alone, without grepping. Three consequences are intended, not accidents:

  - The same basename recurs across subpackages (`data/team.py` and `models/team/` both map to a `team_test.py`). `--import-mode=importlib` in `pyproject.toml` is what stops pytest tripping over that. Never rename a file to dodge the collision.
  - A module with nothing to test at one level simply has no tests at that level: `identity.py` resolves names without touching the network, so it has unit tests and no `integration`-marked ones. A missing section is fine; a misnamed file is not.
  - `conftest.py` is fixture machinery and mirrors nothing.


- **Every fixture lives in `tests/conftest.py`, and a test module contains nothing but tests.** That means the one `conftest.py` at the root of the suite — not a per-directory one — holds every `@pytest.fixture`, every seed constant, every row-building helper (`_involvement_rows`, `_points_row`), and every concrete stand-in for an abstract base (`_StubTeam`, `_StubMinutes`, `_StubInvolvement`). A `*_test.py` file is a docstring, imports of the code under test, and `def test_*` functions; no `@pytest.fixture`, no module-level `_helper()`, no `class` of any kind. Setup in a test module is hiding among the assertions, and it splits "how did this frame get populated?" across as many files as there are modules. "Only this file uses it" is the reason a local fixture drifts, not an exemption. Private names in `conftest.py` (`_SEASON`, `_points_row`) are seed material for the fixtures beside them and are never reached for from a test; the fixture is the interface, injected through the test's argument list. A builder a test needs to call with its own arguments is a fixture that returns the callable (`minutes_row`, `unplayed_fixtures`), so it still arrives by injection. A fitted or composed object that several tests start from is its own fixture (`fitted_points_model` on top of `points_model`), not a helper each test calls.
- **Never import from a test module — a test tree is not a package.** `from conftest import NO_MATCH` is banned, and so is importing from a sibling `*_test.py`. Whether it resolves at all depends on how pytest put the directory on `sys.path`, and it re-introduces the hidden coupling that moving fixtures to `conftest.py` just removed. The only supported channel out of `conftest.py` is a fixture, injected through a test's arguments.

  This applies to plain values too, not just objects that need setup. Seasons, team names, scoring constants, row-building callables are all fixtures:

  ```python
  # conftest.py — the value is private; the fixture is the interface
  _UNKNOWN_TEAM = "Zzz Wanderers"


  @pytest.fixture(scope="session")
  def unknown_team():
      """A club name no alias table contains."""
      return _UNKNOWN_TEAM


  # identity_test.py
  def test_unknown_team_is_not_silently_canonicalised(unknown_team):
      """A miss must surface as a miss, never as a guess that joins to nothing."""
      assert not is_known_team(unknown_team)
  ```

  A test that reads a long argument list is telling you what it depends on, which is the point.
- **Every test has a docstring.** One line is enough when the name already says it; say more when the test pins something non-obvious — why this many games, why this threshold, what breaks if the assertion flips. Same rule as everywhere else in the tree: state the contract and the *why*, not the mechanics. A test that documents a known defect says so in its docstring, with a `NOTE:` and what should replace it once the defect is fixed.
- **Model tests keep inference tiny and assert on direction, not draws.** A handful of warmup and sample steps, a fixed seed, and assertions of the form "more data shrinks less" or "the favourite's win probability exceeds a half" — never an exact posterior mean. A test that only passes at one seed is testing the sampler, not the model.
- **`parametrize` values are written inline, at the test that uses them.** A literal list at the decorator, never a module-level `CASES` dict fed in as `CASES.values()` with `ids=CASES.keys()` — that reads as indirection and you have to scroll to find out what a case even is. Skip `pytest.param` unless a case genuinely needs a marker; auto-generated ids are a fine price for seeing the values at the point of use. If two tests want the same case list, they are usually one parametrized test.

  ```python
  @pytest.mark.parametrize(
      "raw",
      ["Man City", "Manchester City", "man-city", "MANCHESTER CITY"],
  )
  def test_every_spelling_of_a_club_resolves_to_one_id(raw):
      """Identity is resolved at the source; a join must never depend on the spelling."""
  ```

- **Mock with the `@patch` decorator, and nothing else.** The whole toolbox is one import and one form:

  ```python
  from unittest.mock import patch


  @patch("fopol.sources.fpl.cached_get")
  def test_archive_results_fetch_teams_then_fixtures(get, archive_pages, season):
      """Team ids resolve through ``teams.csv``, so it is read before ``fixtures.csv``."""
      get.side_effect = lambda url, **_: archive_pages.get(url)
  ```

  The decorator injects the mock as the first positional argument, before the fixtures; the test configures `return_value` / `side_effect` on it and asserts on it. `@patch.object(Cls, "attr")` is the same form for a method or class attribute. Not allowed: `with patch(...)` as a context manager, `patch.start()`, a bare `MagicMock()` / `AsyncMock()` / `Mock()` built in the test body, `create_autospec`, `mock_open`, `sentinel`, `monkeypatch`, or any other name from `unittest.mock`. A collaborator the unit takes as an argument is patched where the unit looks it up, not constructed by hand. And a test never defines a `_FakeResponse` / `_FakeSource` class to impersonate a collaborator — a hand-written stand-in silently keeps passing after the API it imitates has changed shape, which is exactly the failure the test existed to catch. The one exception is an abstract base: a concrete subclass that implements the family's abstract methods with constants (the `_Stub*` models in `tests/conftest.py`, reached only through the `points_model` fixture) is the contract exercised, not a fake of it — and like every other piece of setup it lives in `conftest.py`, never in a test module.
- **A unit test mocks every call that leaves the unit.** Anything that reaches an external system — `httpx`, the on-disk cache, a feed — is `@patch`ed or given a `tmp_path`, so unit tests need no network. Feed bodies a source parses are seed constants in `conftest.py` (`archive_pages`, `api_pages`, `understat_pages` map URL to body), and the patched `cached_get` looks them up, so the source's own URL construction is under test too. Isolation is the goal, not purity: mocking a *deterministic* collaborator is fine and often right, because it pins what the unit under test asked for instead of testing the collaborator a second time. Mocking a fitted team model to return a fixed goal rate is a better test of `PointsModel` than letting MCMC compute one. In-tree value objects (`Result`, `TeamRef`, `Dataset`) are cheap and stable, so just build them.
- **Assert on the mock, not on a bookkeeping structure.** `assert_called_once_with`, `call_args.kwargs`, `assert_not_called` — never a bespoke `record` dict or `calls` list. Since the mocks accept any signature, the call assertion is what pins the contract: check the arguments, not just that something was called.
- **No code in `__init__.py` — imports are the only exception.** A package's `__init__.py` holds its docstring, re-export imports and `__all__`, nothing else. Classes, functions, constants and type aliases live in a named module (`base/source.py`, not `base/__init__.py`), which the package then re-exports. Code in `__init__.py` has no module to mirror under the test layout above, forces submodules into import cycles with their own package, and hides itself from anyone reading the tree.
- Docstrings follow the **Google** style — a one-line summary, then `Args:` / `Returns:` / `Raises:` / `Attributes:` sections, as in `fopol/models/team/dixon_coles.py`. Not NumPy, not reST field lists. Enforced by `convention = "google"` in `pyproject.toml`; an `Args:` section that names some parameters must name them all.
- Optional extras import their library inside the function that needs it, never at module scope, so `import fopol.models` stays free of `arviz` and `matplotlib`. A type-only import goes under `if TYPE_CHECKING:`.
- `ruff` formats and lints; `ty` type-checks `fopol/` and `tests/` (`[tool.ty.src]` in `pyproject.toml`). Both are configured in `pyproject.toml` and both run from `.pre-commit-config.yaml`, which pins their versions; bump the hook `rev` and the `dev` extra together. Type errors are fixed in the code, not silenced: a numpyro sample is wrapped in `jnp.asarray` so it is an array to the checker, a record field takes the `Ref` it declares (`TeamRef(id="a")`, never the bare string the runtime validator would also accept), and an override keeps the base's parameter names.
- Docs are MkDocs with Material and mkdocstrings, published on Read the Docs. The API reference under `docs/api/` mirrors the package: a new public module gets a `::: fopol.<module>` directive on the page for its subpackage, and a new subpackage gets a page and a `nav` entry in `mkdocs.yml`. Math is written as `$...$` and `$$...$$` in docstrings and pages alike; a docstring that carries a backslash is a raw string.

## Commands

The environment is managed by `uv`; every command runs through it so the lockfile is what gets exercised.

- `uv sync --all-extras` — create or refresh `.venv` from `uv.lock`.
- `uv run pytest -m "not integration"` — unit tests; no network.
- `uv run pytest -m integration` — the tests that fetch from FPL, the archive and Understat.
- `uv run pre-commit run --all-files` — the lint gate: ruff (fixing), ruff format, ty, file hygiene. `uv run pre-commit install` makes it run on every commit.
- `uv run mkdocs build --strict` — the docs gate. Read the Docs builds the same `mkdocs.yml`; a broken cross-reference or a module missing from `docs/api/` fails here before it fails there.

When a change alters a contract (a record's fields or `Agg`, a `Source`'s declared record types, a model's posterior surface), update every implementor and every test that pins it in the same change — the suite pins these on purpose, and a failing pin is information, not an obstacle to sed away.

## Pull requests

Three conventions apply to every PR. The first two are what release-please reads, so they are enforced on every push.

- **Branch name**: `type/short-description` — `feat/understat-shots`, `fix/empty-archive-season`, `docs/agents-md`. Type is one of `feat` `fix` `docs` `style` `refactor` `perf` `test` `build` `ci` `chore` `revert`; the description is lowercase letters and digits separated by `-` or `.`, nothing else.
- **PR title**: Conventional Commits, `type(optional-scope): subject`. Subject lowercase, no trailing period, 3–60 characters. Type is one of `feat` `fix` `docs` `refactor` `perf` `test` `build` `ci` `chore` `deps` `revert`. Scope is the subpackage the change lands in — `(sources)`, `(models)`, `(optim)`, `(base)` — and is omitted for a change that spans them. A breaking change carries `!` after the type/scope.
- **PR description**: type of change, what it does, why (`Closes #N` goes here), whether it breaks anything, how it was tested, and notes for reviewers. Never skip a section — write "n/a" or "not tested — here's why".

When asked for PR artifacts, produce exactly the title and the description as above; branch creation and git are the author's.

## Releases: release-please

Versions, tags, the changelog and GitHub releases are all produced by release-please from the commit history on `main`. Nothing about a release is done by hand.

- **PRs are squash-merged, so the PR title becomes the commit release-please parses.** That is the whole reason the title format above is enforced: a title that does not parse is a change that never reaches the changelog, and a `feat` mislabelled `chore` is a release that never happens.
- **How a title maps to a version.** `fix` bumps patch, `feat` bumps minor, and a `!` after the type (or a `BREAKING CHANGE:` footer in the PR body) bumps major. While the version is below 1.0 the config sets `bump-minor-pre-major`, so a breaking change bumps minor rather than major and `feat` still bumps minor. `perf` and `deps` land in the changelog without bumping more than patch; `docs`, `refactor`, `test`, `build`, `ci` and `chore` are recorded but do not trigger a release on their own.
- **The version is owned by release-please.** It lives in two places — `version` in `pyproject.toml` and `__version__` in `fopol/__init__.py` — and release-please updates both from `release-please-config.json` (the `__init__` line carries the `# x-release-please-version` marker so the updater can find it). Never edit either by hand, never append `.devN`, and never let the two disagree: a hand bump makes the next release PR conflict, and a mismatch means `fopol.__version__` lies about what was installed.
- **Never edit `CHANGELOG.md`, and never create a tag.** The changelog is regenerated from commit titles on every release PR; a hand edit is overwritten. A tag pushed by hand desynchronises `.release-please-manifest.json` from git and the next release PR starts from the wrong base. To change what a release says, fix the commit titles it is built from.
- **The release PR is merged, not edited.** release-please keeps one open PR titled `chore(main): release x.y.z` that accumulates every releasable commit. Merging it is the release: release-please tags `vX.Y.Z` and writes the GitHub release, and that tag triggers `release.yaml`, which checks the tag against `pyproject.toml`, builds the wheel and sdist, runs the unit suite, and publishes to PyPI through trusted publishing (the `pypi` environment). Nothing else is pushed to it. To force a particular version, put a `Release-As: x.y.z` footer in the PR body of a normal change instead.
- **The tag is what publishes, and only release-please pushes tags.** `release.yaml` fires on `v[0-9]+.[0-9]+.[0-9]+` and nothing else. A tag pushed with the default `GITHUB_TOKEN` does not trigger other workflows, so `release-please.yaml` runs with the `RELEASE_PLEASE_TOKEN` repository secret — a fine-grained PAT with contents and pull-requests write on this repo. If that secret is missing the tag still lands but nothing publishes; run `release.yaml` by hand from the Actions tab with the tag as input rather than pushing anything.
- **Where the setup lives.** `release-please-config.json` (release type `python`, the extra-files entry for `fopol/__init__.py`, the changelog sections), `.release-please-manifest.json` (the last released version — the single source of truth for "what is on main"), `.github/workflows/release-please.yaml` (the release PR and the tag) and `.github/workflows/release.yaml` (everything the tag sets off). A change to any of these is a `ci` PR and touches nothing else.
