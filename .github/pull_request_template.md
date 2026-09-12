## Type of change

<!-- feat / fix / docs / refactor / perf / test / build / ci / chore / deps / revert -- the same type as the title -->

## Description

<!-- What the change does. -->

## Motivation

<!-- Why. `Closes #N` goes here. -->

## Breaking change

<!-- What breaks and how a caller migrates, or "n/a". A breaking change also carries `!` in the title. -->

## How it was tested

<!-- The commands run and what they showed, or "not tested -- here's why". -->

## Checklist

- [ ] `uv run pre-commit run --all-files` is clean
- [ ] `uv run pytest -m "not integration"` passes
- [ ] New or changed behaviour has tests in the mirrored `*_test.py`, every test has a docstring, every fixture is in `conftest.py`
- [ ] Docstrings state the contract and the why
- [ ] No version, changelog or tag edited by hand

## Notes for reviewers

<!-- Where to start, what to look hardest at, or "n/a". -->
