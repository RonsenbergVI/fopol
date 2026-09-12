"""FPL sources: the archive mirror and the live API, parsed through one row translator."""

from unittest.mock import patch

import pandas as pd
import pytest

from fopol.base.data import Position
from fopol.base.source import Kind
from fopol.data.player import PlayerStat
from fopol.data.result import Result
from fopol.sources.fpl import FPLApiSource, FPLArchiveSource


def test_archive_seasons_span_first_to_last_inclusive():
    """Seasons are generated from the bounds, in archive form, oldest first."""
    assert FPLArchiveSource(first_season="2022-23", last_season="2024-25").seasons() == (
        "2022-23",
        "2023-24",
        "2024-25",
    )


@patch("fopol.sources.fpl.cached_get")
def test_archive_results_are_canonical_and_keep_unplayed_fixtures(get, archive_pages, season):
    """Every fixture becomes a Result with canonical ids.

    An unplayed one has ``None`` goals, not zero.
    """
    get.side_effect = lambda url, **_: archive_pages.get(url)

    results = FPLArchiveSource().results(season)

    assert results.record_type is Result and results.source == "fpl-archive"
    assert [str(r.home) for r in results] == ["arsenal", "nottm-forest", "man-city"]
    played = results[0]
    assert (played.home_goals, played.away_goals, played.gameweek) == (1, 0, 1)
    assert played.kickoff == pd.Timestamp("2024-08-16T19:00:00Z")
    unscheduled = results[2]
    assert not unscheduled.played
    assert unscheduled.gameweek is None and unscheduled.kickoff is None


@patch("fopol.sources.fpl.cached_get")
def test_archive_results_fetch_teams_then_fixtures(get, archive_pages, archive_url, season):
    """Team ids resolve through ``teams.csv``, so it is read before ``fixtures.csv``."""
    get.side_effect = lambda url, **_: archive_pages.get(url)

    FPLArchiveSource().results(season)

    urls = [c.args[0] for c in get.call_args_list]
    assert urls == [f"{archive_url}/{season}/teams.csv", f"{archive_url}/{season}/fixtures.csv"]


@patch("fopol.sources.fpl.cached_get")
def test_archive_appearances_drop_managers_and_duplicates(get, archive_pages, season):
    """The archive repeats a player on a fixture and lists managers as 'AM'.

    Neither is a PlayerStat.
    """
    get.side_effect = lambda url, **_: archive_pages.get(url)

    rows = FPLArchiveSource().appearances(season)

    assert rows.record_type is PlayerStat
    assert [r.player_name for r in rows] == ["Bukayo Saka", "Erling Haaland", "Chris Wood"]


@patch("fopol.sources.fpl.cached_get")
def test_archive_appearance_fields_translate_the_dialect(get, archive_pages, season):
    """Price is tenths, opponent is a team id, GW is the gameweek, kickoff is UTC."""
    get.side_effect = lambda url, **_: archive_pages.get(url)

    saka = FPLArchiveSource().appearances(season)[0]

    assert saka.position is Position.MID
    assert (str(saka.team), str(saka.opponent), saka.is_home) == ("arsenal", "man-city", True)
    assert (saka.minutes, saka.started, saka.goals, saka.total_points) == (90, 1, 1, 12)
    assert saka.price == pytest.approx(10.0)
    assert saka.ownership == pytest.approx(45.2)
    assert saka.gameweek == 1
    assert saka.kickoff == pd.Timestamp("2024-08-16T19:00:00Z")
    assert saka.band == 2


@patch("fopol.sources.fpl.cached_get")
def test_archive_unplayed_row_has_zero_minutes_not_none(get, archive_pages, season):
    """An archive row exists only for a fixture that happened, so zero minutes means benched."""
    get.side_effect = lambda url, **_: archive_pages.get(url)

    wood = FPLArchiveSource().appearances(season)[-1]

    assert wood.minutes == 0 and wood.played and wood.band == 0


@patch("fopol.sources.fpl.cached_get")
def test_archive_missing_season_is_an_error_not_an_empty_dataset(get, season):
    """A 404 on ``teams.csv`` raises; silently returning nothing would fit a model to no data."""
    get.return_value = None

    with pytest.raises(FileNotFoundError, match="teams.csv"):
        FPLArchiveSource().results(season)


@patch("fopol.sources.fpl.cached_get")
def test_api_season_comes_from_the_first_deadline(get, api_pages):
    """The live API has one season, named from the year of the first deadline."""
    get.side_effect = lambda url, **_: api_pages.get(url)

    assert FPLApiSource().seasons() == ("2025-26",)


@patch("fopol.sources.fpl.cached_get")
def test_api_bootstrap_is_fetched_fresh_and_once(get, api_pages, api_url):
    """Bootstrap changes under the same URL, so it bypasses the cache.

    Still one request per client: the answer is memoised for the client's life.
    """
    get.side_effect = lambda url, **_: api_pages.get(url)
    source = FPLApiSource()

    source.seasons()
    source.seasons()

    get.assert_called_once_with(f"{api_url}/bootstrap-static/", refresh=True)


@patch("fopol.sources.fpl.cached_get")
def test_api_results_translate_like_the_archive(get, api_pages):
    """The fixtures endpoint is the archive's ``fixtures.csv`` as JSON.

    The same parser handles both.
    """
    get.side_effect = lambda url, **_: api_pages.get(url)

    results = FPLApiSource().results("2025-26")

    assert results.source == "fpl-api"
    assert [(str(r.home), str(r.away)) for r in results] == [
        ("arsenal", "man-city"),
        ("man-city", "arsenal"),
    ]
    assert (results[0].home_goals, results[0].away_goals) == (2, 2)
    assert results[1].home_goals is None and results[1].kickoff is None


@patch("fopol.sources.fpl.cached_get_many")
@patch("fopol.sources.fpl.cached_get")
def test_api_appearances_join_history_to_the_element(get, get_many, api_pages, api_url):
    """Per-player history carries no name, club, position or ownership.

    Those come from bootstrap, joined on the element id.
    """
    get.side_effect = lambda url, **_: api_pages.get(url)
    get_many.side_effect = lambda urls, **_: [api_pages.get(u) for u in urls]

    rows = FPLApiSource().appearances("2025-26")

    get_many.assert_called_once()
    assert get_many.call_args.args[0] == [
        f"{api_url}/element-summary/401/",
        f"{api_url}/element-summary/999/",
    ]
    assert len(rows) == 1
    saka = rows[0]
    assert (saka.player_name, str(saka.team), saka.position) == ("Bukayo Saka", "arsenal", "MID")
    assert saka.ownership == pytest.approx(45.2)
    assert saka.gameweek == 1 and saka.price == pytest.approx(10.2)
    assert saka.defensive_contribution == 4


@patch("fopol.sources.fpl.cached_get_many")
@patch("fopol.sources.fpl.cached_get")
def test_api_skips_a_player_whose_summary_is_missing(get, get_many, api_pages):
    """A 404 element-summary is skipped rather than raising.

    The roster is not the request's to fix.
    """
    get.side_effect = lambda url, **_: api_pages.get(url)
    get_many.side_effect = lambda urls, **_: [api_pages.get(u) for u in urls]

    assert [r.player_name for r in FPLApiSource().appearances("2025-26")] == ["Bukayo Saka"]


@pytest.mark.integration
def test_archive_serves_a_full_season(season):
    """A finished season is 380 fixtures over 20 clubs, and every club's players appear."""
    source = FPLArchiveSource()

    results = source.results(season)
    appearances = source.appearances(season)

    assert len(results) == 380
    assert all(r.played for r in results)
    assert len({str(r.home) for r in results}) == 20
    assert {str(r.team) for r in appearances} == {str(r.home) for r in results}
    assert len(appearances) > 20_000
    assert appearances.record_type is source.provides[Kind.APPEARANCES]


@pytest.mark.integration
def test_archive_appearances_agree_with_results_on_the_scoreline(season):
    """Goals per side summed over a fixture's players equal the scoreline in ``fixtures.csv``."""
    source = FPLArchiveSource()
    results = {r.fixture_id: r for r in source.results(season)}

    scored: dict[tuple[str, str], int] = {}
    for row in source.appearances(season):
        key = (row.fixture_id, str(row.team))
        scored[key] = scored.get(key, 0) + row.goals

    mismatches = [
        fid
        for fid, r in results.items()
        if scored.get((fid, str(r.home)), 0) + scored.get((fid, str(r.away)), 0)
        != r.home_goals + r.away_goals
    ]
    # Own goals are credited to no scorer, so a handful of fixtures legitimately differ.
    assert len(mismatches) < 40


@pytest.mark.integration
def test_api_serves_the_season_in_progress():
    """The live API names one season and returns fixtures for it under the declared record type."""
    source = FPLApiSource()

    (season,) = source.seasons()
    results = source.results(season)

    assert len(results) == 380
    assert results.record_type is source.provides[Kind.RESULTS]
    assert len({str(r.home) for r in results}) == 20
