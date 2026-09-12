"""Understat via the archive mirror.

Career files are cut down to one PL season, and results are derived from them.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fopol.base.data import Position
from fopol.data.player import PlayerStat
from fopol.data.result import Result
from fopol.sources.fpl import FPLArchiveSource
from fopol.sources.understat import UnderstatSource, _position


@pytest.mark.parametrize(
    ("code", "fallback", "expected"),
    [
        ("GK", "GK", Position.GK),
        ("D", "D", Position.DEF),
        ("DR", "D", Position.DEF),
        ("DMC", "M", Position.MID),
        ("MC", "M", Position.MID),
        ("AMR", "F M", Position.MID),
        ("FW", "F", Position.FWD),
        ("Sub", "F M", Position.FWD),
        ("", "D", Position.DEF),
    ],
)
def test_position_codes_map_to_fpl_positions(code, fallback, expected):
    """Understat's fine-grained codes collapse to FPL's four.

    'Sub' carries no position, so it defers to the season position.
    """
    assert _position(code, fallback) is expected


def test_seasons_are_the_four_the_mirror_carries():
    """The mirror stops at 2024-25; asking for a later season should not even be offered."""
    assert UnderstatSource().seasons() == ("2021-22", "2022-23", "2023-24", "2024-25")


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_player_files_are_requested_from_the_index(
    get, get_many, understat_pages, archive_url, season
):
    """One URL per indexed player, named ``<name_with_underscores>_<id>.csv`` under the season."""
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]

    UnderstatSource(workers=2).appearances(season)

    get.assert_called_once_with(f"{archive_url}/{season}/understat/understat_player.csv")
    get_many.assert_called_once()
    assert get_many.call_args.args[0] == [
        f"{archive_url}/{season}/understat/Mohamed_Salah_1250.csv",
        f"{archive_url}/{season}/understat/Erling_Haaland_8260.csv",
        f"{archive_url}/{season}/understat/Nomad_Winger_6552.csv",
    ]
    assert get_many.call_args.kwargs == {"workers": 2}


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_career_rows_outside_the_season_and_league_are_dropped(
    get, get_many, understat_pages, season
):
    """A player file is a whole career.

    Keep this year's rows between two clubs in the season's index; drop the rest.
    """
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]

    rows = UnderstatSource().appearances(season)

    assert rows.record_type is PlayerStat and rows.source == "understat"
    assert [(r.player_name, r.fixture_id) for r in rows] == [
        ("Mohamed Salah", "26001"),
        ("Mohamed Salah", "26002"),
        ("Erling Haaland", "26001"),
    ]


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_a_player_listed_for_both_clubs_cannot_be_placed(get, get_many, understat_pages, season):
    """A mid-season transfer between the two clubs in a match has no side.

    The row is dropped, not guessed.
    """
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]

    assert "Nomad Winger" not in {r.player_name for r in UnderstatSource().appearances(season)}


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_side_is_worked_out_from_the_players_club(get, get_many, understat_pages, season):
    """Home and away are match facts; which side the player was on comes from the index."""
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]

    at_anfield, at_etihad, haaland = UnderstatSource().appearances(season)

    assert (str(at_anfield.team), str(at_anfield.opponent), at_anfield.is_home) == (
        "liverpool",
        "man-city",
        True,
    )
    assert (str(at_etihad.team), at_etihad.is_home) == ("liverpool", False)
    assert (str(haaland.team), haaland.is_home) == ("man-city", False)


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_appearance_fields_carry_shots_and_key_passes(get, get_many, understat_pages, season):
    """The fields FPL lacks are the reason this source exists; 'Sub' means not started."""
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]

    started, sub, _ = UnderstatSource().appearances(season)

    assert (started.shots, started.key_passes, started.goals, started.assists) == (4, 2, 1, 1)
    assert started.xg == pytest.approx(0.9) and started.xa == pytest.approx(0.3)
    assert started.started == 1 and started.position is Position.MID
    assert sub.started == 0 and sub.position is Position.FWD
    assert started.kickoff == pd.Timestamp("2024-12-01", tz="UTC")


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_results_are_one_per_match_with_xg_summed_by_side(get, get_many, understat_pages, season):
    """Every player row repeats the match; results dedupe on match id and sum each side's xG."""
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]

    results = UnderstatSource().results(season)

    assert results.record_type is Result
    assert [r.fixture_id for r in results] == ["26001", "26002"]
    anfield = results[0]
    assert (str(anfield.home), str(anfield.away)) == ("liverpool", "man-city")
    assert (anfield.home_goals, anfield.away_goals) == (2, 0)
    assert anfield.home_xg == pytest.approx(0.9) and anfield.away_xg == pytest.approx(0.7)


@patch("fopol.sources.understat.cached_get_many")
@patch("fopol.sources.understat.cached_get")
def test_results_and_appearances_share_one_parse(get, get_many, understat_pages, season):
    """Asking for both kinds fetches the ~800 player files once, not twice."""
    get.side_effect = lambda url, **_: understat_pages.get(url)
    get_many.side_effect = lambda urls, **_: [understat_pages.get(u) for u in urls]
    source = UnderstatSource()

    source.results(season)
    source.appearances(season)

    get_many.assert_called_once()


@patch("fopol.sources.understat.cached_get")
def test_missing_index_is_an_error(get, season):
    """No index means no season on the mirror; that is raised, not returned as an empty dataset."""
    get.return_value = None

    with pytest.raises(FileNotFoundError, match="understat index"):
        UnderstatSource().results(season)


@pytest.mark.integration
def test_mirror_serves_most_of_a_season(season):
    """The mirror's snapshot stops short of the season's end, so expect nearly, not exactly, 380.

    A couple of matches were mirrored before Understat had processed them and
    carry zero xG on both sides, so xG is asserted on almost every match, not all.
    """
    source = UnderstatSource()

    results = source.results(season)
    appearances = source.appearances(season)

    assert 300 <= len(results) <= 380
    assert len({str(r.home) for r in results}) == 20
    with_xg = sum(1 for r in results if r.home_xg is not None and r.home_xg > 0)
    assert with_xg >= 0.98 * len(results)
    assert len(appearances) > 8_000
    assert all(r.kickoff is not None for r in appearances)


@pytest.mark.integration
def test_scorelines_agree_with_the_fpl_archive(season):
    """Two feeds, one league.

    Every match Understat carries has the same scoreline in FPL's fixtures.
    """
    fpl = {
        (str(r.home), str(r.away)): (r.home_goals, r.away_goals)
        for r in FPLArchiveSource().results(season)
    }

    understat = UnderstatSource().results(season)

    disagreements = [
        r for r in understat if fpl[(str(r.home), str(r.away))] != (r.home_goals, r.away_goals)
    ]
    assert disagreements == []
