"""Fixtures for the fopol suite.

Every fixture lives here, including the seed data it is built from, so "how did
this frame get populated?" has exactly one answer. Values are private; the
fixture is the interface, and a test's argument list is its dependency list.
"""

import json
from collections.abc import Callable

import numpy as np
import numpyro
import pandas as pd
import pytest

from fopol.base.dataset import Dataset
from fopol.base.model import Inference, InvolvementModel, MinutesModel, TeamModel
from fopol.data.player import PlayerRef, PlayerStat
from fopol.data.result import Result
from fopol.data.team import TeamRef
from fopol.models import DixonColesTeamModel, SequentialMinutesModel, SimulatedPointsModel

numpyro.set_host_device_count(4)


# -- models.team ---------------------------------------------------------------

_TEAMS = ["Strong", "Average", "Weak"]
_TRUE_TEAM_EFFECTS = {
    "attack": {"Strong": 0.5, "Average": 0.0, "Weak": -0.5},
    "defence": {"Strong": 0.4, "Average": 0.0, "Weak": -0.4},
    "home_advantage": 0.3,
    "intercept": 0.1,
}


@pytest.fixture(scope="session")
def true_team_effects() -> dict:
    """The generative parameters behind ``synthetic_matches``, to check recovery against."""
    return _TRUE_TEAM_EFFECTS


@pytest.fixture(scope="session")
def synthetic_matches(true_team_effects: dict) -> Dataset[Result]:
    """720 played results simulated from the team model's own generative process.

    A week apart, so time decay has something to bite on.
    """
    rng = np.random.default_rng(0)
    attack, defence = true_team_effects["attack"], true_team_effects["defence"]
    home_adv, intercept = true_team_effects["home_advantage"], true_team_effects["intercept"]
    refs = {t: TeamRef(id=t) for t in _TEAMS}
    results: list[Result] = []
    week = 0

    for _ in range(120):  # 120 round-robins -> 720 matches
        for h in range(3):
            for a in range(3):
                if h == a:
                    continue
                lam_h = np.exp(intercept + home_adv + attack[_TEAMS[h]] - defence[_TEAMS[a]])
                lam_a = np.exp(intercept + attack[_TEAMS[a]] - defence[_TEAMS[h]])
                week += 1
                results.append(
                    Result(
                        season="2024-25",
                        fixture_id=str(week),
                        kickoff=pd.Timestamp("2000-01-01", tz="UTC") + pd.Timedelta(days=7 * week),
                        home=refs[_TEAMS[h]],
                        away=refs[_TEAMS[a]],
                        home_goals=int(rng.poisson(lam_h)),
                        away_goals=int(rng.poisson(lam_a)),
                    )
                )
    return Dataset(results, record_type=Result, source="synthetic")


@pytest.fixture(scope="session")
def fitted_team_model(synthetic_matches: Dataset[Result]) -> DixonColesTeamModel:
    """The team model fitted once on the synthetic matches; chains kept small for speed."""
    return DixonColesTeamModel(
        inference=Inference(num_warmup=500, num_samples=500, num_chains=2, seed=0)
    ).fit(synthetic_matches)


# -- models.minutes ------------------------------------------------------------

_ROLES = {"nailed": 0.95, "rotation": 0.55, "fringe": 0.12}


@pytest.fixture(scope="session")
def synthetic_appearances() -> Dataset[PlayerStat]:
    """Players with persistent roles -- nailed, rotation, fringe -- over 30 fixtures.

    Appearances are drawn independently given the role, so recent form carries
    no information beyond the role itself; that is what makes the player-effect
    ablation testable.
    """
    rng = np.random.default_rng(0)
    rows: list[PlayerStat] = []
    for role, p_play in _ROLES.items():
        for n in range(8):
            name = f"{role}-{n}"
            for gw in range(1, 31):
                plays = rng.random() < p_play
                minutes = 0 if not plays else (90 if rng.random() < 0.8 else 30)
                rows.append(
                    PlayerStat(
                        season="2024-25",
                        fixture_id=f"f{gw}",
                        player=PlayerRef(id=name),
                        player_name=name,
                        team=TeamRef(id="anytown"),
                        opponent=TeamRef(id="elsewhere"),
                        position="MID",
                        is_home=gw % 2 == 0,
                        kickoff=pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(days=7 * gw),
                        minutes=minutes,
                        started=int(minutes > 45),
                        price=5.0 + 4.0 * p_play,
                    )
                )
    return Dataset(rows, record_type=PlayerStat, source="synthetic")


@pytest.fixture(scope="session")
def minutes_row() -> Callable[..., PlayerStat]:
    """Builder for one midfielder's row: ``minutes_row(name, gw, minutes, season=, day0=)``.

    Kickoffs are a week apart from ``day0``, so a second season started from a
    later ``day0`` opens after a gap long enough to trip the staleness gate.
    """

    def build(
        name: str, gw: int, minutes: int | None, season: str = "2024-25", day0: str = "2024-08-01"
    ) -> PlayerStat:
        return PlayerStat(
            season=season,
            fixture_id=f"{season}-{gw}",
            player=PlayerRef(id=name),
            player_name=name,
            team=TeamRef(id="a"),
            opponent=TeamRef(id="b"),
            position="MID",
            is_home=True,
            kickoff=pd.Timestamp(day0, tz="UTC") + pd.Timedelta(days=7 * gw),
            minutes=minutes,
            price=5.0,
        )

    return build


@pytest.fixture(scope="session")
def fitted_minutes_model(synthetic_appearances: Dataset[PlayerStat]) -> SequentialMinutesModel:
    """The minutes model fitted once by SVI on the synthetic roster."""
    return SequentialMinutesModel(
        inference=Inference(method="svi", num_steps=4000, num_samples=200, seed=0)
    ).fit(synthetic_appearances)


@pytest.fixture(scope="session")
def unplayed_fixtures() -> Callable[..., Dataset[Result]]:
    """Builder for scheduled-but-unplayed results between named teams: ``unplayed_fixtures(("A", "B"), ...)``."""

    def build(*pairs: tuple[str, str]) -> Dataset[Result]:
        return Dataset(
            [
                Result(
                    season="2024-25",
                    fixture_id=f"x{i}",
                    kickoff=pd.Timestamp("2001-01-01", tz="UTC"),
                    home=TeamRef(id=h),
                    away=TeamRef(id=a),
                )
                for i, (h, a) in enumerate(pairs)
            ],
            record_type=Result,
        )

    return build


def _involvement_rows(name, position, goals, assists, games, minutes=90):
    return [
        PlayerStat(
            season="2024-25",
            fixture_id=f"{name}-{g}",
            player=PlayerRef(id=name),
            player_name=name,
            team=TeamRef(id="a"),
            opponent=TeamRef(id="b"),
            position=position,
            is_home=True,
            kickoff=pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(days=7 * g),
            minutes=minutes,
            goals=goals,
            assists=assists,
        )
        for g in range(games)
    ]


@pytest.fixture(scope="session")
def share_row() -> Callable[..., PlayerStat]:
    """Builder for one unplayed row: ``share_row(name, position, penalties_order=None, ...)``."""

    def build(name: str, position: str, **fields) -> PlayerStat:
        return PlayerStat(
            season="2024-25",
            fixture_id="x",
            player=PlayerRef(id=name),
            player_name=name,
            team=TeamRef(id="anytown"),
            opponent=TeamRef(id="elsewhere"),
            position=position,
            is_home=True,
            kickoff=pd.Timestamp("2025-01-01", tz="UTC"),
            **fields,
        )

    return build


def _share_rows(name, team, position, goals_per_game, games, **fields):
    return [
        PlayerStat(
            season="2024-25",
            fixture_id=f"{team}-{g}",
            player=PlayerRef(id=name),
            player_name=name,
            team=TeamRef(id=team),
            opponent=TeamRef(id="elsewhere"),
            position=position,
            is_home=True,
            kickoff=pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(days=7 * g),
            minutes=90,
            goals=goals_per_game,
            **fields,
        )
        for g in range(games)
    ]


@pytest.fixture(scope="session")
def share_history() -> Dataset[PlayerStat]:
    """Two clubs over 30 fixtures: a low-scoring one carried by its talisman, a big one that spreads goals.

    ``lowtown`` scores one a game and the talisman gets it every time; ``bigcity``
    scores three a game shared by three players. ``blank-taker`` is a first-choice
    penalty taker at bigcity who never scores.
    """
    rows = _share_rows("talisman", "lowtown", "FWD", 1, 30) + _share_rows(
        "lowtown-mid", "lowtown", "MID", 0, 30
    )
    for name in ("big-club-striker", "big-club-winger", "big-club-mid"):
        rows += _share_rows(name, "bigcity", "FWD", 1, 30)
    rows += _share_rows("blank-taker", "bigcity", "MID", 0, 30, penalties_order=1)
    return Dataset(rows, record_type=PlayerStat)


@pytest.fixture(scope="session")
def involvement_history() -> Dataset[PlayerStat]:
    """Four players with very different output: a prolific, a blank and a lucky-sub forward, a playmaker."""
    rows = (
        _involvement_rows("prolific", "FWD", 1, 0, 60)
        + _involvement_rows("average", "FWD", 0, 0, 60)
        + _involvement_rows("lucky-sub", "FWD", 1, 0, 1, minutes=10)
        + _involvement_rows("playmaker", "MID", 0, 1, 60)
    )
    return Dataset(rows, record_type=PlayerStat)


class _StubTeam(TeamModel):
    """Constant goal rates, so the points arithmetic is checked without a sampler."""

    def __init__(self, home=1.6, away=1.1):
        self.home, self.away = home, away

    def _fit(self, data):
        self.teams_ = []

    def goal_rate_draws(self, home, away):
        return np.full((10, len(home)), self.home), np.full((10, len(away)), self.away)


class _StubMinutes(MinutesModel):
    """Fixed band probabilities per player."""

    def __init__(self, probs):
        self.probs = probs

    def _fit(self, data):
        self.history_, self.players_ = [], ()

    def features_for(self, data):
        return data

    def band_probs(self, features):
        return np.array([self.probs[str(r.player)] for r in features])

    def observed_bands(self, features):
        return np.array([-1 if r.band is None else r.band for r in features])


class _StubInvolvement(InvolvementModel):
    """Fixed per-90 rates per player."""

    def __init__(self, g, a):
        self.g, self.a = g, a

    def _fit(self, data):
        self.rates_ = {}

    def rates(self, data):
        return np.array([self.g[str(r.player)] for r in data]), np.array(
            [self.a[str(r.player)] for r in data]
        )


def _points_row(name, position, is_home=True):
    return PlayerStat(
        season="2024-25",
        fixture_id="f1",
        player=PlayerRef(id=name),
        player_name=name,
        team=TeamRef(id="a"),
        opponent=TeamRef(id="b"),
        position=position,
        is_home=is_home,
        kickoff=pd.Timestamp("2024-08-17", tz="UTC"),
    )


@pytest.fixture
def points_rows() -> Dataset[PlayerStat]:
    """A striker, a keeper and a fringe midfielder in one home fixture."""
    return Dataset(
        [_points_row("striker", "FWD"), _points_row("keeper", "GK"), _points_row("fringe", "MID")],
        record_type=PlayerStat,
    )


@pytest.fixture
def points_model() -> SimulatedPointsModel:
    """A points model composed from stub sub-models; nothing fitted yet."""
    nailed, bench = [0.05, 0.05, 0.90], [0.85, 0.10, 0.05]
    return SimulatedPointsModel(
        team=_StubTeam(),
        minutes=_StubMinutes({"striker": nailed, "keeper": nailed, "fringe": bench}),
        involvement=_StubInvolvement(
            {"striker": 0.8, "keeper": 0.0, "fringe": 0.2},
            {"striker": 0.2, "keeper": 0.0, "fringe": 0.2},
        ),
        n_draws=3000,
    )


@pytest.fixture
def fitted_points_model(
    points_model: SimulatedPointsModel, points_rows: Dataset[PlayerStat]
) -> SimulatedPointsModel:
    """``points_model`` with its stubs and itself fitted on ``points_rows``."""
    points_model.team._fit(points_rows)
    points_model.minutes._fit(points_rows)
    points_model.involvement._fit(points_rows)
    return points_model.fit(points_rows)


_SEASON = "2024-25"
_ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
_API = "https://fantasy.premierleague.com/api"

_TEAMS_CSV = """code,id,name,short_name
3,1,Arsenal,ARS
43,13,Man City,MCI
17,17,Nott'm Forest,NFO
"""

_FIXTURES_CSV = """code,event,finished,id,kickoff_time,team_a,team_a_score,team_h,team_h_score
2444470,1,True,1,2024-08-16T19:00:00Z,13,0,1,1
2444471,2,False,2,2024-08-24T14:00:00Z,1,,17,
2444472,,False,3,,17,,13,
"""

_MERGED_GW_CSV = """name,position,team,element,fixture,opponent_team,was_home,kickoff_time,minutes,starts,goals_scored,assists,expected_goals,expected_assists,goals_conceded,clean_sheets,saves,defensive_contribution,bps,bonus,yellow_cards,red_cards,own_goals,total_points,value,selected_by_percent,GW
Bukayo Saka,MID,Arsenal,401,1,13,True,2024-08-16T19:00:00Z,90,1,1,0,0.45,0.12,0,1,0,2,38,3,0,0,0,12,100,45.2,1
Bukayo Saka,MID,Arsenal,401,1,13,True,2024-08-16T19:00:00Z,90,1,1,0,0.45,0.12,0,1,0,2,38,3,0,0,0,12,100,45.2,1
Erling Haaland,FWD,Man City,355,1,1,False,2024-08-16T19:00:00Z,90,1,0,0,0.80,0.05,1,0,0,0,10,0,1,0,0,2,151,60.1,1
Mikel Arteta,AM,Arsenal,700,1,13,True,2024-08-16T19:00:00Z,90,0,0,0,,,0,0,0,0,0,0,0,0,0,6,15,1.0,1
Chris Wood,FWD,Nott'm Forest,447,2,1,True,2024-08-24T14:00:00Z,0,0,0,0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,62,8.3,2
"""

_BOOTSTRAP = {
    "events": [{"id": 1, "deadline_time": "2025-08-15T17:30:00Z"}],
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 13, "name": "Man City"}],
    "elements": [
        {
            "id": 401,
            "first_name": "Bukayo",
            "second_name": "Saka",
            "team": 1,
            "element_type": 3,
            "selected_by_percent": "45.2",
            "now_cost": 102,
        },
        {
            "id": 999,
            "first_name": "Gone",
            "second_name": "Player",
            "team": 13,
            "element_type": 4,
            "selected_by_percent": "0.1",
            "now_cost": 45,
        },
    ],
}

_API_FIXTURES = [
    {
        "id": 1,
        "event": 1,
        "kickoff_time": "2025-08-16T19:00:00Z",
        "team_h": 1,
        "team_a": 13,
        "team_h_score": 2,
        "team_a_score": 2,
    },
    {
        "id": 2,
        "event": None,
        "kickoff_time": None,
        "team_h": 13,
        "team_a": 1,
        "team_h_score": None,
        "team_a_score": None,
    },
]

_SAKA_SUMMARY = {
    "history": [
        {
            "fixture": 1,
            "opponent_team": 13,
            "was_home": True,
            "kickoff_time": "2025-08-16T19:00:00Z",
            "round": 1,
            "minutes": 77,
            "starts": 1,
            "goals_scored": 1,
            "assists": 1,
            "expected_goals": "0.31",
            "expected_assists": "0.22",
            "goals_conceded": 2,
            "clean_sheets": 0,
            "saves": 0,
            "defensive_contribution": 4,
            "bps": 41,
            "bonus": 3,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "total_points": 13,
            "value": 102,
        }
    ]
}

_UNDERSTAT_INDEX_CSV = """id,player_name,games,time,goals,xG,assists,xA,shots,key_passes,position,team_title
1250,Mohamed Salah,2,180,2,1.5,1,0.4,7,3,F M,Liverpool
8260,Erling Haaland,1,90,0,0.7,0,0.0,4,0,F,Manchester City
6552,Nomad Winger,1,90,0,0.1,0,0.0,1,0,M,"Liverpool,Manchester City"
"""

# Salah's file is his career: a 2024 PL row, a 2024 cup row against a club
# outside the league index, and a 2014 row that predates the season.
_SALAH_CSV = """goals,shots,xG,time,position,h_team,a_team,h_goals,a_goals,date,id,season,xA,assists,key_passes
1,4,0.9,90,AMR,Liverpool,Manchester City,2,0,2024-12-01,26001,2024,0.3,1,2
1,3,0.6,90,Sub,Manchester City,Liverpool,1,1,2025-02-23,26002,2024,0.1,0,1
0,2,0.4,90,FW,Liverpool,Plymouth,0,1,2025-02-09,26500,2024,0.0,0,0
0,1,0.06,6,Sub,Chelsea,Swansea,4,2,2014-09-13,4720,2014,0,0,0
"""

_HAALAND_CSV = """goals,shots,xG,time,position,h_team,a_team,h_goals,a_goals,date,id,season,xA,assists,key_passes
0,4,0.7,90,FW,Liverpool,Manchester City,2,0,2024-12-01,26001,2024,0.0,0,0
"""

# Listed for both clubs, so neither side of a match between them can be his.
_NOMAD_CSV = """goals,shots,xG,time,position,h_team,a_team,h_goals,a_goals,date,id,season,xA,assists,key_passes
0,1,0.1,90,MC,Liverpool,Manchester City,2,0,2024-12-01,26001,2024,0.0,0,0
"""


@pytest.fixture(scope="session")
def season() -> str:
    """The season every source fixture is keyed on."""
    return _SEASON


@pytest.fixture(scope="session")
def archive_url() -> str:
    """Root of the vaastav archive mirror, as the sources build URLs from it."""
    return _ARCHIVE


@pytest.fixture(scope="session")
def archive_pages(archive_url: str) -> dict[str, str]:
    """URL -> body for a three-club, three-fixture slice of the FPL archive.

    Saka appears twice on one fixture (the archive's known duplicate), Arteta is
    a manager row, and one fixture has no kickoff yet; the parser must survive
    all three.
    """
    root = f"{archive_url}/{_SEASON}"
    return {
        f"{root}/teams.csv": _TEAMS_CSV,
        f"{root}/fixtures.csv": _FIXTURES_CSV,
        f"{root}/gws/merged_gw.csv": _MERGED_GW_CSV,
    }


@pytest.fixture(scope="session")
def api_url() -> str:
    """Root of the live FPL API."""
    return _API


@pytest.fixture(scope="session")
def api_pages(api_url: str) -> dict[str, str | None]:
    """URL -> JSON body for a two-player live API; one element-summary is a 404 (``None``)."""
    return {
        f"{api_url}/bootstrap-static/": json.dumps(_BOOTSTRAP),
        f"{api_url}/fixtures/": json.dumps(_API_FIXTURES),
        f"{api_url}/element-summary/401/": json.dumps(_SAKA_SUMMARY),
        f"{api_url}/element-summary/999/": None,
    }


_MIRROR = "https://raw.githubusercontent.com/TopMarx/fpl/main/data"

_MIRROR_LIVE_CSV = """gw,fpl_id,minutes,starts,goals_scored,assists,clean_sheets,goals_conceded,own_goals,yellow_cards,red_cards,saves,bonus,bps,defensive_contribution,total_points
1,401,77,1,1,1,0,2,0,0,0,0,3,41,4,13
1,999,0,0,0,0,0,0,0,0,0,0,0,0,0,0
1,555,90,1,0,0,0,0,0,0,0,0,0,0,0,0
"""


@pytest.fixture(scope="session")
def mirror_url() -> str:
    """Root of the TopMarx/fpl mirror."""
    return _MIRROR


@pytest.fixture(scope="session")
def mirror_pages(mirror_url: str) -> dict[str, str]:
    """URL -> body for a one-gameweek mirror: the API bootstrap and fixtures plus ``live.csv``.

    The CSV carries an unknown ``fpl_id`` (555) that must be skipped, and the
    fixtures are the live-API ones, so a row is joined to its fixture by club
    and gameweek.
    """
    root = f"{mirror_url}/2025"
    return {
        f"{root}/fpl-bootstrap_2025.json": json.dumps(_BOOTSTRAP),
        f"{root}/fpl-fixtures_2025.json": json.dumps(_API_FIXTURES),
        f"{root}/csv/live.csv": _MIRROR_LIVE_CSV,
    }


@pytest.fixture(scope="session")
def understat_pages(archive_url: str) -> dict[str, str]:
    """URL -> body for a three-player Understat mirror over one league match plus noise."""
    root = f"{archive_url}/{_SEASON}/understat"
    return {
        f"{root}/understat_player.csv": _UNDERSTAT_INDEX_CSV,
        f"{root}/Mohamed_Salah_1250.csv": _SALAH_CSV,
        f"{root}/Erling_Haaland_8260.csv": _HAALAND_CSV,
        f"{root}/Nomad_Winger_6552.csv": _NOMAD_CSV,
    }
