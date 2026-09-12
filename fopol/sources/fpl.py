"""FPL data: the community archive, and the live API it mirrors.

Two clients over the same dialect. :class:`FPLArchiveSource` reads the
vaastav/Fantasy-Premier-League CSV mirror from GitHub -- every season back to
2016-17, refreshed a few times a season, and the only thing reachable from a
restricted network. :class:`FPLApiSource` reads ``fantasy.premierleague.com``
directly for the season in progress. They share one row parser, because the
archive is a dump of the API and the columns match.

What FPL carries per player per fixture: minutes, goals, assists, xG, xA, BPS,
bonus, cards, saves, clean sheets, defensive contribution, price, and (live
only) ownership. What it lacks: shots and key passes -- see
:mod:`fopol.sources.understat` for those.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from fopol.base.data import Position
from fopol.base.dataset import Dataset
from fopol.base.source import Kind, Source, cached_get, cached_get_many
from fopol.data.player import PlayerRef, PlayerStat
from fopol.data.result import Result
from fopol.data.team import TeamRef

__all__ = ["FPLApiSource", "FPLArchiveSource"]

ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
API = "https://fantasy.premierleague.com/api"

_POSITIONS = {
    "GK": Position.GK,
    "GKP": Position.GK,
    "DEF": Position.DEF,
    "MID": Position.MID,
    "FWD": Position.FWD,
}
_ELEMENT_TYPES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return default if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _appearance(row: dict[str, Any], season: str, teams: dict[int, str]) -> PlayerStat | None:
    """One archive/API history row to a PlayerStat. Managers ('AM') return None."""
    position = _POSITIONS.get(str(row.get("position", "")).upper())
    if position is None:
        return None
    name = str(row["name"])
    team_name = row.get("team")
    opponent_name = teams.get(_int(row.get("opponent_team"), -1), "")
    return PlayerStat(
        season=season,
        fixture_id=str(_int(row["fixture"])),
        player=PlayerRef.from_name(name),
        player_name=name,
        team=TeamRef.from_name(str(team_name)),
        opponent=TeamRef.from_name(opponent_name),
        position=position,
        is_home=bool(row.get("was_home", False)),
        kickoff=_utc(row.get("kickoff_time")),
        gameweek=_gameweek(row),
        minutes=_int(row.get("minutes")),
        started=_int(row.get("starts")),
        goals=_int(row.get("goals_scored")),
        assists=_int(row.get("assists")),
        xg=_float(row.get("expected_goals")),
        xa=_float(row.get("expected_assists")),
        goals_conceded=_int(row.get("goals_conceded")),
        clean_sheet=_int(row.get("clean_sheets")),
        saves=_int(row.get("saves")),
        defensive_contribution=_int(row.get("defensive_contribution")),
        bps=_int(row.get("bps")),
        bonus=_int(row.get("bonus")),
        yellow_cards=_int(row.get("yellow_cards")),
        red_cards=_int(row.get("red_cards")),
        own_goals=_int(row.get("own_goals")),
        total_points=_int(row.get("total_points")),
        price=(_float(row.get("value")) or 0.0) / 10.0 if row.get("value") is not None else None,
        ownership=_float(row.get("selected_by_percent")),
    )


def _gameweek(row: dict[str, Any]) -> int | None:
    """Archive rows call it ``GW``, the API ``round``; both may be blank."""
    value = row.get("GW", row.get("round"))
    return None if value is None or pd.isna(value) else _int(value)


def _utc(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    stamp = pd.Timestamp(value)
    if not isinstance(stamp, pd.Timestamp):
        return None  # NaT
    return stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")


def _result(row: dict[str, Any], season: str, teams: dict[int, str]) -> Result:
    return Result(
        season=season,
        fixture_id=str(_int(row["id"])),
        gameweek=None if pd.isna(row.get("event")) else _int(row["event"]),
        kickoff=_utc(row["kickoff_time"]),
        home=TeamRef.from_name(teams[_int(row["team_h"])]),
        away=TeamRef.from_name(teams[_int(row["team_a"])]),
        home_goals=None if pd.isna(row.get("team_h_score")) else _int(row["team_h_score"]),
        away_goals=None if pd.isna(row.get("team_a_score")) else _int(row["team_a_score"]),
    )


class FPLArchiveSource(Source):
    """The vaastav archive on GitHub. Historical seasons; a few refreshes a year."""

    name = "fpl-archive"
    provides = {Kind.RESULTS: Result, Kind.APPEARANCES: PlayerStat}

    def __init__(self, *, first_season: str = "2021-22", last_season: str = "2026-27") -> None:
        super().__init__()
        self._first, self._last = first_season, last_season

    def seasons(self) -> tuple[str, ...]:
        start, end = int(self._first[:4]), int(self._last[:4])
        return tuple(f"{y}-{str(y + 1)[2:]}" for y in range(start, end + 1))

    def _csv(self, season: str, path: str) -> pd.DataFrame | None:
        text = cached_get(f"{ARCHIVE}/{season}/{path}")
        return None if text is None else pd.read_csv(io.StringIO(text))

    def _teams(self, season: str) -> dict[int, str]:
        frame = self._csv(season, "teams.csv")
        if frame is None:
            raise FileNotFoundError(f"{self.name}: no teams.csv for {season}")
        return dict(zip(frame["id"].astype(int), frame["name"].astype(str), strict=True))

    def _fetch(self, kind: Kind, season: str) -> Dataset:
        teams = self._teams(season)
        if kind is Kind.RESULTS:
            frame = self._csv(season, "fixtures.csv")
            if frame is None:
                raise FileNotFoundError(f"{self.name}: no fixtures.csv for {season}")
            results = [_result(r, season, teams) for r in frame.to_dict("records")]
            return Dataset(results, record_type=Result, source=self.name)

        frame = self._csv(season, "gws/merged_gw.csv")
        if frame is None:
            raise FileNotFoundError(f"{self.name}: no merged_gw.csv for {season}")
        frame = frame.drop_duplicates(subset=["element", "fixture"])
        parsed = [_appearance(r, season, teams) for r in frame.to_dict("records")]
        appearances = [r for r in parsed if r is not None]
        return Dataset(appearances, record_type=PlayerStat, source=self.name)


class FPLApiSource(Source):
    """The live API, for the season in progress.

    Same records as the archive plus live ownership. Roughly one request per
    player for appearances, all cached. Not reachable from every network.
    """

    name = "fpl-api"
    provides = {Kind.RESULTS: Result, Kind.APPEARANCES: PlayerStat}

    def __init__(self) -> None:
        super().__init__()
        self._bootstrap: dict[str, Any] | None = None

    def bootstrap(self) -> dict[str, Any]:
        if self._bootstrap is None:
            import json

            text = cached_get(f"{API}/bootstrap-static/", refresh=True)
            if text is None:
                raise FileNotFoundError(f"{self.name}: bootstrap-static returned 404")
            self._bootstrap = json.loads(text)
        return self._bootstrap

    def seasons(self) -> tuple[str, ...]:
        year = pd.Timestamp(self.bootstrap()["events"][0]["deadline_time"]).year
        return (f"{year}-{str(year + 1)[2:]}",)

    def _fetch(self, kind: Kind, season: str) -> Dataset:
        import json

        boot = self.bootstrap()
        teams = {int(t["id"]): str(t["name"]) for t in boot["teams"]}
        if kind is Kind.RESULTS:
            text = cached_get(f"{API}/fixtures/", refresh=True) or "[]"
            results = [_result(r, season, teams) for r in json.loads(text)]
            return Dataset(results, record_type=Result, source=self.name)

        elements = boot["elements"]
        urls = [f"{API}/element-summary/{e['id']}/" for e in elements]
        bodies = cached_get_many(urls, workers=4)
        appearances: list[PlayerStat] = []
        for element, body in zip(elements, bodies, strict=True):
            if body is None:
                continue
            base = {
                "name": f"{element['first_name']} {element['second_name']}".strip(),
                "team": teams[int(element["team"])],
                "position": _ELEMENT_TYPES.get(int(element["element_type"]), ""),
                "selected_by_percent": element.get("selected_by_percent"),
            }
            for history in json.loads(body).get("history", []):
                row = _appearance({**history, **base}, season, teams)
                if row is not None:
                    appearances.append(row)
        return Dataset(appearances, record_type=PlayerStat, source=self.name)
