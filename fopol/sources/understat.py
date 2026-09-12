"""Understat: shot-based player statistics per match, via the archive mirror.

Understat itself is not reachable from a restricted network, but the vaastav
archive mirrors it season by season under ``understat/``: an index of players
and one CSV per player with a row per match -- goals, shots, xG, xA, key passes,
minutes, and, on every row, both teams and the scoreline. Available 2021-22
through 2024-25.

That last detail matters. Because every player row carries the match, PL
*results* can be derived from Understat appearances by de-duplicating on match
id, and each team's xG for the match is the sum of its players' xG. That is the
superposition the models are built on, coming straight from the data rather
than assumed.
"""

from __future__ import annotations

import io
from collections import defaultdict
from urllib.parse import quote

import pandas as pd

from fopol.base.data import Position
from fopol.base.dataset import Dataset
from fopol.base.source import Kind, Source, cached_get, cached_get_many
from fopol.data.player import PlayerRef, PlayerStat
from fopol.data.result import Result
from fopol.data.team import TeamRef

__all__ = ["UnderstatSource"]

ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def _position(code: str, fallback: str) -> Position:
    """Understat position codes -> FPL positions.

    D, DL, DR, DC are defenders; DM* are defensive midfielders; F* forwards;
    everything else midfield. 'Sub' carries no information, so use the
    player's season position instead.
    """
    token = str(code).strip().split(" ")[0].upper()
    if token in {"", "SUB"}:
        token = str(fallback).strip().split(" ")[0].upper()
    if token.startswith("GK") or token == "G":
        return Position.GK
    if token.startswith("DM"):
        return Position.MID
    if token.startswith("D"):
        return Position.DEF
    if token.startswith("F"):
        return Position.FWD
    return Position.MID


class UnderstatSource(Source):
    """Understat per-match player statistics, and the results derived from them."""

    name = "understat"
    provides = {Kind.RESULTS: Result, Kind.APPEARANCES: PlayerStat}

    def __init__(self, *, workers: int = 8) -> None:
        super().__init__()
        self._workers = workers
        self._rows_memo: dict[str, list[dict]] = {}

    def seasons(self) -> tuple[str, ...]:
        return ("2021-22", "2022-23", "2023-24", "2024-25")

    # ---------------------------------------------------------------- parsing

    def _index(self, season: str) -> pd.DataFrame:
        text = cached_get(f"{ARCHIVE}/{season}/understat/understat_player.csv")
        if text is None:
            raise FileNotFoundError(f"{self.name}: no understat index for {season}")
        return pd.read_csv(io.StringIO(text))

    def _rows(self, season: str) -> list[dict]:
        """Every placed player-match row for one PL season, parsed once.

        A per-player file is that player's whole career, every league included,
        so rows are kept only if they are from this season's year *and* between
        two clubs that appear in this season's PL index. Each kept row is
        tagged with the player's side, worked out from the clubs the index
        lists for him. A player the index lists for *both* clubs in a match
        -- a mid-season transfer between them -- cannot be placed and is dropped
        rather than guessed.
        """
        if season in self._rows_memo:
            return self._rows_memo[season]

        index = self._index(season)
        year = int(season[:4])
        league = {c.strip() for cell in index.team_title.astype(str) for c in cell.split(",")}
        players = index.to_dict("records")
        files = [
            f"{ARCHIVE}/{season}/understat/"
            + quote(f"{str(p['player_name']).replace(' ', '_')}_{int(p['id'])}.csv")
            for p in players
        ]
        bodies = cached_get_many(files, workers=self._workers)

        rows: list[dict] = []
        for meta, body in zip(players, bodies, strict=True):
            if body is None:
                continue
            clubs = {c.strip() for c in str(meta["team_title"]).split(",")}
            for r in pd.read_csv(io.StringIO(body)).to_dict("records"):
                if int(r.get("season", -1)) != year:
                    continue
                home, away = str(r["h_team"]), str(r["a_team"])
                if home not in league or away not in league:
                    continue
                if (home in clubs) == (away in clubs):
                    continue  # neither club is his, or both are: the row cannot be placed
                if home in clubs:
                    side, opponent, is_home = home, away, True
                else:
                    side, opponent, is_home = away, home, False
                rows.append(
                    {
                        **r,
                        "player_name": str(meta["player_name"]),
                        "season_position": str(meta["position"]),
                        "team": side,
                        "opponent": opponent,
                        "is_home": is_home,
                    }
                )
        self._rows_memo[season] = rows
        return rows

    def _appearances(self, season: str) -> list[PlayerStat]:
        return [
            PlayerStat(
                season=season,
                fixture_id=str(int(r["id"])),
                player=PlayerRef.from_name(r["player_name"]),
                player_name=r["player_name"],
                team=TeamRef.from_name(r["team"]),
                opponent=TeamRef.from_name(r["opponent"]),
                position=_position(str(r.get("position", "")), r["season_position"]),
                is_home=r["is_home"],
                kickoff=pd.Timestamp(str(r["date"]), tz="UTC"),
                minutes=int(r.get("time", 0) or 0),
                started=int(str(r.get("position", "")).strip().lower() != "sub"),
                goals=int(r.get("goals", 0) or 0),
                assists=int(r.get("assists", 0) or 0),
                xg=float(r.get("xG", 0.0) or 0.0),
                xa=float(r.get("xA", 0.0) or 0.0),
                shots=int(r.get("shots", 0) or 0),
                key_passes=int(r.get("key_passes", 0) or 0),
            )
            for r in self._rows(season)
        ]

    def _results(self, season: str) -> list[Result]:
        """Results are a view over appearances: one per match id, xG summed by side."""
        by_match: dict[str, dict] = {}
        xg: dict[tuple[str, str], float] = defaultdict(float)
        for r in self._rows(season):
            mid = str(int(r["id"]))
            by_match.setdefault(
                mid,
                {
                    "home": str(r["h_team"]),
                    "away": str(r["a_team"]),
                    "hg": int(r["h_goals"]),
                    "ag": int(r["a_goals"]),
                    "date": str(r["date"]),
                },
            )
            xg[(mid, r["team"])] += float(r.get("xG", 0.0) or 0.0)

        return [
            Result(
                season=season,
                fixture_id=mid,
                kickoff=pd.Timestamp(m["date"], tz="UTC"),
                home=TeamRef.from_name(m["home"]),
                away=TeamRef.from_name(m["away"]),
                home_goals=m["hg"],
                away_goals=m["ag"],
                home_xg=round(xg[(mid, m["home"])], 4),
                away_xg=round(xg[(mid, m["away"])], 4),
            )
            for mid, m in sorted(by_match.items(), key=lambda kv: kv[1]["date"])
        ]

    def _fetch(self, kind: Kind, season: str) -> Dataset:
        if kind is Kind.RESULTS:
            return Dataset(self._results(season), record_type=Result, source=self.name)
        return Dataset(self._appearances(season), record_type=PlayerStat, source=self.name)
