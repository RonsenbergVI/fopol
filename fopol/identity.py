"""Canonical identity for teams and players, resolved once at the source.

Every source names things differently. FPL says "Man City", "Nott'm Forest",
"Spurs"; Understat says "Manchester City", "Nottingham Forest", "Tottenham";
the FPL archive itself says "Ipswich" one season and "Ipswich Town" the next. A
join on raw names does not raise -- it silently drops rows -- so resolution
happens here, at the boundary, and nowhere else.

This module deals only in strings so that ``base`` depends on nothing concrete.
The record types wrap these ids: ``TeamRef.from_name("Spurs")``.

Team identity is an alias table over a normalised form. Player identity is a
normalised name slug: accents stripped, case folded. Not a perfect key -- two
players can share a name -- but the only one that exists across sources, and a
crosswalk can replace it later without changing any caller.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["canonical_player_id", "canonical_team_id", "is_known_team", "normalise", "slug"]


def normalise(text: str) -> str:
    """Accent-, case- and punctuation-insensitive key."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stripped.lower()).split())


def slug(text: str) -> str:
    return normalise(text).replace(" ", "-")


# canonical id -> every spelling seen across sources, in normalised form
_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "arsenal": ("arsenal",),
    "aston-villa": ("aston villa",),
    "bournemouth": ("bournemouth", "afc bournemouth"),
    "brentford": ("brentford",),
    "brighton": ("brighton", "brighton and hove albion", "brighton hove albion"),
    "burnley": ("burnley",),
    "chelsea": ("chelsea",),
    "coventry": ("coventry", "coventry city"),
    "crystal-palace": ("crystal palace",),
    "everton": ("everton",),
    "fulham": ("fulham",),
    "hull": ("hull", "hull city"),
    "ipswich": ("ipswich", "ipswich town"),
    "leeds": ("leeds", "leeds united"),
    "leicester": ("leicester", "leicester city"),
    "liverpool": ("liverpool",),
    "luton": ("luton", "luton town"),
    "man-city": ("man city", "manchester city"),
    "man-utd": ("man utd", "manchester united", "man united"),
    "newcastle": ("newcastle", "newcastle united"),
    "norwich": ("norwich", "norwich city"),
    "nottm-forest": ("nott m forest", "nottm forest", "nottingham forest"),
    "sheffield-utd": ("sheffield utd", "sheffield united"),
    "southampton": ("southampton",),
    "spurs": ("spurs", "tottenham", "tottenham hotspur"),
    "sunderland": ("sunderland",),
    "watford": ("watford",),
    "west-ham": ("west ham", "west ham united"),
    "wolves": ("wolves", "wolverhampton wanderers", "wolverhampton"),
}

_LOOKUP: dict[str, str] = {
    alias: canonical for canonical, aliases in _TEAM_ALIASES.items() for alias in aliases
}


def is_known_team(name: str) -> bool:
    return normalise(name) in _LOOKUP


def canonical_team_id(name: str) -> str:
    """Any spelling of a club to its canonical id.

    Unknown names are slugged rather than rejected, so a newly promoted club
    cannot crash a load -- but they will not join to anything until an alias is
    added, so check :func:`is_known_team` when it matters.
    """
    key = normalise(name)
    return _LOOKUP.get(key, key.replace(" ", "-"))


def canonical_player_id(name: str) -> str:
    return slug(name)
