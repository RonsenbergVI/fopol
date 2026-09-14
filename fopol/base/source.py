"""Sources: clients that fetch per-game data from one place and return Data.

A source is a client and nothing more. It knows one place data lives, how to
translate that place's dialect into fopol's records, and nothing about models.
``base`` knows nothing concrete either: a source declares which record type it
returns for each :class:`Kind`, and the base only checks that promise is kept.

Three rules hold for every source.

**Per-game granularity, always.** One record per fixture, or one per player
per fixture; never season totals. Seasons are derived by
:meth:`~fopol.base.dataset.Dataset.aggregate`.

**Identity is resolved here.** Sources emit canonical refs via
:mod:`fopol.identity`. No analysis code ever writes an alias table.

**Fetches are cached on disk.** Some feeds are one file per player per season
-- Understat is ~800 requests. The cache makes the second load free and a load
reproducible: the same bytes come back until you clear it.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path

import httpx

from fopol.base.data import Data
from fopol.base.dataset import Dataset

__all__ = ["CACHE_DIR", "Kind", "Source", "UnsupportedKind", "cached_get", "cached_get_many"]

CACHE_DIR = Path.home() / ".cache" / "fopol"
_HEADERS = {"User-Agent": "fopol/0.5"}


class UnsupportedKind(NotImplementedError):
    """This source does not carry that kind of data."""


class Kind(StrEnum):
    """What a source can be asked for. The record type is the source's to declare."""

    RESULTS = "results"
    APPEARANCES = "appearances"

def cached_get(
    url: str, *, cache_dir: Path = CACHE_DIR, timeout: float = 60.0, refresh: bool = False
) -> str | None:
    """GET a URL as text, caching the body on disk keyed by the URL.

    A 404 returns ``None`` and is cached too, so a missing per-player file is a
    fact rather than a retry. Any other HTTP error raises.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / hashlib.sha1(url.encode()).hexdigest()
    body, gone = stem.with_suffix(".txt"), stem.with_suffix(".404")

    if not refresh:
        if body.exists():
            return body.read_text(encoding="utf-8")
        if gone.exists():
            return None

    response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
    if response.status_code == 404:
        gone.touch()
        return None
    response.raise_for_status()
    body.write_text(response.text, encoding="utf-8")
    return response.text


def cached_get_many(urls: list[str], *, workers: int = 8, **kwargs) -> list[str | None]:
    """Fetch several URLs concurrently, preserving order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda u: cached_get(u, **kwargs), urls))


# --------------------------------------------------------------------- client


class Source(ABC):
    """Base class for a client that fetches per-game data from one place.

    Subclasses implement :meth:`_fetch` and :meth:`seasons`, and declare
    :attr:`provides` -- a mapping from kind to the record type they return.

    Not a pydantic model: a source is behaviour and mutable state, not a record.
    """

    name: str
    """Short identifier for provenance, e.g. 'fpl-archive'."""

    provides: Mapping[Kind, type[Data]]

    def __init__(self) -> None:
        self._memo: dict[tuple[Kind, str], Dataset] = {}

    @abstractmethod
    def seasons(self) -> tuple[str, ...]:
        """Seasons this source can serve, oldest first, in archive form."""

    @abstractmethod
    def _fetch(self, kind: Kind, season: str) -> Dataset:
        """Records for one kind and season, tagged with this source's name."""


    def get(self, kind: Kind | str, season: str) -> Dataset:
        kind = Kind(kind)
        if kind not in self.provides:
            raise UnsupportedKind(
                f"{self.name} provides {sorted(k.value for k in self.provides)}, not {kind.value!r}"
            )
        key = (kind, season)
        if key not in self._memo:
            data = self._fetch(kind, season)
            expected = self.provides[kind]
            if data.record_type is not expected:
                raise TypeError(
                    f"{self.name}._fetch({kind.value!r}) returned "
                    f"{data.record_type.__name__}, declared {expected.__name__}"
                )
            self._memo[key] = data
        return self._memo[key]

    def results(self, season: str) -> Dataset:
        return self.get(Kind.RESULTS, season)

    def appearances(self, season: str) -> Dataset:
        return self.get(Kind.APPEARANCES, season)

    def load(self, kind: Kind | str, seasons: list[str] | None = None) -> Dataset:
        """Several seasons of one kind, concatenated."""
        kind = Kind(kind)
        wanted = list(self.seasons()) if seasons is None else list(seasons)
        records = [r for s in wanted for r in self.get(kind, s)]
        return Dataset(records, record_type=self.provides[kind], source=self.name)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{type(self).__name__}(provides={sorted(k.value for k in self.provides)})"
