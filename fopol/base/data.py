"""Data: the typed records everything else is written against.

Models take ``Data``, not frames. A frame is a processing convenience -- fast,
vectorised and untyped -- so it is allowed inside a function and never in a
signature. :meth:`fopol.base.dataset.Dataset.to_frame` and ``from_frame`` are the only
places that boundary is crossed, and they round-trip, so a processing step is
always ``Data -> frame -> work -> Data``.

Every column declares how it aggregates, in its own annotation::

    goals: Annotated[int, Agg.SUM]

That is the whole schema. There is no second description to keep in sync, and a
column cannot be added without someone deciding what it means over a season.

Grain lives in the type, not in a field. ``PlayerStat`` is per fixture;
``PlayerSeason`` is a season. They are different types because they have
different keys and different meanings -- a season row has no fixture, a per-game
row has no notion of games played. One type with nullable keys would push that
check into every consumer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, get_args, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_serializer,
    model_validator,
)

from fopol.constants import Grain, Position

__all__ = ["Agg", "Data", "FixtureId", "Grain", "Position", "Ref", "Season"]

Season = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}$")]
"""A season in archive form, e.g. 2026-27."""

FixtureId = Annotated[str, StringConstraints(min_length=1)]


class Agg(StrEnum):
    """How a column combines when per-game records collapse into a season."""

    KEY = "key"
    """Identifies the record. Grouped on, never aggregated."""
    SUM = "sum"
    """Counts: goals, minutes, saves, bonus."""
    MEAN = "mean"
    """Per-game quantities where each game weighs the same."""
    LAST = "last"
    """State at a moment: price, ownership, current club."""
    FIRST = "first"
    """Fixed attributes that merely repeat on every row."""
    MAX = "max"
    """Whether it ever happened: ever started, ever sent off."""
    DROP = "drop"
    """Meaningful per game, meaningless over a season."""


class Ref(BaseModel):
    """A canonical identifier.

    A distinct type rather than a bare string, because every silent join failure
    this project has hit was an identity bug -- a club renamed between seasons, a
    name that lost its accent. Serialises to its id, so a frame column holds the
    plain string and the round-trip stays lossless.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, value: Any) -> Any:
        return {"id": value} if isinstance(value, str) else value

    @model_serializer
    def _as_id(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.id


class Data(BaseModel):
    """Base for every record fopol consumes or produces."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    grain: ClassVar[Grain]
    aggregates_to: ClassVar[type[Data] | None] = None

    @classmethod
    def aggregations(cls) -> dict[str, Agg]:
        """Each field's aggregation rule, read off its own annotation."""
        hints = get_type_hints(cls, include_extras=True)
        return {
            name: next((m for m in get_args(hints.get(name)) if isinstance(m, Agg)), Agg.DROP)
            for name in cls.model_fields
        }

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(n for n, a in cls.aggregations().items() if a is Agg.KEY)

    @classmethod
    def columns(cls) -> tuple[str, ...]:
        return tuple(cls.model_fields)

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, k)) for k in self.keys())
