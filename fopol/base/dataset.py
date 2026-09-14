"""Dataset: a typed collection of records, and the only bridge to frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Generic, Self, TypeVar

import pandas as pd

from fopol.base.data import Agg, Data

__all__ = ["Dataset"]

T = TypeVar("T", bound=Data)


class Dataset(Generic[T]):
    """A homogeneous collection of records, carrying where they came from.

    Exists so provenance travels with the data and so ``to_frame`` and
    ``aggregate`` are methods rather than loose functions. Not a pydantic model:
    its records are already validated, and re-validating thirty thousand of them
    on every construction would be pure cost.
    """

    __slots__ = ("_records", "record_type", "source")

    def __init__(
        self,
        records: Iterable[T],
        *,
        record_type: type[T] | None = None,
        source: str = "",
    ) -> None:
        self._records: tuple[T, ...] = tuple(records)
        if record_type is None:
            if not self._records:
                raise ValueError("an empty Dataset needs an explicit record_type")
            record_type = type(self._records[0])
        self.record_type: type[T] = record_type
        self.source = source

        wrong = next((r for r in self._records if not isinstance(r, record_type)), None)
        if wrong is not None:
            raise TypeError(
                f"Dataset[{record_type.__name__}] was given a "
                f"{type(wrong).__name__}; a dataset holds one record type"
            )

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[T]:
        return iter(self._records)

    def __getitem__(self, index: int) -> T:
        return self._records[index]

    @property
    def records(self) -> tuple[T, ...]:
        return self._records

    def filter(self, predicate: Callable[[T], bool]) -> Self:
        return type(self)(
            [r for r in self._records if predicate(r)],
            record_type=self.record_type,
            source=self.source,
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        where = f", source={self.source!r}" if self.source else ""
        return f"Dataset[{self.record_type.__name__}]({len(self)} records{where})"

    # ---------------------------------------------------------- frame bridge

    def to_frame(self) -> pd.DataFrame:
        """Flatten to a frame. Refs become their id; column order follows fields."""
        if not self._records:
            return pd.DataFrame(columns=list(self.record_type.columns()))
        frame = pd.DataFrame([r.model_dump() for r in self._records])
        return frame.loc[:, [c for c in self.record_type.columns() if c in frame.columns]]

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, record_type: type[T], *, source: str = ""
    ) -> Dataset[T]:
        """Rebuild records from a frame. Inverse of :meth:`to_frame`."""
        missing = [
            name
            for name, field in record_type.model_fields.items()
            if field.is_required() and name not in frame.columns
        ]
        if missing:
            raise ValueError(f"{record_type.__name__}: frame is missing {missing}")
        rows = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        return cls([record_type(**row) for row in rows], record_type=record_type, source=source)

    def aggregate(self) -> Dataset[Any]:
        """Collapse per-fixture records into their season type.

        Grouping is on the *target* type's keys, so what you group by is a
        property of where you are going rather than an argument to remember.
        """
        target = self.record_type.aggregates_to
        if target is None:
            raise TypeError(f"{self.record_type.__name__} does not aggregate")

        by = [k for k in target.keys() if k in self.record_type.columns()]  # noqa: SIM118
        frame = self.to_frame()
        rules = self.record_type.aggregations()
        keep = {Agg.SUM, Agg.MEAN, Agg.LAST, Agg.FIRST, Agg.MAX}
        how = {
            name: rule.value
            for name, rule in rules.items()
            if rule in keep
            and name in frame.columns
            and name not in by
            and name in target.model_fields
        }

        grouped = frame.groupby(by, as_index=False, observed=True, dropna=False).agg(how)
        counts = (
            frame.groupby(by, observed=True, dropna=False).size().to_frame("games").reset_index()
        )
        grouped = grouped.merge(counts, on=by, how="left")

        for name, field in target.model_fields.items():
            if name not in grouped.columns and not field.is_required():
                grouped[name] = None
        return Dataset.from_frame(grouped, target, source=self.source)
