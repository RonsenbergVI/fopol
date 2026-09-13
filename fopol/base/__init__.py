"""Foundations: abstract records, the typed collection, and clients.

Record types live in :mod:`fopol.data`, clients in :mod:`fopol.sources`, model
variants in :mod:`fopol.models`, and name resolution in :mod:`fopol.identity`.

:mod:`fopol.base.model` is deliberately not re-exported here. The model family
bases declare the records they consume and produce, so that module imports
:mod:`fopol.data` -- and every record imports :mod:`fopol.base.data`. Pulling
``model`` into this package's namespace would make importing a record import
the models and close a cycle; import ``Model``, ``Inference`` and the family
bases from :mod:`fopol.base.model` directly.
"""

from fopol.base.data import Agg, Data, FixtureId, Grain, Position, Ref, Season
from fopol.base.dataset import Dataset
from fopol.base.source import Kind, Source, UnsupportedKind, cached_get, cached_get_many

__all__ = [
    "Agg",
    "Data",
    "Dataset",
    "FixtureId",
    "Grain",
    "Kind",
    "Position",
    "Ref",
    "Season",
    "Source",
    "UnsupportedKind",
    "cached_get",
    "cached_get_many",
]
