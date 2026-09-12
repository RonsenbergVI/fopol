"""Clients. Each knows one place data lives and returns fopol records."""

from fopol.sources.fpl import FPLApiSource, FPLArchiveSource
from fopol.sources.understat import UnderstatSource

__all__ = ["FPLApiSource", "FPLArchiveSource", "UnderstatSource"]
