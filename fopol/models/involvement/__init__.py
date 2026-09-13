"""Involvement models. All satisfy :class:`InvolvementModel`; pit them freely."""

from fopol.base.model import InvolvementModel
from fopol.models.involvement.empirical_bayes import EmpiricalBayesInvolvement
from fopol.models.involvement.share import ShareInvolvement

__all__ = ["EmpiricalBayesInvolvement", "InvolvementModel", "ShareInvolvement"]
