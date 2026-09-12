"""Involvement models. All satisfy :class:`InvolvementModel`; pit them freely."""

from fopol.base.model import InvolvementModel
from fopol.models.involvement.empirical_bayes import EmpiricalBayesInvolvement

__all__ = ["EmpiricalBayesInvolvement", "InvolvementModel"]
