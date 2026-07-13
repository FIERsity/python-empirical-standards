"""Heterogeneity and robustness workflows."""

from empirical_standards.diagnostics.heterogeneity import (
    HeterogeneityResult,
    InteractionHeterogeneityResult,
    fit_fe_by_group,
    fit_fe_heterogeneity,
)
from empirical_standards.diagnostics.robustness import covariance_sensitivity, placebo_did

__all__ = [
    "HeterogeneityResult",
    "InteractionHeterogeneityResult",
    "covariance_sensitivity",
    "fit_fe_by_group",
    "fit_fe_heterogeneity",
    "placebo_did",
]
