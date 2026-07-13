"""Heterogeneity and robustness workflows."""

from empirical_standards.diagnostics.heterogeneity import HeterogeneityResult, fit_fe_by_group
from empirical_standards.diagnostics.robustness import covariance_sensitivity, placebo_did

__all__ = ["HeterogeneityResult", "covariance_sensitivity", "fit_fe_by_group", "placebo_did"]
