"""Constrained teaching and exploratory tools outside the V1.0 stable API.

These functions remain importable for reproducibility, but their assumptions are
narrower than the named production estimators.  Prefer the declared R backends
for staggered treatment and wild-cluster inference.
"""

from empirical_standards.causal.staggered import StaggeredDIDResult, fit_staggered_did
from empirical_standards.causal.sun_abraham import SunAbrahamResult, fit_sun_abraham
from empirical_standards.diagnostics.heterogeneity import HeterogeneityResult, fit_fe_by_group
from empirical_standards.diagnostics.inference import (
    PermutationResult,
    WildClusterBootstrapResult,
    permutation_did,
    wild_cluster_bootstrap_fe,
)

__all__ = [
    "HeterogeneityResult",
    "PermutationResult",
    "StaggeredDIDResult",
    "SunAbrahamResult",
    "WildClusterBootstrapResult",
    "fit_fe_by_group",
    "fit_staggered_did",
    "fit_sun_abraham",
    "permutation_did",
    "wild_cluster_bootstrap_fe",
]
