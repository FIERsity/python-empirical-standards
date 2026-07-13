"""Difference-in-differences estimators."""

from empirical_standards.causal.did import DIDResult, EventStudyResult, fit_did, fit_event_study
from empirical_standards.causal.r_backends import (
    RStaggeredDIDResult,
    RSunAbrahamResult,
    fit_staggered_did_r,
    fit_sun_abraham_r,
)

__all__ = [
    "DIDResult",
    "EventStudyResult",
    "RStaggeredDIDResult",
    "RSunAbrahamResult",
    "fit_did",
    "fit_event_study",
    "fit_staggered_did_r",
    "fit_sun_abraham_r",
]
