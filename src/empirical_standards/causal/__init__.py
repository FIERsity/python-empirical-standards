"""Difference-in-differences estimators."""

from empirical_standards.causal.did import DIDResult, EventStudyResult, fit_did, fit_event_study
from empirical_standards.causal.staggered import StaggeredDIDResult, fit_staggered_did

__all__ = [
    "DIDResult",
    "EventStudyResult",
    "StaggeredDIDResult",
    "fit_did",
    "fit_event_study",
    "fit_staggered_did",
]
