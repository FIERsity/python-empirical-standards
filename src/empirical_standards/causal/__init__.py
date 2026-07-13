"""Difference-in-differences estimators."""

from empirical_standards.causal.did import DIDResult, EventStudyResult, fit_did, fit_event_study
from empirical_standards.causal.staggered import StaggeredDIDResult, fit_staggered_did
from empirical_standards.causal.sun_abraham import SunAbrahamResult, fit_sun_abraham

__all__ = [
    "DIDResult",
    "EventStudyResult",
    "StaggeredDIDResult",
    "SunAbrahamResult",
    "fit_did",
    "fit_event_study",
    "fit_staggered_did",
    "fit_sun_abraham",
]
