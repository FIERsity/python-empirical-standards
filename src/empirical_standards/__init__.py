"""Transparent building blocks for empirical research."""

from empirical_standards.causal.did import DIDResult, EventStudyResult, fit_did, fit_event_study
from empirical_standards.causal.staggered import StaggeredDIDResult, fit_staggered_did
from empirical_standards.causal.sun_abraham import SunAbrahamResult, fit_sun_abraham
from empirical_standards.models.iv import IV2SLSResult, fit_iv_2sls
from empirical_standards.models.ols import OLSResult, fit_ols
from empirical_standards.panel.fixed_effects import FixedEffectsResult, fit_fixed_effects

__all__ = [
    "DIDResult",
    "EventStudyResult",
    "FixedEffectsResult",
    "IV2SLSResult",
    "OLSResult",
    "StaggeredDIDResult",
    "SunAbrahamResult",
    "fit_did",
    "fit_event_study",
    "fit_fixed_effects",
    "fit_iv_2sls",
    "fit_ols",
    "fit_staggered_did",
    "fit_sun_abraham",
]
__version__ = "0.9.0"
