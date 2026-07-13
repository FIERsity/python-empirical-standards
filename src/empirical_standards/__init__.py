"""Transparent building blocks for empirical research."""

from empirical_standards.causal.did import DIDResult, EventStudyResult, fit_did, fit_event_study
from empirical_standards.causal.r_backends import (
    RStaggeredDIDResult,
    RSunAbrahamResult,
    fit_staggered_did_r,
    fit_sun_abraham_r,
)
from empirical_standards.diagnostics.r_inference import (
    RWildClusterBootstrapResult,
    wild_cluster_bootstrap_fe_r,
)
from empirical_standards.models.iv import IV2SLSResult, fit_iv_2sls
from empirical_standards.models.iv_diagnostics import summarize_first_stage
from empirical_standards.models.iv_inference import (
    AndersonRubinConfidenceSet,
    AndersonRubinResult,
    anderson_rubin_confidence_set,
    anderson_rubin_test,
)
from empirical_standards.models.iv_relevance import IVRelevanceDiagnostics, diagnose_iv_relevance
from empirical_standards.models.ols import OLSResult, fit_ols
from empirical_standards.panel.fixed_effects import FixedEffectsResult, fit_fixed_effects
from empirical_standards.panel.iv import PanelIV2SLSResult, fit_panel_iv_2sls
from empirical_standards.panel.r_iv import RPanelIV2SLSResult, fit_panel_iv_2sls_r

__all__ = [
    "AndersonRubinConfidenceSet",
    "AndersonRubinResult",
    "DIDResult",
    "EventStudyResult",
    "FixedEffectsResult",
    "IV2SLSResult",
    "IVRelevanceDiagnostics",
    "OLSResult",
    "PanelIV2SLSResult",
    "RPanelIV2SLSResult",
    "RStaggeredDIDResult",
    "RSunAbrahamResult",
    "RWildClusterBootstrapResult",
    "anderson_rubin_confidence_set",
    "anderson_rubin_test",
    "diagnose_iv_relevance",
    "fit_did",
    "fit_event_study",
    "fit_fixed_effects",
    "fit_iv_2sls",
    "fit_ols",
    "fit_panel_iv_2sls",
    "fit_panel_iv_2sls_r",
    "fit_staggered_did_r",
    "fit_sun_abraham_r",
    "summarize_first_stage",
    "wild_cluster_bootstrap_fe_r",
]
__version__ = "1.0.0"
