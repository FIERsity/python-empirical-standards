"""Econometric model implementations."""

from empirical_standards.models.iv import IV2SLSResult, fit_iv_2sls
from empirical_standards.models.ols import OLSResult, fit_ols

__all__ = ["IV2SLSResult", "OLSResult", "fit_iv_2sls", "fit_ols"]
