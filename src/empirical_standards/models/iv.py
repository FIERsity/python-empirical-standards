"""Explicit two-stage least squares with first-stage and specification diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

from empirical_standards.results import ModelMetadata, build_metadata

IVCovariance = Literal["unadjusted", "robust", "cluster"]


@dataclass(frozen=True)
class IV2SLSResult:
    coefficients: pd.Series
    standard_errors: pd.Series
    statistic: pd.Series
    p_values: pd.Series
    confidence_intervals: pd.DataFrame
    first_stage: pd.DataFrame
    nobs: int
    r_squared: float
    adjusted_r_squared: float
    covariance: IVCovariance
    outcome: str
    exogenous: tuple[str, ...]
    endogenous: tuple[str, ...]
    instruments: tuple[str, ...]
    add_intercept: bool
    cluster: str | None
    wu_hausman_statistic: float
    wu_hausman_p_value: float
    sargan_statistic: float
    sargan_p_value: float
    robust_overidentification_statistic: float
    robust_overidentification_p_value: float
    raw_result: Any
    metadata: ModelMetadata

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": self.coefficients.index,
                "estimate": self.coefficients.to_numpy(),
                "std_error": self.standard_errors.to_numpy(),
                "statistic": self.statistic.to_numpy(),
                "p_value": self.p_values.to_numpy(),
                "conf_low": self.confidence_intervals.iloc[:, 0].to_numpy(),
                "conf_high": self.confidence_intervals.iloc[:, 1].to_numpy(),
            }
        )

    def glance(self) -> pd.Series:
        return pd.Series(
            {
                "estimator": "iv_2sls",
                "nobs": self.nobs,
                "r_squared": self.r_squared,
                "adjusted_r_squared": self.adjusted_r_squared,
                "covariance": self.covariance,
                "wu_hausman_statistic": self.wu_hausman_statistic,
                "wu_hausman_p_value": self.wu_hausman_p_value,
                "sargan_statistic": self.sargan_statistic,
                "sargan_p_value": self.sargan_p_value,
                "robust_overidentification_statistic": self.robust_overidentification_statistic,
                "robust_overidentification_p_value": self.robust_overidentification_p_value,
            }
        )

    def model_spec(self) -> dict[str, Any]:
        return self.metadata.spec.to_dict()

    def sample_info(self) -> dict[str, Any]:
        return self.metadata.sample.to_dict()

    def provenance(self) -> dict[str, Any]:
        return self.metadata.provenance.to_dict()


def fit_iv_2sls(
    data: pd.DataFrame,
    outcome: str,
    *,
    exogenous: list[str] | tuple[str, ...],
    endogenous: list[str] | tuple[str, ...],
    instruments: list[str] | tuple[str, ...],
    add_intercept: bool = True,
    covariance: IVCovariance = "robust",
    cluster: str | None = None,
    drop_missing: bool = False,
) -> IV2SLSResult:
    """Fit 2SLS with explicit variable roles and covariance settings."""
    exog, endog, excluded = tuple(exogenous), tuple(endogenous), tuple(instruments)
    if not endog or not excluded:
        raise ValueError("endogenous and instruments must each contain at least one column")
    all_model_columns = (outcome, *exog, *endog, *excluded)
    if len(set(all_model_columns)) != len(all_model_columns):
        raise ValueError("outcome, exogenous, endogenous, and instruments must not overlap")
    if len(excluded) < len(endog):
        raise ValueError(
            "model is underidentified: fewer excluded instruments than endogenous variables"
        )
    if covariance not in {"unadjusted", "robust", "cluster"}:
        raise ValueError("unsupported IV covariance")
    if covariance == "cluster" and cluster is None:
        raise ValueError("cluster is required for clustered IV covariance")
    if covariance != "cluster" and cluster is not None:
        raise ValueError("cluster may only be set with covariance='cluster'")
    required = [*all_model_columns]
    if cluster is not None:
        required.append(cluster)
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    for column in all_model_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"column {column!r} must be numeric")
    sample = data.loc[:, required].copy()
    original_nobs = len(sample)
    missing_rows = sample.isna().any(axis=1)
    if missing_rows.any() and not drop_missing:
        raise ValueError("IV model columns contain missing values; set drop_missing=True")
    sample = sample.loc[~missing_rows].copy()
    if not np.isfinite(sample[list(all_model_columns)].to_numpy(dtype=float)).all():
        raise ValueError("IV model columns must contain only finite values")
    y = sample[outcome].astype(float)
    x = sample.loc[:, list(exog)].astype(float) if exog else pd.DataFrame(index=sample.index)
    if add_intercept:
        x.insert(0, "const", 1.0)
    endogenous_data = sample.loc[:, list(endog)].astype(float)
    instrument_data = sample.loc[:, list(excluded)].astype(float)
    fit_options: dict[str, Any] = {"debiased": True}
    if covariance == "unadjusted":
        fit_options["cov_type"] = "unadjusted"
    elif covariance == "robust":
        fit_options["cov_type"] = "robust"
    else:
        fit_options["cov_type"] = "clustered"
        fit_options["clusters"] = sample[cluster]
        if sample[cluster].nunique() < 2:
            raise ValueError("clustered IV covariance requires at least two clusters")
    fitted: Any = IV2SLS(y, x, endogenous_data, instrument_data).fit(**fit_options)
    first_stage = fitted.first_stage.diagnostics.reset_index(names="endogenous")
    first_stage = first_stage.rename(
        columns={
            "rsquared": "r_squared",
            "partial.rsquared": "partial_r_squared",
            "shea.rsquared": "shea_r_squared",
            "f.stat": "excluded_instruments_statistic",
            "f.pval": "excluded_instruments_p_value",
            "f.dist": "excluded_instruments_distribution",
        }
    )
    wu = fitted.wu_hausman()
    sargan = fitted.sargan
    robust_overid = fitted.wooldridge_overid
    intervals = fitted.conf_int()
    intervals.columns = ["conf_low", "conf_high"]
    predictors = (*exog, *endog)
    metadata = build_metadata(
        estimator="iv_2sls",
        outcome=outcome,
        predictors=predictors,
        settings={
            "exogenous": exog,
            "endogenous": endog,
            "instruments": excluded,
            "add_intercept": add_intercept,
            "covariance": covariance,
            "cluster": cluster,
            "drop_missing": drop_missing,
            "first_stage_statistic_label": "partial F or robust Wald; inspect distribution",
            "robust_overidentification_test": "Wooldridge score",
        },
        sample=sample,
        original_nobs=original_nobs,
    )
    return IV2SLSResult(
        coefficients=fitted.params.rename("estimate"),
        standard_errors=fitted.std_errors.rename("std_error"),
        statistic=fitted.tstats.rename("statistic"),
        p_values=fitted.pvalues.rename("p_value"),
        confidence_intervals=intervals,
        first_stage=first_stage,
        nobs=int(fitted.nobs),
        r_squared=float(fitted.rsquared),
        adjusted_r_squared=float(fitted.rsquared_adj),
        covariance=covariance,
        outcome=outcome,
        exogenous=exog,
        endogenous=endog,
        instruments=excluded,
        add_intercept=add_intercept,
        cluster=cluster,
        wu_hausman_statistic=float(wu.stat),
        wu_hausman_p_value=float(wu.pval),
        sargan_statistic=float(sargan.stat),
        sargan_p_value=float(sargan.pval),
        robust_overidentification_statistic=float(robust_overid.stat),
        robust_overidentification_p_value=float(robust_overid.pval),
        raw_result=fitted,
        metadata=metadata,
    )
