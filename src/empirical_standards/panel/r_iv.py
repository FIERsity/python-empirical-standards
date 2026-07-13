"""High-dimensional fixed-effects IV backed explicitly by R ``fixest``."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from empirical_standards.backends.r import run_r_backend
from empirical_standards.results import build_metadata

RIVCovariance = Literal["unadjusted", "robust", "cluster"]


@dataclass(frozen=True)
class RPanelIV2SLSResult:
    """Structured second-stage and diagnostic output from ``fixest::feols``."""

    coefficients: pd.DataFrame
    first_stage_diagnostics: pd.DataFrame
    metadata: dict[str, object]

    def tidy(self) -> pd.DataFrame:
        return self.coefficients.copy()

    def glance(self) -> pd.Series:
        return pd.Series(self.metadata)

    def model_spec(self) -> dict[str, object]:
        return dict(cast(dict[str, object], self.metadata["specification"]))

    def sample_info(self) -> dict[str, object]:
        return dict(cast(dict[str, object], self.metadata["sample"]))

    def provenance(self) -> dict[str, object]:
        return {
            key: self.metadata[key]
            for key in ("backend", "package", "r_version", "package_versions", "script")
        }


def _script() -> Path:
    return Path(str(files("empirical_standards.backends.r_scripts").joinpath("panel_iv_fixest.R")))


def fit_panel_iv_2sls_r(
    data: pd.DataFrame,
    outcome: str,
    *,
    exogenous: list[str] | tuple[str, ...],
    endogenous: list[str] | tuple[str, ...],
    instruments: list[str] | tuple[str, ...],
    fixed_effects: list[str] | tuple[str, ...],
    covariance: RIVCovariance = "cluster",
    cluster: str | None = None,
    drop_missing: bool = False,
    confidence_level: float = 0.95,
) -> RPanelIV2SLSResult:
    """Fit IV/2SLS with one or more high-dimensional fixed effects in R."""
    exog, endog, excluded, effects = (
        tuple(exogenous), tuple(endogenous), tuple(instruments), tuple(fixed_effects)
    )
    if not endog or not excluded or not effects:
        raise ValueError("endogenous, instruments, and fixed_effects must be non-empty")
    roles = (outcome, *exog, *endog, *excluded)
    if len(set(roles)) != len(roles):
        raise ValueError("outcome, exogenous, endogenous, and instruments must not overlap")
    if len(excluded) < len(endog):
        raise ValueError("fewer excluded instruments than endogenous variables")
    if covariance not in {"unadjusted", "robust", "cluster"}:
        raise ValueError("unsupported R panel IV covariance")
    if covariance == "cluster" and cluster is None:
        raise ValueError("cluster is required for clustered covariance")
    if covariance != "cluster" and cluster is not None:
        raise ValueError("cluster may only be set with covariance='cluster'")
    required = list(dict.fromkeys([*roles, *effects, *([cluster] if cluster else [])]))
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    sample = data.loc[:, required].copy()
    original_nobs = len(sample)
    missing_rows = sample.isna().any(axis=1)
    if missing_rows.any() and not drop_missing:
        raise ValueError("R panel IV columns contain missing values; set drop_missing=True")
    sample = sample.loc[~missing_rows].copy()
    if any(not pd.api.types.is_numeric_dtype(sample[column]) for column in roles):
        raise TypeError("outcome, exogenous, endogenous, and instruments must be numeric")
    if not np.isfinite(sample[list(roles)].to_numpy(dtype=float)).all():
        raise ValueError("R panel IV model columns must contain only finite values")
    if any(sample[column].nunique() < 2 for column in effects):
        raise ValueError("each fixed-effect dimension must contain at least two levels")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    specification: dict[str, object] = {
        "outcome": outcome,
        "exogenous": list(exog),
        "endogenous": list(endog),
        "instruments": list(excluded),
        "fixed_effects": list(effects),
        "covariance": covariance,
        "cluster": cluster,
        "confidence_level": confidence_level,
        "small_sample_correction": "fixest defaults; recorded in result metadata",
    }
    metadata, tables = run_r_backend(
        sample, specification, script=_script(), required_packages=("fixest", "jsonlite")
    )
    shared = build_metadata(
        estimator="r_panel_iv_2sls",
        outcome=outcome,
        predictors=(*exog, *endog),
        settings=specification,
        sample=sample,
        original_nobs=original_nobs,
    )
    metadata["specification"] = specification
    metadata["sample"] = shared.sample.to_dict()
    metadata["python_provenance"] = shared.provenance.to_dict()
    return RPanelIV2SLSResult(tables["coefficients"], tables["first_stage_diagnostics"], metadata)
