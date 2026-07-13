"""Weak-identification-robust Anderson-Rubin tests and grid inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

ARCovariance = Literal["unadjusted", "robust", "cluster"]


@dataclass(frozen=True)
class AndersonRubinResult:
    null_value: float
    statistic: float
    p_value: float
    distribution: str
    instrument_count: int
    covariance: ARCovariance


@dataclass(frozen=True)
class AndersonRubinConfidenceSet:
    grid_results: pd.DataFrame
    accepted_values: tuple[float, ...]
    confidence_level: float
    lower_bound: float
    upper_bound: float
    bounded_below: bool
    bounded_above: bool

    def plot_data(self) -> pd.DataFrame:
        """Return the full inversion grid for external plotting or auditing."""
        return self.grid_results.rename(columns={"null_value": "parameter_value"}).copy()


def _ar_design(
    data: pd.DataFrame,
    exogenous: tuple[str, ...],
    instruments: tuple[str, ...],
    fixed_effects: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    design = data.loc[:, [*exogenous, *instruments]].astype(float).copy()
    for index, column in enumerate(fixed_effects):
        indicators = pd.get_dummies(
            data[column], prefix=f"__ar_fe_{index}", drop_first=True, dtype=float
        )
        design = pd.concat([design, indicators], axis=1)
    design = sm.add_constant(design, has_constant="add")
    return design, list(instruments)


def anderson_rubin_test(
    data: pd.DataFrame,
    outcome: str,
    endogenous: str,
    instruments: list[str] | tuple[str, ...],
    *,
    null_value: float,
    exogenous: list[str] | tuple[str, ...] = (),
    fixed_effects: list[str] | tuple[str, ...] = (),
    covariance: ARCovariance = "robust",
    cluster: str | None = None,
) -> AndersonRubinResult:
    """Test a single endogenous coefficient by joint instrument exclusion under the null."""
    excluded, controls, effects = tuple(instruments), tuple(exogenous), tuple(fixed_effects)
    if not excluded:
        raise ValueError("instruments must contain at least one column")
    if covariance == "cluster" and cluster is None:
        raise ValueError("cluster is required for clustered Anderson-Rubin inference")
    if covariance != "cluster" and cluster is not None:
        raise ValueError("cluster may only be set with covariance='cluster'")
    required = [outcome, endogenous, *excluded, *controls, *effects]
    if cluster is not None:
        required.append(cluster)
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    sample = data.loc[:, required]
    if sample.isna().any().any():
        raise ValueError("Anderson-Rubin test currently requires complete data")
    adjusted_outcome = sample[outcome].astype(float) - null_value * sample[endogenous].astype(float)
    design, instrument_terms = _ar_design(sample, controls, excluded, effects)
    model = sm.OLS(adjusted_outcome, design, missing="raise")
    if covariance == "unadjusted":
        fitted = model.fit()
    elif covariance == "robust":
        fitted = model.fit(cov_type="HC1")
    else:
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": sample[cluster]})
    restriction = np.zeros((len(instrument_terms), len(design.columns)))
    for row, term in enumerate(instrument_terms):
        restriction[row, design.columns.get_loc(term)] = 1
    test = fitted.wald_test(restriction, scalar=True)
    return AndersonRubinResult(
        null_value,
        float(test.statistic),
        float(test.pvalue),
        str(test.distribution),
        len(excluded),
        covariance,
    )


def anderson_rubin_confidence_set(
    data: pd.DataFrame,
    outcome: str,
    endogenous: str,
    instruments: list[str] | tuple[str, ...],
    *,
    grid: list[float] | tuple[float, ...] | np.ndarray,
    exogenous: list[str] | tuple[str, ...] = (),
    fixed_effects: list[str] | tuple[str, ...] = (),
    covariance: ARCovariance = "robust",
    cluster: str | None = None,
    confidence_level: float = 0.95,
) -> AndersonRubinConfidenceSet:
    """Invert Anderson-Rubin tests over an explicit grid without assuming a connected set."""
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("grid must be a strictly increasing one-dimensional sequence")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    rows: list[dict[str, float | bool]] = []
    for value in values:
        result = anderson_rubin_test(
            data,
            outcome,
            endogenous,
            instruments,
            null_value=float(value),
            exogenous=exogenous,
            fixed_effects=fixed_effects,
            covariance=covariance,
            cluster=cluster,
        )
        rows.append(
            {
                "null_value": float(value),
                "statistic": result.statistic,
                "p_value": result.p_value,
                "accepted": result.p_value >= 1 - confidence_level,
            }
        )
    table = pd.DataFrame(rows)
    accepted = table.loc[table["accepted"], "null_value"].to_numpy(dtype=float)
    if len(accepted):
        lower, upper = float(accepted.min()), float(accepted.max())
        bounded_below = bool(accepted.min() > values.min())
        bounded_above = bool(accepted.max() < values.max())
    else:
        lower = upper = float("nan")
        bounded_below = bounded_above = False
    return AndersonRubinConfidenceSet(
        table,
        tuple(accepted.tolist()),
        confidence_level,
        lower,
        upper,
        bounded_below,
        bounded_above,
    )
