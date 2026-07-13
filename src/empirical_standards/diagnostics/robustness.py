"""Common robustness workflows that preserve model specifications."""

from __future__ import annotations

import pandas as pd

from empirical_standards.causal.did import fit_did
from empirical_standards.panel.fixed_effects import PanelCovariance, fit_fixed_effects


def covariance_sensitivity(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str] | tuple[str, ...],
    *,
    entity: str,
    time: str,
    entity_effects: bool = True,
    time_effects: bool = True,
    covariances: tuple[PanelCovariance, ...] = ("robust", "cluster_entity", "cluster_two_way"),
) -> pd.DataFrame:
    """Re-estimate one FE specification under several declared covariance estimators."""
    tables: list[pd.DataFrame] = []
    for covariance in covariances:
        result = fit_fixed_effects(
            data,
            outcome,
            predictors,
            entity=entity,
            time=time,
            entity_effects=entity_effects,
            time_effects=time_effects,
            covariance=covariance,
        )
        table = result.tidy()
        table.insert(0, "covariance", covariance)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def placebo_did(
    data: pd.DataFrame,
    outcome: str,
    treated: str,
    *,
    entity: str,
    time: str,
    placebo_periods: list[int] | tuple[int, ...],
    controls: list[str] | tuple[str, ...] = (),
    covariance: PanelCovariance = "cluster_entity",
) -> pd.DataFrame:
    """Estimate DID effects at declared placebo adoption dates using pre-treatment data."""
    rows: list[dict[str, float]] = []
    for period in placebo_periods:
        subset = data.loc[data[time] < period + 1].copy()
        subset["__placebo_post"] = (subset[time] >= period).astype(int)
        result = fit_did(
            subset,
            outcome,
            treated,
            "__placebo_post",
            entity=entity,
            time=time,
            controls=controls,
            covariance=covariance,
        )
        rows.append(
            {
                "placebo_period": float(period),
                "estimate": result.effect,
                "std_error": result.standard_error,
                "p_value": result.p_value,
            }
        )
    return pd.DataFrame(rows)
